//! Owned inference operations using Singe's common submission ledger.

use std::{
    mem::MaybeUninit,
    sync::{Arc, Mutex},
};

use singe_cuda::{
    error::{Error as CudaError, Result as CudaResult},
    graph::owned::{Recordable, Recording},
    memory::DeviceMemory,
    stream::Stream,
    submission::{
        AccessMode, DeviceOperation, Execution, Operation, Resources, transfer::DeviceValue,
    },
    view::{DeviceRegionProvider, DeviceRepr},
};

use crate::{
    context::Context,
    gemm::{self, DeviceMatrix, DeviceMatrixMut, GemmConfig, GemmElement},
};

/// An owned initialized column-major matrix, including its leading dimension.
#[derive(Debug)]
pub struct Matrix<T> {
    memory: DeviceMemory<T>,
    leading_dimension: usize,
}

impl<T> Matrix<T> {
    /// Takes ownership of initialized matrix storage and its column stride, in elements.
    ///
    /// The layout is checked when the matrix is used by an operation.
    #[must_use]
    pub fn new(memory: DeviceMemory<T>, leading_dimension: usize) -> Self {
        Self {
            memory,
            leading_dimension,
        }
    }

    /// Returns the owned device storage.
    #[must_use]
    pub fn memory(&self) -> &DeviceMemory<T> {
        &self.memory
    }

    /// Returns the stride between adjacent columns, in elements.
    #[must_use]
    pub fn leading_dimension(&self) -> usize {
        self.leading_dimension
    }
}

/// Owned GEMM retaining its handle and all matrix allocations through completion.
/// The destination must already contain initialized values, including layout padding.
#[derive(Debug)]
pub struct Gemm<T> {
    context: Arc<Mutex<Context>>,
    config: GemmConfig,
    alpha: T,
    a: Matrix<T>,
    b: Matrix<T>,
    beta: T,
    c: Matrix<T>,
}

impl<T> Gemm<T> {
    /// Creates an owned GEMM operation, retaining its context, scalars, and matrices.
    ///
    /// Construction does not submit GPU work.
    pub fn new(
        context: Arc<Mutex<Context>>,
        config: GemmConfig,
        alpha: T,
        a: Matrix<T>,
        b: Matrix<T>,
        beta: T,
        c: Matrix<T>,
    ) -> Self {
        Self {
            context,
            config,
            alpha,
            a,
            b,
            beta,
            c,
        }
    }
}

impl<T: GemmElement + DeviceRepr + Send + Sync> Gemm<T> {
    fn resources(&self) -> Resources {
        let mut resources = Resources::new();
        resources.retain(Arc::clone(&self.context));
        resources.retain(Arc::clone(self.a.memory.allocation()));
        resources.retain(Arc::clone(self.b.memory.allocation()));
        resources.retain(Arc::clone(self.c.memory.allocation()));
        resources.access(self.a.memory.region(), AccessMode::Read);
        resources.access(self.b.memory.region(), AccessMode::Read);
        resources.access(self.c.memory.region(), AccessMode::ReadWrite);
        resources
    }

    fn execute_on(&mut self, stream: &Stream) -> CudaResult<()> {
        let context = self
            .context
            .lock()
            .map_err(|_| CudaError::ResourcePoisoned {
                name: "cublas context",
            })?;
        if context.cuda_context() != stream.context() {
            return Err(CudaError::StreamContextMismatch);
        }
        context
            .set_stream(Some(stream))
            .map_err(|error| CudaError::Operation(Box::new(error)))?;
        // SAFETY: the operation owns its matrices, and the stream set above orders every access to them.
        let result = unsafe {
            gemm::gemm(
                &context,
                self.config,
                &self.alpha,
                DeviceMatrix::new(&self.a.memory, self.a.leading_dimension),
                DeviceMatrix::new(&self.b.memory, self.b.leading_dimension),
                &self.beta,
                DeviceMatrixMut::new(&mut self.c.memory, self.c.leading_dimension),
            )
            .map_err(|error| CudaError::Operation(Box::new(error)))
        };
        drop(context);
        result
    }
}

unsafe impl<T: GemmElement + DeviceRepr + Send + Sync> Operation for Gemm<T> {
    type Pending = Self::Output;

    type Output = DeviceValue<T>;

    fn enqueue(mut self, execution: &mut Execution) -> CudaResult<Self::Output> {
        let resources = self.resources();
        // SAFETY: the declared resources retain every buffer and the stream orders all access to them.
        unsafe {
            execution.enqueue(resources, |stream| self.execute_on(stream))?;
            DeviceValue::from_pending(self.c.memory.into_uninit(), execution.stream())
        }
    }

    unsafe fn resolve(pending: Self::Pending) -> CudaResult<Self::Output> {
        Ok(pending)
    }
}

// SAFETY: enqueue returns pending device metadata and resolution is the identity function.
unsafe impl<T: GemmElement + DeviceRepr + Send + Sync> DeviceOperation for Gemm<T> {}

unsafe impl<T: GemmElement + DeviceRepr + Send + Sync> Recordable for Gemm<T> {
    type Output = DeviceMemory<MaybeUninit<T>>;

    fn record(mut self, recording: &mut Recording) -> CudaResult<Self::Output> {
        let resources = self.resources();
        // SAFETY: the declared resources retain every buffer and the stream orders all access to them.
        unsafe {
            recording.enqueue(resources, |stream| self.execute_on(stream))?;
        }
        Ok(self.c.memory.into_uninit())
    }
}
