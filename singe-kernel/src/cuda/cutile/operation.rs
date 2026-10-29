//! Owned submission of already compiled cuTile specializations.

use std::sync::Arc;

use cutile::{cuda_async::device_operation::DeviceOp, tile_kernel::TileFunctionKey};
use singe_cuda::{
    context::Context,
    error::{Error as CudaError, Result as CudaResult},
    graph::owned::{Recordable, Recording},
    stream::Stream,
    submission::{DeviceOperation, Execution, Operation, Resources},
};

use crate::{cuda::cutile::runtime::tile_kernel::prepared_function, error::Result};

/// A cuTile operation retaining its exact compiled native function and declared resources.
#[derive(Debug)]
pub struct TileOperation<O, T> {
    operation: O,
    resources: Resources,
    output: T,
    context: Arc<Context>,
}

impl<O, T> TileOperation<O, T> {
    /// Binds an already compiled specialization to an owned operation.
    /// Use the generated kernel's `compile_on` and `l1_cache_key_on` before constructing this value.
    ///
    /// # Safety
    /// The key must identify every specialization executed by `operation` in this actual context.
    /// The operation must enqueue only already prepared kernels without allocation or host readback.
    /// Retain all pointed-to storage and declare every byte-range access in `resources`.
    /// The output must preserve pending validity until successful execution, including graph capture.
    pub unsafe fn create(
        stream: &Stream,
        operation: O,
        key: &TileFunctionKey,
        mut resources: Resources,
        output: T,
    ) -> Result<Self> {
        resources.retain(prepared_function(stream.context(), key)?);
        Ok(Self {
            operation,
            resources,
            output,
            context: Arc::clone(stream.context()),
        })
    }
}

impl<O: DeviceOp, T> TileOperation<O, T> {
    unsafe fn execute_on(operation: O, context: &Context, stream: &Stream) -> CudaResult<()> {
        if stream.context().as_ref() != context {
            return Err(CudaError::StreamContextMismatch);
        }
        let foreign = stream.to_cuda_core()?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { operation.async_on(&foreign) }
            .map(|_| ())
            .map_err(|error| CudaError::Operation(Box::new(error)))
    }
}

unsafe impl<O: DeviceOp, T: 'static> Operation for TileOperation<O, T> {
    type Pending = Self::Output;

    type Output = T;

    fn enqueue(self, execution: &mut Execution) -> CudaResult<T> {
        // SAFETY: `create` required the operation to declare every access in `resources`, which the execution retains.
        unsafe {
            execution.enqueue(self.resources, |stream| {
                Self::execute_on(self.operation, &self.context, stream)
            })?;
        }
        Ok(self.output)
    }

    unsafe fn resolve(pending: Self::Pending) -> CudaResult<Self::Output> {
        Ok(pending)
    }
}

// SAFETY: enqueue returns pending device metadata and resolution is the identity function.
unsafe impl<O: DeviceOp, T: 'static> DeviceOperation for TileOperation<O, T> {}

unsafe impl<O: DeviceOp, T: 'static> Recordable for TileOperation<O, T> {
    type Output = T;

    fn record(self, recording: &mut Recording) -> CudaResult<T> {
        // SAFETY: `create` required the operation to declare every access in `resources`, which the recording retains.
        unsafe {
            recording.enqueue(self.resources, |stream| {
                Self::execute_on(self.operation, &self.context, stream)
            })?;
        }
        Ok(self.output)
    }
}
