//! Prepared owned cuDNN frontend execution with common CUDA submission retention.

use std::{
    mem::{MaybeUninit, take},
    sync::{Arc, Mutex},
};

use singe_cuda::{
    error::{Error as CudaError, Result as CudaResult},
    graph::owned::{Recordable, Recording},
    memory::DeviceMemory,
    stream::Stream,
    submission::{AccessMode, DeviceOperation, Execution, Operation, Resources},
    view::DeviceRegionProvider,
};

use crate::{
    backend::execution::RetainedExecution,
    context::Context,
    error::{Error, Result},
    frontend::{
        bindings::{Bindings, RuntimeOverrides},
        plan::CompiledGraph,
    },
    utility::validate_context_pair,
};

/// A prepared frontend execution, including native descriptors, scalar bindings and workspace.
/// Allocation and variant-pack construction happen before enqueue or capture.
#[derive(Debug)]
pub struct FrontendExecution<T> {
    context: Arc<Mutex<Context>>,
    graph: Arc<CompiledGraph>,
    invocation: Arc<Mutex<RetainedExecution>>,
    resources: Resources,
    output: T,
}

impl<T: 'static> FrontendExecution<T> {
    /// Prepares an execution from established tensor bindings and owned replay resources.
    ///
    /// # Safety
    /// Retain borrowed and raw tensor allocations in `resources` through completion or replay.
    /// Typed, encoded-byte and retained-region bindings derive graph access declarations automatically.
    /// Raw pointers require complete access declarations and producer dependencies in `resources`.
    /// Bindings registered with [`Bindings::set_owned_region`] also retain their allocation owners.
    /// Establish producer dependencies not represented by recorded readiness or tracked CUDA accesses.
    /// Bindings must satisfy the plan's layout, validity and aliasing requirements for every replay.
    /// Runtime overrides describe the binding layouts for every submission or replay.
    /// Writable outputs must preserve pending initialization until successful execution.
    ///
    /// # Errors
    ///
    /// Returns an error if the context mutex is poisoned, context targets disagree, runtime overrides or tensor bindings are invalid, workspace is insufficient, variant-pack construction fails, or native execution-context and resource retention cannot be prepared.
    pub unsafe fn create(
        context: Arc<Mutex<Context>>,
        graph: Arc<CompiledGraph>,
        bindings: Bindings<'_>,
        mut workspace: DeviceMemory<MaybeUninit<u8>>,
        overrides: &RuntimeOverrides,
        mut resources: Resources,
        output: T,
    ) -> Result<Self> {
        let invocation = {
            let context = context.lock().map_err(|_| {
                Error::Cuda(CudaError::ResourcePoisoned {
                    name: "cudnn context",
                })
            })?;
            validate_context_pair(
                Some(context.cuda_context()),
                graph.execution_plan().cuda_context(),
                "execution plan",
            )?;
            let pack = graph.variant_pack_with_overrides(
                context.cuda_context(),
                &bindings,
                Some(&mut workspace),
                overrides,
            )?;
            let invocation = RetainedExecution::create(graph.execution_plan(), &context, pack)?;
            drop(context);
            invocation
        };
        bindings.retain_resources(graph.required_tensor_bindings(), &mut resources);
        resources.retain(Arc::clone(workspace.allocation()));
        resources.access(workspace.region(), AccessMode::ReadWrite);
        let invocation = Arc::new(Mutex::new(invocation));
        resources.retain(Arc::clone(&context));
        resources.retain(Arc::clone(&graph));
        resources.retain(Arc::clone(&invocation));
        Ok(Self {
            context,
            graph,
            invocation,
            resources,
            output,
        })
    }

    fn execute_on(&self, stream: &Stream) -> CudaResult<()> {
        let mut context = self
            .context
            .lock()
            .map_err(|_| CudaError::ResourcePoisoned {
                name: "cudnn context",
            })?;
        if context.cuda_context() != stream.context() {
            return Err(CudaError::StreamContextMismatch);
        }
        let invocation = self
            .invocation
            .lock()
            .map_err(|_| CudaError::ResourcePoisoned {
                name: "cudnn prepared execution",
            })?;
        context
            .with_stream(Some(stream), |context| {
                // SAFETY: creation retained the bindings, scalar values, layout metadata and all declared replay resources.
                unsafe {
                    self.graph.log_execution(
                        context,
                        invocation.variant_pack(),
                        self.graph.default_plan(),
                    )?;
                }
                // SAFETY: prepared execution retains native state, scalar storage and declared buffers through completion or replay.
                unsafe { invocation.execute(context) }
            })
            .map_err(|error| CudaError::Operation(Box::new(error)))
    }
}

unsafe impl<T: 'static> Operation for FrontendExecution<T> {
    type Pending = Self::Output;

    type Output = T;

    fn enqueue(mut self, execution: &mut Execution) -> CudaResult<T> {
        let resources = take(&mut self.resources);
        // SAFETY: the declared resources retain every buffer and the stream orders all access to them.
        unsafe {
            execution.enqueue(resources, |stream| self.execute_on(stream))?;
        }
        Ok(self.output)
    }

    unsafe fn resolve(pending: Self::Pending) -> CudaResult<Self::Output> {
        Ok(pending)
    }
}

// SAFETY: enqueue returns pending device metadata and resolution is the identity function.
unsafe impl<T: 'static> DeviceOperation for FrontendExecution<T> {}

unsafe impl<T: 'static> Recordable for FrontendExecution<T> {
    type Output = T;

    fn record(mut self, recording: &mut Recording) -> CudaResult<T> {
        let resources = take(&mut self.resources);
        // SAFETY: the declared resources retain every buffer and the stream orders all access to them.
        unsafe {
            recording.enqueue(resources, |stream| self.execute_on(stream))?;
        }
        Ok(self.output)
    }
}
