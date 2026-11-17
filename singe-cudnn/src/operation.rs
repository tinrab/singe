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
    ///
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
            let context = context.lock().map_err(|_| Error::LockPoisoned {
                name: "cudnn context".into(),
            })?;
            validate_context_pair(
                Some(context.cuda_context()),
                graph.execution_plan().cuda_context(),
                "execution plan",
            )?;
            let pack = graph.variant_pack_with_overrides(
                context.cuda_context(),
                &bindings,
                Some(workspace.output()),
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
        let mut context = self.context.lock().map_err(|_| CudaError::LockPoisoned {
            name: "cudnn context".into(),
        })?;
        if context.cuda_context() != stream.context() {
            return Err(CudaError::StreamContextMismatch);
        }

        let invocation = self
            .invocation
            .lock()
            .map_err(|_| CudaError::LockPoisoned {
                name: "cudnn prepared execution".into(),
            })?;
        context
            .with_stream(Some(stream), |context| {
                // SAFETY: logging reads the variant pack that `create` built from these bindings, and the retained invocation keeps its scalar storage alive.
                unsafe {
                    self.graph.log_execution(
                        context,
                        invocation.variant_pack(),
                        self.graph.default_plan(),
                    )?;
                }
                // SAFETY: `create` validated the plan's context against this handle, the stream context matches it, and `resources` keeps every buffer alive through completion or replay, per `create`'s contract.
                unsafe { invocation.execute(context) }
            })
            .map_err(|error| CudaError::Operation(Box::new(error)))
    }
}

// SAFETY: `enqueue` hands every retained owner and declared access in `resources` to `Execution::enqueue` before cuDNN runs, and `create` already allocated the workspace and built the variant pack, so nothing capture-incompatible runs here.
// The pending output is the caller's `T`, whose own contract under `create` keeps it uninitialized until completion.
unsafe impl<T: 'static> Operation for FrontendExecution<T> {
    type Pending = Self::Output;

    type Output = T;

    fn enqueue(mut self, execution: &mut Execution) -> CudaResult<T> {
        let resources = take(&mut self.resources);
        // SAFETY: `resources` holds the bindings, workspace, context, graph and invocation that `create` retained and the accesses it declared, and `execute_on` enqueues only on the supplied stream.
        unsafe {
            execution.enqueue(resources, |stream| self.execute_on(stream))?;
        }
        Ok(self.output)
    }

    unsafe fn resolve(pending: Self::Pending) -> CudaResult<Self::Output> {
        Ok(pending)
    }
}

// SAFETY: `resolve` returns `pending` unchanged, and `enqueue` schedules the native work without host resolution.
unsafe impl<T: 'static> DeviceOperation for FrontendExecution<T> {}

// SAFETY: `record` registers the owners and accesses that `create` collected before recording, the retained invocation and workspace stay valid for every replay, and the output exposes only pending storage.
unsafe impl<T: 'static> Recordable for FrontendExecution<T> {
    type Output = T;

    fn record(mut self, recording: &mut Recording) -> CudaResult<T> {
        let resources = take(&mut self.resources);
        // SAFETY: `resources` retains every binding owner, the workspace and the invocation for replay, and `execute_on` only enqueues the prepared plan on the supplied stream, which is graph-safe.
        unsafe {
            recording.enqueue(resources, |stream| self.execute_on(stream))?;
        }
        Ok(self.output)
    }
}
