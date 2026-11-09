#[cfg(test)]
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::{
    collections::HashMap,
    mem::MaybeUninit,
    ptr,
    sync::{Arc, Mutex, OnceLock},
};

use singe_cuda::{
    graph::Graph as NativeCudaGraph,
    memory::DeviceMemory,
    stream::{StreamCaptureMode, StreamFlags},
    submission::{AccessMode, Resources},
    view::DeviceRegionProvider,
};
use singe_cudnn::{
    backend::{
        behavior::BackendBehaviorNote,
        execution::{
            EngineChoice, ExecutionPlan as CudnnExecutionPlan, VariantPack as CudnnVariantPack,
        },
        tensor::BackendTensorUid as CudnnTensorUid,
    },
    context::Context as CudnnContext,
    error::Result as CudnnResult,
};

use crate::{
    backend::cuda::{
        CudaStream,
        cudnn_backend::{
            cudnn_backend_tensor_binding, cudnn_engine_choices, types::CudnnBackendGraphTemplate,
        },
        graph_value::CudaGraphValue,
        runtime::CudaRuntime,
        utility::device_ptr,
    },
    data_type::DataType,
    error::{Error, Result},
    graph::ir::ValueId,
    utility::Locked,
};

/// Forces every executable to record graphs by stream capture, as for engines without native CUDA graph support.
#[cfg(test)]
pub static FORCE_CAPTURED_GRAPHS: AtomicBool = AtomicBool::new(false);

/// Counts graphs recorded by the stream-capture fallback.
#[cfg(test)]
pub static CAPTURED_GRAPHS: AtomicUsize = AtomicUsize::new(0);

/// Retains a shared cuDNN graph template, workspace, and graph-value binding descriptions.
/// Native graph population and update use the plan's native graph API when available and stream capture otherwise.
#[derive(Debug)]
pub struct CudnnBackendExecutable {
    template: Arc<Mutex<CudnnBackendGraphTemplate>>,
    workspace: DeviceMemory<MaybeUninit<u8>>,
    bindings: Vec<CudnnGraphBinding>,
    // Graph values no node holds, which this executable retains for its graph's replays.
    values: Vec<(CudnnTensorUid, CudaGraphValue)>,
    // Scalars the plan reads by value, as bits, from an address that lasts as long as this executable.
    scalars: Box<[(CudnnTensorUid, u32)]>,
    /// Whether the plan's engine supports the native CUDA graph API, queried once.
    native_cuda_graph: OnceLock<bool>,
}

/// Maps a native cuDNN tensor ID to a graph node or the current output.
#[derive(Debug, Clone, Copy)]
pub struct CudnnGraphBinding {
    id: CudnnTensorUid,
    source: CudnnGraphBindingSource,
}

/// Source of an allocation address in a cuDNN graph binding.
#[derive(Debug, Clone, Copy)]
pub enum CudnnGraphBindingSource {
    /// Uses the retained graph value associated with this node ID.
    Node(ValueId),
    /// Uses the output value supplied for the current graph binding.
    Output,
}

impl CudnnBackendExecutable {
    /// Retains the supplied template, pending workspace, and binding descriptions.
    /// The constructor does not validate workspace capacity or binding compatibility.
    pub fn new(
        template: Arc<Mutex<CudnnBackendGraphTemplate>>,
        workspace: DeviceMemory<MaybeUninit<u8>>,
        bindings: Vec<CudnnGraphBinding>,
    ) -> Self {
        Self {
            template,
            workspace,
            bindings,
            values: Vec::new(),
            scalars: Box::default(),
            native_cuda_graph: OnceLock::new(),
        }
    }

    // Also binds these graph values, which no node of the graph holds.
    #[must_use]
    pub fn with_values(mut self, values: Vec<(CudnnTensorUid, CudaGraphValue)>) -> Self {
        self.values = values;
        self
    }

    // Also binds these scalars, which the plan reads by value.
    #[must_use]
    pub fn with_scalars(mut self, scalars: Vec<(CudnnTensorUid, u32)>) -> Self {
        self.scalars = scalars.into_boxed_slice();
        self
    }

    // What cuDNN could run this graph with, for tuning.
    pub fn engine_choices(
        &self,
        runtime: &CudaRuntime,
        storage_data_type: DataType,
        matmul: bool,
    ) -> Result<Vec<EngineChoice>> {
        let template = self.template.locked("cudnn backend template")?;
        let cudnn = runtime.cudnn().locked("cuda runtime cache")?;
        cudnn_engine_choices(
            &cudnn,
            template.operation_graph(),
            storage_data_type,
            matmul,
        )
    }

    pub(crate) fn submission_resources(&self) -> Resources {
        let mut resources = Resources::new();
        resources.retain(Arc::clone(&self.template));
        resources.retain(Arc::clone(self.workspace.allocation()));
        resources.access(self.workspace.region(), AccessMode::ReadWrite);
        resources
    }

    /// Returns the retained workspace allocation size in bytes.
    pub fn workspace_byte_size(&self) -> usize {
        self.workspace.len()
    }

    /// Records current graph-value addresses without retaining their allocations.
    /// Native graph population and update retain their separate replay lifetime contracts.
    ///
    /// # Errors
    ///
    /// Returns an error if the template mutex is poisoned, a source node has no retained value, a tensor region is invalid, binding context, type, capacity, or alignment checks fail, or cuDNN cannot create the variant pack.
    pub fn variant_pack(
        &mut self,
        output: &CudaGraphValue,
        cache: &HashMap<ValueId, CudaGraphValue>,
    ) -> Result<CudnnVariantPack> {
        let context = Arc::clone(self.workspace.context());
        let template = self.template.locked("cuda runtime cache")?;
        let mut bindings = self
            .bindings
            .iter()
            .map(|binding| {
                let tensor = match binding.source() {
                    CudnnGraphBindingSource::Node(id) => cache
                        .get(&id)
                        .ok_or_else(|| Error::missing_input("variant pack"))?,
                    CudnnGraphBindingSource::Output => output,
                };
                Ok((binding.id(), tensor))
            })
            .chain(self.values.iter().map(|(id, value)| Ok((*id, value))))
            .map(|binding: Result<(CudnnTensorUid, &CudaGraphValue)>| {
                let (id, tensor) = binding?;
                cudnn_backend_tensor_binding(
                    &template,
                    &context,
                    id,
                    tensor.data_type(),
                    tensor.region()?.region(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        drop(template);
        for (id, bits) in &self.scalars {
            // SAFETY: cuDNN reads this host address as the scalar it is, and the boxed slice keeps it for as long as this executable.
            bindings.push((*id, unsafe { device_ptr(ptr::from_ref(bits)) }));
        }
        // SAFETY: every binding points into a live tensor allocation that the caller retains through execution.
        Ok(unsafe { CudnnVariantPack::create(&context, &bindings, Some(&mut self.workspace))? })
    }

    /// Populates a native graph with the addresses in `variant_pack`.
    ///
    /// # Safety
    /// Retain every bound tensor and this executable's workspace through all graph clones, executable snapshots and pending launches that reference them.
    /// Inputs must be initialized before replay and outputs must permit exclusive writes, with producer dependencies, matching layouts and permitted aliases.
    /// The compiler's graph-value cache and replay resources establish this contract internally; this wrapper does not retain external bindings in the native graph.
    ///
    /// Plans whose engine lacks native CUDA graph support are recorded by stream capture, replacing `graph` with the captured graph.
    ///
    /// # Errors
    ///
    /// Returns an error if the template mutex is poisoned, cuDNN cannot query engine capabilities or populate the graph, or CUDA stream creation, capture, or captured plan execution fails.
    pub unsafe fn populate_graph(
        &self,
        cudnn: &mut CudnnContext,
        variant_pack: &CudnnVariantPack,
        graph: &mut NativeCudaGraph,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's contract, which the helper states for both of its paths, and compiler and replay caches retain the plan, tensors and workspace for the generated graph.
        unsafe {
            self.write_graph(cudnn, variant_pack, graph, |plan, cudnn, graph| {
                plan.populate_graph(cudnn, variant_pack, graph)
            })
        }
    }

    /// Replaces bindings in a previously populated native graph.
    ///
    /// # Safety
    /// Uphold the allocation, layout and access contract of [`Self::populate_graph`] for the replacement bindings and workspace.
    /// Retain previous bindings and workspace while any old executable snapshot or pending launch still uses them, including after a partially failed update.
    ///
    /// Captured graphs are re-captured with the same topology and replace `graph`.
    ///
    /// # Errors
    ///
    /// Returns an error if the template mutex is poisoned, cuDNN cannot query engine capabilities or update the graph, or CUDA stream creation, capture, or captured plan execution fails.
    pub unsafe fn update_graph(
        &self,
        cudnn: &mut CudnnContext,
        variant_pack: &CudnnVariantPack,
        graph: &mut NativeCudaGraph,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's contract, which the helper states for both of its paths, and compiler and replay caches retain the plan, tensors and workspace for the generated graph.
        unsafe {
            self.write_graph(cudnn, variant_pack, graph, |plan, cudnn, graph| {
                plan.update_graph(cudnn, variant_pack, graph)
            })
        }
    }

    // Writes the plan into `graph` through `native`, or captures it where the engine has no native graph API.
    //
    // # Safety
    //
    // The caller upholds the retention and access contract of [`Self::populate_graph`].
    unsafe fn write_graph(
        &self,
        cudnn: &mut CudnnContext,
        variant_pack: &CudnnVariantPack,
        graph: &mut NativeCudaGraph,
        native: impl FnOnce(
            &CudnnExecutionPlan,
            &mut CudnnContext,
            &mut NativeCudaGraph,
        ) -> CudnnResult<()>,
    ) -> Result<()> {
        let plan = self.plan()?;
        if self.supports_native_cuda_graph(&plan)? {
            native(&plan, cudnn, graph)?;
        } else {
            // SAFETY: the caller upholds the same retention and access contract for capture.
            *graph = unsafe { capture_plan(cudnn, &plan, variant_pack) }?;
            #[cfg(test)]
            CAPTURED_GRAPHS.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    fn supports_native_cuda_graph(&self, plan: &CudnnExecutionPlan) -> Result<bool> {
        #[cfg(test)]
        if FORCE_CAPTURED_GRAPHS.load(Ordering::Relaxed) {
            return Ok(false);
        }

        if let Some(&native) = self.native_cuda_graph.get() {
            return Ok(native);
        }

        let native = plan
            .engine_config()?
            .engine()?
            .behavior_notes()?
            .contains(&BackendBehaviorNote::SupportsCudaGraphNativeApi);
        Ok(*self.native_cuda_graph.get_or_init(|| native))
    }

    fn plan(&self) -> Result<CudnnExecutionPlan> {
        Ok(self.template.locked("cuda runtime cache")?.plan().clone())
    }
}

/// Records one plan execution on a dedicated stream of the plan's CUDA context.
///
/// # Safety
/// Uphold the retention and access contract of [`CudnnBackendExecutable::populate_graph`].
unsafe fn capture_plan(
    cudnn: &mut CudnnContext,
    plan: &CudnnExecutionPlan,
    variant_pack: &CudnnVariantPack,
) -> Result<NativeCudaGraph> {
    let stream = CudaStream::new(Arc::new(
        cudnn
            .cuda_context()
            .create_stream_with_flags(StreamFlags::DEFAULT)?,
    ));
    cudnn.with_stream(Some(stream.raw()), |cudnn| {
        let (graph, ()) = stream.capture(StreamCaptureMode::Relaxed, || {
            // SAFETY: the caller retains the plan, bindings and workspace through every replay.
            unsafe { plan.execute(cudnn, variant_pack) }?;
            Ok(())
        })?;
        Ok(graph.into_native())
    })
}

impl CudnnGraphBinding {
    /// Associates a native tensor ID with a graph-value source without validating either.
    #[must_use]
    pub fn new(id: CudnnTensorUid, source: CudnnGraphBindingSource) -> Self {
        Self { id, source }
    }

    /// Returns the native tensor ID for this binding.
    #[must_use]
    pub fn id(&self) -> CudnnTensorUid {
        self.id
    }

    /// Returns the graph-value source that supplies the allocation address.
    #[must_use]
    pub fn source(&self) -> CudnnGraphBindingSource {
        self.source
    }
}
