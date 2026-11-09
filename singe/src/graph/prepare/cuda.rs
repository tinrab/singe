use std::{collections::HashMap, sync::Arc};

use singe_cuda::architecture::GpuArchitecture;

use crate::{
    backend::cuda::{
        DynamicShapePolicy, KernelLanguage, KernelRecorder, LoweringPolicy,
        compiled::plan::memory::GraphArena,
        tuning::{LibraryRecorder, TuningRecords},
    },
    error::{Error, Result},
    tensor::Tensor,
    utility::to_u64,
};

/// Optional CUDA provider and shape controls for common graph preparation.
/// Buckets requiring padding are rejected before any replay is submitted.
#[derive(Debug, Clone)]
pub struct GraphCudaOptions {
    pub(crate) lowering: LoweringPolicy,
    pub(crate) shapes: DynamicShapePolicy,
    // What the tuning step chose, which this compile reads and never measures.
    pub(crate) tuning_records: Option<Arc<TuningRecords>>,
    // The language generated kernels are emitted in.
    pub(crate) kernel_language: KernelLanguage,
    // Collects every generated kernel the graph plans, for printing or offline tuning.
    pub(crate) kernel_recorder: Option<KernelRecorder>,
    // Collects what libraries could run each node with, for tuning.
    pub(crate) library_recorder: Option<LibraryRecorder>,
    // The architecture generated kernels are planned for, or `None` for the graph's device.
    pub(crate) target_architecture: Option<GpuArchitecture>,
    /// The runtime inputs replays read where these tensors are, by input name.
    pub(crate) inputs_in_place: HashMap<String, Tensor>,
    // The buffers an executor's specializations share; each executor instance has its own.
    pub(crate) arena: Option<GraphArena>,
    // The buffers that executor shares with others that run one at a time, in place of ones of its own.
    pub(crate) shared_arena: Option<GraphArena>,
}

impl GraphCudaOptions {
    /// Creates CUDA preparation settings with default providers and exact-shape specialization.
    #[must_use]
    pub fn new() -> Self {
        Self {
            lowering: LoweringPolicy::default(),
            shapes: DynamicShapePolicy::exact_shape(),
            tuning_records: None,
            kernel_language: KernelLanguage::default(),
            kernel_recorder: None,
            library_recorder: None,
            target_architecture: None,
            inputs_in_place: HashMap::new(),
            arena: None,
            shared_arena: None,
        }
    }

    // Lists what libraries could run each node with, as this graph compiles.
    pub(crate) fn with_library_recorder(mut self, recorder: LibraryRecorder) -> Self {
        self.library_recorder = Some(recorder);
        self
    }

    // Reads what `tune_graph` chose; a record that no longer fits falls back, and the report says so.
    pub(crate) fn merges_projections(&self) -> bool {
        self.tuning_records
            .as_ref()
            .is_some_and(|records| records.merges_projections())
    }

    #[must_use]
    pub fn with_tuning_records(mut self, records: TuningRecords) -> Self {
        self.tuning_records = Some(Arc::new(records));
        self
    }

    #[must_use]
    pub const fn with_kernel_language(mut self, language: KernelLanguage) -> Self {
        self.kernel_language = language;
        self
    }

    #[must_use]
    pub fn with_kernel_recorder(mut self, recorder: KernelRecorder) -> Self {
        self.kernel_recorder = Some(recorder);
        self
    }

    // Plans generated kernels for `architecture` instead of the device's, so their artifacts can be rendered for a GPU the machine lacks; such a graph cannot run here.
    #[must_use]
    pub const fn with_target_architecture(mut self, architecture: GpuArchitecture) -> Self {
        self.target_architecture = Some(architecture);
        self
    }

    /// Reads the runtime inputs named in `inputs` where those tensors are, instead of copying them into the graph's placeholders on every replay.
    ///
    /// Replays must pass the same tensors for those inputs, as a server passes its persistent KV caches; their contents may change between replays.
    /// A scatter whose result replaces such an input writes it where it is, if nothing else reads its old values and no output keeps its new ones, so a replay's cost does not depend on the input's size.
    /// An input whose tensor does not fit a compiled specialization's input metadata is copied, as other inputs are.
    #[must_use]
    pub fn with_inputs_in_place(
        mut self,
        inputs: impl IntoIterator<Item = (String, Tensor)>,
    ) -> Self {
        self.inputs_in_place.extend(inputs);
        self
    }

    pub(crate) fn with_only_inputs_in_place(
        mut self,
        inputs: impl IntoIterator<Item = (String, Tensor)>,
    ) -> Self {
        self.inputs_in_place = inputs.into_iter().collect();
        self.shared_arena = None;
        self
    }

    /// Replaces the CUDA provider eligibility policy.
    #[must_use]
    pub fn with_lowering_policy(mut self, policy: LoweringPolicy) -> Self {
        self.lowering = policy;
        self
    }

    /// Replaces the dynamic shape specialization and recording policy.
    #[must_use]
    pub fn with_dynamic_shape_policy(mut self, policy: DynamicShapePolicy) -> Self {
        self.shapes = policy;
        self
    }

    pub(crate) fn validate_next_recording(&self, recordings: u64) -> Result<()> {
        if let Some(max) = self.shapes.max_recordings_per_graph() {
            let max = to_u64(max, "cuda graph recording limit")?;
            if recordings >= max {
                return Err(Error::unsupported_operation_with_details(
                    "cuda graph dynamic shape recording limit",
                    format!("recorded {recordings} specializations, limit {max}"),
                ));
            }
        }
        Ok(())
    }
}

impl Default for GraphCudaOptions {
    fn default() -> Self {
        Self::new()
    }
}
