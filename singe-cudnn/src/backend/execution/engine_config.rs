use std::{
    collections::HashSet,
    fmt::Write as _,
    sync::{Arc, Mutex, OnceLock},
};

use singe_cuda::context::Context as CudaContext;

use crate::{
    backend::{
        attribute::{BackendAttributeId, BackendAttributeName, BackendAttributeType},
        descriptor::{BackendDescriptor, BackendDescriptorType},
        device::{DeviceProperties, DevicePropertiesSource},
        engine::{Engine, EngineIndex},
        execution::{IntermediateInfo, OperationGraph},
        heuristic::{BackendHeuristicMode, HeuristicQueryPolicy},
        knob::BackendKnobType,
        layout::BackendLayoutType,
        tensor::BackendTensorUid,
    },
    context::Context,
    error::{Error, Result, Status},
    frontend::error::Error as FrontendError,
    utility::{bind_context, bind_for_drop, to_i32, to_usize, validate_context_pair},
};

/// Native resource targets used during engine selection and reconstruction.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EnginePlanningConfig {
    /// Number of SMs to target. Zero uses every SM; `None` leaves the target unset.
    pub sm_count_target: Option<u32>,
    /// Dynamic shared-memory budget in bytes. Zero uses the device limit.
    pub shared_memory_limit: Option<u32>,
}

/// Heuristic query over a retained operation graph and its actual live CUDA target.
/// Returned configurations retain that target even when the operation graph was prepared deviceless.
#[derive(Debug)]
pub struct EngineHeuristics<'a> {
    planning: EnginePlanningConfig,
    descriptor: BackendDescriptor,
    cuda_context: Option<Arc<CudaContext>>,
    _operation_graph: &'a OperationGraph,
}

/// One attempted heuristic mode and its native result before frontend metadata filtering.
#[derive(Debug)]
pub struct HeuristicQueryStatus {
    /// Native heuristic mode attempted by this query.
    pub mode: BackendHeuristicMode,
    /// Number of configurations returned, or the original typed query error.
    /// A successful query may return zero configurations.
    pub result: Result<usize>,
}

/// Owned configurations and per-mode outcomes from an ordered heuristic query.
/// Successful modes contribute configurations in their requested order, including duplicates.
/// Configurations retain their actual CUDA target and planning settings after queries are released.
#[derive(Debug)]
pub struct EngineHeuristicsResult {
    /// Owned configurations from successful modes, in query and native result order.
    pub engine_configs: Vec<EngineConfig>,
    /// Only attempted modes appear here; stop-on-error omits later modes.
    pub statuses: Vec<HeuristicQueryStatus>,
}

/// Supported value range and increment for an engine-specific tuning knob.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineKnobInfo {
    /// Native tuning knob described by these bounds.
    pub knob_type: BackendKnobType,
    /// Native lower bound for this knob.
    pub minimum_value: i64,
    /// Native upper bound for this knob.
    pub maximum_value: i64,
    /// Native increment between supported knob values.
    pub stride: i64,
}

/// One requested value for a native engine tuning knob.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct KnobChoiceInfo {
    /// Native tuning knob to set.
    pub knob_type: BackendKnobType,
    /// Requested engine-specific knob value.
    pub value: i64,
}

impl KnobChoiceInfo {
    /// Chooses a native knob value; engine-specific support is resolved during configuration.
    #[must_use]
    pub const fn new(knob_type: BackendKnobType, value: i64) -> Self {
        Self { knob_type, value }
    }
}

/// Engine index and performance-knob choices shared by explicit planning and engine-choice caches.
/// Native construction resolves graph, target and engine-specific support independently.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EngineChoice {
    /// Global native engine index selected for the operation graph.
    pub engine_index: EngineIndex,
    /// Requested tuning values; an empty list selects engine defaults.
    pub knob_choices: Vec<KnobChoiceInfo>,
}

impl EngineChoice {
    /// Records an explicit engine selection without allocating native descriptors.
    /// Empty knob choices leave the engine defaults in effect.
    pub fn new(engine_index: EngineIndex, knob_choices: impl Into<Vec<KnobChoiceInfo>>) -> Self {
        Self {
            engine_index,
            knob_choices: knob_choices.into(),
        }
    }

    /// Describes this choice using the canonical native engine/knob tag.
    /// Sorts a copy by numeric knob identity without changing the recorded request order.
    /// Formatting requires no CUDA context, native descriptors or support queries.
    #[must_use]
    pub fn tag(&self) -> String {
        let mut choices = self.knob_choices.clone();
        Self::format_tag(self.engine_index, &mut choices)
    }

    fn format_tag(engine_index: EngineIndex, choices: &mut [KnobChoiceInfo]) -> String {
        let mut tag = format!("eng{engine_index}");
        choices.sort_unstable_by_key(|choice| choice.knob_type);
        for choice in choices {
            write!(tag, "_k{}={}", choice.knob_type.raw(), choice.value)
                .expect("formatting into a string cannot fail");
        }
        tag
    }
}

/// Native preferred memory layouts associated with a tensor identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutInfo {
    /// Native tensor UID whose preferred layouts are described here.
    pub tensor_uid: BackendTensorUid,
    /// Native preferred memory-layout types for this tensor.
    pub types: Vec<BackendLayoutType>,
}

/// Owns a finalized engine configuration and retains its immutable engine descriptor.
/// Live configurations retain their actual CUDA context through native destruction.
#[derive(Debug)]
pub struct EngineConfig {
    planning: OnceLock<EnginePlanningConfig>,
    descriptor: BackendDescriptor,
    cuda_context: Option<Arc<CudaContext>>,
    engine: OnceLock<Engine>,
    choice: OnceLock<EngineChoice>,
    tag: OnceLock<String>,
}

impl EngineConfig {
    /// Resource targets retained when this configuration is rebuilt.
    /// Configurations extracted from native plans recover these values from the engine descriptor.
    /// The first successful read is retained; later reads require no native query or context binding.
    /// Failed native queries remain retryable and run outside the cache initializer.
    ///
    /// # Errors
    ///
    /// Returns an error if the retained CUDA context cannot be bound, the engine cannot be extracted, or a native resource-target query fails for a reason other than an absent or unsupported attribute.
    pub fn planning_config(&self) -> Result<EnginePlanningConfig> {
        if let Some(planning) = self.planning.get() {
            return Ok(*planning);
        }
        bind_context(self.cuda_context())?;
        let engine = self.engine()?.descriptor();
        let attribute = |name| match engine.attribute_i32(name) {
            Ok(value) => Ok(Some(value)),
            Err(
                Error::DescriptorAttributeNotFound(_)
                | Error::Cudnn {
                    code: Status::NotSupported,
                    ..
                },
            ) => Ok(None),
            Err(error) => Err(error),
        };
        let planning = EnginePlanningConfig {
            // cuDNN reports an unset target as a negative value.
            sm_count_target: attribute(BackendAttributeName::EngineSmCountTarget)?
                .and_then(|count| u32::try_from(count).ok()),
            shared_memory_limit: attribute(BackendAttributeName::EngineSharedMemoryLimit)?
                .and_then(|limit| u32::try_from(limit).ok()),
        };
        Ok(*self.planning.get_or_init(|| planning))
    }

    pub(crate) fn from_descriptor(
        descriptor: BackendDescriptor,
        cuda_context: Option<Arc<CudaContext>>,
    ) -> Self {
        Self {
            descriptor,
            cuda_context,
            planning: OnceLock::new(),
            engine: OnceLock::new(),
            choice: OnceLock::new(),
            tag: OnceLock::new(),
        }
    }

    /// Creates an explicit engine configuration for a live context using default resource targets.
    ///
    /// Obtains device properties from the retained live target. Empty knob choices use engine defaults.
    ///
    /// # Errors
    ///
    /// Returns an error if live contexts disagree or cannot be bound, target properties cannot be obtained, a resource target does not fit `i32`, knob choices repeat a knob type, or native descriptor creation, attribute assignment, or finalization fails. cuDNN validates engine and knob support for the selected graph and target.
    pub fn create(
        context: &Context,
        operation_graph: &OperationGraph,
        choice: &EngineChoice,
    ) -> Result<Self> {
        Self::create_with_config(
            context,
            operation_graph,
            choice,
            None,
            EnginePlanningConfig::default(),
        )
    }

    /// Creates an explicit engine configuration with optional device properties and resource targets.
    ///
    /// When `device_properties` is `None`, queries properties from the retained live target.
    ///
    /// # Errors
    ///
    /// Returns an error if live contexts disagree or cannot be bound, target properties cannot be obtained, a resource target does not fit `i32`, knob choices repeat a knob type, or native descriptor creation, attribute assignment, or finalization fails. cuDNN validates engine and knob support for the selected graph and target.
    pub fn create_with_config(
        context: &Context,
        operation_graph: &OperationGraph,
        choice: &EngineChoice,
        device_properties: Option<&DeviceProperties>,
        planning: EnginePlanningConfig,
    ) -> Result<Self> {
        Self::create_for_target(
            Some(context),
            operation_graph,
            choice,
            device_properties,
            planning,
        )
    }

    /// Creates an explicit engine configuration using supplied device properties without a cuDNN handle.
    ///
    /// A graph that retains a live CUDA target still supplies that context for native calls and resource ownership.
    ///
    /// # Errors
    ///
    /// Returns an error if live contexts disagree or cannot be bound, target properties cannot be obtained, a resource target does not fit `i32`, knob choices repeat a knob type, or native descriptor creation, attribute assignment, or finalization fails. cuDNN validates engine and knob support for the selected graph and target.
    pub fn create_deviceless(
        operation_graph: &OperationGraph,
        choice: &EngineChoice,
        device_properties: &DeviceProperties,
        planning: EnginePlanningConfig,
    ) -> Result<Self> {
        Self::create_for_target(
            None,
            operation_graph,
            choice,
            Some(device_properties),
            planning,
        )
    }

    pub(crate) fn create_for_target(
        context: Option<&Context>,
        operation_graph: &OperationGraph,
        choice: &EngineChoice,
        device_properties: Option<&DeviceProperties>,
        planning: EnginePlanningConfig,
    ) -> Result<Self> {
        let knob_choices = &choice.knob_choices;
        let mut knob_types = HashSet::with_capacity(knob_choices.len());
        for choice in knob_choices {
            if !knob_types.insert(choice.knob_type) {
                return Err(Error::DuplicateEngineKnobChoice {
                    knob_type: choice.knob_type,
                });
            }
        }
        let engine = Engine::create_with_context(
            context,
            operation_graph,
            choice.engine_index,
            device_properties,
            planning,
        )?;

        let knob_choice_descriptors = knob_choices
            .iter()
            .map(|choice| {
                let mut descriptor = BackendDescriptor::create(BackendDescriptorType::KnobChoice)?;
                descriptor.set_attribute_enum(
                    BackendAttributeName::KnobChoiceKnobType,
                    choice.knob_type,
                )?;
                descriptor
                    .set_attribute_i64(BackendAttributeName::KnobChoiceKnobValue, choice.value)?;
                descriptor.finalize()?;
                Ok(descriptor)
            })
            .collect::<Result<Vec<_>>>()?;

        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::EngineCfg)?;
        descriptor
            .set_attribute_descriptor(BackendAttributeName::EngineCfgEngine, engine.descriptor())?;
        if !knob_choice_descriptors.is_empty() {
            let knob_choice_refs = knob_choice_descriptors.iter().collect::<Vec<_>>();
            descriptor.set_attribute_descriptor_slice(
                BackendAttributeName::EngineCfgKnobChoices,
                &knob_choice_refs,
            )?;
        }
        descriptor.finalize()?;

        Ok(Self {
            descriptor,
            cuda_context: context
                .map(|context| Arc::clone(context.cuda_context()))
                .or_else(|| operation_graph.cuda_context().map(Arc::clone)),
            planning: OnceLock::from(planning),
            engine: OnceLock::from(engine),
            choice: OnceLock::new(),
            tag: OnceLock::new(),
        })
    }

    /// Returns the native execution-workspace requirement in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding or the native attribute query fails, or the reported size is negative or cannot be represented as `usize`.
    pub fn workspace_size(&self) -> Result<usize> {
        bind_context(self.cuda_context())?;
        to_usize(
            self.descriptor
                .attribute_i64(BackendAttributeName::EngineCfgWorkspaceSize)?,
            "workspace_size",
        )
    }

    /// Queries the engine configuration's shared-memory requirement in bytes.
    /// Returns `None` when the native attribute is absent or unsupported.
    /// A reported zero-byte requirement is `Some(0)`.
    ///
    /// # Errors
    /// Returns native query errors other than an unavailable attribute, or an
    /// error if the reported size cannot be represented as a byte count.
    pub fn shared_memory_size(&self) -> Result<Option<usize>> {
        bind_context(self.cuda_context())?;
        match self
            .descriptor
            .attribute_i32(BackendAttributeName::EngineCfgSharedMemoryUsed)
        {
            Ok(shared_memory_size) => to_usize(shared_memory_size, "shared_memory_size").map(Some),
            Err(
                Error::DescriptorAttributeNotFound(_)
                | Error::Cudnn {
                    code: Status::NotSupported,
                    ..
                },
            ) => Ok(None),
            Err(error) => Err(error),
        }
    }

    /// Borrows the supported knobs retained by this configuration's finalized engine.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding, engine extraction, or native knob-metadata querying or decoding fails.
    pub fn knob_infos(&self) -> Result<&[EngineKnobInfo]> {
        self.engine()?.knob_infos()
    }

    /// Borrows the native-order knob choices from the retained engine-choice snapshot.
    /// The first successful read retains the selection; failed native queries remain retryable.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding, engine extraction, or native knob-choice querying or decoding fails.
    pub fn knob_choices(&self) -> Result<&[KnobChoiceInfo]> {
        Ok(&self.choice()?.knob_choices)
    }

    fn read_native_knob_choices(&self) -> Result<Vec<KnobChoiceInfo>> {
        bind_context(self.cuda_context())?;
        // SAFETY: this attribute returns objects of the explicitly selected native class.
        let descriptors = unsafe {
            self.descriptor.attribute_descriptor_slice(
                BackendAttributeName::EngineCfgKnobChoices,
                BackendDescriptorType::KnobChoice,
                None,
            )
        }?;

        descriptors
            .into_iter()
            .map(|descriptor| {
                Ok(KnobChoiceInfo {
                    knob_type: descriptor
                        .attribute_enum(BackendAttributeName::KnobChoiceKnobType)?,
                    value: descriptor.attribute_i64(BackendAttributeName::KnobChoiceKnobValue)?,
                })
            })
            .collect()
    }

    /// Queries reusable execution intermediates and their storage and tensor dependencies.
    /// Unavailable attribute-dependency metadata does not discard the remaining information.
    /// Other native query and size-conversion errors propagate.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding, native descriptor or attribute queries, or metadata decoding fails, or an intermediate size cannot be represented as `usize`. Missing or unsupported attribute-dependency metadata is returned as `None`.
    pub fn intermediate_infos(&self) -> Result<Vec<IntermediateInfo>> {
        bind_context(self.cuda_context())?;
        // SAFETY: this attribute returns objects of the explicitly selected native class.
        let descriptors = unsafe {
            self.descriptor.attribute_descriptor_slice(
                BackendAttributeName::EngineCfgIntermediateInfo,
                BackendDescriptorType::IntermediateInfo,
                None,
            )
        }?;

        descriptors
            .into_iter()
            .map(|descriptor| {
                let size = to_usize(
                    descriptor.attribute_i64(BackendAttributeName::IntermediateInfoSize)?,
                    "size",
                )?;
                let dependent_attributes = match descriptor
                    .attribute_enum_slice::<BackendAttributeId>(
                        BackendAttributeName::IntermediateInfoDependentAttributes,
                    ) {
                    Ok(attributes) => Some(attributes),
                    Err(
                        Error::DescriptorAttributeNotFound(_)
                        | Error::Cudnn {
                            code: Status::NotSupported,
                            ..
                        },
                    ) => None,
                    Err(error) => return Err(error),
                };

                Ok(IntermediateInfo {
                    unique_id: descriptor
                        .attribute_i64(BackendAttributeName::IntermediateInfoUniqueId)?
                        .into(),
                    size,
                    dependent_tensor_uids: descriptor
                        .attribute_i64_slice(
                            BackendAttributeName::IntermediateInfoDependentTensorUids,
                        )?
                        .into_iter()
                        .map(BackendTensorUid::from)
                        .collect(),
                    dependent_attributes,
                })
            })
            .collect()
    }

    /// Returns the global native index of the retained engine.
    ///
    /// # Errors
    ///
    /// Returns an error if an uncached engine cannot be extracted, its context cannot be bound, or its native index is invalid.
    pub fn engine_index(&self) -> Result<EngineIndex> {
        Ok(self.engine()?.index())
    }

    /// Borrows the retained engine and knob choices for explicit planning or a choice cache.
    /// The first successful read records native metadata; later reads reuse the same snapshot.
    /// Failed queries remain retryable, and native queries run outside the cache initializer.
    /// Device properties and resource targets are supplied separately when rebuilding.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding, engine extraction, or native knob-choice querying or decoding fails.
    pub fn choice(&self) -> Result<&EngineChoice> {
        if let Some(choice) = self.choice.get() {
            return Ok(choice);
        }
        let choice = EngineChoice::new(self.engine_index()?, self.read_native_knob_choices()?);
        Ok(self.choice.get_or_init(|| choice))
    }

    /// Borrows the canonical engine/knob description used by plan names, filters and deduplication.
    /// Successful reads retain the string without changing the snapshot's native knob order.
    ///
    /// # Errors
    ///
    /// Returns an error if the engine and knob-choice snapshot cannot be obtained.
    pub fn tag(&self) -> Result<&str> {
        if let Some(tag) = self.tag.get() {
            return Ok(tag.as_str());
        }
        let tag = self.choice()?.tag();
        Ok(self.tag.get_or_init(|| tag).as_str())
    }

    /// Borrows the retained finalized engine for metadata and knob queries.
    /// Explicit construction retains its engine immediately; extracted configurations read it lazily.
    /// Successful reads are reused, while failed reads remain retryable.
    /// Native queries complete before installing the cached owner.
    ///
    /// # Errors
    /// Returns native extraction or engine-index errors when the engine has not been retained yet.
    pub fn engine(&self) -> Result<&Engine> {
        if let Some(engine) = self.engine.get() {
            return Ok(engine);
        }
        bind_context(self.cuda_context())?;
        // SAFETY: the named attribute requires the explicitly selected native class.
        let descriptor = unsafe {
            self.descriptor.attribute_descriptor(
                BackendAttributeName::EngineCfgEngine,
                BackendDescriptorType::Engine,
            )
        }?;
        let engine =
            Engine::from_descriptor(descriptor, self.cuda_context.as_ref().map(Arc::clone))?;
        Ok(self.engine.get_or_init(|| engine))
    }

    /// Borrows the finalized native engine-configuration descriptor.
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }

    /// Returns the actual retained CUDA context, or None for deviceless metadata.
    pub fn cuda_context(&self) -> Option<&Arc<CudaContext>> {
        self.cuda_context.as_ref()
    }
}

impl<'a> EngineHeuristics<'a> {
    /// Queries native modes in order, preserving successful configurations and every attempted outcome.
    /// Pass a live handle, attached device properties or a graph retaining its live target.
    /// Resource settings apply to each mode and follow the extracted configurations.
    /// Empty mode lists perform no queries. No engine enumeration or implicit mode fallback is added.
    /// Query construction and result-extraction failures are retained in `statuses`.
    /// `StopOnError` stops on the first failure, rather than on the first successful mode.
    #[must_use]
    pub fn query_modes(
        context: Option<&Context>,
        graph: &'a OperationGraph,
        modes: &[BackendHeuristicMode],
        device_properties: Option<&DeviceProperties>,
        planning: EnginePlanningConfig,
        policy: HeuristicQueryPolicy,
    ) -> EngineHeuristicsResult {
        let mut result = EngineHeuristicsResult {
            engine_configs: Vec::new(),
            statuses: Vec::with_capacity(modes.len()),
        };
        for &mode in modes {
            let configurations =
                Self::create_for_target(context, graph, mode, device_properties, planning)
                    .and_then(|query| query.engine_configs(None));
            let status = configurations.map(|configurations| {
                let count = configurations.len();
                result.engine_configs.extend(configurations);
                count
            });
            let failed = status.is_err();
            result.statuses.push(HeuristicQueryStatus {
                mode,
                result: status,
            });
            if failed && policy == HeuristicQueryPolicy::StopOnError {
                break;
            }
        }
        result
    }

    /// Creates a heuristic query for a live context with an optional SM-count target.
    ///
    /// Queries device properties from the live target and leaves the shared-memory limit unset.
    ///
    /// # Errors
    ///
    /// Returns an error if live contexts disagree or cannot be bound, target properties cannot be obtained, a resource target does not fit `i32`, native descriptor creation, attribute assignment, or finalization fails, or the mutex serializing mode B finalization is poisoned.
    pub fn create(
        context: &Context,
        graph: &'a OperationGraph,
        mode: BackendHeuristicMode,
        sm_count_target: Option<u32>,
    ) -> Result<Self> {
        Self::create_with_config(context, graph, mode, sm_count_target, None, None)
    }

    /// Creates a heuristic query with optional device properties and resource targets.
    ///
    /// An absent device-property argument queries the live target. Shared-memory limits are measured in bytes; SM-count and shared-memory targets use native zero-value defaults.
    ///
    /// # Errors
    ///
    /// Returns an error if live contexts disagree or cannot be bound, target properties cannot be obtained, a resource target does not fit `i32`, native descriptor creation, attribute assignment, or finalization fails, or the mutex serializing mode B finalization is poisoned.
    pub fn create_with_config(
        context: &Context,
        graph: &'a OperationGraph,
        mode: BackendHeuristicMode,
        sm_count_target: Option<u32>,
        device_properties: Option<&DeviceProperties>,
        shared_memory_limit: Option<u32>,
    ) -> Result<Self> {
        Self::create_for_target(
            Some(context),
            graph,
            mode,
            device_properties,
            EnginePlanningConfig {
                sm_count_target,
                shared_memory_limit,
            },
        )
    }

    /// Creates a heuristic query using explicit device properties without a cuDNN handle.
    ///
    /// A graph retaining a live CUDA target still supplies that context for native calls. Resource targets follow the extracted configurations.
    ///
    /// # Errors
    ///
    /// Returns an error if live contexts disagree or cannot be bound, target properties cannot be obtained, a resource target does not fit `i32`, native descriptor creation, attribute assignment, or finalization fails, or the mutex serializing mode B finalization is poisoned.
    pub fn create_deviceless(
        graph: &'a OperationGraph,
        mode: BackendHeuristicMode,
        sm_count_target: Option<u32>,
        device_properties: &DeviceProperties,
        shared_memory_limit: Option<u32>,
    ) -> Result<Self> {
        Self::create_for_target(
            None,
            graph,
            mode,
            Some(device_properties),
            EnginePlanningConfig {
                sm_count_target,
                shared_memory_limit,
            },
        )
    }

    pub(crate) fn create_for_target(
        context: Option<&Context>,
        graph: &'a OperationGraph,
        mode: BackendHeuristicMode,
        device_properties: Option<&DeviceProperties>,
        planning: EnginePlanningConfig,
    ) -> Result<Self> {
        validate_context_pair(
            context.map(Context::cuda_context),
            graph.cuda_context(),
            "operation graph",
        )?;
        let cuda_context = context
            .map(Context::cuda_context)
            .or_else(|| graph.cuda_context());
        bind_context(cuda_context)?;
        let automatic_properties = if device_properties.is_none() {
            cuda_context
                .map(|cuda_context| {
                    DeviceProperties::create(DevicePropertiesSource::Device(cuda_context.device()))
                })
                .transpose()?
        } else {
            None
        };
        let properties = device_properties
            .or(automatic_properties.as_ref())
            .ok_or(FrontendError::DevicelessRequiresDeviceProperties)?;

        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::EngineHeur)?;
        descriptor.set_attribute_enum(BackendAttributeName::EngineHeurMode, mode)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::EngineHeurOperationGraph,
            graph.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::EngineHeurDeviceProp,
            properties.descriptor(),
        )?;
        if let Some(sm_count_target) = planning.sm_count_target {
            descriptor.set_attribute_i32(
                BackendAttributeName::EngineHeurSmCountTarget,
                to_i32(sm_count_target, "sm_count_target")?,
            )?;
        }
        if let Some(limit) = planning.shared_memory_limit {
            descriptor.set_attribute_i32(
                BackendAttributeName::EngineHeurSharedMemoryLimit,
                to_i32(limit, "shared_memory_limit")?,
            )?;
        }
        {
            static MODE_B_FINALIZATION: Mutex<()> = Mutex::new(());
            let _guard = if mode == BackendHeuristicMode::B {
                Some(
                    MODE_B_FINALIZATION
                        .lock()
                        .map_err(|_| Error::EngineHeuristicsLockPoisoned)?,
                )
            } else {
                None
            };
            descriptor.finalize()?;
        }

        Ok(Self {
            descriptor,
            cuda_context: cuda_context.map(Arc::clone),
            _operation_graph: graph,
            planning,
        })
    }

    /// Queries the number of available engine configurations without materializing their descriptors.
    ///
    /// # Errors
    /// Returns context-binding, native query or unrepresentable-count errors.
    pub fn engine_config_count(&self) -> Result<usize> {
        bind_context(self.cuda_context())?;
        self.descriptor.attribute_count(
            BackendAttributeName::EngineHeurResults,
            BackendAttributeType::BackendDescriptor,
        )
    }

    /// Extracts a prefix of owned engine configurations in native heuristic order.
    /// `None` retrieves every result; a maximum of zero materializes no descriptors.
    /// Returned configurations retain this query's resource targets and CUDA context.
    /// The configurations remain usable after the heuristic query is released.
    ///
    /// # Errors
    /// Returns context-binding, native attribute-query or descriptor-count errors.
    pub fn engine_configs(&self, maximum: impl Into<Option<usize>>) -> Result<Vec<EngineConfig>> {
        bind_context(self.cuda_context())?;
        // SAFETY: heuristic results populate owned engine-configuration descriptors.
        let descriptors = unsafe {
            self.descriptor.attribute_descriptor_slice(
                BackendAttributeName::EngineHeurResults,
                BackendDescriptorType::EngineCfg,
                maximum.into(),
            )
        }?;
        Ok(descriptors
            .into_iter()
            .map(|descriptor| {
                let mut config =
                    EngineConfig::from_descriptor(descriptor, self.cuda_context.clone());
                config.planning = OnceLock::from(self.planning);
                config
            })
            .collect())
    }

    /// Actual retained CUDA context, including live queries of a deviceless operation graph.
    #[must_use]
    pub fn cuda_context(&self) -> Option<&Arc<CudaContext>> {
        self.cuda_context.as_ref()
    }

    /// Borrows the finalized native heuristic-query descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

impl Drop for EngineConfig {
    fn drop(&mut self) {
        bind_for_drop(self.cuda_context(), "cudnn engine configuration");
    }
}

impl Drop for EngineHeuristics<'_> {
    fn drop(&mut self) {
        bind_for_drop(self.cuda_context(), "cudnn engine heuristics");
    }
}
