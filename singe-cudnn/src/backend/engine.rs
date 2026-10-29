use std::{
    fmt::{self, Display, Formatter},
    result::Result as StdResult,
    sync::{Arc, OnceLock},
};

use bomboni_serde::{self as serde_helpers, as_string};
use serde::{Deserialize, Deserializer, Serialize, de::Error as DeError};
use singe_cuda::context::Context as CudaContext;

use crate::{
    backend::{
        attribute::BackendAttributeName,
        behavior::{BackendBehaviorNote, BackendNumericalNote},
        descriptor::{BackendDescriptor, BackendDescriptorType},
        device::{DeviceProperties, DevicePropertiesSource},
        execution::{EngineKnobInfo, EnginePlanningConfig, LayoutInfo, OperationGraph},
    },
    context::Context,
    error::{Error, Result},
    frontend::error::Error as FrontendError,
    utility::{bind_context, bind_for_drop, to_i32, to_i64, validate_context_pair},
};

/// Owns a finalized native engine descriptor for one operation graph and compilation target.
/// Metadata queries use this descriptor without rebuilding an engine for the current device.
/// Retains its actual CUDA context when created against a live target and destroys the descriptor on drop.
#[derive(Debug)]
pub struct Engine {
    descriptor: BackendDescriptor,
    cuda_context: Option<Arc<CudaContext>>,
    index: EngineIndex,
    knob_infos: OnceLock<Vec<EngineKnobInfo>>,
    behavior_notes: OnceLock<Vec<BackendBehaviorNote>>,
    numerical_notes: OnceLock<Vec<BackendNumericalNote>>,
    layout_infos: OnceLock<Vec<LayoutInfo>>,
}

/// Global native engine index used by metadata queries and explicit planning.
/// Nonnegative by construction; runtime values use [`Self::try_new`] or `TryFrom<i64>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct EngineIndex(#[serde(with = "serde_helpers::as_string")] i64);

impl EngineIndex {
    /// Creates a known nonnegative index, including in constant declarations.
    /// Use [`Self::try_new`] for runtime values that may be negative.
    ///
    /// # Panics
    /// Panics if `value` is negative.
    #[must_use]
    pub const fn new(value: i64) -> Self {
        assert!(value >= 0, "engine index must be nonnegative");
        Self(value)
    }

    /// Returns a valid index, or `None` for a negative runtime value.
    #[must_use]
    pub const fn try_new(value: i64) -> Option<Self> {
        if value >= 0 { Some(Self(value)) } else { None }
    }

    /// Returns the nonnegative global native engine index.
    #[must_use]
    pub const fn as_i64(self) -> i64 {
        self.0
    }
}

impl Display for EngineIndex {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl TryFrom<i64> for EngineIndex {
    type Error = Error;

    fn try_from(value: i64) -> Result<Self> {
        Self::try_new(value).ok_or(Error::EngineIndexInvalid { value })
    }
}

impl<'de> Deserialize<'de> for EngineIndex {
    fn deserialize<D>(deserializer: D) -> StdResult<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value: i64 = as_string::deserialize(deserializer)?;
        Self::try_from(value).map_err(DeError::custom)
    }
}

impl Engine {
    /// Creates an engine using the operation graph and the supplied context's device.
    ///
    /// # Errors
    ///
    /// Returns an error if live contexts disagree or cannot be bound, device properties cannot be queried, or native descriptor creation, attribute assignment, or engine finalization fails.
    pub fn create(context: &Context, graph: &OperationGraph, index: EngineIndex) -> Result<Self> {
        Self::create_with_config(context, graph, index, None, EnginePlanningConfig::default())
    }

    /// Creates an engine with explicit device properties and native resource targets.
    ///
    /// # Errors
    /// Returns context, target-property, attribute or native engine-finalization errors.
    pub fn create_with_config(
        context: &Context,
        graph: &OperationGraph,
        index: EngineIndex,
        properties: Option<&DeviceProperties>,
        planning: EnginePlanningConfig,
    ) -> Result<Self> {
        Self::create_with_context(Some(context), graph, index, properties, planning)
    }

    /// Creates an engine for explicit device properties without a live cuDNN context.
    ///
    /// # Errors
    ///
    /// Returns an error if a retained graph context cannot be bound, a resource target does not fit `i32`, or native descriptor creation, attribute assignment, or engine finalization fails.
    pub fn create_deviceless(
        graph: &OperationGraph,
        index: EngineIndex,
        properties: &DeviceProperties,
        planning: EnginePlanningConfig,
    ) -> Result<Self> {
        Self::create_with_context(None, graph, index, Some(properties), planning)
    }

    pub(crate) fn create_with_context(
        context: Option<&Context>,
        graph: &OperationGraph,
        index: EngineIndex,
        properties: Option<&DeviceProperties>,
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
        let automatic_properties = if properties.is_none() {
            cuda_context
                .map(|cuda_context| {
                    DeviceProperties::create(DevicePropertiesSource::Device(cuda_context.device()))
                })
                .transpose()?
        } else {
            None
        };
        let properties = properties
            .or(automatic_properties.as_ref())
            .ok_or(FrontendError::DevicelessRequiresDeviceProperties)?;
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::Engine)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::EngineOperationGraph,
            graph.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::EngineDeviceProp,
            properties.descriptor(),
        )?;
        descriptor.set_attribute_i64(BackendAttributeName::EngineGlobalIndex, index.as_i64())?;
        if let Some(count) = planning.sm_count_target {
            descriptor.set_attribute_i32(
                BackendAttributeName::EngineSmCountTarget,
                to_i32(count, "sm_count_target")?,
            )?;
        }
        if let Some(limit) = planning.shared_memory_limit {
            descriptor.set_attribute_i32(
                BackendAttributeName::EngineSharedMemoryLimit,
                to_i32(limit, "shared_memory_limit")?,
            )?;
        }
        descriptor.finalize()?;
        Ok(Self {
            descriptor,
            cuda_context: cuda_context.map(Arc::clone),
            index,
            knob_infos: OnceLock::new(),
            behavior_notes: OnceLock::new(),
            numerical_notes: OnceLock::new(),
            layout_infos: OnceLock::new(),
        })
    }

    pub(crate) fn from_descriptor(
        descriptor: BackendDescriptor,
        cuda_context: Option<Arc<CudaContext>>,
    ) -> Result<Self> {
        bind_context(cuda_context.as_ref())?;
        let index = EngineIndex::try_from(
            descriptor.attribute_i64(BackendAttributeName::EngineGlobalIndex)?,
        )?;
        Ok(Self {
            descriptor,
            cuda_context,
            index,
            knob_infos: OnceLock::new(),
            behavior_notes: OnceLock::new(),
            numerical_notes: OnceLock::new(),
            layout_infos: OnceLock::new(),
        })
    }

    /// Returns the global native index recorded by this engine.
    pub fn index(&self) -> EngineIndex {
        self.index
    }

    /// Borrows the finalized native engine descriptor.
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }

    /// Actual retained context, or None for a deviceless engine.
    pub fn cuda_context(&self) -> Option<&Arc<CudaContext>> {
        self.cuda_context.as_ref()
    }

    /// Global engine indices available for an operation graph.
    ///
    /// # Errors
    ///
    /// Returns an error if querying the graph's engine count fails or the count does not fit `i64`.
    pub fn engine_indices_for_operation_graph(graph: &OperationGraph) -> Result<Vec<EngineIndex>> {
        let count = to_i64(graph.engine_count()?, "engine count")?;
        Ok((0..count).map(EngineIndex::new).collect())
    }

    /// Finds the first engine whose default knob configuration can be finalized for this graph's live target.
    /// Returns the retained engine, with its original graph and device metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if engine enumeration fails. If no engine configuration can be finalized, returns the last attempted construction error, or [`Error::NoAvailableEngines`] when enumeration yields no engines.
    pub fn find_compatible(graph: &OperationGraph) -> Result<Self> {
        let mut last_error = None;
        for index in Self::engine_indices_for_operation_graph(graph)? {
            let result = (|| {
                let engine = Self::create_with_context(
                    None,
                    graph,
                    index,
                    None,
                    EnginePlanningConfig::default(),
                )?;
                let mut config = BackendDescriptor::create(BackendDescriptorType::EngineCfg)?;
                config.set_attribute_descriptor(
                    BackendAttributeName::EngineCfgEngine,
                    engine.descriptor(),
                )?;
                config.finalize()?;
                Ok::<_, Error>(engine)
            })();
            match result {
                Ok(engine) => return Ok(engine),
                Err(error) => last_error = Some(error),
            }
        }
        Err(last_error.unwrap_or(Error::NoAvailableEngines))
    }

    /// Behavior notes from this engine's actual finalized descriptor.
    /// Successful reads are retained; cached reads require no allocation or CUDA context binding.
    /// Failed native queries remain retryable.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding, the native attribute query, or note decoding fails.
    pub fn behavior_notes(&self) -> Result<&[BackendBehaviorNote]> {
        if let Some(notes) = self.behavior_notes.get() {
            return Ok(notes);
        }
        bind_context(self.cuda_context())?;
        let notes = self
            .descriptor
            .attribute_enum_slice(BackendAttributeName::EngineBehaviorNote)?;
        Ok(self.behavior_notes.get_or_init(|| notes))
    }

    /// Numerical notes from this engine's actual finalized descriptor.
    /// Successful reads are retained; cached reads require no allocation or CUDA context binding.
    /// Failed native queries remain retryable.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding, the native attribute query, or note decoding fails.
    pub fn numerical_notes(&self) -> Result<&[BackendNumericalNote]> {
        if let Some(notes) = self.numerical_notes.get() {
            return Ok(notes);
        }
        bind_context(self.cuda_context())?;
        let notes = self
            .descriptor
            .attribute_enum_slice(BackendAttributeName::EngineNumericalNote)?;
        Ok(self.numerical_notes.get_or_init(|| notes))
    }

    /// Adjustable knobs from this finalized engine and its retained compilation target.
    /// Successful reads are retained, including empty lists; failed native queries remain retryable.
    /// Cached reads borrow the same metadata without allocating or binding the CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding, native knob-descriptor or attribute queries, or knob-type decoding fails.
    pub fn knob_infos(&self) -> Result<&[EngineKnobInfo]> {
        if let Some(knobs) = self.knob_infos.get() {
            return Ok(knobs);
        }
        bind_context(self.cuda_context())?;
        // SAFETY: this attribute returns objects of the explicitly selected native class.
        let knobs = unsafe {
            self.descriptor.attribute_descriptor_slice(
                BackendAttributeName::EngineKnobInfo,
                BackendDescriptorType::KnobInfo,
                None,
            )
        }?;
        let knobs = knobs
            .into_iter()
            .map(|knob| {
                Ok(EngineKnobInfo {
                    knob_type: knob.attribute_enum(BackendAttributeName::KnobInfoType)?,
                    minimum_value: knob
                        .attribute_i64(BackendAttributeName::KnobInfoMinimumValue)?,
                    maximum_value: knob
                        .attribute_i64(BackendAttributeName::KnobInfoMaximumValue)?,
                    stride: knob.attribute_i64(BackendAttributeName::KnobInfoStride)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(self.knob_infos.get_or_init(|| knobs))
    }

    /// Borrows preferred tensor layouts from this finalized engine and compilation target.
    /// Successful reads retain the complete tensor-UID and layout snapshot, including empty lists.
    /// Cached reads require no allocation or CUDA context binding; failed queries remain retryable.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding, native layout-descriptor or attribute queries, or layout-type decoding fails.
    pub fn layout_infos(&self) -> Result<&[LayoutInfo]> {
        if let Some(layouts) = self.layout_infos.get() {
            return Ok(layouts);
        }
        bind_context(self.cuda_context())?;
        // SAFETY: the named attribute requires the explicitly selected native class.
        let layouts = unsafe {
            self.descriptor.attribute_descriptor_slice(
                BackendAttributeName::EngineLayoutInfo,
                BackendDescriptorType::LayoutInfo,
                None,
            )
        }?;

        let layouts = layouts
            .into_iter()
            .map(|layout| {
                Ok(LayoutInfo {
                    tensor_uid: layout
                        .attribute_i64(BackendAttributeName::LayoutInfoTensorUid)?
                        .into(),
                    types: layout.attribute_enum_slice(BackendAttributeName::LayoutInfoTypes)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(self.layout_infos.get_or_init(|| layouts))
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        bind_for_drop(self.cuda_context(), "cudnn engine");
    }
}
