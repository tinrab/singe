use std::{mem::ManuallyDrop, ptr, rc::Rc, sync::Arc};

use singe_cuda::context::Context as CudaContext;
use singe_cudnn_sys as sys;

use crate::{
    attribute::{BackendAttributeName, BackendAttributeType},
    context::Context,
    descriptor::{BackendDescriptor, BackendDescriptorType},
    device::DeviceProperties,
    engine::{Engine, EngineIndex},
    error::{Error, Result, Status},
    execution::{OperationGraph, validate_context_pair},
    heuristic::BackendHeuristicMode,
    knob::BackendKnobType,
    layout::BackendLayoutType,
    tensor::TensorId,
    try_ffi,
    utility::{to_i64, to_usize},
};

#[derive(Debug)]
pub struct EngineHeuristics<'a> {
    descriptor: BackendDescriptor,
    _operation_graph: &'a OperationGraph,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineKnobInfo {
    pub knob_type: BackendKnobType,
    pub minimum_value: i64,
    pub maximum_value: i64,
    pub stride: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KnobChoiceInfo {
    pub knob_type: BackendKnobType,
    pub value: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntermediateInfo {
    pub unique_id: i64,
    pub size: usize,
    pub dependent_data_ids: Vec<i64>,
    pub dependent_attributes: Vec<BackendAttributeName>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutInfo {
    pub tensor_id: TensorId,
    pub types: Vec<BackendLayoutType>,
}

#[derive(Debug)]
pub struct EngineConfig {
    descriptor: BackendDescriptor,
    cuda_ctx: Option<Arc<CudaContext>>,
    _keepalive: Option<Rc<BackendDescriptor>>,
}

impl EngineConfig {
    pub(crate) fn from_raw_borrowed(
        descriptor: sys::cudnnBackendDescriptor_t,
        keepalive: Rc<BackendDescriptor>,
    ) -> Result<Self> {
        Ok(Self {
            descriptor: BackendDescriptor::from_raw_finalized_borrowed(
                descriptor,
                BackendDescriptorType::EngineCfg,
            )?,
            cuda_ctx: None,
            _keepalive: Some(keepalive),
        })
    }

    pub(crate) fn from_descriptor_keepalive(
        descriptor: BackendDescriptor,
        cuda_ctx: Option<Arc<CudaContext>>,
        keepalive: Rc<BackendDescriptor>,
    ) -> Self {
        Self {
            descriptor,
            cuda_ctx,
            _keepalive: Some(keepalive),
        }
    }

    pub fn create(
        ctx: &Context,
        operation_graph: &OperationGraph,
        engine_index: EngineIndex,
        knob_choices: &[(BackendKnobType, i64)],
    ) -> Result<Self> {
        Self::create_with_config(ctx, operation_graph, engine_index, knob_choices, None)
    }

    pub fn create_with_config(
        ctx: &Context,
        operation_graph: &OperationGraph,
        engine_index: EngineIndex,
        knob_choices: &[(BackendKnobType, i64)],
        device_properties: Option<&DeviceProperties>,
    ) -> Result<Self> {
        Self::create_from_engine_descriptor(
            Some(ctx),
            operation_graph,
            engine_index,
            knob_choices,
            device_properties,
        )
    }

    pub fn create_deviceless(
        operation_graph: &OperationGraph,
        engine_index: EngineIndex,
        knob_choices: &[(BackendKnobType, i64)],
        device_properties: &DeviceProperties,
    ) -> Result<Self> {
        Self::create_from_engine_descriptor(
            None,
            operation_graph,
            engine_index,
            knob_choices,
            Some(device_properties),
        )
    }

    fn create_from_engine_descriptor(
        ctx: Option<&Context>,
        operation_graph: &OperationGraph,
        engine_index: EngineIndex,
        knob_choices: &[(BackendKnobType, i64)],
        device_properties: Option<&DeviceProperties>,
    ) -> Result<Self> {
        validate_context_pair(
            ctx.map(|ctx| ctx.cuda_context()),
            operation_graph.cuda_context(),
            "operation graph",
        )?;

        let mut engine = BackendDescriptor::create(BackendDescriptorType::Engine)?;
        engine.set_attribute_descriptor(
            BackendAttributeName::EngineOperationGraph,
            operation_graph.descriptor(),
        )?;
        if let Some(device_properties) = device_properties {
            let _ = engine.set_attribute_descriptor(
                BackendAttributeName::EngineDeviceProp,
                device_properties.descriptor(),
            );
        } else if let Some(ctx) = ctx
            && let Ok(device_prop) = Engine::finalized_device_prop(ctx)
        {
            let _ = engine
                .set_attribute_descriptor(BackendAttributeName::EngineDeviceProp, &device_prop);
        }
        engine.set_attribute_i64(
            BackendAttributeName::EngineGlobalIndex,
            engine_index.as_i64(),
        )?;
        engine.finalize()?;

        let knob_choice_descriptors = knob_choices
            .iter()
            .map(|(knob_type, value)| {
                let mut descriptor = BackendDescriptor::create(BackendDescriptorType::KnobChoice)?;
                descriptor.set_attribute_enum(
                    BackendAttributeName::KnobChoiceKnobType,
                    BackendAttributeType::KnobType,
                    *knob_type,
                )?;
                descriptor.set_attribute_i64(BackendAttributeName::KnobChoiceKnobValue, *value)?;
                descriptor.finalize()?;
                Ok(descriptor)
            })
            .collect::<Result<Vec<_>>>()?;

        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::EngineCfg)?;
        descriptor.set_attribute_descriptor(BackendAttributeName::EngineCfgEngine, &engine)?;
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
            cuda_ctx: ctx
                .map(|ctx| Arc::clone(ctx.cuda_context()))
                .or_else(|| operation_graph.cuda_context().map(Arc::clone)),
            _keepalive: None,
        })
    }

    pub fn workspace_size(&self) -> Result<usize> {
        to_usize(
            self.descriptor
                .attribute_i64(BackendAttributeName::EngineCfgWorkspaceSize)?,
            "workspace_size",
        )
    }

    fn extracted_engine_descriptor(&self) -> Result<BackendDescriptor> {
        self.descriptor.attribute_descriptor_of_type(
            BackendAttributeName::EngineCfgEngine,
            BackendDescriptorType::Engine,
        )
    }

    pub fn shared_memory_size(&self) -> Result<usize> {
        match self
            .descriptor
            .attribute_i32(BackendAttributeName::EngineCfgSharedMemoryUsed)
        {
            Ok(shared_memory_size) => to_usize(shared_memory_size, "shared_memory_size"),
            Err(Error::DescriptorAttributeNotFound(_))
            | Err(Error::Cudnn {
                code: Status::NotSupported,
                ..
            }) => Ok(0),
            Err(error) => Err(error),
        }
    }

    pub fn knob_infos(&self) -> Result<Vec<EngineKnobInfo>> {
        let engine = self.extracted_engine_descriptor()?;
        let knobs = engine.attribute_descriptor_slice(
            BackendAttributeName::EngineKnobInfo,
            BackendAttributeType::BackendDescriptor,
        )?;

        knobs
            .into_iter()
            .map(|knob| {
                let knob = BackendDescriptor::from_raw_finalized_borrowed(
                    knob,
                    BackendDescriptorType::KnobInfo,
                )?;
                Ok(EngineKnobInfo {
                    knob_type: knob.attribute_enum(
                        BackendAttributeName::KnobInfoType,
                        BackendAttributeType::KnobType,
                    )?,
                    minimum_value: knob
                        .attribute_i64(BackendAttributeName::KnobInfoMinimumValue)?,
                    maximum_value: knob
                        .attribute_i64(BackendAttributeName::KnobInfoMaximumValue)?,
                    stride: knob.attribute_i64(BackendAttributeName::KnobInfoStride)?,
                })
            })
            .collect()
    }

    pub fn knob_choices(&self) -> Result<Vec<KnobChoiceInfo>> {
        let descriptors = self.descriptor.attribute_descriptor_slice(
            BackendAttributeName::EngineCfgKnobChoices,
            BackendAttributeType::BackendDescriptor,
        )?;

        descriptors
            .into_iter()
            .map(|descriptor| {
                let descriptor = BackendDescriptor::from_raw_finalized_borrowed(
                    descriptor,
                    BackendDescriptorType::KnobChoice,
                )?;

                Ok(KnobChoiceInfo {
                    knob_type: descriptor.attribute_enum(
                        BackendAttributeName::KnobChoiceKnobType,
                        BackendAttributeType::KnobType,
                    )?,
                    value: descriptor.attribute_i64(BackendAttributeName::KnobChoiceKnobValue)?,
                })
            })
            .collect()
    }

    pub fn intermediate_infos(&self) -> Result<Vec<IntermediateInfo>> {
        let descriptors = self.descriptor.attribute_descriptor_slice(
            BackendAttributeName::EngineCfgIntermediateInfo,
            BackendAttributeType::BackendDescriptor,
        )?;

        descriptors
            .into_iter()
            .map(|descriptor| {
                let descriptor = BackendDescriptor::from_raw_finalized_borrowed(
                    descriptor,
                    BackendDescriptorType::IntermediateInfo,
                )?;
                let size = to_usize(
                    descriptor.attribute_i64(BackendAttributeName::IntermediateInfoSize)?,
                    "size",
                )?;
                let dependent_attributes = descriptor.attribute_enum_slice(
                    BackendAttributeName::IntermediateInfoDependentAttributes,
                    BackendAttributeType::AttribName,
                )?;

                Ok(IntermediateInfo {
                    unique_id: descriptor
                        .attribute_i64(BackendAttributeName::IntermediateInfoUniqueId)?,
                    size,
                    dependent_data_ids: descriptor.attribute_i64_slice(
                        BackendAttributeName::IntermediateInfoDependentDataIds,
                    )?,
                    dependent_attributes,
                })
            })
            .collect()
    }

    pub fn engine_index(&self) -> Result<EngineIndex> {
        Ok(self
            .extracted_engine_descriptor()?
            .attribute_i64(BackendAttributeName::EngineGlobalIndex)?
            .into())
    }

    pub fn engine(&self) -> Result<Engine> {
        Ok(Engine::from_index(self.engine_index()?))
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }

    pub(crate) fn cuda_context(&self) -> Option<&Arc<CudaContext>> {
        self.cuda_ctx.as_ref()
    }
}

impl<'a> EngineHeuristics<'a> {
    pub fn create(
        ctx: &Context,
        graph: &'a OperationGraph,
        mode: BackendHeuristicMode,
        sm_count_target: Option<i64>,
    ) -> Result<Self> {
        Self::create_with_config(ctx, graph, mode, sm_count_target, None)
    }

    pub fn create_with_config(
        ctx: &Context,
        graph: &'a OperationGraph,
        mode: BackendHeuristicMode,
        sm_count_target: Option<i64>,
        device_properties: Option<&DeviceProperties>,
    ) -> Result<Self> {
        Self::create_with_context(Some(ctx), graph, mode, sm_count_target, device_properties)
    }

    pub fn create_deviceless(
        graph: &'a OperationGraph,
        mode: BackendHeuristicMode,
        sm_count_target: Option<i64>,
        device_properties: &DeviceProperties,
    ) -> Result<Self> {
        Self::create_with_context(None, graph, mode, sm_count_target, Some(device_properties))
    }

    fn create_with_context(
        ctx: Option<&Context>,
        graph: &'a OperationGraph,
        mode: BackendHeuristicMode,
        sm_count_target: Option<i64>,
        device_properties: Option<&DeviceProperties>,
    ) -> Result<Self> {
        validate_context_pair(
            ctx.map(|ctx| ctx.cuda_context()),
            graph.cuda_context(),
            "operation graph",
        )?;

        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::EngineHeur)?;
        descriptor.set_attribute_enum(
            BackendAttributeName::EngineHeurMode,
            BackendAttributeType::HeurMode,
            mode,
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::EngineHeurOperationGraph,
            graph.descriptor(),
        )?;
        if let Some(device_properties) = device_properties {
            let _ = descriptor.set_attribute_descriptor(
                BackendAttributeName::EngineHeurDeviceProp,
                device_properties.descriptor(),
            );
        } else if let Some(ctx) = ctx
            && let Ok(device_prop) = Engine::finalized_device_prop(ctx)
        {
            let _ = descriptor
                .set_attribute_descriptor(BackendAttributeName::EngineHeurDeviceProp, &device_prop);
        }
        if let Some(sm_count_target) = sm_count_target {
            descriptor.set_attribute_i64(
                BackendAttributeName::EngineHeurSmCountTarget,
                sm_count_target,
            )?;
        }
        descriptor.finalize()?;

        Ok(Self {
            descriptor,
            _operation_graph: graph,
        })
    }

    /// Returns an attribute from this finalized backend descriptor.
    ///
    /// The `name` selects the attribute, and this method supplies the expected cuDNN
    /// attribute type from its Rust return type or `attribute_type` argument.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor has not been finalized, if cuDNN
    /// does not return the requested attribute value, or if cuDNN rejects
    /// the attribute name or expected type for this descriptor.
    pub fn engine_configs(&self) -> Result<Vec<EngineConfig>> {
        let mut count = 0_i64;
        unsafe {
            try_ffi!(sys::cudnnBackendGetAttribute(
                self.descriptor.as_raw(),
                BackendAttributeName::EngineHeurResults.into(),
                BackendAttributeType::BackendDescriptor.into(),
                0,
                &raw mut count,
                ptr::null_mut(),
            ))?;
        }
        let count = to_usize(count, "heuristic_result_count")?;
        if count == 0 {
            return Ok(Vec::new());
        }

        let keepalive = Rc::new(BackendDescriptor::from_raw_finalized_borrowed(
            self.descriptor.as_raw(),
            BackendDescriptorType::EngineHeur,
        )?);
        let mut configs = Vec::with_capacity(count);
        let mut raw_descriptors = Vec::with_capacity(count);
        for _ in 0..count {
            let config = BackendDescriptor::create(BackendDescriptorType::EngineCfg)?;
            raw_descriptors.push(config.as_raw());
            configs.push(config);
        }

        let mut count_written = 0;
        unsafe {
            try_ffi!(sys::cudnnBackendGetAttribute(
                self.descriptor.as_raw(),
                BackendAttributeName::EngineHeurResults.into(),
                BackendAttributeType::BackendDescriptor.into(),
                to_i64(raw_descriptors.len(), "heuristic_result_count")?,
                &raw mut count_written,
                raw_descriptors.as_mut_ptr().cast(),
            ))?;
        }

        let count_written = to_usize(count_written, "heuristic_result_count")?;
        configs.truncate(count_written);

        let mut result = Vec::with_capacity(configs.len());
        for config in configs {
            let config = ManuallyDrop::new(config);
            let descriptor = config.as_raw();
            result.push(EngineConfig::from_raw_borrowed(
                descriptor,
                Rc::clone(&keepalive),
            )?);
        }

        Ok(result)
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}
