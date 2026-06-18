use std::fmt::{self, Display, Formatter};

use singe_cuda::device::Device;
use singe_cudnn_sys as sys;

use crate::{
    attribute::{BackendAttributeName, BackendAttributeType},
    behavior::{BackendBehaviorNote, BackendNumericalNote},
    context::Context,
    descriptor::{BackendDescriptor, BackendDescriptorType},
    error::{Error, Result},
    execution::{EngineKnobInfo, LayoutInfo, OperationGraph},
    try_ffi,
    utility::{to_i64, to_usize},
};

pub struct Engine {
    index: EngineIndex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EngineIndex(i64);

impl EngineIndex {
    pub fn new(value: i64) -> Self {
        EngineIndex(value)
    }

    pub fn as_i64(self) -> i64 {
        self.0
    }
}

impl Display for EngineIndex {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<i64> for EngineIndex {
    fn from(value: i64) -> Self {
        EngineIndex(value)
    }
}

impl Engine {
    pub(crate) fn finalized_device_prop(ctx: &Context) -> Result<BackendDescriptor> {
        ctx.bind()?;

        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::DeviceProp)?;
        descriptor.set_attribute_i32(
            BackendAttributeName::DevicePropDeviceId,
            Device::current()?.id(),
        )?;
        descriptor.finalize()?;
        Ok(descriptor)
    }

    fn finalized_descriptor_without_graph(&self) -> Result<BackendDescriptor> {
        let mut engine = BackendDescriptor::create(BackendDescriptorType::Engine)?;
        engine.set_attribute_i64(BackendAttributeName::EngineGlobalIndex, self.index.as_i64())?;
        engine.finalize()?;
        Ok(engine)
    }

    fn finalized_descriptor_for_graph(
        &self,
        op_graph: &BackendDescriptor,
    ) -> Result<BackendDescriptor> {
        let mut engine = BackendDescriptor::create(BackendDescriptorType::Engine)?;
        engine.set_attribute_descriptor(BackendAttributeName::EngineOperationGraph, op_graph)?;
        engine.set_attribute_i64(BackendAttributeName::EngineGlobalIndex, self.index.as_i64())?;
        engine.finalize()?;
        Ok(engine)
    }

    fn finalized_descriptor(
        &self,
        ctx: &Context,
        op_graph: &BackendDescriptor,
    ) -> Result<BackendDescriptor> {
        let mut engine = BackendDescriptor::create(BackendDescriptorType::Engine)?;
        engine.set_attribute_descriptor(BackendAttributeName::EngineOperationGraph, op_graph)?;
        if let Ok(device_prop) = Self::finalized_device_prop(ctx) {
            let _ = engine
                .set_attribute_descriptor(BackendAttributeName::EngineDeviceProp, &device_prop);
        }
        engine.set_attribute_i64(BackendAttributeName::EngineGlobalIndex, self.index.as_i64())?;
        engine.finalize()?;
        Ok(engine)
    }

    pub fn index(&self) -> EngineIndex {
        self.index
    }

    pub fn from_index(index: EngineIndex) -> Self {
        Self { index }
    }

    /// Returns the global indices of all engines associated with a given operation graph.
    pub fn engine_indices_for_operation_graph(op_graph: &BackendDescriptor) -> Result<Vec<i64>> {
        let engine_count =
            op_graph.attribute_i64(BackendAttributeName::OperationGraphEngineGlobalCount)?;
        let engine_indices: Vec<i64> = (0..engine_count).collect();
        Ok(engine_indices)
    }

    /// Finds the index of the first engine that is truly compatible with the operation graph
    /// and the current hardware by attempting to finalize its EngineCfg descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDNN cannot enumerate engines, create or finalize
    /// the temporary descriptors, or if no compatible engine can be found.
    pub fn find_compatible(op_graph: &BackendDescriptor) -> Result<Self> {
        let engine_indices = Self::engine_indices_for_operation_graph(op_graph)?;

        for engine_index in engine_indices {
            let result: Result<i64> = (|| {
                let mut engine = BackendDescriptor::create(BackendDescriptorType::Engine)?;

                engine.set_attribute_descriptor(
                    BackendAttributeName::EngineOperationGraph,
                    op_graph,
                )?;
                engine.set_attribute_i64(BackendAttributeName::EngineGlobalIndex, engine_index)?;
                engine.finalize()?;

                let mut engine_config =
                    BackendDescriptor::create(BackendDescriptorType::EngineCfg)?;
                engine_config
                    .set_attribute_descriptor(BackendAttributeName::EngineCfgEngine, &engine)?;

                match engine_config.finalize() {
                    Ok(()) => Ok(engine_index),
                    Err(_) => Err(Error::NoAvailableEngines),
                }
            })();

            if let Ok(compatible_index) = result {
                return Ok(Self::from_index(compatible_index.into()));
            }
        }

        Err(Error::NoAvailableEngines)
    }

    pub fn find_compatible_for_graph(graph: &OperationGraph) -> Result<Self> {
        Self::find_compatible(graph.descriptor())
    }

    pub fn behavior_notes(&self, op_graph: &BackendDescriptor) -> Result<Vec<BackendBehaviorNote>> {
        self.finalized_descriptor_for_graph(op_graph)?
            .attribute_enum_slice(
                BackendAttributeName::EngineBehaviorNote,
                BackendAttributeType::BehaviorNote,
            )
    }

    pub fn numerical_notes(
        &self,
        op_graph: &BackendDescriptor,
    ) -> Result<Vec<BackendNumericalNote>> {
        self.finalized_descriptor_for_graph(op_graph)?
            .attribute_enum_slice(
                BackendAttributeName::EngineNumericalNote,
                BackendAttributeType::NumericalNote,
            )
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
    pub fn knob_infos(
        &self,
        ctx: &Context,
        op_graph: &BackendDescriptor,
    ) -> Result<Vec<EngineKnobInfo>> {
        let engine = self.finalized_descriptor(ctx, op_graph)?;
        let capacity = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_COUNTS as usize;

        let mut knobs = (0..capacity)
            .map(|_| BackendDescriptor::create(BackendDescriptorType::KnobInfo))
            .collect::<Result<Vec<_>>>()?;
        let mut raw_knobs = knobs
            .iter_mut()
            .map(|descriptor| descriptor.as_raw())
            .collect::<Vec<_>>();

        let mut count_written = 0_i64;
        unsafe {
            try_ffi!(sys::cudnnBackendGetAttribute(
                engine.as_raw(),
                BackendAttributeName::EngineKnobInfo.into(),
                BackendAttributeType::BackendDescriptor.into(),
                to_i64(raw_knobs.len(), "knob_count_capacity")?,
                &raw mut count_written,
                raw_knobs.as_mut_ptr().cast(),
            ))?;
        }

        knobs.truncate(to_usize(count_written, "knob_count")?);

        knobs.iter_mut().for_each(BackendDescriptor::mark_finalized);

        knobs
            .into_iter()
            .map(|knob| {
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

    pub fn layout_infos(&self) -> Result<Vec<LayoutInfo>> {
        let engine = self.finalized_descriptor_without_graph()?;
        let layouts = engine.attribute_descriptor_slice(
            BackendAttributeName::EngineLayoutInfo,
            BackendAttributeType::BackendDescriptor,
        )?;

        layouts
            .into_iter()
            .map(|layout| {
                let layout = BackendDescriptor::from_raw_finalized_borrowed(
                    layout,
                    BackendDescriptorType::LayoutInfo,
                )?;
                Ok(LayoutInfo {
                    tensor_id: layout
                        .attribute_i64(BackendAttributeName::LayoutInfoTensorId)?
                        .into(),
                    types: layout.attribute_enum_slice(
                        BackendAttributeName::LayoutInfoTypes,
                        BackendAttributeType::LayoutType,
                    )?,
                })
            })
            .collect()
    }
}
