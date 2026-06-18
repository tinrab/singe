use crate::{
    attribute::{BackendAttributeName, BackendAttributeType},
    data_type::DataType,
    descriptor::{BackendDescriptor, BackendDescriptorType},
    error::Result,
    reduction::ReduceTensorOperator,
    tensor::Tensor,
};

#[derive(Debug)]
pub struct ReductionDescriptor {
    descriptor: BackendDescriptor,
}

impl ReductionDescriptor {
    pub fn create(
        op: ReduceTensorOperator,
        compute_type: DataType,
        is_deterministic: bool,
    ) -> Result<Self> {
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::Reduction)?;
        descriptor.set_attribute_enum(
            BackendAttributeName::ReductionOperator,
            BackendAttributeType::ReductionOperatorType,
            op,
        )?;
        descriptor.set_attribute_enum(
            BackendAttributeName::ReductionCompType,
            BackendAttributeType::DataType,
            compute_type,
        )?;
        descriptor.set_attribute_bool(
            BackendAttributeName::ReductionIsDeterministic,
            is_deterministic,
        )?;
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

#[derive(Debug)]
pub struct ReductionOperation {
    descriptor: BackendDescriptor,
}

impl ReductionOperation {
    pub fn create(reduction: &ReductionDescriptor, x: &Tensor, y: &Tensor) -> Result<Self> {
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::OperationReduction)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationReductionDesc,
            reduction.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationReductionXDesc,
            x.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationReductionYDesc,
            y.descriptor(),
        )?;
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}
