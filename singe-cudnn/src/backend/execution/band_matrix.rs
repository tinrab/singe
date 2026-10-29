use crate::{
    backend::{
        attribute::BackendAttributeName,
        descriptor::{BackendDescriptor, BackendDescriptorType},
        tensor::Tensor,
    },
    error::{Error, Result},
    utility::check_range,
};

/// Finalized native band-matrix expansion with declared bandwidths, padding, and optional masking operands.
#[derive(Debug)]
pub struct ExpandBandMatrixOperation {
    descriptor: BackendDescriptor,
}

impl ExpandBandMatrixOperation {
    /// Creates band-matrix expansion from input/output descriptors and optional token-offset and speculative-mask operands.
    ///
    /// # Errors
    ///
    /// Returns an error for negative bandwidths, an axis outside the input rank, or native descriptor creation, attribute assignment, or finalization failure.
    pub fn create(
        x: &Tensor,
        y: &Tensor,
        lower_bandwidth: i64,
        upper_bandwidth: i64,
        axis: i64,
        pad_value: f32,
        kv_token_offset: Option<&Tensor>,
        speculative_mask: Option<&Tensor>,
    ) -> Result<Self> {
        check_range!("lower_bandwidth", lower_bandwidth >= 0)?;
        check_range!("upper_bandwidth", upper_bandwidth >= 0)?;

        check_range!(
            "axis",
            usize::try_from(axis).is_ok_and(|axis| axis < x.rank())
        )?;

        let mut descriptor =
            BackendDescriptor::create(BackendDescriptorType::OperationExpandBandMatrix)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationExpandBandMatrixXDesc,
            x.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationExpandBandMatrixYDesc,
            y.descriptor(),
        )?;
        descriptor.set_attribute_i64(
            BackendAttributeName::OperationExpandBandMatrixLowerBandwidth,
            lower_bandwidth,
        )?;
        descriptor.set_attribute_i64(
            BackendAttributeName::OperationExpandBandMatrixUpperBandwidth,
            upper_bandwidth,
        )?;
        descriptor.set_attribute_i64(BackendAttributeName::OperationExpandBandMatrixAxis, axis)?;
        descriptor.set_attribute_f32(
            BackendAttributeName::OperationExpandBandMatrixPadValue,
            pad_value,
        )?;
        if let Some(kv_token_offset) = kv_token_offset {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationExpandBandMatrixKvTokenOffsetDesc,
                kv_token_offset.descriptor(),
            )?;
        }
        if let Some(speculative_mask) = speculative_mask {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationExpandBandMatrixSpeculativeMaskDesc,
                speculative_mask.descriptor(),
            )?;
        }
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    /// Borrows the finalized native operation descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

/// Finalized native band-matrix contraction with declared bandwidths, padding, and token bound.
#[derive(Debug)]
pub struct ContractBandMatrixOperation {
    descriptor: BackendDescriptor,
}

impl ContractBandMatrixOperation {
    /// Creates band-matrix contraction from input/output descriptors and explicit band geometry.
    ///
    /// # Errors
    ///
    /// Returns an error for negative bandwidths or maximum token value, an axis outside the input rank, or native descriptor creation, attribute assignment, or finalization failure.
    pub fn create(
        x: &Tensor,
        y: &Tensor,
        lower_bandwidth: i64,
        upper_bandwidth: i64,
        axis: i64,
        pad_value: f32,
        max_token_value: i64,
    ) -> Result<Self> {
        if lower_bandwidth < 0 {
            return Err(Error::out_of_range("lower_bandwidth"));
        }
        if upper_bandwidth < 0 {
            return Err(Error::out_of_range("upper_bandwidth"));
        }
        if max_token_value < 0 {
            return Err(Error::out_of_range("max_token_value"));
        }
        if !usize::try_from(axis).is_ok_and(|axis| axis < x.rank()) {
            return Err(Error::out_of_range("axis"));
        }

        let mut descriptor =
            BackendDescriptor::create(BackendDescriptorType::OperationContractBandMatrix)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationContractBandMatrixXDesc,
            x.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationContractBandMatrixYDesc,
            y.descriptor(),
        )?;
        descriptor.set_attribute_i64(
            BackendAttributeName::OperationContractBandMatrixLowerBandwidth,
            lower_bandwidth,
        )?;
        descriptor.set_attribute_i64(
            BackendAttributeName::OperationContractBandMatrixUpperBandwidth,
            upper_bandwidth,
        )?;
        descriptor
            .set_attribute_i64(BackendAttributeName::OperationContractBandMatrixAxis, axis)?;
        descriptor.set_attribute_f32(
            BackendAttributeName::OperationContractBandMatrixPadValue,
            pad_value,
        )?;
        descriptor.set_attribute_i64(
            BackendAttributeName::OperationContractBandMaxTokenValue,
            max_token_value,
        )?;
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    /// Borrows the finalized native operation descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}
