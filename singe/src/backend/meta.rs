//! Shape-only execution: every operation produces meta tensors whose metadata comes from the operation's inference rule.

use std::sync::Arc;

use crate::{
    backend::registry::{
        AliasBehavior, CheckedOperationRequest, OperationAttributes, operation_schema,
    },
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    graph::view_output_layout,
    shape::Shape,
    storage::meta::MetaStorage,
    tensor::Tensor,
};

// Meta tensors describe allocations without making them.
pub fn zeros(device: Device, data_type: DataType, shape: &Shape) -> Result<Tensor> {
    validate_meta_device(device)?;
    meta_tensor(data_type, shape)
}

pub fn from_bytes(
    device: Device,
    data_type: DataType,
    shape: &Shape,
    data: &[u8],
) -> Result<Tensor> {
    validate_meta_device(device)?;
    let expected = shape.byte_size(data_type.size_bytes())?;
    if data.len() != expected {
        return Err(Error::length_mismatch(
            "meta tensor data",
            expected,
            data.len(),
        ));
    }
    meta_tensor(data_type, shape)
}

/// Runs `request` on meta `inputs` without a kernel.
///
/// A view shares its input's storage with the layout its operation gives, as a backend view would; other results get fresh contiguous meta storage.
/// A copy off the meta device succeeds only for a literal that retained its host value.
///
/// # Errors
///
/// Returns an error if an input is not a meta tensor, the request copies a tensor without a retained value off the meta device, or a result's byte size overflows.
pub fn run(request: &CheckedOperationRequest, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
    if let Some(input) = inputs.iter().find(|input| !input.device().is_meta()) {
        return Err(Error::device_mismatch(
            "operation operands",
            Device::Meta,
            input.device(),
        ));
    }

    if let OperationAttributes::CopyTo(device) = request.attributes()
        && !device.is_meta()
    {
        // Only a literal that retained its host value can leave the meta device.
        let input = inputs
            .first()
            .ok_or_else(|| Error::missing_input("meta run"))?;
        let storage = input.storage().clone_to_device(*device)?;
        return Ok(vec![Tensor::create_from_storage(
            Arc::new(storage),
            input.data_type(),
            input.layout().clone(),
        )?]);
    }

    let operation = request.operation();
    let view = operation_schema(operation).alias_behavior() == AliasBehavior::View;
    request
        .inference()
        .outputs()
        .iter()
        .enumerate()
        .map(|(index, spec)| {
            if view
                && index == 0
                && let Some(input) = inputs.first()
                && let Ok(layout) =
                    view_output_layout(operation, request.attributes(), input.layout())
            {
                return Tensor::create_from_storage(
                    Arc::clone(input.storage()),
                    spec.data_type(),
                    layout,
                );
            }
            meta_tensor(spec.data_type(), spec.shape())
        })
        .collect()
}

fn meta_tensor(data_type: DataType, shape: &Shape) -> Result<Tensor> {
    let bytes = shape.byte_size(data_type.size_bytes())?;
    Tensor::create(MetaStorage::new(bytes), data_type, shape.clone())
}

fn validate_meta_device(device: Device) -> Result<()> {
    if device.is_meta() {
        Ok(())
    } else {
        Err(Error::device_mismatch("device", Device::Meta, device))
    }
}
