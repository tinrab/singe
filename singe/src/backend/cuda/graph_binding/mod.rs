mod alias;
mod dependency;
mod in_place;
#[cfg(test)]
mod in_place_tests;
mod transfer;
mod validation;

pub use self::{
    alias::validate_graph_alias_tensor_metadata,
    dependency::{
        graph_native_dependencies, graph_node_positions, graph_users, operation_dependencies,
        push_unique_cuda_graph_dependency,
    },
    in_place::{InPlaceWrites, ScatterTarget, WritableInputs},
    transfer::copy_cuda_tensor_data_on,
    validation::{
        graph_matmul_is_zero_work, validate_graph_input_value_tensor, validate_graph_matmul,
        validate_graph_matmul_add, validate_graph_node_value_device,
        validate_graph_node_value_operand, validate_graph_node_value_tensor,
    },
};
use crate::{
    device::Device,
    error::{Error, Result},
    graph::ir::{Graph, ValueId},
};

pub fn infer_cuda_graph_device(ir: &Graph, order: &[ValueId]) -> Result<Device> {
    let mut device = None;
    for &id in order {
        let value = id;
        let value_device = ir.required_value(value)?.device();
        if !value_device.is_cuda() {
            return Err(Error::device_mismatch(
                format!("node {id:?} output device"),
                Device::Cuda(0),
                value_device,
            ));
        }
        match device {
            Some(existing) if existing != value_device => {
                return Err(Error::device_mismatch(
                    format!("node {id:?} output device"),
                    existing,
                    value_device,
                ));
            }
            Some(_) => {}
            None => device = Some(value_device),
        }
    }
    device.ok_or_else(|| Error::missing_output("infer cuda graph device"))
}
