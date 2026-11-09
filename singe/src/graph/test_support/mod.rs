mod assertions;
mod graph;
mod operations;
mod tensors;

pub use std::collections::HashMap;

pub(crate) use singe_core::assert_close;

pub use self::{assertions::*, graph::*, operations::*, tensors::*};
pub use crate::{
    TensorDataType,
    backend::registry::{
        ContractionAttributes, Operation, OperationAttributes, ReductionAttributes,
        SliceAttributes, TopKAttributes,
    },
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    graph::{
        Graph, GraphCompileOptions, GraphFingerprint, GraphFingerprintOptions, GraphInputClass,
        GraphOperationInputs, GraphOperationOutput, GraphOperationRequest,
        guards::{GraphGuardFailure, GraphGuards},
        ir::*,
        passes::{GraphPassKind, GraphPassPipeline},
    },
    shape::Shape,
    storage::cpu::CpuStorage,
    tensor::Tensor,
};
use crate::{
    graph::node::{Node, NodeKind},
    layout::Layout,
};

impl Graph {
    /// Constructs compiler fixtures, including deliberately malformed requests.
    /// Public insertion checks have separate public API coverage.
    pub(crate) fn add_operation_fixture(&mut self, request: GraphOperationRequest) -> ValueId {
        self.push_node(request.into_node())
    }

    pub(crate) fn add_copy_to_with_declared_output_for_test(
        &mut self,
        input: ValueId,
        device: Device,
        output: &Tensor,
    ) -> Result<ValueId> {
        self.add_named_node_with_layout(Node::copy_to_with_declared_output_for_test(
            input, device, output,
        ))
    }

    pub(crate) fn add_cast_with_declared_output_for_test(
        &mut self,
        input: ValueId,
        data_type: DataType,
        output: &Tensor,
    ) -> Result<ValueId> {
        self.add_named_node_with_layout(Node::cast_with_declared_output_for_test(
            input, data_type, output,
        ))
    }

    pub(crate) fn add_reshape_with_declared_output_for_test(
        &mut self,
        input: ValueId,
        shape: Shape,
        output: &Tensor,
    ) -> Result<ValueId> {
        self.add_named_node_with_layout(Node::reshape_with_declared_output_for_test(
            input, shape, output,
        ))
    }

    pub(crate) fn replace_node_copy_to_for_test(
        &mut self,
        id: ValueId,
        input: ValueId,
        device: Device,
    ) {
        self.update_node(id, |node| {
            node.replace_kind(NodeKind::copy_to(input, device));
        })
        .unwrap();
    }

    pub(crate) fn replace_node_slice_for_test(
        &mut self,
        id: ValueId,
        input: ValueId,
        dim: i64,
        start: i64,
        length: i64,
    ) {
        self.update_node(id, |node| {
            node.replace_kind(NodeKind::slice(input, dim, start, length));
        })
        .unwrap();
    }

    pub(crate) fn replace_node_concat_for_test(
        &mut self,
        id: ValueId,
        inputs: Vec<ValueId>,
        dim: i64,
    ) {
        self.update_node(id, |node| node.replace_kind(NodeKind::concat(inputs, dim)))
            .unwrap();
    }

    pub(crate) fn replace_node_output_metadata_for_test(
        &mut self,
        id: ValueId,
        data_type: DataType,
        layout: Layout,
        device: Device,
    ) {
        self.update_node(id, |node| {
            node.replace_output_metadata(data_type, layout, device);
        })
        .unwrap();
    }

    pub(crate) fn replace_node_output_data_type_for_test(
        &mut self,
        id: ValueId,
        data_type: DataType,
    ) {
        self.update_node(id, |node| node.replace_output_data_type(data_type))
            .unwrap();
    }

    pub(crate) fn replace_node_output_layout_for_test(&mut self, id: ValueId, layout: Layout) {
        self.update_node(id, |node| node.replace_output_layout(layout))
            .unwrap();
    }

    pub(crate) fn replace_node_output_device_for_test(&mut self, id: ValueId, device: Device) {
        self.update_node(id, |node| node.replace_output_device(device))
            .unwrap();
    }
}

// A tensor of varied values in [-1, 1) that `seed` tells apart from others of its shape.
pub fn patterned_tensor(device: Device, dimensions: &[i64], seed: usize) -> Tensor {
    let shape = Shape::create(dimensions).unwrap();
    let data = (0..shape.element_count())
        .map(|index| ((index * 37 + seed * 11) % 97) as f32 / 48.0 - 1.0)
        .collect::<Vec<_>>();
    Tensor::from_vec(device, &shape, data).unwrap()
}
