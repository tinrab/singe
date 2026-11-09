use std::collections::HashMap;

use super::in_place::{InPlaceWrites, ScatterTarget, WritableInputs};
use crate::{
    backend::{cuda::graph_binding::graph_users, registry::Operation},
    data_type::DataType,
    device::Device,
    error::Result,
    graph::{Graph, GraphCompileOptions, ir::ValueId},
    index::TensorIndex,
    shape::Shape,
    tensor::Tensor,
};

fn meta(dimensions: &[i64], data_type: DataType) -> Tensor {
    Tensor::zeroed_on(
        Device::Meta,
        data_type,
        Shape::create(dimensions.to_vec()).unwrap(),
    )
    .unwrap()
}

fn index(tensor: &Tensor) -> [TensorIndex; 1] {
    [TensorIndex::Indices(tensor.clone())]
}

// What the graph's one scatter writes under `inputs`, as a word: `base`, `input`, or `fresh`.
fn target_in(
    function: impl Fn(&[Tensor]) -> Result<Vec<Tensor>>,
    inputs: WritableInputs<'_>,
) -> &'static str {
    let (graph, _) = Graph::capture(
        [
            ("cache", meta(&[2, 2], DataType::F32)),
            ("value", meta(&[1, 2], DataType::F32)),
            ("where", meta(&[1], DataType::I64)),
        ],
        function,
    )
    .unwrap();
    let compiled = graph
        .compile_with_options(GraphCompileOptions::new())
        .unwrap();
    let ir = compiled.shared_ir();
    let order = compiled.value_order().to_vec();
    let users = graph_users(&ir, &order).unwrap();
    let outputs = ir.output_value_set();
    let analysis = InPlaceWrites::analyze(&ir, &order, &users, &outputs, inputs).unwrap();
    let scatter = order
        .iter()
        .copied()
        .find(|&id| {
            ir.operation_for_value(id)
                .is_some_and(|operation| operation.opcode() == Some(Operation::IndexPut))
        })
        .unwrap();
    match analysis.target(scatter) {
        Some(ScatterTarget::Base(_)) => "base",
        Some(ScatterTarget::Input(_)) => "input",
        None => "fresh",
    }
}

fn write_cache(inputs: &[Tensor]) -> Result<Vec<Tensor>> {
    inputs[0].index_put_(&index(&inputs[2]), &inputs[1])?;
    Ok(vec![inputs[1].relu()?])
}

#[test]
fn test_a_written_input_is_written_only_where_the_inputs_allow_it() {
    let empty = HashMap::<ValueId, Tensor>::new();
    assert_eq!(target_in(write_cache, WritableInputs::Every), "input");
    assert_eq!(target_in(write_cache, WritableInputs::Nothing), "fresh");
    assert_eq!(
        target_in(write_cache, WritableInputs::Bound(&empty)),
        "fresh"
    );
}

#[test]
fn test_what_else_reads_an_input_keeps_it_from_being_written() {
    let read_too = |inputs: &[Tensor]| {
        let before = inputs[0].relu()?;
        inputs[0].index_put_(&index(&inputs[2]), &inputs[1])?;
        Ok(vec![before])
    };
    assert_eq!(target_in(read_too, WritableInputs::Every), "fresh");
    // A result the caller keeps cannot live where the next run writes.
    let returned = |inputs: &[Tensor]| {
        inputs[0].index_put_(&index(&inputs[2]), &inputs[1])?;
        Ok(vec![inputs[0].clone()])
    };
    assert_eq!(target_in(returned, WritableInputs::Every), "fresh");
}

#[test]
fn test_a_result_only_the_scatter_reads_is_written_where_it_is() {
    let result = |inputs: &[Tensor]| {
        let base = inputs[0].relu()?;
        base.index_put_(&index(&inputs[2]), &inputs[1])?;
        Ok(vec![base])
    };
    assert_eq!(target_in(result, WritableInputs::Nothing), "base");
    // A scatter that keeps its base takes a copy of it first, which only it reads.
    let copy = |inputs: &[Tensor]| {
        let base = inputs[0].relu()?;
        Ok(vec![
            base.index_put(&index(&inputs[2]), &inputs[1], false)?,
            base,
        ])
    };
    assert_eq!(target_in(copy, WritableInputs::Nothing), "base");
}

#[test]
fn test_a_result_something_else_reads_keeps_its_contents() {
    let read_too = |inputs: &[Tensor]| {
        let base = inputs[0].relu()?;
        let before = base.exp()?;
        base.index_put_(&index(&inputs[2]), &inputs[1])?;
        Ok(vec![base, before])
    };
    assert_eq!(target_in(read_too, WritableInputs::Every), "fresh");
}

#[test]
fn test_a_view_of_a_result_is_not_written_where_it_is() {
    let view = |inputs: &[Tensor]| {
        let base = inputs[0].relu()?.transpose(0, 1)?;
        base.index_put_(&index(&inputs[2]), &inputs[1])?;
        Ok(vec![base])
    };
    assert_eq!(target_in(view, WritableInputs::Every), "fresh");
}
