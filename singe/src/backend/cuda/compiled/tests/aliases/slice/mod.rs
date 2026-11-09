mod cases;
mod validation;

use super::*;
use crate::{
    backend::cuda::graph_value::CudaGraphValue,
    graph::{GraphCudaOptions, ir::ValueId},
};

#[derive(Clone, Copy)]
struct SliceAliasCase<'a> {
    dim: i64,
    start: i64,
    length: i64,
    expected_strides: &'a [i64],
    expected_storage_offset: usize,
}

fn cuda_device() -> Device {
    Device::Cuda(0)
}

fn contiguous_shape(dimensions: &[i64]) -> Shape {
    Shape::create(dimensions.iter().copied()).unwrap()
}

fn maybe_i64_tensor(shape: &Shape, values: Vec<i64>) -> Option<Tensor> {
    skip_if_stub(Tensor::from_vec(cuda_device(), shape, values))
}

fn maybe_i64_output(shape: &Shape) -> Option<Tensor> {
    skip_if_stub(Tensor::zeroed_on(cuda_device(), DataType::I64, shape))
}

fn add_i64_input(graph: &mut Graph, shape: Shape) -> ValueId {
    graph.add_input("x", DataType::I64, shape, cuda_device())
}

fn add_i64_slice_operation(
    graph: &mut Graph,
    input: ValueId,
    output: &Tensor,
    case: SliceAliasCase<'_>,
) -> Result<ValueId, Error> {
    graph.add_named_slice_operation(
        None,
        input,
        case.dim,
        case.start,
        case.length,
        output.data_type(),
        &output.shape().clone(),
        output.device(),
    )
}

fn assert_i64_slice_alias(
    input_dimensions: &[i64],
    output_dimensions: &[i64],
    input_values: Vec<i64>,
    case: SliceAliasCase<'_>,
) {
    let Some(outputs) =
        run_i64_slice_alias(input_dimensions, output_dimensions, input_values, case)
    else {
        return;
    };

    assert_eq!(outputs[0].data_type(), DataType::I64);
    assert_eq!(outputs[0].shape().dimensions(), output_dimensions);
    assert_eq!(outputs[0].layout().strides(), case.expected_strides);
    assert_eq!(outputs[0].storage_offset(), case.expected_storage_offset);
}

fn run_i64_slice_alias(
    input_dimensions: &[i64],
    output_dimensions: &[i64],
    input_values: Vec<i64>,
    case: SliceAliasCase<'_>,
) -> Option<Vec<CudaGraphValue>> {
    let shape = contiguous_shape(input_dimensions);
    let output_shape = contiguous_shape(output_dimensions);
    let input = maybe_i64_tensor(&shape, input_values)?;
    let output = maybe_i64_output(&output_shape)?;

    let mut graph = Graph::new();
    let x = add_i64_input(&mut graph, shape);
    let slice = add_i64_slice_operation(&mut graph, x, &output, case).unwrap();
    graph.set_outputs(vec![slice]).unwrap();

    let _runtime = skip_if_stub(runtime_for_device(cuda_device()))?;
    let mut compiled = cuda::compile_graph(&graph, &GraphCudaOptions::default()).unwrap();
    let inputs = HashMap::from([("x".to_string(), input)]);

    Some(compiled.run_graph_owned_outputs(&inputs).unwrap())
}
