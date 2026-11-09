use super::*;
use crate::graph::GraphCudaOptions;

#[test]
fn test_cuda_compiled_graph_accepts_permute_alias_for_non_f32_dtype() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let shape = Shape::create([2, 3]).unwrap();
    let output_shape = Shape::create([3, 2]).unwrap();
    let Some(input) = skip_if_stub(Tensor::from_vec(
        Device::Cuda(0),
        &shape,
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    )) else {
        return;
    };
    let Some(output) = skip_if_stub(Tensor::zeroed_on(
        Device::Cuda(0),
        DataType::F64,
        &output_shape,
    )) else {
        return;
    };

    let graph = build_permute_graph(shape, &output, vec![1, 0]);

    let Some(_) = skip_if_stub(runtime_for_device(Device::Cuda(0))) else {
        return;
    };
    let mut compiled = cuda::compile_graph(&graph, &GraphCudaOptions::default()).unwrap();
    let inputs = HashMap::from([("x".to_string(), input)]);
    let outputs = compiled.run_graph_owned_outputs(&inputs).unwrap();

    assert_eq!(outputs[0].shape().dimensions(), output_shape.dimensions());
    assert_eq!(outputs[0].layout().strides(), &[1, 3]);
    assert_eq!(outputs[0].storage_offset(), 0);
}
