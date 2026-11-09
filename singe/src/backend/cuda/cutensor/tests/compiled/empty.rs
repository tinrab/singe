use super::*;
use crate::graph::GraphCudaOptions;

#[test]
fn test_cuda_compiled_graph_skips_empty_cutensor_contraction() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let a_shape = Shape::create([0, 3]).unwrap();
    let b_shape = Shape::create([3, 2]).unwrap();
    let c_shape = Shape::create([0, 2]).unwrap();
    let Some(a) = maybe_cuda_f32_tensor(&a_shape, Vec::<f32>::new()) else {
        return;
    };
    let b = f32_tensor(&b_shape, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let c = f32_tensor(&c_shape, Vec::<f32>::new());
    let output = cuda_f32_output(&c_shape);
    let (graph, _) = contraction_graph(a_shape, b_shape, c_shape, &output, 1.0, 0.0);

    let Some(mut compiled) =
        skip_if_stub(cuda::compile_graph(&graph, &GraphCudaOptions::default()))
    else {
        return;
    };
    assert_eq!(compiled.stats().cutensor_child_graph_nodes(), 0);

    let outputs = compiled.run(&contraction_inputs(a, b, c)).unwrap();

    assert_eq!(outputs[0].shape().dimensions(), &[0, 2]);
    assert!(outputs[0].to_vec::<f32>().unwrap().is_empty());
}
