use super::*;
use crate::graph::{GraphCudaOptions, node::NodeKind};

#[test]
fn test_cuda_graph_run_infers_strided_alias_layout_for_non_f32() {
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

    let mut graph = Graph::new();
    let x = graph.add_input("x", DataType::F64, shape, Device::Cuda(0));
    let transposed = graph
        .add_named_operation(
            None,
            NodeKind::transpose(x, 0, 1),
            output.data_type(),
            output.shape().clone(),
            output.device(),
        )
        .unwrap();
    graph.set_outputs(vec![transposed]).unwrap();

    let inputs = HashMap::from([("x".to_string(), input)]);
    let outputs = graph.run(&inputs).unwrap();
    assert_eq!(outputs[0].layout().strides(), &[1, 3]);
    assert_eq!(
        outputs[0].to_vec::<f64>().unwrap(),
        [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
}

#[test]
fn test_cuda_compiled_graph_accepts_alias_ops_with_non_f32_dtype() {
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

    let mut graph = Graph::new();
    let x = graph.add_input("x", DataType::F64, shape, Device::Cuda(0));
    let transposed = graph
        .add_named_operation(
            None,
            NodeKind::transpose(x, 0, 1),
            output.data_type(),
            output.shape().clone(),
            output.device(),
        )
        .unwrap();
    graph.set_outputs(vec![transposed]).unwrap();

    let Some(_) = skip_if_stub(runtime_for_device(Device::Cuda(0))) else {
        return;
    };
    let mut compiled = cuda::compile_graph(&graph, &GraphCudaOptions::default()).unwrap();
    let inputs = HashMap::from([("x".to_string(), input)]);
    let outputs = compiled.run_graph_owned_outputs(&inputs).unwrap();

    assert_eq!(outputs[0].data_type(), DataType::F64);
    assert_eq!(outputs[0].shape().dimensions(), output_shape.dimensions());
    assert_eq!(outputs[0].layout().strides(), &[1, 3]);
    assert_eq!(outputs[0].storage_offset(), 0);
}
