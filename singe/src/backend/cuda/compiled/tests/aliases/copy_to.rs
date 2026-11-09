use super::*;
use crate::graph::{GraphCudaOptions, node::NodeKind};

#[test]
fn test_cuda_compiled_graph_rejects_mixed_device_copy_to() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let shape = Shape::create([3]).unwrap();
    let Some(output) = skip_if_stub(Tensor::zeroed_on(Device::Cuda(0), DataType::F32, &shape))
    else {
        return;
    };

    let mut graph = Graph::new();
    let source = graph.add_input("source", DataType::F32, shape, Device::Cpu);
    let copied = graph
        .add_named_operation(
            None,
            NodeKind::copy_to(source, Device::Cuda(0)),
            output.data_type(),
            output.shape().clone(),
            output.device(),
        )
        .unwrap();
    graph.set_outputs(vec![copied]).unwrap();

    let Some(_) = skip_if_stub(runtime_for_device(Device::Cuda(0))) else {
        return;
    };
    assert_device_mismatch(cuda::compile_graph(&graph, &GraphCudaOptions::default()));
}

#[test]
fn test_cuda_graph_run_accepts_copy_to_for_non_f32_dtype() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let shape = Shape::create([3]).unwrap();
    let i64_values = vec![1_i64, 2_i64, 3_i64];
    let Some(input) = skip_if_stub(Tensor::from_vec(Device::Cpu, &shape, i64_values.clone()))
    else {
        return;
    };
    let Some(output) = skip_if_stub(Tensor::zeroed_on(Device::Cuda(0), DataType::I64, &shape))
    else {
        return;
    };

    let mut graph = Graph::new();
    let source = graph.add_input("source", DataType::I64, shape.clone(), Device::Cpu);
    let copied = graph
        .add_named_operation(
            None,
            NodeKind::copy_to(source, Device::Cuda(0)),
            output.data_type(),
            output.shape().clone(),
            output.device(),
        )
        .unwrap();
    graph.set_outputs(vec![copied]).unwrap();

    let inputs = HashMap::from([("source".to_string(), input)]);
    let outputs = graph.run(&inputs).unwrap();

    assert_eq!(outputs[0].device(), Device::Cuda(0));
    assert_eq!(outputs[0].data_type(), DataType::I64);
    assert_eq!(outputs[0].shape().dimensions(), shape.dimensions());
    assert_eq!(outputs[0].to_vec::<i64>().unwrap(), i64_values);
}

#[test]
fn test_cuda_compiled_graph_accepts_copy_to_for_non_f32_dtype() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let shape = Shape::create([3]).unwrap();
    let i64_values = vec![4_i64, 5_i64, 6_i64];
    let Some(input) = skip_if_stub(Tensor::from_vec(
        Device::Cuda(0),
        &shape,
        i64_values.clone(),
    )) else {
        return;
    };
    let Some(output) = skip_if_stub(Tensor::zeroed_on(Device::Cuda(0), DataType::I64, &shape))
    else {
        return;
    };

    let mut graph = Graph::new();
    let source = graph.add_input("source", DataType::I64, shape.clone(), Device::Cuda(0));
    let copied = graph
        .add_named_operation(
            None,
            NodeKind::copy_to(source, Device::Cuda(0)),
            output.data_type(),
            output.shape().clone(),
            output.device(),
        )
        .unwrap();
    graph.set_outputs(vec![copied]).unwrap();

    let Some(_) = skip_if_stub(runtime_for_device(Device::Cuda(0))) else {
        return;
    };
    let mut compiled = cuda::compile_graph(&graph, &GraphCudaOptions::default()).unwrap();
    let inputs = HashMap::from([("source".to_string(), input)]);
    let outputs = compiled.run(&inputs).unwrap();

    assert_eq!(outputs[0].device(), Device::Cuda(0));
    assert_eq!(outputs[0].data_type(), DataType::I64);
    assert_eq!(outputs[0].shape().dimensions(), shape.dimensions());
    assert_eq!(outputs[0].to_vec::<i64>().unwrap(), i64_values);
}
