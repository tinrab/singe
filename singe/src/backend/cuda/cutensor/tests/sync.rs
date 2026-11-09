use super::*;
use crate::{backend::cuda::CudaStream, graph::GraphCudaOptions};

#[test]
fn test_cuda_compiled_graph_drop_waits_for_pending_cutensor_workspace_work() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let a_shape = Shape::create([2, 3]).unwrap();
    let b_shape = Shape::create([3, 2]).unwrap();
    let c_shape = Shape::create([2, 2]).unwrap();
    let Some(a) = skip_if_stub(Tensor::from_vec(
        Device::Cuda(0),
        &a_shape,
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
    )) else {
        return;
    };
    let b = Tensor::from_vec(
        Device::Cuda(0),
        &b_shape,
        vec![7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0],
    )
    .unwrap();
    let c = Tensor::from_vec(Device::Cuda(0), &c_shape, vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let output = Tensor::zeroed_on(Device::Cuda(0), DataType::F32, &c_shape).unwrap();

    let mut graph = Graph::new();
    let a_id = graph.add_input("a", DataType::F32, a_shape, Device::Cuda(0));
    let b_id = graph.add_input("b", DataType::F32, b_shape, Device::Cuda(0));
    let c_id = graph.add_input("c", DataType::F32, c_shape, Device::Cuda(0));
    let contraction = graph
        .add_operation_request(GraphOperationRequest::contraction(
            None,
            a_id,
            vec![0, 2],
            b_id,
            vec![2, 1],
            c_id,
            vec![0, 1],
            vec![0, 1],
            0.5,
            2.0,
            operation_output(&output),
        ))
        .unwrap();
    graph.set_outputs(vec![contraction]).unwrap();

    let stream = CudaStream::create(Device::Cuda(0)).unwrap();
    let outputs = {
        let Some(mut compiled) =
            skip_if_stub(cuda::compile_graph(&graph, &GraphCudaOptions::default()))
        else {
            return;
        };
        assert_eq!(compiled.stats().cutensor_child_graph_nodes(), 1);
        let outputs = compiled
            .enqueue_on(
                &HashMap::from([
                    ("a".to_string(), a),
                    ("b".to_string(), b),
                    ("c".to_string(), c),
                ]),
                &stream,
            )
            .unwrap();
        assert_eq!(compiled.stats().pending_runs(), 1);
        outputs
    };

    assert!(stream.raw().query().unwrap());
    let output = outputs.into_iter().next().unwrap();
    // SAFETY: graph destruction completed the retained producer and copy-out, leaving this fully initialized contiguous result uniquely owned.
    let output = unsafe { output.assume_init() }.unwrap();
    assert_close!(&output.to_vec::<f32>().unwrap(), &[31.0, 36.0, 75.5, 85.0],);
}
