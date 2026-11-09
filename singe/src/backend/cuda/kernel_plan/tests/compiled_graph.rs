use super::*;
use crate::{
    backend::cuda::compiled::executable::GraphCompileProviderDecision, graph::GraphCudaOptions,
};

#[test]
fn test_cuda_compiled_graph_add_f32_contiguous_runs_a_generated_kernel_child_graph() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let len = 129usize;
    let shape = Shape::create([i64::try_from(len).unwrap()]).unwrap();
    let lhs_values = (0..len)
        .map(|value| (len as f32).mul_add(-0.5, value as f32))
        .collect::<Vec<_>>();
    let rhs_values = (0..len)
        .map(|value| (value as f32).mul_add(0.25, 1.0))
        .collect::<Vec<_>>();
    let expected = lhs_values
        .iter()
        .zip(&rhs_values)
        .map(|(lhs, rhs)| lhs + rhs)
        .collect::<Vec<_>>();
    let Some(lhs) = skip_if_stub(Tensor::from_vec(Device::Cuda(0), &shape, lhs_values)) else {
        return;
    };
    let rhs = Tensor::from_vec(Device::Cuda(0), &shape, rhs_values).unwrap();

    let mut graph = Graph::new();
    let lhs_id = graph.add_input("lhs", DataType::F32, shape.clone(), Device::Cuda(0));
    let rhs_id = graph.add_input("rhs", DataType::F32, shape.clone(), Device::Cuda(0));
    let output = graph
        .add_operation_request(GraphOperationRequest::binary(
            None,
            Operation::Add,
            [lhs_id, rhs_id],
            GraphOperationOutput::new(DataType::F32, shape, Device::Cuda(0)),
        ))
        .unwrap();
    graph.set_outputs(vec![output]).unwrap();

    let Some(mut compiled) =
        skip_if_stub(cuda::compile_graph(&graph, &GraphCudaOptions::default()))
    else {
        return;
    };
    let stats = compiled.stats();
    assert_eq!(stats.kernel_child_graph_nodes(), 1);
    assert_eq!(stats.child_graph_nodes(), 1);
    assert_eq!(stats.native_provider_nodes(), 1);

    let add_report = compiled
        .report()
        .operations()
        .iter()
        .find(|operation| operation.operation() == operation_name(Operation::Add))
        .expect("missing add report");
    assert!(
        add_report
            .candidates()
            .iter()
            .any(|candidate| { candidate.provider() == CudaGraphProvider::Kernel }),
        "{:?}",
        add_report.candidates()
    );
    assert_eq!(
        add_report
            .realized_decision()
            .and_then(GraphCompileProviderDecision::provider),
        Some(CudaGraphProvider::Kernel),
    );
    assert_eq!(add_report.selected_algorithm(), Some("kernel add"));

    let outputs = compiled
        .run(&HashMap::from([
            ("lhs".to_string(), lhs),
            ("rhs".to_string(), rhs),
        ]))
        .unwrap();

    assert_close!(&outputs[0].to_vec::<f32>().unwrap(), &expected);
}
