use crate::{backend::registry::Operation as RegistryOperation, graph::test_support::*};

#[test]
fn test_graph_pass_pipeline_reports_pass_results() {
    let shape = Shape::create(vec![2]).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, shape.clone(), Device::Cpu);
    graph
        .add_initializer(
            "x",
            Tensor::from_vec(Device::Cpu, &shape, vec![1.0_f32, 2.0]).unwrap(),
        )
        .unwrap();
    graph.set_outputs(vec![input]).unwrap();

    let (_, results) = GraphPassPipeline::default().normalize(&graph).unwrap();
    assert_eq!(
        results
            .iter()
            .map(|result| result.pass_name)
            .collect::<Vec<_>>(),
        vec![
            "validate_initializers",
            "validate_value_metadata",
            "validate_node_contracts",
            "validate_outputs",
            "decompose_operations",
            "fold_small_constants",
            "remove_redundant_ops",
            "recognize_row_operations",
            "decompose_scaled_dot_product_attention",
            "fold_grouped_query_attention",
            "elide_contiguous_copies",
            "materialize_contiguous_operands",
        ]
    );
    for (index, result) in results.iter().enumerate() {
        if index == 6 {
            continue;
        }
        assert_eq!(result.checked_nodes, value_count(&graph));
        assert!(result.remapped_nodes.is_empty());
    }
    assert_eq!(results[6].checked_nodes, 1);
    assert_eq!(results[6].remapped_nodes, [(input, input)]);
    assert_eq!(results[3].checked_nodes, 1);
}

#[test]
fn test_graph_pass_pipeline_runs_single_pass() {
    let shape = Shape::create(vec![2]).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, shape, Device::Cpu);
    graph.set_outputs(vec![input]).unwrap();

    let (_, mut results) = GraphPassPipeline::new(vec![GraphPassKind::ValidateOutputs])
        .normalize(&graph)
        .unwrap();
    let result = results.remove(0);

    assert_eq!(result.pass_name, "validate_outputs");
    assert_eq!(result.checked_nodes, 1);
    assert!(result.remapped_nodes.is_empty());
}

#[test]
fn test_graph_compile_with_config_and_passes_uses_supplied_pipeline() {
    let shape = Shape::create(vec![2]).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, shape.clone(), Device::Cpu);
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let invalid = graph.add_operation_request(GraphOperationRequest::scalar(
        None,
        RegistryOperation::Scale,
        ValueId::new(99),
        2.0,
        graph_operation_output(&output),
    ));
    assert!(matches!(invalid, Err(Error::MissingNode { .. })));

    let empty_pipeline = GraphPassPipeline::new(Vec::new());
    let valid = add_scalar_operation(&mut graph, RegistryOperation::Scale, input, 2.0, &output);
    graph.set_outputs(vec![valid]).unwrap();
    let compiled = graph
        .compile_with_options_and_passes(GraphCompileOptions::new(), &empty_pipeline)
        .unwrap();
    assert_eq!(compiled.value_order(), [input, valid]);
}

#[test]
fn test_graph_compile_with_passes_honors_explicit_pruning_stage_when_config_keeps_dead_nodes() {
    let shape = Shape::create(vec![2]).unwrap();
    let mut graph = Graph::new();
    let live = graph.add_input("live", DataType::F32, shape.clone(), Device::Cpu);
    let dead = graph.add_input("dead", DataType::F32, shape.clone(), Device::Cpu);
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let live_output =
        add_scalar_operation(&mut graph, RegistryOperation::Scale, live, 2.0, &output);
    let _dead_output =
        add_scalar_operation(&mut graph, RegistryOperation::Scale, dead, 99.0, &output);
    graph.set_outputs(vec![live_output]).unwrap();

    let config = GraphCompileOptions::new().with_dead_node_pruning(false);
    let validation_only = GraphPassPipeline::default();
    let unpruned = graph
        .compile_with_options_and_passes(config, &validation_only)
        .unwrap();
    assert_eq!(unpruned.to_ir().values().len(), 4);

    let pruning_pipeline = GraphPassPipeline::new(vec![
        GraphPassKind::ValidateInitializers,
        GraphPassKind::PruneDeadNodes,
        GraphPassKind::ValidateOutputs,
    ]);
    let pruned = graph
        .compile_with_options_and_passes(config, &pruning_pipeline)
        .unwrap();
    let pruned_ir = pruned.to_ir();
    assert_eq!(pruned_ir.values().len(), 2);
    assert_eq!(pruned_ir.output_values(), &[ValueId::new(1)]);
}
