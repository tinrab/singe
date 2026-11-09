use super::*;

#[test]
fn test_graph_prune_dead_nodes_pass_remaps_dependencies_outputs_and_guards() {
    let shape = Shape::create(vec![2]).unwrap();
    let mut graph = Graph::new();
    let dead = graph.add_input("dead", DataType::F32, shape.clone(), Device::Cpu);
    graph
        .add_initializer(
            "dead",
            Tensor::from_vec(Device::Cpu, &shape, vec![9.0_f32, 9.0]).unwrap(),
        )
        .unwrap();
    graph
        .add_input_dimension_guard("dead", 0, "dead_batch", Some(1), Some(4))
        .unwrap();
    let live = graph.add_input("live", DataType::F32, shape.clone(), Device::Cpu);
    graph
        .add_initializer(
            "live",
            Tensor::from_vec(Device::Cpu, &shape, vec![1.0_f32, 2.0]).unwrap(),
        )
        .unwrap();
    graph
        .add_input_dimension_guard("live", 0, "batch", Some(1), Some(4))
        .unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let _dead_output =
        add_scalar_operation(&mut graph, RegistryOperation::Scale, dead, 99.0, &output);
    let live_output =
        add_scalar_operation(&mut graph, RegistryOperation::Scale, live, 2.0, &output);
    graph.set_outputs(vec![live_output]).unwrap();

    let (_, mut prune_results) = GraphPassPipeline::new(vec![GraphPassKind::PruneDeadNodes])
        .normalize(&graph)
        .unwrap();
    let result = prune_results.remove(0);
    assert_eq!(result.pass_name, "prune_dead_nodes");
    assert_eq!(result.checked_nodes, 2);
    assert_eq!(
        result.remapped_nodes,
        vec![(live, ValueId::new(0)), (live_output, ValueId::new(1))]
    );

    let pipeline = GraphPassPipeline::new(vec![
        GraphPassKind::PruneDeadNodes,
        GraphPassKind::ValidateOutputs,
    ]);
    let (normalized, results) = pipeline.normalize(&graph).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].pass_name, "prune_dead_nodes");
    assert_eq!(results[0].checked_nodes, 2);
    assert_eq!(
        results[0].remapped_nodes,
        vec![(live, ValueId::new(0)), (live_output, ValueId::new(1))]
    );
    assert_eq!(results[1].pass_name, "validate_outputs");
    assert_eq!(results[1].checked_nodes, 1);
    assert!(results[1].remapped_nodes.is_empty());
    let normalized_ir = normalized.to_ir().unwrap();
    assert_eq!(normalized_ir.values().len(), 2);
    assert_eq!(normalized_ir.output_values(), &[ValueId::new(1)]);
    let normalized_input = normalized_ir.value(ValueId::new(0)).unwrap().binding();
    assert!(normalized_input.is_initializer_input());
    assert_eq!(normalized_input.input_name(), Some("live"));
    assert!(!normalized_ir.values().iter().any(|value| {
        value.binding().is_initializer_input() && value.binding().input_name() == Some("dead")
    }));
    assert!(normalized_ir.input_dimension_guard("live", 0).is_some());
    assert!(normalized_ir.input_dimension_guard("dead", 0).is_none());
    let normalized_scale = normalized_ir
        .operation(normalized_ir.result_for_value(ValueId::new(1)).unwrap().0)
        .unwrap();
    assert_eq!(normalized_scale.opcode(), Some(Operation::Scale));
    assert_eq!(normalized_scale.inputs(), &[ValueId::new(0)]);
    assert_eq!(
        normalized_scale.attributes(),
        &OperationAttributes::Scalar((2.0).into())
    );

    let (_, run_results) = pipeline.normalize(&graph).unwrap();
    assert_eq!(run_results, results);

    let compiled = graph.compile().unwrap();
    let compiled_ir = compiled.to_ir();
    assert_eq!(compiled_ir.values().len(), 2);
    assert_eq!(compiled_ir.output_values(), &[ValueId::new(1)]);
    let compiled_input = compiled_ir.value(ValueId::new(0)).unwrap().binding();
    assert!(compiled_input.is_initializer_input());
    assert_eq!(compiled_input.input_name(), Some("live"));
    assert!(!compiled_ir.values().iter().any(|value| {
        value.binding().is_initializer_input() && value.binding().input_name() == Some("dead")
    }));
    assert!(compiled_ir.input_dimension_guard("live", 0).is_some());
    assert!(compiled_ir.input_dimension_guard("dead", 0).is_none());
    let compiled_scale = compiled_ir
        .operation(compiled_ir.result_for_value(ValueId::new(1)).unwrap().0)
        .unwrap();
    assert_eq!(compiled_scale.opcode(), Some(Operation::Scale));
    assert_eq!(compiled_scale.inputs(), &[ValueId::new(0)]);
    assert_eq!(
        compiled_scale.attributes(),
        &OperationAttributes::Scalar((2.0).into())
    );
}
