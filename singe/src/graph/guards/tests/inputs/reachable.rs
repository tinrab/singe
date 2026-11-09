use super::*;

#[test]
fn test_graph_signature_validates_unused_inputs() {
    let shape = Shape::create(vec![2]).unwrap();
    let mut graph = Graph::new();
    let live = graph.add_input("live", DataType::F32, shape.clone(), Device::Cpu);
    let dead = graph.add_input("dead", DataType::F32, shape.clone(), Device::Cpu);
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let live_output =
        add_binary_operation(&mut graph, RegistryOperation::Add, [live, live], &output);
    let _dead_output =
        add_binary_operation(&mut graph, RegistryOperation::Add, [dead, dead], &output);
    graph.set_outputs(vec![live_output]).unwrap();

    let compiled = graph.compile().unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("live".to_string(), cpu_tensor(&shape, vec![1.0_f32, 2.0]));
    inputs.insert("dead".to_string(), cpu_tensor(&shape, vec![10.0_f32, 20.0]));

    let outputs = compiled.run(&inputs).unwrap();
    assert_close!(&outputs[0].to_vec::<f32>().unwrap(), &[2.0, 4.0], 1e-6);
    inputs.remove("dead");
    assert!(matches!(
        compiled.run(&inputs),
        Err(Error::MissingInput { .. })
    ));
}
