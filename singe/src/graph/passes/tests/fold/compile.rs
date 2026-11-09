use super::*;

#[test]
fn test_graph_compile_folds_and_prunes_small_constants_by_default() {
    let shape = Shape::create(vec![2]).unwrap();
    let lhs = Tensor::from_vec(Device::Cpu, &shape, vec![2.0_f32, 4.0]).unwrap();
    let rhs = Tensor::from_vec(Device::Cpu, &shape, vec![1.0_f32, 3.0]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let mut graph = Graph::new();
    let lhs = graph.add_constant(lhs);
    let rhs = graph.add_constant(rhs);
    let sub = add_binary_operation(&mut graph, RegistryOperation::Sub, [lhs, rhs], &output);
    graph.set_outputs(vec![sub]).unwrap();

    let compiled = graph.compile().unwrap();

    let compiled_ir = compiled.to_ir();
    assert_eq!(compiled_ir.values().len(), 1);
    assert_eq!(compiled_ir.output_values(), &[ValueId::new(0)]);
    let tensor = compiled_ir
        .constant_tensor(ValueId::new(0))
        .expect("expected folded constant");
    assert_close!(&tensor.to_vec::<f32>().unwrap(), &[1.0, 1.0]);
}

#[test]
fn test_graph_compile_deduplicates_equivalent_folded_constants_by_default() {
    let shape = Shape::create(vec![2]).unwrap();
    let lhs = Tensor::from_vec(Device::Cpu, &shape, vec![1.0_f32, 2.0]).unwrap();
    let rhs = Tensor::from_vec(Device::Cpu, &shape, vec![3.0_f32, 4.0]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let mut graph = Graph::new();
    let lhs = graph.add_constant(lhs);
    let rhs = graph.add_constant(rhs);
    let first = add_binary_operation(&mut graph, RegistryOperation::Add, [lhs, rhs], &output);
    let second = add_binary_operation(&mut graph, RegistryOperation::Add, [rhs, lhs], &output);
    graph.set_outputs(vec![first, second]).unwrap();

    let compiled = graph.compile().unwrap();

    let compiled_ir = compiled.to_ir();
    assert_eq!(compiled_ir.values().len(), 1);
    assert_eq!(
        compiled_ir.output_values(),
        &[ValueId::new(0), ValueId::new(0)]
    );
    let tensor = compiled_ir
        .constant_tensor(ValueId::new(0))
        .expect("expected folded constant");
    assert_close!(&tensor.to_vec::<f32>().unwrap(), &[4.0, 6.0]);
}

#[test]
fn test_fold_small_constants_skips_unsupported_cpu_dtypes_and_keeps_folding() {
    let shape = Shape::create([2]).unwrap();
    // The CPU computes no packed FP4 values, so the FP4 rectification cannot fold.
    let fp4 = Tensor::zeroed_on(Device::Cpu, DataType::F4E2M1, &shape).unwrap();
    let float = Tensor::from_vec(Device::Cpu, &shape, vec![-1_f32, 2.]).unwrap();
    let mut graph = Graph::new();
    let fp4_input = graph.add_constant(fp4.clone());
    let fp4_relu = add_unary_operation(&mut graph, RegistryOperation::Relu, fp4_input, &fp4);
    let float_input = graph.add_constant(float.clone());
    let float_relu = add_unary_operation(&mut graph, RegistryOperation::Relu, float_input, &float);
    graph.set_outputs(vec![fp4_relu, float_relu]).unwrap();

    let pipeline = GraphPassPipeline::new(vec![GraphPassKind::FoldSmallConstants]);
    let (folded, _) = pipeline.normalize(&graph).unwrap();

    assert_eq!(
        folded.node(fp4_relu).unwrap().operation(),
        Some(RegistryOperation::Relu)
    );
    assert_eq!(
        folded_constant_tensor(&folded, float_relu)
            .to_vec::<f32>()
            .unwrap(),
        [0., 2.]
    );
}
