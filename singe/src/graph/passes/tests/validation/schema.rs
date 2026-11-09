use super::*;
use crate::graph::{GraphPrepareOptions, node::NodeKind};

#[test]
fn test_graph_compile_rejects_constant_contract_mismatch() {
    let shape = Shape::create(vec![2]).unwrap();
    let mut graph = Graph::new();
    let constant =
        graph.add_constant(Tensor::from_vec(Device::Cpu, &shape, vec![1.0_f32, 2.0]).unwrap());
    graph.replace_node_output_data_type_for_test(constant, DataType::I32);
    graph.set_outputs(vec![constant]).unwrap();

    assert!(matches!(
        graph.compile(),
        Err(Error::DataTypeMismatch { name, expected, actual })
            if name == "graph pass validate_value_metadata node 0 constant"
                && expected == DataType::F32
                && actual == DataType::I32
    ));
}

#[test]
fn test_graph_compile_rejects_cast_contract_mismatch() {
    let shape = Shape::create(vec![2]).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, shape.clone(), Device::Cpu);
    let output = Tensor::zeroed_on(Device::Cpu, DataType::I32, &shape).unwrap();
    let cast = graph
        .add_named_operation(
            None,
            NodeKind::cast(input, DataType::I32),
            DataType::I32,
            output.shape().clone(),
            output.device(),
        )
        .unwrap();
    graph.replace_node_output_data_type_for_test(cast, DataType::F32);
    graph.set_outputs(vec![cast]).unwrap();

    assert!(matches!(
        graph.compile(),
        Err(Error::DataTypeMismatch { name, expected, actual })
            if name == "graph pass validate_node_contracts node 1 cast"
                && expected == DataType::I32
                && actual == DataType::F32
    ));
}

#[test]
fn test_graph_compile_rejects_copy_to_contract_mismatch() {
    let shape = Shape::create(vec![2]).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, shape.clone(), Device::Cpu);
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let copy = graph
        .add_named_operation(
            None,
            NodeKind::copy_to(input, Device::Cpu),
            output.data_type(),
            output.shape().clone(),
            output.device(),
        )
        .unwrap();
    graph.replace_node_output_device_for_test(copy, Device::Cuda(0));
    graph.set_outputs(vec![copy]).unwrap();

    assert!(matches!(
        graph.compile(),
        Err(Error::DeviceMismatch { name, expected, actual })
            if name == "graph pass validate_node_contracts node 1 copy_to"
                && expected == Device::Cpu
                && actual == Device::Cuda(0)
    ));
}

#[test]
fn test_graph_compile_rejects_schema_operand_count_mismatch() {
    let shape = Shape::create(vec![0]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let mut graph = Graph::new();
    let concat = add_operation_fixture(
        &mut graph,
        Operation::Concat,
        OperationAttributes::Concat(0),
        Vec::new(),
        &output,
    );
    graph.set_outputs(vec![concat]).unwrap();

    assert!(matches!(
        graph.compile(),
        Err(Error::Unsupported { operation, details, .. })
            if operation == "graph pass validate_node_contracts"
                && details.contains("operation concat has 0 operands")
                && details.contains("Variadic")
    ));
}

#[test]
fn test_graph_preparation_rejects_unavailable_compute_dtype() {
    let shape = Shape::create(vec![2]).unwrap();
    let mut graph = Graph::new();
    let lhs = graph.add_input("lhs", DataType::F4E2M1, shape.clone(), Device::Cpu);
    let rhs = graph.add_input("rhs", DataType::F4E2M1, shape.clone(), Device::Cpu);
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F4E2M1, &shape).unwrap();
    let add = add_binary_operation(&mut graph, RegistryOperation::Add, [lhs, rhs], &output);
    graph.set_outputs(vec![add]).unwrap();

    graph.compile().unwrap();
    assert!(matches!(
        graph.prepare(GraphPrepareOptions::new(Device::Cpu)),
        Err(Error::Unsupported { operation, details, .. })
            if operation == "add"
                && details.contains("execution CpuEager")
                && details.contains("input dtypes [F4E2M1, F4E2M1]")
    ));
}

#[test]
fn test_graph_compile_rejects_schema_input_dtype_mismatch() {
    let shape = Shape::create(vec![2]).unwrap();
    let mut graph = Graph::new();
    let lhs = graph.add_input("lhs", DataType::F32, shape.clone(), Device::Cpu);
    let rhs = graph.add_input("rhs", DataType::F4E2M1, shape.clone(), Device::Cpu);
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let add = add_binary_operation(&mut graph, RegistryOperation::Add, [lhs, rhs], &output);
    graph.set_outputs(vec![add]).unwrap();

    assert!(
        matches!(graph.compile(), Err(Error::Unsupported { operation, .. }) if operation == "promotion of an fp8 or fp4 type")
    );
}

#[test]
fn test_validate_node_contracts_accepts_schema_index_operand_dtype() {
    let weight_shape = Shape::create(vec![4, 3]).unwrap();
    let indices_shape = Shape::create(vec![2]).unwrap();
    let output_shape = Shape::create(vec![2, 3]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &output_shape).unwrap();
    let mut graph = Graph::new();
    let weight = graph.add_input("weight", DataType::F32, weight_shape, Device::Cpu);
    let indices = graph.add_input("indices", DataType::I64, indices_shape, Device::Cpu);
    let embedding = graph
        .add_named_operation(
            None,
            NodeKind::embedding(weight, indices),
            output.data_type(),
            output.shape().clone(),
            output.device(),
        )
        .unwrap();
    graph.set_outputs(vec![embedding]).unwrap();

    GraphPassPipeline::new(vec![GraphPassKind::ValidateNodeContracts])
        .normalize(&graph)
        .unwrap();
}

#[test]
fn test_graph_preparation_rejects_quantize_output_dtype_mismatch() {
    let shape = Shape::create(vec![2]).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, shape.clone(), Device::Cpu);
    let quantize = graph
        .add_named_quantization_operation(
            None,
            RegistryOperation::Quantize,
            input,
            0.5,
            0,
            DataType::I32,
            shape,
            Device::Cpu,
        )
        .unwrap();
    graph.set_outputs(vec![quantize]).unwrap();

    graph.compile().unwrap();
    assert!(matches!(
        graph.prepare(GraphPrepareOptions::new(Device::Cpu)),
        Err(Error::Unsupported { operation, details, .. })
            if operation == "quantize" && details.contains("execution CpuEager") && details.contains("F32")
    ));
}

#[test]
fn test_graph_preparation_rejects_dequantize_input_dtype_mismatch() {
    let shape = Shape::create(vec![2]).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, shape.clone(), Device::Cpu);
    let dequantize = graph
        .add_named_quantization_operation(
            None,
            RegistryOperation::Dequantize,
            input,
            0.5,
            0,
            DataType::F32,
            shape,
            Device::Cpu,
        )
        .unwrap();
    graph.set_outputs(vec![dequantize]).unwrap();

    graph.compile().unwrap();
    assert!(matches!(
        graph.prepare(GraphPrepareOptions::new(Device::Cpu)),
        Err(Error::Unsupported { operation, details, .. })
            if operation == "dequantize" && details.contains("execution CpuEager") && details.contains("F32")
    ));
}

#[test]
fn test_graph_compile_rejects_quantized_matmul_result_contract_mismatch() {
    let lhs_shape = Shape::create(vec![2, 2]).unwrap();
    let rhs_shape = Shape::create(vec![2, 2]).unwrap();
    let output_shape = Shape::create(vec![2, 2]).unwrap();
    let mut graph = Graph::new();
    let lhs = graph.add_input("lhs", DataType::U8, lhs_shape, Device::Cpu);
    let rhs = graph.add_input("rhs", DataType::U8, rhs_shape, Device::Cpu);
    let output_tensor = Tensor::zeroed_on(Device::Cpu, DataType::U8, &output_shape).unwrap();
    let output =
        add_quantized_matmul_operation(&mut graph, [lhs, rhs], 0.5, 1, 0.25, 2, &output_tensor);
    graph.set_outputs(vec![output]).unwrap();

    assert!(matches!(
        graph.compile(),
        Err(Error::DataTypeMismatch {
            expected: DataType::F32,
            actual: DataType::U8,
            ..
        })
    ));
}
