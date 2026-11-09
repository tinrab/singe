use super::*;

#[test]
fn test_graph_fingerprint_distinguishes_quantization_attributes() {
    fn graph_with_quantize(scale: f32, zero_point: i64, data_type: DataType) -> GraphFingerprint {
        let shape = Shape::create([4]).unwrap();
        let output = Tensor::zeroed_on(Device::Cpu, data_type, &shape).unwrap();
        let mut graph = Graph::new();
        let input = graph.add_input("x", DataType::F32, shape, Device::Cpu);
        let output = add_quantization_operation(
            &mut graph,
            RegistryOperation::Quantize,
            input,
            scale,
            zero_point,
            data_type,
            &output,
        );
        graph.set_outputs(vec![output]).unwrap();
        graph.fingerprint().unwrap()
    }

    fn graph_with_dequantize(scale: f32, zero_point: i64) -> GraphFingerprint {
        let shape = Shape::create([4]).unwrap();
        let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
        let mut graph = Graph::new();
        let input = graph.add_input("x", DataType::U8, shape, Device::Cpu);
        let output = add_quantization_operation(
            &mut graph,
            RegistryOperation::Dequantize,
            input,
            scale,
            zero_point,
            DataType::F32,
            &output,
        );
        graph.set_outputs(vec![output]).unwrap();
        graph.fingerprint().unwrap()
    }

    let quantize_base = graph_with_quantize(0.5, 2, DataType::U8);
    assert!(quantize_base.is_cacheable());
    assert_ne!(quantize_base, graph_with_quantize(0.25, 2, DataType::U8));
    assert_ne!(quantize_base, graph_with_quantize(0.5, 3, DataType::U8));
    assert_ne!(quantize_base, graph_with_quantize(0.5, 2, DataType::I8));

    let dequantize_base = graph_with_dequantize(0.5, 2);
    assert!(dequantize_base.is_cacheable());
    assert_ne!(dequantize_base, graph_with_dequantize(0.25, 2));
    assert_ne!(dequantize_base, graph_with_dequantize(0.5, 3));
}

#[test]
fn test_graph_fingerprint_distinguishes_quantized_matmul_attributes() {
    fn graph_with_quantized_matmul(
        lhs_scale: f32,
        lhs_zero_point: i64,
        rhs_scale: f32,
        rhs_zero_point: i64,
    ) -> GraphFingerprint {
        let lhs_shape = Shape::create([2, 3]).unwrap();
        let rhs_shape = Shape::create([3, 2]).unwrap();
        let output_shape = Shape::create([2, 2]).unwrap();
        let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &output_shape).unwrap();
        let mut graph = Graph::new();
        let lhs = graph.add_input("lhs", DataType::I8, lhs_shape, Device::Cpu);
        let rhs = graph.add_input("rhs", DataType::U8, rhs_shape, Device::Cpu);
        let output = add_quantized_matmul_operation(
            &mut graph,
            [lhs, rhs],
            lhs_scale,
            lhs_zero_point,
            rhs_scale,
            rhs_zero_point,
            &output,
        );
        graph.set_outputs(vec![output]).unwrap();
        graph.fingerprint().unwrap()
    }

    let base = graph_with_quantized_matmul(0.5, 1, 0.25, 2);
    assert!(base.is_cacheable());
    assert_ne!(base, graph_with_quantized_matmul(0.25, 1, 0.25, 2));
    assert_ne!(base, graph_with_quantized_matmul(0.5, 3, 0.25, 2));
    assert_ne!(base, graph_with_quantized_matmul(0.5, 1, 0.5, 2));
    assert_ne!(base, graph_with_quantized_matmul(0.5, 1, 0.25, 4));
}
