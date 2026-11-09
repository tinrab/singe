mod ir;
mod outputs;

use super::*;

fn cpu_tensor<T: TensorDataType>(shape: &Shape, values: Vec<T>) -> Tensor {
    Tensor::from_vec(Device::Cpu, shape, values).unwrap()
}

fn core_capture_inputs(matrix_shape: &Shape, vector_shape: &Shape) -> [(&'static str, Tensor); 5] {
    [
        (
            "lhs",
            cpu_tensor(matrix_shape, vec![1.0_f32, 2.0, 3.0, 4.0]),
        ),
        (
            "rhs",
            cpu_tensor(matrix_shape, vec![5.0_f32, 6.0, 7.0, 8.0]),
        ),
        ("qlhs", cpu_tensor(matrix_shape, vec![3_i8, 5, 7, 9])),
        ("qrhs", cpu_tensor(matrix_shape, vec![1_u8, 3, 5, 7])),
        ("x", cpu_tensor(vector_shape, vec![1.0_f32, 2.0])),
    ]
}

fn core_replay_inputs(matrix_shape: &Shape, vector_shape: &Shape) -> HashMap<String, Tensor> {
    HashMap::from([
        (
            "lhs".to_string(),
            cpu_tensor(matrix_shape, vec![1.0_f32, 2.0, 3.0, 4.0]),
        ),
        (
            "rhs".to_string(),
            cpu_tensor(matrix_shape, vec![1.0_f32, 0.0, 0.0, 1.0]),
        ),
        (
            "qlhs".to_string(),
            cpu_tensor(matrix_shape, vec![1_i8, 3, 5, 7]),
        ),
        (
            "qrhs".to_string(),
            cpu_tensor(matrix_shape, vec![2_u8, 4, 6, 8]),
        ),
        (
            "x".to_string(),
            cpu_tensor(vector_shape, vec![2.0_f32, 3.0]),
        ),
    ])
}

#[test]
fn test_graph_capture_records_tensor_core_methods() {
    let matrix_shape = Shape::create(vec![2, 2]).unwrap();
    let vector_shape = Shape::create(vec![2]).unwrap();

    let (captured, captured_outputs) = Graph::capture(
        core_capture_inputs(&matrix_shape, &vector_shape),
        |inputs| {
            let matmul = inputs[0].matmul(&inputs[1])?;
            let quantized_matmul = inputs[2].quantized_matmul(&inputs[3], 0.5, 1, 0.25, 2)?;
            let cast = inputs[4].cast(DataType::I32)?;
            let quantized = inputs[4].quantize(0.5, 1, DataType::I8)?;
            let dequantized = quantized.dequantize(0.5, 1)?;
            let scaled = inputs[4].mul_scalar(2.0)?;
            let padded = inputs[4].pad(&[(1, 1)], -1.0)?;
            let clamped = inputs[4].clamp(1.25, 1.75)?;
            Ok(vec![
                matmul,
                quantized_matmul,
                cast,
                dequantized,
                scaled,
                padded,
                clamped,
            ])
        },
    )
    .unwrap();

    outputs::assert_core_capture_outputs(&captured_outputs);
    let ir = captured.to_ir().unwrap();
    let replay_outputs = captured
        .run(&core_replay_inputs(&matrix_shape, &vector_shape))
        .unwrap();
    assert_close!(
        &replay_outputs[1].to_vec::<f32>().unwrap(),
        &[1.0, 1.5, 3.0, 5.5],
        1e-6,
    );

    ir::assert_matmul_capture_ir(&ir);
    ir::assert_numeric_capture_ir(&ir);
}
