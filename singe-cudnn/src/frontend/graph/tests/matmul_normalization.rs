use super::*;

#[test]
fn test_matmul_accepts_explicit_output_data_type_distinct_from_compute_type() -> Result<()> {
    let mut graph = Graph::new();
    let a = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8])?,
    ));
    let b = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 8, 16])?,
    ));
    let c = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 16])?,
    ));

    graph.matmul(a, b, c, DataType::F32)?;

    Ok(())
}

#[test]
fn test_matmul_infer_with_config_uses_graph_intermediate_data_type() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_intermediate_data_type(DataType::BF16);
    let a = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8])?,
    ));
    let b = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 8, 16])?,
    ));

    let c = graph.matmul_infer_with_config(a, b, MatmulConfig::new(DataType::F32))?;

    assert_eq!(graph.tensor_config(c)?.data_type, DataType::BF16);

    Ok(())
}

#[test]
fn test_matmul_with_config_preserves_operation_name() -> Result<()> {
    let mut graph = Graph::new();
    let a = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8])?,
    ));
    let b = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 8, 16])?,
    ));
    let c = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 16])?,
    ));

    graph.matmul_with_config(a, b, c, MatmulConfig::new(DataType::F32).with_name("GEMM"))?;

    let Operation::Matmul { config, .. } = &graph.operations[0] else {
        panic!("expected matmul operation");
    };
    assert_eq!(config.name(), Some("GEMM"));

    Ok(())
}

#[test]
fn test_matmul_fp8_infer_creates_expected_output_shape() -> Result<()> {
    let mut graph = Graph::new();
    let a = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 4, 8])?,
    ));
    let b = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 8, 16])?,
    ));
    let descale_a = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let descale_b = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));

    let output = graph.matmul_fp8_infer(
        MatmulFp8Inputs {
            a: a,
            b: b,
            descale_a: descale_a,
            descale_b: descale_b,
        },
        DataType::F32,
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[2, 4, 16]);

    Ok(())
}

#[test]
fn test_matmul_fp8_infer_uses_graph_io_data_type() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_io_data_type(DataType::BF16);
    let a = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 4, 8])?,
    ));
    let b = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 8, 16])?,
    ));
    let descale_a = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let descale_b = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));

    let output = graph.matmul_fp8_infer(
        MatmulFp8Inputs {
            a: a,
            b: b,
            descale_a: descale_a,
            descale_b: descale_b,
        },
        DataType::F32,
    )?;

    assert_eq!(graph.tensor_config(output)?.data_type, DataType::BF16);

    Ok(())
}

#[test]
fn test_matmul_fp8_error_removes_generated_intermediates() -> Result<()> {
    let mut graph = Graph::new();
    let a = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 4, 8])?,
    ));
    let b = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 8, 16])?,
    ));
    let descale_a = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let descale_b = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 15])?,
    ));
    let tensor_count = graph.tensors.len();
    let operation_count = graph.operations.len();

    let error = graph
        .matmul_fp8(
            MatmulFp8Tensors::new(
                MatmulFp8Inputs {
                    a: a,
                    b: b,
                    descale_a: descale_a,
                    descale_b: descale_b,
                },
                output,
            ),
            DataType::F32,
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == output
            && operation == "matmul fp8 output"
            && expected == vec![2, 4, 16]
            && actual == vec![2, 4, 15]
    ));
    assert_eq!(graph.tensors.len(), tensor_count);
    assert_eq!(graph.operations.len(), operation_count);

    Ok(())
}

#[test]
fn test_matmul_fp8_rejects_mismatched_output_data_type_without_mutation() -> Result<()> {
    let mut graph = Graph::new();
    let a = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 4, 8])?,
    ));
    let b = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 8, 16])?,
    ));
    let descale_a = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let descale_b = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 16])?,
    ));
    let tensor_count = graph.tensors.len();
    let operation_count = graph.operations.len();

    let error = graph
        .matmul_fp8(
            MatmulFp8Tensors::new(
                MatmulFp8Inputs {
                    a: a,
                    b: b,
                    descale_a: descale_a,
                    descale_b: descale_b,
                },
                output,
            ),
            DataType::F32,
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorDataTypeMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == output
            && operation == "matmul fp8 output"
            && expected == DataType::F32
            && actual == DataType::F16
    ));
    assert_eq!(graph.tensors.len(), tensor_count);
    assert_eq!(graph.operations.len(), operation_count);

    Ok(())
}

#[test]
fn test_matmul_fp8_quantize_infer_creates_output_and_absolute_max_shapes() -> Result<()> {
    let mut graph = Graph::new();
    let a = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 4, 8])?,
    ));
    let b = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 8, 16])?,
    ));
    let descale_a = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let descale_b = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let scale_output = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));

    let outputs = graph.matmul_fp8_quantize_infer(
        MatmulFp8Inputs {
            a: a,
            b: b,
            descale_a: descale_a,
            descale_b: descale_b,
        },
        scale_output,
        DataType::F32,
    )?;

    assert_eq!(graph.shape(outputs.output)?.dimensions(), &[2, 4, 16]);
    assert_eq!(
        graph.shape(outputs.absolute_max_output)?.dimensions(),
        &[1, 1, 1]
    );

    Ok(())
}

#[test]
fn test_matmul_fp8_quantize_rejects_mismatched_output_shape_without_mutation() -> Result<()> {
    let mut graph = Graph::new();
    let a = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 4, 8])?,
    ));
    let b = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 8, 16])?,
    ));
    let descale_a = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let descale_b = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let scale_output = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 15])?,
    ));
    let absolute_max_output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1])?,
    ));
    let tensor_count = graph.tensors.len();
    let operation_count = graph.operations.len();

    let error = graph
        .matmul_fp8_quantize(
            MatmulFp8QuantizeTensors::new(
                MatmulFp8Inputs {
                    a: a,
                    b: b,
                    descale_a: descale_a,
                    descale_b: descale_b,
                },
                scale_output,
                output,
                absolute_max_output,
            ),
            DataType::F32,
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == output
            && operation == "matmul fp8 quantize output"
            && expected == vec![2, 4, 16]
            && actual == vec![2, 4, 15]
    ));
    assert_eq!(graph.tensors.len(), tensor_count);
    assert_eq!(graph.operations.len(), operation_count);

    Ok(())
}

#[test]
fn test_matmul_fp8_quantize_rejects_unsupported_output_data_type_without_mutation() -> Result<()> {
    let mut graph = Graph::new();
    let a = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 4, 8])?,
    ));
    let b = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 8, 16])?,
    ));
    let descale_a = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let descale_b = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let scale_output = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let output = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 4, 16])?,
    ));
    let absolute_max_output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1])?,
    ));
    let tensor_count = graph.tensors.len();
    let operation_count = graph.operations.len();

    let error = graph
        .matmul_fp8_quantize(
            MatmulFp8QuantizeTensors::new(
                MatmulFp8Inputs {
                    a: a,
                    b: b,
                    descale_a: descale_a,
                    descale_b: descale_b,
                },
                scale_output,
                output,
                absolute_max_output,
            ),
            DataType::F32,
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorDataTypeUnsupported {
            tensor_id,
            operation,
            supported,
            actual,
        } if tensor_id == output
            && operation == "matmul fp8 quantize output"
            && supported == vec![
                DataType::F32,
                DataType::F16,
                DataType::BF16,
                DataType::F8E4M3,
                DataType::F8E5M2,
            ]
            && actual == DataType::I32
    ));
    assert_eq!(graph.tensors.len(), tensor_count);
    assert_eq!(graph.operations.len(), operation_count);

    Ok(())
}

#[test]
fn test_matmul_fp8_quantize_rejects_mismatched_absolute_max_shape_without_mutation() -> Result<()> {
    let mut graph = Graph::new();
    let a = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 4, 8])?,
    ));
    let b = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 8, 16])?,
    ));
    let descale_a = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let descale_b = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let scale_output = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 16])?,
    ));
    let absolute_max_output =
        graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1, 1])?));
    let tensor_count = graph.tensors.len();
    let operation_count = graph.operations.len();

    let error = graph
        .matmul_fp8_quantize(
            MatmulFp8QuantizeTensors::new(
                MatmulFp8Inputs {
                    a: a,
                    b: b,
                    descale_a: descale_a,
                    descale_b: descale_b,
                },
                scale_output,
                output,
                absolute_max_output,
            ),
            DataType::F32,
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == absolute_max_output
            && operation == "matmul fp8 quantize absolute_max output"
            && expected == vec![1, 1, 1]
            && actual == vec![1, 1]
    ));
    assert_eq!(graph.tensors.len(), tensor_count);
    assert_eq!(graph.operations.len(), operation_count);

    Ok(())
}

#[test]
fn test_matmul_fp8_quantize_rejects_mismatched_absolute_max_data_type_without_mutation()
-> Result<()> {
    let mut graph = Graph::new();
    graph.set_io_data_type(DataType::BF16);
    let a = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 4, 8])?,
    ));
    let b = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 8, 16])?,
    ));
    let descale_a = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let descale_b = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let scale_output = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let output = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([2, 4, 16])?,
    ));
    let absolute_max_output = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 1, 1])?,
    ));
    let tensor_count = graph.tensors.len();
    let operation_count = graph.operations.len();

    let error = graph
        .matmul_fp8_quantize(
            MatmulFp8QuantizeTensors::new(
                MatmulFp8Inputs {
                    a: a,
                    b: b,
                    descale_a: descale_a,
                    descale_b: descale_b,
                },
                scale_output,
                output,
                absolute_max_output,
            ),
            DataType::F32,
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorDataTypeMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == absolute_max_output
            && operation == "matmul fp8 quantize absolute_max output"
            && expected == DataType::F32
            && actual == DataType::BF16
    ));
    assert_eq!(graph.tensors.len(), tensor_count);
    assert_eq!(graph.operations.len(), operation_count);

    Ok(())
}

#[test]
fn test_matmul_fp8_op_rejects_mismatched_output_shape_without_mutation() -> Result<()> {
    let mut graph = Graph::new();
    let a = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 4, 8])?,
    ));
    let b = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 8, 16])?,
    ));
    let descale_a = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let descale_b = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let scale_c = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let c = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 15])?,
    ));
    let absolute_max_c = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1])?,
    ));
    let tensor_count = graph.tensors.len();
    let operation_count = graph.operations.len();

    let error = graph
        .matmul_fp8_op(
            MatmulFp8OperationTensors::new(
                MatmulFp8Inputs {
                    a: a,
                    b: b,
                    descale_a: descale_a,
                    descale_b: descale_b,
                },
                scale_c,
                c,
                absolute_max_c,
            ),
            MatmulFp8Config::new(DataType::F32),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == c
            && operation == "matmul fp8 output"
            && expected == vec![2, 4, 16]
            && actual == vec![2, 4, 15]
    ));
    assert_eq!(graph.tensors.len(), tensor_count);
    assert_eq!(graph.operations.len(), operation_count);

    Ok(())
}

#[test]
fn test_matmul_fp8_op_rejects_mismatched_output_data_type_without_mutation() -> Result<()> {
    let mut graph = Graph::new();
    let a = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 4, 8])?,
    ));
    let b = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 8, 16])?,
    ));
    let descale_a = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let descale_b = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let scale_c = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let c = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 16])?,
    ));
    let absolute_max_c = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1])?,
    ));
    let tensor_count = graph.tensors.len();
    let operation_count = graph.operations.len();

    let error = graph
        .matmul_fp8_op(
            MatmulFp8OperationTensors::new(
                MatmulFp8Inputs {
                    a: a,
                    b: b,
                    descale_a: descale_a,
                    descale_b: descale_b,
                },
                scale_c,
                c,
                absolute_max_c,
            ),
            MatmulFp8Config::new(DataType::F32),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorDataTypeMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == c
            && operation == "matmul fp8 output"
            && expected == DataType::F32
            && actual == DataType::F16
    ));
    assert_eq!(graph.tensors.len(), tensor_count);
    assert_eq!(graph.operations.len(), operation_count);

    Ok(())
}

#[test]
fn test_matmul_fp8_op_rejects_mismatched_absolute_max_shape_without_mutation() -> Result<()> {
    let mut graph = Graph::new();
    let a = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 4, 8])?,
    ));
    let b = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 8, 16])?,
    ));
    let descale_a = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let descale_b = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let scale_c = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let c = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 16])?,
    ));
    let absolute_max_c = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1, 1])?));
    let tensor_count = graph.tensors.len();
    let operation_count = graph.operations.len();

    let error = graph
        .matmul_fp8_op(
            MatmulFp8OperationTensors::new(
                MatmulFp8Inputs {
                    a: a,
                    b: b,
                    descale_a: descale_a,
                    descale_b: descale_b,
                },
                scale_c,
                c,
                absolute_max_c,
            ),
            MatmulFp8Config::new(DataType::F32),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == absolute_max_c
            && operation == "matmul fp8 absolute_max output"
            && expected == vec![1, 1, 1]
            && actual == vec![1, 1]
    ));
    assert_eq!(graph.tensors.len(), tensor_count);
    assert_eq!(graph.operations.len(), operation_count);

    Ok(())
}

#[test]
fn test_matmul_fp8_op_rejects_mismatched_absolute_max_data_type_without_mutation() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_io_data_type(DataType::BF16);
    let a = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 4, 8])?,
    ));
    let b = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 8, 16])?,
    ));
    let descale_a = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let descale_b = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let scale_c = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let c = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([2, 4, 16])?,
    ));
    let absolute_max_c = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 1, 1])?,
    ));
    let tensor_count = graph.tensors.len();
    let operation_count = graph.operations.len();

    let error = graph
        .matmul_fp8_op(
            MatmulFp8OperationTensors::new(
                MatmulFp8Inputs {
                    a: a,
                    b: b,
                    descale_a: descale_a,
                    descale_b: descale_b,
                },
                scale_c,
                c,
                absolute_max_c,
            ),
            MatmulFp8Config::new(DataType::F32),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorDataTypeMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == absolute_max_c
            && operation == "matmul fp8 absolute_max output"
            && expected == DataType::F32
            && actual == DataType::BF16
    ));
    assert_eq!(graph.tensors.len(), tensor_count);
    assert_eq!(graph.operations.len(), operation_count);

    Ok(())
}

#[test]
fn test_matmul_fp8_quantize_infer_uses_graph_io_data_type() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_io_data_type(DataType::BF16);
    let a = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 4, 8])?,
    ));
    let b = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 8, 16])?,
    ));
    let descale_a = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let descale_b = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let scale_output = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));

    let outputs = graph.matmul_fp8_quantize_infer(
        MatmulFp8Inputs {
            a: a,
            b: b,
            descale_a: descale_a,
            descale_b: descale_b,
        },
        scale_output,
        DataType::F32,
    )?;

    assert_eq!(
        graph.tensor_config(outputs.output)?.data_type,
        DataType::BF16
    );
    assert_eq!(
        graph.tensor_config(outputs.absolute_max_output)?.data_type,
        DataType::F32
    );

    Ok(())
}

#[test]
fn test_matmul_block_scale_infer_creates_expected_output_shape() -> Result<()> {
    let mut graph = Graph::new();
    let a = graph.tensor(TensorSpec::new(
        DataType::F4E2M1,
        Shape::contiguous([2, 16, 32])?,
    ));
    let b = graph.tensor(TensorSpec::new(
        DataType::F4E2M1,
        Shape::contiguous([2, 32, 24])?,
    ));
    let scale_a = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 16, 8])?,
    ));
    let scale_b = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 8, 24])?,
    ));

    let output = graph.matmul_block_scale_infer(
        BlockScaleMatmulInputs {
            a: a,
            b: b,
            scale_a: scale_a,
            scale_b: scale_b,
        },
        BlockScaleDequantizeConfig::new(DataType::F32, vec![1, 1, 4]),
        BlockScaleDequantizeConfig::new(DataType::F32, vec![1, 4, 1]),
        DataType::F32,
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[2, 16, 24]);
    assert_eq!(graph.tensor_config(output)?.data_type, DataType::F32);

    Ok(())
}

#[test]
fn test_matmul_block_scale_infer_uses_graph_io_data_type() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_io_data_type(DataType::BF16);
    let a = graph.tensor(TensorSpec::new(
        DataType::F4E2M1,
        Shape::contiguous([2, 16, 32])?,
    ));
    let b = graph.tensor(TensorSpec::new(
        DataType::F4E2M1,
        Shape::contiguous([2, 32, 24])?,
    ));
    let scale_a = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 16, 8])?,
    ));
    let scale_b = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 8, 24])?,
    ));

    let output = graph.matmul_block_scale_infer(
        BlockScaleMatmulInputs {
            a: a,
            b: b,
            scale_a: scale_a,
            scale_b: scale_b,
        },
        BlockScaleDequantizeConfig::new(DataType::F32, vec![1, 1, 4]),
        BlockScaleDequantizeConfig::new(DataType::F32, vec![1, 4, 1]),
        DataType::F32,
    )?;

    assert_eq!(graph.tensor_config(output)?.data_type, DataType::BF16);

    Ok(())
}

#[test]
fn test_matmul_block_scale_infer_error_removes_generated_intermediates() -> Result<()> {
    let mut graph = Graph::new();
    let a = graph.tensor(TensorSpec::new(
        DataType::F4E2M1,
        Shape::contiguous([2, 16, 32])?,
    ));
    let b = graph.tensor(TensorSpec::new(
        DataType::F4E2M1,
        Shape::contiguous([2, 32, 24])?,
    ));
    let scale_a = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 16, 8])?,
    ));
    let bad_scale_b = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 7, 24])?,
    ));
    let tensor_count = graph.tensors.len();

    let error = graph
        .matmul_block_scale_infer(
            BlockScaleMatmulInputs {
                a: a,
                b: b,
                scale_a: scale_a,
                scale_b: bad_scale_b,
            },
            BlockScaleDequantizeConfig::new(DataType::F32, vec![1, 1, 4]),
            BlockScaleDequantizeConfig::new(DataType::F32, vec![1, 4, 1]),
            DataType::F32,
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == bad_scale_b
            && operation == "block scale dequantize scale shape"
            && expected == vec![2, 8, 24]
            && actual == vec![2, 7, 24]
    ));
    assert_eq!(graph.tensors.len(), tensor_count);
    assert!(graph.operations.is_empty());

    Ok(())
}

#[test]
fn test_matmul_block_scale_quantize_infer_creates_output_and_scale_shapes() -> Result<()> {
    let mut graph = Graph::new();
    let a = graph.tensor(TensorSpec::new(
        DataType::F4E2M1,
        Shape::contiguous([2, 16, 32])?,
    ));
    let b = graph.tensor(TensorSpec::new(
        DataType::F4E2M1,
        Shape::contiguous([2, 32, 24])?,
    ));
    let scale_a = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 16, 8])?,
    ));
    let scale_b = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 8, 24])?,
    ));

    let outputs = graph.matmul_block_scale_quantize_infer(
        BlockScaleMatmulInputs {
            a: a,
            b: b,
            scale_a: scale_a,
            scale_b: scale_b,
        },
        BlockScaleDequantizeConfig::new(DataType::F32, vec![1, 1, 4]),
        BlockScaleDequantizeConfig::new(DataType::F32, vec![1, 4, 1]),
        BlockScaleQuantizeConfig::new(DataType::F32, 4).with_axis(2),
        DataType::F32,
    )?;

    assert_eq!(graph.shape(outputs.output)?.dimensions(), &[2, 16, 24]);
    assert_eq!(graph.shape(outputs.scale)?.dimensions(), &[2, 16, 6]);
    assert!(graph.tensor_config(outputs.output)?.is_virtual);
    assert!(graph.tensor_config(outputs.scale)?.is_virtual);
    assert_eq!(
        graph.tensor_config(outputs.output)?.data_type,
        DataType::F32
    );
    assert_eq!(graph.tensor_config(outputs.scale)?.data_type, DataType::F32);

    Ok(())
}

#[test]
fn test_matmul_block_scale_quantize_infer_error_removes_generated_intermediates() -> Result<()> {
    let mut graph = Graph::new();
    let a = graph.tensor(TensorSpec::new(
        DataType::F4E2M1,
        Shape::contiguous([2, 16, 32])?,
    ));
    let b = graph.tensor(TensorSpec::new(
        DataType::F4E2M1,
        Shape::contiguous([2, 32, 24])?,
    ));
    let scale_a = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 16, 8])?,
    ));
    let scale_b = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 8, 24])?,
    ));
    let tensor_count = graph.tensors.len();

    let error = graph
        .matmul_block_scale_quantize_infer(
            BlockScaleMatmulInputs {
                a: a,
                b: b,
                scale_a: scale_a,
                scale_b: scale_b,
            },
            BlockScaleDequantizeConfig::new(DataType::F32, vec![1, 1, 4]),
            BlockScaleDequantizeConfig::new(DataType::F32, vec![1, 4, 1]),
            BlockScaleQuantizeConfig::new(DataType::F32, 0).with_axis(2),
            DataType::F32,
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::OutOfRange { .. } | Error::DescriptorMismatch { .. }
    ));
    assert_eq!(graph.tensors.len(), tensor_count);
    assert!(graph.operations.is_empty());

    Ok(())
}

#[test]
fn test_matmul_nvfp4_infer_creates_expected_output_shape() -> Result<()> {
    let mut graph = Graph::new();
    let a = graph.tensor(TensorSpec::new(
        DataType::F4E2M1,
        Shape::contiguous([2, 16, 32])?,
    ));
    let b = graph.tensor(TensorSpec::new(
        DataType::F4E2M1,
        Shape::contiguous([2, 32, 24])?,
    ));
    let scale_a = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 16, 8])?,
    ));
    let scale_b = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 8, 24])?,
    ));

    let output = graph.matmul_nvfp4_infer(
        BlockScaleMatmulInputs {
            a: a,
            b: b,
            scale_a: scale_a,
            scale_b: scale_b,
        },
        4,
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[2, 16, 24]);
    assert_eq!(graph.tensor_config(output)?.data_type, DataType::F32);

    Ok(())
}

#[test]
fn test_matmul_nvfp4_quantize_infer_creates_output_and_scale_shapes() -> Result<()> {
    let mut graph = Graph::new();
    let a = graph.tensor(TensorSpec::new(
        DataType::F4E2M1,
        Shape::contiguous([2, 16, 32])?,
    ));
    let b = graph.tensor(TensorSpec::new(
        DataType::F4E2M1,
        Shape::contiguous([2, 32, 24])?,
    ));
    let scale_a = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 16, 8])?,
    ));
    let scale_b = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 8, 24])?,
    ));

    let outputs = graph.matmul_nvfp4_quantize_infer(
        BlockScaleMatmulInputs {
            a: a,
            b: b,
            scale_a: scale_a,
            scale_b: scale_b,
        },
        4,
    )?;

    assert_eq!(graph.shape(outputs.output)?.dimensions(), &[2, 16, 24]);
    assert_eq!(graph.shape(outputs.scale)?.dimensions(), &[2, 16, 6]);
    assert!(graph.tensor_config(outputs.output)?.is_virtual);
    assert!(graph.tensor_config(outputs.scale)?.is_virtual);
    assert_eq!(
        graph.tensor_config(outputs.output)?.data_type,
        DataType::F32
    );
    assert_eq!(graph.tensor_config(outputs.scale)?.data_type, DataType::F32);

    Ok(())
}

#[test]
fn test_matmul_nvfp4_upstream_like_graph_reaches_compile() -> Result<()> {
    use crate::{context::Context, frontend::operation::HeuristicMode};
    use singe_cuda::context::Context as CudaContext;

    let cuda = match CudaContext::create() {
        Ok(cuda) => cuda,
        Err(_) => return Ok(()),
    };
    let context = match Context::create(&cuda) {
        Ok(ctx) => ctx,
        Err(_) => return Ok(()),
    };
    let stream = match cuda.create_stream() {
        Ok(stream) => stream,
        Err(_) => return Ok(()),
    };
    context.set_stream(Some(&stream))?;

    let mut graph = Graph::new();
    let batch = 1;
    let m = 128;
    let n = 128;
    let k = 64;
    let block_size = 16;
    let block_scale_dim_m = 128;
    let block_scale_dim_n = 128;
    let block_scale_dim_k = 4;

    let a = graph.tensor(TensorSpec::new(
        DataType::F4E2M1,
        Shape::contiguous([batch, m, k])?.with_strides([m * k, k, 1])?,
    ));
    let b = graph.tensor(TensorSpec::new(
        DataType::F4E2M1,
        Shape::contiguous([batch, k, n])?.with_strides([k * n, 1, k])?,
    ));
    let scale_a =
        graph.tensor(TensorSpec::block_scale_tensor(
            DataType::F8E4M3,
            Shape::contiguous([batch, block_scale_dim_m, block_scale_dim_k])?.with_strides(
                vec![block_scale_dim_m * block_scale_dim_k, block_scale_dim_k, 1],
            )?,
        ));
    let scale_b =
        graph.tensor(TensorSpec::block_scale_tensor(
            DataType::F8E4M3,
            Shape::contiguous([batch, block_scale_dim_k, block_scale_dim_n])?.with_strides(
                vec![block_scale_dim_n * block_scale_dim_k, 1, block_scale_dim_k],
            )?,
        ));

    graph.set_io_data_type(DataType::F16);
    let _ = graph.matmul_nvfp4_infer(
        BlockScaleMatmulInputs {
            a: a,
            b: b,
            scale_a: scale_a,
            scale_b: scale_b,
        },
        block_size,
    )?;

    match graph.compile(&context, &[HeuristicMode::A]) {
        Ok(_) => Ok(()),
        Err(error) if is_expected_compile_status(&error) => Ok(()),
        Err(error) => Err(error),
    }
}

#[test]
fn test_matmul_nvfp4_swiglu_upstream_like_graph_reaches_support() -> Result<()> {
    use crate::{
        context::Context,
        frontend::operation::{BlockScaleDequantizeConfig, HeuristicMode, PointwiseOperation},
        math::NanPropagation,
        pointwise::PointwiseMode,
    };
    use singe_cuda::context::Context as CudaContext;

    let cuda = match CudaContext::create() {
        Ok(cuda) => cuda,
        Err(_) => return Ok(()),
    };
    let context = match Context::create(&cuda) {
        Ok(ctx) => ctx,
        Err(_) => return Ok(()),
    };
    let stream = match cuda.create_stream() {
        Ok(stream) => stream,
        Err(_) => return Ok(()),
    };
    context.set_stream(Some(&stream))?;

    let mut graph = Graph::new();
    let b = 1;
    let m = 256;
    let n = 256;
    let k = 256;
    let block_size = 16;
    let indestructible_128x4_block_m_n = 128;
    let indestructible_128x4_block_k = 4;
    let block_scale_dim_m = ((m + indestructible_128x4_block_m_n - 1)
        / indestructible_128x4_block_m_n)
        * indestructible_128x4_block_m_n;
    let block_scale_dim_n = ((n + indestructible_128x4_block_m_n - 1)
        / indestructible_128x4_block_m_n)
        * indestructible_128x4_block_m_n;
    let block_scale_dim_k = (((k + block_size - 1) / block_size + indestructible_128x4_block_k
        - 1)
        / indestructible_128x4_block_k)
        * indestructible_128x4_block_k;

    graph.set_intermediate_data_type(DataType::F32);
    graph.set_compute_data_type(DataType::F32);

    let mut tensor_a = TensorSpec::new(
        DataType::F4E2M1,
        Shape::contiguous([b, m, k])?.with_strides([m * k, k, 1])?,
    );
    tensor_a.set_name("tensor_a");
    let tensor_a = graph.tensor(tensor_a);

    let mut tensor_b0 = TensorSpec::new(
        DataType::F4E2M1,
        Shape::contiguous([b, k, n])?.with_strides([k * n, 1, k])?,
    );
    tensor_b0.set_name("tensor_b");
    let tensor_b0 = graph.tensor(tensor_b0);

    let mut tensor_b1 = TensorSpec::new(
        DataType::F4E2M1,
        Shape::contiguous([b, k, n])?.with_strides([k * n, 1, k])?,
    );
    tensor_b1.set_name("tensor_b");
    let tensor_b1 = graph.tensor(tensor_b1);

    let mut block_descale_a = TensorSpec::block_scale_tensor(
        DataType::F8E4M3,
        Shape::contiguous([b, block_scale_dim_m, block_scale_dim_k])?.with_strides(vec![
            block_scale_dim_m * block_scale_dim_k,
            block_scale_dim_k,
            1,
        ])?,
    );
    block_descale_a.set_name("block_descale_a");
    let block_descale_a = graph.tensor(block_descale_a);

    let mut block_descale_b0 = TensorSpec::block_scale_tensor(
        DataType::F8E4M3,
        Shape::contiguous([b, block_scale_dim_k, block_scale_dim_n])?.with_strides(vec![
            block_scale_dim_m * block_scale_dim_k,
            1,
            block_scale_dim_k,
        ])?,
    );
    block_descale_b0.set_name("block_descale_b");
    let block_descale_b0 = graph.tensor(block_descale_b0);

    let mut block_descale_b1 = TensorSpec::block_scale_tensor(
        DataType::F8E4M3,
        Shape::contiguous([b, block_scale_dim_k, block_scale_dim_n])?.with_strides(vec![
            block_scale_dim_m * block_scale_dim_k,
            1,
            block_scale_dim_k,
        ])?,
    );
    block_descale_b1.set_name("block_descale_b");
    let block_descale_b1 = graph.tensor(block_descale_b1);

    let dequan_tensor_a = graph.block_scale_dequantize_infer(
        tensor_a,
        block_descale_a,
        BlockScaleDequantizeConfig::without_compute_type(vec![1, block_size as i32]),
    )?;
    let dequan_tensor_b0 = graph.block_scale_dequantize_infer(
        tensor_b0,
        block_descale_b0,
        BlockScaleDequantizeConfig::without_compute_type(vec![block_size as i32, 1]),
    )?;
    let dequan_tensor_b1 = graph.block_scale_dequantize_infer(
        tensor_b1,
        block_descale_b1,
        BlockScaleDequantizeConfig::without_compute_type(vec![block_size as i32, 1]),
    )?;

    let tensor_c0 = graph.matmul_infer_with_config(
        dequan_tensor_a,
        dequan_tensor_b0,
        MatmulConfig::new(DataType::F32),
    )?;
    let tensor_c1 = graph.matmul_infer_with_config(
        dequan_tensor_a,
        dequan_tensor_b1,
        MatmulConfig::new(DataType::F32),
    )?;

    let mut tensor_after_swish = TensorSpec::new(
        DataType::F32,
        Shape::contiguous([b, m, n])?.with_strides([m * n, n, 1])?,
    )
    .virtual_tensor();
    tensor_after_swish.set_name("swish::OUT_0");
    let tensor_after_swish = graph.tensor(tensor_after_swish);
    graph.pointwise(PointwiseOperation::Unary {
        mode: PointwiseMode::SwishFwd,
        input: tensor_c0,
        output: tensor_after_swish,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        axis: None,
    });

    let mut tensor_d = TensorSpec::new(
        DataType::F4E2M1,
        Shape::contiguous([b, m, n])?.with_strides([m * n, n, 1])?,
    );
    tensor_d.set_name("mul::OUT_0");
    let tensor_d = graph.tensor(tensor_d);
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::Mul,
        lhs: tensor_after_swish,
        rhs: tensor_c1,
        output: tensor_d,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });

    match graph.compile_with_config(
        &context,
        &CompileConfig::new().with_heuristic_modes(vec![HeuristicMode::A]),
    ) {
        Ok(_) => Ok(()),
        Err(error) if is_expected_compile_status(&error) => Ok(()),
        Err(error) => Err(error),
    }
}

#[test]
fn test_layer_norm_infer_creates_output_and_stats_shapes() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);

    let outputs =
        graph.layer_normalization_infer(input, LayerNormalizationConfig::training(epsilon))?;
    let output = outputs.output;
    let mean = outputs.mean;
    let inv_variance = outputs.inv_variance;
    let scale = outputs.scale;
    let bias = outputs.bias;

    assert_eq!(graph.shape(output)?.dimensions(), &[8, 16, 32]);
    assert_eq!(graph.shape(mean)?.dimensions(), &[8, 16, 1]);
    assert_eq!(graph.shape(inv_variance)?.dimensions(), &[8, 16, 1]);
    assert_eq!(graph.shape(scale)?.dimensions(), &[1, 1, 32]);
    assert_eq!(graph.shape(bias)?.dimensions(), &[1, 1, 32]);

    Ok(())
}

#[test]
fn test_layer_norm_rejects_non_scalar_epsilon() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 1, 32])?,
    ));
    let bias = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 1, 32])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32])?,
    ));
    let mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([8, 16, 1])?,
    ));
    let inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([8, 16, 1])?,
    ));
    let epsilon = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2])?));

    let err = graph
        .layer_normalization(
            input,
            scale,
            bias,
            output,
            mean,
            inv_variance,
            LayerNormalizationConfig::inference(epsilon),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorElementCountMismatch {
            tensor_id,
            operation,
            expected: 1,
            actual: 2,
        } if tensor_id == epsilon && operation == "layer norm epsilon"
    ));

    Ok(())
}

#[test]
fn test_layer_norm_rejects_mismatched_output_shape() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 1, 32])?,
    ));
    let bias = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 1, 32])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 31])?,
    ));
    let mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([8, 16, 1])?,
    ));
    let inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([8, 16, 1])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);

    let err = graph
        .layer_normalization(
            input,
            scale,
            bias,
            output,
            mean,
            inv_variance,
            LayerNormalizationConfig::inference(epsilon),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == output
            && operation == "layer norm output shape"
            && expected == vec![8, 16, 32]
            && actual == vec![8, 16, 31]
    ));

    Ok(())
}

#[test]
fn test_layer_norm_infer_error_removes_generated_outputs() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32])?,
    ));
    let epsilon = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2])?));
    let tensor_count = graph.tensors.len();

    let error = graph
        .layer_normalization_infer(input, LayerNormalizationConfig::training(epsilon))
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorElementCountMismatch {
            tensor_id,
            operation,
            expected: 1,
            actual: 2,
        } if tensor_id == epsilon && operation == "layer norm epsilon"
    ));
    assert_eq!(graph.tensors.len(), tensor_count);
    assert!(graph.operations.is_empty());

    Ok(())
}

#[test]
fn test_layer_norm_graph_compiles_when_plan_is_available() -> Result<()> {
    let context = setup_context()?;

    let mut graph = Graph::new();
    let input = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([4 * 1024, 128, 1, 1])?.with_strides([128, 1, 128, 128])?,
        )
        .with_id(41),
    );
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?.with_id(42));
    let outputs =
        graph.layer_normalization_infer(input, LayerNormalizationConfig::training(epsilon))?;
    let output = outputs.output;

    match graph.compile(&context, &[HeuristicMode::A, HeuristicMode::Fallback]) {
        Ok(compiled) => {
            assert_eq!(compiled.tensor_record(input)?.id, 41.into());
            assert_eq!(compiled.tensor_record(epsilon)?.id, 42.into());
            let _ = compiled.tensor_record(output)?;
        }
        Err(Error::NoAvailableEngines) => {}
        Err(error) if is_expected_compile_status(&error) => {}
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_rms_norm_infer_creates_output_and_stats_shapes() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);

    let outputs =
        graph.rms_normalization_infer(input, RmsNormalizationConfig::training(epsilon))?;
    let output = outputs.output;
    let inv_variance = outputs.inv_variance;
    let scale = outputs.scale;

    assert_eq!(graph.shape(output)?.dimensions(), &[8, 16, 32]);
    assert_eq!(graph.shape(inv_variance)?.dimensions(), &[8, 16, 1]);
    assert_eq!(graph.shape(scale)?.dimensions(), &[1, 1, 32]);

    Ok(())
}

#[test]
fn test_rms_norm_rejects_mismatched_optional_bias_shape() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 1, 32])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32])?,
    ));
    let inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([8, 16, 1])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);
    let bias = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 8, 32])?,
    ));

    let err = graph
        .rms_normalization(
            input,
            scale,
            output,
            inv_variance,
            RmsNormalizationConfig::inference(epsilon).with_bias(bias),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == bias
            && operation == "rms norm bias shape"
            && expected == vec![1, 1, 32]
            && actual == vec![1, 8, 32]
    ));

    Ok(())
}

#[test]
fn test_rms_norm_graph_compiles_when_plan_is_available() -> Result<()> {
    let context = setup_context()?;

    let mut graph = Graph::new();
    let input = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([4 * 1024, 128, 1, 1])?.with_strides([128, 1, 128, 128])?,
        )
        .with_id(51),
    );
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?.with_id(52));
    let outputs =
        graph.rms_normalization_infer(input, RmsNormalizationConfig::training(epsilon))?;
    let output = outputs.output;

    match graph.compile(&context, &[HeuristicMode::A, HeuristicMode::Fallback]) {
        Ok(compiled) => {
            assert_eq!(compiled.tensor_record(input)?.id, 51.into());
            assert_eq!(compiled.tensor_record(epsilon)?.id, 52.into());
            let _ = compiled.tensor_record(output)?;
        }
        Err(Error::NoAvailableEngines) => {}
        Err(error) if is_expected_compile_status(&error) => {}
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_layer_norm_backward_infer_creates_dx_dscale_bias_gradient_shapes() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32])?,
    ));
    let dy = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 1, 32])?,
    ));
    let mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([8, 16, 1])?,
    ));
    let inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([8, 16, 1])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);

    let outputs = graph.layer_normalization_backward_infer(
        input,
        mean,
        inv_variance,
        dy,
        scale,
        LayerNormalizationBackwardConfig::new(epsilon),
    )?;
    let dx = outputs.dx;
    let dscale = outputs.dscale;
    let bias_gradient = outputs.bias_gradient;

    assert_eq!(graph.shape(dx)?.dimensions(), &[8, 16, 32]);
    assert_eq!(graph.shape(dscale)?.dimensions(), &[1, 1, 32]);
    assert_eq!(graph.shape(bias_gradient)?.dimensions(), &[1, 1, 32]);

    Ok(())
}

#[test]
fn test_rms_norm_backward_infer_can_skip_bias_gradient() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32])?,
    ));
    let dy = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 1, 32])?,
    ));
    let inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([8, 16, 1])?,
    ));
    let outputs = graph.rms_normalization_backward_infer(
        input,
        inv_variance,
        dy,
        scale,
        RmsNormalizationBackwardConfig::new(),
    )?;
    let dx = outputs.dx;
    let dscale = outputs.dscale;
    let bias_gradient = outputs.bias_gradient;

    assert_eq!(graph.shape(dx)?.dimensions(), &[8, 16, 32]);
    assert_eq!(graph.shape(dscale)?.dimensions(), &[1, 1, 32]);
    assert!(bias_gradient.is_none());

    Ok(())
}

#[test]
fn test_rms_norm_backward_infer_error_removes_generated_outputs() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32])?,
    ));
    let dy = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 31])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 1, 32])?,
    ));
    let inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([8, 16, 1])?,
    ));
    let tensor_count = graph.tensors.len();

    let error = graph
        .rms_normalization_backward_infer(
            input,
            inv_variance,
            dy,
            scale,
            RmsNormalizationBackwardConfig::new().with_bias_gradient(),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == dy
            && operation == "rms norm backward dy shape"
            && expected == vec![8, 16, 32]
            && actual == vec![8, 16, 31]
    ));
    assert_eq!(graph.tensors.len(), tensor_count);
    assert!(graph.operations.is_empty());

    Ok(())
}

#[test]
fn test_layer_norm_backward_graph_compiles_when_plan_is_available() -> Result<()> {
    let context = setup_context()?;

    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4 * 1024, 128, 1, 1])?.with_strides([128, 1, 128, 128])?,
    ));
    let dy = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4 * 1024, 128, 1, 1])?.with_strides([128, 1, 128, 128])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 128, 1, 1])?.with_strides([128, 1, 128, 128])?,
    ));
    let mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([4 * 1024, 1, 1, 1])?.with_strides([1, 1, 1, 1])?,
    ));
    let inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([4 * 1024, 1, 1, 1])?.with_strides([1, 1, 1, 1])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);

    let _outputs = graph.layer_normalization_backward_infer(
        input,
        mean,
        inv_variance,
        dy,
        scale,
        LayerNormalizationBackwardConfig::new(epsilon),
    )?;

    match graph.compile(&context, &[HeuristicMode::A, HeuristicMode::Fallback]) {
        Ok(compiled) => {
            let _ = compiled.workspace_size()?;
        }
        Err(Error::NoAvailableEngines) => {}
        Err(error) if is_expected_compile_status(&error) => {}
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_rms_norm_backward_graph_compiles_when_plan_is_available() -> Result<()> {
    let context = match setup_context() {
        Ok(ctx) => ctx,
        Err(Error::Cuda(singe_cuda::error::Error::Cuda { code, .. }))
            if code == singe_cuda::error::Status::NoDevice =>
        {
            return Ok(());
        }
        Err(error) => return Err(error),
    };

    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([4 * 1024, 128, 1, 1])?.with_strides([128, 1, 128, 128])?,
    ));
    let dy = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([4 * 1024, 128, 1, 1])?.with_strides([128, 1, 128, 128])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 128, 1, 1])?.with_strides([128, 1, 128, 128])?,
    ));
    let inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([4 * 1024, 1, 1, 1])?.with_strides([1, 1, 1, 1])?,
    ));
    let _outputs = graph.rms_normalization_backward_infer(
        input,
        inv_variance,
        dy,
        scale,
        RmsNormalizationBackwardConfig::new().with_bias_gradient(),
    )?;

    match graph.compile(&context, &[HeuristicMode::A, HeuristicMode::Fallback]) {
        Ok(compiled) => {
            let _ = compiled.workspace_size()?;
        }
        Err(Error::NoAvailableEngines) => {}
        Err(error) if is_expected_compile_status(&error) => {}
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_instance_norm_infer_creates_output_and_stats_shapes() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32, 32])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);

    let outputs = graph
        .instance_normalization_infer(input, InstanceNormalizationConfig::training(epsilon))?;
    let output = outputs.output;
    let mean = outputs.mean;
    let inv_variance = outputs.inv_variance;
    let scale = outputs.scale;
    let bias = outputs.bias;

    assert_eq!(graph.shape(output)?.dimensions(), &[8, 16, 32, 32]);
    assert_eq!(graph.shape(mean)?.dimensions(), &[8, 16, 1, 1]);
    assert_eq!(graph.shape(inv_variance)?.dimensions(), &[8, 16, 1, 1]);
    assert_eq!(graph.shape(scale)?.dimensions(), &[1, 16, 1, 1]);
    assert_eq!(graph.shape(bias)?.dimensions(), &[1, 16, 1, 1]);

    Ok(())
}

#[test]
fn test_instance_norm_rejects_mismatched_scale_shape() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32, 32])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 15, 1, 1])?,
    ));
    let bias = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 16, 1, 1])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32, 32])?,
    ));
    let mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([8, 16, 1, 1])?,
    ));
    let inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([8, 16, 1, 1])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);

    let err = graph
        .instance_normalization(
            input,
            scale,
            bias,
            output,
            mean,
            inv_variance,
            InstanceNormalizationConfig::training(epsilon),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == scale
            && operation == "instance norm scale shape"
            && expected == vec![1, 16, 1, 1]
            && actual == vec![1, 15, 1, 1]
    ));

    Ok(())
}

#[test]
fn test_instance_norm_backward_infer_creates_dx_dscale_bias_gradient_shapes() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32, 32])?,
    ));
    let dy = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32, 32])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 16, 1, 1])?,
    ));
    let mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([8, 16, 1, 1])?,
    ));
    let inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([8, 16, 1, 1])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);

    let outputs = graph.instance_normalization_backward_infer(
        input,
        mean,
        inv_variance,
        dy,
        scale,
        InstanceNormalizationBackwardConfig::new(epsilon),
    )?;
    let dx = outputs.dx;
    let dscale = outputs.dscale;
    let bias_gradient = outputs.bias_gradient;

    assert_eq!(graph.shape(dx)?.dimensions(), &[8, 16, 32, 32]);
    assert_eq!(graph.shape(dscale)?.dimensions(), &[1, 16, 1, 1]);
    assert_eq!(graph.shape(bias_gradient)?.dimensions(), &[1, 16, 1, 1]);

    Ok(())
}

#[test]
fn test_instance_norm_backward_rejects_mismatched_dy_shape() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32, 32])?,
    ));
    let dy = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 31, 32])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 16, 1, 1])?,
    ));
    let mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([8, 16, 1, 1])?,
    ));
    let inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([8, 16, 1, 1])?,
    ));
    let dscale = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 16, 1, 1])?,
    ));
    let bias_gradient = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 16, 1, 1])?,
    ));
    let dx = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32, 32])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);

    let err = graph
        .instance_normalization_backward(
            input,
            mean,
            inv_variance,
            dy,
            scale,
            dscale,
            bias_gradient,
            dx,
            InstanceNormalizationBackwardConfig::new(epsilon),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == dy
            && operation == "instance norm backward dy shape"
            && expected == vec![8, 16, 32, 32]
            && actual == vec![8, 16, 31, 32]
    ));

    Ok(())
}

#[test]
fn test_instance_norm_graph_compiles_when_plan_is_available() -> Result<()> {
    let context = setup_context()?;

    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32, 32])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);
    let _outputs = graph
        .instance_normalization_infer(input, InstanceNormalizationConfig::inference(epsilon))?;

    match graph.compile(&context, &[HeuristicMode::Instant, HeuristicMode::Fallback]) {
        Ok(compiled) => {
            let _ = compiled.workspace_size()?;
        }
        Err(error) if is_expected_compile_status(&error) => {}
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_instance_norm_backward_graph_compiles_when_plan_is_available() -> Result<()> {
    let context = setup_context()?;

    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32, 32])?,
    ));
    let dy = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32, 32])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 16, 1, 1])?,
    ));
    let mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([8, 16, 1, 1])?,
    ));
    let inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([8, 16, 1, 1])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);

    let _outputs = graph.instance_normalization_backward_infer(
        input,
        mean,
        inv_variance,
        dy,
        scale,
        InstanceNormalizationBackwardConfig::new(epsilon),
    )?;

    match graph.compile(&context, &[HeuristicMode::Instant, HeuristicMode::Fallback]) {
        Ok(compiled) => {
            let _ = compiled.workspace_size()?;
        }
        Err(error) if is_expected_compile_status(&error) => {}
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_batch_norm_infer_creates_output_and_channel_stats_shapes() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32, 32])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);

    let outputs = graph.batch_normalization_infer(input, BatchNormalizationConfig::new(epsilon))?;
    let output = outputs.output;
    let mean = outputs.mean;
    let inv_variance = outputs.inv_variance;
    let scale = outputs.scale;
    let bias = outputs.bias;

    assert_eq!(graph.shape(output)?.dimensions(), &[8, 16, 32, 32]);
    assert_eq!(graph.shape(mean)?.dimensions(), &[1, 16, 1, 1]);
    assert_eq!(graph.shape(inv_variance)?.dimensions(), &[1, 16, 1, 1]);
    assert_eq!(graph.shape(scale)?.dimensions(), &[1, 16, 1, 1]);
    assert_eq!(graph.shape(bias)?.dimensions(), &[1, 16, 1, 1]);

    Ok(())
}

#[test]
fn test_batch_norm_infer_error_removes_generated_outputs() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32, 32])?,
    ));
    let epsilon = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2])?));
    let tensor_count = graph.tensors.len();

    let error = graph
        .batch_normalization_infer(input, BatchNormalizationConfig::new(epsilon))
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorElementCountMismatch {
            tensor_id,
            operation,
            expected: 1,
            actual: 2,
        } if tensor_id == epsilon && operation == "batch norm epsilon"
    ));
    assert_eq!(graph.tensors.len(), tensor_count);
    assert!(graph.operations.is_empty());

    Ok(())
}

#[test]
fn test_batch_norm_rejects_mismatched_scale_shape() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32, 32])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 15, 1, 1])?,
    ));
    let bias = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 16, 1, 1])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32, 32])?,
    ));
    let mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 16, 1, 1])?,
    ));
    let inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 16, 1, 1])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);

    let error = graph
        .batch_normalization(
            input,
            scale,
            bias,
            output,
            mean,
            inv_variance,
            BatchNormalizationConfig::new(epsilon),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == scale
            && operation == "batch norm scale shape"
            && expected == vec![1, 16, 1, 1]
            && actual == vec![1, 15, 1, 1]
    ));

    Ok(())
}

#[test]
fn test_batch_norm_rejects_mismatched_output_shape() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32, 32])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 16, 1, 1])?,
    ));
    let bias = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 16, 1, 1])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32, 31])?,
    ));
    let mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 16, 1, 1])?,
    ));
    let inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 16, 1, 1])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);

    let error = graph
        .batch_normalization(
            input,
            scale,
            bias,
            output,
            mean,
            inv_variance,
            BatchNormalizationConfig::new(epsilon),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == output
            && operation == "batch norm output shape"
            && expected == vec![8, 16, 32, 32]
            && actual == vec![8, 16, 32, 31]
    ));

    Ok(())
}

#[test]
fn test_batch_norm_backward_infer_creates_dx_dscale_bias_gradient_shapes() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32, 32])?,
    ));
    let dy = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32, 32])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 16, 1, 1])?,
    ));
    let mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 16, 1, 1])?,
    ));
    let inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 16, 1, 1])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);

    let outputs = graph.batch_normalization_backward_infer(
        input,
        mean,
        inv_variance,
        dy,
        scale,
        BatchNormalizationBackwardConfig::new(epsilon),
    )?;
    let dx = outputs.dx;
    let dscale = outputs.dscale;
    let bias_gradient = outputs.bias_gradient;

    assert_eq!(graph.shape(dx)?.dimensions(), &[8, 16, 32, 32]);
    assert_eq!(graph.shape(dscale)?.dimensions(), &[1, 16, 1, 1]);
    assert_eq!(graph.shape(bias_gradient)?.dimensions(), &[1, 16, 1, 1]);

    Ok(())
}

#[test]
fn test_batch_norm_backward_rejects_mismatched_dy_shape() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32, 32])?,
    ));
    let dy = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32, 31])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 16, 1, 1])?,
    ));
    let mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 16, 1, 1])?,
    ));
    let inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 16, 1, 1])?,
    ));
    let dscale = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 16, 1, 1])?,
    ));
    let bias_gradient = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 16, 1, 1])?,
    ));
    let dx = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32, 32])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);

    let error = graph
        .batch_normalization_backward(
            input,
            mean,
            inv_variance,
            dy,
            scale,
            dscale,
            bias_gradient,
            dx,
            BatchNormalizationBackwardConfig::new(epsilon),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == dy
            && operation == "batch norm backward dy shape"
            && expected == vec![8, 16, 32, 32]
            && actual == vec![8, 16, 32, 31]
    ));

    Ok(())
}

#[test]
fn test_batch_norm_graph_compiles_when_plan_is_available() -> Result<()> {
    let context = setup_context()?;

    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 16])?.with_strides(vec![32 * 16 * 16, 1, 32 * 16, 32])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);
    let _outputs =
        graph.batch_normalization_infer(input, BatchNormalizationConfig::new(epsilon))?;

    match graph.compile(&context, &[HeuristicMode::A, HeuristicMode::Fallback]) {
        Ok(compiled) => {
            let _ = compiled.workspace_size()?;
        }
        Err(Error::NoAvailableEngines) => {}
        Err(error) if is_expected_compile_status(&error) => {}
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_compiled_graph_autotune_rejects_zero_iterations() -> Result<()> {
    let context = setup_context()?;

    let mut graph = Graph::new();
    let x = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([4])?).with_id(11));
    let y = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([4])?).with_id(22));
    graph.pointwise(PointwiseOperation::Unary {
        mode: PointwiseMode::Identity,
        input: x,
        output: y,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        axis: None,
    });

    match graph.compile(&context, &[HeuristicMode::Instant, HeuristicMode::Fallback]) {
        Ok(mut compiled) => {
            let bindings = Bindings::new();
            let err = compiled
                .autotune_rank(
                    &context,
                    &bindings,
                    None,
                    &AutotuneConfig::new().with_iterations(0),
                )
                .unwrap_err();
            assert!(matches!(
                err,
                Error::OutOfRange { name } if name == "iterations"
            ));
            let err = compiled
                .autotune_in_place(
                    &context,
                    &bindings,
                    None,
                    &AutotuneConfig::new().with_iterations(0),
                )
                .unwrap_err();
            assert!(matches!(
                err,
                Error::OutOfRange { name } if name == "iterations"
            ));
        }
        Err(error) if is_expected_compile_status(&error) => {}
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_compiled_graph_autotune_reports_stable_winner_metadata_when_plan_is_available() -> Result<()>
{
    let context = setup_context()?;

    let mut graph = Graph::new();
    let x = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([4])?).with_id(11));
    let y = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([4])?).with_id(22));
    graph.pointwise(PointwiseOperation::Unary {
        mode: PointwiseMode::Identity,
        input: x,
        output: y,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        axis: None,
    });

    match graph.compile(&context, &[HeuristicMode::Instant, HeuristicMode::Fallback]) {
        Ok(mut compiled) => {
            let plan_count = compiled.execution_plans().len();
            let x_dev = DeviceMemory::<f32>::create(4)?;
            let y_dev = DeviceMemory::<f32>::create(4)?;
            let mut bindings = Bindings::new();
            unsafe {
                bindings.set_id(
                    compiled.tensor_record(x)?.id,
                    DevicePtr::from_raw(x_dev.as_ptr().cast_mut().cast()),
                );
                bindings.set_id(
                    compiled.tensor_record(y)?.id,
                    DevicePtr::from_raw(y_dev.as_ptr().cast_mut().cast()),
                );
            }
            let default_plan_index = compiled.default_plan_index();
            let ranked_result = compiled.autotune_rank(
                &context,
                &bindings,
                None,
                &AutotuneConfig::new().with_iterations(1).with_warmup(1),
            );
            match ranked_result {
                Ok(result) => {
                    assert_eq!(compiled.default_plan_index(), default_plan_index);
                    assert!(!result.timings.is_empty());
                    assert_eq!(result.winner_ranked_plan_index, 0);
                    assert!(result.winner_original_plan_index < plan_count);
                    for (ranked_index, timing) in result.timings.iter().enumerate() {
                        assert_eq!(timing.ranked_plan_index, ranked_index);
                        assert!(timing.original_plan_index < plan_count);
                    }
                }
                Err(error) if is_expected_compile_status(&error) => {}
                Err(error) => return Err(error),
            }
            let result = compiled.autotune_in_place(
                &context,
                &bindings,
                None,
                &AutotuneConfig::new().with_iterations(1).with_warmup(1),
            );

            match result {
                Ok(result) => {
                    assert_eq!(
                        compiled.default_plan_index(),
                        result.winner_ranked_plan_index
                    );
                    assert_eq!(result.winner_ranked_plan_index, 0);
                    assert!(!result.timings.is_empty());
                    assert!(result.winner_original_plan_index < plan_count);
                    for (ranked_index, timing) in result.timings.iter().enumerate() {
                        assert_eq!(timing.ranked_plan_index, ranked_index);
                        assert!(timing.original_plan_index < plan_count);
                    }
                }
                Err(error) if is_expected_compile_status(&error) => {}
                Err(error) => return Err(error),
            }
        }
        Err(error) if is_expected_compile_status(&error) => {}
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_batch_norm_backward_graph_compiles_when_plan_is_available() -> Result<()> {
    let context = setup_context()?;

    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 16])?.with_strides(vec![32 * 16 * 16, 1, 32 * 16, 32])?,
    ));
    let dy = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 16])?.with_strides(vec![32 * 16 * 16, 1, 32 * 16, 32])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 32, 1, 1])?.with_strides([32, 1, 32, 32])?,
    ));
    let mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 32, 1, 1])?.with_strides([32, 1, 32, 32])?,
    ));
    let inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 32, 1, 1])?.with_strides([32, 1, 32, 32])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);

    let _outputs = graph.batch_normalization_backward_infer(
        input,
        mean,
        inv_variance,
        dy,
        scale,
        BatchNormalizationBackwardConfig::new(epsilon),
    )?;

    match graph.compile(&context, &[HeuristicMode::A, HeuristicMode::Fallback]) {
        Ok(compiled) => {
            let _ = compiled.workspace_size()?;
        }
        Err(Error::NoAvailableEngines) => {}
        Err(error) if is_expected_compile_status(&error) => {}
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_bn_inference_drelu_batch_norm_backward_executes_when_plan_is_available() -> Result<()> {
    let context = setup_context()?;

    let n = 4;
    let c = 32;
    let h = 16;
    let w = 16;

    let mut graph = Graph::new().with_io_data_type(DataType::F32);
    let x = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([n, c, h, w])?.with_strides([c * h * w, 1, c * w, c])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
    ));
    let bias = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
    ));
    let mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
    ));
    let inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);
    let bn_y = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([n, c, h, w])?.with_strides(vec![c * h * w, 1, c * w, c])?,
        )
        .virtual_tensor(),
    );
    graph.batch_normalization_inference(
        x,
        mean,
        inv_variance,
        scale,
        bias,
        bn_y,
        BatchNormalizationInferenceConfig::new(epsilon),
    )?;

    let dy = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([n, c, h, w])?.with_strides([c * h * w, 1, c * w, c])?,
    ));
    let dx_drelu = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([n, c, h, w])?.with_strides(vec![c * h * w, 1, c * w, c])?,
        )
        .virtual_tensor(),
    );
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::ReluBwd,
        lhs: dy,
        rhs: bn_y,
        output: dx_drelu,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });

    let dx = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([n, c, h, w])?.with_strides([c * h * w, 1, c * w, c])?,
    ));
    let dscale = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
    ));
    let bias_gradient = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
    ));
    graph.batch_normalization_backward(
        x,
        mean,
        inv_variance,
        dx_drelu,
        scale,
        dscale,
        bias_gradient,
        dx,
        BatchNormalizationBackwardConfig::new(epsilon),
    )?;

    let compiled = match graph.compile_with_config(
        &context,
        &CompileConfig::new()
            .with_heuristic_modes(vec![HeuristicMode::A, HeuristicMode::Fallback])
            .with_build_policy(BuildPlanPolicy::AllSupported),
    ) {
        Ok(compiled) => compiled,
        Err(Error::NoAvailableEngines) => return Ok(()),
        Err(error) if is_expected_compile_status(&error) => return Ok(()),
        Err(Error::Cudnn {
            code: Status::BadParam,
            ..
        }) => return Ok(()),
        Err(error) => return Err(error),
    };

    let mut x_dev = DeviceMemory::<f16>::zeroes((n * c * h * w) as usize)?;
    let mut dy_dev = DeviceMemory::<f16>::zeroes((n * c * h * w) as usize)?;
    let mut scale_dev = DeviceMemory::<f32>::zeroes(c as usize)?;
    let mut bias_dev = DeviceMemory::<f32>::zeroes(c as usize)?;
    let mut mean_dev = DeviceMemory::<f32>::zeroes(c as usize)?;
    let mut inv_variance_dev = DeviceMemory::<f32>::zeroes(c as usize)?;
    let mut dx_dev = DeviceMemory::<f16>::zeroes((n * c * h * w) as usize)?;
    let mut dscale_dev = DeviceMemory::<f32>::zeroes(c as usize)?;
    let mut bias_gradient_dev = DeviceMemory::<f32>::zeroes(c as usize)?;
    let mut workspace = DeviceMemory::<u8>::zeroes(compiled.max_workspace_size()?)?;
    let mut bindings = compiled.bindings();
    bindings.set(x, &mut x_dev)?;
    bindings.set(dy, &mut dy_dev)?;
    bindings.set(scale, &mut scale_dev)?;
    bindings.set(bias, &mut bias_dev)?;
    bindings.set(mean, &mut mean_dev)?;
    bindings.set(inv_variance, &mut inv_variance_dev)?;
    bindings.set(dx, &mut dx_dev)?;
    bindings.set(dscale, &mut dscale_dev)?;
    bindings.set(bias_gradient, &mut bias_gradient_dev)?;
    match compiled.execute(&context, &bindings, Some(&mut workspace)) {
        Ok(()) => {}
        Err(Error::NoAvailableEngines) => return Ok(()),
        Err(error) if is_expected_compile_status(&error) => return Ok(()),
        Err(Error::Cudnn {
            code: Status::BadParam,
            ..
        }) => return Ok(()),
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_batch_norm_accepts_running_stats_tensors() -> Result<()> {
    let context = setup_context()?;

    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 16])?.with_strides(vec![32 * 16 * 16, 1, 32 * 16, 32])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 32, 1, 1])?.with_strides([32, 1, 32, 32])?,
    ));
    let bias = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 32, 1, 1])?.with_strides([32, 1, 32, 32])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([4, 32, 16, 16])?.with_strides(vec![32 * 16 * 16, 1, 32 * 16, 32])?,
    ));
    let mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 32, 1, 1])?.with_strides([32, 1, 32, 32])?,
    ));
    let inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 32, 1, 1])?.with_strides([32, 1, 32, 32])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);
    let momentum = graph.tensor(TensorSpec::scalar_f32(0.1)?);
    let prev_mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 32, 1, 1])?.with_strides([32, 1, 32, 32])?,
    ));
    let prev_var = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 32, 1, 1])?.with_strides([32, 1, 32, 32])?,
    ));
    let next_mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 32, 1, 1])?.with_strides([32, 1, 32, 32])?,
    ));
    let next_var = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 32, 1, 1])?.with_strides([32, 1, 32, 32])?,
    ));

    graph.batch_normalization(
        input,
        scale,
        bias,
        output,
        mean,
        inv_variance,
        BatchNormalizationConfig::new(epsilon).with_running_stats(
            BatchNormalizationRunningStats::new(momentum, prev_mean, prev_var, next_mean, next_var),
        ),
    )?;

    match graph.compile(&context, &[HeuristicMode::A, HeuristicMode::Fallback]) {
        Ok(compiled) => {
            let _ = compiled.workspace_size()?;
        }
        Err(Error::NoAvailableEngines) => {}
        Err(Error::Cudnn { code, .. }) if code == Status::InternalErrorUnexpectedValue => {}
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_transpose_infer_records_frontend_node_and_alias_shape() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4])?.with_strides([12, 4, 1])?,
    ));

    let output = graph.transpose_infer(input, [1, 2, 0])?;

    let output_tensor = graph.tensor_config(output)?;
    assert_eq!(output_tensor.shape.dimensions(), &[3, 4, 2]);
    assert_eq!(output_tensor.shape.strides(), &[4, 1, 12]);
    assert!(matches!(
        graph.operations.last(),
        Some(Operation::Transpose {
            input: op_input,
            output: op_output,
            permutation,
        }) if *op_input == input && *op_output == output && permutation == &[1, 2, 0]
    ));

    Ok(())
}

#[test]
fn test_transpose_rejects_mismatched_output_strides() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4])?.with_strides([12, 4, 1])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([3, 4, 2])?,
    ));

    let error = graph.transpose(input, output, [1, 2, 0]).unwrap_err();
    assert!(matches!(
        error,
        Error::FrontendTensorStridesMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == output
            && operation == "transpose output"
            && expected == vec![4, 1, 12]
            && actual == vec![8, 2, 1]
    ));

    Ok(())
}

#[test]
fn test_graph_tracks_optional_data_type_defaults() {
    let mut graph = Graph::new();

    assert_eq!(graph.name(), None);
    assert_eq!(graph.sm_count_target(), None);
    assert_eq!(graph.sm_version(), None);
    assert!(!graph.dynamic_shape_enabled());
    assert!(!graph.override_shape_enabled());
    assert!(!graph.kernel_cache_enabled());
    assert_eq!(graph.io_data_type(), None);
    assert_eq!(graph.intermediate_data_type(), None);
    assert_eq!(graph.compute_data_type(), None);

    graph.set_name("attention");
    graph.set_sm_count_target(42);
    graph.set_sm_version(100);
    graph.enable_dynamic_shape();
    graph.enable_override_shape();
    graph.set_io_data_type(DataType::F8E4M3);
    graph.set_intermediate_data_type(DataType::F32);
    graph.set_compute_data_type(DataType::F32);

    assert_eq!(graph.name(), Some("attention"));
    assert_eq!(graph.sm_count_target(), Some(42));
    assert_eq!(graph.sm_version(), Some(100));
    assert!(graph.dynamic_shape_enabled());
    assert!(graph.override_shape_enabled());
    assert!(!graph.kernel_cache_enabled());
    assert_eq!(graph.io_data_type(), Some(DataType::F8E4M3));
    assert_eq!(graph.intermediate_data_type(), Some(DataType::F32));
    assert_eq!(graph.compute_data_type(), Some(DataType::F32));

    graph.clear_name();
    graph.clear_sm_count_target();
    graph.clear_sm_version();
    graph.disable_dynamic_shape();
    graph.disable_override_shape();
    graph.clear_io_data_type();
    graph.clear_intermediate_data_type();
    graph.clear_compute_data_type();

    assert_eq!(graph.name(), None);
    assert_eq!(graph.sm_count_target(), None);
    assert_eq!(graph.sm_version(), None);
    assert!(!graph.dynamic_shape_enabled());
    assert!(!graph.override_shape_enabled());
    assert!(!graph.kernel_cache_enabled());
    assert_eq!(graph.io_data_type(), None);
    assert_eq!(graph.intermediate_data_type(), None);
    assert_eq!(graph.compute_data_type(), None);
}

#[test]
fn test_batch_norm_accepts_peer_stats() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 16])?.with_strides(vec![32 * 16 * 16, 1, 32 * 16, 32])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 32, 1, 1])?.with_strides([32, 1, 32, 32])?,
    ));
    let bias = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 32, 1, 1])?.with_strides([32, 1, 32, 32])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([4, 32, 16, 16])?.with_strides(vec![32 * 16 * 16, 1, 32 * 16, 32])?,
    ));
    let mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 32, 1, 1])?.with_strides([32, 1, 32, 32])?,
    ));
    let inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 32, 1, 1])?.with_strides([32, 1, 32, 32])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);
    let peer_stat = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4 * 32, 1, 1])?.with_strides(vec![4 * 32, 1, 4 * 32, 4 * 32])?,
    ));

    graph.batch_normalization(
        input,
        scale,
        bias,
        output,
        mean,
        inv_variance,
        BatchNormalizationConfig::new(epsilon).with_peer_stats(vec![peer_stat]),
    )?;

    Ok(())
}

#[test]
fn test_batch_norm_backward_rejects_mismatched_peer_stat_shape() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 16])?.with_strides(vec![32 * 16 * 16, 1, 32 * 16, 32])?,
    ));
    let dy = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 16])?.with_strides(vec![32 * 16 * 16, 1, 32 * 16, 32])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 32, 1, 1])?.with_strides([32, 1, 32, 32])?,
    ));
    let mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 32, 1, 1])?.with_strides([32, 1, 32, 32])?,
    ));
    let inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 32, 1, 1])?.with_strides([32, 1, 32, 32])?,
    ));
    let dscale = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 32, 1, 1])?.with_strides([32, 1, 32, 32])?,
    ));
    let bias_gradient = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 32, 1, 1])?.with_strides([32, 1, 32, 32])?,
    ));
    let dx = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 16])?.with_strides(vec![32 * 16 * 16, 1, 32 * 16, 32])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);
    let peer_stat = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 127, 1, 1])?.with_strides([127, 1, 127, 127])?,
    ));

    let err = graph
        .batch_normalization_backward(
            input,
            mean,
            inv_variance,
            dy,
            scale,
            dscale,
            bias_gradient,
            dx,
            BatchNormalizationBackwardConfig::new(epsilon).with_peer_stats(vec![peer_stat]),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "batch norm backward peer_stat shape"
    ));

    Ok(())
}

#[test]
fn test_batch_norm_accepts_sm_carveout_peer_stat_shape() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 32, 16, 16])?.with_strides(vec![32 * 16 * 16, 1, 32 * 16, 32])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 32, 1, 1])?.with_strides([32, 1, 32, 32])?,
    ));
    let bias = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 32, 1, 1])?.with_strides([32, 1, 32, 32])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 32, 16, 16])?.with_strides(vec![32 * 16 * 16, 1, 32 * 16, 32])?,
    ));
    let mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 32, 1, 1])?.with_strides([32, 1, 32, 32])?,
    ));
    let inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 32, 1, 1])?.with_strides([32, 1, 32, 32])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);
    let peer_stat = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4 * 32, 1, 1])?.with_strides(vec![4 * 32, 1, 4 * 32, 4 * 32])?,
    ));

    graph.batch_normalization(
        input,
        scale,
        bias,
        output,
        mean,
        inv_variance,
        BatchNormalizationConfig::new(epsilon).with_peer_stats(vec![peer_stat]),
    )?;

    Ok(())
}

#[test]
fn test_adaptive_layer_norm_infer_creates_output_and_stats_shapes() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);

    let outputs = graph.adaptive_layer_normalization_infer(
        input,
        AdaptiveLayerNormalizationConfig::training(epsilon),
    )?;
    let output = outputs.output;
    let mean = outputs.mean;
    let inv_variance = outputs.inv_variance;
    let scale = outputs.scale;

    assert_eq!(graph.shape(output)?.dimensions(), &[8, 16, 32]);
    assert_eq!(graph.shape(mean)?.dimensions(), &[8, 16, 1]);
    assert_eq!(graph.shape(inv_variance)?.dimensions(), &[8, 16, 1]);
    assert_eq!(graph.shape(scale)?.dimensions(), &[8, 1, 32]);

    Ok(())
}

#[test]
fn test_adaptive_layer_norm_rejects_mismatched_scale_shape() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 2, 32])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32])?,
    ));
    let mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([8, 16, 1])?,
    ));
    let inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([8, 16, 1])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);

    let error = graph
        .adaptive_layer_normalization(
            input,
            scale,
            output,
            mean,
            inv_variance,
            AdaptiveLayerNormalizationConfig::training(epsilon),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == scale
            && operation == "adaptive layer norm scale shape"
            && expected == vec![8, 1, 32]
            && actual == vec![8, 2, 32]
    ));

    Ok(())
}

#[test]
fn test_normalization_infer_uses_graph_intermediate_data_type_for_stats() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_intermediate_data_type(DataType::BF16);

    let vector_input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32])?,
    ));
    let image_input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32, 32])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);

    let layer_outputs = graph
        .layer_normalization_infer(vector_input, LayerNormalizationConfig::training(epsilon))?;
    let layer_mean = layer_outputs.mean;
    let layer_inv_variance = layer_outputs.inv_variance;
    let rms_outputs =
        graph.rms_normalization_infer(vector_input, RmsNormalizationConfig::training(epsilon))?;
    let rms_inv_variance = rms_outputs.inv_variance;
    let instance_outputs = graph.instance_normalization_infer(
        image_input,
        InstanceNormalizationConfig::training(epsilon),
    )?;
    let instance_mean = instance_outputs.mean;
    let instance_inv_variance = instance_outputs.inv_variance;
    let batch_outputs =
        graph.batch_normalization_infer(image_input, BatchNormalizationConfig::new(epsilon))?;
    let batch_mean = batch_outputs.mean;
    let batch_inv_variance = batch_outputs.inv_variance;
    let adaptive_outputs = graph.adaptive_layer_normalization_infer(
        vector_input,
        AdaptiveLayerNormalizationConfig::training(epsilon),
    )?;
    let adaptive_mean = adaptive_outputs.mean;
    let adaptive_inv_variance = adaptive_outputs.inv_variance;

    for tensor in [
        layer_mean,
        layer_inv_variance,
        rms_inv_variance,
        instance_mean,
        instance_inv_variance,
        batch_mean,
        batch_inv_variance,
        adaptive_mean,
        adaptive_inv_variance,
    ] {
        assert_eq!(graph.tensor_config(tensor)?.data_type, DataType::BF16);
    }

    Ok(())
}

#[test]
fn test_normalization_infer_uses_graph_io_data_type_for_non_virtual_tensors() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_io_data_type(DataType::BF16);

    let vector_input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32])?,
    ));
    let image_input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32, 32])?,
    ));
    let batch_channel = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 16, 1, 1])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);

    let layer_outputs = graph
        .layer_normalization_infer(vector_input, LayerNormalizationConfig::training(epsilon))?;
    let layer_output = layer_outputs.output;
    let layer_scale = layer_outputs.scale;
    let layer_bias = layer_outputs.bias;
    let rms_outputs =
        graph.rms_normalization_infer(vector_input, RmsNormalizationConfig::training(epsilon))?;
    let rms_output = rms_outputs.output;
    let rms_scale = rms_outputs.scale;
    let instance_outputs = graph.instance_normalization_infer(
        image_input,
        InstanceNormalizationConfig::training(epsilon),
    )?;
    let instance_output = instance_outputs.output;
    let instance_scale = instance_outputs.scale;
    let instance_bias = instance_outputs.bias;
    let batch_outputs =
        graph.batch_normalization_infer(image_input, BatchNormalizationConfig::new(epsilon))?;
    let batch_output = batch_outputs.output;
    let batch_scale = batch_outputs.scale;
    let batch_bias = batch_outputs.bias;
    let adaptive_outputs = graph.adaptive_layer_normalization_infer(
        vector_input,
        AdaptiveLayerNormalizationConfig::training(epsilon),
    )?;
    let adaptive_output = adaptive_outputs.output;
    let adaptive_scale = adaptive_outputs.scale;
    let batch_inference_output = graph.batch_normalization_inference_infer(
        image_input,
        batch_channel,
        batch_channel,
        batch_channel,
        batch_channel,
        BatchNormalizationInferenceConfig::new(epsilon),
    )?;

    for tensor in [
        layer_output,
        layer_scale,
        layer_bias,
        rms_output,
        rms_scale,
        instance_output,
        instance_scale,
        instance_bias,
        batch_output,
        batch_scale,
        batch_bias,
        adaptive_output,
        adaptive_scale,
        batch_inference_output,
    ] {
        assert_eq!(graph.tensor_config(tensor)?.data_type, DataType::BF16);
    }

    Ok(())
}

#[test]
fn test_normalization_backward_infer_uses_graph_io_data_type_for_outputs() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_io_data_type(DataType::BF16);

    let vector_input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32])?,
    ));
    let vector_dy = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32])?,
    ));
    let vector_scale = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 1, 32])?,
    ));
    let layer_mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([8, 16, 1])?,
    ));
    let layer_inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([8, 16, 1])?,
    ));

    let image_input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32, 32])?,
    ));
    let image_dy = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32, 32])?,
    ));
    let image_scale = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 16, 1, 1])?,
    ));
    let instance_mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([8, 16, 1, 1])?,
    ));
    let instance_inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([8, 16, 1, 1])?,
    ));
    let batch_mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 16, 1, 1])?,
    ));
    let batch_inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 16, 1, 1])?,
    ));

    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);

    let layer_backward_outputs = graph.layer_normalization_backward_infer(
        vector_input,
        layer_mean,
        layer_inv_variance,
        vector_dy,
        vector_scale,
        LayerNormalizationBackwardConfig::new(epsilon),
    )?;
    let layer_dx = layer_backward_outputs.dx;
    let layer_dscale = layer_backward_outputs.dscale;
    let layer_bias_gradient = layer_backward_outputs.bias_gradient;
    let rms_backward_outputs = graph.rms_normalization_backward_infer(
        vector_input,
        layer_inv_variance,
        vector_dy,
        vector_scale,
        RmsNormalizationBackwardConfig::new().with_bias_gradient(),
    )?;
    let rms_dx = rms_backward_outputs.dx;
    let rms_dscale = rms_backward_outputs.dscale;
    let rms_bias_gradient = rms_backward_outputs.bias_gradient;
    let instance_backward_outputs = graph.instance_normalization_backward_infer(
        image_input,
        instance_mean,
        instance_inv_variance,
        image_dy,
        image_scale,
        InstanceNormalizationBackwardConfig::new(epsilon),
    )?;
    let instance_dx = instance_backward_outputs.dx;
    let instance_dscale = instance_backward_outputs.dscale;
    let instance_bias_gradient = instance_backward_outputs.bias_gradient;
    let batch_backward_outputs = graph.batch_normalization_backward_infer(
        image_input,
        batch_mean,
        batch_inv_variance,
        image_dy,
        image_scale,
        BatchNormalizationBackwardConfig::new(epsilon),
    )?;
    let batch_dx = batch_backward_outputs.dx;
    let batch_dscale = batch_backward_outputs.dscale;
    let batch_bias_gradient = batch_backward_outputs.bias_gradient;
    let adaptive_mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([8, 16, 1])?,
    ));
    let adaptive_inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([8, 16, 1])?,
    ));
    let adaptive_scale = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 1, 32])?,
    ));
    let adaptive_backward_outputs = graph.adaptive_layer_normalization_backward_infer(
        vector_input,
        adaptive_mean,
        adaptive_inv_variance,
        vector_dy,
        adaptive_scale,
        AdaptiveLayerNormalizationBackwardConfig::new().with_bias_gradient(),
    )?;
    let adaptive_dx = adaptive_backward_outputs.dx;
    let adaptive_dscale = adaptive_backward_outputs.dscale;
    let adaptive_bias_gradient = adaptive_backward_outputs.bias_gradient;

    for tensor in [
        layer_dx,
        layer_dscale,
        layer_bias_gradient,
        rms_dx,
        rms_dscale,
        rms_bias_gradient.expect("rms bias_gradient"),
        instance_dx,
        instance_dscale,
        instance_bias_gradient,
        batch_dx,
        batch_dscale,
        batch_bias_gradient,
        adaptive_dx,
        adaptive_dscale,
        adaptive_bias_gradient.expect("adaptive bias_gradient"),
    ] {
        let tensor_ref = graph.tensor_config(tensor)?;
        assert_eq!(tensor_ref.data_type, DataType::BF16);
        assert!(!tensor_ref.is_virtual);
    }

    Ok(())
}

#[test]
fn test_adaptive_layer_norm_backward_infer_can_skip_bias_gradient() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32])?,
    ));
    let dy = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 1, 32])?,
    ));
    let mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([8, 16, 1])?,
    ));
    let inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([8, 16, 1])?,
    ));

    let outputs = graph.adaptive_layer_normalization_backward_infer(
        input,
        mean,
        inv_variance,
        dy,
        scale,
        AdaptiveLayerNormalizationBackwardConfig::new(),
    )?;

    assert_eq!(graph.shape(outputs.dx)?.dimensions(), &[8, 16, 32]);
    assert_eq!(graph.shape(outputs.dscale)?.dimensions(), &[8, 1, 32]);
    assert!(outputs.bias_gradient.is_none());

    Ok(())
}

#[test]
fn test_adaptive_layer_norm_backward_rejects_mismatched_dy_shape() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32])?,
    ));
    let dy = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 31])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 1, 32])?,
    ));
    let mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([8, 16, 1])?,
    ));
    let inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([8, 16, 1])?,
    ));
    let dscale = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 1, 32])?,
    ));
    let dx = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 16, 32])?,
    ));

    let error = graph
        .adaptive_layer_normalization_backward(
            input,
            mean,
            inv_variance,
            dy,
            scale,
            dscale,
            None,
            dx,
            AdaptiveLayerNormalizationBackwardConfig::new(),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == dy
            && operation == "adaptive layer norm backward dy shape"
            && expected == vec![8, 16, 32]
            && actual == vec![8, 16, 31]
    ));

    Ok(())
}

#[test]
fn test_adaptive_layer_norm_graph_compiles_when_plan_is_available() -> Result<()> {
    let context = setup_context()?;

    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([4, 1024, 128])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);
    let _outputs = graph.adaptive_layer_normalization_infer(
        input,
        AdaptiveLayerNormalizationConfig::training(epsilon),
    )?;

    match graph.compile(&context, &[HeuristicMode::A, HeuristicMode::Fallback]) {
        Ok(compiled) => {
            let _ = compiled.workspace_size()?;
        }
        Err(Error::NoAvailableEngines) => {}
        Err(error) if is_expected_compile_status(&error) => {}
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_adaptive_layer_norm_backward_graph_compiles_when_plan_is_available() -> Result<()> {
    let context = match setup_context() {
        Ok(ctx) => ctx,
        Err(Error::Cuda(singe_cuda::error::Error::Cuda { code, .. }))
            if code == singe_cuda::error::Status::NoDevice =>
        {
            return Ok(());
        }
        Err(error) => return Err(error),
    };

    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 1024, 128])?,
    ));
    let dy = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 1024, 128])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 1, 128])?,
    ));
    let mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([4, 1024, 1])?,
    ));
    let inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([4, 1024, 1])?,
    ));

    let _outputs = graph.adaptive_layer_normalization_backward_infer(
        input,
        mean,
        inv_variance,
        dy,
        scale,
        AdaptiveLayerNormalizationBackwardConfig::new().with_bias_gradient(),
    )?;

    match graph.compile(&context, &[HeuristicMode::A, HeuristicMode::Fallback]) {
        Ok(compiled) => {
            let _ = compiled.workspace_size()?;
        }
        Err(error) if is_expected_compile_status(&error) => {}
        Err(error) => return Err(error),
    }

    Ok(())
}
