use super::*;
use crate::{
    attribute::{BackendAttributeName, BackendAttributeType},
    frontend::lower::LoweredOperation,
};

#[test]
fn test_moe_grouped_matmul_infer_creates_expected_output_shape() -> Result<()> {
    let mut graph = Graph::new();
    let token = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 8, 16])?,
    ));
    let weight = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 16, 32])?,
    ));
    let first_token_offset = graph.tensor(TensorSpec::new(DataType::I32, Shape::contiguous([5])?));

    let output = graph.moe_grouped_matmul_infer(
        MoeGroupedMatmulInputs::new(token, weight, first_token_offset),
        MoeGroupedMatmulConfig::new(MoeGroupedMatmulMode::None, DataType::F32),
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[1, 8, 32]);

    Ok(())
}

#[test]
fn test_moe_grouped_matmul_gather_uses_token_index_shape() -> Result<()> {
    let mut graph = Graph::new();
    let token = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 8, 16])?,
    ));
    let weight = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 16, 32])?,
    ));
    let first_token_offset = graph.tensor(TensorSpec::new(DataType::I32, Shape::contiguous([5])?));
    let token_index = graph.tensor(TensorSpec::new(DataType::I32, Shape::contiguous([1, 6])?));

    let output = graph.moe_grouped_matmul_infer(
        MoeGroupedMatmulInputs::new(token, weight, first_token_offset),
        MoeGroupedMatmulConfig::new(MoeGroupedMatmulMode::Gather, DataType::F32)
            .with_token_index(token_index),
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[1, 6, 32]);

    Ok(())
}

#[test]
fn test_moe_grouped_matmul_scatter_infer_uses_token_shape() -> Result<()> {
    let mut graph = Graph::new();
    let token = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 8, 16])?,
    ));
    let weight = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 16, 32])?,
    ));
    let first_token_offset = graph.tensor(TensorSpec::new(DataType::I32, Shape::contiguous([5])?));
    let token_index = graph.tensor(TensorSpec::new(DataType::I32, Shape::contiguous([1, 6])?));
    let token_ks = graph.tensor(TensorSpec::new(DataType::I32, Shape::contiguous([1, 6])?));

    let output = graph.moe_grouped_matmul_infer(
        MoeGroupedMatmulInputs::new(token, weight, first_token_offset),
        MoeGroupedMatmulConfig::new(MoeGroupedMatmulMode::Scatter, DataType::F32)
            .with_token_index(token_index)
            .with_token_ks(token_ks)
            .with_top_k(2),
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[1, 8, 32]);

    Ok(())
}

#[test]
fn test_moe_grouped_matmul_infer_error_does_not_insert_output_tensor() -> Result<()> {
    let mut graph = Graph::new();
    let token = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 8, 16])?,
    ));
    let weight = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 16, 32])?,
    ));
    let first_token_offset = graph.tensor(TensorSpec::new(DataType::I32, Shape::contiguous([5])?));
    let token_index = graph.tensor(TensorSpec::new(DataType::I32, Shape::contiguous([1, 6])?));
    let tensor_count = graph.tensors.len();

    let error = graph
        .moe_grouped_matmul_infer(
            MoeGroupedMatmulInputs::new(token, weight, first_token_offset),
            MoeGroupedMatmulConfig::new(MoeGroupedMatmulMode::Scatter, DataType::F32)
                .with_token_index(token_index)
                .with_top_k(2),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "moe grouped matmul token ks"
    ));
    assert_eq!(graph.tensors.len(), tensor_count);
    assert!(graph.operations.is_empty());

    Ok(())
}

#[test]
fn test_moe_grouped_matmul_rejects_mismatched_token_rank() -> Result<()> {
    let mut graph = Graph::new();
    let token = graph.tensor(TensorSpec::new(DataType::F16, Shape::contiguous([8, 16])?));
    let weight = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 16, 32])?,
    ));
    let first_token_offset = graph.tensor(TensorSpec::new(DataType::I32, Shape::contiguous([5])?));
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 8, 32])?,
    ));

    let error = graph
        .moe_grouped_matmul(
            MoeGroupedMatmulTensors::new(
                MoeGroupedMatmulInputs::new(token, weight, first_token_offset),
                output,
            ),
            MoeGroupedMatmulConfig::new(MoeGroupedMatmulMode::None, DataType::F32),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        Error::FrontendTensorRankMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == token
            && operation == "moe grouped matmul token rank"
            && expected == "3"
            && actual == 2
    ));

    Ok(())
}

#[test]
fn test_moe_grouped_matmul_rejects_short_token_index_rank() -> Result<()> {
    let mut graph = Graph::new();
    let token = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 8, 16])?,
    ));
    let weight = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 16, 32])?,
    ));
    let first_token_offset = graph.tensor(TensorSpec::new(DataType::I32, Shape::contiguous([5])?));
    let token_index = graph.tensor(TensorSpec::new(DataType::I32, Shape::contiguous([8])?));
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 8, 32])?,
    ));

    let error = graph
        .moe_grouped_matmul(
            MoeGroupedMatmulTensors::new(
                MoeGroupedMatmulInputs::new(token, weight, first_token_offset),
                output,
            ),
            MoeGroupedMatmulConfig::new(MoeGroupedMatmulMode::Gather, DataType::F32)
                .with_token_index(token_index),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        Error::FrontendTensorRankMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == token_index
            && operation == "moe grouped matmul token index rank"
            && expected == ">= 2"
            && actual == 1
    ));

    Ok(())
}

#[test]
fn test_moe_grouped_matmul_support_surface_rejects_before_cudnn_915() {
    let graph = Graph::new();
    let error = graph
        .validate_moe_grouped_matmul_support_surface_for_version(91499)
        .unwrap_err();
    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "moe grouped matmul version"
    ));
}

#[test]
fn test_moe_grouped_matmul_backward_support_surface_rejects_before_cudnn_922() {
    let graph = Graph::new();
    let error = graph
        .validate_moe_grouped_matmul_backward_support_surface_for_version(92199)
        .unwrap_err();
    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "moe grouped matmul backward version"
    ));
    graph
        .validate_moe_grouped_matmul_backward_support_surface_for_version(92200)
        .unwrap();
}

#[test]
fn test_moe_grouped_matmul_backward_infer_creates_expected_weight_gradient_shape() -> Result<()> {
    let mut graph = Graph::new();
    let output_gradient = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 8, 32])?,
    ));
    let token = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 8, 16])?,
    ));
    let first_token_offset = graph.tensor(TensorSpec::new(DataType::I32, Shape::contiguous([4])?));

    let weight_gradient = graph.moe_grouped_matmul_backward_infer_for_version(
        92200,
        MoeGroupedMatmulBackwardInputs::new(output_gradient, token, first_token_offset),
        MoeGroupedMatmulBackwardConfig::new(DataType::F32),
    )?;

    assert_eq!(graph.shape(weight_gradient)?.dimensions(), &[4, 16, 32]);
    assert!(matches!(
        graph.operations.last(),
        Some(Operation::MoeGroupedMatmulBackward {
            output_gradient: actual_output_gradient,
            token: actual_token,
            first_token_offset: actual_first_token_offset,
            weight_gradient: actual_weight_gradient,
            config,
        }) if *actual_output_gradient == output_gradient
            && *actual_token == token
            && *actual_first_token_offset == first_token_offset
            && *actual_weight_gradient == weight_gradient
            && config.compute_type() == DataType::F32
    ));

    Ok(())
}

#[test]
fn test_moe_grouped_matmul_backward_rejects_wrong_weight_gradient_shape() -> Result<()> {
    let mut graph = Graph::new();
    let output_gradient = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 8, 32])?,
    ));
    let token = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 8, 16])?,
    ));
    let first_token_offset = graph.tensor(TensorSpec::new(DataType::I32, Shape::contiguous([4])?));
    let weight_gradient = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 16, 31])?,
    ));

    let error = graph
        .moe_grouped_matmul_backward_for_version(
            92200,
            MoeGroupedMatmulBackwardTensors::new(
                MoeGroupedMatmulBackwardInputs::new(output_gradient, token, first_token_offset),
                weight_gradient,
            ),
            MoeGroupedMatmulBackwardConfig::new(DataType::F32),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == weight_gradient
            && operation == "moe grouped matmul backward weight gradient shape"
            && expected == vec![4, 16, 32]
            && actual == vec![4, 16, 31]
    ));

    Ok(())
}

#[test]
fn test_moe_grouped_matmul_backward_infer_error_does_not_insert_output() -> Result<()> {
    let mut graph = Graph::new();
    let output_gradient = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 8, 32])?,
    ));
    let token = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 8, 16])?,
    ));
    let first_token_offset = graph.tensor(TensorSpec::new(DataType::I32, Shape::contiguous([4])?));
    let tensor_count = graph.tensors.len();

    let error = graph
        .moe_grouped_matmul_backward_infer_for_version(
            92200,
            MoeGroupedMatmulBackwardInputs::new(output_gradient, token, first_token_offset),
            MoeGroupedMatmulBackwardConfig::new(DataType::F32),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        Error::FrontendTensorDataTypeMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == output_gradient
            && operation == "moe grouped matmul backward output gradient"
            && expected == DataType::F16
            && actual == DataType::F32
    ));
    assert_eq!(graph.tensors.len(), tensor_count);
    assert!(graph.operations.is_empty());

    Ok(())
}

#[test]
fn test_moe_grouped_matmul_rejects_mismatched_weight_data_type() -> Result<()> {
    let mut graph = Graph::new();
    let token = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 8, 16])?,
    ));
    let weight = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([4, 16, 32])?,
    ));
    let first_token_offset = graph.tensor(TensorSpec::new(DataType::I32, Shape::contiguous([5])?));
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 8, 32])?,
    ));

    let error = graph
        .moe_grouped_matmul(
            MoeGroupedMatmulTensors::new(
                MoeGroupedMatmulInputs::new(token, weight, first_token_offset),
                output,
            ),
            MoeGroupedMatmulConfig::new(MoeGroupedMatmulMode::None, DataType::F32),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        Error::FrontendTensorDataTypeMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == weight
            && operation == "moe grouped matmul weight data type"
            && expected == DataType::F16
            && actual == DataType::F32
    ));

    Ok(())
}

#[test]
fn test_moe_grouped_matmul_rejects_wrong_first_token_offset_data_type() -> Result<()> {
    let mut graph = Graph::new();
    let token = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 8, 16])?,
    ));
    let weight = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 16, 32])?,
    ));
    let first_token_offset = graph.tensor(TensorSpec::new(DataType::I64, Shape::contiguous([5])?));
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 8, 32])?,
    ));

    let error = graph
        .moe_grouped_matmul(
            MoeGroupedMatmulTensors::new(
                MoeGroupedMatmulInputs::new(token, weight, first_token_offset),
                output,
            ),
            MoeGroupedMatmulConfig::new(MoeGroupedMatmulMode::None, DataType::F32),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        Error::FrontendTensorDataTypeMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == first_token_offset
            && operation == "moe grouped matmul first token offset data type"
            && expected == DataType::I32
            && actual == DataType::I64
    ));

    Ok(())
}

#[test]
fn test_moe_grouped_matmul_rejects_wrong_token_index_data_type() -> Result<()> {
    let mut graph = Graph::new();
    let token = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 8, 16])?,
    ));
    let weight = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 16, 32])?,
    ));
    let first_token_offset = graph.tensor(TensorSpec::new(DataType::I32, Shape::contiguous([5])?));
    let token_index = graph.tensor(TensorSpec::new(DataType::I64, Shape::contiguous([1, 6])?));
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 6, 32])?,
    ));

    let error = graph
        .moe_grouped_matmul(
            MoeGroupedMatmulTensors::new(
                MoeGroupedMatmulInputs::new(token, weight, first_token_offset),
                output,
            ),
            MoeGroupedMatmulConfig::new(MoeGroupedMatmulMode::Gather, DataType::F32)
                .with_token_index(token_index),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        Error::FrontendTensorDataTypeMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == token_index
            && operation == "moe grouped matmul token index data type"
            && expected == DataType::I32
            && actual == DataType::I64
    ));

    Ok(())
}

#[test]
fn test_moe_grouped_matmul_rejects_wrong_token_ks_data_type() -> Result<()> {
    let mut graph = Graph::new();
    let token = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 8, 16])?,
    ));
    let weight = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 16, 32])?,
    ));
    let first_token_offset = graph.tensor(TensorSpec::new(DataType::I32, Shape::contiguous([5])?));
    let token_index = graph.tensor(TensorSpec::new(DataType::I32, Shape::contiguous([1, 6])?));
    let token_ks = graph.tensor(TensorSpec::new(DataType::I64, Shape::contiguous([1, 6])?));
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 8, 32])?,
    ));

    let error = graph
        .moe_grouped_matmul(
            MoeGroupedMatmulTensors::new(
                MoeGroupedMatmulInputs::new(token, weight, first_token_offset),
                output,
            ),
            MoeGroupedMatmulConfig::new(MoeGroupedMatmulMode::Scatter, DataType::F32)
                .with_token_index(token_index)
                .with_token_ks(token_ks)
                .with_top_k(2),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        Error::FrontendTensorDataTypeMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == token_ks
            && operation == "moe grouped matmul token ks data type"
            && expected == DataType::I32
            && actual == DataType::I64
    ));

    Ok(())
}

#[test]
fn test_moe_grouped_matmul_rejects_gather_without_token_index() -> Result<()> {
    let mut graph = Graph::new();
    let token = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 8, 16])?,
    ));
    let weight = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 16, 32])?,
    ));
    let first_token_offset = graph.tensor(TensorSpec::new(DataType::I32, Shape::contiguous([5])?));
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 8, 32])?,
    ));

    let error = graph
        .moe_grouped_matmul(
            MoeGroupedMatmulTensors::new(
                MoeGroupedMatmulInputs::new(token, weight, first_token_offset),
                output,
            ),
            MoeGroupedMatmulConfig::new(MoeGroupedMatmulMode::Gather, DataType::F32),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "moe grouped matmul token index"
    ));

    Ok(())
}

#[test]
fn test_moe_grouped_matmul_rejects_scatter_without_token_ks() -> Result<()> {
    let mut graph = Graph::new();
    let token = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 8, 16])?,
    ));
    let weight = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 16, 32])?,
    ));
    let first_token_offset = graph.tensor(TensorSpec::new(DataType::I32, Shape::contiguous([5])?));
    let token_index = graph.tensor(TensorSpec::new(DataType::I32, Shape::contiguous([1, 8])?));
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 8, 32])?,
    ));

    let error = graph
        .moe_grouped_matmul(
            MoeGroupedMatmulTensors::new(
                MoeGroupedMatmulInputs::new(token, weight, first_token_offset),
                output,
            ),
            MoeGroupedMatmulConfig::new(MoeGroupedMatmulMode::Scatter, DataType::F32)
                .with_token_index(token_index)
                .with_top_k(2),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "moe grouped matmul token ks"
    ));

    Ok(())
}

#[test]
fn test_moe_grouped_matmul_rejects_scatter_without_top_k() -> Result<()> {
    let mut graph = Graph::new();
    let token = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 8, 16])?,
    ));
    let weight = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 16, 32])?,
    ));
    let first_token_offset = graph.tensor(TensorSpec::new(DataType::I32, Shape::contiguous([5])?));
    let token_index = graph.tensor(TensorSpec::new(DataType::I32, Shape::contiguous([1, 8])?));
    let token_ks = graph.tensor(TensorSpec::new(DataType::I32, Shape::contiguous([1, 8])?));
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 8, 32])?,
    ));

    let error = graph
        .moe_grouped_matmul(
            MoeGroupedMatmulTensors::new(
                MoeGroupedMatmulInputs::new(token, weight, first_token_offset),
                output,
            ),
            MoeGroupedMatmulConfig::new(MoeGroupedMatmulMode::Scatter, DataType::F32)
                .with_token_index(token_index)
                .with_token_ks(token_ks),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "moe grouped matmul top_k"
    ));

    Ok(())
}

#[test]
fn test_moe_grouped_matmul_rejects_mismatched_output_shape() -> Result<()> {
    let mut graph = Graph::new();
    let token = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 8, 16])?,
    ));
    let weight = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 16, 32])?,
    ));
    let first_token_offset = graph.tensor(TensorSpec::new(DataType::I32, Shape::contiguous([5])?));
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 8, 31])?,
    ));

    let error = graph
        .moe_grouped_matmul(
            MoeGroupedMatmulTensors::new(
                MoeGroupedMatmulInputs::new(token, weight, first_token_offset),
                output,
            ),
            MoeGroupedMatmulConfig::new(MoeGroupedMatmulMode::None, DataType::F32),
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
            && operation == "moe grouped matmul output shape"
            && expected == vec![1, 8, 32]
            && actual == vec![1, 8, 31]
    ));

    Ok(())
}

#[test]
fn test_moe_grouped_matmul_rejects_mismatched_output_data_type() -> Result<()> {
    let mut graph = Graph::new();
    let token = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 8, 16])?,
    ));
    let weight = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 16, 32])?,
    ));
    let first_token_offset = graph.tensor(TensorSpec::new(DataType::I32, Shape::contiguous([5])?));
    let output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 8, 32])?,
    ));

    let error = graph
        .moe_grouped_matmul(
            MoeGroupedMatmulTensors::new(
                MoeGroupedMatmulInputs::new(token, weight, first_token_offset),
                output,
            ),
            MoeGroupedMatmulConfig::new(MoeGroupedMatmulMode::None, DataType::F32),
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
            && operation == "moe grouped matmul output"
            && expected == DataType::F16
            && actual == DataType::F32
    ));

    Ok(())
}

#[test]
fn test_convolution_infer_builds_expected_shapes() -> Result<()> {
    let mut graph = Graph::new();
    let x = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8, 8])?,
    ));
    let w = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([4, 3, 3, 3])?,
    ));
    let config = ConvolutionConfig::new(DataType::F32, 2)
        .with_mode(ConvolutionMode::CrossCorrelation)
        .with_padding(vec![1, 1]);

    let y = graph.convolution_forward_infer(x, w, config.clone())?;
    let dx = graph.convolution_dgrad_infer(w, y, config.clone())?;
    let dw = graph.convolution_wgrad_infer(x, y, config)?;

    assert_eq!(graph.shape(y)?.dimensions(), &[2, 4, 8, 8]);
    assert_eq!(graph.shape(y)?.strides(), &[256, 1, 32, 4]);
    assert_eq!(graph.shape(dx)?.dimensions(), &[2, 3, 8, 8]);
    assert_eq!(graph.shape(dx)?.strides(), &[192, 1, 24, 3]);
    assert_eq!(graph.shape(dw)?.dimensions(), &[4, 3, 3, 3]);
    assert_eq!(graph.shape(dw)?.strides(), &[27, 1, 9, 3]);

    Ok(())
}

#[test]
fn test_convolution_infer_uses_graph_io_data_type() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_io_data_type(DataType::BF16);
    let x = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8, 8])?,
    ));
    let w = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([4, 3, 3, 3])?,
    ));
    let config = ConvolutionConfig::new(DataType::F32, 2)
        .with_mode(ConvolutionMode::CrossCorrelation)
        .with_padding(vec![1, 1]);

    let y = graph.convolution_forward_infer(x, w, config.clone())?;
    let dx = graph.convolution_dgrad_infer(w, y, config.clone())?;
    let dw = graph.convolution_wgrad_infer(x, y, config)?;

    assert_eq!(graph.tensor_config(y)?.data_type, DataType::BF16);
    assert_eq!(graph.tensor_config(dx)?.data_type, DataType::BF16);
    assert_eq!(graph.tensor_config(dw)?.data_type, DataType::BF16);

    Ok(())
}

#[test]
fn test_convolution_forward_rejects_mismatched_output_shape() -> Result<()> {
    let mut graph = Graph::new();
    let x = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8, 8])?,
    ));
    let w = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([4, 3, 3, 3])?,
    ));
    let y = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 7, 8])?,
    ));

    let error = graph
        .convolution_forward(
            x,
            w,
            y,
            ConvolutionConfig::new(DataType::F32, 2).with_padding([1, 1]),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == y
            && operation == "convolution output shape"
            && expected == vec![2, 4, 8, 8]
            && actual == vec![2, 4, 7, 8]
    ));

    Ok(())
}

#[test]
fn test_convolution_forward_rejects_mismatched_output_data_type() -> Result<()> {
    let mut graph = Graph::new();
    let x = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8, 8])?,
    ));
    let w = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([4, 3, 3, 3])?,
    ));
    let y = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 8])?,
    ));

    let error = graph
        .convolution_forward(
            x,
            w,
            y,
            ConvolutionConfig::new(DataType::F32, 2).with_padding([1, 1]),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorDataTypeMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == y
            && operation == "convolution output"
            && expected == DataType::F32
            && actual == DataType::F16
    ));

    Ok(())
}

#[test]
fn test_convolution_forward_rejects_mismatched_filter_rank() -> Result<()> {
    let mut graph = Graph::new();
    let x = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8, 8])?,
    ));
    let w = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([4, 3, 3])?,
    ));
    let y = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 8])?,
    ));

    let error = graph
        .convolution_forward(
            x,
            w,
            y,
            ConvolutionConfig::new(DataType::F32, 2).with_padding([1, 1]),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorRankMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == w
            && operation == "convolution filter rank"
            && expected == "4"
            && actual == 3
    ));

    Ok(())
}

#[test]
fn test_convolution_forward_rejects_non_4d_input_rank() -> Result<()> {
    let mut graph = Graph::new();
    let x = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8])?,
    ));
    let w = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([4, 3, 3, 3])?,
    ));

    let error = graph
        .convolution_forward_infer(
            x,
            w,
            ConvolutionConfig::new(DataType::F32, 2).with_padding([1, 1]),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorRankMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == x
            && operation == "convolution input rank"
            && expected == "4"
            && actual == 3
    ));

    Ok(())
}

#[test]
fn test_convolution_dgrad_rejects_mismatched_output_shape() -> Result<()> {
    let mut graph = Graph::new();
    let w = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([4, 3, 3, 3])?,
    ));
    let dy = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 8])?,
    ));
    let dx = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 7, 8])?,
    ));

    let error = graph
        .convolution_dgrad(
            w,
            dy,
            dx,
            ConvolutionConfig::new(DataType::F32, 2).with_padding([1, 1]),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == dx
            && operation == "convolution backward data output shape"
            && expected == vec![2, 3, 8, 8]
            && actual == vec![2, 3, 7, 8]
    ));

    Ok(())
}

#[test]
fn test_convolution_dgrad_rejects_non_4d_output_gradient_rank() -> Result<()> {
    let mut graph = Graph::new();
    let w = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([4, 3, 3, 3])?,
    ));
    let dy = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8])?,
    ));

    let error = graph
        .convolution_dgrad_infer(
            w,
            dy,
            ConvolutionConfig::new(DataType::F32, 2).with_padding([1, 1]),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorRankMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == dy
            && operation == "convolution output-gradient rank"
            && expected == "4"
            && actual == 3
    ));

    Ok(())
}

#[test]
fn test_convolution_dgrad_rejects_mismatched_output_data_type() -> Result<()> {
    let mut graph = Graph::new();
    let w = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([4, 3, 3, 3])?,
    ));
    let dy = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 8])?,
    ));
    let dx = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([2, 3, 8, 8])?,
    ));

    let error = graph
        .convolution_dgrad(
            w,
            dy,
            dx,
            ConvolutionConfig::new(DataType::F32, 2).with_padding([1, 1]),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorDataTypeMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == dx
            && operation == "convolution backward data output"
            && expected == DataType::F32
            && actual == DataType::BF16
    ));

    Ok(())
}

#[test]
fn test_convolution_wgrad_rejects_mismatched_output_shape() -> Result<()> {
    let mut graph = Graph::new();
    let x = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8, 8])?,
    ));
    let dy = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 8])?,
    ));
    let dw = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([4, 3, 3, 2])?,
    ));

    let error = graph
        .convolution_wgrad(
            x,
            dy,
            dw,
            ConvolutionConfig::new(DataType::F32, 2).with_padding([1, 1]),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == dw
            && operation == "convolution backward filter output shape"
            && expected == vec![4, 3, 3, 3]
            && actual == vec![4, 3, 3, 2]
    ));

    Ok(())
}

#[test]
fn test_convolution_wgrad_rejects_non_4d_input_rank() -> Result<()> {
    let mut graph = Graph::new();
    let x = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8])?,
    ));
    let dy = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 8])?,
    ));

    let error = graph
        .convolution_wgrad_infer(
            x,
            dy,
            ConvolutionConfig::new(DataType::F32, 2).with_padding([1, 1]),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorRankMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == x
            && operation == "convolution input rank"
            && expected == "4"
            && actual == 3
    ));

    Ok(())
}

#[test]
fn test_convolution_wgrad_rejects_mismatched_output_data_type() -> Result<()> {
    let mut graph = Graph::new();
    let x = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8, 8])?,
    ));
    let dy = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 8])?,
    ));
    let dw = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 3, 3, 3])?,
    ));

    let error = graph
        .convolution_wgrad(
            x,
            dy,
            dw,
            ConvolutionConfig::new(DataType::F32, 2).with_padding([1, 1]),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorDataTypeMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == dw
            && operation == "convolution backward filter output"
            && expected == DataType::F32
            && actual == DataType::F16
    ));

    Ok(())
}

#[test]
fn test_convolution_graph_compiles_when_plan_is_available() -> Result<()> {
    let context = setup_context()?;

    let mut graph = Graph::new();
    let x = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8, 8])?,
    ));
    let w = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([4, 3, 3, 3])?,
    ));
    let _y = graph.convolution_forward_infer(
        x,
        w,
        ConvolutionConfig::new(DataType::F32, 2).with_padding(vec![1, 1]),
    )?;

    match graph.compile(&context, &[HeuristicMode::Instant, HeuristicMode::Fallback]) {
        Ok(compiled) => {
            assert!(!compiled.execution_plans().is_empty());
            let _ = compiled.workspace_size_at(0)?;
        }
        Err(error) if is_expected_compile_status(&error) => {}
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_resample_backward_infer_builds_input_gradient_shape() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8, 8])?,
    ));
    let output = graph.resample_infer(
        input,
        None,
        ResampleConfig::new(ResampleMode::MaxPool, DataType::F32)
            .with_window_dims([2, 2])
            .with_strides([2, 2]),
    )?;
    let dy = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 4])?,
    ));

    let dx = graph.resample_backward_infer(
        input,
        output,
        dy,
        None,
        ResampleConfig::new(ResampleMode::MaxPool, DataType::F32)
            .with_window_dims([2, 2])
            .with_strides([2, 2]),
    )?;

    assert_eq!(graph.shape(dx)?.dimensions(), &[2, 3, 8, 8]);

    Ok(())
}

#[test]
fn test_resample_backward_infer_uses_graph_io_data_type_for_input_gradient() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_io_data_type(DataType::F16);
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8, 8])?,
    ));
    let output = graph.resample_infer(
        input,
        None,
        ResampleConfig::new(ResampleMode::MaxPool, DataType::F32)
            .with_window_dims([2, 2])
            .with_strides([2, 2]),
    )?;
    let dy = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 3, 4, 4])?,
    ));

    let dx = graph.resample_backward_infer(
        input,
        output,
        dy,
        None,
        ResampleConfig::new(ResampleMode::MaxPool, DataType::F32)
            .with_window_dims([2, 2])
            .with_strides([2, 2]),
    )?;

    assert_eq!(graph.tensor_config(dx)?.data_type, DataType::F16);

    Ok(())
}

#[test]
fn test_resample_backward_infer_supports_3d_spatial_dims() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8, 8, 8])?,
    ));
    let output = graph.resample_infer(
        input,
        None,
        ResampleConfig::new(ResampleMode::MaxPool, DataType::F32)
            .with_window_dims(vec![2, 2, 2])
            .with_strides([2, 2, 2]),
    )?;
    let dy = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 4, 4])?,
    ));

    let dx = graph.resample_backward_infer(
        input,
        output,
        dy,
        None,
        ResampleConfig::new(ResampleMode::MaxPool, DataType::F32)
            .with_window_dims(vec![2, 2, 2])
            .with_strides([2, 2, 2]),
    )?;

    assert_eq!(graph.shape(dx)?.dimensions(), &[2, 3, 8, 8, 8]);

    Ok(())
}

#[test]
fn test_resample_backward_infer_invalid_indices_do_not_insert_gradient_tensor() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8, 8])?,
    ));
    let output = graph.resample_infer(
        input,
        None,
        ResampleConfig::new(ResampleMode::MaxPool, DataType::F32)
            .with_window_dims([2, 2])
            .with_strides([2, 2]),
    )?;
    let dy = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 4])?,
    ));
    let indices = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 4])?,
    ));
    let tensor_count = graph.tensors.len();
    let operation_count = graph.operations.len();

    let error = graph
        .resample_backward_infer(
            input,
            output,
            dy,
            Some(indices),
            ResampleConfig::new(ResampleMode::AvgPoolIncludePadding, DataType::F32)
                .with_window_dims([2, 2])
                .with_strides([2, 2]),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "resample backward indices"
    ));
    assert_eq!(graph.tensors.len(), tensor_count);
    assert_eq!(graph.operations.len(), operation_count);

    Ok(())
}

#[test]
fn test_resample_backward_infer_error_removes_generated_gradient_tensor() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8, 8])?,
    ));
    let output = graph.resample_infer(
        input,
        None,
        ResampleConfig::new(ResampleMode::MaxPool, DataType::F32)
            .with_window_dims([2, 2])
            .with_strides([2, 2]),
    )?;
    let dy = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 5, 4])?,
    ));
    let tensor_count = graph.tensors.len();
    let operation_count = graph.operations.len();

    let error = graph
        .resample_backward_infer(
            input,
            output,
            dy,
            None,
            ResampleConfig::new(ResampleMode::MaxPool, DataType::F32)
                .with_window_dims([2, 2])
                .with_strides([2, 2]),
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
            && operation == "resample backward output gradient shape"
            && expected == vec![2, 3, 4, 4]
            && actual == vec![2, 3, 5, 4]
    ));
    assert_eq!(graph.tensors.len(), tensor_count);
    assert_eq!(graph.operations.len(), operation_count);

    Ok(())
}

#[test]
fn test_genstats_bn_helpers_build_expected_shapes() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 16])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);
    let momentum = graph.tensor(TensorSpec::scalar_f32(0.1)?);
    let accum_count = graph.tensor(
        TensorSpec::new(DataType::I64, Shape::contiguous([1])?)
            .with_scalar_value(ScalarValue::I64(64))?,
    );

    let stats = graph.genstats_infer(input, GenStatsConfig::new())?;
    let scale = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 32, 1, 1])?,
    ));
    let bias = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 32, 1, 1])?,
    ));
    let prev_mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 32, 1, 1])?,
    ));
    let prev_var = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 32, 1, 1])?,
    ));

    let bn_finalize_outputs = graph.batch_normalization_finalize_infer(
        stats.sum,
        stats.square_sum,
        scale,
        bias,
        accum_count,
        BatchNormalizationFinalizeConfig::training(epsilon).with_running_stats(
            BatchNormalizationFinalizeRunningStats::new(momentum, prev_mean, prev_var),
        ),
    )?;
    let eq_scale = bn_finalize_outputs.eq_scale;
    let eq_bias = bn_finalize_outputs.eq_bias;
    let saved_mean = bn_finalize_outputs.saved_mean;
    let saved_inv_variance = bn_finalize_outputs.saved_inv_variance;
    let next_mean = bn_finalize_outputs.next_running_mean;
    let next_var = bn_finalize_outputs.next_running_var;

    let y = graph.batch_normalization_inference_infer(
        input,
        saved_mean,
        saved_inv_variance,
        scale,
        bias,
        BatchNormalizationInferenceConfig::new(epsilon),
    )?;

    let dbn_weight_outputs = graph.dbn_weight_infer(
        y,
        input,
        scale,
        saved_mean,
        saved_inv_variance,
        DbnWeightConfig::new(),
    )?;
    let dscale = dbn_weight_outputs.dscale;
    let bias_gradient = dbn_weight_outputs.bias_gradient;
    let eq_bias_bwd = dbn_weight_outputs.eq_bias;
    let eq_scale_dy = dbn_weight_outputs.eq_scale_dy;
    let eq_scale_x = dbn_weight_outputs.eq_scale_x;

    assert_eq!(graph.shape(stats.sum)?.dimensions(), &[1, 32, 1, 1]);
    assert_eq!(graph.shape(stats.sum)?.strides(), &[32, 1, 32, 32]);
    assert_eq!(graph.shape(eq_scale)?.dimensions(), &[1, 32, 1, 1]);
    assert_eq!(graph.shape(eq_bias)?.dimensions(), &[1, 32, 1, 1]);
    assert_eq!(
        graph
            .shape(next_mean.expect("next running mean"))?
            .dimensions(),
        &[1, 32, 1, 1]
    );
    assert_eq!(
        graph
            .shape(next_var.expect("next running var"))?
            .dimensions(),
        &[1, 32, 1, 1]
    );
    assert_eq!(graph.shape(y)?.dimensions(), &[4, 32, 16, 16]);
    assert_eq!(graph.shape(dscale)?.dimensions(), &[1, 32, 1, 1]);
    assert_eq!(graph.shape(dscale)?.strides(), &[32, 1, 32, 32]);
    assert_eq!(graph.shape(bias_gradient)?.dimensions(), &[1, 32, 1, 1]);
    assert_eq!(graph.shape(eq_bias_bwd)?.dimensions(), &[1, 32, 1, 1]);
    assert_eq!(graph.shape(eq_scale_dy)?.dimensions(), &[1, 32, 1, 1]);
    assert_eq!(graph.shape(eq_scale_x)?.dimensions(), &[1, 32, 1, 1]);

    Ok(())
}

#[test]
fn test_genstats_rejects_mismatched_sum_shape() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 16])?,
    ));
    let sum = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 31, 1, 1])?,
    ));
    let sq_sum = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 32, 1, 1])?,
    ));

    let error = graph
        .genstats(input, sum, sq_sum, GenStatsConfig::new())
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == sum
            && operation == "genstats sum shape"
            && expected == vec![1, 32, 1, 1]
            && actual == vec![1, 31, 1, 1]
    ));

    Ok(())
}

#[test]
fn test_genstats_rejects_mismatched_sq_sum_shape() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 16])?,
    ));
    let sum = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 32, 1, 1])?,
    ));
    let sq_sum = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 33, 1, 1])?,
    ));

    let error = graph
        .genstats(input, sum, sq_sum, GenStatsConfig::new())
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == sq_sum
            && operation == "genstats sq_sum shape"
            && expected == vec![1, 32, 1, 1]
            && actual == vec![1, 33, 1, 1]
    ));

    Ok(())
}

#[test]
fn test_genstats_rejects_mismatched_output_data_type() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 16])?,
    ));
    let sum = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 32, 1, 1])?,
    ));
    let sq_sum = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 32, 1, 1])?,
    ));

    let error = graph
        .genstats(input, sum, sq_sum, GenStatsConfig::new())
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorDataTypeMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == sum
            && operation == "genstats sum"
            && expected == DataType::F32
            && actual == DataType::F16
    ));

    Ok(())
}

#[test]
fn test_genstats_uses_graph_io_data_type_for_explicit_outputs() -> Result<()> {
    let mut graph = Graph::new().with_io_data_type(DataType::BF16);
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 16])?,
    ));
    let sum = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 32, 1, 1])?,
    ));
    let sq_sum = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 32, 1, 1])?,
    ));

    graph.genstats(input, sum, sq_sum, GenStatsConfig::new())?;
    assert_eq!(graph.operations.len(), 1);

    Ok(())
}

#[test]
fn test_bn_finalize_rejects_mismatched_sq_sum_shape() -> Result<()> {
    let mut graph = Graph::new();
    let shape = Shape::contiguous([1, 32, 1, 1])?;
    let sum = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let sq_sum = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 31, 1, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let bias = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let saved_mean = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let saved_inv_variance = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let eq_scale = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let eq_bias = graph.tensor(TensorSpec::new(DataType::F32, shape));
    let accum_count = graph.tensor(
        TensorSpec::new(DataType::I64, Shape::contiguous([1])?)
            .with_scalar_value(ScalarValue::I64(64))?,
    );
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);

    let err = graph
        .batch_normalization_finalize(
            sum,
            sq_sum,
            scale,
            bias,
            None,
            None,
            saved_mean,
            saved_inv_variance,
            eq_scale,
            eq_bias,
            accum_count,
            BatchNormalizationFinalizeConfig::training(epsilon),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == sq_sum
            && operation == "bn finalize sq_sum shape"
            && expected == vec![1, 32, 1, 1]
            && actual == vec![1, 31, 1, 1]
    ));

    Ok(())
}

#[test]
fn test_bn_finalize_rejects_non_scalar_epsilon() -> Result<()> {
    let mut graph = Graph::new();
    let shape = Shape::contiguous([1, 32, 1, 1])?;
    let sum = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let sq_sum = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let scale = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let bias = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let prev_mean = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let prev_var = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let next_mean = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let next_var = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let saved_mean = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let saved_inv_variance = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let eq_scale = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let eq_bias = graph.tensor(TensorSpec::new(DataType::F32, shape));
    let accum_count = graph.tensor(
        TensorSpec::new(DataType::I64, Shape::contiguous([1])?)
            .with_scalar_value(ScalarValue::I64(64))?,
    );
    let epsilon = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 1, 1])?,
    ));
    let momentum = graph.tensor(TensorSpec::scalar_f32(0.1)?);

    let err = graph
        .batch_normalization_finalize(
            sum,
            sq_sum,
            scale,
            bias,
            Some(next_mean),
            Some(next_var),
            saved_mean,
            saved_inv_variance,
            eq_scale,
            eq_bias,
            accum_count,
            BatchNormalizationFinalizeConfig::training(epsilon).with_running_stats(
                BatchNormalizationFinalizeRunningStats::new(momentum, prev_mean, prev_var),
            ),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorElementCountMismatch {
            tensor_id,
            operation,
            expected: 1,
            actual: 2,
        } if tensor_id == epsilon && operation == "bn finalize epsilon"
    ));

    Ok(())
}

#[test]
fn test_bn_finalize_rejects_non_scalar_momentum() -> Result<()> {
    let mut graph = Graph::new();
    let shape = Shape::contiguous([1, 32, 1, 1])?;
    let sum = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let sq_sum = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let scale = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let bias = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let prev_mean = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let prev_var = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let next_mean = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let next_var = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let saved_mean = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let saved_inv_variance = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let eq_scale = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let eq_bias = graph.tensor(TensorSpec::new(DataType::F32, shape));
    let accum_count = graph.tensor(
        TensorSpec::new(DataType::I64, Shape::contiguous([1])?)
            .with_scalar_value(ScalarValue::I64(64))?,
    );
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);
    let momentum = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 1, 1])?,
    ));

    let err = graph
        .batch_normalization_finalize(
            sum,
            sq_sum,
            scale,
            bias,
            Some(next_mean),
            Some(next_var),
            saved_mean,
            saved_inv_variance,
            eq_scale,
            eq_bias,
            accum_count,
            BatchNormalizationFinalizeConfig::training(epsilon).with_running_stats(
                BatchNormalizationFinalizeRunningStats::new(momentum, prev_mean, prev_var),
            ),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorElementCountMismatch {
            tensor_id,
            operation,
            expected: 1,
            actual: 2,
        } if tensor_id == momentum && operation == "bn finalize momentum"
    ));

    Ok(())
}

#[test]
fn test_bn_finalize_infer_can_skip_running_stats() -> Result<()> {
    let mut graph = Graph::new();
    let shape = Shape::contiguous([1, 32, 1, 1])?;
    let sum = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let sq_sum = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let scale = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let bias = graph.tensor(TensorSpec::new(DataType::F32, shape));
    let accum_count = graph.tensor(
        TensorSpec::new(DataType::I64, Shape::contiguous([1])?)
            .with_scalar_value(ScalarValue::I64(64))?,
    );
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);

    let outputs = graph.batch_normalization_finalize_infer(
        sum,
        sq_sum,
        scale,
        bias,
        accum_count,
        BatchNormalizationFinalizeConfig::training(epsilon),
    )?;

    assert!(outputs.next_running_mean.is_none());
    assert!(outputs.next_running_var.is_none());

    Ok(())
}

#[test]
fn test_bn_finalize_infer_error_removes_generated_outputs() -> Result<()> {
    let mut graph = Graph::new();
    let shape = Shape::contiguous([1, 32, 1, 1])?;
    let sum = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let sq_sum = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let scale = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let bias = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let prev_mean = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let prev_var = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let accum_count = graph.tensor(
        TensorSpec::new(DataType::I64, Shape::contiguous([1])?)
            .with_scalar_value(ScalarValue::I64(64))?,
    );
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);
    let bad_momentum = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 1, 1])?,
    ));
    let tensor_count = graph.tensors.len();

    let error = graph
        .batch_normalization_finalize_infer(
            sum,
            sq_sum,
            scale,
            bias,
            accum_count,
            BatchNormalizationFinalizeConfig::training(epsilon).with_running_stats(
                BatchNormalizationFinalizeRunningStats::new(bad_momentum, prev_mean, prev_var),
            ),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorElementCountMismatch {
            tensor_id,
            operation,
            expected: 1,
            actual: 2,
        } if tensor_id == bad_momentum && operation == "bn finalize momentum"
    ));
    assert_eq!(graph.tensors.len(), tensor_count);
    assert!(graph.operations.is_empty());

    Ok(())
}

#[test]
fn test_lowered_bn_finalize_uses_f32_math_precision() -> Result<()> {
    let mut graph = Graph::new();
    let shape = Shape::contiguous([1, 32, 1, 1])?;
    let sum = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let sq_sum = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let scale = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let bias = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let saved_mean = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let saved_inv_variance = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let eq_scale = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()));
    let eq_bias = graph.tensor(TensorSpec::new(DataType::F32, shape));
    let accum_count = graph.tensor(
        TensorSpec::new(DataType::I64, Shape::contiguous([1])?)
            .with_scalar_value(ScalarValue::I64(64))?,
    );
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);

    graph.batch_normalization_finalize(
        sum,
        sq_sum,
        scale,
        bias,
        None,
        None,
        saved_mean,
        saved_inv_variance,
        eq_scale,
        eq_bias,
        accum_count,
        BatchNormalizationFinalizeConfig::training(epsilon),
    )?;

    let lowered = match LoweredGraph::lower(&graph) {
        Ok(lowered) => lowered,
        Err(Error::Cudnn {
            code: Status::NotSupported,
            ..
        }) => return Ok(()),
        Err(Error::Cudnn {
            code: Status::BadParam,
            ..
        }) => return Ok(()),
        Err(error) => return Err(error),
    };
    let op = match lowered.operations.first() {
        Some(LoweredOperation::BatchNormalizationFinalize(op)) => op,
        other => panic!("expected lowered batch norm finalize op, got {other:?}"),
    };
    let math_precision: DataType = match op.descriptor().attribute_enum(
        BackendAttributeName::OperationBnFinalizeMathPrec,
        BackendAttributeType::DataType,
    ) {
        Ok(math_precision) => math_precision,
        Err(Error::Cudnn {
            code: Status::NotSupported,
            ..
        }) => return Ok(()),
        Err(Error::Cudnn {
            code: Status::BadParam,
            ..
        }) => return Ok(()),
        Err(error) => return Err(error),
    };
    assert_eq!(math_precision, DataType::F32);

    Ok(())
}

#[test]
fn test_genstats_graph_compiles_when_plan_is_available() -> Result<()> {
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
        Shape::contiguous([4, 32, 16, 16])?,
    ));
    let stats = graph.genstats_infer(input, GenStatsConfig::new())?;
    graph.reshape(stats.sum, stats.sum)?;
    graph.reshape(stats.square_sum, stats.square_sum)?;

    match graph.compile(&context, &[HeuristicMode::Instant, HeuristicMode::Fallback]) {
        Ok(compiled) => {
            assert!(!compiled.execution_plans().is_empty());
            let _ = compiled.workspace_size()?;
        }
        Err(Error::NoAvailableEngines) => {}
        Err(Error::Cudnn {
            code: Status::BadParam,
            ..
        }) => {}
        Err(Error::Cudnn {
            code: Status::BadParamAttributeType,
            ..
        }) => {}
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_genstats_and_dbn_weight_infer_use_graph_io_data_type() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_io_data_type(DataType::BF16);

    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 16])?,
    ));
    let dy = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 16])?,
    ));
    let channel_shape = Shape::contiguous([1, 32, 1, 1])?;
    let scale = graph.tensor(TensorSpec::new(DataType::F32, channel_shape.clone()));
    let mean = graph.tensor(TensorSpec::new(DataType::F32, channel_shape.clone()));
    let inv_variance = graph.tensor(TensorSpec::new(DataType::F32, channel_shape));

    let stats = graph.genstats_infer(input, GenStatsConfig::new())?;
    let dbn_weight_outputs =
        graph.dbn_weight_infer(dy, input, scale, mean, inv_variance, DbnWeightConfig::new())?;
    let dscale = dbn_weight_outputs.dscale;
    let bias_gradient = dbn_weight_outputs.bias_gradient;
    let eq_bias = dbn_weight_outputs.eq_bias;
    let eq_scale_dy = dbn_weight_outputs.eq_scale_dy;
    let eq_scale_x = dbn_weight_outputs.eq_scale_x;

    for tensor in [
        stats.sum,
        stats.square_sum,
        dscale,
        bias_gradient,
        eq_bias,
        eq_scale_dy,
        eq_scale_x,
    ] {
        assert_eq!(graph.tensor_config(tensor)?.data_type, DataType::BF16);
    }

    Ok(())
}

#[test]
fn test_dbn_weight_infer_error_removes_generated_outputs() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 16])?,
    ));
    let dy = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 31, 16, 16])?,
    ));
    let channel_shape = Shape::contiguous([1, 32, 1, 1])?;
    let scale = graph.tensor(TensorSpec::new(DataType::F32, channel_shape.clone()));
    let mean = graph.tensor(TensorSpec::new(DataType::F32, channel_shape.clone()));
    let inv_variance = graph.tensor(TensorSpec::new(DataType::F32, channel_shape));
    let tensor_count = graph.tensors.len();

    let error = graph
        .dbn_weight_infer(dy, input, scale, mean, inv_variance, DbnWeightConfig::new())
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == dy
            && operation == "dbn weight dy shape"
            && expected == vec![4, 32, 16, 16]
            && actual == vec![4, 31, 16, 16]
    ));
    assert_eq!(graph.tensors.len(), tensor_count);
    assert!(graph.operations.is_empty());

    Ok(())
}

#[test]
fn test_lowered_batch_norm_inference_includes_epsilon_descriptor() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 16])?,
    ));
    let channel = Shape::contiguous([1, 32, 1, 1])?;
    let mean = graph.tensor(TensorSpec::new(DataType::F32, channel.clone()));
    let inv_variance = graph.tensor(TensorSpec::new(DataType::F32, channel.clone()));
    let scale = graph.tensor(TensorSpec::new(DataType::F32, channel.clone()));
    let bias = graph.tensor(TensorSpec::new(DataType::F32, channel));
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 16])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);

    graph.batch_normalization_inference(
        input,
        mean,
        inv_variance,
        scale,
        bias,
        output,
        BatchNormalizationInferenceConfig::new(epsilon),
    )?;

    let lowered = LoweredGraph::lower(&graph)?;
    let op = match lowered.operations.first() {
        Some(LoweredOperation::BatchNormalizationInference(op)) => op,
        other => panic!("expected lowered batch norm inference op, got {other:?}"),
    };
    match op.descriptor().attribute_descriptor(
        BackendAttributeName::OperationNormFwdEpsilonDesc,
        BackendAttributeType::BackendDescriptor,
    ) {
        Ok(_) => {}
        Err(Error::Cudnn {
            code: Status::NotSupported,
            ..
        }) => return Ok(()),
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_batch_norm_inference_rejects_mismatched_parameter_data_types() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 16])?,
    ));
    let channel = Shape::contiguous([1, 32, 1, 1])?;
    let mean = graph.tensor(TensorSpec::new(DataType::F32, channel.clone()));
    let inv_variance = graph.tensor(TensorSpec::new(DataType::F32, channel.clone()));
    let scale = graph.tensor(TensorSpec::new(DataType::F16, channel.clone()));
    let bias = graph.tensor(TensorSpec::new(DataType::F32, channel.clone()));
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 16])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);

    let err = graph
        .batch_normalization_inference(
            input,
            mean,
            inv_variance,
            scale,
            bias,
            output,
            BatchNormalizationInferenceConfig::new(epsilon),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorDataTypeMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == scale
            && operation == "batch norm inference scale data type"
            && expected == DataType::F32
            && actual == DataType::F16
    ));

    Ok(())
}

#[test]
fn test_batch_norm_inference_rejects_mismatched_epsilon_data_type() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 16])?,
    ));
    let channel = Shape::contiguous([1, 32, 1, 1])?;
    let mean = graph.tensor(TensorSpec::new(DataType::F32, channel.clone()));
    let inv_variance = graph.tensor(TensorSpec::new(DataType::F32, channel.clone()));
    let scale = graph.tensor(TensorSpec::new(DataType::F32, channel.clone()));
    let bias = graph.tensor(TensorSpec::new(DataType::F32, channel));
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 16])?,
    ));
    let epsilon = graph.tensor(TensorSpec::new(DataType::F16, Shape::contiguous([1])?));

    let err = graph
        .batch_normalization_inference(
            input,
            mean,
            inv_variance,
            scale,
            bias,
            output,
            BatchNormalizationInferenceConfig::new(epsilon),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorDataTypeMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == epsilon
            && operation == "batch norm inference epsilon data type"
            && expected == DataType::F32
            && actual == DataType::F16
    ));

    Ok(())
}

#[test]
fn test_batch_norm_inference_rejects_mismatched_output_data_type() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 16])?,
    ));
    let channel = Shape::contiguous([1, 32, 1, 1])?;
    let mean = graph.tensor(TensorSpec::new(DataType::F32, channel.clone()));
    let inv_variance = graph.tensor(TensorSpec::new(DataType::F32, channel.clone()));
    let scale = graph.tensor(TensorSpec::new(DataType::F32, channel.clone()));
    let bias = graph.tensor(TensorSpec::new(DataType::F32, channel));
    let output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([4, 32, 16, 16])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);

    let err = graph
        .batch_normalization_inference(
            input,
            mean,
            inv_variance,
            scale,
            bias,
            output,
            BatchNormalizationInferenceConfig::new(epsilon),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorDataTypeMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == output
            && operation == "batch norm inference output data type"
            && expected == DataType::F16
            && actual == DataType::F32
    ));

    Ok(())
}

#[test]
fn test_batch_norm_inference_uses_graph_io_data_type_for_explicit_output() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_io_data_type(DataType::BF16);
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 16])?,
    ));
    let channel = Shape::contiguous([1, 32, 1, 1])?;
    let mean = graph.tensor(TensorSpec::new(DataType::F32, channel.clone()));
    let inv_variance = graph.tensor(TensorSpec::new(DataType::F32, channel.clone()));
    let scale = graph.tensor(TensorSpec::new(DataType::F32, channel.clone()));
    let bias = graph.tensor(TensorSpec::new(DataType::F32, channel));
    let output = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([4, 32, 16, 16])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);

    graph.batch_normalization_inference(
        input,
        mean,
        inv_variance,
        scale,
        bias,
        output,
        BatchNormalizationInferenceConfig::new(epsilon),
    )?;

    assert!(matches!(
        graph.operations.last(),
        Some(Operation::BatchNormalizationInference { .. })
    ));

    Ok(())
}

#[test]
fn test_batch_norm_inference_rejects_non_scalar_epsilon() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 16])?,
    ));
    let channel = Shape::contiguous([1, 32, 1, 1])?;
    let mean = graph.tensor(TensorSpec::new(DataType::F32, channel.clone()));
    let inv_variance = graph.tensor(TensorSpec::new(DataType::F32, channel.clone()));
    let scale = graph.tensor(TensorSpec::new(DataType::F32, channel.clone()));
    let bias = graph.tensor(TensorSpec::new(DataType::F32, channel.clone()));
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 16])?,
    ));
    let epsilon = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2])?));

    let err = graph
        .batch_normalization_inference(
            input,
            mean,
            inv_variance,
            scale,
            bias,
            output,
            BatchNormalizationInferenceConfig::new(epsilon),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorElementCountMismatch {
            tensor_id,
            operation,
            expected: 1,
            actual: 2,
        } if tensor_id == epsilon && operation == "batch norm inference epsilon"
    ));

    Ok(())
}

#[test]
fn test_batch_norm_inference_rejects_mismatched_scale_shape() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 16])?,
    ));
    let channel = Shape::contiguous([1, 32, 1, 1])?;
    let mean = graph.tensor(TensorSpec::new(DataType::F32, channel.clone()));
    let inv_variance = graph.tensor(TensorSpec::new(DataType::F32, channel));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 31, 1, 1])?,
    ));
    let bias = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 32, 1, 1])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 16])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);

    let err = graph
        .batch_normalization_inference(
            input,
            mean,
            inv_variance,
            scale,
            bias,
            output,
            BatchNormalizationInferenceConfig::new(epsilon),
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
            && operation == "batch norm inference scale shape"
            && expected == vec![1, 32, 1, 1]
            && actual == vec![1, 31, 1, 1]
    ));

    Ok(())
}

#[test]
fn test_batch_norm_inference_rejects_mismatched_output_shape() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 16])?,
    ));
    let channel = Shape::contiguous([1, 32, 1, 1])?;
    let mean = graph.tensor(TensorSpec::new(DataType::F32, channel.clone()));
    let inv_variance = graph.tensor(TensorSpec::new(DataType::F32, channel.clone()));
    let scale = graph.tensor(TensorSpec::new(DataType::F32, channel.clone()));
    let bias = graph.tensor(TensorSpec::new(DataType::F32, channel));
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 15])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);

    let err = graph
        .batch_normalization_inference(
            input,
            mean,
            inv_variance,
            scale,
            bias,
            output,
            BatchNormalizationInferenceConfig::new(epsilon),
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
            && operation == "batch norm inference output shape"
            && expected == vec![4, 32, 16, 16]
            && actual == vec![4, 32, 16, 15]
    ));

    Ok(())
}

#[test]
fn test_dbn_weight_graph_compiles_when_plan_is_available() -> Result<()> {
    let context = match setup_context() {
        Ok(ctx) => ctx,
        Err(Error::Cuda(singe_cuda::error::Error::Cuda { code, .. }))
            if code == singe_cuda::error::Status::NoDevice =>
        {
            return Ok(());
        }
        Err(error) => return Err(error),
    };

    let n = 4;
    let c = 32;
    let h = 16;
    let w = 16;
    let k = 64;

    let mut graph = Graph::new().with_io_data_type(DataType::F32);
    let dy = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([n, k, h, w])?.with_strides([k * h * w, 1, k * w, k])?,
    ));
    let weight = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([k, c, 3, 3])?.with_strides([c * 3 * 3, 1, c * 3, c])?,
    ));
    let dgrad = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([n, c, h, w])?.with_strides(vec![c * h * w, 1, c * w, c])?,
        )
        .virtual_tensor(),
    );
    graph.convolution_dgrad(
        weight,
        dy,
        dgrad,
        ConvolutionConfig::new(DataType::F32, 2)
            .with_padding(vec![1, 1])
            .with_strides([1, 1])
            .with_dilations(vec![1, 1]),
    )?;

    let x = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([n, c, h, w])?.with_strides([c * h * w, 1, c * w, c])?,
    ));
    let mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
    ));
    let centered = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([n, c, h, w])?.with_strides(vec![c * h * w, 1, c * w, c])?,
        )
        .virtual_tensor(),
    );
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::Sub,
        lhs: x,
        rhs: mean,
        output: centered,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });

    let inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
    ));
    let normalized = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([n, c, h, w])?.with_strides(vec![c * h * w, 1, c * w, c])?,
        )
        .virtual_tensor(),
    );
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::Mul,
        lhs: centered,
        rhs: inv_variance,
        output: normalized,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });

    let scale = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
    ));
    let scaled = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([n, c, h, w])?.with_strides(vec![c * h * w, 1, c * w, c])?,
        )
        .virtual_tensor(),
    );
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::Mul,
        lhs: normalized,
        rhs: scale,
        output: scaled,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });

    let bias = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
    ));
    let bn_output = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([n, c, h, w])?.with_strides(vec![c * h * w, 1, c * w, c])?,
        )
        .virtual_tensor(),
    );
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::Add,
        lhs: scaled,
        rhs: bias,
        output: bn_output,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });

    let drelu = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([n, c, h, w])?.with_strides(vec![c * h * w, 1, c * w, c])?,
        )
        .output_tensor(),
    );
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::ReluBwd,
        lhs: dgrad,
        rhs: bn_output,
        output: drelu,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });

    let _ = graph.dbn_weight_infer(drelu, x, scale, mean, inv_variance, DbnWeightConfig::new())?;

    match graph.compile_with_config(
        &context,
        &CompileConfig::new()
            .with_heuristic_modes(vec![HeuristicMode::A])
            .with_build_policy(BuildPlanPolicy::AllSupported),
    ) {
        Ok(compiled) => {
            assert!(!compiled.execution_plans().is_empty());
            let _ = compiled.max_workspace_size()?;
        }
        Err(Error::NoAvailableEngines) => {}
        Err(Error::Cudnn {
            code: Status::BadParam,
            ..
        }) => {}
        Err(Error::Cudnn {
            code: Status::BadParamAttributeType,
            ..
        }) => {}
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_lowered_dbn_weight_uses_f32_math_precision() -> Result<()> {
    let mut graph = Graph::new();
    let dy = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 16])?,
    ));
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 16])?,
    ));
    let channel = Shape::contiguous([1, 32, 1, 1])?;
    let scale = graph.tensor(TensorSpec::new(DataType::F32, channel.clone()));
    let mean = graph.tensor(TensorSpec::new(DataType::F32, channel.clone()));
    let inv_variance = graph.tensor(TensorSpec::new(DataType::F32, channel.clone()));
    let dscale = graph.tensor(TensorSpec::new(DataType::F32, channel.clone()));
    let bias_gradient = graph.tensor(TensorSpec::new(DataType::F32, channel.clone()));
    let eq_bias = graph.tensor(TensorSpec::new(DataType::F32, channel.clone()));
    let eq_scale_dy = graph.tensor(TensorSpec::new(DataType::F32, channel.clone()));
    let eq_scale_x = graph.tensor(TensorSpec::new(DataType::F32, channel));

    graph.dbn_weight(
        dy,
        input,
        scale,
        mean,
        inv_variance,
        dscale,
        bias_gradient,
        eq_bias,
        eq_scale_dy,
        eq_scale_x,
        DbnWeightConfig::new(),
    )?;

    let lowered = match LoweredGraph::lower(&graph) {
        Ok(lowered) => lowered,
        Err(Error::Cudnn {
            code: Status::NotSupported,
            ..
        }) => return Ok(()),
        Err(error) => return Err(error),
    };
    let op = match lowered.operations.first() {
        Some(LoweredOperation::DbnWeight(op)) => op,
        other => panic!("expected lowered dbn weight op, got {other:?}"),
    };
    let math_precision: DataType = match op.descriptor().attribute_enum(
        BackendAttributeName::OperationBnBwdWeightsMathPrec,
        BackendAttributeType::DataType,
    ) {
        Ok(math_precision) => math_precision,
        Err(Error::Cudnn {
            code: Status::NotSupported,
            ..
        }) => return Ok(()),
        Err(error) => return Err(error),
    };
    assert_eq!(math_precision, DataType::F32);

    Ok(())
}

#[test]
fn test_dbn_weight_graph_prefix_diagnostics() -> Result<()> {
    let n = 4;
    let c = 32;
    let h = 16;
    let w = 16;
    let k = 64;

    let mut graph = Graph::new().with_io_data_type(DataType::F32);
    let dy = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([n, k, h, w])?.with_strides([k * h * w, 1, k * w, k])?,
    ));
    let weight = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([k, c, 3, 3])?.with_strides([c * 3 * 3, 1, c * 3, c])?,
    ));
    let dgrad = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([n, c, h, w])?.with_strides(vec![c * h * w, 1, c * w, c])?,
        )
        .virtual_tensor(),
    );
    graph.convolution_dgrad(
        weight,
        dy,
        dgrad,
        ConvolutionConfig::new(DataType::F32, 2)
            .with_padding(vec![1, 1])
            .with_strides([1, 1])
            .with_dilations(vec![1, 1]),
    )?;

    let x = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([n, c, h, w])?.with_strides([c * h * w, 1, c * w, c])?,
    ));
    let mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
    ));
    let centered = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([n, c, h, w])?.with_strides(vec![c * h * w, 1, c * w, c])?,
        )
        .virtual_tensor(),
    );
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::Sub,
        lhs: x,
        rhs: mean,
        output: centered,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });

    let inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
    ));
    let normalized = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([n, c, h, w])?.with_strides(vec![c * h * w, 1, c * w, c])?,
        )
        .virtual_tensor(),
    );
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::Mul,
        lhs: centered,
        rhs: inv_variance,
        output: normalized,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });

    let scale = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
    ));
    let scaled = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([n, c, h, w])?.with_strides(vec![c * h * w, 1, c * w, c])?,
        )
        .virtual_tensor(),
    );
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::Mul,
        lhs: normalized,
        rhs: scale,
        output: scaled,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });

    let bias = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
    ));
    let bn_output = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([n, c, h, w])?.with_strides(vec![c * h * w, 1, c * w, c])?,
        )
        .virtual_tensor(),
    );
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::Add,
        lhs: scaled,
        rhs: bias,
        output: bn_output,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });

    let drelu = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([n, c, h, w])?.with_strides(vec![c * h * w, 1, c * w, c])?,
        )
        .output_tensor(),
    );
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::ReluBwd,
        lhs: dgrad,
        rhs: bn_output,
        output: drelu,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });

    let _ = graph.dbn_weight_infer(drelu, x, scale, mean, inv_variance, DbnWeightConfig::new())?;

    Ok(())
}

#[test]
fn test_pointwise_sub_mixed_dtype_graph_prepares() -> Result<()> {
    let context = setup_context()?;

    let mut graph = Graph::new();
    let x = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 16])?.with_strides(vec![32 * 16 * 16, 1, 32 * 16, 32])?,
    ));
    let mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 32, 1, 1])?.with_strides([32, 1, 32, 32])?,
    ));
    let y = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([4, 32, 16, 16])?.with_strides(vec![32 * 16 * 16, 1, 32 * 16, 32])?,
        )
        .virtual_tensor()
        .output_tensor(),
    );
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::Sub,
        lhs: x,
        rhs: mean,
        output: y,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });

    graph.prepare(&context).map(|_| ())
}

#[test]
fn test_pointwise_relu_bwd_mixed_dtype_graph_prepares() -> Result<()> {
    let context = setup_context()?;

    let mut graph = Graph::new();
    let dy = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([4, 32, 16, 16])?.with_strides(vec![32 * 16 * 16, 1, 32 * 16, 32])?,
    ));
    let x = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([4, 32, 16, 16])?.with_strides(vec![32 * 16 * 16, 1, 32 * 16, 32])?,
    ));
    let dx = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([4, 32, 16, 16])?.with_strides(vec![32 * 16 * 16, 1, 32 * 16, 32])?,
        )
        .virtual_tensor()
        .output_tensor(),
    );
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::ReluBwd,
        lhs: dy,
        rhs: x,
        output: dx,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });

    graph.prepare(&context).map(|_| ())
}

#[test]
fn test_dbn_weight_graph_executes_when_plan_is_available() -> Result<()> {
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
    let dy = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 16])?.with_strides(vec![32 * 16 * 16, 1, 32 * 16, 32])?,
    ));
    let x = graph.tensor(TensorSpec::new(
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

    let dbn_weight_outputs =
        graph.dbn_weight_infer(dy, x, scale, mean, inv_variance, DbnWeightConfig::new())?;
    let dscale = dbn_weight_outputs.dscale;
    let bias_gradient = dbn_weight_outputs.bias_gradient;
    let eq_bias = dbn_weight_outputs.eq_bias;
    let eq_scale_dy = dbn_weight_outputs.eq_scale_dy;
    let eq_scale_x = dbn_weight_outputs.eq_scale_x;

    let compiled = match graph.compile_with_config(
        &context,
        &CompileConfig::new()
            .with_heuristic_modes(vec![HeuristicMode::A, HeuristicMode::Fallback])
            .with_build_policy(BuildPlanPolicy::AllSupported),
    ) {
        Ok(compiled) => compiled,
        Err(Error::NoAvailableEngines) => return Ok(()),
        Err(Error::Cudnn {
            code: Status::NotSupported,
            ..
        }) => return Ok(()),
        Err(Error::Cudnn {
            code: Status::BadParam,
            ..
        }) => return Ok(()),
        Err(Error::Cudnn {
            code: Status::InternalErrorUnexpectedValue,
            ..
        }) => return Ok(()),
        Err(error) if is_expected_compile_status(&error) => return Ok(()),
        Err(error) => return Err(error),
    };

    let mut dy_dev = DeviceMemory::<f16>::zeroes(4 * 32 * 16 * 16)?;
    let mut x_dev = DeviceMemory::<f16>::zeroes(4 * 32 * 16 * 16)?;
    let mut scale_dev = DeviceMemory::<f32>::zeroes(32)?;
    let mut mean_dev = DeviceMemory::<f32>::zeroes(32)?;
    let mut inv_variance_dev = DeviceMemory::<f32>::zeroes(32)?;
    let mut dscale_dev = DeviceMemory::<f32>::zeroes(32)?;
    let mut bias_gradient_dev = DeviceMemory::<f32>::zeroes(32)?;
    let mut eq_bias_dev = DeviceMemory::<f32>::zeroes(32)?;
    let mut eq_scale_dy_dev = DeviceMemory::<f32>::zeroes(32)?;
    let mut eq_scale_x_dev = DeviceMemory::<f32>::zeroes(32)?;
    let mut workspace = DeviceMemory::<u8>::zeroes(compiled.max_workspace_size()?)?;
    let mut bindings = compiled.bindings();
    bindings.set(dy, &mut dy_dev)?;
    bindings.set(x, &mut x_dev)?;
    bindings.set(scale, &mut scale_dev)?;
    bindings.set(mean, &mut mean_dev)?;
    bindings.set(inv_variance, &mut inv_variance_dev)?;
    bindings.set(dscale, &mut dscale_dev)?;
    bindings.set(bias_gradient, &mut bias_gradient_dev)?;
    bindings.set(eq_bias, &mut eq_bias_dev)?;
    bindings.set(eq_scale_dy, &mut eq_scale_dy_dev)?;
    bindings.set(eq_scale_x, &mut eq_scale_x_dev)?;

    match compiled.execute(&context, &bindings, Some(&mut workspace)) {
        Ok(()) => {}
        Err(Error::NoAvailableEngines) => return Ok(()),
        Err(Error::Cudnn {
            code: Status::NotSupported,
            ..
        }) => return Ok(()),
        Err(Error::Cudnn {
            code: Status::BadParam,
            ..
        }) => return Ok(()),
        Err(Error::Cudnn {
            code: Status::InternalErrorUnexpectedValue,
            ..
        }) => return Ok(()),
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_dgrad_drelu_dbn_weight_graph_executes_when_plan_is_available() -> Result<()> {
    let context = match setup_context() {
        Ok(ctx) => ctx,
        Err(Error::Cuda(singe_cuda::error::Error::Cuda { code, .. }))
            if code == singe_cuda::error::Status::NoDevice =>
        {
            return Ok(());
        }
        Err(error) => return Err(error),
    };

    let n = 4;
    let c = 32;
    let h = 16;
    let w = 16;
    let k = 64;

    let mut graph = Graph::new().with_io_data_type(DataType::F32);
    let dy = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([n, k, h, w])?.with_strides([k * h * w, 1, k * w, k])?,
    ));
    let weight = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([k, c, 3, 3])?.with_strides([c * 3 * 3, 1, c * 3, c])?,
    ));
    let dgrad = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([n, c, h, w])?.with_strides(vec![c * h * w, 1, c * w, c])?,
        )
        .virtual_tensor(),
    );
    graph.convolution_dgrad(
        weight,
        dy,
        dgrad,
        ConvolutionConfig::new(DataType::F32, 2)
            .with_padding(vec![1, 1])
            .with_strides([1, 1])
            .with_dilations(vec![1, 1]),
    )?;

    let bn_output = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([n, c, h, w])?.with_strides(vec![c * h * w, 1, c * w, c])?,
        )
        .virtual_tensor(),
    );
    let bias = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
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
    let x = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([n, c, h, w])?.with_strides([c * h * w, 1, c * w, c])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);

    graph.batch_normalization_inference(
        x,
        mean,
        inv_variance,
        scale,
        bias,
        bn_output,
        BatchNormalizationInferenceConfig::new(epsilon),
    )?;

    let drelu = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([n, c, h, w])?.with_strides(vec![c * h * w, 1, c * w, c])?,
        )
        .output_tensor(),
    );
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::ReluBwd,
        lhs: dgrad,
        rhs: bn_output,
        output: drelu,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });

    let dbn_weight_outputs =
        graph.dbn_weight_infer(drelu, x, scale, mean, inv_variance, DbnWeightConfig::new())?;
    let dscale = dbn_weight_outputs.dscale;
    let bias_gradient = dbn_weight_outputs.bias_gradient;
    let eq_bias = dbn_weight_outputs.eq_bias;
    let eq_scale_dy = dbn_weight_outputs.eq_scale_dy;
    let eq_scale_x = dbn_weight_outputs.eq_scale_x;

    let compiled = match graph.compile_with_config(
        &context,
        &CompileConfig::new()
            .with_heuristic_modes(vec![HeuristicMode::A, HeuristicMode::Fallback])
            .with_build_policy(BuildPlanPolicy::AllSupported),
    ) {
        Ok(compiled) => compiled,
        Err(Error::NoAvailableEngines) => return Ok(()),
        Err(Error::Cudnn {
            code: Status::NotSupported,
            ..
        }) => return Ok(()),
        Err(Error::Cudnn {
            code: Status::BadParam,
            ..
        }) => return Ok(()),
        Err(Error::Cudnn {
            code: Status::InternalErrorUnexpectedValue,
            ..
        }) => return Ok(()),
        Err(error) if is_expected_compile_status(&error) => return Ok(()),
        Err(error) => return Err(error),
    };

    let mut dy_dev = DeviceMemory::<f16>::zeroes((n * k * h * w) as usize)?;
    let mut weight_dev = DeviceMemory::<f16>::zeroes((k * c * 3 * 3) as usize)?;
    let mut x_dev = DeviceMemory::<f16>::zeroes((n * c * h * w) as usize)?;
    let mut scale_dev = DeviceMemory::<f32>::zeroes(c as usize)?;
    let mut mean_dev = DeviceMemory::<f32>::zeroes(c as usize)?;
    let mut inv_variance_dev = DeviceMemory::<f32>::zeroes(c as usize)?;
    let mut bias_dev = DeviceMemory::<f32>::zeroes(c as usize)?;
    let mut drelu_dev = DeviceMemory::<f16>::zeroes((n * c * h * w) as usize)?;
    let mut dscale_dev = DeviceMemory::<f32>::zeroes(c as usize)?;
    let mut bias_gradient_dev = DeviceMemory::<f32>::zeroes(c as usize)?;
    let mut eq_bias_dev = DeviceMemory::<f32>::zeroes(c as usize)?;
    let mut eq_scale_dy_dev = DeviceMemory::<f32>::zeroes(c as usize)?;
    let mut eq_scale_x_dev = DeviceMemory::<f32>::zeroes(c as usize)?;
    let mut workspace = DeviceMemory::<u8>::zeroes(compiled.max_workspace_size()?)?;
    let mut bindings = compiled.bindings();
    bindings.set(dy, &mut dy_dev)?;
    bindings.set(weight, &mut weight_dev)?;
    bindings.set(x, &mut x_dev)?;
    bindings.set(scale, &mut scale_dev)?;
    bindings.set(mean, &mut mean_dev)?;
    bindings.set(inv_variance, &mut inv_variance_dev)?;
    bindings.set(bias, &mut bias_dev)?;
    bindings.set(drelu, &mut drelu_dev)?;
    bindings.set(dscale, &mut dscale_dev)?;
    bindings.set(bias_gradient, &mut bias_gradient_dev)?;
    bindings.set(eq_bias, &mut eq_bias_dev)?;
    bindings.set(eq_scale_dy, &mut eq_scale_dy_dev)?;
    bindings.set(eq_scale_x, &mut eq_scale_x_dev)?;

    match compiled.execute(&context, &bindings, Some(&mut workspace)) {
        Ok(()) => {}
        Err(Error::NoAvailableEngines) => return Ok(()),
        Err(Error::Cudnn {
            code: Status::NotSupported,
            ..
        }) => return Ok(()),
        Err(Error::Cudnn {
            code: Status::BadParam,
            ..
        }) => return Ok(()),
        Err(Error::Cudnn {
            code: Status::InternalErrorUnexpectedValue,
            ..
        }) => return Ok(()),
        Err(error) if is_expected_compile_status(&error) => return Ok(()),
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_bn_inference_drelu_dbn_weight_graph_executes_when_plan_is_available() -> Result<()> {
    let context = match setup_context() {
        Ok(ctx) => ctx,
        Err(Error::Cuda(singe_cuda::error::Error::Cuda { code, .. }))
            if code == singe_cuda::error::Status::NoDevice =>
        {
            return Ok(());
        }
        Err(error) => return Err(error),
    };

    let n = 4;
    let c = 32;
    let h = 16;
    let w = 16;

    let mut graph = Graph::new();
    let x = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([n, c, h, w])?.with_strides([c * h * w, 1, c * w, c])?,
    ));
    let mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
    ));
    let inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
    ));
    let bias = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
    ));
    let bn_output = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([n, c, h, w])?.with_strides(vec![c * h * w, 1, c * w, c])?,
        )
        .virtual_tensor(),
    );
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);
    graph.batch_normalization_inference(
        x,
        mean,
        inv_variance,
        scale,
        bias,
        bn_output,
        BatchNormalizationInferenceConfig::new(epsilon),
    )?;

    let dy = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([n, c, h, w])?.with_strides([c * h * w, 1, c * w, c])?,
    ));
    let drelu = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([n, c, h, w])?.with_strides(vec![c * h * w, 1, c * w, c])?,
        )
        .output_tensor(),
    );
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::ReluBwd,
        lhs: dy,
        rhs: bn_output,
        output: drelu,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });

    let dbn_weight_outputs =
        graph.dbn_weight_infer(drelu, x, scale, mean, inv_variance, DbnWeightConfig::new())?;
    let dscale = dbn_weight_outputs.dscale;
    let bias_gradient = dbn_weight_outputs.bias_gradient;
    let eq_bias = dbn_weight_outputs.eq_bias;
    let eq_scale_dy = dbn_weight_outputs.eq_scale_dy;
    let eq_scale_x = dbn_weight_outputs.eq_scale_x;

    let compiled = match graph.compile_with_config(
        &context,
        &CompileConfig::new()
            .with_heuristic_modes(vec![HeuristicMode::A, HeuristicMode::Fallback])
            .with_build_policy(BuildPlanPolicy::AllSupported),
    ) {
        Ok(compiled) => compiled,
        Err(Error::NoAvailableEngines) => return Ok(()),
        Err(Error::Cudnn {
            code: Status::NotSupported,
            ..
        }) => return Ok(()),
        Err(Error::Cudnn {
            code: Status::BadParam,
            ..
        }) => return Ok(()),
        Err(Error::Cudnn {
            code: Status::InternalErrorUnexpectedValue,
            ..
        }) => return Ok(()),
        Err(error) if is_expected_compile_status(&error) => return Ok(()),
        Err(error) => return Err(error),
    };

    let mut x_dev = DeviceMemory::<f16>::zeroes((n * c * h * w) as usize)?;
    let mut dy_dev = DeviceMemory::<f16>::zeroes((n * c * h * w) as usize)?;
    let mut mean_dev = DeviceMemory::<f32>::zeroes(c as usize)?;
    let mut inv_variance_dev = DeviceMemory::<f32>::zeroes(c as usize)?;
    let mut scale_dev = DeviceMemory::<f32>::zeroes(c as usize)?;
    let mut bias_dev = DeviceMemory::<f32>::zeroes(c as usize)?;
    let mut drelu_dev = DeviceMemory::<f16>::zeroes((n * c * h * w) as usize)?;
    let mut dscale_dev = DeviceMemory::<f32>::zeroes(c as usize)?;
    let mut bias_gradient_dev = DeviceMemory::<f32>::zeroes(c as usize)?;
    let mut eq_bias_dev = DeviceMemory::<f32>::zeroes(c as usize)?;
    let mut eq_scale_dy_dev = DeviceMemory::<f32>::zeroes(c as usize)?;
    let mut eq_scale_x_dev = DeviceMemory::<f32>::zeroes(c as usize)?;
    let mut workspace = DeviceMemory::<u8>::zeroes(compiled.max_workspace_size()?)?;
    let mut bindings = compiled.bindings();
    bindings.set(x, &mut x_dev)?;
    bindings.set(dy, &mut dy_dev)?;
    bindings.set(mean, &mut mean_dev)?;
    bindings.set(inv_variance, &mut inv_variance_dev)?;
    bindings.set(scale, &mut scale_dev)?;
    bindings.set(bias, &mut bias_dev)?;
    bindings.set(drelu, &mut drelu_dev)?;
    bindings.set(dscale, &mut dscale_dev)?;
    bindings.set(bias_gradient, &mut bias_gradient_dev)?;
    bindings.set(eq_bias, &mut eq_bias_dev)?;
    bindings.set(eq_scale_dy, &mut eq_scale_dy_dev)?;
    bindings.set(eq_scale_x, &mut eq_scale_x_dev)?;
    match compiled.execute(&context, &bindings, Some(&mut workspace)) {
        Ok(()) => {}
        Err(Error::NoAvailableEngines) => return Ok(()),
        Err(Error::Cudnn {
            code: Status::NotSupported,
            ..
        }) => return Ok(()),
        Err(Error::Cudnn {
            code: Status::BadParam,
            ..
        }) => return Ok(()),
        Err(Error::Cudnn {
            code: Status::InternalErrorUnexpectedValue,
            ..
        }) => return Ok(()),
        Err(error) if is_expected_compile_status(&error) => return Ok(()),
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_batch_norm_inference_graph_compiles_when_plan_is_available() -> Result<()> {
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
    let scale = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 32, 1, 1])?.with_strides([32, 1, 32, 32])?,
    ));
    let bias = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 32, 1, 1])?.with_strides([32, 1, 32, 32])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);
    let _y = graph.batch_normalization_inference_infer(
        input,
        mean,
        inv_variance,
        scale,
        bias,
        BatchNormalizationInferenceConfig::new(epsilon),
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
fn test_rng_infer_creates_requested_output_shape() -> Result<()> {
    let mut graph = Graph::new();

    let output = graph.random_number_generator_infer(
        Shape::contiguous([2, 3, 4])?,
        RandomNumberGeneratorConfig::bernoulli(0.5, 7),
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[2, 3, 4]);

    Ok(())
}

#[test]
fn test_rng_infer_uses_graph_io_data_type() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_io_data_type(DataType::BF16);

    let output = graph.random_number_generator_infer(
        Shape::contiguous([2, 3, 4])?,
        RandomNumberGeneratorConfig::bernoulli(0.5, 7),
    )?;

    assert_eq!(graph.tensor_config(output)?.data_type, DataType::BF16);

    Ok(())
}

#[test]
fn test_rng_infer_with_dimensions_uses_nhwc_default_strides() -> Result<()> {
    let mut graph = Graph::new();

    let output = graph.random_number_generator_infer_with_dimensions(
        vec![2, 3, 4, 5],
        RandomNumberGeneratorConfig::bernoulli(0.5, 7),
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[2, 3, 4, 5]);
    assert_eq!(graph.shape(output)?.strides(), &[60, 1, 15, 3]);

    Ok(())
}

#[test]
fn test_rng_infer_with_dimensions_falls_back_to_contiguous_for_rank_one() -> Result<()> {
    let mut graph = Graph::new();

    let output = graph.random_number_generator_infer_with_dimensions(
        vec![7],
        RandomNumberGeneratorConfig::bernoulli(0.5, 7),
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[7]);
    assert_eq!(graph.shape(output)?.strides(), &[1]);

    Ok(())
}

#[test]
fn test_rng_rejects_unsupported_output_data_type() -> Result<()> {
    let mut graph = Graph::new();
    let output = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 3, 4])?,
    ));

    let err = graph
        .random_number_generator(output, RandomNumberGeneratorConfig::bernoulli(0.5, 7))
        .unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorDataTypeUnsupported {
            tensor_id,
            operation,
            supported,
            actual,
        } if tensor_id == output
            && operation == "rng output"
            && supported == vec![DataType::F16, DataType::BF16, DataType::F32]
            && actual == DataType::I32
    ));

    Ok(())
}

#[test]
fn test_rng_infer_error_removes_generated_output() -> Result<()> {
    let mut graph = Graph::new();
    let seed = graph.tensor(TensorSpec::new(DataType::I32, Shape::contiguous([1])?));
    let offset = graph.tensor(TensorSpec::new(DataType::I64, Shape::contiguous([1])?));
    let tensor_count = graph.tensors.len();

    let error = graph
        .random_number_generator_infer(
            Shape::contiguous([2, 3, 4])?,
            RandomNumberGeneratorConfig::bernoulli(0.5, 7).with_seed_tensor(seed, offset),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorDataTypeMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == seed
            && operation == "rng seed"
            && expected == DataType::I64
            && actual == DataType::I32
    ));
    assert_eq!(graph.tensors.len(), tensor_count);
    assert!(graph.operations.is_empty());

    Ok(())
}

#[test]
fn test_rng_rejects_non_scalar_offset() -> Result<()> {
    let mut graph = Graph::new();
    let output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4])?,
    ));
    let offset = graph.tensor(TensorSpec::new(DataType::I64, Shape::contiguous([2])?));

    let err = graph
        .random_number_generator(
            output,
            RandomNumberGeneratorConfig::bernoulli(0.5, 7).with_offset(offset),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == offset
            && operation == "rng offset"
            && expected == vec![1]
            && actual == vec![2]
    ));

    Ok(())
}

#[test]
fn test_rng_rejects_offset_with_non_unit_strides() -> Result<()> {
    let mut graph = Graph::new();
    let output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4])?,
    ));
    let offset = graph.tensor(TensorSpec::new(
        DataType::I64,
        Shape::contiguous([1, 1])?.with_strides([2, 1])?,
    ));

    let err = graph
        .random_number_generator(
            output,
            RandomNumberGeneratorConfig::bernoulli(0.5, 7).with_offset(offset),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorStridesMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == offset
            && operation == "rng offset"
            && expected == vec![1, 1]
            && actual == vec![2, 1]
    ));

    Ok(())
}

#[test]
fn test_rng_rejects_seed_with_non_scalar_dimensions() -> Result<()> {
    let mut graph = Graph::new();
    let output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4])?,
    ));
    let seed = graph.tensor(TensorSpec::new(DataType::I64, Shape::contiguous([2])?));
    let offset = graph.tensor(TensorSpec::new(DataType::I64, Shape::contiguous([1])?));

    let err = graph
        .random_number_generator(
            output,
            RandomNumberGeneratorConfig::bernoulli(0.5, 7).with_seed_tensor(seed, offset),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == seed
            && operation == "rng seed"
            && expected == vec![1]
            && actual == vec![2]
    ));

    Ok(())
}

#[test]
fn test_rng_rejects_seed_with_non_unit_strides() -> Result<()> {
    let mut graph = Graph::new();
    let output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4])?,
    ));
    let seed = graph.tensor(TensorSpec::new(
        DataType::I64,
        Shape::contiguous([1, 1])?.with_strides([2, 1])?,
    ));
    let offset = graph.tensor(TensorSpec::new(DataType::I64, Shape::contiguous([1])?));

    let err = graph
        .random_number_generator(
            output,
            RandomNumberGeneratorConfig::bernoulli(0.5, 7).with_seed_tensor(seed, offset),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorStridesMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == seed
            && operation == "rng seed"
            && expected == vec![1, 1]
            && actual == vec![2, 1]
    ));

    Ok(())
}

#[test]
fn test_rng_accepts_host_seed_offset() -> Result<()> {
    let mut graph = Graph::new();
    let output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4])?,
    ));
    let offset = graph.tensor(TensorSpec::new(
        DataType::I64,
        Shape::contiguous([1, 1, 1, 1])?,
    ));

    graph.random_number_generator(
        output,
        RandomNumberGeneratorConfig::bernoulli(0.5, 7).with_offset(offset),
    )?;
    assert!(matches!(
        graph.operations.last(),
        Some(Operation::RandomNumberGenerator { .. })
    ));

    Ok(())
}

#[test]
fn test_rng_accepts_device_seed_and_offset() -> Result<()> {
    let mut graph = Graph::new();
    let output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4])?,
    ));
    let seed = graph.tensor(TensorSpec::new(
        DataType::I64,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let offset = graph.tensor(TensorSpec::new(
        DataType::I64,
        Shape::contiguous([1, 1, 1, 1])?,
    ));

    graph.random_number_generator(
        output,
        RandomNumberGeneratorConfig::bernoulli(0.5, 7).with_seed_tensor(seed, offset),
    )?;

    assert!(
        graph
            .operations
            .iter()
            .any(|operation| matches!(operation, Operation::RandomNumberGenerator { .. }))
    );

    Ok(())
}

#[test]
fn test_rng_rejects_device_seed_with_wrong_data_type() -> Result<()> {
    let mut graph = Graph::new();
    let output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4])?,
    ));
    let seed = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let offset = graph.tensor(TensorSpec::new(
        DataType::I64,
        Shape::contiguous([1, 1, 1, 1])?,
    ));

    let err = graph
        .random_number_generator(
            output,
            RandomNumberGeneratorConfig::bernoulli(0.5, 7).with_seed_tensor(seed, offset),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorDataTypeMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == seed
            && operation == "rng seed"
            && expected == DataType::I64
            && actual == DataType::I32
    ));

    Ok(())
}

#[test]
fn test_lowered_rng_includes_device_seed_and_offset_bindings() -> Result<()> {
    let mut graph = Graph::new();
    let seed =
        graph.tensor(TensorSpec::new(DataType::I64, Shape::contiguous([1, 1, 1, 1])?).with_id(11));
    let offset =
        graph.tensor(TensorSpec::new(DataType::I64, Shape::contiguous([1, 1, 1, 1])?).with_id(22));
    let output =
        graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2, 3, 4])?).with_id(33));

    graph.random_number_generator(
        output,
        RandomNumberGeneratorConfig::bernoulli(0.5, 7).with_seed_tensor(seed, offset),
    )?;

    let lowered = LoweredGraph::lower(&graph)?;
    assert_eq!(
        lowered.binding_template_ids,
        vec![11.into(), 22.into(), 33.into()]
    );

    Ok(())
}

#[test]
fn test_rng_graph_compiles_when_plan_is_available() -> Result<()> {
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
    let output =
        graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2, 3, 4])?).with_id(33));
    graph.random_number_generator(output, RandomNumberGeneratorConfig::bernoulli(0.5, 7))?;

    match graph.compile(&context, &[HeuristicMode::Instant, HeuristicMode::Fallback]) {
        Ok(compiled) => {
            assert_eq!(compiled.tensor_record(output)?.id, 33.into());
            let _ = compiled.workspace_size()?;
        }
        Err(error) if is_expected_compile_status(&error) => {}
        Err(Error::Cuda(singe_cuda::error::Error::Cuda { code, .. }))
            if code == singe_cuda::error::Status::NoDevice => {}
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_block_scale_quantize_infer_creates_output_and_scale_shapes() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 64])?,
    ));

    let outputs = graph
        .block_scale_quantize_infer(input, BlockScaleQuantizeConfig::new(DataType::F32, 16))?;

    assert_eq!(graph.shape(outputs.output)?.dimensions(), &[2, 3, 64]);
    assert_eq!(graph.shape(outputs.scale)?.dimensions(), &[2, 3, 4]);
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
fn test_block_scale_quantize_infer_supports_transposed_layouts() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 16, 8])?,
    ));

    let outputs = graph.block_scale_quantize_infer(
        input,
        BlockScaleQuantizeConfig::new(DataType::F32, 4)
            .with_axis(2)
            .with_transpose(),
    )?;

    assert_eq!(graph.shape(outputs.output)?.dimensions(), &[2, 3, 16, 8]);
    assert_eq!(graph.shape(outputs.output)?.strides(), &[48, 16, 1, 96]);
    assert_eq!(graph.shape(outputs.scale)?.dimensions(), &[2, 3, 4, 8]);
    assert_eq!(graph.shape(outputs.scale)?.strides(), &[12, 4, 1, 24]);

    Ok(())
}

#[test]
fn test_block_scale_quantize_infer_uses_graph_intermediate_data_type() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_intermediate_data_type(DataType::BF16);
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 64])?,
    ));

    let outputs = graph
        .block_scale_quantize_infer(input, BlockScaleQuantizeConfig::new(DataType::F32, 16))?;

    assert_eq!(
        graph.tensor_config(outputs.output)?.data_type,
        DataType::BF16
    );
    assert_eq!(
        graph.tensor_config(outputs.scale)?.data_type,
        DataType::BF16
    );
    assert!(graph.tensor_config(outputs.output)?.is_virtual);
    assert!(graph.tensor_config(outputs.scale)?.is_virtual);

    Ok(())
}

#[test]
fn test_block_scale_quantize_infer_error_removes_generated_tensors() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 64])?,
    ));
    let tensor_count = graph.tensors.len();

    let error = graph
        .block_scale_quantize_infer(input, BlockScaleQuantizeConfig::new(DataType::F32, 0))
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
fn test_layer_norm_scale_bias_inference_preserves_innermost_nonunit_axis() -> Result<()> {
    let input =
        Shape::contiguous([4, 1024, 128, 1])?.with_strides(vec![1024 * 128, 128, 1, 128])?;
    let scale = infer_layer_norm_scale_bias_shape(&input)?;
    let stats = infer_layer_norm_stats_shape(&input, &scale)?;

    assert_eq!(scale.dimensions(), &[1, 1, 128, 1]);
    assert_eq!(stats.dimensions(), &[4, 1024, 1, 1]);

    Ok(())
}

#[test]
fn test_block_scale_dequantize_infer_creates_virtual_output_shape() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 3, 64])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F8E8M0,
        Shape::contiguous([2, 3, 4])?,
    ));

    let output = graph.block_scale_dequantize_infer(
        input,
        scale,
        BlockScaleDequantizeConfig::new(DataType::F32, vec![1, 1, 16]),
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[2, 3, 64]);
    assert_eq!(graph.tensor_config(output)?.data_type, DataType::F32);

    Ok(())
}

#[test]
fn test_block_scale_dequantize_infer_uses_graph_intermediate_data_type() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_intermediate_data_type(DataType::BF16);
    let input = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 3, 64])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F8E8M0,
        Shape::contiguous([2, 3, 4])?,
    ));

    let output = graph.block_scale_dequantize_infer(
        input,
        scale,
        BlockScaleDequantizeConfig::new(DataType::F32, vec![1, 1, 16]),
    )?;

    assert_eq!(graph.tensor_config(output)?.data_type, DataType::BF16);

    Ok(())
}

#[test]
fn test_block_scale_dequantize_infer_error_removes_generated_tensor() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 3, 64])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F8E8M0,
        Shape::contiguous([2, 3, 5])?,
    ));
    let tensor_count = graph.tensors.len();

    let error = graph
        .block_scale_dequantize_infer(
            input,
            scale,
            BlockScaleDequantizeConfig::new(DataType::F32, vec![1, 1, 16]),
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
            && operation == "block scale dequantize scale shape"
            && expected == vec![2, 3, 4]
            && actual == vec![2, 3, 5]
    ));
    assert_eq!(graph.tensors.len(), tensor_count);
    assert!(graph.operations.is_empty());

    Ok(())
}

#[test]
fn test_block_scale_dequantize_rejects_non_virtual_output() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 3, 64])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F8E8M0,
        Shape::contiguous([2, 3, 4])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 64])?,
    ));

    let error = graph
        .block_scale_dequantize(
            input,
            scale,
            output,
            BlockScaleDequantizeConfig::new(DataType::F32, vec![1, 1, 16]),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        Error::FrontendBindingTypeMismatch {
            tensor_id,
            expected,
            actual,
        } if tensor_id == output
            && expected == "virtual tensor"
            && actual == "bindable tensor"
    ));

    Ok(())
}

#[test]
fn test_block_scale_dequantize_rejects_mismatched_output_shape() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 3, 64])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F8E8M0,
        Shape::contiguous([2, 3, 4])?,
    ));
    let output = graph
        .tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2, 3, 63])?).virtual_tensor());

    let error = graph
        .block_scale_dequantize(
            input,
            scale,
            output,
            BlockScaleDequantizeConfig::new(DataType::F32, vec![1, 1, 16]),
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
            && operation == "block scale dequantize output shape"
            && expected == vec![2, 3, 64]
            && actual == vec![2, 3, 63]
    ));

    Ok(())
}

#[test]
fn test_block_scale_quantize_rejects_mismatched_scale_shape() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 64])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 3, 64])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F8E8M0,
        Shape::contiguous([2, 3, 8])?,
    ));

    let err = graph
        .block_scale_quantize(
            input,
            output,
            scale,
            BlockScaleQuantizeConfig::new(DataType::F32, 16),
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
            && operation == "block scale quantize scale shape"
            && expected == vec![2, 3, 4]
            && actual == vec![2, 3, 8]
    ));

    Ok(())
}

#[test]
fn test_block_scale_quantize_accepts_transposed_explicit_layouts() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 16, 8])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 3, 16, 8])?.with_strides([48, 16, 1, 96])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F8E8M0,
        Shape::contiguous([2, 3, 4, 8])?.with_strides([12, 4, 1, 24])?,
    ));

    graph.block_scale_quantize(
        input,
        output,
        scale,
        BlockScaleQuantizeConfig::new(DataType::F32, 4)
            .with_axis(2)
            .with_transpose(),
    )?;

    assert!(matches!(
        graph.operations.last(),
        Some(Operation::BlockScaleQuantize { .. })
    ));

    Ok(())
}

#[test]
fn test_block_scale_quantize_accepts_noncontiguous_explicit_layouts() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 16, 8])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 3, 16, 8])?.with_strides([400, 120, 8, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F8E8M0,
        Shape::contiguous([2, 3, 4, 8])?.with_strides([160, 48, 8, 1])?,
    ));

    graph.block_scale_quantize(
        input,
        output,
        scale,
        BlockScaleQuantizeConfig::new(DataType::F32, 4)
            .with_axis(2)
            .with_transpose(),
    )?;

    assert!(matches!(
        graph.operations.last(),
        Some(Operation::BlockScaleQuantize { .. })
    ));

    Ok(())
}

#[test]
fn test_block_scale_quantize_rejects_wrong_transposed_output_shape() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 16, 8])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 3, 8, 16])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F8E8M0,
        Shape::contiguous([2, 3, 4, 8])?.with_strides([12, 4, 1, 24])?,
    ));

    let err = graph
        .block_scale_quantize(
            input,
            output,
            scale,
            BlockScaleQuantizeConfig::new(DataType::F32, 4)
                .with_axis(2)
                .with_transpose(),
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
            && operation == "block scale quantize output shape"
            && expected == vec![2, 3, 16, 8]
            && actual == vec![2, 3, 8, 16]
    ));

    Ok(())
}

#[test]
fn test_lowp_storage_types_map_to_block_scale_fp8_data_types() {
    assert_eq!(<f8e4m3 as DataTypeLike>::data_type(), DataType::F8E4M3);
    assert_eq!(<f8ue8m0 as DataTypeLike>::data_type(), DataType::F8E8M0);
}

#[test]
fn test_matmul_accepts_noncontiguous_explicit_output_strides() -> Result<()> {
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
        DataType::F32,
        Shape::contiguous([2, 4, 16])?.with_strides([128, 32, 2])?,
    ));

    graph.matmul(a, b, c, DataType::F32)?;

    Ok(())
}

#[test]
fn test_matmul_rejects_wrong_explicit_output_shape() -> Result<()> {
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
        DataType::F32,
        Shape::contiguous([2, 5, 16])?,
    ));

    let err = graph.matmul(a, b, c, DataType::F32).unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == c
            && operation == "matmul output shape"
            && expected == vec![2, 4, 16]
            && actual == vec![2, 5, 16]
    ));

    Ok(())
}

#[test]
fn test_matmul_rejects_incompatible_contracted_dimensions() -> Result<()> {
    let mut graph = Graph::new();
    let a = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8])?,
    ));
    let b = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 7, 16])?,
    ));

    let err = graph
        .matmul_infer_with_config(a, b, MatmulConfig::new(DataType::F32))
        .unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendMatmulShapeMismatch {
            lhs_id,
            rhs_id,
            operation,
            lhs,
            rhs,
        } if lhs_id == a
            && rhs_id == b
            && operation == "matmul shapes"
            && lhs == vec![2, 4, 8]
            && rhs == vec![2, 7, 16]
    ));

    Ok(())
}

#[test]
fn test_matmul_rejects_incompatible_batch_dimensions() -> Result<()> {
    let mut graph = Graph::new();
    let a = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8])?,
    ));
    let b = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([3, 8, 16])?,
    ));

    let err = graph
        .matmul_infer_with_config(a, b, MatmulConfig::new(DataType::F32))
        .unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendMatmulShapeMismatch {
            lhs_id,
            rhs_id,
            operation,
            lhs,
            rhs,
        } if lhs_id == a
            && rhs_id == b
            && operation == "matmul shapes"
            && lhs == vec![2, 4, 8]
            && rhs == vec![3, 8, 16]
    ));

    Ok(())
}
