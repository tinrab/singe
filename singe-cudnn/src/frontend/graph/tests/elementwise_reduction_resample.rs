use super::*;

#[test]
fn test_pointwise_binary_infer_broadcasts_output_shape() -> Result<()> {
    let mut graph = Graph::new();
    let lhs = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4])?,
    ));
    let rhs = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 1])?,
    ));

    let output = graph.pointwise_binary_infer(lhs, rhs, PointwiseMode::Add, DataType::F32)?;

    assert_eq!(graph.shape(output)?.dimensions(), &[2, 3, 4]);

    Ok(())
}

#[test]
fn test_pointwise_binary_infer_broadcasts_lower_rank_rhs() -> Result<()> {
    let mut graph = Graph::new();
    let lhs = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4])?,
    ));
    let rhs = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([4])?));

    let output = graph.pointwise_binary_infer(lhs, rhs, PointwiseMode::Add, DataType::F32)?;

    assert_eq!(graph.shape(output)?.dimensions(), &[2, 3, 4]);

    Ok(())
}

#[test]
fn test_pointwise_binary_infer_reports_broadcast_shape_mismatch() -> Result<()> {
    let mut graph = Graph::new();
    let lhs = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4])?,
    ));
    let rhs = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 5, 4])?,
    ));

    let error = graph
        .pointwise_binary_infer(lhs, rhs, PointwiseMode::Add, DataType::F32)
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendPointwiseBroadcastShapeMismatch { lhs, rhs }
            if lhs == vec![2, 3, 4] && rhs == vec![2, 5, 4]
    ));

    Ok(())
}

#[test]
fn test_pointwise_prepare_reports_output_shape_mismatch() -> Result<()> {
    let mut graph = Graph::new();
    let lhs = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4])?,
    ));
    let rhs = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 5])?,
    ));
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::Add,
        lhs,
        rhs,
        output,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });

    let mut expanded = graph.expand_for_runtime(version()?.raw())?;
    expanded.expand_fused_scalar_shapes_for_pointwise()?;
    let error = expanded.validate_pointwise_operations().unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendPointwiseOutputShapeMismatch {
            operation,
            expected,
            actual,
        } if operation == "binary" && expected == vec![2, 3, 4] && actual == vec![2, 3, 5]
    ));

    Ok(())
}

#[test]
fn test_pointwise_prepare_expands_fused_scalar_rank() -> Result<()> {
    let mut graph = Graph::new();
    let lhs = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 3, 4])?,
    ));
    let rhs = graph.tensor(TensorSpec::scalar_f32(5.0)?);
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 3, 4])?,
    ));
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::Add,
        lhs,
        rhs,
        output,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });

    let mut expanded = graph.expand_for_runtime(version()?.raw())?;
    expanded.expand_fused_scalar_shapes_for_pointwise()?;

    assert_eq!(expanded.shape(rhs)?.dimensions(), &[1, 1, 1]);
    assert_eq!(expanded.shape(rhs)?.strides(), &[1, 1, 1]);

    Ok(())
}

#[test]
fn test_pointwise_prepare_rejects_unsupported_binary_alpha2() -> Result<()> {
    let mut graph = Graph::new();
    let lhs = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 3, 4])?,
    ));
    let rhs = graph.tensor(TensorSpec::scalar_f32(5.0)?);
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 3, 4])?,
    ));
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::Add,
        lhs,
        rhs,
        output,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 0.0,
    });

    let mut expanded = graph.expand_for_runtime(version()?.raw())?;
    let error = expanded
        .expand_fused_scalar_shapes_for_pointwise()
        .unwrap_err();
    assert!(matches!(
        error,
        Error::FrontendPointwiseBinaryAlpha2Unsupported { alpha2 } if alpha2 == 0.0
    ));

    Ok(())
}

#[test]
fn test_pointwise_binary_infer_uses_graph_io_data_type() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_io_data_type(DataType::BF16);
    let lhs = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4])?,
    ));
    let rhs = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4])?,
    ));

    let output = graph.pointwise_binary_infer(lhs, rhs, PointwiseMode::Add, DataType::F32)?;

    assert_eq!(graph.tensor_config(output)?.data_type, DataType::BF16);

    Ok(())
}

#[test]
fn test_reduction_infer_reduces_last_axis_shape() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4])?,
    ));

    let output = graph.reduction_infer(
        input,
        ReductionConfig::new(ReduceTensorOperator::Add, DataType::F32),
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[2, 3, 1]);

    Ok(())
}

#[test]
fn test_reduction_infer_uses_nhwc_default_strides() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 5])?,
    ));

    let output = graph.reduction_infer(
        input,
        ReductionConfig::new(ReduceTensorOperator::Add, DataType::F32),
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[2, 3, 4, 1]);
    assert_eq!(graph.shape(output)?.strides(), &[12, 1, 3, 3]);

    Ok(())
}

#[test]
fn test_reduction_infer_uses_graph_io_data_type() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_io_data_type(DataType::BF16);
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4])?,
    ));

    let output = graph.reduction_infer(
        input,
        ReductionConfig::new(ReduceTensorOperator::Add, DataType::F32),
    )?;

    assert_eq!(graph.tensor_config(output)?.data_type, DataType::BF16);

    Ok(())
}

#[test]
fn test_reduction_infer_reduces_all_axes_shape() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4])?,
    ));

    let output = graph.reduction_infer(
        input,
        ReductionConfig::new(ReduceTensorOperator::Add, DataType::F32)
            .with_axes(ReductionAxes::All),
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[1, 1, 1]);

    Ok(())
}

#[test]
fn test_reduction_support_surface_rejects_deterministic_before_cudnn_911() {
    let graph = Graph::new();
    let error = graph
        .validate_reduction_support_surface_for_version(
            91099,
            ReductionConfig::new(ReduceTensorOperator::Add, DataType::F32).with_deterministic(),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "reduction deterministic version"
    ));
}

#[test]
fn test_reduction_infer_records_deterministic_flag() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4])?,
    ));

    let _output = graph.reduction_infer(
        input,
        ReductionConfig::new(ReduceTensorOperator::Add, DataType::F32).with_deterministic(),
    )?;

    let Operation::Reduction(ReductionOperation::Reduce {
        is_deterministic, ..
    }) = &graph.operations[0]
    else {
        panic!("expected reduction operation");
    };
    assert!(*is_deterministic);

    Ok(())
}

#[test]
fn test_concat_infer_sums_dimensions_along_axis() -> Result<()> {
    let mut graph = Graph::new();
    let a = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2, 3])?));
    let b = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2, 5])?));

    let output = graph.concat_infer(&[a, b], 1)?;

    assert_eq!(graph.shape(output)?.dimensions(), &[2, 8]);

    Ok(())
}

#[test]
fn test_concat_infer_uses_graph_io_data_type() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_io_data_type(DataType::BF16);
    let a = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2, 3])?));
    let b = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2, 5])?));

    let output = graph.concat_infer(&[a, b], 1)?;

    assert_eq!(graph.tensor_config(output)?.data_type, DataType::BF16);

    Ok(())
}

#[test]
fn test_concat_infer_preserves_input_layout() -> Result<()> {
    let mut graph = Graph::new();
    let a = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4])?.with_strides([12, 1, 3])?,
    ));
    let b = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 5, 4])?.with_strides([20, 1, 5])?,
    ));

    let output = graph.concat_infer(&[a, b], 1)?;

    assert_eq!(graph.shape(output)?.dimensions(), &[2, 8, 4]);
    assert_eq!(graph.shape(output)?.strides(), &[32, 1, 8]);

    Ok(())
}

#[test]
fn test_concat_support_surface_rejects_before_cudnn_907() {
    let graph = Graph::new();
    let error = graph
        .validate_concat_support_surface_for_version(90699)
        .unwrap_err();
    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "concat version"
    ));
}

#[test]
fn test_resample_infer_computes_spatial_output_shape() -> Result<()> {
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

    assert_eq!(graph.shape(output)?.dimensions(), &[2, 3, 4, 4]);

    Ok(())
}

#[test]
fn test_resample_infer_uses_graph_io_data_type_for_output() -> Result<()> {
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

    assert_eq!(graph.tensor_config(output)?.data_type, DataType::F16);

    Ok(())
}

#[test]
fn test_resample_infer_with_indices_creates_maxpool_index_output() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_io_data_type(DataType::F16);
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8, 8])?,
    ));

    let outputs = graph.resample_infer_with_indices(
        input,
        ResampleConfig::new(ResampleMode::MaxPool, DataType::F32)
            .with_window_dims([2, 2])
            .with_strides([2, 2]),
    )?;
    let indices = outputs
        .indices
        .expect("maxpool resample should infer indices");

    assert_eq!(graph.shape(outputs.output)?.dimensions(), &[2, 3, 4, 4]);
    assert_eq!(graph.shape(indices)?.dimensions(), &[2, 3, 4, 4]);
    assert_eq!(
        graph.tensor_config(outputs.output)?.data_type,
        DataType::F16
    );
    assert_eq!(graph.tensor_config(indices)?.data_type, DataType::I8);

    Ok(())
}

#[test]
fn test_resample_infer_preserves_nspatialc_packed_layout() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([8, 8, 56, 56])?.with_strides(vec![56 * 56 * 8, 1, 56 * 8, 8])?,
    ));

    let output = graph.resample_infer(
        input,
        None,
        ResampleConfig::new(ResampleMode::MaxPool, DataType::F32)
            .with_padding_mode(PaddingMode::NegInf)
            .with_window_dims([2, 3])
            .with_strides([4, 5])
            .with_pre_paddings([2, 3])
            .with_post_paddings([4, 5]),
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[8, 8, 16, 13]);
    assert_eq!(graph.shape(output)?.strides(), &[1664, 1, 104, 8]);

    Ok(())
}

#[test]
fn test_resample_infer_with_indices_skips_index_output_for_non_maxpool_mode() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8, 8])?,
    ));

    let outputs = graph.resample_infer_with_indices(
        input,
        ResampleConfig::new(ResampleMode::AvgPoolIncludePadding, DataType::F32)
            .with_window_dims([2, 2])
            .with_strides([2, 2]),
    )?;

    assert_eq!(graph.shape(outputs.output)?.dimensions(), &[2, 3, 4, 4]);
    assert_eq!(outputs.indices, None);

    Ok(())
}

#[test]
fn test_resample_rejects_mismatched_output_shape() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8, 8])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 5, 5])?,
    ));

    let error = graph
        .resample(
            input,
            output,
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
        } if tensor_id == output
            && operation == "resample output shape"
            && expected == vec![2, 3, 4, 4]
            && actual == vec![2, 3, 5, 5]
    ));

    Ok(())
}

#[test]
fn test_resample_rejects_mismatched_output_data_type() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8, 8])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 3, 4, 4])?,
    ));

    let error = graph
        .resample(
            input,
            output,
            None,
            ResampleConfig::new(ResampleMode::MaxPool, DataType::F32)
                .with_window_dims([2, 2])
                .with_strides([2, 2]),
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
            && operation == "resample output"
            && expected == DataType::F32
            && actual == DataType::F16
    ));

    Ok(())
}

#[test]
fn test_resample_rejects_indices_for_non_maxpool_mode() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8, 8])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 4])?,
    ));
    let indices = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 4])?,
    ));

    let error = graph
        .resample(
            input,
            output,
            Some(indices),
            ResampleConfig::new(ResampleMode::AvgPoolIncludePadding, DataType::F32)
                .with_window_dims([2, 2])
                .with_strides([2, 2]),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "resample indices"
    ));

    Ok(())
}

#[test]
fn test_resample_rejects_mismatched_indices_shape() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8, 8])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 4])?,
    ));
    let indices = graph.tensor(TensorSpec::new(
        DataType::I8,
        Shape::contiguous([2, 3, 4, 5])?,
    ));

    let error = graph
        .resample(
            input,
            output,
            Some(indices),
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
        } if tensor_id == indices
            && operation == "resample indices shape"
            && expected == vec![2, 3, 4, 4]
            && actual == vec![2, 3, 4, 5]
    ));

    Ok(())
}

#[test]
fn test_resample_rejects_mismatched_indices_data_type() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8, 8])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 4])?,
    ));
    let indices = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 4])?,
    ));

    let error = graph
        .resample(
            input,
            output,
            Some(indices),
            ResampleConfig::new(ResampleMode::MaxPool, DataType::F32)
                .with_window_dims([2, 2])
                .with_strides([2, 2]),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        Error::FrontendTensorDataTypeMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == indices
            && operation == "resample indices shape"
            && expected == DataType::I8
            && actual == DataType::F32
    ));

    Ok(())
}

#[test]
fn test_resample_infer_invalid_indices_do_not_insert_output_tensor() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8, 8])?,
    ));
    let indices = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 4])?,
    ));
    let tensor_count = graph.tensors.len();

    let error = graph
        .resample_infer(
            input,
            Some(indices),
            ResampleConfig::new(ResampleMode::AvgPoolIncludePadding, DataType::F32)
                .with_window_dims([2, 2])
                .with_strides([2, 2]),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "resample indices"
    ));
    assert_eq!(graph.tensors.len(), tensor_count);
    assert!(graph.operations.is_empty());

    Ok(())
}

#[test]
fn test_resample_accepts_noncontiguous_output_and_indices_strides() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8, 8])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 4])?.with_strides([96, 32, 8, 2])?,
    ));
    let indices = graph.tensor(TensorSpec::new(
        DataType::I8,
        Shape::contiguous([2, 3, 4, 4])?.with_strides([96, 32, 8, 2])?,
    ));

    graph.resample(
        input,
        output,
        Some(indices),
        ResampleConfig::new(ResampleMode::MaxPool, DataType::F32)
            .with_window_dims(vec![2, 2])
            .with_strides([2, 2]),
    )?;

    assert!(matches!(
        graph.operations.last(),
        Some(Operation::Resample { .. })
    ));

    Ok(())
}

#[test]
fn test_resample_backward_accepts_noncontiguous_gradient_strides() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8, 8])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 4])?.with_strides([96, 32, 8, 2])?,
    ));
    let output_gradient = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 4])?.with_strides([96, 32, 8, 2])?,
    ));
    let input_gradient = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8, 8])?.with_strides([256, 80, 10, 1])?,
    ));

    graph.resample_backward(
        input,
        output,
        output_gradient,
        input_gradient,
        None,
        ResampleConfig::new(ResampleMode::MaxPool, DataType::F32)
            .with_window_dims(vec![2, 2])
            .with_strides([2, 2]),
    )?;

    assert!(matches!(
        graph.operations.last(),
        Some(Operation::ResampleBackward { .. })
    ));

    Ok(())
}

#[test]
fn test_resample_backward_rejects_mismatched_input_gradient_shape() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8, 8])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 4])?,
    ));
    let output_gradient = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 4])?,
    ));
    let input_gradient = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 7, 8])?,
    ));

    let error = graph
        .resample_backward(
            input,
            output,
            output_gradient,
            input_gradient,
            None,
            ResampleConfig::new(ResampleMode::MaxPool, DataType::F32)
                .with_window_dims(vec![2, 2])
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
        } if tensor_id == input_gradient
            && operation == "resample backward input gradient shape"
            && expected == vec![2, 3, 8, 8]
            && actual == vec![2, 3, 7, 8]
    ));

    Ok(())
}

#[test]
fn test_resample_backward_rejects_mismatched_output_gradient_shape() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8, 8])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 4])?,
    ));
    let output_gradient = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 5, 4])?,
    ));
    let input_gradient = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8, 8])?,
    ));

    let error = graph
        .resample_backward(
            input,
            output,
            output_gradient,
            input_gradient,
            None,
            ResampleConfig::new(ResampleMode::MaxPool, DataType::F32)
                .with_window_dims(vec![2, 2])
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
        } if tensor_id == output_gradient
            && operation == "resample backward output gradient shape"
            && expected == vec![2, 3, 4, 4]
            && actual == vec![2, 3, 5, 4]
    ));

    Ok(())
}

#[test]
fn test_resample_backward_rejects_mismatched_gradient_data_types() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8, 8])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 4])?,
    ));
    let output_gradient = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 3, 4, 4])?,
    ));
    let input_gradient = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8, 8])?,
    ));

    let error = graph
        .resample_backward(
            input,
            output,
            output_gradient,
            input_gradient,
            None,
            ResampleConfig::new(ResampleMode::MaxPool, DataType::F32)
                .with_window_dims(vec![2, 2])
                .with_strides([2, 2]),
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
            && operation == "resample backward output gradient"
            && expected == DataType::F32
            && actual == DataType::F16
    ));

    let output_gradient = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 4])?,
    ));
    let input_gradient = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 3, 8, 8])?,
    ));
    let error = graph
        .resample_backward(
            input,
            output,
            output_gradient,
            input_gradient,
            None,
            ResampleConfig::new(ResampleMode::MaxPool, DataType::F32)
                .with_window_dims(vec![2, 2])
                .with_strides([2, 2]),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        Error::FrontendTensorDataTypeMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == input_gradient
            && operation == "resample backward input gradient"
            && expected == DataType::F32
            && actual == DataType::F16
    ));

    Ok(())
}

#[test]
fn test_resample_backward_rejects_indices_for_non_maxpool_mode() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8, 8])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 4])?,
    ));
    let output_gradient = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 4])?,
    ));
    let input_gradient = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8, 8])?,
    ));
    let indices = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 4])?,
    ));

    let error = graph
        .resample_backward(
            input,
            output,
            output_gradient,
            input_gradient,
            Some(indices),
            ResampleConfig::new(ResampleMode::AvgPoolIncludePadding, DataType::F32)
                .with_window_dims(vec![2, 2])
                .with_strides([2, 2]),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "resample backward indices"
    ));

    Ok(())
}

#[test]
fn test_resample_backward_rejects_mismatched_indices_shape() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8, 8])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 4])?,
    ));
    let output_gradient = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 4])?,
    ));
    let input_gradient = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8, 8])?,
    ));
    let indices = graph.tensor(TensorSpec::new(
        DataType::I8,
        Shape::contiguous([2, 3, 4, 5])?,
    ));

    let error = graph
        .resample_backward(
            input,
            output,
            output_gradient,
            input_gradient,
            Some(indices),
            ResampleConfig::new(ResampleMode::MaxPool, DataType::F32)
                .with_window_dims(vec![2, 2])
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
        } if tensor_id == indices
            && operation == "resample backward indices shape"
            && expected == vec![2, 3, 4, 4]
            && actual == vec![2, 3, 4, 5]
    ));

    Ok(())
}

#[test]
fn test_resample_backward_rejects_mismatched_indices_data_type() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8, 8])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 4])?,
    ));
    let output_gradient = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 4])?,
    ));
    let input_gradient = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8, 8])?,
    ));
    let indices = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 4])?,
    ));

    let error = graph
        .resample_backward(
            input,
            output,
            output_gradient,
            input_gradient,
            Some(indices),
            ResampleConfig::new(ResampleMode::MaxPool, DataType::F32)
                .with_window_dims(vec![2, 2])
                .with_strides([2, 2]),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        Error::FrontendTensorDataTypeMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == indices
            && operation == "resample backward indices shape"
            && expected == DataType::I8
            && actual == DataType::F32
    ));

    Ok(())
}

#[test]
fn test_resample_infer_supports_3d_spatial_dims() -> Result<()> {
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

    assert_eq!(graph.shape(output)?.dimensions(), &[2, 3, 4, 4, 4]);

    Ok(())
}

#[test]
fn test_block_scale_quantize_support_surface_rejects_before_cudnn_907() {
    let graph = Graph::new();
    let error = graph
        .validate_block_scale_quantize_support_surface_for_version(90699)
        .unwrap_err();
    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "block scale quantize version"
    ));
}

#[test]
fn test_block_scale_dequantize_support_surface_rejects_before_cudnn_907() {
    let graph = Graph::new();
    let error = graph
        .validate_block_scale_dequantize_support_surface_for_version(90699)
        .unwrap_err();
    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "block scale dequantize version"
    ));
}

#[test]
fn test_adaptive_layer_norm_support_surface_rejects_before_cudnn_909() {
    let graph = Graph::new();
    let error = graph
        .validate_adaptive_layer_normalization_support_surface_for_version(90899)
        .unwrap_err();
    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "adaptive layer norm version"
    ));
}

#[test]
fn test_paged_cache_load_infer_builds_transposed_output_shape() -> Result<()> {
    let mut graph = Graph::new();
    let container = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 16, 8])?,
    ));
    let sequence = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    let page_table = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 3, 1])?,
    ));

    let output = graph.paged_cache_load_infer(container, sequence, page_table, true)?;

    assert_eq!(graph.shape(output)?.dimensions(), &[2, 4, 8, 48]);
    assert_eq!(graph.shape(output)?.strides(), &[1536, 384, 1, 8]);
    assert_eq!(graph.tensor_config(output)?.data_type, DataType::F16);
    assert!(graph.tensor_config(output)?.is_virtual);

    Ok(())
}

#[test]
fn test_paged_cache_load_infer_uses_graph_intermediate_data_type() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_intermediate_data_type(DataType::BF16);
    let container = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 16, 8])?,
    ));
    let sequence = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    let page_table = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 3, 1])?,
    ));

    let output = graph.paged_cache_load_infer(container, sequence, page_table, true)?;

    assert_eq!(graph.tensor_config(output)?.data_type, DataType::BF16);
    assert!(graph.tensor_config(output)?.is_virtual);

    Ok(())
}

#[test]
fn test_paged_cache_load_infer_error_removes_generated_output() -> Result<()> {
    let mut graph = Graph::new();
    let container = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 16, 8])?,
    ));
    let sequence = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    let page_table = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 3, 1])?,
    ));
    let tensor_count = graph.tensors.len();

    let error = graph
        .paged_cache_load_infer(container, sequence, page_table, true)
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorDataTypeUnsupported {
            tensor_id,
            operation,
            supported,
            actual,
        } if tensor_id == sequence
            && operation == "paged cache load sequence"
            && supported == vec![DataType::I32, DataType::I64]
            && actual == DataType::F32
    ));
    assert_eq!(graph.tensors.len(), tensor_count);
    assert!(graph.operations.is_empty());

    Ok(())
}

#[test]
fn test_paged_cache_load_rejects_non_integer_sequence() -> Result<()> {
    let mut graph = Graph::new();
    let container = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 16, 8])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 48])?.with_strides([1536, 384, 1, 8])?,
    ));
    let sequence = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    let page_table = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 3, 1])?,
    ));

    let err = graph
        .paged_cache_load(container, output, sequence, page_table)
        .unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorDataTypeUnsupported {
            tensor_id,
            operation,
            supported,
            actual,
        } if tensor_id == sequence
            && operation == "paged cache load sequence"
            && supported == vec![DataType::I32, DataType::I64]
            && actual == DataType::F32
    ));

    Ok(())
}

#[test]
fn test_paged_cache_load_rejects_non_integer_page_table() -> Result<()> {
    let mut graph = Graph::new();
    let container = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 16, 8])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 48])?.with_strides([1536, 384, 1, 8])?,
    ));
    let sequence = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    let page_table = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 1, 3, 1])?,
    ));

    let err = graph
        .paged_cache_load(container, output, sequence, page_table)
        .unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorDataTypeUnsupported {
            tensor_id,
            operation,
            supported,
            actual,
        } if tensor_id == page_table
            && operation == "paged cache load page table"
            && supported == vec![DataType::I32, DataType::I64]
            && actual == DataType::F32
    ));

    Ok(())
}

#[test]
fn test_paged_cache_load_rejects_mismatched_output_data_type() -> Result<()> {
    let mut graph = Graph::new();
    let container = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 16, 8])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 48])?.with_strides([1536, 384, 1, 8])?,
    ));
    let sequence = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    let page_table = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 3, 1])?,
    ));

    let err = graph
        .paged_cache_load(container, output, sequence, page_table)
        .unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorDataTypeMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == output
            && operation == "paged cache load output"
            && expected == DataType::F16
            && actual == DataType::F32
    ));

    Ok(())
}

#[test]
fn test_paged_cache_load_uses_graph_intermediate_data_type_for_explicit_output() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_intermediate_data_type(DataType::BF16);
    let container = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 16, 8])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([2, 4, 8, 48])?.with_strides([1536, 384, 1, 8])?,
    ));
    let sequence = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    let page_table = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 3, 1])?,
    ));

    graph.paged_cache_load(container, output, sequence, page_table)?;

    assert!(matches!(
        graph.operations.last(),
        Some(Operation::PagedCacheLoad { .. })
    ));

    Ok(())
}

#[test]
fn test_paged_cache_load_accepts_i64_sequence() -> Result<()> {
    let mut graph = Graph::new();
    let container = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 16, 8])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 48])?.with_strides([1536, 384, 1, 8])?,
    ));
    let sequence = graph.tensor(TensorSpec::new(
        DataType::I64,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    let page_table = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 3, 1])?,
    ));

    graph.paged_cache_load(container, output, sequence, page_table)?;

    Ok(())
}

#[test]
fn test_paged_cache_load_support_surface_rejects_before_cudnn_905() {
    let graph = Graph::new();
    let error = graph
        .validate_paged_cache_load_support_surface_for_version(90499)
        .unwrap_err();
    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "paged cache load version"
    ));
}
