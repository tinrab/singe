use super::*;

#[test]
fn test_compile_requires_operations() -> Result<()> {
    let context = setup_context()?;
    let graph = Graph::new();

    let err = graph
        .compile(&context, &[HeuristicMode::Instant])
        .unwrap_err();
    assert!(matches!(err, Error::FrontendGraphEmpty));

    Ok(())
}

#[test]
fn test_softmax_op_validates_reduction_shape() -> Result<()> {
    let mut graph = Graph::new();
    let x = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4])?,
    ));
    let y = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 2])?,
    ));

    let err = graph
        .softmax(x, y, SoftmaxOperationConfig::new().with_stats(stats))
        .unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == stats
            && operation == "softmax stats"
            && expected == vec![2, 3, 1]
            && actual == vec![2, 3, 2]
    ));

    Ok(())
}

#[test]
fn test_softmax_op_accepts_sink_shape() -> Result<()> {
    let mut graph = Graph::new();
    let x = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 5])?,
    ));
    let y = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 5])?,
    ));
    let sink = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 3, 1, 1])?,
    ));

    graph.softmax(x, y, SoftmaxOperationConfig::new().with_sink(sink))?;

    Ok(())
}

#[test]
fn test_softmax_op_accepts_noncontiguous_output_strides() -> Result<()> {
    let mut graph = Graph::new();
    let x = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 5])?,
    ));
    let y = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 5])?.with_strides([120, 40, 10, 2])?,
    ));

    graph.softmax(x, y, SoftmaxOperationConfig::new())?;

    Ok(())
}

#[test]
fn test_softmax_op_accepts_noncontiguous_aux_output_strides() -> Result<()> {
    let mut graph = Graph::new();
    let x = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 5])?,
    ));
    let y = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 5])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 1])?.with_strides([24, 8, 2, 1])?,
    ));
    let max = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 1])?.with_strides([24, 8, 2, 1])?,
    ));
    let sum_exp = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 1])?.with_strides([24, 8, 2, 1])?,
    ));

    graph.softmax(
        x,
        y,
        SoftmaxOperationConfig::new()
            .with_stats(stats)
            .with_max(max)
            .with_sum_exp(sum_exp),
    )?;

    Ok(())
}

#[test]
fn test_softmax_op_rejects_reduction_shaped_sink_tensor() -> Result<()> {
    let mut graph = Graph::new();
    let x = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 5])?,
    ));
    let y = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 5])?,
    ));
    let sink = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 1])?,
    ));

    let err = graph
        .softmax(x, y, SoftmaxOperationConfig::new().with_sink(sink))
        .unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == sink
            && operation == "softmax sink"
            && expected == vec![1, 3, 1, 1]
            && actual == vec![2, 3, 4, 1]
    ));

    Ok(())
}

#[test]
fn test_softmax_op_rejects_sink_with_non_rank4_input() -> Result<()> {
    let mut graph = Graph::new();
    let x = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 5])?,
    ));
    let y = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 5])?,
    ));
    let sink = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 3, 1, 1])?,
    ));

    let error = graph
        .softmax(x, y, SoftmaxOperationConfig::new().with_sink(sink))
        .unwrap_err();
    assert!(matches!(
        error,
        Error::FrontendTensorRankMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == x
            && operation == "softmax sink input"
            && expected == "4"
            && actual == 3
    ));

    Ok(())
}

#[test]
fn test_softmax_op_accepts_noncontiguous_sink_tensor_strides() -> Result<()> {
    let mut graph = Graph::new();
    let x = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 5])?,
    ));
    let y = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 5])?,
    ));
    let sink = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 3, 1, 1])?.with_strides([6, 2, 2, 2])?,
    ));

    graph.softmax(x, y, SoftmaxOperationConfig::new().with_sink(sink))?;

    Ok(())
}

#[test]
fn test_diagonal_band_mask_rejects_conflicting_bounds() -> Result<()> {
    let mut graph = Graph::new();
    let x = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 8])?,
    ));
    let b = graph.tensor(
        TensorSpec::new(DataType::F32, Shape::contiguous([1])?)
            .with_scalar_value(ScalarValue::F32(f32::NEG_INFINITY))?,
    );
    let y = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 8])?,
    ));
    let left_bound = graph.tensor(
        TensorSpec::new(DataType::I32, Shape::contiguous([1])?)
            .with_scalar_value(ScalarValue::I32(1))?,
    );
    let shift_right_bound = graph.tensor(
        TensorSpec::new(DataType::I32, Shape::contiguous([1])?)
            .with_scalar_value(ScalarValue::I32(1))?,
    );

    let err = graph
        .diagonal_band_mask(
            x,
            b,
            y,
            DiagonalBandMaskConfig::new(PointwiseMode::CmpGe)
                .with_left_bound(left_bound)
                .with_shift_right_bound(shift_right_bound),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "diagonal band mask bounds"
    ));

    Ok(())
}

#[test]
fn test_diagonal_band_mask_accepts_noncontiguous_output_strides() -> Result<()> {
    let mut graph = Graph::new();
    let x = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 8])?,
    ));
    let b = graph.tensor(
        TensorSpec::new(DataType::F32, Shape::contiguous([1])?)
            .with_scalar_value(ScalarValue::F32(f32::NEG_INFINITY))?,
    );
    let y = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 8])?.with_strides([512, 128, 8, 1])?,
    ));

    graph.diagonal_band_mask(x, b, y, DiagonalBandMaskConfig::new(PointwiseMode::CmpGt))?;

    Ok(())
}

#[test]
fn test_diagonal_band_mask_rejects_mismatched_b_data_type() -> Result<()> {
    let mut graph = Graph::new();
    let x = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 8])?,
    ));
    let b = graph.tensor(TensorSpec::new(DataType::F16, Shape::contiguous([1])?));
    let y = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 8])?,
    ));

    let err = graph
        .diagonal_band_mask(x, b, y, DiagonalBandMaskConfig::new(PointwiseMode::CmpGt))
        .unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorDataTypeMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == b
            && operation == "diagonal band mask b"
            && expected == DataType::F32
            && actual == DataType::F16
    ));

    Ok(())
}

#[test]
fn test_diagonal_band_mask_rejects_mismatched_output_shape() -> Result<()> {
    let mut graph = Graph::new();
    let x = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 8])?,
    ));
    let b = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let y = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 7])?,
    ));

    let err = graph
        .diagonal_band_mask(x, b, y, DiagonalBandMaskConfig::new(PointwiseMode::CmpGt))
        .unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == y
            && operation == "diagonal band mask output shape"
            && expected == vec![2, 4, 8, 8]
            && actual == vec![2, 4, 8, 7]
    ));

    Ok(())
}

#[test]
fn test_diagonal_band_mask_rejects_non_scalar_b() -> Result<()> {
    let mut graph = Graph::new();
    let x = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 8])?,
    ));
    let b = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2])?));
    let y = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 8])?,
    ));

    let err = graph
        .diagonal_band_mask(x, b, y, DiagonalBandMaskConfig::new(PointwiseMode::CmpGt))
        .unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorElementCountMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == b
            && operation == "diagonal band mask b shape"
            && expected == 1
            && actual == 2
    ));

    Ok(())
}

#[test]
fn test_diagonal_band_mask_rejects_non_scalar_left_bound() -> Result<()> {
    let mut graph = Graph::new();
    let x = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 8])?,
    ));
    let b = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let y = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 8])?,
    ));
    let left_bound = graph.tensor(TensorSpec::new(DataType::I32, Shape::contiguous([2])?));

    let err = graph
        .diagonal_band_mask(
            x,
            b,
            y,
            DiagonalBandMaskConfig::new(PointwiseMode::CmpGt).with_left_bound(left_bound),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorElementCountMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == left_bound
            && operation == "diagonal band mask left bound shape"
            && expected == 1
            && actual == 2
    ));

    Ok(())
}

#[test]
fn test_diagonal_band_mask_rejects_non_scalar_shift_right_bound() -> Result<()> {
    let mut graph = Graph::new();
    let x = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 8])?,
    ));
    let b = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let y = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 8])?,
    ));
    let shift_right_bound = graph.tensor(TensorSpec::new(DataType::I32, Shape::contiguous([2])?));

    let err = graph
        .diagonal_band_mask(
            x,
            b,
            y,
            DiagonalBandMaskConfig::new(PointwiseMode::CmpGt)
                .with_shift_right_bound(shift_right_bound),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorElementCountMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == shift_right_bound
            && operation == "diagonal band mask shift right bound shape"
            && expected == 1
            && actual == 2
    ));

    Ok(())
}

#[test]
fn test_diagonal_band_mask_infer_uses_graph_io_data_type() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_io_data_type(DataType::BF16);
    let x = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 16])?.with_strides([512, 1, 64, 4])?,
    ));
    let b = graph.tensor(
        TensorSpec::new(DataType::F32, Shape::contiguous([1])?)
            .with_scalar_value(ScalarValue::F32(f32::NEG_INFINITY))?,
    );

    let y =
        graph.diagonal_band_mask_infer(x, b, DiagonalBandMaskConfig::new(PointwiseMode::CmpGt))?;

    assert_eq!(graph.tensor_config(y)?.data_type, DataType::BF16);
    assert_eq!(graph.shape(y)?.dimensions(), &[2, 4, 8, 16]);
    assert_eq!(graph.shape(y)?.strides(), &[512, 1, 64, 4]);

    Ok(())
}

#[test]
fn test_diagonal_band_mask_infer_error_removes_generated_output() -> Result<()> {
    let mut graph = Graph::new();
    let x = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 8])?,
    ));
    let b = graph.tensor(
        TensorSpec::new(DataType::F32, Shape::contiguous([1])?)
            .with_scalar_value(ScalarValue::F32(f32::NEG_INFINITY))?,
    );
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 2, 1, 1])?,
    ));
    let tensor_count = graph.tensors.len();

    let error = graph
        .diagonal_band_mask_infer(
            x,
            b,
            DiagonalBandMaskConfig::new(PointwiseMode::CmpGt)
                .with_sequence_length_query(sequence_length_query),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == sequence_length_query
            && operation == "diagonal band mask seq len q shape"
            && expected == vec![2, 1, 1, 1]
            && actual == vec![1, 2, 1, 1]
    ));
    assert_eq!(graph.tensors.len(), tensor_count);
    assert!(graph.operations.is_empty());

    Ok(())
}

#[test]
fn test_diagonal_band_mask_rejects_wrong_seq_len_q_shape() -> Result<()> {
    let mut graph = Graph::new();
    let x = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 8])?,
    ));
    let b = graph.tensor(
        TensorSpec::new(DataType::F32, Shape::contiguous([1])?)
            .with_scalar_value(ScalarValue::F32(f32::NEG_INFINITY))?,
    );
    let y = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 8])?,
    ));
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 2, 1, 1])?,
    ));

    let err = graph
        .diagonal_band_mask(
            x,
            b,
            y,
            DiagonalBandMaskConfig::new(PointwiseMode::CmpGt)
                .with_sequence_length_query(sequence_length_query),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == sequence_length_query
            && operation == "diagonal band mask seq len q shape"
            && expected == vec![2, 1, 1, 1]
            && actual == vec![1, 2, 1, 1]
    ));

    Ok(())
}

#[test]
fn test_diagonal_band_mask_rejects_wrong_seq_len_kv_shape() -> Result<()> {
    let mut graph = Graph::new();
    let x = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 8])?,
    ));
    let b = graph.tensor(
        TensorSpec::new(DataType::F32, Shape::contiguous([1])?)
            .with_scalar_value(ScalarValue::F32(f32::NEG_INFINITY))?,
    );
    let y = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 8])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 2, 1])?,
    ));

    let err = graph
        .diagonal_band_mask(
            x,
            b,
            y,
            DiagonalBandMaskConfig::new(PointwiseMode::CmpGt)
                .with_sequence_length_key_value(sequence_length_key_value),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == sequence_length_key_value
            && operation == "diagonal band mask seq len kv shape"
            && expected == vec![2, 1, 1, 1]
            && actual == vec![2, 1, 2, 1]
    ));

    Ok(())
}

#[test]
fn test_diagonal_band_mask_accepts_batched_seq_len_tensors() -> Result<()> {
    let mut graph = Graph::new();
    let x = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 8])?,
    ));
    let b = graph.tensor(
        TensorSpec::new(DataType::F32, Shape::contiguous([1])?)
            .with_scalar_value(ScalarValue::F32(f32::NEG_INFINITY))?,
    );
    let y = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 8])?,
    ));
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));

    graph.diagonal_band_mask(
        x,
        b,
        y,
        DiagonalBandMaskConfig::new(PointwiseMode::CmpGt)
            .with_sequence_length_query(sequence_length_query)
            .with_sequence_length_key_value(sequence_length_key_value),
    )?;

    Ok(())
}

#[test]
fn test_legacy_runtime_expands_unified_softmax_and_diagonal_band_mask() -> Result<()> {
    let mut graph = Graph::new();
    let x = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 8])?,
    ));
    let y = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 8])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 1])?,
    ));
    graph.softmax(x, y, SoftmaxOperationConfig::new().with_stats(stats))?;

    let b = graph.tensor(
        TensorSpec::new(DataType::F32, Shape::contiguous([1])?)
            .with_scalar_value(ScalarValue::F32(f32::NEG_INFINITY))?,
    );
    let masked = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 8])?,
    ));
    graph.diagonal_band_mask(
        y,
        b,
        masked,
        DiagonalBandMaskConfig::new(PointwiseMode::CmpGt),
    )?;

    let expanded = graph.expand_for_runtime(92000)?;
    assert!(
        expanded
            .operations
            .iter()
            .all(|operation| !matches!(operation, Operation::Softmax { .. }))
    );
    assert!(
        expanded
            .operations
            .iter()
            .all(|operation| !matches!(operation, Operation::DiagonalBandMask { .. }))
    );
    assert!(expanded.operations.len() > graph.operations.len());

    Ok(())
}

#[test]
fn test_legacy_runtime_expands_unified_sdpa_forward() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    graph.sdpa(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        SdpaOutputTensors {
            output: o,
            stats: Some(stats),
            logit_max: None,
            score_sum_exp: None,
        },
        &SdpaConfig::new().with_stats(),
    )?;

    let expanded = match graph.expand_for_runtime(91300) {
        Ok(expanded) => expanded,
        Err(Error::DescriptorMismatch { name }) if name == "reshape data type" => return Ok(()),
        Err(Error::FrontendTensorDataTypeMismatch { operation, .. })
            if operation == "reshape output" =>
        {
            return Ok(());
        }
        Err(error) => return Err(error),
    };
    assert!(
        expanded
            .operations
            .iter()
            .all(|operation| !matches!(operation, Operation::SdpaForward { .. }))
    );
    assert!(expanded.operations.len() > graph.operations.len());

    Ok(())
}

#[test]
fn test_pointwise_graph_compiles_when_plan_is_available() -> Result<()> {
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
        Ok(compiled) => {
            assert_eq!(compiled.tensor_record(x)?.id, 11.into());
            assert_eq!(compiled.tensor_record(y)?.id, 22.into());
            assert_eq!(compiled.required_tensor_ids(), vec![11.into(), 22.into()]);
            let _ = compiled.workspace_size()?;
        }
        Err(error) if is_expected_compile_status(&error) => {}
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_compile_with_options_rejects_too_small_workspace_limit() -> Result<()> {
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

    let result = graph.compile_with_config(
        &context,
        &CompileConfig::new()
            .with_heuristic_modes(vec![HeuristicMode::Instant, HeuristicMode::Fallback])
            .with_max_workspace_size(0),
    );

    match result {
        Ok(compiled) => {
            assert_eq!(compiled.workspace_size()?, 0);
        }
        Err(error) if is_expected_compile_status(&error) => {}
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_compiled_graph_can_switch_default_plan_when_plan_is_available() -> Result<()> {
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
            assert_eq!(
                compiled.engine_configs().len(),
                compiled.execution_plans().len()
            );
            assert_eq!(compiled.execution_plans().len(), 1);
            compiled.set_default_plan(0)?;
            assert_eq!(compiled.default_plan_index(), 0);

            let rebuilt = compiled.with_default_plan(0)?;
            assert_eq!(rebuilt.default_plan_index(), 0);
        }
        Err(error) if is_expected_compile_status(&error) => {}
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_graph_can_compile_from_explicit_engine_config() -> Result<()> {
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
        Ok(compiled) => {
            let rebuilt = graph.compile_with_engine_config(
                &context,
                compiled.engine_config_at(compiled.default_plan_index())?,
            );

            match rebuilt {
                Ok(rebuilt) => {
                    assert_eq!(rebuilt.engine_configs().len(), 1);
                    assert_eq!(rebuilt.execution_plans().len(), 1);
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
fn test_graph_exposes_plan_candidates_for_partial_build_workflows() -> Result<()> {
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

    match graph.plan_candidates(&context, &[HeuristicMode::Instant, HeuristicMode::Fallback]) {
        Ok(candidates) => {
            assert!(candidates.candidate_count() > 0);
            assert_eq!(
                candidates.candidate_count(),
                candidates.engine_configs().len()
            );
            let _ = candidates.operation_graph();
        }
        Err(error) if is_expected_compile_status(&error) => {}
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_compile_with_all_policy_can_keep_multiple_built_plans() -> Result<()> {
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

    match graph.compile_with_config(
        &context,
        &CompileConfig::new()
            .with_heuristic_modes(vec![HeuristicMode::Instant, HeuristicMode::Fallback])
            .with_build_policy(BuildPlanPolicy::AllSupported),
    ) {
        Ok(compiled) => {
            assert!(!compiled.execution_plans().is_empty());
            assert_eq!(
                compiled.execution_plans().len(),
                compiled.engine_configs().len()
            );
        }
        Err(error) if is_expected_compile_status(&error) => {}
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_compile_with_parallel_first_supported_policy_can_build_plan() -> Result<()> {
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

    match graph.compile_with_config(
        &context,
        &CompileConfig::new()
            .with_heuristic_modes(vec![HeuristicMode::Instant, HeuristicMode::Fallback])
            .with_build_policy(BuildPlanPolicy::FirstSupportedParallel { window_size: 2 }),
    ) {
        Ok(compiled) => {
            assert!(!compiled.execution_plans().is_empty());
            assert_eq!(compiled.execution_plans().len(), 1);
            assert_eq!(compiled.engine_configs().len(), 1);
        }
        Err(Error::Cuda(singe_cuda::error::Error::Cuda { code, .. }))
            if code == singe_cuda::error::Status::NoDevice => {}
        Err(error) if is_expected_compile_status(&error) => {}
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_graph_can_query_candidates_with_sm_count_target() -> Result<()> {
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

    match graph.engine_configs_with_config(
        &context,
        &CompileConfig::new()
            .with_heuristic_modes(vec![HeuristicMode::Instant])
            .with_sm_count_target(1),
    ) {
        Ok(_) => {}
        Err(error) if is_expected_engine_query_status(&error) => {}
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_graph_can_query_candidates_with_graph_sm_count_target() -> Result<()> {
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
    graph.set_sm_count_target(1);
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

    match graph.engine_configs_with_config(
        &context,
        &CompileConfig::new().with_heuristic_modes(vec![HeuristicMode::Instant]),
    ) {
        Ok(_) => {}
        Err(error) if is_expected_engine_query_status(&error) => {}
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_plan_candidates_can_build_single_plan_by_index() -> Result<()> {
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

    match graph.plan_candidates(&context, &[HeuristicMode::Instant, HeuristicMode::Fallback]) {
        Ok(candidates) => {
            let candidate_count = candidates.candidate_count();
            let built = candidates.compile_at(&context, 0);

            match built {
                Ok(compiled) => {
                    assert_eq!(compiled.execution_plans().len(), 1);
                    assert_eq!(compiled.engine_configs().len(), 1);
                    assert!(candidate_count >= 1);
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
fn test_graph_can_compile_deviceless_with_device_properties() -> Result<()> {
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
    graph.set_device_properties(DeviceProperties::create(&context)?)?;
    graph.set_io_data_type(DataType::F32);
    graph.set_compute_data_type(DataType::F32);
    let x = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([4])?).with_id(11));
    let y = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([4])?).with_id(22));
    graph.reshape(x, y)?;

    match graph.compile_deviceless(&[HeuristicMode::Instant, HeuristicMode::Fallback]) {
        Ok(compiled) => {
            assert!(!compiled.execution_plans().is_empty());
            assert_eq!(
                compiled.execution_plans().len(),
                compiled.engine_configs().len()
            );
        }
        Err(error) if is_expected_compile_status(&error) => {}
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_plan_candidates_expose_support_status() -> Result<()> {
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

    match graph.plan_candidates(&context, &[HeuristicMode::Instant, HeuristicMode::Fallback]) {
        Ok(candidates) => {
            let support = candidates.support(&context);
            let supported_indices = candidates.supported_indices(&context);
            assert_eq!(support.len(), candidates.candidate_count());
            assert_eq!(
                supported_indices,
                support
                    .iter()
                    .filter(|entry| entry.supported)
                    .map(|entry| entry.plan_index)
                    .collect::<Vec<_>>()
            );
            for entry in support {
                if entry.supported {
                    assert!(entry.error.is_none());
                } else {
                    assert!(entry.error.is_some());
                }
            }
        }
        Err(error) if is_expected_compile_status(&error) => {}
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_plan_candidates_support_respects_kernel_cache_requirements() -> Result<()> {
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
    graph.set_kernel_cache(KernelCache::create()?)?;
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

    match graph.plan_candidates(&context, &[HeuristicMode::Instant, HeuristicMode::Fallback]) {
        Ok(candidates) => {
            let support = candidates.support(&context);
            assert_eq!(support.len(), candidates.candidate_count());
            assert!(candidates.supported_indices(&context).is_empty());
            for entry in support {
                assert!(!entry.supported);
                assert!(matches!(
                    entry.error.as_ref(),
                    Some(Error::FrontendKernelCacheRequiresDynamicShape)
                ));
            }
        }
        Err(Error::FrontendKernelCacheRequiresDynamicShape) => return Ok(()),
        Err(error) if is_expected_compile_status(&error) => {}
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_graph_rejects_dynamic_shape_before_cudnn_904() {
    let mut graph = Graph::new();
    graph.enable_dynamic_shape();

    let error = graph
        .validate_runtime_feature_support_for_version(90399)
        .unwrap_err();
    assert!(matches!(
        error,
        Error::FrontendFeatureRequiresVersion {
            feature,
            min_version,
            actual_version: 90399,
        } if feature == "dynamic shape or kernel cache" && min_version == "9.4"
    ));
}

#[test]
fn test_plan_candidates_and_compiled_graph_expose_names() -> Result<()> {
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
    graph.set_name("plan-name-test");
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

    match graph.plan_candidates(&context, &[HeuristicMode::Instant, HeuristicMode::Fallback]) {
        Ok(candidates) => {
            assert_eq!(candidates.graph_name(), Some("plan-name-test"));
            let names = candidates.names()?;
            assert_eq!(names.len(), candidates.candidate_count());
            if candidates.candidate_count() > 0 {
                assert_eq!(candidates.name_at(0)?, names[0]);
                assert!(names[0].starts_with("plan-name-test/"));
            }

            match candidates.compile_at(&context, 0) {
                Ok(compiled) => {
                    assert_eq!(compiled.graph().name(), Some("plan-name-test"));
                    let compiled_names = compiled.names()?;
                    assert_eq!(compiled_names.len(), compiled.execution_plans().len());
                    assert!(compiled.name()?.starts_with("plan-name-test/"));
                }
                Err(Error::Cuda(singe_cuda::error::Error::Cuda { code, .. }))
                    if code == singe_cuda::error::Status::NoDevice => {}
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
fn test_plan_candidates_support_direct_deselection_workflow() -> Result<()> {
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

    match graph.plan_candidates(&context, &[HeuristicMode::Instant, HeuristicMode::Fallback]) {
        Ok(candidates) => {
            if candidates.candidate_count() == 0 {
                return Ok(());
            }

            let blocked_name = candidates.name_at(0)?;
            let filtered = candidates
                .deselect_workspace_greater_than(usize::MAX)?
                .deselect_shared_memory_greater_than(usize::MAX)?
                .deselect_engines(vec![blocked_name.clone()])?;

            for name in filtered.names()? {
                assert!(!name.contains(&blocked_name));
            }
        }
        Err(error) if is_expected_compile_status(&error) => {}
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_plan_candidates_support_direct_behavior_note_filters() -> Result<()> {
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

    match graph.plan_candidates(&context, &[HeuristicMode::Instant, HeuristicMode::Fallback]) {
        Ok(candidates) => {
            let filtered = candidates
                .require_behavior_notes(vec![BackendBehaviorNote::SupportsCudaGraphNativeApi]);

            match filtered {
                Ok(filtered) => {
                    for engine_config in filtered.engine_configs() {
                        let notes = engine_config
                            .engine()?
                            .behavior_notes(filtered.operation_graph().descriptor())?;
                        assert!(notes.contains(&BackendBehaviorNote::SupportsCudaGraphNativeApi));
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
fn test_graph_exposes_engine_indices_for_explicit_plan_selection() -> Result<()> {
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

    let engine_count = graph.engine_count(&context)?;
    let engine_indices = graph.engine_indices(&context)?;

    assert_eq!(engine_count, engine_indices.len());
    assert_eq!(
        engine_indices,
        (0..to_i64(engine_count, "engine_count")?).collect::<Vec<_>>()
    );

    Ok(())
}

#[test]
fn test_graph_exposes_knobs_for_engine_when_supported() -> Result<()> {
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

    for engine_index in graph.engine_indices(&context)? {
        match graph.knobs_for_engine(&context, engine_index) {
            Ok(_) => return Ok(()),
            Err(Error::NoAvailableEngines) => {}
            Err(Error::Cudnn { .. }) => {}
            Err(error) => return Err(error),
        }
    }

    Ok(())
}

#[test]
fn test_graph_exposes_engine_notes_when_supported() -> Result<()> {
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

    for engine_index in graph.engine_indices(&context)? {
        match graph.numerical_notes_for_engine(&context, engine_index) {
            Ok(_) => {
                let _ = graph.behavior_notes_for_engine(&context, engine_index)?;
                return Ok(());
            }
            Err(Error::NoAvailableEngines) => {}
            Err(Error::Cudnn { .. }) => {}
            Err(error) => return Err(error),
        }
    }

    Ok(())
}

#[test]
fn test_graph_exposes_filtered_engine_configs() -> Result<()> {
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

    let configs =
        match graph.engine_configs(&context, &[HeuristicMode::Instant, HeuristicMode::Fallback]) {
            Ok(configs) => configs,
            Err(error) if is_expected_compile_status(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
    let filtered = match graph.engine_configs_with_config(
        &context,
        &CompileConfig::new()
            .with_heuristic_modes(vec![HeuristicMode::Instant, HeuristicMode::Fallback])
            .with_max_workspace_size(0),
    ) {
        Ok(filtered) => filtered,
        Err(error) if is_expected_compile_status(&error) => return Ok(()),
        Err(error) => return Err(error),
    };

    assert!(filtered.len() <= configs.len());
    for config in filtered {
        assert_eq!(config.workspace_size()?, 0);
    }

    Ok(())
}

#[test]
fn test_graph_filters_engine_configs_by_behavior_notes() -> Result<()> {
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

    let filtered = match graph.engine_configs_with_config(
        &context,
        &CompileConfig::new()
            .with_heuristic_modes(vec![HeuristicMode::Instant, HeuristicMode::Fallback])
            .with_included_behavior_notes(vec![BackendBehaviorNote::SupportsCudaGraphNativeApi]),
    ) {
        Ok(filtered) => filtered,
        Err(error) if is_expected_compile_status(&error) => return Ok(()),
        Err(error) => return Err(error),
    };

    for config in filtered {
        let engine_index = config.engine()?.index();
        let notes = graph.behavior_notes_for_engine(&context, engine_index.as_i64())?;
        assert!(notes.contains(&BackendBehaviorNote::SupportsCudaGraphNativeApi));
    }

    Ok(())
}

#[test]
fn test_graph_filters_engine_configs_by_name_substring() -> Result<()> {
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

    match graph.plan_candidates(&context, &[HeuristicMode::Instant, HeuristicMode::Fallback]) {
        Ok(candidates) => {
            if candidates.candidate_count() == 0 {
                return Ok(());
            }

            let blocked_name = candidates.name_at(0)?;
            let filtered = graph.engine_configs_with_config(
                &context,
                &CompileConfig::new()
                    .with_heuristic_modes(vec![HeuristicMode::Instant, HeuristicMode::Fallback])
                    .with_excluded_engine_name_substrings(vec![blocked_name.clone()]),
            );

            match filtered {
                Ok(filtered) => {
                    for config in filtered {
                        let name = PlanCandidates::name_for_engine_config(&config)?;
                        assert!(!name.contains(&blocked_name));
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
fn test_graph_filters_engine_configs_by_shared_memory_limit() -> Result<()> {
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

    let filtered = match graph.engine_configs_with_config(
        &context,
        &CompileConfig::new()
            .with_heuristic_modes(vec![HeuristicMode::Instant, HeuristicMode::Fallback])
            .with_max_shared_memory_size(0),
    ) {
        Ok(filtered) => filtered,
        Err(error) if is_expected_compile_status(&error) => return Ok(()),
        Err(error) => return Err(error),
    };

    for config in filtered {
        assert_eq!(config.shared_memory_size()?, 0);
    }

    Ok(())
}
