use super::*;
use crate::{
    attribute::{BackendAttributeName, BackendAttributeType},
    descriptor::{BackendDescriptor, BackendDescriptorType},
    frontend::{
        composite::sdpa::{SdpaBackwardOutputs, SdpaOutputs},
        operation::Operation,
        support::SdpaBackwardSupport,
    },
};

#[test]
fn test_sdpa_backward_direct_infer_creates_gradient_shapes() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);

    let SdpaBackwardOutputs {
        query_gradient: d_q,
        key_gradient: d_k,
        value_gradient: d_v,
        bias_gradient: _,
    } = graph.sdpa_backward_direct_infer(
        SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
        DirectSdpaBackwardConfig::new(),
    )?;

    assert_eq!(graph.shape(d_q)?.dimensions(), &[2, 4, 8, 16]);
    assert_eq!(graph.shape(d_k)?.dimensions(), &[2, 4, 8, 16]);
    assert_eq!(graph.shape(d_v)?.dimensions(), &[2, 4, 8, 16]);

    Ok(())
}

#[test]
fn test_sdpa_backward_direct_accepts_sequence_lengths() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));

    let _ = graph.sdpa_backward_direct_infer(
        SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
        DirectSdpaBackwardConfig::new()
            .with_sequence_lengths(sequence_length_query, sequence_length_key_value),
    )?;

    Ok(())
}

#[test]
fn test_sdpa_backward_direct_rejects_wrong_seq_len_data_type() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let d_q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));

    let error = graph
        .sdpa_backward_direct(
            SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
            SdpaBackwardGradientTensors {
                query_gradient: d_q,
                key_gradient: d_k,
                value_gradient: d_v,
                bias_gradient: None,
                rng_dump: None,
                sink_token_gradient: None,
            },
            DirectSdpaBackwardConfig::new()
                .with_sequence_lengths(sequence_length_query, sequence_length_key_value),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorDataTypeMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == sequence_length_query
            && operation == "sdpa backward seq len data type"
            && expected == DataType::I32
            && actual == DataType::F32
    ));

    Ok(())
}

#[test]
fn test_sdpa_backward_direct_rejects_wrong_seq_len_shape() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let d_q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let sequence_length_query =
        graph.tensor(TensorSpec::new(DataType::I32, Shape::contiguous([2, 1])?));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));

    let error = graph
        .sdpa_backward_direct(
            SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
            SdpaBackwardGradientTensors {
                query_gradient: d_q,
                key_gradient: d_k,
                value_gradient: d_v,
                bias_gradient: None,
                rng_dump: None,
                sink_token_gradient: None,
            },
            DirectSdpaBackwardConfig::new()
                .with_sequence_lengths(sequence_length_query, sequence_length_key_value),
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
            && operation == "sdpa seq len q shape"
            && expected == vec![2, 1, 1, 1]
            && actual == vec![2, 1]
    ));

    Ok(())
}

#[test]
fn test_sdpa_backward_direct_accepts_max_total_sequence_lengths() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let q_offsets = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    graph.set_ragged_offset(q, q_offsets)?;

    let _ = graph.sdpa_backward_direct_infer(
        SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
        DirectSdpaBackwardConfig::new()
            .with_max_total_sequence_length_query(128)
            .with_max_total_sequence_length_key_value(256),
    )?;

    Ok(())
}

#[test]
fn test_sdpa_backward_direct_rejects_max_total_sequence_lengths_without_ragged() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let tensor_count = graph.tensors.len();

    let error = graph
        .sdpa_backward_direct_infer(
            SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
            DirectSdpaBackwardConfig::new().with_max_total_sequence_length_query(128),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa backward ragged max_total_seq_len"
    ));
    assert_eq!(graph.tensors.len(), tensor_count);
    assert!(graph.operations.is_empty());

    Ok(())
}

#[test]
fn test_sdpa_backward_direct_clears_max_total_sequence_lengths_before_cudnn_906() {
    let graph = Graph::new();
    let config = DirectSdpaBackwardConfig::new()
        .with_max_total_sequence_length_query(128)
        .with_max_total_sequence_length_key_value(256);

    let support = SdpaBackwardSupport::new(90599, graph.sm_version());
    let normalized =
        graph.normalize_sdpa_backward_max_total_seq_len_for_version(config, 16, 16, support);

    assert_eq!(normalized.max_total_sequence_length_query(), None);
    assert_eq!(normalized.max_total_sequence_length_key_value(), None);
}

#[test]
fn test_sdpa_backward_direct_clears_max_total_sequence_lengths_for_non_multiple_of_16_heads()
-> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 24])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 24])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 24])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 24])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 24])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let q_offsets = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    graph.set_ragged_offset(q, q_offsets)?;

    let _outputs = graph.sdpa_backward_direct_infer(
        SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
        DirectSdpaBackwardConfig::new()
            .with_max_total_sequence_length_query(128)
            .with_max_total_sequence_length_key_value(256),
    )?;

    let Operation::SdpaBackward { config, .. } = &graph.operations[0] else {
        panic!("expected sdpa backward operation");
    };
    assert_eq!(config.max_total_sequence_length_query(), None);
    assert_eq!(config.max_total_sequence_length_key_value(), None);

    Ok(())
}

#[test]
fn test_sdpa_backward_direct_clears_max_total_sequence_lengths_on_sm8x_after_cudnn_91801() {
    let mut graph = Graph::new();
    graph.set_sm_version(80);
    let config = DirectSdpaBackwardConfig::new()
        .with_max_total_sequence_length_query(128)
        .with_max_total_sequence_length_key_value(256);

    let support = SdpaBackwardSupport::new(91801, graph.sm_version());
    let normalized =
        graph.normalize_sdpa_backward_max_total_seq_len_for_version(config, 16, 16, support);

    assert_eq!(normalized.max_total_sequence_length_query(), None);
    assert_eq!(normalized.max_total_sequence_length_key_value(), None);
}

#[test]
fn test_sdpa_backward_direct_rejects_ragged_gqa_before_cudnn_906() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 8, 16])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 8, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let q_offsets = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    graph.set_ragged_offset(q, q_offsets)?;
    let d_q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 8, 16])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 8, 16])?,
    ));

    let error = graph
        .sdpa_backward_direct_for_version(
            90599,
            SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
            SdpaBackwardGradientTensors {
                query_gradient: d_q,
                key_gradient: d_k,
                value_gradient: d_v,
                bias_gradient: None,
                rng_dump: None,
                sink_token_gradient: None,
            },
            DirectSdpaBackwardConfig::new(),
        )
        .unwrap_err();

    assert!(
        matches!(
            error,
            Error::DescriptorMismatch { ref name } if name == "sdpa backward ragged gqa"
        ),
        "{error:?}"
    );

    Ok(())
}

#[test]
fn test_sdpa_backward_direct_rejects_invalid_gqa_head_factor() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 3, 8, 16])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 8, 16])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 8, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 3, 8, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 3, 8, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 8, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let d_q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 3, 8, 16])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 8, 16])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 8, 16])?,
    ));

    let error = graph
        .sdpa_backward_direct_for_version(
            92100,
            SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
            SdpaBackwardGradientTensors {
                query_gradient: d_q,
                key_gradient: d_k,
                value_gradient: d_v,
                bias_gradient: None,
                rng_dump: None,
                sink_token_gradient: None,
            },
            DirectSdpaBackwardConfig::new(),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa backward gqa heads"
    ));

    Ok(())
}

#[test]
fn test_sdpa_backward_direct_rejects_ragged_stats_on_sm8x_and_sm12x() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(80);
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let stats_offsets = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    graph.set_ragged_offset(stats, stats_offsets)?;
    let d_q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));

    let error = graph
        .sdpa_backward_direct_for_version(
            91801,
            SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
            SdpaBackwardGradientTensors {
                query_gradient: d_q,
                key_gradient: d_k,
                value_gradient: d_v,
                bias_gradient: None,
                rng_dump: None,
                sink_token_gradient: None,
            },
            DirectSdpaBackwardConfig::new(),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa backward ragged stats"
    ));

    Ok(())
}

#[test]
fn test_sdpa_backward_direct_rejects_ampere_head_dim_over_128_before_910() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(80);
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 136])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 136])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 144])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 144])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 144])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 8, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let d_q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 136])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 136])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 144])?,
    ));

    let error = graph
        .sdpa_backward_direct_for_version(
            90900,
            SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
            SdpaBackwardGradientTensors {
                query_gradient: d_q,
                key_gradient: d_k,
                value_gradient: d_v,
                bias_gradient: None,
                rng_dump: None,
                sink_token_gradient: None,
            },
            DirectSdpaBackwardConfig::new(),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa backward head dimension"
    ));

    Ok(())
}

#[test]
fn test_sdpa_backward_direct_rejects_hopper_non_deepseek_head_dims_in_911_and_912() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(90);
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 160])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 160])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 96])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 96])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 96])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 8, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let d_q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 160])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 160])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 96])?,
    ));

    let error = graph
        .sdpa_backward_direct_for_version(
            91100,
            SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
            SdpaBackwardGradientTensors {
                query_gradient: d_q,
                key_gradient: d_k,
                value_gradient: d_v,
                bias_gradient: None,
                rng_dump: None,
                sink_token_gradient: None,
            },
            DirectSdpaBackwardConfig::new(),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa backward head dimension"
    ));

    Ok(())
}

#[test]
fn test_sdpa_backward_direct_rejects_blackwell_192_without_128_v() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(100);
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 192])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 192])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 120])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 120])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 120])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 8, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let d_q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 192])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 192])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 120])?,
    ));

    let error = graph
        .sdpa_backward_direct_for_version(
            91100,
            SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
            SdpaBackwardGradientTensors {
                query_gradient: d_q,
                key_gradient: d_k,
                value_gradient: d_v,
                bias_gradient: None,
                rng_dump: None,
                sink_token_gradient: None,
            },
            DirectSdpaBackwardConfig::new(),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa backward head dimension"
    ));

    Ok(())
}

#[test]
fn test_sdpa_backward_direct_rejects_cudnn_9100_and_9101_for_low_precision() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(90);
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 64])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 64])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 64])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 64])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 64])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 8, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let d_q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 64])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 64])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 64])?,
    ));

    let error = graph
        .sdpa_backward_direct_for_version(
            91000,
            SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
            SdpaBackwardGradientTensors {
                query_gradient: d_q,
                key_gradient: d_k,
                value_gradient: d_v,
                bias_gradient: None,
                rng_dump: None,
                sink_token_gradient: None,
            },
            DirectSdpaBackwardConfig::new(),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa backward cuDNN version"
    ));

    Ok(())
}

#[test]
fn test_sdpa_backward_direct_rejects_unit_sequence_lengths() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(90);
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 1, 64])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 1, 64])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 1, 64])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 1, 64])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 1, 64])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 1, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let d_q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 1, 64])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 1, 64])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 1, 64])?,
    ));

    let error = graph
        .sdpa_backward_direct_for_version(
            92100,
            SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
            SdpaBackwardGradientTensors {
                query_gradient: d_q,
                key_gradient: d_k,
                value_gradient: d_v,
                bias_gradient: None,
                rng_dump: None,
                sink_token_gradient: None,
            },
            DirectSdpaBackwardConfig::new(),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa backward sequence lengths"
    ));

    Ok(())
}

#[test]
fn test_sdpa_backward_direct_rejects_d_sink_without_sink() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(90);
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 64])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 64])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 64])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 64])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 64])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 8, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let d_q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 64])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 64])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 64])?,
    ));
    let sink_gradient = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 8, 1])?,
    ));

    let error = graph
        .sdpa_backward_direct_for_version(
            92100,
            SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
            SdpaBackwardGradientTensors {
                query_gradient: d_q,
                key_gradient: d_k,
                value_gradient: d_v,
                bias_gradient: None,
                rng_dump: None,
                sink_token_gradient: None,
            },
            DirectSdpaBackwardConfig::new().with_sink_gradient(sink_gradient),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa backward sink_gradient requires sink"
    ));

    Ok(())
}

#[test]
fn test_graph_ragged_offset_is_reflected_in_tensor_attributes() -> Result<()> {
    let mut graph = Graph::new();
    let data = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let offsets = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));

    graph.set_ragged_offset(data, offsets)?;

    assert_eq!(graph.tensor_config(data)?.ragged_offset, Some(offsets));

    graph.clear_ragged_offset(data)?;
    assert_eq!(graph.tensor_config(data)?.ragged_offset, None);

    Ok(())
}

#[test]
fn test_graph_builder_methods_configure_common_attributes() {
    let graph = Graph::new()
        .with_name("builder")
        .with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F16)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        )
        .with_dynamic_shape()
        .with_override_shape()
        .with_sm_count_target(8)
        .with_sm_version(90);

    assert_eq!(graph.name(), Some("builder"));
    assert_eq!(graph.io_data_type(), Some(DataType::F16));
    assert_eq!(graph.intermediate_data_type(), Some(DataType::F32));
    assert_eq!(graph.compute_data_type(), Some(DataType::F32));
    assert_eq!(
        graph.data_type_policy(),
        DataTypePolicy::new()
            .with_io(DataType::F16)
            .with_intermediate(DataType::F32)
            .with_compute(DataType::F32)
    );
    assert!(graph.dynamic_shape_enabled());
    assert!(graph.override_shape_enabled());
    assert_eq!(graph.sm_count_target(), Some(8));
    assert_eq!(graph.sm_version(), Some(90));
}

#[test]
fn test_alias_tensor_rejects_unaligned_byte_offset() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(DataType::F16, Shape::contiguous([4])?));
    let alias_tensor = TensorSpec::new(DataType::F16, Shape::contiguous([2])?);

    let error = graph.alias_tensor(input, alias_tensor, 1).unwrap_err();
    assert!(matches!(
        error,
        Error::FrontendAliasByteOffsetUnaligned {
            source_id,
            target_id: _,
            byte_offset: 1,
            element_size: 2,
        } if source_id == input
    ));

    Ok(())
}

#[test]
fn test_alias_tensor_rejects_mismatched_data_type() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(DataType::F16, Shape::contiguous([4])?));
    let alias_tensor = TensorSpec::new(DataType::F32, Shape::contiguous([2])?).with_id(99);
    let alias_id = alias_tensor.id().unwrap();

    let error = graph.alias_tensor(input, alias_tensor, 0).unwrap_err();
    assert!(matches!(
        error,
        Error::FrontendTensorDataTypeMismatch {
            tensor_id,
            operation,
            expected: DataType::F16,
            actual: DataType::F32,
        } if tensor_id == alias_id && operation == "alias target"
    ));

    Ok(())
}

#[test]
fn test_alias_tensor_rejects_negative_byte_offset() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(DataType::F16, Shape::contiguous([4])?));
    let alias_tensor = TensorSpec::new(DataType::F16, Shape::contiguous([2])?).with_id(99);

    let error = graph.alias_tensor(input, alias_tensor, -2).unwrap_err();
    assert!(matches!(
        error,
        Error::FrontendAliasByteRangeOutOfBounds {
            source_id,
            target_id,
            start: 0,
            end: 4,
            source_byte_len: 8,
        } if source_id == input && target_id == 99.into()
    ));

    Ok(())
}

#[test]
fn test_alias_tensor_rejects_byte_range_past_source() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(DataType::F16, Shape::contiguous([4])?));
    let alias_tensor = TensorSpec::new(DataType::F16, Shape::contiguous([2])?).with_id(99);

    let error = graph.alias_tensor(input, alias_tensor, 6).unwrap_err();
    assert!(matches!(
        error,
        Error::FrontendAliasByteRangeOutOfBounds {
            source_id,
            target_id,
            start: 6,
            end: 10,
            source_byte_len: 8,
        } if source_id == input && target_id == 99.into()
    ));

    Ok(())
}

#[test]
fn test_alias_tensor_rejects_virtual_source() -> Result<()> {
    let mut graph = Graph::new();
    let input =
        graph.tensor(TensorSpec::new(DataType::F16, Shape::contiguous([4])?).virtual_tensor());
    let alias_tensor = TensorSpec::new(DataType::F16, Shape::contiguous([2])?);

    let error = graph.alias_tensor(input, alias_tensor, 0).unwrap_err();
    assert!(matches!(
        error,
        Error::FrontendAliasSourceNotBindable {
            source_id,
            reason,
        } if source_id == input && reason == "source tensor is virtual"
    ));

    Ok(())
}

#[test]
fn test_alias_tensor_rejects_by_value_source() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let alias_tensor = TensorSpec::new(DataType::F32, Shape::contiguous([1])?);

    let error = graph.alias_tensor(input, alias_tensor, 0).unwrap_err();
    assert!(matches!(
        error,
        Error::FrontendAliasSourceNotBindable {
            source_id,
            reason,
        } if source_id == input && reason == "source tensor is by value"
    ));

    Ok(())
}

#[test]
fn test_graph_tensor_role_helpers_update_existing_tensor() -> Result<()> {
    let mut graph = Graph::new();
    let tensor = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([4])?));

    graph.mark_as_virtual(tensor)?;
    assert!(graph.tensor_config(tensor)?.is_virtual);

    graph.mark_as_output(tensor)?;
    assert!(!graph.tensor_config(tensor)?.is_virtual);

    graph.modify_tensor(tensor, |tensor| tensor.with_name("renamed"))?;
    assert_eq!(
        graph.tensor_config(tensor)?.name.as_deref(),
        Some("renamed")
    );

    Ok(())
}

#[test]
fn test_tensor_lookup_and_clone_helpers_preserve_shape() -> Result<()> {
    let mut graph = Graph::new();
    let mut tensor = TensorSpec::new(DataType::F16, Shape::contiguous([2, 4, 8, 16])?);
    tensor = tensor.with_id(41);
    tensor.set_name("input");
    let input = graph.tensor(tensor);

    assert_eq!(input, 41.into());
    assert_eq!(
        graph.tensor_config(41.into())?.name.as_deref(),
        Some("input")
    );

    let clone = graph.tensor_like(input)?;
    let clone_attributes = graph.tensor_config(clone)?;
    assert_eq!(clone_attributes.shape, graph.shape(input)?.clone());
    assert_eq!(clone_attributes.data_type, DataType::F16);
    assert_eq!(clone_attributes.id(), Some(clone));
    assert_ne!(clone, input);
    assert_eq!(clone_attributes.name.as_deref(), None);

    let renamed = graph.tensor_like_named(input, "output")?;
    let renamed_attributes = graph.tensor_config(renamed)?;
    assert_eq!(renamed_attributes.shape, graph.shape(input)?.clone());
    assert_eq!(renamed_attributes.data_type, DataType::F16);
    assert_eq!(renamed_attributes.id(), Some(renamed));
    assert_ne!(renamed, input);
    assert_eq!(renamed_attributes.name.as_deref(), Some("output"));

    Ok(())
}

#[test]
fn test_dynamic_shape_enabled_graph_marks_backend_operation_graph() -> Result<()> {
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
    graph.enable_dynamic_shape();
    let x = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([4])?));
    let y = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([4])?));
    graph.reshape(x, y)?;

    let Some(prepared) = prepare_or_skip(&graph, &context)? else {
        return Ok(());
    };
    assert!(prepared.operation_graph.is_dynamic_shape_enabled()?);
    assert!(!prepared.operation_graph.is_override_shape_enabled()?);
    let engine_count = match prepared.operation_graph.engine_count() {
        Ok(engine_count) => engine_count,
        Err(error) if is_expected_prepare_status(&error) => return Ok(()),
        Err(error) => return Err(error),
    };
    assert!(engine_count > 0);

    let _successful_engine_configs = (0..engine_count)
        .filter_map(|engine_index| {
            let engine_index = to_i64(engine_index, "engine_index").ok()?;
            EngineConfig::create_with_config(
                &context,
                &prepared.operation_graph,
                EngineIndex::new(engine_index),
                &[],
                None,
            )
            .ok()
        })
        .count();

    let candidates = match graph.plan_candidates(&context, &[HeuristicMode::A]) {
        Ok(candidates) => candidates,
        Err(error) if is_expected_prepare_status(&error) => return Ok(()),
        Err(error) => return Err(error),
    };
    assert!(candidates.candidate_count() > 0);
    match candidates.compile(&context) {
        Ok(_compiled) => {}
        Err(error) if is_expected_prepare_status(&error) => return Ok(()),
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_override_shape_enabled_graph_marks_backend_operation_graph() -> Result<()> {
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
    graph.enable_override_shape();
    let x = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([4])?));
    let y = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([4])?));
    graph.reshape(x, y)?;

    let Some(prepared) = prepare_or_skip(&graph, &context)? else {
        return Ok(());
    };
    assert!(prepared.operation_graph.is_override_shape_enabled()?);

    Ok(())
}

#[test]
fn test_prepare_attaches_unified_sdpa_subgraph_for_bias_modifier() -> Result<()> {
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
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 6, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 6, 16])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);
    let bias = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 6])?,
    ));

    let SdpaOutputs {
        output: _,
        stats: _,
    } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32).with_bias(bias),
    )?;

    let Some(prepared) = prepare_or_skip(&graph, &context)? else {
        return Ok(());
    };
    let Some(operation_descriptors) = operation_descriptors_or_skip(&prepared)? else {
        return Ok(());
    };
    assert_eq!(operation_descriptors.len(), 1);

    Ok(())
}

#[test]
fn test_prepare_attaches_unified_sdpa_subgraph_for_causal_mask() -> Result<()> {
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

    let SdpaOutputs {
        output: _,
        stats: _,
    } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32).with_causal_mask(),
    )?;

    let Some(prepared) = prepare_or_skip(&graph, &context)? else {
        return Ok(());
    };
    let Some(operation_descriptors) = operation_descriptors_or_skip(&prepared)? else {
        return Ok(());
    };
    assert_eq!(operation_descriptors.len(), 1);

    Ok(())
}

#[test]
fn test_prepare_supports_override_shape_sdpa_with_causal_and_padding_mask() -> Result<()> {
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
    graph.set_io_data_type(DataType::BF16);
    graph.set_intermediate_data_type(DataType::F32);
    graph.set_compute_data_type(DataType::F32);
    graph.enable_override_shape();

    let q = graph.tensor(
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([2, 4, 1024, 128])?.with_strides(vec![
                4 * 1024 * 128,
                1024 * 128,
                128,
                1,
            ])?,
        )
        .with_id(1),
    );
    let k = graph.tensor(
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([2, 4, 1024, 128])?.with_strides(vec![
                4 * 1024 * 128,
                1024 * 128,
                128,
                1,
            ])?,
        )
        .with_id(2),
    );
    let v = graph.tensor(
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([2, 4, 1024, 128])?.with_strides(vec![
                4 * 1024 * 128,
                1024 * 128,
                128,
                1,
            ])?,
        )
        .with_id(3),
    );
    let sequence_length_query = graph.tensor(
        TensorSpec::new(
            DataType::I32,
            Shape::contiguous([2, 1, 1, 1])?.with_strides([1, 1, 1, 1])?,
        )
        .with_id(7),
    );
    let sequence_length_key_value = graph.tensor(
        TensorSpec::new(
            DataType::I32,
            Shape::contiguous([2, 1, 1, 1])?.with_strides([1, 1, 1, 1])?,
        )
        .with_id(8),
    );
    let scale = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?.with_strides([1, 1, 1, 1])?,
    ));

    let SdpaOutputs { output, stats } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32)
            .with_stats()
            .with_causal_mask()
            .with_padding_mask(sequence_length_query, sequence_length_key_value),
    )?;

    graph.replace_tensor(
        output,
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([2, 4, 1024, 128])?.with_strides(vec![
                4 * 128,
                128,
                2 * 4 * 128,
                1,
            ])?,
        )
        .output_tensor()
        .with_id(4),
    )?;
    let stats = stats.expect("stats requested");
    let stats_tensor = graph
        .tensor_config(stats)?
        .clone()
        .output_tensor()
        .with_id(5);
    graph.replace_tensor(stats, stats_tensor)?;

    let Some(prepared) = prepare_or_skip(&graph, &context)? else {
        return Ok(());
    };
    assert!(prepared.operation_graph.is_override_shape_enabled()?);

    Ok(())
}

#[test]
fn test_unified_sdpa_subgraph_scalar_bindings_are_included_in_binding_template() -> Result<()> {
    let context = match setup_context() {
        Ok(ctx) => ctx,
        Err(Error::Cuda(singe_cuda::error::Error::Cuda { code, .. }))
            if code == singe_cuda::error::Status::NoDevice =>
        {
            return Ok(());
        }
        Err(error) => return Err(error),
    };

    let b = 3;
    let h = 4;
    let s_q = 1024;
    let s_kv = 1024;
    let d_qk = 128;
    let d_v = 128;

    let mut graph = Graph::new();
    graph.set_io_data_type(DataType::BF16);
    graph.set_intermediate_data_type(DataType::F32);
    graph.set_compute_data_type(DataType::F32);

    let q = graph
        .tensor(TensorSpec::new(DataType::BF16, Shape::contiguous([b, h, s_q, d_qk])?).with_id(1));
    let k = graph
        .tensor(TensorSpec::new(DataType::BF16, Shape::contiguous([b, h, s_kv, d_qk])?).with_id(2));
    let v = graph
        .tensor(TensorSpec::new(DataType::BF16, Shape::contiguous([b, h, s_kv, d_v])?).with_id(3));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.123)?);
    let sequence_length_query =
        graph.tensor(TensorSpec::new(DataType::I32, Shape::contiguous([b, 1, 1, 1])?).with_id(7));
    let sequence_length_key_value =
        graph.tensor(TensorSpec::new(DataType::I32, Shape::contiguous([b, 1, 1, 1])?).with_id(8));

    let SdpaOutputs { output, stats } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32)
            .with_stats()
            .with_causal_mask()
            .with_padding_mask(sequence_length_query, sequence_length_key_value),
    )?;

    graph.replace_tensor(
        output,
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([b, h, s_q, d_v])?.with_strides(vec![
                h * d_v,
                d_v,
                b * h * d_v,
                1,
            ])?,
        )
        .output_tensor()
        .with_id(4),
    )?;
    let stats = stats.expect("stats requested");
    let stats_tensor = graph
        .tensor_config(stats)?
        .clone()
        .output_tensor()
        .with_id(5);
    graph.replace_tensor(stats, stats_tensor)?;

    let Some(prepared) = prepare_or_skip(&graph, &context)? else {
        return Ok(());
    };
    assert!(
        prepared
            .lowered
            .binding_template_ids
            .contains(&prepared.scalar_bindings[0].0)
    );

    Ok(())
}

#[test]
fn test_prepare_keeps_padding_mask_on_unified_seq_len_descriptors() -> Result<()> {
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
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 4, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 6, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 6, 16])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));

    let SdpaOutputs {
        output: _,
        stats: _,
    } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32)
            .with_padding_mask(sequence_length_query, sequence_length_key_value),
    )?;

    let Some(prepared) = prepare_or_skip(&graph, &context)? else {
        return Ok(());
    };
    let Some(operation_descriptors) = operation_descriptors_or_skip(&prepared)? else {
        return Ok(());
    };
    assert_eq!(operation_descriptors.len(), 1);

    let sdpa_forward = match BackendDescriptor::from_raw_finalized_borrowed(
        operation_descriptors[0],
        BackendDescriptorType::OperationSdpaForward,
    ) {
        Ok(sdpa_forward) => sdpa_forward,
        Err(error) if is_expected_prepare_status(&error) => return Ok(()),
        Err(error) => return Err(error),
    };
    match sdpa_forward.attribute_descriptor(
        BackendAttributeName::OperationSdpaFwdSeqLenQDesc,
        BackendAttributeType::BackendDescriptor,
    ) {
        Ok(_) => {}
        Err(error) if is_expected_prepare_status(&error) => return Ok(()),
        Err(error) => return Err(error),
    }
    match sdpa_forward.attribute_descriptor(
        BackendAttributeName::OperationSdpaFwdSeqLenKvDesc,
        BackendAttributeType::BackendDescriptor,
    ) {
        Ok(_) => {}
        Err(error) if is_expected_prepare_status(&error) => return Ok(()),
        Err(error) => return Err(error),
    }
    let subgraph_err = match sdpa_forward.attribute_descriptor(
        BackendAttributeName::OperationSdpaFwdSubgraph,
        BackendAttributeType::BackendDescriptor,
    ) {
        Ok(_) => {
            return Err(Error::DescriptorMismatch {
                name: "sdpa forward subgraph".into(),
            });
        }
        Err(error) => error,
    };
    assert!(matches!(
        subgraph_err,
        Error::DescriptorAttributeNotFound(BackendAttributeName::OperationSdpaFwdSubgraph)
    ));

    Ok(())
}

#[test]
fn test_prepare_combines_unified_subgraph_with_seq_len_descriptors() -> Result<()> {
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
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 4, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 6, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 6, 16])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);
    let bias = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 4, 6])?,
    ));
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));

    let SdpaOutputs {
        output: _,
        stats: _,
    } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32)
            .with_bias(bias)
            .with_padding_mask(sequence_length_query, sequence_length_key_value),
    )?;

    let Some(prepared) = prepare_or_skip(&graph, &context)? else {
        return Ok(());
    };
    let Some(operation_descriptors) = operation_descriptors_or_skip(&prepared)? else {
        return Ok(());
    };
    assert_eq!(operation_descriptors.len(), 1);

    let sdpa_forward = BackendDescriptor::from_raw_finalized_borrowed(
        operation_descriptors[0],
        BackendDescriptorType::OperationSdpaForward,
    )?;
    match sdpa_forward.attribute_descriptor(
        BackendAttributeName::OperationSdpaFwdSubgraph,
        BackendAttributeType::BackendDescriptor,
    ) {
        Ok(_) => {}
        Err(error) if is_expected_prepare_status(&error) => return Ok(()),
        Err(error) => return Err(error),
    }
    match sdpa_forward.attribute_descriptor(
        BackendAttributeName::OperationSdpaFwdSeqLenQDesc,
        BackendAttributeType::BackendDescriptor,
    ) {
        Ok(_) => {}
        Err(error) if is_expected_prepare_status(&error) => return Ok(()),
        Err(error) => return Err(error),
    }
    match sdpa_forward.attribute_descriptor(
        BackendAttributeName::OperationSdpaFwdSeqLenKvDesc,
        BackendAttributeType::BackendDescriptor,
    ) {
        Ok(_) => {}
        Err(error) if is_expected_prepare_status(&error) => return Ok(()),
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_prepare_attaches_unified_sdpa_subgraph_for_custom_modifier() -> Result<()> {
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
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 6, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 6, 16])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);

    let mut subgraph = Graph::new();
    let subgraph_input = subgraph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 6])?,
    ));
    let bias = subgraph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 6])?,
    ));
    let subgraph_output =
        subgraph.pointwise_binary_infer(subgraph_input, bias, PointwiseMode::Add, DataType::F32)?;

    let SdpaOutputs {
        output: _,
        stats: _,
    } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32).with_score_subgraph(SdpaScoreSubgraph::new(
            subgraph,
            subgraph_input,
            subgraph_output,
        )),
    )?;

    let Some(prepared) = prepare_or_skip(&graph, &context)? else {
        return Ok(());
    };
    let Some(operation_descriptors) = operation_descriptors_or_skip(&prepared)? else {
        return Ok(());
    };
    assert_eq!(operation_descriptors.len(), 1);

    Ok(())
}

#[test]
fn test_prepare_combines_unified_custom_sdpa_subgraph_with_bias() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 6, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 6, 16])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);
    let bias =
        graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1, 2, 4, 6])?).with_id(6));

    let mut subgraph = Graph::new();
    let subgraph_input = subgraph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 6])?,
    ));
    let captured_bias = subgraph
        .tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1, 2, 4, 6])?).with_id(6));
    let biased = subgraph.pointwise_binary_infer(
        subgraph_input,
        captured_bias,
        PointwiseMode::Add,
        DataType::F32,
    )?;
    let soft_cap = subgraph.tensor(TensorSpec::scalar_f32(0.8)?);
    let subgraph_output =
        subgraph.pointwise_binary_infer(biased, soft_cap, PointwiseMode::Mul, DataType::F32)?;

    let config = AttentionConfig::new(DataType::F32)
        .with_bias(bias)
        .with_score_subgraph(SdpaScoreSubgraph::new(
            subgraph,
            subgraph_input,
            subgraph_output,
        ));

    let SdpaOutputs {
        output: _,
        stats: _,
    } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        config,
    )?;

    assert!(
        graph
            .operations
            .iter()
            .any(|operation| matches!(operation, Operation::SdpaForward { .. }))
    );

    Ok(())
}

#[test]
fn test_prepare_combines_unified_custom_sdpa_subgraph_with_additive_mask() -> Result<()> {
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
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 6, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 6, 16])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);
    let additive_mask = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 4, 6])?,
    ));

    let mut subgraph = Graph::new();
    let subgraph_input = subgraph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 6])?,
    ));
    let soft_cap = subgraph.tensor(TensorSpec::scalar_f32(0.8)?);
    let subgraph_output = subgraph.pointwise_binary_infer(
        subgraph_input,
        soft_cap,
        PointwiseMode::Mul,
        DataType::F32,
    )?;

    let SdpaOutputs {
        output: _,
        stats: _,
    } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32)
            .with_additive_mask(additive_mask)
            .with_score_subgraph(SdpaScoreSubgraph::new(
                subgraph,
                subgraph_input,
                subgraph_output,
            )),
    )?;

    let Some(prepared) = prepare_or_skip(&graph, &context)? else {
        return Ok(());
    };
    let Some(operation_descriptors) = operation_descriptors_or_skip(&prepared)? else {
        return Ok(());
    };
    assert_eq!(operation_descriptors.len(), 1);

    Ok(())
}

#[test]
fn test_prepare_combines_unified_custom_sdpa_subgraph_with_seq_lens() -> Result<()> {
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
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 4, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 6, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 6, 16])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));

    let mut subgraph = Graph::new();
    let subgraph_input = subgraph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 2, 4, 6])?,
    ));
    let soft_cap = subgraph.tensor(TensorSpec::scalar_f32(0.8)?);
    let subgraph_output = subgraph.pointwise_binary_infer(
        subgraph_input,
        soft_cap,
        PointwiseMode::Mul,
        DataType::F32,
    )?;

    let SdpaOutputs {
        output: _,
        stats: _,
    } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32)
            .with_sequence_lengths(sequence_length_query, sequence_length_key_value)
            .with_score_subgraph(SdpaScoreSubgraph::new(
                subgraph,
                subgraph_input,
                subgraph_output,
            )),
    )?;

    let Some(prepared) = prepare_or_skip(&graph, &context)? else {
        return Ok(());
    };
    let Some(operation_descriptors) = operation_descriptors_or_skip(&prepared)? else {
        return Ok(());
    };
    assert_eq!(operation_descriptors.len(), 1);

    let sdpa_forward = BackendDescriptor::from_raw_finalized_borrowed(
        operation_descriptors[0],
        BackendDescriptorType::OperationSdpaForward,
    )?;
    match sdpa_forward.attribute_descriptor(
        BackendAttributeName::OperationSdpaFwdSubgraph,
        BackendAttributeType::BackendDescriptor,
    ) {
        Ok(_) => {}
        Err(error) if is_expected_prepare_status(&error) => return Ok(()),
        Err(error) => return Err(error),
    }
    match sdpa_forward.attribute_descriptor(
        BackendAttributeName::OperationSdpaFwdSeqLenQDesc,
        BackendAttributeType::BackendDescriptor,
    ) {
        Ok(_) => {}
        Err(error) if is_expected_prepare_status(&error) => return Ok(()),
        Err(error) => return Err(error),
    }
    match sdpa_forward.attribute_descriptor(
        BackendAttributeName::OperationSdpaFwdSeqLenKvDesc,
        BackendAttributeType::BackendDescriptor,
    ) {
        Ok(_) => {}
        Err(error) if is_expected_prepare_status(&error) => return Ok(()),
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_prepare_attaches_unified_sdpa_subgraph_for_ragged_custom_modifier() -> Result<()> {
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
    graph.set_sm_version(90);
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 4, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 4, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 4, 16])?,
    ));
    let offsets = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    graph.set_ragged_offset(q, offsets)?;
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);

    let mut subgraph = Graph::new();
    let subgraph_input = subgraph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 2, 4, 4])?,
    ));
    let soft_cap = subgraph.tensor(TensorSpec::scalar_f32(0.8)?);
    let subgraph_output = subgraph.pointwise_binary_infer(
        subgraph_input,
        soft_cap,
        PointwiseMode::Mul,
        DataType::F32,
    )?;

    let SdpaOutputs {
        output: _,
        stats: _,
    } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32).with_score_subgraph(SdpaScoreSubgraph::new(
            subgraph,
            subgraph_input,
            subgraph_output,
        )),
    )?;

    let Some(prepared) = prepare_or_skip(&graph, &context)? else {
        return Ok(());
    };
    let Some(operation_descriptors) = operation_descriptors_or_skip(&prepared)? else {
        return Ok(());
    };
    assert_eq!(operation_descriptors.len(), 1);

    Ok(())
}

#[test]
fn test_prepare_combines_unified_ragged_custom_sdpa_subgraph_with_seq_lens() -> Result<()> {
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
    graph.set_sm_version(90);
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 4, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 4, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 4, 16])?,
    ));
    let offsets = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    graph.set_ragged_offset(q, offsets)?;
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));

    let mut subgraph = Graph::new();
    let subgraph_input = subgraph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 2, 4, 4])?,
    ));
    let soft_cap = subgraph.tensor(TensorSpec::scalar_f32(0.8)?);
    let subgraph_output = subgraph.pointwise_binary_infer(
        subgraph_input,
        soft_cap,
        PointwiseMode::Mul,
        DataType::F32,
    )?;

    let SdpaOutputs {
        output: _,
        stats: _,
    } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32)
            .with_sequence_lengths(sequence_length_query, sequence_length_key_value)
            .with_score_subgraph(SdpaScoreSubgraph::new(
                subgraph,
                subgraph_input,
                subgraph_output,
            )),
    )?;

    let Some(prepared) = prepare_or_skip(&graph, &context)? else {
        return Ok(());
    };
    let Some(operation_descriptors) = operation_descriptors_or_skip(&prepared)? else {
        return Ok(());
    };
    assert_eq!(operation_descriptors.len(), 1);

    let sdpa_forward = BackendDescriptor::from_raw_finalized_borrowed(
        operation_descriptors[0],
        BackendDescriptorType::OperationSdpaForward,
    )?;
    match sdpa_forward.attribute_descriptor(
        BackendAttributeName::OperationSdpaFwdSubgraph,
        BackendAttributeType::BackendDescriptor,
    ) {
        Ok(_) => {}
        Err(error) if is_expected_prepare_status(&error) => return Ok(()),
        Err(error) => return Err(error),
    }
    let _seq_len_q = sdpa_forward.attribute_descriptor(
        BackendAttributeName::OperationSdpaFwdSeqLenQDesc,
        BackendAttributeType::BackendDescriptor,
    )?;
    let _seq_len_kv = sdpa_forward.attribute_descriptor(
        BackendAttributeName::OperationSdpaFwdSeqLenKvDesc,
        BackendAttributeType::BackendDescriptor,
    )?;

    Ok(())
}

#[test]
fn test_prepare_keeps_unified_seq_len_descriptors_for_ragged_padding_mask() -> Result<()> {
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
    graph.set_sm_version(90);
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 4, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 6, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 6, 16])?,
    ));
    let offsets = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    graph.set_ragged_offset(q, offsets)?;
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));

    let SdpaOutputs {
        output: _,
        stats: _,
    } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32)
            .with_padding_mask(sequence_length_query, sequence_length_key_value),
    )?;

    let Some(prepared) = prepare_or_skip(&graph, &context)? else {
        return Ok(());
    };
    let Some(operation_descriptors) = operation_descriptors_or_skip(&prepared)? else {
        return Ok(());
    };
    assert_eq!(operation_descriptors.len(), 1);

    let sdpa_forward = BackendDescriptor::from_raw_finalized_borrowed(
        operation_descriptors[0],
        BackendDescriptorType::OperationSdpaForward,
    )?;
    let _seq_len_q = sdpa_forward.attribute_descriptor(
        BackendAttributeName::OperationSdpaFwdSeqLenQDesc,
        BackendAttributeType::BackendDescriptor,
    )?;
    let _seq_len_kv = sdpa_forward.attribute_descriptor(
        BackendAttributeName::OperationSdpaFwdSeqLenKvDesc,
        BackendAttributeType::BackendDescriptor,
    )?;

    Ok(())
}

#[test]
fn test_deserialized_graph_preserves_unified_sdpa_custom_subgraph() -> Result<()> {
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
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 6, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 6, 16])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);

    let mut subgraph = Graph::new();
    let subgraph_input = subgraph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 6])?,
    ));
    let bias = subgraph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 6])?,
    ));
    let subgraph_output =
        subgraph.pointwise_binary_infer(subgraph_input, bias, PointwiseMode::Add, DataType::F32)?;

    let SdpaOutputs {
        output: _,
        stats: _,
    } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32).with_score_subgraph(SdpaScoreSubgraph::new(
            subgraph,
            subgraph_input,
            subgraph_output,
        )),
    )?;

    let restored = serde_json::from_str::<Graph>(&serde_json::to_string(&graph)?)?;

    let Operation::SdpaForward { score_subgraph, .. } = &restored.operations[0] else {
        panic!("expected sdpa forward operation with custom score subgraph");
    };
    let score_subgraph = score_subgraph
        .as_ref()
        .as_ref()
        .expect("expected sdpa forward operation with custom score subgraph");
    assert_eq!(score_subgraph.graph().operations.len(), 1);
    assert_eq!(
        score_subgraph
            .graph()
            .shape(score_subgraph.input())?
            .dimensions(),
        &[1, 2, 4, 6]
    );
    assert_eq!(
        score_subgraph
            .graph()
            .shape(score_subgraph.output())?
            .dimensions(),
        &[1, 2, 4, 6]
    );

    let Some(prepared) = prepare_or_skip(&restored, &context)? else {
        return Ok(());
    };
    let Some(operation_descriptors) = operation_descriptors_or_skip(&prepared)? else {
        return Ok(());
    };
    assert_eq!(operation_descriptors.len(), 1);

    let sdpa_forward = BackendDescriptor::from_raw_finalized_borrowed(
        operation_descriptors[0],
        BackendDescriptorType::OperationSdpaForward,
    )?;
    let _subgraph = sdpa_forward.attribute_descriptor(
        BackendAttributeName::OperationSdpaFwdSubgraph,
        BackendAttributeType::BackendDescriptor,
    )?;

    Ok(())
}

#[test]
fn test_deserialized_graph_preserves_unified_sdpa_custom_subgraph_with_seq_lens() -> Result<()> {
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
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 4, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 6, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 6, 16])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));

    let mut subgraph = Graph::new();
    let subgraph_input = subgraph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 2, 4, 6])?,
    ));
    let soft_cap = subgraph.tensor(TensorSpec::scalar_f32(0.8)?);
    let subgraph_output = subgraph.pointwise_binary_infer(
        subgraph_input,
        soft_cap,
        PointwiseMode::Mul,
        DataType::F32,
    )?;

    let SdpaOutputs {
        output: _,
        stats: _,
    } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32)
            .with_sequence_lengths(sequence_length_query, sequence_length_key_value)
            .with_score_subgraph(SdpaScoreSubgraph::new(
                subgraph,
                subgraph_input,
                subgraph_output,
            )),
    )?;

    let restored = serde_json::from_str::<Graph>(&serde_json::to_string(&graph)?)?;

    let Operation::SdpaForward {
        score_subgraph,
        config,
        ..
    } = &restored.operations[0]
    else {
        panic!("expected sdpa forward operation with custom score subgraph");
    };
    let score_subgraph = score_subgraph
        .as_ref()
        .as_ref()
        .expect("expected sdpa forward operation with custom score subgraph");
    assert_eq!(score_subgraph.graph().operations.len(), 1);
    assert_eq!(
        config.sequence_length_query().unwrap(),
        sequence_length_query
    );
    assert_eq!(
        config.sequence_length_key_value().unwrap(),
        sequence_length_key_value
    );
    assert!(!config.padding_mask());

    let Some(prepared) = prepare_or_skip(&restored, &context)? else {
        return Ok(());
    };
    let Some(operation_descriptors) = operation_descriptors_or_skip(&prepared)? else {
        return Ok(());
    };
    assert_eq!(operation_descriptors.len(), 1);

    let sdpa_forward = match BackendDescriptor::from_raw_finalized_borrowed(
        operation_descriptors[0],
        BackendDescriptorType::OperationSdpaForward,
    ) {
        Ok(sdpa_forward) => sdpa_forward,
        Err(error) if is_expected_prepare_status(&error) => return Ok(()),
        Err(error) => return Err(error),
    };
    match sdpa_forward.attribute_descriptor(
        BackendAttributeName::OperationSdpaFwdSubgraph,
        BackendAttributeType::BackendDescriptor,
    ) {
        Ok(_) => {}
        Err(error) if is_expected_prepare_status(&error) => return Ok(()),
        Err(error) => return Err(error),
    }
    match sdpa_forward.attribute_descriptor(
        BackendAttributeName::OperationSdpaFwdSeqLenQDesc,
        BackendAttributeType::BackendDescriptor,
    ) {
        Ok(_) => {}
        Err(error) if is_expected_prepare_status(&error) => return Ok(()),
        Err(error) => return Err(error),
    }
    match sdpa_forward.attribute_descriptor(
        BackendAttributeName::OperationSdpaFwdSeqLenKvDesc,
        BackendAttributeType::BackendDescriptor,
    ) {
        Ok(_) => {}
        Err(error) if is_expected_prepare_status(&error) => return Ok(()),
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn tensor_accepts_negative_explicit_id() {
    let mut graph = Graph::new();
    let tensor = TensorSpec::new(DataType::F32, Shape::contiguous([4]).unwrap()).with_id(-1);
    let id = graph.tensor(tensor);
    assert_eq!(id, (-1).into());
}

#[test]
fn tensor_config_new_is_idless_until_graph_insertion() -> Result<()> {
    let tensor = TensorSpec::new(DataType::F32, Shape::contiguous([4])?);
    assert_eq!(tensor.id(), None);

    let mut graph = Graph::new();
    let first_id = graph.tensor(tensor.clone());
    let second_id = graph.tensor(tensor);

    assert_ne!(first_id, second_id);
    assert_eq!(graph.tensor_config(first_id)?.id(), Some(first_id));
    assert_eq!(graph.tensor_config(second_id)?.id(), Some(second_id));

    Ok(())
}

#[test]
fn checked_tensor_insertion_requires_explicit_id() -> Result<()> {
    let mut graph = Graph::new();
    let error = graph
        .insert_tensor(TensorSpec::new(DataType::F32, Shape::contiguous([4])?))
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "frontend tensor id"
    ));

    Ok(())
}

#[test]
fn try_tensor_rejects_duplicate_explicit_id_without_panicking() -> Result<()> {
    let mut graph = Graph::new();
    let tensor = TensorSpec::new(DataType::F32, Shape::contiguous([4])?).with_id(17);
    graph.insert_tensor(tensor.clone())?;

    let error = graph.insert_tensor(tensor).unwrap_err();
    assert!(matches!(error, Error::FrontendTensorIdConflict(id) if id == 17.into()));

    Ok(())
}

#[test]
fn try_tensor_is_checked_insert_alias() -> Result<()> {
    let mut graph = Graph::new();
    let tensor = TensorSpec::new(DataType::F32, Shape::contiguous([4])?).with_id(17);
    graph.try_tensor(tensor.clone())?;

    let error = graph.try_tensor(tensor).unwrap_err();
    assert!(matches!(error, Error::FrontendTensorIdConflict(id) if id == 17.into()));

    Ok(())
}

#[test]
fn tensor_assigns_fresh_id_for_duplicate_explicit_id() {
    let mut graph = Graph::new();
    let tensor = TensorSpec::new(DataType::F32, Shape::contiguous([4]).unwrap()).with_id(17);
    let first = graph.tensor(tensor.clone());

    assert_eq!(first, 17.into());
    let second = graph.tensor(tensor);
    assert_ne!(second, first);
}

#[test]
fn tensor_with_fresh_id_ignores_configured_id() -> Result<()> {
    let mut graph = Graph::new();
    graph.insert_tensor(TensorSpec::new(DataType::F32, Shape::contiguous([4])?).with_id(17))?;

    let fresh = graph
        .tensor_with_fresh_id(TensorSpec::new(DataType::F32, Shape::contiguous([4])?).with_id(17));

    assert_ne!(fresh, 17.into());
    assert!(graph.tensor_config(17.into()).is_ok());
    assert!(graph.tensor_config(fresh).is_ok());

    Ok(())
}

#[test]
fn graph_dynamic_shape_constraints_validate_and_serialize() -> Result<()> {
    let mut graph = Graph::new();
    let tensor = graph.try_tensor(
        TensorSpec::new(DataType::F32, Shape::contiguous([2, 3])?)
            .with_id(17)
            .with_name("dynamic"),
    )?;

    graph.set_dynamic_shape_constraints(tensor, [1, 3], [8, 16])?;
    let constraints = graph
        .dynamic_shape_constraints(tensor)
        .expect("constraints should be present");
    assert_eq!(constraints.min_dimensions, vec![1, 3]);
    assert_eq!(constraints.max_dimensions, vec![8, 16]);
    assert!(constraints.contains(&[4, 5]));
    assert!(!constraints.contains(&[9, 5]));

    let restored = serde_json::from_str::<Graph>(&serde_json::to_string(&graph)?)?;
    assert_eq!(
        restored.dynamic_shape_constraints(tensor),
        graph.dynamic_shape_constraints(tensor)
    );

    let error = graph
        .set_dynamic_shape_constraints(tensor, [1], [8])
        .unwrap_err();
    assert!(matches!(
        error,
        Error::LengthMismatch {
            name,
            expected: 2,
            actual: 1
        } if name == "dynamic_shape_constraints_rank"
    ));

    let error = graph
        .set_dynamic_shape_constraints(tensor, [4, 3], [2, 16])
        .unwrap_err();
    assert!(matches!(
        error,
        Error::FrontendDynamicShapeBoundsInvalid {
            min_dimensions,
            max_dimensions,
        } if min_dimensions == vec![4, 3] && max_dimensions == vec![2, 16]
    ));

    graph.clear_dynamic_shape_constraints(tensor);
    assert!(graph.dynamic_shape_constraints(tensor).is_none());

    Ok(())
}

#[test]
fn graph_operation_json_uses_tagged_representation() -> Result<()> {
    let mut graph = Graph::new();
    let shape = Shape::contiguous([1, 2, 2])?;
    let a = graph.try_tensor(TensorSpec::new(DataType::F32, shape.clone()).with_id(1))?;
    let b = graph.try_tensor(TensorSpec::new(DataType::F32, shape.clone()).with_id(2))?;
    let c = graph.try_tensor(TensorSpec::new(DataType::F32, shape).with_id(3))?;
    graph.matmul(a, b, c, DataType::F32)?;

    let json = serde_json::to_string(&graph)?;
    let value = serde_json::from_str::<serde_json::Value>(&json)?;
    let operation = &value["operations"][0];
    assert_eq!(operation["kind"], "Matmul");
    assert_eq!(operation["operation"]["a"], "1");
    assert_eq!(operation["operation"]["b"], "2");
    assert_eq!(operation["operation"]["c"], "3");

    let restored = serde_json::from_str::<Graph>(&json)?;
    assert_eq!(restored.key()?, graph.key()?);

    Ok(())
}

#[test]
fn graph_enable_dynamic_shape_for_tensor_sets_flag_and_constraints() -> Result<()> {
    let mut graph = Graph::new();
    let tensor = graph.try_tensor(
        TensorSpec::new(DataType::F32, Shape::contiguous([2, 3])?)
            .with_id(17)
            .with_name("dynamic"),
    )?;

    assert!(!graph.dynamic_shape_enabled());
    graph.enable_dynamic_shape_for_tensor(tensor, [1, 3], [8, 16])?;

    assert!(graph.dynamic_shape_enabled());
    let constraints = graph
        .dynamic_shape_constraints(tensor)
        .expect("constraints should be present");
    assert_eq!(constraints.min_dimensions, vec![1, 3]);
    assert_eq!(constraints.max_dimensions, vec![8, 16]);

    Ok(())
}

#[test]
fn graph_with_dynamic_shape_for_tensor_sets_flag_and_constraints() -> Result<()> {
    let mut graph = Graph::new();
    let tensor = graph.try_tensor(
        TensorSpec::new(DataType::F32, Shape::contiguous([2, 3])?)
            .with_id(17)
            .with_name("dynamic"),
    )?;

    let graph = graph.with_dynamic_shape_for_tensor(tensor, [1, 3], [8, 16])?;

    assert!(graph.dynamic_shape_enabled());
    assert!(
        graph
            .dynamic_shape_constraints(tensor)
            .expect("constraints should be present")
            .contains(&[4, 5])
    );

    Ok(())
}

#[test]
fn graph_enable_dynamic_shape_for_tensor_does_not_enable_on_invalid_bounds() -> Result<()> {
    let mut graph = Graph::new();
    let tensor = graph.try_tensor(
        TensorSpec::new(DataType::F32, Shape::contiguous([2, 3])?)
            .with_id(17)
            .with_name("dynamic"),
    )?;

    let error = graph
        .enable_dynamic_shape_for_tensor(tensor, [1], [8])
        .unwrap_err();

    assert!(matches!(
        error,
        Error::LengthMismatch {
            name,
            expected: 2,
            actual: 1
        } if name == "dynamic_shape_constraints_rank"
    ));
    assert!(!graph.dynamic_shape_enabled());
    assert!(graph.dynamic_shape_constraints(tensor).is_none());

    Ok(())
}

#[test]
fn graph_key_digest_is_stable_for_equal_graphs() -> Result<()> {
    let mut first = Graph::new();
    let mut second = Graph::new();
    let tensor = TensorSpec::new(DataType::F32, Shape::contiguous([2, 3])?).with_id(101);
    first.try_tensor(tensor.clone())?;
    second.try_tensor(tensor)?;

    assert_eq!(first.key_digest()?, second.key_digest()?);
    assert_eq!(first.key()?, second.key()?);

    Ok(())
}

#[test]
fn test_graph_ragged_offset_rejects_cycles() -> Result<()> {
    let mut graph = Graph::new();
    let a = graph.tensor(TensorSpec::new(DataType::F16, Shape::contiguous([4])?));
    let b = graph.tensor(TensorSpec::new(DataType::I32, Shape::contiguous([4])?));

    graph.set_ragged_offset(a, b)?;
    let error = graph.set_ragged_offset(b, a).unwrap_err();
    assert!(matches!(
        error,
        Error::FrontendRaggedOffsetCycle {
            tensor_id,
            ragged_offset,
        } if tensor_id == b && ragged_offset == a
    ));

    Ok(())
}

#[test]
fn test_lowered_graph_binds_ragged_offset_tensors() -> Result<()> {
    let mut graph = Graph::new();
    let input =
        graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2, 3])?).with_id(11));
    let offsets =
        graph.tensor(TensorSpec::new(DataType::I32, Shape::contiguous([2, 1])?).with_id(22));
    let output =
        graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2, 3])?).with_id(33));
    graph.set_ragged_offset(input, offsets)?;
    graph.pointwise(PointwiseOperation::Unary {
        mode: PointwiseMode::ReluFwd,
        input,
        output,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::NotPropagate,
        alpha1: 1.0,
        axis: None,
    });

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
    assert_eq!(
        lowered.binding_template_ids,
        vec![11.into(), 22.into(), 33.into()]
    );

    Ok(())
}

#[test]
fn test_sdpa_backward_direct_accepts_sink_tensors() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let sink = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 4, 1, 1])?,
    ));
    let sink_gradient = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 4, 1, 1])?,
    ));

    let _ = graph.sdpa_backward_direct_infer(
        SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
        DirectSdpaBackwardConfig::new()
            .with_sink(sink)
            .with_sink_gradient(sink_gradient),
    )?;

    Ok(())
}

#[test]
fn test_sdpa_backward_direct_rejects_stats_shaped_sink_tensors() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let sink = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 1])?,
    ));

    let err = graph
        .sdpa_backward_direct_infer(
            SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
            DirectSdpaBackwardConfig::new().with_sink(sink),
        )
        .unwrap_err();

    assert!(matches!(
        err,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == sink
            && operation == "sdpa backward sink shape"
            && expected == vec![1, 4, 1, 1]
            && actual == vec![2, 4, 8, 1]
    ));

    Ok(())
}

#[test]
fn test_sdpa_backward_direct_rejects_mismatched_gradient_data_type() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let d_q = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));

    let error = graph
        .sdpa_backward_direct(
            SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
            SdpaBackwardGradientTensors {
                query_gradient: d_q,
                key_gradient: d_k,
                value_gradient: d_v,
                bias_gradient: None,
                rng_dump: None,
                sink_token_gradient: None,
            },
            DirectSdpaBackwardConfig::new(),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorDataTypeMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == d_q
            && operation == "sdpa backward d_q data type"
            && expected == DataType::F16
            && actual == DataType::F32
    ));

    Ok(())
}

#[test]
fn test_sdpa_backward_direct_rejects_mismatched_gradient_shape() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 4, 8, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let d_q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 7, 16])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 16])?,
    ));

    let error = graph
        .sdpa_backward_direct(
            SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
            SdpaBackwardGradientTensors {
                query_gradient: d_q,
                key_gradient: d_k,
                value_gradient: d_v,
                bias_gradient: None,
                rng_dump: None,
                sink_token_gradient: None,
            },
            DirectSdpaBackwardConfig::new(),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == d_q
            && operation == "sdpa backward d_q shape"
            && expected == vec![2, 4, 8, 16]
            && actual == vec![2, 4, 7, 16]
    ));

    Ok(())
}
