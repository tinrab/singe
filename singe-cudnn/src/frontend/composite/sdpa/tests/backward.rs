use super::*;
use crate::frontend::operation::AttentionPagedCache;

#[test]
fn test_sdpa_backward_rejects_causal_bottom_right_with_dropout() -> Result<()> {
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
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);
    let d_q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 6, 8])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 6, 16])?,
    ));
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let dropout_mask = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 6])?,
    ));
    let dropout_scale = graph.tensor(TensorSpec::scalar_f32(2.0)?);

    let err = graph
        .sdpa_backward(
            SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
            SdpaBackwardGradientTensors {
                query_gradient: d_q,
                key_gradient: d_k,
                value_gradient: d_v,
                bias_gradient: None,
                rng_dump: None,
                sink_token_gradient: None,
            },
            AttentionBackwardConfig::new(DataType::F32)
                .with_causal_bottom_right(sequence_length_query, sequence_length_key_value)
                .with_dropout(dropout_mask, dropout_scale),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa causal bottom right modifiers"
    ));

    Ok(())
}

#[test]
fn test_attention_backward_support_surface_rejects_causal_bottom_right_before_cudnn_907()
-> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(100);
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 64, 1])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 16])?,
    ));
    let d_q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 8])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 8])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 16])?,
    ));
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));

    let err = graph
        .validate_attention_backward_support_surface_for_version(
            90699,
            q,
            k,
            v,
            o,
            stats,
            d_o,
            d_q,
            d_k,
            d_v,
            &AttentionBackwardConfig::new(DataType::F32)
                .with_causal_bottom_right(sequence_length_query, sequence_length_key_value),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa backward causal bottom right version"
    ));

    Ok(())
}

#[test]
fn test_attention_backward_support_surface_rejects_causal_bottom_right_before_blackwell()
-> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(90);
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 64, 1])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 16])?,
    ));
    let d_q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 8])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 8])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 16])?,
    ));
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));

    let err = graph
        .validate_attention_backward_support_surface_for_version(
            90700,
            q,
            k,
            v,
            o,
            stats,
            d_o,
            d_q,
            d_k,
            d_v,
            &AttentionBackwardConfig::new(DataType::F32)
                .with_causal_bottom_right(sequence_length_query, sequence_length_key_value),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa backward causal bottom right architecture"
    ));

    Ok(())
}

#[test]
fn test_attention_backward_support_surface_rejects_causal_bottom_right_with_sq_gt_skv() -> Result<()>
{
    let mut graph = Graph::new();
    graph.set_sm_version(100);
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 128, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 128, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 128, 1])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 128, 16])?,
    ));
    let d_q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 128, 8])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 8])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 16])?,
    ));
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));

    let err = graph
        .validate_attention_backward_support_surface_for_version(
            90700,
            q,
            k,
            v,
            o,
            stats,
            d_o,
            d_q,
            d_k,
            d_v,
            &AttentionBackwardConfig::new(DataType::F32)
                .with_causal_bottom_right(sequence_length_query, sequence_length_key_value),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa backward causal bottom right sequence lengths"
    ));

    Ok(())
}

#[test]
fn test_attention_backward_support_surface_rejects_causal_bottom_right_non_multiple_of_64()
-> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(100);
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 96, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 96, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 96, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 96, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 96, 1])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 96, 16])?,
    ));
    let d_q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 96, 8])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 96, 8])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 96, 16])?,
    ));
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));

    let err = graph
        .validate_attention_backward_support_surface_for_version(
            90700,
            q,
            k,
            v,
            o,
            stats,
            d_o,
            d_q,
            d_k,
            d_v,
            &AttentionBackwardConfig::new(DataType::F32)
                .with_causal_bottom_right(sequence_length_query, sequence_length_key_value),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa backward causal bottom right multiples of 64"
    ));

    Ok(())
}

#[test]
fn test_attention_backward_support_surface_rejects_causal_bottom_right_with_internal_dropout()
-> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(100);
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 64, 1])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 16])?,
    ));
    let d_q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 8])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 8])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 16])?,
    ));
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));

    let err = graph
        .validate_attention_backward_support_surface_for_version(
            90700,
            q,
            k,
            v,
            o,
            stats,
            d_o,
            d_q,
            d_k,
            d_v,
            &AttentionBackwardConfig::new(DataType::F32)
                .with_causal_bottom_right(sequence_length_query, sequence_length_key_value)
                .with_dropout_probability(0.25, 7),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa backward causal bottom right dropout"
    ));

    Ok(())
}

#[test]
fn test_sdpa_with_modifiers_rejects_boolean_bias() -> Result<()> {
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
        Shape::contiguous([2, 2, 6, 12])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 4, 12])?,
    ));
    let bias = graph.tensor(TensorSpec::new(
        DataType::Boolean,
        Shape::contiguous([1, 2, 4, 6])?,
    ));

    let err = graph
        .sdpa_with_modifiers(
            SdpaInputs {
                query: q,
                key: k,
                value: v,
                scale: scale,
            },
            SdpaOutputTensors {
                output: output,
                stats: None,
                logit_max: None,
                score_sum_exp: None,
            },
            SdpaScoreModifiers::new().with_bias(bias),
            SoftmaxConfig::new(DataType::F32),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa bias data type"
    ));

    Ok(())
}

#[test]
fn test_sdpa_with_modifiers_rejects_non_divisible_head_counts() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 3, 4, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 6, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 6, 12])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 3, 4, 12])?,
    ));

    let err = graph
        .sdpa_with_modifiers(
            SdpaInputs {
                query: q,
                key: k,
                value: v,
                scale: scale,
            },
            SdpaOutputTensors {
                output: output,
                stats: None,
                logit_max: None,
                score_sum_exp: None,
            },
            SdpaScoreModifiers::new(),
            SoftmaxConfig::new(DataType::F32),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa attention heads"
    ));

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_rejects_sink_token_for_decode_only() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 1, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 6, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 6, 12])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);
    let sink_token = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 1, 1])?,
    ));

    let err = graph
        .sdpa_auto_infer(
            SdpaInputs {
                query: q,
                key: k,
                value: v,
                scale: scale,
            },
            AttentionConfig::new(DataType::F32).with_sink_token(sink_token),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa sink token decode"
    ));

    Ok(())
}

#[test]
fn test_attention_support_surface_rejects_decode_only_blackwell_large_heads_before_910()
-> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(100);
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 1, 192])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 192])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 160])?,
    ));

    let error = graph
        .validate_attention_support_surface_for_version(
            90900,
            q,
            k,
            v,
            AttentionConfig::new(DataType::F32),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa decode head dimension"
    ));

    Ok(())
}

#[test]
fn test_attention_support_surface_rejects_decode_only_sliding_window_before_910() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 1, 64])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 64])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 64])?,
    ));

    let error = graph
        .validate_attention_support_surface_for_version(
            90900,
            q,
            k,
            v,
            AttentionConfig::new(DataType::F32).with_sliding_window(4, 0),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa sliding window modifiers"
    ));

    Ok(())
}

#[test]
fn test_attention_support_surface_rejects_non_causal_sliding_window_before_cudnn_91002()
-> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 64])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 64])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 64])?,
    ));

    let error = graph
        .validate_attention_support_surface_for_version(
            90999,
            q,
            k,
            v,
            AttentionConfig::new(DataType::F32).with_sliding_window(4, 0),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa sliding window modifiers"
    ));

    Ok(())
}

#[test]
fn test_attention_support_surface_rejects_sliding_window_with_bias_before_cudnn_91002() -> Result<()>
{
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 64])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 64])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 64])?,
    ));
    let bias = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 8])?,
    ));

    let error = graph
        .validate_attention_support_surface_for_version(
            90999,
            q,
            k,
            v,
            AttentionConfig::new(DataType::F32)
                .with_bias(bias)
                .with_causal_mask()
                .with_sliding_window(4, 0),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa sliding window modifiers"
    ));

    Ok(())
}

#[test]
fn test_attention_support_surface_rejects_sliding_window_with_dropout_before_cudnn_91002()
-> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 64])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 64])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 64])?,
    ));

    let error = graph
        .validate_attention_support_surface_for_version(
            90999,
            q,
            k,
            v,
            AttentionConfig::new(DataType::F32)
                .with_causal_mask()
                .with_sliding_window(4, 0)
                .with_dropout_probability(0.25, 7),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa sliding window modifiers"
    ));

    Ok(())
}

#[test]
fn test_attention_support_surface_rejects_bottom_right_non_multiple_of_64_before_cudnn_906()
-> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 65, 64])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 65, 64])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 65, 64])?,
    ));
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));

    let error = graph
        .validate_attention_support_surface_for_version(
            90599,
            q,
            k,
            v,
            AttentionConfig::new(DataType::F32)
                .with_causal_bottom_right(sequence_length_query, sequence_length_key_value),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa causal bottom right sequence lengths"
    ));

    Ok(())
}

#[test]
fn test_attention_support_surface_rejects_bottom_right_sliding_window_with_sq_ne_skv_before_cudnn_906()
-> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 128, 64])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 128, 64])?,
    ));
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));

    let error = graph
        .validate_attention_support_surface_for_version(
            90599,
            q,
            k,
            v,
            AttentionConfig::new(DataType::F32)
                .with_causal_bottom_right(sequence_length_query, sequence_length_key_value)
                .with_sliding_window(16, 0),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa sliding window alignment"
    ));

    Ok(())
}

#[test]
fn test_attention_support_surface_rejects_paged_attention_large_heads_before_910() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(90);
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 160])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 16, 160])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 16, 144])?,
    ));
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let page_table_k = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let page_table_v = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));

    let error = graph
        .validate_attention_support_surface_for_version(
            90900,
            q,
            k,
            v,
            AttentionConfig::new(DataType::F32)
                .with_padding_mask(sequence_length_query, sequence_length_key_value)
                .with_paged_k(AttentionPagedCache::new(k, page_table_k))
                .with_paged_v(AttentionPagedCache::new(v, page_table_v)),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa paged attention head dimension"
    ));

    Ok(())
}

#[test]
fn test_attention_support_surface_rejects_paged_attention_with_ragged_offsets_before_907()
-> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(90);
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 64])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 16, 64])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 16, 64])?,
    ));
    let offsets = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    graph.set_ragged_offset(q, offsets)?;
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let page_table_k = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));

    let error = graph
        .validate_attention_support_surface_for_version(
            90600,
            q,
            k,
            v,
            AttentionConfig::new(DataType::F32)
                .with_padding_mask(sequence_length_query, sequence_length_key_value)
                .with_paged_k(AttentionPagedCache::new(k, page_table_k)),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa paged attention ragged offsets"
    ));

    Ok(())
}

#[test]
fn test_attention_support_surface_rejects_packed_paged_tables_before_91002() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(90);
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 64])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 16, 64])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 16, 64])?,
    ));
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let page_table_k = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let page_offsets = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    graph.set_ragged_offset(page_table_k, page_offsets)?;

    let error = graph
        .validate_attention_support_surface_for_version(
            90999,
            q,
            k,
            v,
            AttentionConfig::new(DataType::F32)
                .with_padding_mask(sequence_length_query, sequence_length_key_value)
                .with_paged_k(AttentionPagedCache::new(k, page_table_k)),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa paged k packed page table version"
    ));

    Ok(())
}

#[test]
fn test_sdpa_backward_rejects_internal_and_custom_dropout_together() -> Result<()> {
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
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);
    let d_q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 6, 8])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 6, 16])?,
    ));
    let dropout_mask = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 6])?,
    ));
    let dropout_scale = graph.tensor(TensorSpec::scalar_f32(2.0)?);

    let err = graph
        .sdpa_backward(
            SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
            SdpaBackwardGradientTensors {
                query_gradient: d_q,
                key_gradient: d_k,
                value_gradient: d_v,
                bias_gradient: None,
                rng_dump: None,
                sink_token_gradient: None,
            },
            AttentionBackwardConfig::new(DataType::F32)
                .with_dropout(dropout_mask, dropout_scale)
                .with_dropout_probability(0.25, 7),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa dropout"
    ));

    Ok(())
}

#[test]
fn test_sdpa_backward_infer_creates_gradient_shapes() -> Result<()> {
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
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);

    let outputs = graph.sdpa_backward_infer(
        SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
        AttentionBackwardConfig::new(DataType::F32).with_bias_gradient(),
    )?;
    let d_q = outputs.query_gradient;
    let d_k = outputs.key_gradient;
    let d_v = outputs.value_gradient;
    let d_bias = outputs.bias_gradient;

    assert_eq!(graph.shape(d_q)?.dimensions(), &[1, 2, 4, 8]);
    assert_eq!(graph.shape(d_k)?.dimensions(), &[1, 2, 6, 8]);
    assert_eq!(graph.shape(d_v)?.dimensions(), &[1, 2, 6, 16]);
    assert_eq!(
        graph.shape(d_bias.expect("bias_gradient"))?.dimensions(),
        &[1, 2, 4, 6]
    );

    Ok(())
}

#[test]
fn test_sdpa_fp8_backward_infer_creates_gradient_and_absolute_max_shapes() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_q = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let descale_k = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let descale_v = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let descale_o = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let descale_d_o = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let descale_s = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let descale_d_p = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let scale_s = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let scale_d_q = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let scale_d_k = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let scale_d_v = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));
    let scale_d_p = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1])?));

    let outputs = graph.sdpa_fp8_backward_infer(
        SdpaFp8BackwardInputs::new(
            q,
            k,
            v,
            o,
            d_o,
            stats,
            attention_scale,
            descale_q,
            descale_k,
            descale_v,
            descale_o,
            descale_d_o,
            descale_s,
            descale_d_p,
            scale_s,
            scale_d_q,
            scale_d_k,
            scale_d_v,
            scale_d_p,
        ),
        AttentionBackwardConfig::new(DataType::F32),
    )?;
    let d_q = outputs.query_gradient;
    let d_k = outputs.key_gradient;
    let d_v = outputs.value_gradient;
    let absolute_max_d_q = outputs.absolute_max_query_gradient;
    let absolute_max_d_k = outputs.absolute_max_key_gradient;
    let absolute_max_d_v = outputs.absolute_max_value_gradient;
    let absolute_max_d_p = outputs.absolute_max_probability_gradient;

    assert_eq!(graph.shape(d_q)?.dimensions(), &[1, 2, 4, 8]);
    assert_eq!(graph.shape(d_k)?.dimensions(), &[1, 2, 6, 8]);
    assert_eq!(graph.shape(d_v)?.dimensions(), &[1, 2, 6, 16]);
    assert_eq!(graph.shape(absolute_max_d_q)?.dimensions(), &[1, 1, 1, 1]);
    assert_eq!(graph.shape(absolute_max_d_k)?.dimensions(), &[1, 1, 1, 1]);
    assert_eq!(graph.shape(absolute_max_d_v)?.dimensions(), &[1, 1, 1, 1]);
    assert_eq!(graph.shape(absolute_max_d_p)?.dimensions(), &[1, 1, 1, 1]);

    Ok(())
}

#[test]
fn test_sdpa_fp8_backward_infer_uses_bf16_outputs_on_blackwell() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(100);
    graph.set_io_data_type(DataType::BF16);
    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 128])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 128])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 128])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 128])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 128])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_d_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_d_p = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_p = graph.tensor(TensorSpec::scalar_f32(1.0)?);

    let outputs = graph.sdpa_fp8_backward_infer(
        SdpaFp8BackwardInputs::new(
            q,
            k,
            v,
            o,
            d_o,
            stats,
            attention_scale,
            descale_q,
            descale_k,
            descale_v,
            descale_o,
            descale_d_o,
            descale_s,
            descale_d_p,
            scale_s,
            scale_d_q,
            scale_d_k,
            scale_d_v,
            scale_d_p,
        ),
        AttentionBackwardConfig::new(DataType::F32),
    )?;
    let d_q = outputs.query_gradient;
    let d_k = outputs.key_gradient;
    let d_v = outputs.value_gradient;

    assert_eq!(graph.tensor_config(d_q)?.data_type, DataType::BF16);
    assert_eq!(graph.tensor_config(d_k)?.data_type, DataType::BF16);
    assert_eq!(graph.tensor_config(d_v)?.data_type, DataType::BF16);

    Ok(())
}

#[test]
fn test_sdpa_fp8_backward_infer_accepts_bottom_right_causal_inputs_on_blackwell() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(100);

    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 64, 128])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 128, 128])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 128, 128])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 64, 128])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 64, 128])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 64, 1])?,
    ));
    let d_q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 64, 128])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 128, 128])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 128, 128])?,
    ));
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let config = AttentionBackwardConfig::new(DataType::F32)
        .with_causal_bottom_right(sequence_length_query, sequence_length_key_value);

    graph.validate_fp8_backward_support_surface_for_version(
        92100,
        DataType::F32,
        128,
        128,
        false,
        &config,
    )?;
    graph.validate_attention_backward_support_surface_for_version(
        92100, q, k, v, o, stats, d_o, d_q, d_k, d_v, &config,
    )?;

    Ok(())
}

#[test]
fn test_sdpa_fp8_backward_rejects_sliding_window_when_q_exceeds_k_without_padding() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 8, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 8, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 8, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 8, 1])?,
    ));
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_d_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_d_p = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_p = graph.tensor(TensorSpec::scalar_f32(1.0)?);

    let error = graph
        .sdpa_fp8_backward_infer(
            SdpaFp8BackwardInputs::new(
                q,
                k,
                v,
                o,
                d_o,
                stats,
                attention_scale,
                descale_q,
                descale_k,
                descale_v,
                descale_o,
                descale_d_o,
                descale_s,
                descale_d_p,
                scale_s,
                scale_d_q,
                scale_d_k,
                scale_d_v,
                scale_d_p,
            ),
            AttentionBackwardConfig::new(DataType::F32).with_sliding_window(2, 0),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa fp8 backward sliding window sequence lengths"
    ));

    Ok(())
}

#[test]
fn test_sdpa_fp8_backward_infer_accepts_custom_score_subgraphs() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_d_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_d_p = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_p = graph.tensor(TensorSpec::scalar_f32(1.0)?);

    let mut score_subgraph = Graph::new();
    let score_input = score_subgraph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 6])?,
    ));
    let score_scale = score_subgraph.tensor(TensorSpec::scalar_f32(0.8)?);
    let score_output = score_subgraph.pointwise_binary_infer(
        score_input,
        score_scale,
        PointwiseMode::Mul,
        DataType::F32,
    )?;

    let mut score_subgraph_bprop = Graph::new();
    let bprop_input = score_subgraph_bprop.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 6])?,
    ));
    let bprop_scale = score_subgraph_bprop.tensor(TensorSpec::scalar_f32(0.5)?);
    let bprop_output = score_subgraph_bprop.pointwise_binary_infer(
        bprop_input,
        bprop_scale,
        PointwiseMode::Mul,
        DataType::F32,
    )?;

    let outputs = graph.sdpa_fp8_backward_infer(
        SdpaFp8BackwardInputs::new(
            q,
            k,
            v,
            o,
            d_o,
            stats,
            attention_scale,
            descale_q,
            descale_k,
            descale_v,
            descale_o,
            descale_d_o,
            descale_s,
            descale_d_p,
            scale_s,
            scale_d_q,
            scale_d_k,
            scale_d_v,
            scale_d_p,
        ),
        AttentionBackwardConfig::new(DataType::F32)
            .with_score_subgraph(SdpaScoreSubgraph::new(
                score_subgraph,
                score_input,
                score_output,
            ))
            .with_score_subgraph_bprop(SdpaScoreSubgraph::new(
                score_subgraph_bprop,
                bprop_input,
                bprop_output,
            )),
    )?;
    let d_q = outputs.query_gradient;
    let d_k = outputs.key_gradient;
    let d_v = outputs.value_gradient;
    let absolute_max_d_q = outputs.absolute_max_query_gradient;
    let absolute_max_d_k = outputs.absolute_max_key_gradient;
    let absolute_max_d_v = outputs.absolute_max_value_gradient;
    let absolute_max_d_p = outputs.absolute_max_probability_gradient;

    assert_eq!(graph.shape(d_q)?.dimensions(), &[1, 2, 4, 8]);
    assert_eq!(graph.shape(d_k)?.dimensions(), &[1, 2, 6, 8]);
    assert_eq!(graph.shape(d_v)?.dimensions(), &[1, 2, 6, 16]);
    assert_eq!(graph.shape(absolute_max_d_q)?.dimensions(), &[1, 1, 1, 1]);
    assert_eq!(graph.shape(absolute_max_d_k)?.dimensions(), &[1, 1, 1, 1]);
    assert_eq!(graph.shape(absolute_max_d_v)?.dimensions(), &[1, 1, 1, 1]);
    assert_eq!(graph.shape(absolute_max_d_p)?.dimensions(), &[1, 1, 1, 1]);

    Ok(())
}

#[test]
fn test_sdpa_fp8_backward_infer_accepts_inline_attn_scale_value() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_d_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_d_p = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_p = graph.tensor(TensorSpec::scalar_f32(1.0)?);

    let outputs = graph.sdpa_fp8_backward_infer(
        SdpaFp8BackwardInputs::new(
            q,
            k,
            v,
            o,
            d_o,
            stats,
            attention_scale,
            descale_q,
            descale_k,
            descale_v,
            descale_o,
            descale_d_o,
            descale_s,
            descale_d_p,
            scale_s,
            scale_d_q,
            scale_d_k,
            scale_d_v,
            scale_d_p,
        ),
        AttentionBackwardConfig::new(DataType::F32).with_attention_scale(0.5),
    )?;
    let d_q = outputs.query_gradient;
    let d_k = outputs.key_gradient;
    let d_v = outputs.value_gradient;
    let absolute_max_d_q = outputs.absolute_max_query_gradient;
    let absolute_max_d_k = outputs.absolute_max_key_gradient;
    let absolute_max_d_v = outputs.absolute_max_value_gradient;
    let absolute_max_d_p = outputs.absolute_max_probability_gradient;

    assert_eq!(graph.shape(d_q)?.dimensions(), &[1, 2, 4, 8]);
    assert_eq!(graph.shape(d_k)?.dimensions(), &[1, 2, 6, 8]);
    assert_eq!(graph.shape(d_v)?.dimensions(), &[1, 2, 6, 16]);
    assert_eq!(graph.shape(absolute_max_d_q)?.dimensions(), &[1, 1, 1, 1]);
    assert_eq!(graph.shape(absolute_max_d_k)?.dimensions(), &[1, 1, 1, 1]);
    assert_eq!(graph.shape(absolute_max_d_v)?.dimensions(), &[1, 1, 1, 1]);
    assert_eq!(graph.shape(absolute_max_d_p)?.dimensions(), &[1, 1, 1, 1]);

    Ok(())
}

#[test]
fn test_sdpa_fp8_backward_infer_accepts_custom_score_subgraph_with_bias() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_d_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_d_p = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_p = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let bias = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 6])?,
    ));

    let mut score_subgraph = Graph::new();
    let score_input = score_subgraph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 6])?,
    ));
    let score_scale = score_subgraph.tensor(TensorSpec::scalar_f32(0.8)?);
    let score_output = score_subgraph.pointwise_binary_infer(
        score_input,
        score_scale,
        PointwiseMode::Mul,
        DataType::F32,
    )?;

    let outputs = graph.sdpa_fp8_backward_infer(
        SdpaFp8BackwardInputs::new(
            q,
            k,
            v,
            o,
            d_o,
            stats,
            attention_scale,
            descale_q,
            descale_k,
            descale_v,
            descale_o,
            descale_d_o,
            descale_s,
            descale_d_p,
            scale_s,
            scale_d_q,
            scale_d_k,
            scale_d_v,
            scale_d_p,
        ),
        AttentionBackwardConfig::new(DataType::F32)
            .with_bias(bias)
            .with_score_subgraph(SdpaScoreSubgraph::new(
                score_subgraph,
                score_input,
                score_output,
            )),
    )?;
    let d_q = outputs.query_gradient;
    let d_k = outputs.key_gradient;
    let d_v = outputs.value_gradient;
    let absolute_max_d_q = outputs.absolute_max_query_gradient;
    let absolute_max_d_k = outputs.absolute_max_key_gradient;
    let absolute_max_d_v = outputs.absolute_max_value_gradient;
    let absolute_max_d_p = outputs.absolute_max_probability_gradient;

    assert_eq!(graph.shape(d_q)?.dimensions(), &[1, 2, 4, 8]);
    assert_eq!(graph.shape(d_k)?.dimensions(), &[1, 2, 6, 8]);
    assert_eq!(graph.shape(d_v)?.dimensions(), &[1, 2, 6, 16]);
    assert_eq!(graph.shape(absolute_max_d_q)?.dimensions(), &[1, 1, 1, 1]);
    assert_eq!(graph.shape(absolute_max_d_k)?.dimensions(), &[1, 1, 1, 1]);
    assert_eq!(graph.shape(absolute_max_d_v)?.dimensions(), &[1, 1, 1, 1]);
    assert_eq!(graph.shape(absolute_max_d_p)?.dimensions(), &[1, 1, 1, 1]);

    Ok(())
}

#[test]
fn test_sdpa_fp8_backward_infer_with_aux_creates_d_sink_token_shape() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_d_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_d_p = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_p = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let sink_token = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 1, 1])?,
    ));

    let outputs = graph.sdpa_fp8_backward_infer_with_aux(
        SdpaFp8BackwardInputs::new(
            q,
            k,
            v,
            o,
            d_o,
            stats,
            attention_scale,
            descale_q,
            descale_k,
            descale_v,
            descale_o,
            descale_d_o,
            descale_s,
            descale_d_p,
            scale_s,
            scale_d_q,
            scale_d_k,
            scale_d_v,
            scale_d_p,
        ),
        AttentionBackwardConfig::new(DataType::F32)
            .with_sink_token(sink_token)
            .with_sink_token_gradient(),
    )?;
    let d_sink_token = outputs.sink_token_gradient;

    assert_eq!(
        graph
            .shape(d_sink_token.expect("d_sink_token"))?
            .dimensions(),
        &[1, 2, 1, 1]
    );
    assert_eq!(
        graph
            .tensor_config(d_sink_token.expect("d_sink_token"))?
            .data_type,
        DataType::F32
    );

    Ok(())
}

#[test]
fn test_sdpa_fp8_backward_rejects_bias_gradient() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_d_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_d_p = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_p = graph.tensor(TensorSpec::scalar_f32(1.0)?);

    let error = graph
        .sdpa_fp8_backward_infer_with_aux(
            SdpaFp8BackwardInputs::new(
                q,
                k,
                v,
                o,
                d_o,
                stats,
                attention_scale,
                descale_q,
                descale_k,
                descale_v,
                descale_o,
                descale_d_o,
                descale_s,
                descale_d_p,
                scale_s,
                scale_d_q,
                scale_d_k,
                scale_d_v,
                scale_d_p,
            ),
            AttentionBackwardConfig::new(DataType::F32).with_bias_gradient(),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa fp8 backward d_bias"
    ));

    Ok(())
}

#[test]
fn test_sdpa_fp8_backward_rejects_rng_dump() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_d_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_d_p = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_p = graph.tensor(TensorSpec::scalar_f32(1.0)?);

    let error = graph
        .sdpa_fp8_backward_infer_with_aux(
            SdpaFp8BackwardInputs::new(
                q,
                k,
                v,
                o,
                d_o,
                stats,
                attention_scale,
                descale_q,
                descale_k,
                descale_v,
                descale_o,
                descale_d_o,
                descale_s,
                descale_d_p,
                scale_s,
                scale_d_q,
                scale_d_k,
                scale_d_v,
                scale_d_p,
            ),
            AttentionBackwardConfig::new(DataType::F32).with_random_number_generator_dump(),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa fp8 backward rng_dump"
    ));

    Ok(())
}

#[test]
fn test_sdpa_fp8_backward_with_aux_uses_provided_outputs() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_d_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_d_p = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_p = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let d_q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 8])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 16])?,
    ));
    let absolute_max_d_q = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let absolute_max_d_k = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let absolute_max_d_v = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let absolute_max_d_p = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));

    graph.sdpa_fp8_backward_with_aux(
        SdpaFp8BackwardInputs::new(
            q,
            k,
            v,
            o,
            d_o,
            stats,
            attention_scale,
            descale_q,
            descale_k,
            descale_v,
            descale_o,
            descale_d_o,
            descale_s,
            descale_d_p,
            scale_s,
            scale_d_q,
            scale_d_k,
            scale_d_v,
            scale_d_p,
        ),
        SdpaFp8BackwardGradientTensors {
            query_gradient: d_q,
            key_gradient: d_k,
            value_gradient: d_v,
            sink_token_gradient: None,
            absolute_max_query_gradient: absolute_max_d_q,
            absolute_max_key_gradient: absolute_max_d_k,
            absolute_max_value_gradient: absolute_max_d_v,
            absolute_max_probability_gradient: absolute_max_d_p,
        },
        AttentionBackwardConfig::new(DataType::F32),
    )?;

    for output in [
        d_q,
        d_k,
        d_v,
        absolute_max_d_q,
        absolute_max_d_k,
        absolute_max_d_v,
        absolute_max_d_p,
    ] {
        assert!(
            graph
                .operations
                .iter()
                .any(|operation| matches!(operation, Operation::Reshape { output: op_output, .. } if *op_output == output))
        );
    }

    Ok(())
}

#[test]
fn test_sdpa_fp8_backward_with_aux_uses_bf16_outputs_on_blackwell() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(100);
    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 128])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 128])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 128])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 128])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 128])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_d_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_d_p = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_p = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let d_q = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 4, 128])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 6, 128])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 6, 128])?,
    ));
    let absolute_max_d_q = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let absolute_max_d_k = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let absolute_max_d_v = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let absolute_max_d_p = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));

    graph.sdpa_fp8_backward_with_aux(
        SdpaFp8BackwardInputs::new(
            q,
            k,
            v,
            o,
            d_o,
            stats,
            attention_scale,
            descale_q,
            descale_k,
            descale_v,
            descale_o,
            descale_d_o,
            descale_s,
            descale_d_p,
            scale_s,
            scale_d_q,
            scale_d_k,
            scale_d_v,
            scale_d_p,
        ),
        SdpaFp8BackwardGradientTensors {
            query_gradient: d_q,
            key_gradient: d_k,
            value_gradient: d_v,
            sink_token_gradient: None,
            absolute_max_query_gradient: absolute_max_d_q,
            absolute_max_key_gradient: absolute_max_d_k,
            absolute_max_value_gradient: absolute_max_d_v,
            absolute_max_probability_gradient: absolute_max_d_p,
        },
        AttentionBackwardConfig::new(DataType::F32),
    )?;

    Ok(())
}

#[test]
fn test_sdpa_fp8_backward_with_aux_accepts_mixed_output_data_types_on_blackwell() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(100);
    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 128])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 128])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 128])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 128])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 128])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_d_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_d_p = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_p = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let d_q = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 4, 128])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 6, 128])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 128])?,
    ));
    let absolute_max_d_q = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let absolute_max_d_k = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let absolute_max_d_v = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let absolute_max_d_p = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));

    graph.sdpa_fp8_backward_with_aux(
        SdpaFp8BackwardInputs::new(
            q,
            k,
            v,
            o,
            d_o,
            stats,
            attention_scale,
            descale_q,
            descale_k,
            descale_v,
            descale_o,
            descale_d_o,
            descale_s,
            descale_d_p,
            scale_s,
            scale_d_q,
            scale_d_k,
            scale_d_v,
            scale_d_p,
        ),
        SdpaFp8BackwardGradientTensors {
            query_gradient: d_q,
            key_gradient: d_k,
            value_gradient: d_v,
            sink_token_gradient: None,
            absolute_max_query_gradient: absolute_max_d_q,
            absolute_max_key_gradient: absolute_max_d_k,
            absolute_max_value_gradient: absolute_max_d_v,
            absolute_max_probability_gradient: absolute_max_d_p,
        },
        AttentionBackwardConfig::new(DataType::F32),
    )?;

    Ok(())
}

#[test]
fn test_sdpa_fp8_backward_infer_with_aux_and_output_types_accepts_mixed_output_data_types_on_blackwell()
-> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(100);
    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 128])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 128])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 128])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 128])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 128])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_d_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_d_p = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_p = graph.tensor(TensorSpec::scalar_f32(1.0)?);

    let outputs = graph.sdpa_fp8_backward_infer_with_aux_and_output_types(
        SdpaFp8BackwardInputs::new(
            q,
            k,
            v,
            o,
            d_o,
            stats,
            attention_scale,
            descale_q,
            descale_k,
            descale_v,
            descale_o,
            descale_d_o,
            descale_s,
            descale_d_p,
            scale_s,
            scale_d_q,
            scale_d_k,
            scale_d_v,
            scale_d_p,
        ),
        DataType::BF16,
        DataType::F16,
        DataType::F8E4M3,
        AttentionBackwardConfig::new(DataType::F32),
    )?;
    let d_q = outputs.query_gradient;
    let d_k = outputs.key_gradient;
    let d_v = outputs.value_gradient;

    assert_eq!(graph.tensor_config(d_q)?.data_type, DataType::BF16);
    assert_eq!(graph.tensor_config(d_k)?.data_type, DataType::F16);
    assert_eq!(graph.tensor_config(d_v)?.data_type, DataType::F8E4M3);

    Ok(())
}

#[test]
fn test_sdpa_fp8_backward_with_aux_rejects_ragged_outputs_on_hopper() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(90);
    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_d_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_d_p = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_p = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let d_q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 8])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 16])?,
    ));
    let d_q_offsets = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 4, 1])?,
    ));
    graph.set_ragged_offset(d_q, d_q_offsets)?;
    let absolute_max_d_q = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let absolute_max_d_k = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let absolute_max_d_v = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let absolute_max_d_p = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));

    let error = graph
        .sdpa_fp8_backward_with_aux(
            SdpaFp8BackwardInputs::new(
                q,
                k,
                v,
                o,
                d_o,
                stats,
                attention_scale,
                descale_q,
                descale_k,
                descale_v,
                descale_o,
                descale_d_o,
                descale_s,
                descale_d_p,
                scale_s,
                scale_d_q,
                scale_d_k,
                scale_d_v,
                scale_d_p,
            ),
            SdpaFp8BackwardGradientTensors {
                query_gradient: d_q,
                key_gradient: d_k,
                value_gradient: d_v,
                sink_token_gradient: None,
                absolute_max_query_gradient: absolute_max_d_q,
                absolute_max_key_gradient: absolute_max_d_k,
                absolute_max_value_gradient: absolute_max_d_v,
                absolute_max_probability_gradient: absolute_max_d_p,
            },
            AttentionBackwardConfig::new(DataType::F32),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa fp8 backward ragged outputs"
    ));

    Ok(())
}

#[test]
fn test_sdpa_fp8_backward_builder_uses_provided_outputs() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_d_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_d_p = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_p = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let d_q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 8])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 16])?,
    ));
    let absolute_max_d_q = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let absolute_max_d_k = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let absolute_max_d_v = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let absolute_max_d_p = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));

    graph.sdpa_fp8_backward(
        SdpaFp8BackwardInputs::new(
            q,
            k,
            v,
            o,
            d_o,
            stats,
            attention_scale,
            descale_q,
            descale_k,
            descale_v,
            descale_o,
            descale_d_o,
            descale_s,
            descale_d_p,
            scale_s,
            scale_d_q,
            scale_d_k,
            scale_d_v,
            scale_d_p,
        ),
        SdpaFp8BackwardGradientTensors::new(
            d_q,
            d_k,
            d_v,
            absolute_max_d_q,
            absolute_max_d_k,
            absolute_max_d_v,
            absolute_max_d_p,
        ),
        AttentionBackwardConfig::new(DataType::F32),
    )?;

    assert!(
        graph.operations.iter().any(
            |operation| matches!(operation, Operation::Reshape { output, .. } if *output == d_q)
        )
    );

    Ok(())
}

#[test]
fn test_sdpa_fp8_backward_builder_uses_bf16_outputs_on_blackwell() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(100);
    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 128])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 128])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 128])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 128])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 128])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_d_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_d_p = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_p = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let d_q = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 4, 128])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 6, 128])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 6, 128])?,
    ));
    let absolute_max_d_q = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let absolute_max_d_k = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let absolute_max_d_v = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let absolute_max_d_p = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));

    graph.sdpa_fp8_backward(
        SdpaFp8BackwardInputs::new(
            q,
            k,
            v,
            o,
            d_o,
            stats,
            attention_scale,
            descale_q,
            descale_k,
            descale_v,
            descale_o,
            descale_d_o,
            descale_s,
            descale_d_p,
            scale_s,
            scale_d_q,
            scale_d_k,
            scale_d_v,
            scale_d_p,
        ),
        SdpaFp8BackwardGradientTensors::new(
            d_q,
            d_k,
            d_v,
            absolute_max_d_q,
            absolute_max_d_k,
            absolute_max_d_v,
            absolute_max_d_p,
        ),
        AttentionBackwardConfig::new(DataType::F32),
    )?;

    Ok(())
}

#[test]
fn test_sdpa_fp8_backward_infer_rejects_non_scalar_scale_tensor() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let attention_scale = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2])?));
    let descale_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_d_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_d_p = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_d_p = graph.tensor(TensorSpec::scalar_f32(1.0)?);

    let err = graph
        .sdpa_fp8_backward_infer(
            SdpaFp8BackwardInputs::new(
                q,
                k,
                v,
                o,
                d_o,
                stats,
                attention_scale,
                descale_q,
                descale_k,
                descale_v,
                descale_o,
                descale_d_o,
                descale_s,
                descale_d_p,
                scale_s,
                scale_d_q,
                scale_d_k,
                scale_d_v,
                scale_d_p,
            ),
            AttentionBackwardConfig::new(DataType::F32),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa fp8 backward attention_scale"
    ));

    Ok(())
}

#[test]
fn test_sdpa_backward_infer_accepts_internal_dropout() -> Result<()> {
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
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);
    let offset = graph.tensor(
        TensorSpec::new(DataType::I64, Shape::contiguous([1, 1, 1, 1])?)
            .with_scalar_value(ScalarValue::I64(17))?,
    );

    let outputs = graph.sdpa_backward_infer(
        SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
        AttentionBackwardConfig::new(DataType::F32)
            .with_dropout_probability(0.25, 7)
            .with_dropout_offset(offset),
    )?;
    let d_q = outputs.query_gradient;
    let d_k = outputs.key_gradient;
    let d_v = outputs.value_gradient;
    let d_bias = outputs.bias_gradient;

    assert_eq!(graph.shape(d_q)?.dimensions(), &[1, 2, 4, 8]);
    assert_eq!(graph.shape(d_k)?.dimensions(), &[1, 2, 6, 8]);
    assert_eq!(graph.shape(d_v)?.dimensions(), &[1, 2, 6, 16]);
    assert!(d_bias.is_none());

    Ok(())
}

#[test]
fn test_sdpa_backward_infer_accepts_device_seed_internal_dropout() -> Result<()> {
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
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);
    let seed = graph.tensor(
        TensorSpec::new(DataType::I64, Shape::contiguous([1, 1, 1, 1])?)
            .with_scalar_value(ScalarValue::I64(7))?,
    );
    let offset = graph.tensor(
        TensorSpec::new(DataType::I64, Shape::contiguous([1, 1, 1, 1])?)
            .with_scalar_value(ScalarValue::I64(17))?,
    );

    let outputs = graph.sdpa_backward_infer_with_aux(
        SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
        AttentionBackwardConfig::new(DataType::F32)
            .with_dropout_probability_from_tensor(0.25, seed, offset)
            .with_random_number_generator_dump(),
    )?;
    let rng_dump = outputs.rng_dump;

    assert_eq!(
        graph.shape(rng_dump.expect("rng dump"))?.dimensions(),
        &[1, 2, 4, 6]
    );

    Ok(())
}

#[test]
fn test_sdpa_backward_infer_with_aux_creates_rng_dump_for_internal_dropout() -> Result<()> {
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
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);
    let offset = graph.tensor(
        TensorSpec::new(DataType::I64, Shape::contiguous([1, 1, 1, 1])?)
            .with_scalar_value(ScalarValue::I64(17))?,
    );

    let outputs = graph.sdpa_backward_infer_with_aux(
        SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
        AttentionBackwardConfig::new(DataType::F32)
            .with_dropout_probability(0.25, 7)
            .with_dropout_offset(offset)
            .with_random_number_generator_dump(),
    )?;
    let rng_dump = outputs.rng_dump;

    assert_eq!(
        graph.shape(rng_dump.expect("rng dump"))?.dimensions(),
        &[1, 2, 4, 6]
    );

    Ok(())
}

#[test]
fn test_sdpa_backward_rejects_rng_dump_without_internal_dropout() -> Result<()> {
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
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);

    let error = graph
        .sdpa_backward_infer_with_aux(
            SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
            AttentionBackwardConfig::new(DataType::F32).with_random_number_generator_dump(),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa backward rng dump"
    ));

    Ok(())
}

#[test]
fn test_sdpa_backward_rejects_dropout_offset_without_probability() -> Result<()> {
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
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let offset = graph.tensor(
        TensorSpec::new(DataType::I64, Shape::contiguous([1, 1, 1, 1])?)
            .with_scalar_value(ScalarValue::I64(123))?,
    );

    let error = graph
        .sdpa_backward_infer_with_aux(
            SdpaBackwardInputs::new(q, k, v, o, d_o, stats, attention_scale),
            AttentionBackwardConfig::new(DataType::F32).with_dropout_offset(offset),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa dropout offset"
    ));

    Ok(())
}

#[test]
fn test_sdpa_backward_rejects_non_scalar_scale() -> Result<()> {
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
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2])?));
    let d_q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 6, 8])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 6, 16])?,
    ));

    let err = graph
        .sdpa_backward(
            SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
            SdpaBackwardGradientTensors {
                query_gradient: d_q,
                key_gradient: d_k,
                value_gradient: d_v,
                bias_gradient: None,
                rng_dump: None,
                sink_token_gradient: None,
            },
            AttentionBackwardConfig::new(DataType::F32),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa backward scale"
    ));

    Ok(())
}

#[test]
fn test_sdpa_backward_rejects_wrong_stats_data_type() -> Result<()> {
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
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);
    let d_q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 6, 8])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 6, 16])?,
    ));

    let err = graph
        .sdpa_backward(
            SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
            SdpaBackwardGradientTensors {
                query_gradient: d_q,
                key_gradient: d_k,
                value_gradient: d_v,
                bias_gradient: None,
                rng_dump: None,
                sink_token_gradient: None,
            },
            AttentionBackwardConfig::new(DataType::F32),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa backward stats data type"
    ));

    Ok(())
}

#[test]
fn test_sdpa_backward_infer_with_aux_creates_d_sink_token_shape() -> Result<()> {
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
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);
    let sink_token = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 1, 1])?,
    ));

    let outputs = graph.sdpa_backward_infer_with_aux(
        SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
        AttentionBackwardConfig::new(DataType::F32)
            .with_sink_token(sink_token)
            .with_sink_token_gradient(),
    )?;
    let d_sink_token = outputs.sink_token_gradient;

    assert_eq!(
        graph
            .shape(d_sink_token.expect("d_sink_token"))?
            .dimensions(),
        &[1, 2, 1, 1]
    );

    Ok(())
}

#[test]
fn test_sdpa_backward_infer_accepts_custom_score_subgraphs() -> Result<()> {
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
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);

    let mut score_subgraph = Graph::new();
    let score_input = score_subgraph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 6])?,
    ));
    let soft_cap = score_subgraph.tensor(TensorSpec::scalar_f32(0.8)?);
    let divided = score_subgraph.pointwise_binary_infer(
        score_input,
        soft_cap,
        PointwiseMode::Div,
        DataType::F32,
    )?;
    let activated = score_subgraph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 6])?,
    ));
    score_subgraph.pointwise(PointwiseOperation::Unary {
        mode: PointwiseMode::TanhFwd,
        input: divided,
        output: activated,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        axis: None,
    });
    let score_output = score_subgraph.pointwise_binary_infer(
        activated,
        soft_cap,
        PointwiseMode::Mul,
        DataType::F32,
    )?;

    let mut score_subgraph_bprop = Graph::new();
    let bprop_input = score_subgraph_bprop.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 6])?,
    ));
    let bprop_scale = score_subgraph_bprop.tensor(TensorSpec::scalar_f32(0.5)?);
    let bprop_output = score_subgraph_bprop.pointwise_binary_infer(
        bprop_input,
        bprop_scale,
        PointwiseMode::Mul,
        DataType::F32,
    )?;

    let outputs = graph.sdpa_backward_infer(
        SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
        AttentionBackwardConfig::new(DataType::F32)
            .with_score_subgraph(SdpaScoreSubgraph::new(
                score_subgraph,
                score_input,
                score_output,
            ))
            .with_score_subgraph_bprop(SdpaScoreSubgraph::new(
                score_subgraph_bprop,
                bprop_input,
                bprop_output,
            )),
    )?;
    let d_q = outputs.query_gradient;
    let d_k = outputs.key_gradient;
    let d_v = outputs.value_gradient;

    assert_eq!(graph.shape(d_q)?.dimensions(), &[1, 2, 4, 8]);
    assert_eq!(graph.shape(d_k)?.dimensions(), &[1, 2, 6, 8]);
    assert_eq!(graph.shape(d_v)?.dimensions(), &[1, 2, 6, 16]);

    Ok(())
}

#[test]
fn test_sdpa_backward_infer_accepts_inline_attn_scale_value() -> Result<()> {
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
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);

    let outputs = graph.sdpa_backward_infer(
        SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
        AttentionBackwardConfig::new(DataType::F32).with_attention_scale(0.5),
    )?;
    let d_q = outputs.query_gradient;
    let d_k = outputs.key_gradient;
    let d_v = outputs.value_gradient;
    let d_bias = outputs.bias_gradient;

    assert_eq!(graph.shape(d_q)?.dimensions(), &[1, 2, 4, 8]);
    assert_eq!(graph.shape(d_k)?.dimensions(), &[1, 2, 6, 8]);
    assert_eq!(graph.shape(d_v)?.dimensions(), &[1, 2, 6, 16]);
    assert!(d_bias.is_none());

    Ok(())
}

#[test]
fn test_sdpa_backward_infer_accepts_custom_score_subgraph_with_additive_mask() -> Result<()> {
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
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);
    let additive_mask = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 4, 6])?,
    ));

    let mut score_subgraph = Graph::new();
    let score_input = score_subgraph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 6])?,
    ));
    let soft_cap = score_subgraph.tensor(TensorSpec::scalar_f32(0.8)?);
    let score_output = score_subgraph.pointwise_binary_infer(
        score_input,
        soft_cap,
        PointwiseMode::Mul,
        DataType::F32,
    )?;

    let outputs = graph.sdpa_backward_infer(
        SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
        AttentionBackwardConfig::new(DataType::F32)
            .with_additive_mask(additive_mask)
            .with_score_subgraph(SdpaScoreSubgraph::new(
                score_subgraph,
                score_input,
                score_output,
            )),
    )?;
    let d_q = outputs.query_gradient;
    let d_k = outputs.key_gradient;
    let d_v = outputs.value_gradient;

    assert_eq!(graph.shape(d_q)?.dimensions(), &[1, 2, 4, 8]);
    assert_eq!(graph.shape(d_k)?.dimensions(), &[1, 2, 6, 8]);
    assert_eq!(graph.shape(d_v)?.dimensions(), &[1, 2, 6, 16]);

    Ok(())
}

#[test]
fn test_sdpa_backward_infer_accepts_custom_score_subgraph_with_seq_lens() -> Result<()> {
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
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 2, 4, 1])?,
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

    let mut score_subgraph = Graph::new();
    let score_input = score_subgraph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 2, 4, 6])?,
    ));
    let soft_cap = score_subgraph.tensor(TensorSpec::scalar_f32(0.8)?);
    let score_output = score_subgraph.pointwise_binary_infer(
        score_input,
        soft_cap,
        PointwiseMode::Mul,
        DataType::F32,
    )?;

    let outputs = graph.sdpa_backward_infer(
        SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
        AttentionBackwardConfig::new(DataType::F32)
            .with_sequence_lengths(sequence_length_query, sequence_length_key_value)
            .with_score_subgraph(SdpaScoreSubgraph::new(
                score_subgraph,
                score_input,
                score_output,
            )),
    )?;
    let d_q = outputs.query_gradient;
    let d_k = outputs.key_gradient;
    let d_v = outputs.value_gradient;

    assert_eq!(graph.shape(d_q)?.dimensions(), &[2, 2, 4, 8]);
    assert_eq!(graph.shape(d_k)?.dimensions(), &[2, 2, 6, 8]);
    assert_eq!(graph.shape(d_v)?.dimensions(), &[2, 2, 6, 16]);

    Ok(())
}

#[test]
fn test_sdpa_backward_infer_accepts_custom_score_subgraph_with_bias() -> Result<()> {
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
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);
    let bias = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 6])?,
    ));

    let mut score_subgraph = Graph::new();
    let score_input = score_subgraph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 6])?,
    ));
    let soft_cap = score_subgraph.tensor(TensorSpec::scalar_f32(0.8)?);
    let score_output = score_subgraph.pointwise_binary_infer(
        score_input,
        soft_cap,
        PointwiseMode::Mul,
        DataType::F32,
    )?;

    let outputs = graph.sdpa_backward_infer(
        SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
        AttentionBackwardConfig::new(DataType::F32)
            .with_bias(bias)
            .with_score_subgraph(SdpaScoreSubgraph::new(
                score_subgraph,
                score_input,
                score_output,
            )),
    )?;
    let d_q = outputs.query_gradient;
    let d_k = outputs.key_gradient;
    let d_v = outputs.value_gradient;

    assert_eq!(graph.shape(d_q)?.dimensions(), &[1, 2, 4, 8]);
    assert_eq!(graph.shape(d_k)?.dimensions(), &[1, 2, 6, 8]);
    assert_eq!(graph.shape(d_v)?.dimensions(), &[1, 2, 6, 16]);

    Ok(())
}

#[test]
fn test_sdpa_backward_rejects_custom_score_subgraph_with_causal_mask() -> Result<()> {
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

    let mut score_subgraph = Graph::new();
    let score_input = score_subgraph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 6])?,
    ));
    let soft_cap = score_subgraph.tensor(TensorSpec::scalar_f32(0.8)?);
    let score_output = score_subgraph.pointwise_binary_infer(
        score_input,
        soft_cap,
        PointwiseMode::Mul,
        DataType::F32,
    )?;
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 6, 8])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 6, 16])?,
    ));

    let err = graph
        .validate_attention_backward_support_surface_for_version(
            92100,
            q,
            k,
            v,
            o,
            stats,
            d_o,
            d_q,
            d_k,
            d_v,
            &AttentionBackwardConfig::new(DataType::F32)
                .with_causal_mask()
                .with_score_subgraph(SdpaScoreSubgraph::new(
                    score_subgraph,
                    score_input,
                    score_output,
                )),
        )
        .unwrap_err();

    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa backward score subgraph modifiers"
    ));

    Ok(())
}

#[test]
fn test_sdpa_backward_rejects_bprop_score_subgraph_without_forward_score_subgraph() -> Result<()> {
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
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);

    let mut score_subgraph_bprop = Graph::new();
    let bprop_input = score_subgraph_bprop.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 6])?,
    ));
    let bprop_scale = score_subgraph_bprop.tensor(TensorSpec::scalar_f32(0.5)?);
    let bprop_output = score_subgraph_bprop.pointwise_binary_infer(
        bprop_input,
        bprop_scale,
        PointwiseMode::Mul,
        DataType::F32,
    )?;

    let err = graph
        .sdpa_backward_infer(
            SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
            AttentionBackwardConfig::new(DataType::F32).with_score_subgraph_bprop(
                SdpaScoreSubgraph::new(score_subgraph_bprop, bprop_input, bprop_output),
            ),
        )
        .unwrap_err();

    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa backward score subgraph bprop"
    ));

    Ok(())
}

#[test]
fn test_sdpa_backward_rejects_seq_lens_without_padding_mask_or_custom_score_subgraph() -> Result<()>
{
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
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 2, 4, 1])?,
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

    let err = graph
        .sdpa_backward_infer(
            SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
            AttentionBackwardConfig::new(DataType::F32)
                .with_sequence_lengths(sequence_length_query, sequence_length_key_value),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa backward seq lens"
    ));

    Ok(())
}

#[test]
fn test_attention_backward_support_surface_rejects_bias_gradient_with_padding_mask_before_cudnn_905()
-> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 64, 1])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let d_q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));

    let error = graph
        .validate_attention_backward_support_surface_for_version(
            90499,
            q,
            k,
            v,
            o,
            stats,
            d_o,
            d_q,
            d_k,
            d_v,
            &AttentionBackwardConfig::new(DataType::F32)
                .with_padding_mask(sequence_length_query, sequence_length_key_value)
                .with_bias_gradient(),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa backward d_bias version"
    ));

    Ok(())
}

#[test]
fn test_attention_backward_support_surface_rejects_bias_gradient_non_multiple_of_64_before_cudnn_905()
-> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 32, 64])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 96, 64])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 96, 64])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 32, 64])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 32, 1])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 32, 64])?,
    ));
    let d_q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 32, 64])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 96, 64])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 96, 64])?,
    ));

    let error = graph
        .validate_attention_backward_support_surface_for_version(
            90499,
            q,
            k,
            v,
            o,
            stats,
            d_o,
            d_q,
            d_k,
            d_v,
            &AttentionBackwardConfig::new(DataType::F32).with_bias_gradient(),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa backward d_bias sequence lengths"
    ));

    Ok(())
}

#[test]
fn test_attention_backward_support_surface_rejects_ragged_bias_gradient_on_sm10() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(100);
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 64, 1])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let d_q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let offsets = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    graph.set_ragged_offset(q, offsets)?;

    let error = graph
        .validate_attention_backward_support_surface_for_version(
            92100,
            q,
            k,
            v,
            o,
            stats,
            d_o,
            d_q,
            d_k,
            d_v,
            &AttentionBackwardConfig::new(DataType::F32).with_bias_gradient(),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa backward d_bias ragged"
    ));

    Ok(())
}

#[test]
fn test_attention_backward_support_surface_rejects_sink_token_before_cudnn_913() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 64, 1])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let d_q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let sink_token = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 1, 1])?,
    ));

    let error = graph
        .validate_attention_backward_support_surface_for_version(
            91299,
            q,
            k,
            v,
            o,
            stats,
            d_o,
            d_q,
            d_k,
            d_v,
            &AttentionBackwardConfig::new(DataType::F32).with_sink_token(sink_token),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa backward sink token version"
    ));

    Ok(())
}

#[test]
fn test_attention_backward_support_surface_rejects_deterministic_algorithm_on_blackwell_before_918()
-> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(100);
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 64, 1])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let d_q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));

    let error = graph
        .validate_attention_backward_support_surface_for_version(
            91799,
            q,
            k,
            v,
            o,
            stats,
            d_o,
            d_q,
            d_k,
            d_v,
            &AttentionBackwardConfig::new(DataType::F32).with_deterministic_algorithm(),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa backward deterministic algorithm version"
    ));

    Ok(())
}

#[test]
fn test_attention_backward_support_surface_rejects_deterministic_algorithm_with_dropout_on_blackwell()
-> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(100);
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 64, 1])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let d_q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let dropout_mask = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let dropout_scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);

    let error = graph
        .validate_attention_backward_support_surface_for_version(
            91800,
            q,
            k,
            v,
            o,
            stats,
            d_o,
            d_q,
            d_k,
            d_v,
            &AttentionBackwardConfig::new(DataType::F32)
                .with_dropout(dropout_mask, dropout_scale)
                .with_deterministic_algorithm(),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa backward deterministic algorithm modifiers"
    ));

    Ok(())
}

#[test]
fn test_attention_backward_support_surface_rejects_deterministic_algorithm_for_ragged_sm8x()
-> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(80);
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 64, 1])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let d_q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let offsets = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    graph.set_ragged_offset(q, offsets)?;

    let error = graph
        .validate_attention_backward_support_surface_for_version(
            91801,
            q,
            k,
            v,
            o,
            stats,
            d_o,
            d_q,
            d_k,
            d_v,
            &AttentionBackwardConfig::new(DataType::F32).with_deterministic_algorithm(),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa backward deterministic algorithm ragged"
    ));

    Ok(())
}

#[test]
fn test_mxfp8_backward_support_surface_rejects_before_cudnn_921() {
    let graph = Graph::new();
    let error = graph
        .validate_mxfp8_backward_support_surface_for_version(
            92099,
            DataType::BF16,
            false,
            &AttentionBackwardConfig::new(DataType::F32),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa mxfp8 backward version"
    ));
}

#[test]
fn test_mxfp8_backward_support_surface_rejects_pre_hopper() {
    let mut graph = Graph::new();
    graph.set_sm_version(80);

    let error = graph
        .validate_mxfp8_backward_support_surface_for_version(
            92100,
            DataType::F32,
            false,
            &AttentionBackwardConfig::new(DataType::F32),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa mxfp8 backward sm version"
    ));
}

#[test]
fn test_mxfp8_backward_support_surface_rejects_bf16_output_before_blackwell() {
    let mut graph = Graph::new();
    graph.set_sm_version(90);

    let error = graph
        .validate_mxfp8_backward_support_surface_for_version(
            92100,
            DataType::BF16,
            false,
            &AttentionBackwardConfig::new(DataType::F32),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa mxfp8 backward output data type"
    ));
}

#[test]
fn test_mxfp8_backward_support_surface_rejects_blackwell_dropout() {
    let mut graph = Graph::new();
    graph.set_sm_version(100);

    let error = graph
        .validate_mxfp8_backward_support_surface_for_version(
            92100,
            DataType::BF16,
            false,
            &AttentionBackwardConfig::new(DataType::F32).with_dropout_probability(0.25, 7),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa mxfp8 backward deterministic algorithm dropout"
    ));
}

#[test]
fn test_mxfp8_backward_support_surface_rejects_rng_dump() {
    let graph = Graph::new();
    let error = graph
        .validate_mxfp8_backward_support_surface_for_version(
            92100,
            DataType::F32,
            false,
            &AttentionBackwardConfig::new(DataType::F32).with_random_number_generator_dump(),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa mxfp8 backward rng_dump"
    ));
}

#[test]
fn test_mxfp8_backward_support_surface_rejects_ragged_on_hopper() {
    let mut graph = Graph::new();
    graph.set_sm_version(90);

    let error = graph
        .validate_mxfp8_backward_support_surface_for_version(
            92100,
            DataType::F32,
            true,
            &AttentionBackwardConfig::new(DataType::F32),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa mxfp8 backward ragged"
    ));
}

#[test]
fn test_mxfp8_backward_rejects_scale_q_shape() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 32])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 32])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let scale_q = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 1, 4, 1])?,
    ));
    let scale_q_t = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 1, 32])?,
    ));
    let scale_k = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 6, 1])?,
    ));
    let scale_k_t = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 1, 32])?,
    ));
    let scale_v = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 6, 1])?,
    ));
    let scale_d_o = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let scale_d_o_t = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 1, 16])?,
    ));

    let error = graph
        .validate_mxfp8_backward_scale_dimensions(
            q,
            k,
            v,
            d_o,
            scale_q,
            scale_q_t,
            scale_k,
            scale_k_t,
            scale_v,
            scale_d_o,
            scale_d_o_t,
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa mxfp8 backward scale_q shape"
    ));

    Ok(())
}

#[test]
fn test_mxfp8_backward_rejects_scale_q_t_shape() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 64, 32])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 32])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 64, 16])?,
    ));
    let scale_q = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 64, 1])?,
    ));
    let scale_q_t = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 1, 32])?,
    ));
    let scale_k = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 6, 1])?,
    ));
    let scale_k_t = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 1, 32])?,
    ));
    let scale_v = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 6, 1])?,
    ));
    let scale_d_o = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 64, 1])?,
    ));
    let scale_d_o_t = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 1, 16])?,
    ));

    let error = graph
        .validate_mxfp8_backward_scale_dimensions(
            q,
            k,
            v,
            d_o,
            scale_q,
            scale_q_t,
            scale_k,
            scale_k_t,
            scale_v,
            scale_d_o,
            scale_d_o_t,
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa mxfp8 backward scale_q_t shape"
    ));

    Ok(())
}

#[test]
fn test_mxfp8_backward_rejects_scale_d_o_t_shape() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 64, 32])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 32])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 64, 16])?,
    ));
    let scale_q = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 64, 1])?,
    ));
    let scale_q_t = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 2, 32])?,
    ));
    let scale_k = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 6, 1])?,
    ));
    let scale_k_t = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 1, 32])?,
    ));
    let scale_v = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 6, 1])?,
    ));
    let scale_d_o = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 64, 1])?,
    ));
    let scale_d_o_t = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 1, 16])?,
    ));

    let error = graph
        .validate_mxfp8_backward_scale_dimensions(
            q,
            k,
            v,
            d_o,
            scale_q,
            scale_q_t,
            scale_k,
            scale_k_t,
            scale_v,
            scale_d_o,
            scale_d_o_t,
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa mxfp8 backward scale_d_o_t shape"
    ));

    Ok(())
}

#[test]
fn test_fp8_backward_support_surface_rejects_bf16_output_before_blackwell() {
    let mut graph = Graph::new();
    graph.set_sm_version(90);

    let error = graph
        .validate_fp8_backward_support_surface_for_version(
            92100,
            DataType::BF16,
            128,
            64,
            false,
            &AttentionBackwardConfig::new(DataType::F32),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa fp8 backward output data type"
    ));
}

#[test]
fn test_fp8_backward_support_surface_rejects_before_cudnn_901() {
    let graph = Graph::new();
    let error = graph
        .validate_fp8_backward_support_surface_for_version(
            90099,
            DataType::F32,
            128,
            128,
            false,
            &AttentionBackwardConfig::new(DataType::F32),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa fp8 backward version"
    ));
}

#[test]
fn test_fp8_backward_support_surface_rejects_cudnn_91000() {
    let graph = Graph::new();
    let error = graph
        .validate_fp8_backward_support_surface_for_version(
            91000,
            DataType::F32,
            128,
            128,
            false,
            &AttentionBackwardConfig::new(DataType::F32),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa fp8 backward version"
    ));
}

#[test]
fn test_fp8_backward_support_surface_rejects_pre_hopper() {
    let mut graph = Graph::new();
    graph.set_sm_version(80);

    let error = graph
        .validate_fp8_backward_support_surface_for_version(
            91300,
            DataType::F32,
            128,
            128,
            false,
            &AttentionBackwardConfig::new(DataType::F32),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa fp8 backward sm version"
    ));
}

#[test]
fn test_fp8_backward_support_surface_rejects_blackwell_supported_dims_before_cudnn_919() {
    let mut graph = Graph::new();
    graph.set_sm_version(100);

    let error = graph
        .validate_fp8_backward_support_surface_for_version(
            91899,
            DataType::BF16,
            192,
            128,
            false,
            &AttentionBackwardConfig::new(DataType::F32),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa fp8 backward deterministic algorithm version"
    ));
}

#[test]
fn test_fp8_backward_support_surface_rejects_blackwell_supported_dims_with_dropout() {
    let mut graph = Graph::new();
    graph.set_sm_version(100);

    let error = graph
        .validate_fp8_backward_support_surface_for_version(
            92100,
            DataType::BF16,
            192,
            128,
            false,
            &AttentionBackwardConfig::new(DataType::F32).with_dropout_probability(0.25, 7),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa fp8 backward deterministic algorithm dropout"
    ));
}

#[test]
fn test_fp8_backward_support_surface_rejects_ragged_on_hopper() {
    let mut graph = Graph::new();
    graph.set_sm_version(90);

    let error = graph
        .validate_fp8_backward_support_surface_for_version(
            91300,
            DataType::F32,
            128,
            128,
            true,
            &AttentionBackwardConfig::new(DataType::F32),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa fp8 backward ragged"
    ));
}

#[test]
fn test_quantized_backward_head_dimensions_reject_hopper_non_128_dims() {
    let mut graph = Graph::new();
    graph.set_sm_version(90);

    let error = graph
        .validate_quantized_backward_head_dimensions(64, 128, "sdpa fp8 backward head dimension")
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa fp8 backward head dimension"
    ));
}

#[test]
fn test_quantized_backward_head_dimensions_reject_blackwell_large_dims() {
    let mut graph = Graph::new();
    graph.set_sm_version(100);

    let error = graph
        .validate_quantized_backward_head_dimensions(256, 128, "sdpa fp8 backward head dimension")
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa fp8 backward head dimension"
    ));
}

#[test]
fn test_quantized_backward_head_dimensions_accept_blackwell_192_128() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(100);

    graph.validate_quantized_backward_head_dimensions(
        192,
        128,
        "sdpa fp8 backward head dimension",
    )?;

    Ok(())
}

#[test]
fn test_attention_backward_support_surface_rejects_alibi_without_right_bound_zero() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 64, 1])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let d_q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 64, 64])?,
    ));
    let alibi = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 1, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);

    let error = graph
        .sdpa_backward_infer(
            SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
            AttentionBackwardConfig::new(DataType::F32).with_alibi(alibi),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa alibi alignment"
    ));

    let _ = (d_q, d_k, d_v);

    Ok(())
}

#[test]
fn test_sdpa_backward_rejects_mismatched_output_gradient_shape() -> Result<()> {
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
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 5, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);
    let d_q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 6, 8])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 6, 16])?,
    ));

    let err = graph
        .sdpa_backward(
            SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
            SdpaBackwardGradientTensors {
                query_gradient: d_q,
                key_gradient: d_k,
                value_gradient: d_v,
                bias_gradient: None,
                rng_dump: None,
                sink_token_gradient: None,
            },
            AttentionBackwardConfig::new(DataType::F32),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa backward d_o shape"
    ));

    Ok(())
}
