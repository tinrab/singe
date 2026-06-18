use super::*;

#[test]
fn test_sdpa_mxfp8_backward_infer_creates_gradient_and_absolute_max_shapes() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 32])?,
    ));
    let q_t = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 32])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 32])?,
    ));
    let k_t = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 32])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o_quantized = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o_t = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let scale_q = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 4, 1])?,
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
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);

    let outputs = graph.sdpa_mxfp8_backward_infer(
        SdpaMxfp8BackwardInputs::new(
            q,
            q_t,
            k,
            k_t,
            v,
            o,
            d_o,
            d_o_quantized,
            d_o_t,
            stats,
            scale_q,
            scale_q_t,
            scale_k,
            scale_k_t,
            scale_v,
            scale_d_o,
            scale_d_o_t,
            attention_scale,
        ),
        AttentionBackwardConfig::new(DataType::F32),
    )?;

    assert_eq!(
        graph.shape(outputs.query_gradient)?.dimensions(),
        &[1, 2, 4, 32]
    );
    assert_eq!(
        graph.shape(outputs.key_gradient)?.dimensions(),
        &[1, 2, 6, 32]
    );
    assert_eq!(
        graph.shape(outputs.value_gradient)?.dimensions(),
        &[1, 2, 6, 16]
    );
    assert_eq!(
        graph
            .shape(outputs.absolute_max_query_gradient)?
            .dimensions(),
        &[1, 1, 1, 1]
    );
    assert_eq!(
        graph.shape(outputs.absolute_max_key_gradient)?.dimensions(),
        &[1, 1, 1, 1]
    );
    assert_eq!(
        graph
            .shape(outputs.absolute_max_value_gradient)?
            .dimensions(),
        &[1, 1, 1, 1]
    );

    Ok(())
}

#[test]
fn test_sdpa_mxfp8_backward_rejects_ragged_outputs_on_hopper() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(90);
    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 32])?,
    ));
    let q_t = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 32])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 32])?,
    ));
    let k_t = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 32])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o_quantized = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o_t = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let scale_q = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 4, 1])?,
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
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let d_q = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 4, 32])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 6, 32])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::BF16,
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

    let error = graph
        .sdpa_mxfp8_backward(
            SdpaMxfp8BackwardInputs::new(
                q,
                q_t,
                k,
                k_t,
                v,
                o,
                d_o,
                d_o_quantized,
                d_o_t,
                stats,
                scale_q,
                scale_q_t,
                scale_k,
                scale_k_t,
                scale_v,
                scale_d_o,
                scale_d_o_t,
                attention_scale,
            ),
            SdpaMxfp8BackwardGradientTensors::new(
                d_q,
                d_k,
                d_v,
                absolute_max_d_q,
                absolute_max_d_k,
                absolute_max_d_v,
            ),
            AttentionBackwardConfig::new(DataType::F32),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa mxfp8 backward ragged outputs"
    ));

    Ok(())
}

#[test]
fn test_sdpa_mxfp8_backward_rejects_sliding_window_when_q_exceeds_k_without_padding() -> Result<()>
{
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 8, 32])?,
    ));
    let q_t = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 8, 32])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 32])?,
    ));
    let k_t = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 32])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 8, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 8, 16])?,
    ));
    let d_o_quantized = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 8, 16])?,
    ));
    let d_o_t = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 8, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 8, 1])?,
    ));
    let scale_q = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 8, 1])?,
    ));
    let scale_q_t = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 1, 32])?,
    ));
    let scale_k = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let scale_k_t = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 1, 32])?,
    ));
    let scale_v = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let scale_d_o = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 8, 1])?,
    ));
    let scale_d_o_t = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 1, 16])?,
    ));
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);

    let error = graph
        .sdpa_mxfp8_backward_infer(
            SdpaMxfp8BackwardInputs::new(
                q,
                q_t,
                k,
                k_t,
                v,
                o,
                d_o,
                d_o_quantized,
                d_o_t,
                stats,
                scale_q,
                scale_q_t,
                scale_k,
                scale_k_t,
                scale_v,
                scale_d_o,
                scale_d_o_t,
                attention_scale,
            ),
            AttentionBackwardConfig::new(DataType::F32).with_sliding_window(2, 0),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa mxfp8 backward sliding window sequence lengths"
    ));

    Ok(())
}

#[test]
fn test_sdpa_mxfp8_backward_infer_accepts_custom_score_subgraphs() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 32])?,
    ));
    let q_t = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 32])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 32])?,
    ));
    let k_t = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 32])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o_quantized = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o_t = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let scale_q = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 4, 1])?,
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
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);

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

    let outputs = graph.sdpa_mxfp8_backward_infer(
        SdpaMxfp8BackwardInputs::new(
            q,
            q_t,
            k,
            k_t,
            v,
            o,
            d_o,
            d_o_quantized,
            d_o_t,
            stats,
            scale_q,
            scale_q_t,
            scale_k,
            scale_k_t,
            scale_v,
            scale_d_o,
            scale_d_o_t,
            attention_scale,
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

    assert_eq!(
        graph.shape(outputs.query_gradient)?.dimensions(),
        &[1, 2, 4, 32]
    );
    assert_eq!(
        graph.shape(outputs.key_gradient)?.dimensions(),
        &[1, 2, 6, 32]
    );
    assert_eq!(
        graph.shape(outputs.value_gradient)?.dimensions(),
        &[1, 2, 6, 16]
    );
    assert_eq!(
        graph
            .shape(outputs.absolute_max_query_gradient)?
            .dimensions(),
        &[1, 1, 1, 1]
    );
    assert_eq!(
        graph.shape(outputs.absolute_max_key_gradient)?.dimensions(),
        &[1, 1, 1, 1]
    );
    assert_eq!(
        graph
            .shape(outputs.absolute_max_value_gradient)?
            .dimensions(),
        &[1, 1, 1, 1]
    );

    Ok(())
}

#[test]
fn test_sdpa_mxfp8_backward_infer_accepts_inline_attn_scale_value() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 32])?,
    ));
    let q_t = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 32])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 32])?,
    ));
    let k_t = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 32])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o_quantized = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o_t = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let scale_q = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 4, 1])?,
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
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);

    let outputs = graph.sdpa_mxfp8_backward_infer(
        SdpaMxfp8BackwardInputs::new(
            q,
            q_t,
            k,
            k_t,
            v,
            o,
            d_o,
            d_o_quantized,
            d_o_t,
            stats,
            scale_q,
            scale_q_t,
            scale_k,
            scale_k_t,
            scale_v,
            scale_d_o,
            scale_d_o_t,
            attention_scale,
        ),
        AttentionBackwardConfig::new(DataType::F32).with_attention_scale(0.5),
    )?;

    assert_eq!(
        graph.shape(outputs.query_gradient)?.dimensions(),
        &[1, 2, 4, 32]
    );
    assert_eq!(
        graph.shape(outputs.key_gradient)?.dimensions(),
        &[1, 2, 6, 32]
    );
    assert_eq!(
        graph.shape(outputs.value_gradient)?.dimensions(),
        &[1, 2, 6, 16]
    );
    assert_eq!(
        graph
            .shape(outputs.absolute_max_query_gradient)?
            .dimensions(),
        &[1, 1, 1, 1]
    );
    assert_eq!(
        graph.shape(outputs.absolute_max_key_gradient)?.dimensions(),
        &[1, 1, 1, 1]
    );
    assert_eq!(
        graph
            .shape(outputs.absolute_max_value_gradient)?
            .dimensions(),
        &[1, 1, 1, 1]
    );

    Ok(())
}

#[test]
fn test_sdpa_mxfp8_backward_builder_uses_provided_outputs() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 32])?,
    ));
    let q_t = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 32])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 32])?,
    ));
    let k_t = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 32])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o_quantized = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o_t = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let scale_q = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 4, 1])?,
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
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let d_q = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 4, 32])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 6, 32])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::BF16,
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

    graph.sdpa_mxfp8_backward(
        SdpaMxfp8BackwardInputs::new(
            q,
            q_t,
            k,
            k_t,
            v,
            o,
            d_o,
            d_o_quantized,
            d_o_t,
            stats,
            scale_q,
            scale_q_t,
            scale_k,
            scale_k_t,
            scale_v,
            scale_d_o,
            scale_d_o_t,
            attention_scale,
        ),
        SdpaMxfp8BackwardGradientTensors::new(
            d_q,
            d_k,
            d_v,
            absolute_max_d_q,
            absolute_max_d_k,
            absolute_max_d_v,
        ),
        AttentionBackwardConfig::new(DataType::F32),
    )?;

    for output in [
        d_q,
        d_k,
        d_v,
        absolute_max_d_q,
        absolute_max_d_k,
        absolute_max_d_v,
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
fn test_sdpa_mxfp8_backward_accepts_mixed_f16_bf16_outputs_on_blackwell() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(100);
    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 32])?,
    ));
    let q_t = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 32])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 32])?,
    ));
    let k_t = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 32])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o_quantized = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o_t = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let scale_q = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 4, 1])?,
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
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let d_q = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 4, 32])?,
    ));
    let d_k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 6, 32])?,
    ));
    let d_v = graph.tensor(TensorSpec::new(
        DataType::BF16,
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

    graph.sdpa_mxfp8_backward(
        SdpaMxfp8BackwardInputs::new(
            q,
            q_t,
            k,
            k_t,
            v,
            o,
            d_o,
            d_o_quantized,
            d_o_t,
            stats,
            scale_q,
            scale_q_t,
            scale_k,
            scale_k_t,
            scale_v,
            scale_d_o,
            scale_d_o_t,
            attention_scale,
        ),
        SdpaMxfp8BackwardGradientTensors::new(
            d_q,
            d_k,
            d_v,
            absolute_max_d_q,
            absolute_max_d_k,
            absolute_max_d_v,
        ),
        AttentionBackwardConfig::new(DataType::F32),
    )?;

    Ok(())
}

#[test]
fn test_sdpa_mxfp8_backward_infer_with_output_types_accepts_mixed_f16_bf16_outputs_on_blackwell()
-> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(100);
    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 32])?,
    ));
    let q_t = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 32])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 32])?,
    ));
    let k_t = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 32])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o_quantized = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o_t = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let scale_q = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 4, 1])?,
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
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);

    let outputs = graph.sdpa_mxfp8_backward_infer_with_output_types(
        SdpaMxfp8BackwardInputs::new(
            q,
            q_t,
            k,
            k_t,
            v,
            o,
            d_o,
            d_o_quantized,
            d_o_t,
            stats,
            scale_q,
            scale_q_t,
            scale_k,
            scale_k_t,
            scale_v,
            scale_d_o,
            scale_d_o_t,
            attention_scale,
        ),
        DataType::BF16,
        DataType::F16,
        DataType::BF16,
        AttentionBackwardConfig::new(DataType::F32),
    )?;

    assert_eq!(
        graph.tensor_config(outputs.query_gradient)?.data_type,
        DataType::BF16
    );
    assert_eq!(
        graph.tensor_config(outputs.key_gradient)?.data_type,
        DataType::F16
    );
    assert_eq!(
        graph.tensor_config(outputs.value_gradient)?.data_type,
        DataType::BF16
    );

    Ok(())
}

#[test]
fn test_sdpa_mxfp8_backward_rejects_non_mxfp8_scale_tensor() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 32])?,
    ));
    let q_t = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 32])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 32])?,
    ));
    let k_t = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 32])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o_quantized = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o_t = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let scale_q = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
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
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);

    let err = graph
        .sdpa_mxfp8_backward_infer(
            SdpaMxfp8BackwardInputs::new(
                q,
                q_t,
                k,
                k_t,
                v,
                o,
                d_o,
                d_o_quantized,
                d_o_t,
                stats,
                scale_q,
                scale_q_t,
                scale_k,
                scale_k_t,
                scale_v,
                scale_d_o,
                scale_d_o_t,
                attention_scale,
            ),
            AttentionBackwardConfig::new(DataType::F32),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa mxfp8 backward scale_q"
    ));

    Ok(())
}

#[test]
fn test_sdpa_mxfp8_backward_graph_reaches_compile() -> Result<()> {
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

    let mut graph = Graph::new();
    graph.set_io_data_type(DataType::F8E4M3);
    graph.set_intermediate_data_type(DataType::F32);
    graph.set_compute_data_type(DataType::F32);

    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 32])?,
    ));
    let q_t = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 32])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 32])?,
    ));
    let k_t = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 32])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o_quantized = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o_t = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let scale_q = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 4, 1])?,
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
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);

    let outputs = graph.sdpa_mxfp8_backward_infer(
        SdpaMxfp8BackwardInputs::new(
            q,
            q_t,
            k,
            k_t,
            v,
            o,
            d_o,
            d_o_quantized,
            d_o_t,
            stats,
            scale_q,
            scale_q_t,
            scale_k,
            scale_k_t,
            scale_v,
            scale_d_o,
            scale_d_o_t,
            attention_scale,
        ),
        AttentionBackwardConfig::new(DataType::F32).with_causal_mask(),
    )?;

    for output in [
        outputs.query_gradient,
        outputs.key_gradient,
        outputs.value_gradient,
        outputs.absolute_max_query_gradient,
        outputs.absolute_max_key_gradient,
        outputs.absolute_max_value_gradient,
    ] {
        let tensor = graph.tensor_config(output)?.clone().output_tensor();
        graph.replace_tensor(output, tensor)?;
    }

    match graph.compile(&context, &[HeuristicMode::A]) {
        Ok(_) => {}
        Err(error) if is_expected_compile_status(&error) => return Ok(()),
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_sdpa_mxfp8_upstream_like_graph_reaches_compile() -> Result<()> {
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

    let mut graph = Graph::new();
    let b = 2;
    let h = 2;
    let s = 512;
    let d = 128;
    let block_size = 32;
    let d_scale = (d + block_size - 1) / block_size;
    let s_scale = (s + block_size - 1) / block_size;
    let s_padded = ((s + 127) / 128) * 128;
    let d_scale_padded = ((d_scale + 3) / 4) * 4;
    let s_scale_padded = ((s_scale + 3) / 4) * 4;
    let d_padded = ((d + 127) / 128) * 128;

    let qkv_dims = vec![b, h, s, d];
    let qkv_strides = vec![s * 3 * h * d, d, 3 * h * d, 1];
    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous(qkv_dims.clone())?.with_strides(qkv_strides.clone())?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous(qkv_dims.clone())?.with_strides(qkv_strides.clone())?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous(qkv_dims)?.with_strides(qkv_strides)?,
    ));
    let sf_q = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([b, h, s_padded, d_scale_padded])?.with_strides(vec![
            h * s_padded * d_scale_padded,
            s_padded * d_scale_padded,
            d_scale_padded,
            1,
        ])?,
    ));
    let sf_k = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([b, h, s_padded, d_scale_padded])?.with_strides(vec![
            h * s_padded * d_scale_padded,
            s_padded * d_scale_padded,
            d_scale_padded,
            1,
        ])?,
    ));
    let sf_v = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([b, h, s_scale_padded, d_padded])?.with_strides(vec![
            h * s_scale_padded * d_padded,
            s_scale_padded * d_padded,
            1,
            s_scale_padded,
        ])?,
    ));
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(0.123)?);

    let _ = graph.sdpa_mxfp8_infer(
        SdpaMxfp8Inputs::new(q, k, v, sf_q, sf_k, sf_v, attention_scale),
        &SdpaConfig::new()
            .with_stats()
            .with_absolute_max_o()
            .with_causal_mask(),
    )?;

    match graph.compile(&context, &[HeuristicMode::A]) {
        Ok(_) => Ok(()),
        Err(error) if is_expected_compile_status(&error) => Ok(()),
        Err(error) => Err(error),
    }
}

#[test]
fn test_mxfp8_transposed_k_dequant_reaches_compile() -> Result<()> {
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

    let mut graph = Graph::new();
    let b = 2;
    let h = 2;
    let s = 512;
    let d = 128;
    let block_size = 32;
    let d_scale = (d + block_size - 1) / block_size;
    let s_padded = ((s + 127) / 128) * 128;
    let d_scale_padded = ((d_scale + 3) / 4) * 4;

    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([b, h, s, d])?.with_strides([s * h * d, d, h * d, 1])?,
    ));
    let sf_k = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([b, h, s_padded, d_scale_padded])?.with_strides(vec![
            h * s_padded * d_scale_padded,
            s_padded * d_scale_padded,
            d_scale_padded,
            1,
        ])?,
    ));

    let mut kt_dimensions = graph.shape(k)?.dimensions().to_vec();
    let mut kt_strides = graph.shape(k)?.strides().to_vec();
    kt_dimensions.swap(2, 3);
    kt_strides.swap(2, 3);
    let k_tensor = graph.tensor_config(k)?.clone();
    graph.replace_tensor(
        k,
        k_tensor.with_shape(Shape::contiguous(kt_dimensions)?.with_strides(kt_strides)?),
    )?;

    let mut sf_k_dimensions = graph.shape(sf_k)?.dimensions().to_vec();
    let mut sf_k_strides = graph.shape(sf_k)?.strides().to_vec();
    sf_k_dimensions.swap(2, 3);
    sf_k_strides.swap(2, 3);
    let sf_k_tensor = graph.tensor_config(sf_k)?.clone();
    graph.replace_tensor(
        sf_k,
        sf_k_tensor.with_shape(Shape::contiguous(sf_k_dimensions)?.with_strides(sf_k_strides)?),
    )?;

    let _ = graph.block_scale_dequantize_infer(
        k,
        sf_k,
        BlockScaleDequantizeConfig::new(DataType::F32, vec![32, 1]),
    )?;

    match graph.compile(&context, &[HeuristicMode::A]) {
        Ok(_) => Ok(()),
        Err(error) if is_expected_compile_status(&error) => Ok(()),
        Err(error) => Err(error),
    }
}

#[test]
fn test_mxfp8_bmm1_reaches_compile() -> Result<()> {
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

    let mut graph = Graph::new();
    let b = 2;
    let h = 2;
    let s = 512;
    let d = 128;
    let block_size = 32;
    let d_scale = (d + block_size - 1) / block_size;
    let s_padded = ((s + 127) / 128) * 128;
    let d_scale_padded = ((d_scale + 3) / 4) * 4;

    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([b, h, s, d])?.with_strides(vec![s * 3 * h * d, d, 3 * h * d, 1])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([b, h, s, d])?.with_strides(vec![s * 3 * h * d, d, 3 * h * d, 1])?,
    ));
    let sf_q = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([b, h, s_padded, d_scale_padded])?.with_strides(vec![
            h * s_padded * d_scale_padded,
            s_padded * d_scale_padded,
            d_scale_padded,
            1,
        ])?,
    ));
    let sf_k = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([b, h, s_padded, d_scale_padded])?.with_strides(vec![
            h * s_padded * d_scale_padded,
            s_padded * d_scale_padded,
            d_scale_padded,
            1,
        ])?,
    ));

    let k_tensor = graph.tensor_config(k)?.clone();
    let mut kt_dimensions = k_tensor.shape.dimensions().to_vec();
    let mut kt_strides = k_tensor.shape.strides().to_vec();
    kt_dimensions.swap(2, 3);
    kt_strides.swap(2, 3);
    graph.replace_tensor(
        k,
        k_tensor.with_shape(Shape::contiguous(kt_dimensions)?.with_strides(kt_strides)?),
    )?;

    let sf_k_tensor = graph.tensor_config(sf_k)?.clone();
    let mut sf_k_dimensions = sf_k_tensor.shape.dimensions().to_vec();
    let mut sf_k_strides = sf_k_tensor.shape.strides().to_vec();
    sf_k_dimensions.swap(2, 3);
    sf_k_strides.swap(2, 3);
    graph.replace_tensor(
        sf_k,
        sf_k_tensor.with_shape(Shape::contiguous(sf_k_dimensions)?.with_strides(sf_k_strides)?),
    )?;

    let q_fp = graph.block_scale_dequantize_infer(
        q,
        sf_q,
        BlockScaleDequantizeConfig::new(DataType::F32, vec![1, 32]),
    )?;
    let k_fp = graph.block_scale_dequantize_infer(
        k,
        sf_k,
        BlockScaleDequantizeConfig::new(DataType::F32, vec![32, 1]),
    )?;
    let q_tensor = graph.tensor_config(q_fp)?.clone();
    let k_tensor = graph.tensor_config(k_fp)?.clone();
    let scores = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous(vec![
                q_tensor.shape.dimensions()[0],
                q_tensor.shape.dimensions()[1],
                q_tensor.shape.dimensions()[2],
                k_tensor.shape.dimensions()[3],
            ])?
            .with_strides(vec![
                q_tensor.shape.dimensions()[1]
                    * q_tensor.shape.dimensions()[2]
                    * k_tensor.shape.dimensions()[3],
                q_tensor.shape.dimensions()[2] * k_tensor.shape.dimensions()[3],
                k_tensor.shape.dimensions()[3],
                1,
            ])?,
        )
        .virtual_tensor(),
    );
    graph.matmul(q_fp, k_fp, scores, DataType::F32)?;

    match graph.compile(&context, &[HeuristicMode::A]) {
        Ok(_) => Ok(()),
        Err(error) if is_expected_compile_status(&error) => Ok(()),
        Err(error) => Err(error),
    }
}

#[test]
fn test_mxfp8_softmax_reaches_compile() -> Result<()> {
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

    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 2, 512, 128])?.with_strides(vec![
            2 * 512 * 128,
            512 * 128,
            128,
            1,
        ])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 2, 128, 512])?.with_strides(vec![
            2 * 128 * 512,
            128 * 512,
            1,
            128,
        ])?,
    ));
    let scores = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([2, 2, 512, 512])?.with_strides(vec![
                2 * 512 * 512,
                512 * 512,
                512,
                1,
            ])?,
        )
        .virtual_tensor(),
    );
    graph.matmul(q, k, scores, DataType::F32)?;
    let scale = graph.tensor(
        TensorSpec::new(DataType::F32, Shape::contiguous([1, 1, 1, 1])?)
            .with_scalar_value(ScalarValue::F32(0.123))?,
    );
    let scaled_scores = graph.tensor(
        TensorSpec::new(DataType::F32, graph.tensor_config(scores)?.shape.clone()).virtual_tensor(),
    );
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::Mul,
        lhs: scores,
        rhs: scale,
        output: scaled_scores,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::NotPropagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });
    let masked_scores = graph.apply_diagonal_band_mask(scaled_scores, 0, Some(0))?;
    let probs = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            graph.tensor_config(masked_scores)?.shape.clone(),
        )
        .virtual_tensor(),
    );
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        reduce_last_axis(graph.shape(masked_scores)?)?,
    ));
    let sum = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            reduce_last_axis(graph.shape(masked_scores)?)?,
        )
        .virtual_tensor(),
    );
    graph.softmax_composite(
        masked_scores,
        stats,
        sum,
        probs,
        SoftmaxConfig::new(DataType::F32),
    )?;

    match graph.compile(&context, &[HeuristicMode::A]) {
        Ok(_) => Ok(()),
        Err(error) if is_expected_compile_status(&error) => Ok(()),
        Err(error) => Err(error),
    }
}

#[test]
fn test_mxfp8_scaled_scores_reaches_compile_without_softmax() -> Result<()> {
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

    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 2, 512, 128])?.with_strides(vec![
            2 * 512 * 128,
            512 * 128,
            128,
            1,
        ])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 2, 128, 512])?.with_strides(vec![
            2 * 128 * 512,
            128 * 512,
            1,
            128,
        ])?,
    ));
    let scores = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([2, 2, 512, 512])?.with_strides(vec![
                2 * 512 * 512,
                512 * 512,
                512,
                1,
            ])?,
        )
        .virtual_tensor(),
    );
    graph.matmul(q, k, scores, DataType::F32)?;
    let scale = graph.tensor(
        TensorSpec::new(DataType::F32, Shape::contiguous([1, 1, 1, 1])?)
            .with_scalar_value(ScalarValue::F32(0.123))?,
    );
    let scaled_scores = graph.tensor(
        TensorSpec::new(DataType::F32, graph.tensor_config(scores)?.shape.clone()).virtual_tensor(),
    );
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::Mul,
        lhs: scores,
        rhs: scale,
        output: scaled_scores,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::NotPropagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });
    let _masked_scores = graph.apply_diagonal_band_mask(scaled_scores, 0, Some(0))?;

    match graph.compile(&context, &[HeuristicMode::A]) {
        Ok(_) => Ok(()),
        Err(error) if is_expected_compile_status(&error) => Ok(()),
        Err(error) => Err(error),
    }
}
