use super::*;
use crate::frontend::composite::sdpa::SdpaOutputs;
use crate::frontend::operation::AttentionPagedCache;

#[test]
fn test_mxfp8_bmm1_plus_scale_reaches_compile() -> Result<()> {
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

    match graph.compile(&context, &[HeuristicMode::A]) {
        Ok(_) => Ok(()),
        Err(error) if is_expected_compile_status(&error) => Ok(()),
        Err(error) => Err(error),
    }
}

#[test]
fn test_sdpa_causal_mask_infer_matches_score_shape() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 6, 8])?,
    ));

    let mask = graph.sdpa_causal_mask_infer(q, k)?;

    assert_eq!(graph.shape(mask)?.dimensions(), &[1, 2, 4, 6]);

    Ok(())
}

#[test]
fn test_sdpa_sliding_window_mask_rejects_negative_bounds() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 6, 8])?,
    ));

    let err = graph
        .sdpa_sliding_window_mask_infer(q, k, -1, 0)
        .unwrap_err();
    assert!(matches!(
        err,
        Error::OutOfRange { name } if name == "left_window"
    ));

    Ok(())
}

#[test]
fn test_sdpa_sliding_window_mask_infer_matches_score_shape() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 6, 8])?,
    ));

    let mask = graph.sdpa_sliding_window_mask_infer(q, k, 2, 1)?;

    assert_eq!(graph.shape(mask)?.dimensions(), &[1, 2, 4, 6]);

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_accepts_padding_mask_inputs() -> Result<()> {
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

    let SdpaOutputs { output, stats: _ } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32)
            .with_padding_mask(sequence_length_query, sequence_length_key_value),
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[2, 2, 4, 16]);

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_accepts_inline_attn_scale_value() -> Result<()> {
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

    let SdpaOutputs { output, stats: _ } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32).with_attention_scale(0.5),
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[2, 2, 4, 16]);

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_accepts_causal_mask_inputs() -> Result<()> {
    let mut graph = Graph::new();
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
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);

    let SdpaOutputs { output, stats } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32)
            .with_stats()
            .with_causal_mask(),
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[2, 2, 4, 16]);
    assert_eq!(
        graph.shape(stats.expect("stats"))?.dimensions(),
        &[2, 2, 4, 1]
    );
    assert!(graph.operations.iter().any(|operation| matches!(
        operation,
        Operation::SdpaForward { score_modifiers, .. } if score_modifiers.causal_mask
    )));

    Ok(())
}

#[test]
fn test_sdpa_unified_subgraph_marks_generated_output_virtual() -> Result<()> {
    let mut graph = Graph::new();
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
    let Operation::SdpaForward {
        score_modifiers,
        config,
        ..
    } = graph.operations.first().expect("sdpa operation")
    else {
        panic!("expected sdpa operation");
    };
    let subgraph = graph
        .build_sdpa_unified_subgraph(q, k, **config, *score_modifiers, None)?
        .expect("causal mask should create a score subgraph");

    let output = subgraph.graph.tensor_config(subgraph.output)?;
    assert!(output.is_virtual);
    assert_eq!(output.data_type, DataType::F32);

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_accepts_sliding_window_inputs() -> Result<()> {
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

    let SdpaOutputs { output, stats: _ } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32).with_sliding_window(2, 1),
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[2, 2, 4, 16]);

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_with_aux_creates_rng_dump_for_internal_dropout() -> Result<()> {
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
    let offset = graph.tensor(
        TensorSpec::new(DataType::I64, Shape::contiguous([1, 1, 1, 1])?)
            .with_scalar_value(ScalarValue::I64(17))?,
    );

    let outputs = graph.sdpa_auto_infer_with_aux(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32)
            .with_stats()
            .with_dropout_probability(0.25, 7)
            .with_dropout_offset(offset),
    )?;
    let output = outputs.output;
    let stats = outputs.stats;
    let rng_dump = outputs.rng_dump;

    assert_eq!(graph.shape(output)?.dimensions(), &[2, 2, 4, 16]);
    assert_eq!(
        graph.shape(stats.expect("stats"))?.dimensions(),
        &[2, 2, 4, 1]
    );
    let rng_dump = rng_dump.expect("rng dump");
    assert_eq!(graph.shape(rng_dump)?.dimensions(), &[2, 2, 4, 6]);
    assert_eq!(graph.tensor_config(rng_dump)?.data_type, DataType::F32);
    assert!(
        graph
            .operations
            .iter()
            .any(|operation| matches!(operation, Operation::SdpaForward { .. }))
    );

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_uses_unified_path_with_padding_mask() -> Result<()> {
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

    let SdpaOutputs { output, stats } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32)
            .with_stats()
            .with_padding_mask(sequence_length_query, sequence_length_key_value),
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[2, 2, 4, 16]);
    assert_eq!(
        graph.shape(stats.expect("stats"))?.dimensions(),
        &[2, 2, 4, 1]
    );
    assert!(
        graph
            .operations
            .iter()
            .any(|operation| matches!(operation, Operation::SdpaForward { .. }))
    );

    Ok(())
}

#[test]
fn test_unified_selector_accepts_ragged_padding_mask_on_hopper() -> Result<()> {
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
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));

    assert!(
        graph.can_lower_attention_as_unified_with_tensors_for_version(
            92100,
            q,
            k,
            v,
            AttentionConfig::new(DataType::F32)
                .with_padding_mask(sequence_length_query, sequence_length_key_value),
        )?
    );

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_uses_unified_path_with_ragged_padding_mask() -> Result<()> {
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

    let SdpaOutputs { output, stats: _ } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32)
            .with_padding_mask(sequence_length_query, sequence_length_key_value),
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[2, 2, 4, 16]);
    assert!(
        graph
            .operations
            .iter()
            .any(|operation| matches!(operation, Operation::SdpaForward { .. }))
    );

    Ok(())
}

#[test]
fn test_unified_selector_rejects_padding_mask_before_cudnn_915() -> Result<()> {
    let mut graph = Graph::new();
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));

    assert!(
        !graph.can_lower_attention_as_unified_for_version(
            91499,
            AttentionConfig::new(DataType::F32)
                .with_padding_mask(sequence_length_query, sequence_length_key_value),
        )
    );
    assert!(
        graph.can_lower_attention_as_unified_for_version(
            91500,
            AttentionConfig::new(DataType::F32)
                .with_padding_mask(sequence_length_query, sequence_length_key_value),
        )
    );

    Ok(())
}

#[test]
fn test_unified_selector_rejects_dropout_before_cudnn_921() -> Result<()> {
    let mut graph = Graph::new();
    let offset = graph.tensor(
        TensorSpec::new(DataType::I64, Shape::contiguous([1, 1, 1, 1])?)
            .with_scalar_value(ScalarValue::I64(17))?,
    );

    assert!(
        !graph.can_lower_attention_as_unified_for_version(
            92099,
            AttentionConfig::new(DataType::F32)
                .with_dropout_probability(0.25, 7)
                .with_dropout_offset(offset),
        )
    );
    assert!(
        graph.can_lower_attention_as_unified_for_version(
            92100,
            AttentionConfig::new(DataType::F32)
                .with_dropout_probability(0.25, 7)
                .with_dropout_offset(offset),
        )
    );

    Ok(())
}

#[test]
fn test_unified_selector_treats_zero_dropout_probability_as_no_dropout() -> Result<()> {
    let mut graph = Graph::new();
    let offset = graph.tensor(
        TensorSpec::new(DataType::I64, Shape::contiguous([1, 1, 1, 1])?)
            .with_scalar_value(ScalarValue::I64(17))?,
    );

    assert!(
        graph.can_lower_attention_as_unified_for_version(
            92099,
            AttentionConfig::new(DataType::F32)
                .with_dropout_probability(0.0, 7)
                .with_dropout_offset(offset),
        )
    );

    Ok(())
}

#[test]
fn test_sdpa_unified_infer_treats_zero_dropout_probability_as_no_dropout() -> Result<()> {
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
    let offset = graph.tensor(
        TensorSpec::new(DataType::I64, Shape::contiguous([1, 1, 1, 1])?)
            .with_scalar_value(ScalarValue::I64(17))?,
    );

    let outputs = graph.sdpa_unified_infer(
        q,
        k,
        v,
        scale,
        AttentionConfig::new(DataType::F32)
            .with_dropout_probability(0.0, 7)
            .with_dropout_offset(offset),
    )?;
    let rng_dump = outputs.rng_dump;

    assert!(rng_dump.is_none());

    Ok(())
}

#[test]
fn test_sdpa_unified_infer_creates_f32_rng_dump_for_internal_dropout() -> Result<()> {
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
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);

    let outputs = graph.sdpa_unified_infer(
        q,
        k,
        v,
        scale,
        AttentionConfig::new(DataType::F32).with_dropout_probability(0.25, 7),
    )?;
    let rng_dump = outputs.rng_dump;

    let rng_dump = rng_dump.expect("rng dump");
    assert_eq!(graph.tensor_config(rng_dump)?.data_type, DataType::F32);
    assert_eq!(graph.shape(rng_dump)?.dimensions(), &[2, 2, 4, 6]);

    Ok(())
}

#[test]
fn test_unified_selector_rejects_unfuse_fma_before_cudnn_921() {
    let graph = Graph::new();

    assert!(!graph.can_lower_attention_as_unified_for_version(
        92099,
        AttentionConfig::new(DataType::F32).with_unfused_fma(),
    ));
    assert!(graph.can_lower_attention_as_unified_for_version(
        92100,
        AttentionConfig::new(DataType::F32).with_unfused_fma(),
    ));
}

#[test]
fn test_unified_selector_rejects_non_float_compute_type() -> Result<()> {
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

    assert!(
        !graph.can_lower_attention_as_unified_with_tensors_for_version(
            92100,
            q,
            k,
            v,
            AttentionConfig::new(DataType::F16),
        )?
    );

    Ok(())
}

#[test]
fn test_unified_selector_rejects_non_multiple_of_eight_head_dims() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 10])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 10])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));

    assert!(
        !graph.can_lower_attention_as_unified_with_tensors_for_version(
            92100,
            q,
            k,
            v,
            AttentionConfig::new(DataType::F32),
        )?
    );

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_uses_composite_path_for_f32_inputs() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);

    let SdpaOutputs { output, stats: _ } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32).with_stats(),
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[1, 2, 4, 16]);
    assert!(
        graph
            .operations
            .iter()
            .all(|operation| !matches!(operation, Operation::SdpaForward { .. }))
    );

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_accepts_bottom_right_causal_inputs() -> Result<()> {
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

    let SdpaOutputs { output, stats: _ } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32)
            .with_causal_bottom_right(sequence_length_query, sequence_length_key_value),
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[2, 2, 4, 16]);

    Ok(())
}

#[test]
fn test_attention_support_surface_rejects_sliding_window_when_q_exceeds_k_without_padding()
-> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 8, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));

    let err = graph
        .validate_attention_support_surface_for_version(
            92100,
            q,
            k,
            v,
            AttentionConfig::new(DataType::F32).with_sliding_window(2, 0),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa sliding window sequence lengths"
    ));

    Ok(())
}

#[test]
fn test_attention_support_surface_rejects_invalid_gqa_head_factor() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 3, 4, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));

    let err = graph
        .validate_attention_support_surface_for_version(
            92100,
            q,
            k,
            v,
            AttentionConfig::new(DataType::F32),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa gqa heads"
    ));

    Ok(())
}

#[test]
fn test_attention_support_surface_rejects_ampere_head_dim_over_128_before_910() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(80);
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 136])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 136])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 144])?,
    ));

    let err = graph
        .validate_attention_support_surface_for_version(
            90900,
            q,
            k,
            v,
            AttentionConfig::new(DataType::F32),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa head dimension"
    ));

    Ok(())
}

#[test]
fn test_attention_support_surface_rejects_blackwell_head_dim_over_128_before_909() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(100);
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 136])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 136])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 144])?,
    ));

    let err = graph
        .validate_attention_support_surface_for_version(
            90899,
            q,
            k,
            v,
            AttentionConfig::new(DataType::F32),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa head dimension"
    ));

    Ok(())
}

#[test]
fn test_attention_support_surface_rejects_paged_attention_before_cudnn_905() -> Result<()> {
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
    let sequence = graph.tensor(TensorSpec::new(
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

    let err = graph
        .validate_attention_support_surface_for_version(
            90499,
            q,
            k,
            v,
            AttentionConfig::new(DataType::F32)
                .with_paged_k(AttentionPagedCache::new(sequence, page_table_k))
                .with_paged_v(AttentionPagedCache::new(sequence, page_table_v)),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa paged attention version"
    ));

    Ok(())
}

#[test]
fn test_attention_support_surface_rejects_cudnn_91400_sliding_window_bug() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 2048, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 2048, 16])?,
    ));

    let err = graph
        .validate_attention_support_surface_for_version(
            91400,
            q,
            k,
            v,
            AttentionConfig::new(DataType::F32).with_sliding_window(2, 0),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa cuDNN version"
    ));

    Ok(())
}

#[test]
fn test_attention_support_surface_rejects_ragged_offsets_on_pre_hopper_before_91801() -> Result<()>
{
    let mut graph = Graph::new();
    graph.set_sm_version(80);
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
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    graph.set_ragged_offset(q, offsets)?;

    let err = graph
        .validate_attention_support_surface_for_version(
            91800,
            q,
            k,
            v,
            AttentionConfig::new(DataType::F32)
                .with_padding_mask(sequence_length_query, sequence_length_key_value),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa ragged offsets"
    ));

    Ok(())
}

#[test]
fn test_attention_support_surface_rejects_ragged_k_without_padding_mask() -> Result<()> {
    let mut graph = Graph::new();
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
    graph.set_ragged_offset(k, offsets)?;

    let err = graph
        .validate_attention_support_surface_for_version(
            92100,
            q,
            k,
            v,
            AttentionConfig::new(DataType::F32),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa ragged offsets"
    ));

    Ok(())
}

#[test]
fn test_attention_support_surface_rejects_ragged_k_on_pre_hopper_before_91801() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(80);
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
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    graph.set_ragged_offset(k, offsets)?;

    let err = graph
        .validate_attention_support_surface_for_version(
            91800,
            q,
            k,
            v,
            AttentionConfig::new(DataType::F32)
                .with_padding_mask(sequence_length_query, sequence_length_key_value),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa ragged offsets"
    ));

    Ok(())
}

#[test]
fn test_attention_support_surface_accepts_ragged_offsets_on_hopper_with_padding_mask() -> Result<()>
{
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
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    graph.set_ragged_offset(q, offsets)?;

    graph.validate_attention_support_surface_for_version(
        91800,
        q,
        k,
        v,
        AttentionConfig::new(DataType::F32)
            .with_padding_mask(sequence_length_query, sequence_length_key_value),
    )?;

    Ok(())
}

#[test]
fn test_attention_support_surface_accepts_ragged_offsets_with_custom_score_subgraph() -> Result<()>
{
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

    let mut subgraph = Graph::new();
    let subgraph_input = subgraph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 2, 4, 4])?,
    ));
    let scale = subgraph.tensor(TensorSpec::scalar_f32(0.8)?);
    let subgraph_output = subgraph.pointwise_binary_infer(
        subgraph_input,
        scale,
        PointwiseMode::Mul,
        DataType::F32,
    )?;

    graph.validate_attention_support_surface_for_version(
        92100,
        q,
        k,
        v,
        AttentionConfig::new(DataType::F32).with_score_subgraph(SdpaScoreSubgraph::new(
            subgraph,
            subgraph_input,
            subgraph_output,
        )),
    )?;

    Ok(())
}

#[test]
fn test_attention_support_surface_accepts_block_mask_on_cudnn_914() -> Result<()> {
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
    let block_mask = graph.tensor(TensorSpec::new(
        DataType::U8,
        Shape::contiguous([1, 2, 1, 1])?,
    ));

    graph.validate_attention_support_surface_for_version(
        91400,
        q,
        k,
        v,
        AttentionConfig::new(DataType::F32).with_block_mask(block_mask),
    )?;
    assert!(
        graph.can_lower_attention_as_unified_with_tensors_for_version(
            91400,
            q,
            k,
            v,
            AttentionConfig::new(DataType::F32).with_block_mask(block_mask),
        )?
    );

    Ok(())
}

#[test]
fn test_attention_unified_path_rejects_block_mask_before_cudnn_914() -> Result<()> {
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
    let block_mask = graph.tensor(TensorSpec::new(
        DataType::U8,
        Shape::contiguous([1, 2, 1, 1])?,
    ));

    assert!(
        !graph.can_lower_attention_as_unified_with_tensors_for_version(
            91399,
            q,
            k,
            v,
            AttentionConfig::new(DataType::F32).with_block_mask(block_mask),
        )?
    );

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_rejects_non_u8_block_mask() -> Result<()> {
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
    let block_mask = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 1, 1])?,
    ));

    let error = graph
        .sdpa_auto_infer(
            SdpaInputs {
                query: q,
                key: k,
                value: v,
                scale: scale,
            },
            AttentionConfig::new(DataType::F32).with_block_mask(block_mask),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa block mask data type"
    ));

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_accepts_alibi_slopes() -> Result<()> {
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
    let alibi = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 1, 1])?,
    ));

    let SdpaOutputs { output, stats: _ } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32)
            .with_alibi(alibi)
            .with_causal_mask(),
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[1, 2, 4, 16]);
    assert!(graph.operations.iter().any(|operation| matches!(
        operation,
        Operation::SdpaForward { score_modifiers, .. }
            if score_modifiers.alibi_slopes == Some(alibi) && score_modifiers.causal_mask
    )));

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_accepts_custom_score_subgraph() -> Result<()> {
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
    let score_subgraph = SdpaScoreSubgraph::new(subgraph, subgraph_input, subgraph_output);

    let SdpaOutputs { output, stats: _ } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32).with_score_subgraph(score_subgraph),
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[1, 2, 4, 16]);
    assert!(graph.operations.iter().any(|operation| matches!(
        operation,
        Operation::SdpaForward { score_subgraph, .. } if score_subgraph.as_ref().as_ref().is_some()
    )));

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_rejects_custom_score_subgraph_with_causal_mask() -> Result<()> {
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
    let subgraph_bias = subgraph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 6])?,
    ));
    let subgraph_output = subgraph.pointwise_binary_infer(
        subgraph_input,
        subgraph_bias,
        PointwiseMode::Add,
        DataType::F32,
    )?;
    let score_subgraph = SdpaScoreSubgraph::new(subgraph, subgraph_input, subgraph_output);

    let err = graph
        .sdpa_auto_infer(
            SdpaInputs {
                query: q,
                key: k,
                value: v,
                scale: scale,
            },
            AttentionConfig::new(DataType::F32)
                .with_causal_mask()
                .with_score_subgraph(score_subgraph),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa score subgraph modifiers"
    ));

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_accepts_custom_score_subgraph_with_bias() -> Result<()> {
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
    let outer_bias = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 6])?,
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
    let score_subgraph = SdpaScoreSubgraph::new(subgraph, subgraph_input, subgraph_output);

    let SdpaOutputs { output, stats: _ } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32)
            .with_bias(outer_bias)
            .with_score_subgraph(score_subgraph),
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[1, 2, 4, 16]);
    assert!(graph.operations.iter().any(|operation| matches!(
        operation,
        Operation::SdpaForward { score_subgraph, .. } if score_subgraph.as_ref().as_ref().is_some()
    )));

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_uses_configured_compute_type_for_composite_matmuls() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_compute_data_type(DataType::F32);

    let q = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 6, 8])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 6, 16])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);
    let outer_bias = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 1, 4, 6])?,
    ));

    let mut subgraph = Graph::new();
    let subgraph_input = subgraph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 6])?,
    ));
    let soft_cap = subgraph.tensor(TensorSpec::scalar_f32(0.8)?);
    let divided = subgraph.pointwise_binary_infer(
        subgraph_input,
        soft_cap,
        PointwiseMode::Div,
        DataType::F32,
    )?;
    let activated = subgraph
        .tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1, 2, 4, 6])?).virtual_tensor());
    subgraph.pointwise(PointwiseOperation::Unary {
        mode: PointwiseMode::TanhFwd,
        input: divided,
        output: activated,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        axis: None,
    });
    let subgraph_output =
        subgraph.pointwise_binary_infer(activated, soft_cap, PointwiseMode::Mul, DataType::F32)?;

    let _outputs = match graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32)
            .with_bias(outer_bias)
            .with_score_subgraph(SdpaScoreSubgraph::new(
                subgraph,
                subgraph_input,
                subgraph_output,
            )),
    ) {
        Ok(result) => result,
        Err(Error::NoAvailableEngines) => return Ok(()),
        Err(Error::Cudnn { code, .. })
            if code == Status::NotSupported
                || code == Status::BadParam
                || code == Status::NotSupportedRuntimePrerequisiteMissing =>
        {
            return Ok(());
        }
        Err(error) => return Err(error),
    };

    let matmul_compute_types = graph
        .operations
        .iter()
        .filter_map(|operation| {
            let Operation::Matmul { config, .. } = operation else {
                return None;
            };
            Some(config.compute_type())
        })
        .collect::<Vec<_>>();

    if matmul_compute_types.is_empty() {
        assert!(
            graph
                .operations
                .iter()
                .any(|operation| matches!(operation, Operation::SdpaForward { .. }))
        );
    } else {
        assert_eq!(matmul_compute_types, vec![DataType::F32, DataType::F32]);
    }

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_accepts_custom_score_subgraph_with_additive_mask() -> Result<()> {
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
    let score_subgraph = SdpaScoreSubgraph::new(subgraph, subgraph_input, subgraph_output);

    let SdpaOutputs { output, stats: _ } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32)
            .with_additive_mask(additive_mask)
            .with_score_subgraph(score_subgraph),
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[1, 2, 4, 16]);
    assert!(graph.operations.iter().any(|operation| matches!(
        operation,
        Operation::SdpaForward { score_subgraph, .. } if score_subgraph.as_ref().as_ref().is_some()
    )));

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_accepts_custom_score_subgraph_with_seq_lens() -> Result<()> {
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
    let score_subgraph = SdpaScoreSubgraph::new(subgraph, subgraph_input, subgraph_output);

    let SdpaOutputs { output, stats: _ } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32)
            .with_sequence_lengths(sequence_length_query, sequence_length_key_value)
            .with_score_subgraph(score_subgraph),
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[2, 2, 4, 16]);
    assert!(graph.operations.iter().any(|operation| matches!(
        operation,
        Operation::SdpaForward { score_subgraph, .. } if score_subgraph.as_ref().as_ref().is_some()
    )));

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_rejects_seq_lens_without_padding_mask_or_custom_score_subgraph()
-> Result<()> {
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

    let err = graph
        .sdpa_auto_infer(
            SdpaInputs {
                query: q,
                key: k,
                value: v,
                scale: scale,
            },
            AttentionConfig::new(DataType::F32)
                .with_sequence_lengths(sequence_length_query, sequence_length_key_value),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa seq lens"
    ));

    Ok(())
}

#[test]
fn test_normalize_attention_config_for_forward_version_clears_zero_dropout_after_8902() {
    let graph = Graph::new();
    let config = AttentionConfig::new(DataType::F32).with_dropout_probability(0.0, 7);

    let normalized = graph.normalize_attention_config_for_forward_version(8903, config.clone());
    assert!(normalized.dropout().is_none());

    let retained = graph.normalize_attention_config_for_forward_version(8902, config);
    assert_eq!(
        retained.dropout().map(|dropout| dropout.probability()),
        Some(0.0)
    );
}

#[test]
fn test_attention_config_clear_dropout_probability_clears_offset() -> Result<()> {
    let mut graph = Graph::new();
    let offset = graph.tensor(
        TensorSpec::new(DataType::I64, Shape::contiguous([1, 1, 1, 1])?)
            .with_scalar_value(ScalarValue::I64(123))?,
    );

    let config = AttentionConfig::new(DataType::F32)
        .with_dropout_probability(0.25, 7)
        .with_dropout_offset(offset)
        .clear_dropout_probability();

    assert!(config.dropout().is_none());
    assert!(config.dropout_offset().is_none());

    Ok(())
}

#[test]
fn test_attention_backward_config_clear_dropout_probability_clears_offset() -> Result<()> {
    let mut graph = Graph::new();
    let offset = graph.tensor(
        TensorSpec::new(DataType::I64, Shape::contiguous([1, 1, 1, 1])?)
            .with_scalar_value(ScalarValue::I64(123))?,
    );

    let config = AttentionBackwardConfig::new(DataType::F32)
        .with_dropout_probability(0.25, 7)
        .with_dropout_offset(offset)
        .clear_dropout_probability();

    assert!(config.dropout().is_none());
    assert!(config.dropout_offset().is_none());

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_accepts_device_seed_internal_dropout() -> Result<()> {
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
    let scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let seed = graph.tensor(
        TensorSpec::new(DataType::I64, Shape::contiguous([1, 1, 1, 1])?)
            .with_scalar_value(ScalarValue::I64(7))?,
    );
    let offset = graph.tensor(
        TensorSpec::new(DataType::I64, Shape::contiguous([1, 1, 1, 1])?)
            .with_scalar_value(ScalarValue::I64(17))?,
    );

    let outputs = graph.sdpa_auto_infer_with_aux(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32)
            .with_dropout_probability_from_tensor(0.25, seed, offset),
    )?;
    let rng_dump = outputs.rng_dump;

    assert_eq!(
        graph.shape(rng_dump.expect("rng dump"))?.dimensions(),
        &[1, 2, 4, 6]
    );

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_rejects_dropout_seed_tensor_with_wrong_data_type() -> Result<()> {
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
    let scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let seed = graph.tensor(
        TensorSpec::new(DataType::I32, Shape::contiguous([1, 1, 1, 1])?)
            .with_scalar_value(ScalarValue::I32(7))?,
    );
    let offset = graph.tensor(
        TensorSpec::new(DataType::I64, Shape::contiguous([1, 1, 1, 1])?)
            .with_scalar_value(ScalarValue::I64(17))?,
    );

    let error = graph
        .sdpa_auto_infer_with_aux(
            SdpaInputs {
                query: q,
                key: k,
                value: v,
                scale: scale,
            },
            AttentionConfig::new(DataType::F32)
                .with_dropout_probability_from_tensor(0.25, seed, offset),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "rng seed data type"
    ));

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_with_aux_treats_zero_dropout_probability_as_no_dropout() -> Result<()> {
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
    let scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);

    let outputs = graph.sdpa_auto_infer_with_aux(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32).with_dropout_probability(0.0, 7),
    )?;
    let rng_dump = outputs.rng_dump;

    if version()? > 8902 {
        assert!(rng_dump.is_none());
        assert!(
            !graph
                .operations
                .iter()
                .any(|operation| matches!(operation, Operation::RandomNumberGenerator { .. }))
        );
    }

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_rejects_dropout_offset_without_probability() -> Result<()> {
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
    let scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let offset = graph.tensor(
        TensorSpec::new(DataType::I64, Shape::contiguous([1, 1, 1, 1])?)
            .with_scalar_value(ScalarValue::I64(123))?,
    );

    let error = graph
        .sdpa_auto_infer_with_aux(
            SdpaInputs {
                query: q,
                key: k,
                value: v,
                scale: scale,
            },
            AttentionConfig::new(DataType::F32).with_dropout_offset(offset),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa dropout offset"
    ));

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_accepts_ragged_custom_score_subgraph_without_padding_mask() -> Result<()> {
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
    let score_subgraph = SdpaScoreSubgraph::new(subgraph, subgraph_input, subgraph_output);

    let SdpaOutputs { output, stats: _ } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32).with_score_subgraph(score_subgraph),
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[2, 2, 4, 16]);
    assert!(graph.operations.iter().any(|operation| matches!(
        operation,
        Operation::SdpaForward { score_subgraph, .. } if score_subgraph.as_ref().as_ref().is_some()
    )));

    Ok(())
}

#[test]
fn test_unified_selector_accepts_ragged_custom_score_subgraph_on_hopper() -> Result<()> {
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
    let score_subgraph = SdpaScoreSubgraph::new(subgraph, subgraph_input, subgraph_output);

    assert!(
        graph.can_lower_attention_as_unified_with_tensors_for_version(
            92100,
            q,
            k,
            v,
            AttentionConfig::new(DataType::F32).with_score_subgraph(score_subgraph),
        )?
    );

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_rejects_custom_score_subgraph_with_padding_mask() -> Result<()> {
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
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));

    let mut subgraph = Graph::new();
    let subgraph_input = subgraph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 6])?,
    ));
    let subgraph_bias = subgraph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 6])?,
    ));
    let subgraph_output = subgraph.pointwise_binary_infer(
        subgraph_input,
        subgraph_bias,
        PointwiseMode::Add,
        DataType::F32,
    )?;
    let score_subgraph = SdpaScoreSubgraph::new(subgraph, subgraph_input, subgraph_output);

    let err = graph
        .sdpa_auto_infer(
            SdpaInputs {
                query: q,
                key: k,
                value: v,
                scale: scale,
            },
            AttentionConfig::new(DataType::F32)
                .with_padding_mask(sequence_length_query, sequence_length_key_value)
                .with_score_subgraph(score_subgraph),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa score subgraph modifiers"
    ));

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_rejects_invalid_alibi_shape() -> Result<()> {
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
    let alibi = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));

    let error = graph
        .sdpa_auto_infer(
            SdpaInputs {
                query: q,
                key: k,
                value: v,
                scale: scale,
            },
            AttentionConfig::new(DataType::F32).with_alibi(alibi),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa alibi shape"
    ));

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_rejects_alibi_without_right_bound_zero() -> Result<()> {
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
    let alibi = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 1, 1])?,
    ));

    let error = graph
        .sdpa_auto_infer(
            SdpaInputs {
                query: q,
                key: k,
                value: v,
                scale: scale,
            },
            AttentionConfig::new(DataType::F32).with_alibi(alibi),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa alibi alignment"
    ));

    Ok(())
}

#[test]
fn test_alibi_bias_uses_relative_position_difference() -> Result<()> {
    let mut graph = Graph::new();
    let scores = graph
        .tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2, 4, 8, 8])?).virtual_tensor());
    let slopes = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 4, 1, 1])?,
    ));

    let _ = graph.alibi_bias(scores, slopes)?;

    let mut has_row_index = false;
    let mut has_col_index = false;
    let mut has_relative_sub = false;
    for operation in graph.operations.iter() {
        match operation {
            Operation::Pointwise(PointwiseOperation::Unary {
                mode: PointwiseMode::GenIndex,
                axis: Some(2),
                ..
            }) => has_row_index = true,
            Operation::Pointwise(PointwiseOperation::Unary {
                mode: PointwiseMode::GenIndex,
                axis: Some(3),
                ..
            }) => has_col_index = true,
            Operation::Pointwise(PointwiseOperation::Binary {
                mode: PointwiseMode::Sub,
                compute_type: DataType::I32,
                ..
            }) => has_relative_sub = true,
            _ => {}
        }
    }

    assert!(has_row_index);
    assert!(has_col_index);
    assert!(has_relative_sub);

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_rejects_invalid_alibi_data_type() -> Result<()> {
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
    let alibi = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 1, 1])?,
    ));

    let error = graph
        .sdpa_auto_infer(
            SdpaInputs {
                query: q,
                key: k,
                value: v,
                scale: scale,
            },
            AttentionConfig::new(DataType::F32)
                .with_alibi(alibi)
                .with_causal_mask(),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa alibi data type"
    ));

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_accepts_sink_token() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 5])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 6, 5])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 6, 7])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let sink_token = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 3, 1, 1])?,
    ));

    let SdpaOutputs { output, stats: _ } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32).with_sink_token(sink_token),
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[2, 3, 4, 7]);

    Ok(())
}

#[test]
fn test_attention_support_surface_rejects_sink_token_before_cudnn_913() -> Result<()> {
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
    let sink_token = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 1, 1])?,
    ));

    let error = graph
        .validate_attention_support_surface_for_version(
            91299,
            q,
            k,
            v,
            AttentionConfig::new(DataType::F32).with_sink_token(sink_token),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa sink token version"
    ));

    Ok(())
}

#[test]
fn test_sdpa_sink_token_rejects_non_rank4_tensor() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 5])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 6, 5])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 6, 7])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 7])?,
    ));
    let sink_token = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([3])?));

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
            SdpaScoreModifiers::new().with_sink_token(sink_token),
            SoftmaxConfig::new(DataType::F32),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa sink token rank"
    ));

    Ok(())
}

#[test]
fn test_sdpa_sink_token_rejects_invalid_shape() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 5])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 6, 5])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 6, 7])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 7])?,
    ));
    let sink_token = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 1, 1])?,
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
            SdpaScoreModifiers::new().with_sink_token(sink_token),
            SoftmaxConfig::new(DataType::F32),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa sink token shape"
    ));

    Ok(())
}

#[test]
fn test_sdpa_sink_token_rejects_invalid_data_type() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 5])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 6, 5])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 6, 7])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 7])?,
    ));
    let sink_token = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 3, 1, 1])?,
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
            SdpaScoreModifiers::new().with_sink_token(sink_token),
            SoftmaxConfig::new(DataType::F32),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa sink token data type"
    ));

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_accepts_paged_kv_inputs() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 4, 8])?,
    ));
    let k_container = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 16, 8])?,
    ));
    let v_container = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 16, 12])?,
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
    let page_table_k = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 3, 1])?,
    ));
    let page_table_v = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 3, 1])?,
    ));

    let SdpaOutputs { output, stats: _ } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k_container,
            value: v_container,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32)
            .with_padding_mask(sequence_length_query, sequence_length_key_value)
            .with_paged_k(AttentionPagedCache::new(
                sequence_length_key_value,
                page_table_k,
            ))
            .with_paged_v(AttentionPagedCache::new(
                sequence_length_key_value,
                page_table_v,
            )),
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[2, 2, 4, 12]);

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_uses_unified_path_with_paged_kv() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 4, 8])?,
    ));
    let k_container = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 16, 8])?,
    ));
    let v_container = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 16, 12])?,
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
    let page_table_k = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 3, 1])?,
    ));
    let page_table_v = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 3, 1])?,
    ));

    let SdpaOutputs { output, stats } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k_container,
            value: v_container,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32)
            .with_stats()
            .with_padding_mask(sequence_length_query, sequence_length_key_value)
            .with_paged_k(AttentionPagedCache::new(
                sequence_length_key_value,
                page_table_k,
            ))
            .with_paged_v(AttentionPagedCache::new(
                sequence_length_key_value,
                page_table_v,
            )),
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[2, 2, 4, 12]);
    assert_eq!(
        graph.shape(stats.expect("stats"))?.dimensions(),
        &[2, 2, 4, 1]
    );
    assert!(
        graph
            .operations
            .iter()
            .any(|operation| matches!(operation, Operation::SdpaForward { .. }))
    );

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_accepts_explicit_paged_max_seq_len_kv() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 4, 8])?,
    ));
    let k_container = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 16, 8])?,
    ));
    let v_container = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 16, 12])?,
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
    let page_table_k = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 3, 1])?,
    ));
    let page_table_v = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 3, 1])?,
    ));

    let SdpaOutputs { output, stats: _ } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k_container,
            value: v_container,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32)
            .with_padding_mask(sequence_length_query, sequence_length_key_value)
            .with_paged_k(AttentionPagedCache::new(
                sequence_length_key_value,
                page_table_k,
            ))
            .with_paged_v(AttentionPagedCache::new(
                sequence_length_key_value,
                page_table_v,
            ))
            .with_max_sequence_length_key_value(20),
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[2, 2, 4, 12]);

    Ok(())
}

#[test]
fn test_sdpa_with_aux_rejects_ragged_output_without_padding_mask() -> Result<()> {
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
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let offsets = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    graph.set_ragged_offset(o, offsets)?;
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);

    let error = graph
        .sdpa_with_aux(
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
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa ragged offsets"
    ));

    Ok(())
}

#[test]
fn test_sdpa_with_aux_rejects_ragged_output_on_pre_hopper_before_91801() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(80);
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
    let offsets = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    graph.set_ragged_offset(o, offsets)?;

    let error = graph
        .validate_sdpa_direct_ragged_support_for_version(
            91800,
            q,
            k,
            v,
            o,
            &DirectSdpaForwardConfig::new()
                .with_sequence_lengths(sequence_length_query, sequence_length_key_value),
            false,
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa ragged offsets"
    ));

    Ok(())
}

#[test]
fn test_validate_sdpa_direct_ragged_support_accepts_custom_score_subgraph_without_padding_mask()
-> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(90);
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
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let offsets = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    graph.set_ragged_offset(o, offsets)?;

    graph.validate_sdpa_direct_ragged_support_for_version(
        92100,
        q,
        k,
        v,
        o,
        &DirectSdpaForwardConfig::new(),
        true,
    )?;

    Ok(())
}

#[test]
fn test_sdpa_with_modifiers_rejects_partial_seq_lens() -> Result<()> {
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
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
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
            SdpaScoreModifiers {
                sequence_length_query: Some(sequence_length_query),
                ..SdpaScoreModifiers::new()
            },
            SoftmaxConfig::new(DataType::F32),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa seq lens"
    ));

    Ok(())
}

#[test]
fn test_sdpa_with_modifiers_rejects_seq_lens_without_padding_mask() -> Result<()> {
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
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
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
            SdpaScoreModifiers::new()
                .with_sequence_lengths(sequence_length_query, sequence_length_key_value),
            SoftmaxConfig::new(DataType::F32),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa seq lens"
    ));

    Ok(())
}

#[test]
fn test_sdpa_with_modifiers_rejects_conflicting_causal_alignments() -> Result<()> {
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
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 8])?,
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
            SdpaScoreModifiers::new()
                .with_causal_mask()
                .with_causal_bottom_right(sequence_length_query, sequence_length_key_value),
            SoftmaxConfig::new(DataType::F32),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa causal alignment"
    ));

    Ok(())
}

#[test]
fn test_sdpa_with_modifiers_rejects_sliding_window_with_bottom_right_alignment() -> Result<()> {
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
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 8])?,
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
            SdpaScoreModifiers::new()
                .with_sliding_window(2, 1)
                .with_causal_bottom_right(sequence_length_query, sequence_length_key_value),
            SoftmaxConfig::new(DataType::F32),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa sliding window alignment"
    ));

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_rejects_internal_and_custom_dropout_together() -> Result<()> {
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
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);
    let dropout_mask = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 4])?,
    ));
    let dropout_scale = graph.tensor(TensorSpec::scalar_f32(2.0)?);

    let err = graph
        .sdpa_auto_infer(
            SdpaInputs {
                query: q,
                key: k,
                value: v,
                scale: scale,
            },
            AttentionConfig::new(DataType::F32)
                .with_dropout(dropout_mask, dropout_scale)
                .with_dropout_probability(0.5, 7),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa dropout"
    ));

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_rejects_max_seq_len_kv_without_paged_attention() -> Result<()> {
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

    let err = graph
        .sdpa_auto_infer(
            SdpaInputs {
                query: q,
                key: k,
                value: v,
                scale: scale,
            },
            AttentionConfig::new(DataType::F32).with_max_sequence_length_key_value(8),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa max_sequence_length_key_value"
    ));

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_rejects_incompatible_max_seq_len_kv_bias() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let k_container = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 16, 8])?,
    ));
    let v_container = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 16, 12])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);
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
        Shape::contiguous([1, 1, 3, 1])?,
    ));
    let page_table_v = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 3, 1])?,
    ));
    let bias = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 18])?,
    ));

    let err = graph
        .sdpa_auto_infer(
            SdpaInputs {
                query: q,
                key: k_container,
                value: v_container,
                scale: scale,
            },
            AttentionConfig::new(DataType::F32)
                .with_padding_mask(sequence_length_query, sequence_length_key_value)
                .with_paged_k(AttentionPagedCache::new(
                    sequence_length_key_value,
                    page_table_k,
                ))
                .with_paged_v(AttentionPagedCache::new(
                    sequence_length_key_value,
                    page_table_v,
                ))
                .with_bias(bias)
                .with_max_sequence_length_key_value(20),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa max_sequence_length_key_value bias"
    ));

    Ok(())
}

#[test]
fn test_sdpa_with_modifiers_rejects_invalid_seq_len_shape() -> Result<()> {
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
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 2, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
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
            SdpaScoreModifiers::new()
                .with_padding_mask(sequence_length_query, sequence_length_key_value),
            SoftmaxConfig::new(DataType::F32),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa seq len shape"
    ));

    Ok(())
}

#[test]
fn test_sdpa_with_modifiers_rejects_partial_dropout() -> Result<()> {
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
    let dropout_mask = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 4, 6])?,
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
            SdpaScoreModifiers {
                dropout_mask: Some(dropout_mask),
                ..SdpaScoreModifiers::new()
            },
            SoftmaxConfig::new(DataType::F32),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa dropout"
    ));

    Ok(())
}

#[test]
fn test_sdpa_with_modifiers_rejects_non_scalar_dropout_scale() -> Result<()> {
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
    let dropout_mask = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 4, 6])?,
    ));
    let dropout_scale = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2])?));

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
            SdpaScoreModifiers::new().with_dropout(dropout_mask, dropout_scale),
            SoftmaxConfig::new(DataType::F32),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa dropout scale"
    ));

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_rejects_paged_kv_without_seq_lens() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 4, 8])?,
    ));
    let k_container = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 16, 8])?,
    ));
    let v_container = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 16, 12])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    let page_table_k = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 3, 1])?,
    ));
    let page_table_v = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 3, 1])?,
    ));

    let err = graph
        .sdpa_auto_infer(
            SdpaInputs {
                query: q,
                key: k_container,
                value: v_container,
                scale: scale,
            },
            AttentionConfig::new(DataType::F32)
                .with_paged_k(AttentionPagedCache::new(
                    sequence_length_key_value,
                    page_table_k,
                ))
                .with_paged_v(AttentionPagedCache::new(
                    sequence_length_key_value,
                    page_table_v,
                )),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa paged seq lens"
    ));

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_rejects_non_i32_page_table() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 4, 8])?,
    ));
    let k_container = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 16, 8])?,
    ));
    let v_container = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 16, 12])?,
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
    let page_table_k = graph.tensor(TensorSpec::new(
        DataType::I64,
        Shape::contiguous([2, 1, 3, 1])?,
    ));
    let page_table_v = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 3, 1])?,
    ));

    let err = graph
        .sdpa_auto_infer(
            SdpaInputs {
                query: q,
                key: k_container,
                value: v_container,
                scale: scale,
            },
            AttentionConfig::new(DataType::F32)
                .with_padding_mask(sequence_length_query, sequence_length_key_value)
                .with_paged_k(AttentionPagedCache::new(
                    sequence_length_key_value,
                    page_table_k,
                ))
                .with_paged_v(AttentionPagedCache::new(
                    sequence_length_key_value,
                    page_table_v,
                )),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa paged k page table data type"
    ));

    Ok(())
}
