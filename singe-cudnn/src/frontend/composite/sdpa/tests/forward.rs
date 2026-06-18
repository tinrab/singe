use super::*;
use crate::frontend::composite::sdpa::SdpaOutputs;

#[test]
fn test_sdpa_rejects_non_scalar_scale() -> Result<()> {
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
    let scale = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2])?));
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 8])?,
    ));

    let err = graph
        .sdpa(
            SdpaInputs {
                query: q,
                key: k,
                value: v,
                scale: scale,
            },
            SdpaOutputTensors {
                output: o,
                stats: None,
                logit_max: None,
                score_sum_exp: None,
            },
            &SdpaConfig::new(),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa scale"
    ));

    Ok(())
}

#[test]
fn test_sdpa_direct_rejects_wrong_stats_data_type() -> Result<()> {
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
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 1])?,
    ));

    let error = graph
        .sdpa_direct(
            q,
            k,
            v,
            scale,
            o,
            Some(stats),
            None,
            None,
            None,
            None,
            None,
            SdpaScoreModifiers::new(),
            None,
            &SdpaConfig::new().with_stats(),
            DirectSdpaForwardConfig::new(),
            None,
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa stats data type"
    ));

    Ok(())
}

#[test]
fn test_sdpa_direct_rejects_wrong_logit_max_data_type() -> Result<()> {
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
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let logit_max = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 1])?,
    ));

    let error = graph
        .sdpa_direct(
            q,
            k,
            v,
            scale,
            o,
            None,
            Some(logit_max),
            None,
            None,
            None,
            None,
            SdpaScoreModifiers::new(),
            None,
            &SdpaConfig::new().with_logit_max(),
            DirectSdpaForwardConfig::new(),
            None,
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa logit max data type"
    ));

    Ok(())
}

#[test]
fn test_sdpa_direct_accepts_noncontiguous_output_and_aux_strides() -> Result<()> {
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
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?.with_strides([256, 128, 32, 1])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?.with_strides([32, 16, 2, 1])?,
    ));
    let logit_max = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?.with_strides([40, 20, 3, 1])?,
    ));
    let score_sum_exp = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?.with_strides([48, 24, 4, 1])?,
    ));

    graph.sdpa_direct(
        q,
        k,
        v,
        scale,
        o,
        Some(stats),
        Some(logit_max),
        Some(score_sum_exp),
        None,
        None,
        None,
        SdpaScoreModifiers::new(),
        None,
        &SdpaConfig::new()
            .with_stats()
            .with_logit_max()
            .with_score_sum_exp(),
        DirectSdpaForwardConfig::new(),
        None,
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
fn test_sdpa_direct_rejects_partial_softmax_descriptor() -> Result<()> {
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
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let softmax_p = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 6])?,
    ));

    let error = graph
        .sdpa_direct(
            q,
            k,
            v,
            scale,
            o,
            None,
            None,
            None,
            Some(softmax_p),
            None,
            None,
            SdpaScoreModifiers::new(),
            None,
            &SdpaConfig::new(),
            DirectSdpaForwardConfig::new(),
            None,
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa unified softmax descriptor"
    ));

    Ok(())
}

#[test]
fn test_sdpa_direct_rejects_sink_without_softmax_descriptor() -> Result<()> {
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
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let sink = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 1, 1])?,
    ));

    let error = graph
        .sdpa_direct(
            q,
            k,
            v,
            scale,
            o,
            None,
            None,
            None,
            None,
            None,
            Some(sink),
            SdpaScoreModifiers::new(),
            None,
            &SdpaConfig::new(),
            DirectSdpaForwardConfig::new(),
            None,
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa unified softmax descriptor"
    ));

    Ok(())
}

#[test]
fn test_sdpa_direct_rejects_wrong_softmax_descriptor_shape() -> Result<()> {
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
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let softmax_p = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 5])?,
    ));
    let softmax_s = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 6])?,
    ));

    let error = graph
        .sdpa_direct(
            q,
            k,
            v,
            scale,
            o,
            None,
            None,
            None,
            Some(softmax_p),
            Some(softmax_s),
            None,
            SdpaScoreModifiers::new(),
            None,
            &SdpaConfig::new(),
            DirectSdpaForwardConfig::new(),
            None,
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa softmax p"
    ));

    Ok(())
}

#[test]
fn test_sdpa_direct_rejects_wrong_rng_dump_data_type() -> Result<()> {
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
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let dropout_seed = graph.tensor(TensorSpec::new(
        DataType::I64,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let rng_dump = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 6])?,
    ));

    let error = graph
        .sdpa_direct(
            q,
            k,
            v,
            scale,
            o,
            None,
            None,
            None,
            None,
            None,
            None,
            SdpaScoreModifiers::new(),
            None,
            &SdpaConfig::new(),
            DirectSdpaForwardConfig::new()
                .with_dropout(AttentionDropoutConfig::new(0.25, 7))
                .with_dropout_seed(dropout_seed)
                .with_random_number_generator_dump(rng_dump),
            None,
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa rng dump"
    ));

    Ok(())
}

#[test]
fn test_sdpa_direct_rejects_dropout_seed_without_dropout() -> Result<()> {
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
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let dropout_seed = graph.tensor(TensorSpec::new(
        DataType::I64,
        Shape::contiguous([1, 1, 1, 1])?,
    ));

    let error = graph
        .sdpa_direct(
            q,
            k,
            v,
            scale,
            o,
            None,
            None,
            None,
            None,
            None,
            None,
            SdpaScoreModifiers::new(),
            None,
            &SdpaConfig::new(),
            DirectSdpaForwardConfig::new().with_dropout_seed(dropout_seed),
            None,
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa dropout"
    ));

    Ok(())
}

#[test]
fn test_sdpa_infer_creates_output_and_optional_stats() -> Result<()> {
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

    let SdpaOutputs { output, stats } = graph.sdpa_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        &SdpaConfig::new().with_stats(),
    )?;
    let stats = stats.expect("stats");

    assert_eq!(graph.shape(output)?.dimensions(), &[1, 2, 4, 8]);
    assert_eq!(graph.tensor_config(stats)?.data_type, DataType::F32);
    assert_eq!(graph.shape(stats)?.dimensions(), &[1, 2, 4, 1]);

    Ok(())
}

#[test]
fn test_sdpa_infer_accepts_inline_attn_scale_value() -> Result<()> {
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

    let SdpaOutputs { output, stats: _ } = graph.sdpa_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        &SdpaConfig::new().with_attention_scale(0.5),
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[1, 2, 4, 8]);
    assert!(graph.operations.iter().any(|operation| matches!(
        operation,
        Operation::Pointwise(PointwiseOperation::Binary {
            mode: PointwiseMode::Mul,
            ..
        })
    )));

    Ok(())
}

#[test]
fn test_sdpa_with_causal_mask_uses_composite_path() -> Result<()> {
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
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 8])?,
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
            stats: None,
            logit_max: None,
            score_sum_exp: None,
        },
        &SdpaConfig::new().with_causal_mask(),
    )?;

    assert!(
        graph
            .operations
            .iter()
            .all(|operation| !matches!(operation, Operation::SdpaForward { .. }))
    );

    Ok(())
}

#[test]
fn test_sdpa_infer_with_aux_creates_logit_max_and_score_sum_exp() -> Result<()> {
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

    let outputs = graph.sdpa_infer_with_aux(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        &SdpaConfig::new()
            .with_stats()
            .with_logit_max()
            .with_score_sum_exp(),
    )?;
    let o = outputs.output;
    let stats = outputs.stats;
    let logit_max = outputs.logit_max;
    let score_sum_exp = outputs.score_sum_exp;
    let rng_dump = outputs.rng_dump;
    let stats = stats.expect("stats");
    let logit_max = logit_max.expect("logit max");
    let score_sum_exp = score_sum_exp.expect("score sum exp");

    assert_eq!(graph.shape(o)?.dimensions(), &[2, 3, 4, 7]);
    assert_eq!(graph.tensor_config(stats)?.data_type, DataType::F32);
    assert_eq!(graph.tensor_config(logit_max)?.data_type, DataType::F32);
    assert_eq!(graph.tensor_config(score_sum_exp)?.data_type, DataType::F32);
    assert_eq!(graph.shape(stats)?.dimensions(), &[2, 3, 4, 1]);
    assert_eq!(graph.shape(logit_max)?.dimensions(), &[2, 3, 4, 1]);
    assert_eq!(graph.shape(score_sum_exp)?.dimensions(), &[2, 3, 4, 1]);
    assert!(rng_dump.is_none());

    Ok(())
}

#[test]
fn test_sdpa_infer_compiles_when_plan_is_available() -> Result<()> {
    let context = setup_context()?;

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

    let SdpaOutputs { output, stats: _ } = graph.sdpa_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        &SdpaConfig::new(),
    )?;

    match graph.compile(&context, &[HeuristicMode::Instant, HeuristicMode::Fallback]) {
        Ok(compiled) => {
            assert!(compiled.tensor_record(output)?.id.as_i64() > 0);
            let _ = compiled.workspace_size()?;
        }
        Err(Error::NoAvailableEngines) => {}
        Err(Error::Cudnn { code, .. })
            if code == Status::BadParam || code == Status::NotSupported => {}
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_sdpa_bias_rejects_non_rank4_q() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8])?,
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
    let bias = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 8])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 8])?,
    ));

    let err = graph
        .sdpa_bias(
            q,
            k,
            v,
            scale,
            bias,
            o,
            None,
            SoftmaxConfig::new(DataType::F32),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa rank"
    ));

    Ok(())
}

#[test]
fn test_sdpa_with_modifiers_accepts_bias_mask_and_dropout_shapes() -> Result<()> {
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
    let additive_mask = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 1, 4, 6])?,
    ));
    let dropout_mask = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 6])?,
    ));
    let dropout_scale = graph.tensor(TensorSpec::scalar_f32(2.0)?);
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));

    graph.sdpa_with_modifiers(
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
            .with_bias(bias)
            .with_additive_mask(additive_mask)
            .with_dropout(dropout_mask, dropout_scale),
        SoftmaxConfig::new(DataType::F32),
    )?;
    assert_eq!(graph.alias_bindings.len(), 1);

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_uses_unified_path_without_modifiers() -> Result<()> {
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

    let SdpaOutputs { output, stats } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32).with_stats(),
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[1, 2, 4, 16]);
    assert_eq!(
        graph.shape(stats.expect("stats"))?.dimensions(),
        &[1, 2, 4, 1]
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
fn test_sdpa_auto_infer_uses_unified_path_with_bias_modifier() -> Result<()> {
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

    let SdpaOutputs { output, stats: _ } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32).with_bias(bias),
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[1, 2, 4, 16]);
    assert!(graph.operations.iter().any(|operation| matches!(
        operation,
        Operation::SdpaForward { score_modifiers, .. }
            if score_modifiers.bias == Some(bias)
    )));

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_respects_forced_composite_implementation() -> Result<()> {
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

    let SdpaOutputs { output, stats } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32)
            .with_stats()
            .with_implementation(AttentionImplementation::Composite),
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[1, 2, 4, 16]);
    assert_eq!(
        graph.shape(stats.expect("stats"))?.dimensions(),
        &[1, 2, 4, 1]
    );
    assert!(
        graph
            .operations
            .iter()
            .all(|operation| !matches!(operation, Operation::SdpaForward { .. }))
    );

    Ok(())
}

#[test]
fn test_resolve_attention_implementation_exposes_auto_and_forced_choices() -> Result<()> {
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

    assert_eq!(
        graph.resolve_attention_implementation_for_version(
            91301,
            q,
            k,
            v,
            AttentionConfig::new(DataType::F32),
        )?,
        AttentionImplementation::Unified
    );
    assert_eq!(
        graph.resolve_attention_implementation_for_version(
            91300,
            q,
            k,
            v,
            AttentionConfig::new(DataType::F32),
        )?,
        AttentionImplementation::Composite
    );
    assert_eq!(
        graph.resolve_attention_implementation_for_version(
            92100,
            q,
            k,
            v,
            AttentionConfig::new(DataType::F32)
                .with_implementation(AttentionImplementation::Composite),
        )?,
        AttentionImplementation::Composite
    );

    let error = graph
        .resolve_attention_implementation_for_version(
            91300,
            q,
            k,
            v,
            AttentionConfig::new(DataType::F32)
                .with_implementation(AttentionImplementation::Unified),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa unified implementation"
    ));

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_rejects_unsupported_forced_unified_implementation() -> Result<()> {
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
        Shape::contiguous([1, 1, 4, 6])?,
    ));

    match graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32)
            .with_bias(bias)
            .with_implementation(AttentionImplementation::Unified),
    ) {
        Ok(_outputs) => {
            assert!(
                graph
                    .operations
                    .iter()
                    .any(|operation| matches!(operation, Operation::SdpaForward { .. }))
            );
        }
        Err(error) => assert!(matches!(
            error,
            Error::DescriptorMismatch { name } if name == "sdpa unified implementation"
        )),
    }

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_falls_back_to_composite_for_dynamic_shape_graph() -> Result<()> {
    let mut graph = Graph::new();
    graph.enable_dynamic_shape();
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

    let SdpaOutputs { output, stats } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32).with_stats(),
    )?;

    assert_eq!(graph.shape(output)?.dimensions(), &[1, 2, 4, 16]);
    assert_eq!(
        graph.shape(stats.expect("stats"))?.dimensions(),
        &[1, 2, 4, 1]
    );
    assert!(
        graph
            .operations
            .iter()
            .all(|operation| !matches!(operation, Operation::SdpaForward { .. }))
    );

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_uses_unified_path_for_override_shape_graph() -> Result<()> {
    let mut graph = Graph::new();
    graph.enable_override_shape();
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
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
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

    assert_eq!(graph.shape(output)?.dimensions(), &[1, 2, 4, 16]);
    assert_eq!(
        graph.shape(stats.expect("stats"))?.dimensions(),
        &[1, 2, 4, 1]
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
fn test_sdpa_auto_infer_rejects_forced_unified_implementation_for_dynamic_shape_graph() -> Result<()>
{
    let mut graph = Graph::new();
    graph.enable_dynamic_shape();
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

    let error = graph
        .sdpa_auto_infer(
            SdpaInputs {
                query: q,
                key: k,
                value: v,
                scale: scale,
            },
            AttentionConfig::new(DataType::F32)
                .with_stats()
                .with_implementation(AttentionImplementation::Unified),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa unified implementation"
    ));

    Ok(())
}

#[test]
fn test_sdpa_unified_infer_supports_logit_max_and_score_sum_exp_on_cudnn_921() -> Result<()> {
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
    let config = AttentionConfig::new(DataType::F32)
        .with_stats()
        .with_logit_max()
        .with_score_sum_exp();

    assert!(
        graph.can_lower_attention_as_unified_with_tensors_for_version(
            92100,
            q,
            k,
            v,
            config.clone(),
        )?
    );

    let outputs = graph.sdpa_unified_infer(q, k, v, scale, config)?;
    let output = outputs.output;
    let stats = outputs.stats;
    let logit_max = outputs.logit_max;
    let score_sum_exp = outputs.score_sum_exp;
    let rng_dump = outputs.rng_dump;

    assert_eq!(graph.shape(output)?.dimensions(), &[1, 2, 4, 16]);
    assert_eq!(
        graph.shape(stats.expect("stats"))?.dimensions(),
        &[1, 2, 4, 1]
    );
    assert_eq!(
        graph.shape(logit_max.expect("logit max"))?.dimensions(),
        &[1, 2, 4, 1]
    );
    assert_eq!(
        graph
            .shape(score_sum_exp.expect("score sum exp"))?
            .dimensions(),
        &[1, 2, 4, 1]
    );
    assert!(rng_dump.is_none());
    assert!(
        graph
            .operations
            .iter()
            .any(|operation| matches!(operation, Operation::SdpaForward { .. }))
    );

    Ok(())
}

#[test]
fn test_sdpa_unified_infer_supports_sink_token_on_cudnn_921() -> Result<()> {
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
    let sink_token = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 1, 1])?,
    ));
    let config = AttentionConfig::new(DataType::F32).with_sink_token(sink_token);

    assert!(
        graph.can_lower_attention_as_unified_with_tensors_for_version(
            92100,
            q,
            k,
            v,
            config.clone(),
        )?
    );

    let outputs = graph.sdpa_unified_infer(q, k, v, scale, config)?;
    let output = outputs.output;
    let stats = outputs.stats;
    let logit_max = outputs.logit_max;
    let score_sum_exp = outputs.score_sum_exp;
    let rng_dump = outputs.rng_dump;

    assert_eq!(graph.shape(output)?.dimensions(), &[1, 2, 4, 16]);
    assert!(stats.is_none());
    assert!(logit_max.is_none());
    assert!(score_sum_exp.is_none());
    assert!(rng_dump.is_none());
    assert!(graph.operations.iter().any(|operation| matches!(
        operation,
        Operation::SdpaForward {
            sink: Some(found_sink),
            score_modifiers,
            ..
        } if *found_sink == sink_token && score_modifiers.sink_token == Some(sink_token)
    )));

    Ok(())
}

#[test]
fn test_sdpa_unified_infer_supports_block_mask_on_cudnn_914() -> Result<()> {
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
        DataType::U8,
        Shape::contiguous([1, 2, 1, 1])?,
    ));
    let config = AttentionConfig::new(DataType::F32).with_block_mask(block_mask);

    assert!(
        graph.can_lower_attention_as_unified_with_tensors_for_version(
            91400,
            q,
            k,
            v,
            config.clone(),
        )?
    );

    let outputs = graph.sdpa_unified_infer(q, k, v, scale, config)?;
    let output = outputs.output;
    let stats = outputs.stats;
    let logit_max = outputs.logit_max;
    let score_sum_exp = outputs.score_sum_exp;
    let rng_dump = outputs.rng_dump;

    assert_eq!(graph.shape(output)?.dimensions(), &[1, 2, 4, 16]);
    assert!(stats.is_none());
    assert!(logit_max.is_none());
    assert!(score_sum_exp.is_none());
    assert!(rng_dump.is_none());
    assert!(
        graph
            .operations
            .iter()
            .any(|operation| matches!(operation, Operation::SdpaForward { .. }))
    );

    Ok(())
}

#[test]
fn test_sdpa_auto_infer_with_aux_creates_optional_outputs() -> Result<()> {
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
    let bias = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 6])?,
    ));

    let outputs = graph.sdpa_auto_infer_with_aux(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32)
            .with_bias(bias)
            .with_stats()
            .with_logit_max()
            .with_score_sum_exp(),
    )?;
    let o = outputs.output;
    let stats = outputs.stats;
    let logit_max = outputs.logit_max;
    let score_sum_exp = outputs.score_sum_exp;
    let rng_dump = outputs.rng_dump;

    assert_eq!(graph.shape(o)?.dimensions(), &[2, 3, 4, 7]);
    assert_eq!(
        graph.shape(stats.expect("stats"))?.dimensions(),
        &[2, 3, 4, 1]
    );
    assert_eq!(
        graph.shape(logit_max.expect("logit max"))?.dimensions(),
        &[2, 3, 4, 1]
    );
    assert_eq!(
        graph
            .shape(score_sum_exp.expect("score sum exp"))?
            .dimensions(),
        &[2, 3, 4, 1]
    );
    assert!(rng_dump.is_none());

    Ok(())
}

#[test]
fn test_sdpa_fp8_infer_creates_stats_and_absolute_max_outputs() -> Result<()> {
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
    let descale_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);

    let outputs = graph.sdpa_fp8_infer(
        SdpaFp8Inputs::new(
            q, k, v, descale_q, descale_k, descale_v, descale_s, scale_s, scale_o,
        ),
        &SdpaConfig::new()
            .with_attention_scale(1.0)
            .with_stats()
            .with_absolute_max_s()
            .with_absolute_max_o(),
    )?;
    let o = outputs.output;
    let stats = outputs.stats;
    let absolute_max_s = outputs.logit_max;
    let absolute_max_o = outputs.score_sum_exp;
    let rng_dump = outputs.rng_dump;

    assert_eq!(graph.shape(o)?.dimensions(), &[1, 2, 4, 16]);
    assert_eq!(
        graph.shape(stats.expect("stats"))?.dimensions(),
        &[1, 2, 4, 1]
    );
    assert_eq!(
        graph
            .shape(absolute_max_s.expect("absolute_max_s"))?
            .dimensions(),
        &[1, 1, 1, 1]
    );
    assert_eq!(
        graph
            .shape(absolute_max_o.expect("absolute_max_o"))?
            .dimensions(),
        &[1, 1, 1, 1]
    );
    assert!(rng_dump.is_none());

    Ok(())
}

#[test]
fn test_sdpa_fp8_with_aux_uses_provided_outputs() -> Result<()> {
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
    let descale_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let absolute_max_s = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let absolute_max_o = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));

    graph.sdpa_fp8_with_aux(
        SdpaFp8Inputs::new(
            q, k, v, descale_q, descale_k, descale_v, descale_s, scale_s, scale_o,
        ),
        SdpaFp8OutputTensors {
            output: o,
            stats: Some(stats),
            absolute_max_scores: Some(absolute_max_s),
            absolute_max_output: Some(absolute_max_o),
        },
        &SdpaConfig::new()
            .with_attention_scale(1.0)
            .with_stats()
            .with_absolute_max_s()
            .with_absolute_max_o(),
    )?;

    assert!(
        graph.operations.iter().any(
            |operation| matches!(operation, Operation::Reshape { output, .. } if *output == o)
        )
    );
    assert!(graph.operations.iter().any(
        |operation| matches!(operation, Operation::Reshape { output, .. } if *output == stats)
    ));
    assert!(graph.operations.iter().any(
        |operation| matches!(operation, Operation::Reshape { output, .. } if *output == absolute_max_s)
    ));
    assert!(graph.operations.iter().any(
        |operation| matches!(operation, Operation::Reshape { output, .. } if *output == absolute_max_o)
    ));

    Ok(())
}

#[test]
fn test_sdpa_fp8_infer_with_output_type_uses_bf16_on_blackwell() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(100);

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
    let descale_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_s = graph.tensor(TensorSpec::scalar_f32(448.0)?);
    let scale_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);

    let outputs = graph.sdpa_fp8_infer_with_output_type(
        SdpaFp8Inputs::new(
            q, k, v, descale_q, descale_k, descale_v, descale_s, scale_s, scale_o,
        ),
        DataType::BF16,
        &SdpaConfig::new()
            .with_attention_scale(1.0)
            .with_stats()
            .with_absolute_max_s()
            .with_absolute_max_o(),
    )?;
    let o = outputs.output;
    let stats = outputs.stats;
    let absolute_max_s = outputs.logit_max;
    let absolute_max_o = outputs.score_sum_exp;
    let rng_dump = outputs.rng_dump;

    assert_eq!(graph.tensor_config(o)?.data_type, DataType::BF16);
    assert_eq!(
        graph.tensor_config(stats.expect("stats"))?.data_type,
        DataType::F32
    );
    assert_eq!(
        graph
            .tensor_config(absolute_max_s.expect("absolute_max_s"))?
            .data_type,
        DataType::F32
    );
    assert_eq!(
        graph
            .tensor_config(absolute_max_o.expect("absolute_max_o"))?
            .data_type,
        DataType::F32
    );
    assert!(rng_dump.is_none());

    Ok(())
}

#[test]
fn test_sdpa_fp8_infer_accepts_bottom_right_causal_inputs_on_blackwell() -> Result<()> {
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
    let descale_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));

    let outputs = graph.sdpa_fp8_infer(
        SdpaFp8Inputs::new(
            q, k, v, descale_q, descale_k, descale_v, descale_s, scale_s, scale_o,
        ),
        &SdpaConfig::new()
            .with_attention_scale(1.0)
            .with_causal_bottom_right(sequence_length_query, sequence_length_key_value)
            .with_stats()
            .with_absolute_max_s()
            .with_absolute_max_o(),
    )?;
    let o = outputs.output;
    let stats = outputs.stats;
    let absolute_max_s = outputs.logit_max;
    let absolute_max_o = outputs.score_sum_exp;
    let rng_dump = outputs.rng_dump;

    assert_eq!(graph.shape(o)?.dimensions(), &[1, 2, 64, 128]);
    assert_eq!(
        graph.shape(stats.expect("stats"))?.dimensions(),
        &[1, 2, 64, 1]
    );
    assert_eq!(
        graph
            .shape(absolute_max_s.expect("absolute_max_s"))?
            .dimensions(),
        &[1, 1, 1, 1]
    );
    assert_eq!(
        graph
            .shape(absolute_max_o.expect("absolute_max_o"))?
            .dimensions(),
        &[1, 1, 1, 1]
    );
    assert!(rng_dump.is_none());

    Ok(())
}

#[test]
fn test_sdpa_fp8_with_aux_accepts_bf16_output_on_blackwell() -> Result<()> {
    let mut graph = Graph::new();
    graph.set_sm_version(100);

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
    let descale_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_s = graph.tensor(TensorSpec::scalar_f32(448.0)?);
    let scale_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let o = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let absolute_max_s = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let absolute_max_o = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));

    graph.sdpa_fp8_with_aux(
        SdpaFp8Inputs::new(
            q, k, v, descale_q, descale_k, descale_v, descale_s, scale_s, scale_o,
        ),
        SdpaFp8OutputTensors {
            output: o,
            stats: Some(stats),
            absolute_max_scores: Some(absolute_max_s),
            absolute_max_output: Some(absolute_max_o),
        },
        &SdpaConfig::new()
            .with_attention_scale(1.0)
            .with_stats()
            .with_absolute_max_s()
            .with_absolute_max_o(),
    )?;

    assert!(
        graph.operations.iter().any(
            |operation| matches!(operation, Operation::Reshape { output, .. } if *output == o)
        )
    );

    Ok(())
}

#[test]
fn test_sdpa_fp8_builder_uses_provided_outputs() -> Result<()> {
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
    let descale_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let absolute_max_s = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let absolute_max_o = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));

    graph.sdpa_fp8(
        SdpaFp8Inputs::new(
            q, k, v, descale_q, descale_k, descale_v, descale_s, scale_s, scale_o,
        ),
        SdpaFp8OutputTensors {
            output: o,
            stats: Some(stats),
            absolute_max_scores: Some(absolute_max_s),
            absolute_max_output: Some(absolute_max_o),
        },
        &SdpaConfig::new()
            .with_attention_scale(1.0)
            .with_stats()
            .with_absolute_max_s()
            .with_absolute_max_o(),
    )?;

    assert!(
        graph.operations.iter().any(
            |operation| matches!(operation, Operation::Reshape { output, .. } if *output == o)
        )
    );

    Ok(())
}

#[test]
fn test_sdpa_fp8_infer_rejects_non_scalar_descale_tensor() -> Result<()> {
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
    let descale_q = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2])?));
    let descale_k = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_v = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_s = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);

    let err = graph
        .sdpa_fp8_infer(
            SdpaFp8Inputs::new(
                q, k, v, descale_q, descale_k, descale_v, descale_s, scale_s, scale_o,
            ),
            &SdpaConfig::new().with_attention_scale(1.0),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa fp8 descale_q"
    ));

    Ok(())
}

#[test]
fn test_sdpa_mxfp8_infer_creates_stats_and_absolute_max_output() -> Result<()> {
    let mut graph = Graph::new();
    let _block_size = 32_i64;
    let q_dims = vec![1, 2, 4, 32];
    let k_dims = vec![1, 2, 6, 32];
    let v_dims = vec![1, 2, 6, 16];
    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous(q_dims.clone())?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous(k_dims.clone())?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous(v_dims.clone())?,
    ));
    let scale_q = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let scale_k = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 6, 1])?,
    ));
    let scale_v = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 1, 16])?,
    ));
    let attention_scale = graph.tensor(
        TensorSpec::new(DataType::F32, Shape::contiguous([1, 1, 1, 1])?)
            .with_scalar_value(ScalarValue::F32(1.0))?,
    );

    let outputs = graph.sdpa_mxfp8_infer(
        SdpaMxfp8Inputs::new(q, k, v, scale_q, scale_k, scale_v, attention_scale),
        &SdpaConfig::new().with_stats().with_absolute_max_o(),
    )?;
    let o = outputs.output;
    let stats = outputs.stats;
    let absolute_max_o = outputs.absolute_max_output;

    assert_eq!(graph.shape(o)?.dimensions(), &[1, 2, 4, 16]);
    assert_eq!(
        graph.shape(stats.expect("stats"))?.dimensions(),
        &[1, 2, 4, 1]
    );
    assert_eq!(
        graph
            .shape(absolute_max_o.expect("absolute_max_o"))?
            .dimensions(),
        &[1, 1, 1, 1]
    );

    Ok(())
}

#[test]
fn test_fp8_forward_support_surface_rejects_bf16_output_before_blackwell_or_913() {
    let mut graph = Graph::new();
    graph.set_sm_version(100);
    let error = graph
        .validate_fp8_forward_support_surface_for_version(91299, DataType::BF16)
        .unwrap_err();
    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa fp8 output data type"
    ));

    graph.set_sm_version(90);
    let error = graph
        .validate_fp8_forward_support_surface_for_version(91300, DataType::BF16)
        .unwrap_err();
    assert!(matches!(
        error,
        Error::DescriptorMismatch { name } if name == "sdpa fp8 output data type"
    ));
}

#[test]
fn test_sdpa_mxfp8_with_aux_uses_provided_outputs() -> Result<()> {
    let mut graph = Graph::new();
    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 64])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 64])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 16])?,
    ));
    let scale_q = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 4, 2])?,
    ));
    let scale_k = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 6, 2])?,
    ));
    let scale_v = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 1, 16])?,
    ));
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let o = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let absolute_max_o = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));

    graph.sdpa_mxfp8_with_aux(
        SdpaMxfp8Inputs::new(q, k, v, scale_q, scale_k, scale_v, attention_scale),
        SdpaMxfp8OutputTensors {
            output: o,
            stats: Some(stats),
            absolute_max_output: Some(absolute_max_o),
        },
        &SdpaConfig::new().with_stats().with_absolute_max_o(),
    )?;

    assert!(
        graph.operations.iter().any(
            |operation| matches!(operation, Operation::Reshape { output, .. } if *output == o)
        )
    );
    assert!(graph.operations.iter().any(
        |operation| matches!(operation, Operation::Reshape { output, .. } if *output == stats)
    ));
    assert!(graph.operations.iter().any(
        |operation| matches!(operation, Operation::Reshape { output, .. } if *output == absolute_max_o)
    ));

    Ok(())
}

#[test]
fn test_sdpa_mxfp8_builder_uses_provided_outputs() -> Result<()> {
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
    let scale_q = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let scale_k = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 6, 1])?,
    ));
    let scale_v = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 1, 16])?,
    ));
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let o = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let absolute_max_o = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));

    graph.sdpa_mxfp8(
        SdpaMxfp8Inputs::new(q, k, v, scale_q, scale_k, scale_v, attention_scale),
        SdpaMxfp8OutputTensors {
            output: o,
            stats: Some(stats),
            absolute_max_output: Some(absolute_max_o),
        },
        &SdpaConfig::new().with_stats().with_absolute_max_o(),
    )?;

    assert!(
        graph.operations.iter().any(
            |operation| matches!(operation, Operation::Reshape { output, .. } if *output == o)
        )
    );

    Ok(())
}
