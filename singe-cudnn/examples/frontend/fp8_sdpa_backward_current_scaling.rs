mod common;

use std::env::args;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, f8e4m3},
    error::{Error, Result},
    frontend::{
        composite::sdpa::{SdpaFp8BackwardGradientTensors, SdpaFp8BackwardInputs},
        graph::{DataTypePolicy, Graph, GraphConfig},
        operation::{
            AttentionBackwardConfig, AttentionDiagonalAlignment, AttentionDiagonalBand,
            AttentionDropoutMask, AttentionDropoutMode, AttentionMaskMode, AttentionScoreConfig,
        },
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
};

fn run() -> Result<()> {
    let example = common::ExampleContext::create()?;

    let gradient_type = if args().any(|argument| argument == "--fp16-gradients") {
        DataType::F16
    } else {
        DataType::BF16
    };
    let b = 2_i64;
    let h = 2_i64;
    let s = 512_i64;
    let d = 128_i64;

    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F8E4M3)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );

    let qkvo_dims = [b, h, s, d];
    let qkv_strides = [s * 3 * h * d, d, 3 * h * d, 1];
    let o_strides = [s * h * d, d, h * d, 1];

    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::with_strides(qkvo_dims, qkv_strides)?,
    ));
    let k = graph.tensor_like(q, Some("K"))?;
    let v = graph.tensor_like(q, Some("V"))?;
    let o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::with_strides(qkvo_dims, o_strides)?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::with_strides(qkvo_dims, o_strides)?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::with_strides([b, h, s, 1], [h * s, s, 1, 1])?,
    ));

    let attention_scale = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let descale_q = graph.tensor_like(attention_scale, Some("Descale_Q"))?;
    let descale_k = graph.tensor_like(descale_q, Some("Descale_K"))?;
    let descale_v = graph.tensor_like(descale_q, Some("Descale_V"))?;
    let descale_o = graph.tensor_like(descale_q, Some("Descale_O"))?;
    let descale_d_o = graph.tensor_like(descale_q, Some("Descale_dO"))?;
    let descale_s = graph.tensor_like(descale_q, Some("Descale_S"))?;
    let descale_d_p = graph.tensor_like(descale_q, Some("Descale_dP"))?;
    let scale_s = graph.tensor(TensorSpec::scalar_f32(448.0));
    let scale_d_qkv = graph.tensor(TensorSpec::scalar_f32(1.0));
    let scale_d_p = graph.tensor_like(descale_q, Some("Scale_dP"))?;

    // Gradients use their own unpacked BSHD storage instead of inheriting the interleaved QKV strides.
    let gradient = TensorSpec::new(
        gradient_type,
        Shape::with_strides([b, h, s, d], [s * h * d, d, h * d, 1])?,
    );
    let d_q = graph.tensor(gradient.clone().with_name("dQ"));
    let d_k = graph.tensor(gradient.clone().with_name("dK"));
    let d_v = graph.tensor(gradient.with_name("dV"));
    let amax = TensorSpec::new(DataType::F32, Shape::contiguous([1, 1, 1, 1])?);
    let absolute_max_d_q = graph.tensor(amax.clone());
    let absolute_max_d_k = graph.tensor(amax.clone());
    let absolute_max_d_v = graph.tensor(amax.clone());
    let absolute_max_d_p = graph.tensor(amax);
    let mut score_config =
        AttentionScoreConfig::new().with_mask_mode(AttentionMaskMode::DiagonalBand(
            AttentionDiagonalBand::causal(AttentionDiagonalAlignment::TopLeft),
        ));
    let mut backward_config = AttentionBackwardConfig::new().with_compute_type(DataType::F32);
    let dropout_mask = if args().any(|argument| argument == "--custom-dropout") {
        let mask = graph.tensor(
            TensorSpec::new(DataType::F32, Shape::contiguous([b, h, s, s])?)
                .with_name("dropout_mask"),
        );
        let scale = graph.tensor(TensorSpec::scalar_f32(1.25));
        let inverse = graph.tensor(TensorSpec::scalar_f32(0.8));
        score_config = score_config.with_dropout_mode(AttentionDropoutMode::CustomMask(
            AttentionDropoutMask::new(mask, scale),
        ));
        backward_config = backward_config
            .with_deterministic_algorithm(true)
            .with_dropout_scale_inverse(inverse);
        if args().any(|argument| argument == "--derive-dropout-inverse") {
            backward_config = backward_config.with_dropout_scale_inverse(None);
        }
        if args().any(|argument| argument == "--clear-dropout") {
            score_config = score_config.with_dropout_mode(AttentionDropoutMode::None);
            None
        } else {
            Some(mask)
        }
    } else {
        None
    };
    let sink_gradient = if args().any(|argument| argument == "--sink-gradient") {
        let shape = Shape::contiguous([1, h, 1, 1])?;
        let sink = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()).with_name("sink"));
        let gradient = graph.tensor(TensorSpec::new(DataType::F32, shape).with_name("dSink"));
        score_config = score_config.with_sink_token(sink);
        Some((sink, gradient))
    } else {
        None
    };
    graph.sdpa_fp8_backward_into(
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
            scale_d_qkv,
            scale_d_qkv,
            scale_d_qkv,
            scale_d_p,
        ),
        SdpaFp8BackwardGradientTensors {
            query_gradient: d_q,
            key_gradient: d_k,
            value_gradient: d_v,
            sink_token_gradient: sink_gradient.map(|(_, gradient)| gradient),
            absolute_max_query_gradient: absolute_max_d_q,
            absolute_max_key_gradient: absolute_max_d_k,
            absolute_max_value_gradient: absolute_max_d_v,
            absolute_max_probability_gradient: absolute_max_d_p,
        },
        backward_config.with_score_config(score_config),
    )?;

    let mut output_ids = vec![
        d_q,
        d_k,
        d_v,
        absolute_max_d_q,
        absolute_max_d_k,
        absolute_max_d_v,
        absolute_max_d_p,
    ];
    output_ids.extend(sink_gradient.map(|(_, gradient)| gradient));
    for &output in &output_ids {
        graph.mark_as_output(output)?;
    }

    let head_elements = Shape::contiguous([h, d])?.element_count()?;
    let value_offset = head_elements.checked_mul(2).ok_or(Error::OutOfRange {
        name: "interleaved value offset".into(),
    })?;
    let qkv_elements = graph
        .shape(q)?
        .storage_span()?
        .checked_add(value_offset)
        .ok_or(Error::OutOfRange {
            name: "interleaved qkv storage".into(),
        })?;
    let output_elements = graph.shape(o)?.storage_span()?;
    let output_gradient_elements = graph.shape(d_o)?.storage_span()?;
    let stats_elements = graph.shape(stats)?.storage_span()?;
    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
    )?;
    let cuda = example.cudnn.cuda_context();
    let qkv_dev = DeviceMemory::<f8e4m3>::zeroed(cuda, qkv_elements)?;
    let output_dev = DeviceMemory::<f8e4m3>::zeroed(cuda, output_elements)?;
    let output_gradient_dev = DeviceMemory::<f8e4m3>::zeroed(cuda, output_gradient_elements)?;
    let stats_dev = DeviceMemory::<f32>::zeroed(cuda, stats_elements)?;
    let sink_dev = if let Some((sink, _)) = sink_gradient {
        let values = vec![0.0_f32; graph.shape(sink)?.storage_span()?];
        Some((sink, DeviceMemory::from_slice(cuda, &values)?))
    } else {
        None
    };
    let dropout_mask_dev = if let Some(mask) = dropout_mask {
        let values = (0..graph.shape(mask)?.storage_span()?)
            .map(|index| if index % 5 == 0 { 0.0_f32 } else { 1.0 })
            .collect::<Vec<_>>();
        Some((mask, DeviceMemory::from_slice(cuda, &values)?))
    } else {
        None
    };
    let unit_scale_dev = DeviceMemory::from_slice(cuda, &[1.0_f32])?;
    let descale_s_dev = DeviceMemory::from_slice(cuda, &[1.0_f32 / 448.0])?;
    let attention_scale_dev = DeviceMemory::from_slice(cuda, &[0.123_f32])?;
    let mut output_buffers = output_ids
        .into_iter()
        .map(|id| {
            let bytes = graph.tensor_spec(id)?.storage_byte_len(None)?;
            Ok((id, DeviceMemory::<u8>::zeroed(cuda, bytes)?))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut workspace = common::workspace(cuda, compiled.workspace_size())?;
    let mut bindings = compiled.bindings()?;
    bindings.set_input(q, &qkv_dev.view())?;
    bindings.set_input(k, &qkv_dev.view().slice(head_elements..)?)?;
    bindings.set_input(v, &qkv_dev.view().slice(value_offset..)?)?;
    bindings.set_input(o, &output_dev.view())?;
    bindings.set_input(d_o, &output_gradient_dev.view())?;
    bindings.set_input(stats, &stats_dev.view())?;
    if let Some((sink, memory)) = &sink_dev {
        bindings.set_input(*sink, &memory.view())?;
    }
    if let Some((mask, memory)) = &dropout_mask_dev {
        bindings.set_input(*mask, &memory.view())?;
    }
    for scale in [
        descale_q,
        descale_k,
        descale_v,
        descale_o,
        descale_d_o,
        descale_d_p,
        scale_d_p,
    ] {
        bindings.set_input(scale, &unit_scale_dev.view())?;
    }
    bindings.set_input(descale_s, &descale_s_dev.view())?;
    bindings.set_input(attention_scale, &attention_scale_dev.view())?;
    for (id, memory) in &mut output_buffers {
        bindings.set_encoded_bytes(*id, memory)?;
    }
    // SAFETY: initialized inputs, optional sink and mask, gradients, scales and workspace remain live through completion.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_fp8_sdpa_backward_current_scaling", run())
}
