mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, bf16},
    error::Result,
    frontend::{
        composite::sdpa::SdpaBackwardInputs,
        graph::{DataTypePolicy, Graph, GraphConfig},
        operation::{
            AttentionBackwardAuxGradientRequest, AttentionBackwardConfig, AttentionMaskMode,
            AttentionScoreConfig, HeuristicMode,
        },
    },
    version,
};

fn init_bf16_image(size: usize, seed: &mut u32) -> Vec<bf16> {
    common::init_f32_image(size, seed)
        .into_iter()
        .map(bf16::from_f32)
        .collect()
}

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;

    if version()? < 91300 {
        println!("frontend_fp16_sdpa_backward_sink_token: skipped, requires cuDNN 9.13+");
        return Ok(());
    }

    let b = 3_i64;
    let h_q = 4_i64;
    let h_k = 4_i64;
    let h_v = 4_i64;
    let s_q = 1024_i64;
    let s_kv = 1024_i64;
    let d_qk = 128_i64;
    let d_v = 128_i64;

    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::BF16)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );

    let q = graph.tensor(
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([b, h_q, s_q, d_qk])?.with_strides(vec![
                h_q * s_q * d_qk,
                s_q * d_qk,
                d_qk,
                1,
            ])?,
        )
        .with_id(1001),
    );
    let k = graph.tensor(
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([b, h_k, d_qk, s_kv])?.with_strides(vec![
                h_k * s_kv * d_qk,
                s_kv * d_qk,
                1,
                d_qk,
            ])?,
        )
        .with_id(1002),
    );
    let v = graph.tensor(
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([b, h_v, d_v, s_kv])?.with_strides(vec![
                h_v * s_kv * d_v,
                s_kv * d_v,
                1,
                d_v,
            ])?,
        )
        .with_id(1003),
    );
    let o = graph.tensor(
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([b, h_q, s_q, d_v])?.with_strides(vec![
                h_q * s_q * d_v,
                s_q * d_v,
                d_v,
                1,
            ])?,
        )
        .with_id(1004),
    );
    let d_o = graph.tensor(
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([b, h_q, s_q, d_v])?.with_strides(vec![
                h_q * s_q * d_v,
                s_q * d_v,
                d_v,
                1,
            ])?,
        )
        .with_id(1101),
    );
    let stats = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([b, h_q, s_q, 1])?.with_strides([h_q * s_q, s_q, 1, 1])?,
        )
        .with_id(1005),
    );
    let scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let sequence_length_query = graph.tensor(
        TensorSpec::new(
            DataType::I32,
            Shape::contiguous([b, 1, 1, 1])?.with_strides([1, 1, 1, 1])?,
        )
        .with_id(1008),
    );
    let sequence_length_key_value = graph.tensor(
        TensorSpec::new(
            DataType::I32,
            Shape::contiguous([b, 1, 1, 1])?.with_strides([1, 1, 1, 1])?,
        )
        .with_id(1009),
    );
    let sink_token = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([1, h_q, 1, 1])?.with_strides([h_q, 1, 1, 1])?,
        )
        .with_id(1010),
    );
    let outputs = graph.sdpa_backward_infer_with_aux(
        SdpaBackwardInputs::new(q, k, v, o, d_o, stats, scale),
        AttentionBackwardConfig::new(DataType::F32)
            .with_attention_scale(0.123)
            .with_score_config(
                AttentionScoreConfig::new()
                    .with_mask_mode(AttentionMaskMode::CausalTopLeftWithPadding {
                        sequence_length_query: sequence_length_query,
                        sequence_length_key_value: sequence_length_key_value,
                    })
                    .with_sink_token(sink_token),
            )
            .with_aux_gradients(AttentionBackwardAuxGradientRequest::sink_token_gradient()),
    )?;
    let d_q = outputs.query_gradient();
    let d_k = outputs.key_gradient();
    let d_v_out = outputs.value_gradient();
    let d_sink_token = outputs.sink_token_gradient();
    graph.replace_tensor(
        d_q,
        graph
            .tensor_config(d_q)?
            .clone()
            .output_tensor()
            .with_id(1102),
    )?;
    graph.replace_tensor(
        d_k,
        graph
            .tensor_config(d_k)?
            .clone()
            .output_tensor()
            .with_id(1103),
    )?;
    graph.replace_tensor(
        d_v_out,
        graph
            .tensor_config(d_v_out)?
            .clone()
            .output_tensor()
            .with_id(1104),
    )?;
    let d_sink_token = d_sink_token.expect("d sink token requested");
    graph.replace_tensor(
        d_sink_token,
        graph
            .tensor_config(d_sink_token)?
            .clone()
            .output_tensor()
            .with_id(1105),
    )?;

    let compiled = graph.compile(&ctx.cudnn, &[HeuristicMode::A])?;

    let mut seed = 123_456_789_u32;
    let mut q_dev =
        DeviceMemory::from_slice(&init_bf16_image((b * h_q * s_q * d_qk) as usize, &mut seed))?;
    let mut k_dev = DeviceMemory::from_slice(&init_bf16_image(
        (b * h_k * s_kv * d_qk) as usize,
        &mut seed,
    ))?;
    let mut v_dev =
        DeviceMemory::from_slice(&init_bf16_image((b * h_v * s_kv * d_v) as usize, &mut seed))?;
    let mut o_dev =
        DeviceMemory::from_slice(&init_bf16_image((b * h_q * s_q * d_v) as usize, &mut seed))?;
    let mut d_o_dev =
        DeviceMemory::from_slice(&init_bf16_image((b * h_q * s_q * d_v) as usize, &mut seed))?;
    let mut stats_dev =
        DeviceMemory::from_slice(&common::init_f32_image((b * h_q * s_q) as usize, &mut seed))?;
    let mut seq_len_q_dev = DeviceMemory::from_slice(&vec![20_i32; b as usize])?;
    let mut seq_len_kv_dev = DeviceMemory::from_slice(&vec![20_i32; b as usize])?;
    let mut sink_token_dev = DeviceMemory::from_slice(&vec![0.0_f32; h_q as usize])?;
    let mut d_q_dev = DeviceMemory::<bf16>::zeroes((b * h_q * s_q * d_qk) as usize)?;
    let mut d_k_dev = DeviceMemory::<bf16>::zeroes((b * h_k * s_kv * d_qk) as usize)?;
    let mut d_v_dev = DeviceMemory::<bf16>::zeroes((b * h_v * s_kv * d_v) as usize)?;
    let mut d_sink_dev = DeviceMemory::from_slice(&vec![0.0_f32; h_q as usize])?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;

    let mut bindings = compiled.bindings();
    bindings.set(q, &mut q_dev)?;
    bindings.set(k, &mut k_dev)?;
    bindings.set(v, &mut v_dev)?;
    bindings.set(o, &mut o_dev)?;
    bindings.set(d_o, &mut d_o_dev)?;
    bindings.set(stats, &mut stats_dev)?;
    bindings.set(sequence_length_query, &mut seq_len_q_dev)?;
    bindings.set(sequence_length_key_value, &mut seq_len_kv_dev)?;
    bindings.set(sink_token, &mut sink_token_dev)?;
    bindings.set(d_q, &mut d_q_dev)?;
    bindings.set(d_k, &mut d_k_dev)?;
    bindings.set(d_v_out, &mut d_v_dev)?;
    bindings.set(d_sink_token, &mut d_sink_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_fp16_sdpa_backward_sink_token", run())
}
