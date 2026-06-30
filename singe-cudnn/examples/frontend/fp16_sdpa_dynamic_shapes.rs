mod common;

use std::sync::{Arc, Mutex};

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::execution::KernelCache,
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, bf16},
    error::{Error, Result, Status},
    frontend::{
        composite::sdpa::SdpaInputs,
        graph::{DataTypePolicy, Graph, GraphConfig},
        operation::{
            AttentionConfig, AttentionMaskMode, AttentionScoreConfig, HeuristicMode,
            SdpaAuxOutputRequest,
        },
    },
};

fn init_bf16_image(size: usize, seed: &mut u32) -> Vec<bf16> {
    common::init_f32_image(size, seed)
        .into_iter()
        .map(bf16::from_f32)
        .collect()
}

fn is_not_supported(error: &Error) -> bool {
    matches!(
        error,
        Error::Cudnn { code, .. }
            if *code == Status::NotSupported
                || *code == Status::NotSupportedGraphPattern
                || *code == Status::NotSupportedShape
                || *code == Status::NotSupportedDataType
                || *code == Status::NotSupportedLayout
    )
}

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;

    if singe_cudnn::version()? < 92_100 {
        println!("skipping: test is disabled till backend is updated");
        return Ok(());
    }

    let b = 2_i64;
    let override_b = 4_i64;
    let h_q = 4_i64;
    let h_k = 4_i64;
    let h_v = 4_i64;
    let s_q = 1024_i64;
    let s_kv = 1024_i64;
    let d_qk = 128_i64;
    let d_v = 128_i64;
    let kernel_cache = Arc::new(Mutex::new(KernelCache::create()?));

    let mut graph = Graph::with_config(
        GraphConfig::new()
            .with_dynamic_shape()
            .with_override_shape()
            .with_data_type_policy(
                DataTypePolicy::new()
                    .with_io(DataType::BF16)
                    .with_intermediate(DataType::F32)
                    .with_compute(DataType::F32),
            ),
    );
    graph.attach_shared_kernel_cache(Arc::clone(&kernel_cache))?;

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
            Shape::contiguous([b, h_k, s_kv, d_qk])?.with_strides(vec![
                h_k * s_kv * d_qk,
                s_kv * d_qk,
                d_qk,
                1,
            ])?,
        )
        .with_id(1002),
    );
    let v = graph.tensor(
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([b, h_v, s_kv, d_v])?.with_strides(vec![
                h_v * s_kv * d_v,
                s_kv * d_v,
                d_v,
                1,
            ])?,
        )
        .with_id(1003),
    );
    let sequence_length_query = graph.tensor(
        TensorSpec::new(
            DataType::I32,
            Shape::contiguous([b, 1, 1, 1])?.with_strides([1, 1, 1, 1])?,
        )
        .with_id(1007),
    );
    let sequence_length_key_value = graph.tensor(
        TensorSpec::new(
            DataType::I32,
            Shape::contiguous([b, 1, 1, 1])?.with_strides([1, 1, 1, 1])?,
        )
        .with_id(1008),
    );
    let scale = graph.tensor(TensorSpec::scalar_f32(0.123)?);

    let outputs = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32)
            .with_aux_outputs(SdpaAuxOutputRequest::stats_only())
            .with_score_config(AttentionScoreConfig::new().with_mask_mode(
                AttentionMaskMode::CausalTopLeftWithPadding {
                    sequence_length_query: sequence_length_query,
                    sequence_length_key_value: sequence_length_key_value,
                },
            )),
    )?;
    let o = outputs.output();
    let stats = outputs.stats();

    graph.replace_tensor(
        o,
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([b, h_q, s_q, d_v])?.with_strides(vec![
                h_q * d_v,
                d_v,
                b * h_q * d_v,
                1,
            ])?,
        )
        .output_tensor()
        .with_id(1004),
    )?;

    let stats = stats.expect("stats requested");
    let stats_tensor = graph
        .tensor_config(stats)?
        .clone()
        .output_tensor()
        .with_id(1005);
    graph.replace_tensor(stats, stats_tensor)?;

    graph.record_dynamic_shape_constraints(q, [1, h_q, s_q, d_qk], [override_b, h_q, s_q, d_qk])?;
    graph.record_dynamic_shape_constraints(
        k,
        [1, h_k, s_kv, d_qk],
        [override_b, h_k, s_kv, d_qk],
    )?;
    graph.record_dynamic_shape_constraints(v, [1, h_v, s_kv, d_v], [override_b, h_v, s_kv, d_v])?;
    graph.record_dynamic_shape_constraints(o, [1, h_q, s_q, d_v], [override_b, h_q, s_q, d_v])?;
    graph.record_dynamic_shape_constraints(stats, [1, h_q, s_q, 1], [override_b, h_q, s_q, 1])?;
    graph.record_dynamic_shape_constraints(
        sequence_length_query,
        [1, 1, 1, 1],
        [override_b, 1, 1, 1],
    )?;
    graph.record_dynamic_shape_constraints(
        sequence_length_key_value,
        [1, 1, 1, 1],
        [override_b, 1, 1, 1],
    )?;

    let compiled = match graph.compile(&ctx.cudnn, &[HeuristicMode::A]) {
        Ok(compiled) => compiled,
        Err(error) if is_not_supported(&error) => {
            println!(
                "frontend_fp16_sdpa_dynamic_shapes: skipped, not supported on current configuration"
            );
            return Ok(());
        }
        Err(error) => return Err(error),
    };

    let mut seed = 123_456_789_u32;
    let mut q_dev =
        DeviceMemory::from_slice(&init_bf16_image((b * h_q * s_q * d_qk) as usize, &mut seed))?;
    let mut k_dev = DeviceMemory::from_slice(&init_bf16_image(
        (b * h_k * s_kv * d_qk) as usize,
        &mut seed,
    ))?;
    let mut v_dev =
        DeviceMemory::from_slice(&init_bf16_image((b * h_v * s_kv * d_v) as usize, &mut seed))?;
    let mut seq_len_q_dev = DeviceMemory::from_slice(&vec![20_i32; b as usize])?;
    let mut seq_len_kv_dev = DeviceMemory::from_slice(&vec![20_i32; b as usize])?;
    let mut o_dev = DeviceMemory::<bf16>::zeroes((b * h_q * s_q * d_v) as usize)?;
    let mut stats_dev = DeviceMemory::<f32>::zeroes((b * h_q * s_q) as usize)?;
    let mut workspace = common::workspace(compiled.workspace_size()?.max(256 * 1024))?;

    let mut bindings = compiled.bindings();
    bindings.set(q, &mut q_dev)?;
    bindings.set(k, &mut k_dev)?;
    bindings.set(v, &mut v_dev)?;
    bindings.set(sequence_length_query, &mut seq_len_q_dev)?;
    bindings.set(sequence_length_key_value, &mut seq_len_kv_dev)?;
    bindings.set(o, &mut o_dev)?;
    bindings.set(stats, &mut stats_dev)?;
    if let Err(error) = compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace)) {
        if is_not_supported(&error) {
            println!(
                "frontend_fp16_sdpa_dynamic_shapes: skipped, not supported on current configuration"
            );
            return Ok(());
        }
        return Err(error);
    }

    let mut q_dev_2 = DeviceMemory::from_slice(&init_bf16_image(
        (override_b * h_q * s_q * d_qk) as usize,
        &mut seed,
    ))?;
    let mut k_dev_2 = DeviceMemory::from_slice(&init_bf16_image(
        (override_b * h_k * s_kv * d_qk) as usize,
        &mut seed,
    ))?;
    let mut v_dev_2 = DeviceMemory::from_slice(&init_bf16_image(
        (override_b * h_v * s_kv * d_v) as usize,
        &mut seed,
    ))?;
    let mut seq_len_q_dev_2 = DeviceMemory::from_slice(&vec![20_i32; override_b as usize])?;
    let mut seq_len_kv_dev_2 = DeviceMemory::from_slice(&vec![20_i32; override_b as usize])?;
    let mut o_dev_2 = DeviceMemory::<bf16>::zeroes((override_b * h_q * s_q * d_v) as usize)?;
    let mut stats_dev_2 = DeviceMemory::<f32>::zeroes((override_b * h_q * s_q) as usize)?;

    let mut bindings_2 = compiled.bindings();
    bindings_2.set(q, &mut q_dev_2)?;
    bindings_2.set(k, &mut k_dev_2)?;
    bindings_2.set(v, &mut v_dev_2)?;
    bindings_2.set(sequence_length_query, &mut seq_len_q_dev_2)?;
    bindings_2.set(sequence_length_key_value, &mut seq_len_kv_dev_2)?;
    bindings_2.set(o, &mut o_dev_2)?;
    bindings_2.set(stats, &mut stats_dev_2)?;

    let mut overrides = compiled.runtime_overrides();
    overrides.set_shape(
        q,
        &Shape::contiguous([override_b, h_q, s_q, d_qk])?.with_strides(vec![
            h_q * s_q * d_qk,
            s_q * d_qk,
            d_qk,
            1,
        ])?,
    )?;
    overrides.set_shape(
        k,
        &Shape::contiguous([override_b, h_k, s_kv, d_qk])?.with_strides(vec![
            h_k * s_kv * d_qk,
            s_kv * d_qk,
            d_qk,
            1,
        ])?,
    )?;
    overrides.set_shape(
        v,
        &Shape::contiguous([override_b, h_v, s_kv, d_v])?.with_strides(vec![
            h_v * s_kv * d_v,
            s_kv * d_v,
            d_v,
            1,
        ])?,
    )?;
    overrides.set_shape(
        o,
        &Shape::contiguous([override_b, h_q, s_q, d_v])?.with_strides(vec![
            h_q * d_v,
            d_v,
            override_b * h_q * d_v,
            1,
        ])?,
    )?;
    overrides.set_shape(
        stats,
        &Shape::contiguous([override_b, h_q, s_q, 1])?.with_strides([h_q * s_q, s_q, 1, 1])?,
    )?;
    overrides.set_shape(
        sequence_length_query,
        &Shape::contiguous([override_b, 1, 1, 1])?.with_strides([1, 1, 1, 1])?,
    )?;
    overrides.set_shape(
        sequence_length_key_value,
        &Shape::contiguous([override_b, 1, 1, 1])?.with_strides([1, 1, 1, 1])?,
    )?;
    let overrides = overrides.build();
    if let Err(error) = compiled.execute_with_runtime_overrides(
        &ctx.cudnn,
        &bindings_2,
        Some(&mut workspace),
        &overrides,
    ) {
        if is_not_supported(&error) {
            println!(
                "frontend_fp16_sdpa_dynamic_shapes: skipped, not supported on current configuration"
            );
            return Ok(());
        }
        return Err(error);
    }

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_fp16_sdpa_dynamic_shapes", run())
}
