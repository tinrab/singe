mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    cudart_version,
    data_type::{DataType, bf16},
    error::{Error, Result, Status},
    frontend::{
        composite::sdpa::SdpaInputs,
        graph::{DataTypePolicy, Graph, GraphConfig},
        operation::{AttentionConfig, AttentionMaskMode, AttentionScoreConfig, HeuristicMode},
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

    if cudart_version()? < 12_000 {
        println!("frontend_fp16_sdpa_block_mask: skipped, requires CUDA toolkit 12.0+");
        return Ok(());
    }
    if version()? < 91_400 {
        println!("frontend_fp16_sdpa_block_mask: skipped, requires cuDNN 9.14.0+");
        return Ok(());
    }

    let properties = ctx.cudnn.cuda_context().device().properties()?;
    let sm = properties.major * 10 + properties.minor;
    if !(100..=120).contains(&sm) {
        println!("frontend_fp16_sdpa_block_mask: skipped, requires Blackwell Computing GPU");
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
    let tile_m = 128_i64;
    let tile_n = 128_i64;

    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::BF16)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );

    let q = graph.tensor(
        TensorSpec::new(DataType::BF16, Shape::contiguous([b, h_q, s_q, d_qk])?).with_id(1001),
    );
    let k = graph.tensor(
        TensorSpec::new(DataType::BF16, Shape::contiguous([b, h_k, s_kv, d_qk])?).with_id(1002),
    );
    let v = graph.tensor(
        TensorSpec::new(DataType::BF16, Shape::contiguous([b, h_v, s_kv, d_v])?).with_id(1003),
    );
    let scale = graph.tensor(TensorSpec::scalar_f32(0.123)?);

    let block_rows = (s_q + tile_m - 1) / tile_m;
    let block_cols = ((s_kv + tile_n - 1) / tile_n + 7) / 8;
    let block_mask = graph.tensor(
        TensorSpec::new(
            DataType::U8,
            Shape::contiguous([b, h_q, block_rows, block_cols])?.with_strides(vec![
                h_q * block_rows * block_cols,
                block_rows * block_cols,
                block_cols,
                1,
            ])?,
        )
        .with_id(1009),
    );

    let outputs = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32).with_score_config(
            AttentionScoreConfig::new().with_mask_mode(AttentionMaskMode::Block(block_mask)),
        ),
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
    if let Some(stats) = stats {
        let tensor = graph
            .tensor_config(stats)?
            .clone()
            .output_tensor()
            .with_id(1005);
        graph.replace_tensor(stats, tensor)?;
    }

    let compiled = match graph.compile(&ctx.cudnn, &[HeuristicMode::A]) {
        Ok(compiled) => compiled,
        Err(error) => {
            if matches!(
                error,
                Error::Cudnn { code, .. }
                    if code == Status::NotSupported
                        || code == Status::NotSupportedGraphPattern
                        || code == Status::NotSupportedShape
                        || code == Status::NotSupportedDataType
                        || code == Status::NotSupportedLayout
            ) {
                println!("frontend_fp16_sdpa_block_mask: skipped, block mask not supported");
                return Ok(());
            }
            return Err(error);
        }
    };

    let mut seed = 987_654_321_u32;
    let mut q_dev =
        DeviceMemory::from_slice(&init_bf16_image((b * h_q * s_q * d_qk) as usize, &mut seed))?;
    let mut k_dev = DeviceMemory::from_slice(&init_bf16_image(
        (b * h_k * s_kv * d_qk) as usize,
        &mut seed,
    ))?;
    let mut v_dev =
        DeviceMemory::from_slice(&init_bf16_image((b * h_v * s_kv * d_v) as usize, &mut seed))?;
    let mut o_dev = DeviceMemory::<bf16>::zeroes((b * h_q * s_q * d_v) as usize)?;
    let mut block_mask_dev =
        DeviceMemory::<u8>::zeroes((b * h_q * block_rows * block_cols) as usize)?;
    let mut stats_dev = DeviceMemory::<f32>::zeroes((b * h_q * s_q) as usize)?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;

    let mut bindings = compiled.bindings();
    bindings.set(q, &mut q_dev)?;
    bindings.set(k, &mut k_dev)?;
    bindings.set(v, &mut v_dev)?;
    bindings.set(block_mask, &mut block_mask_dev)?;
    bindings.set(o, &mut o_dev)?;
    if let Some(stats) = stats {
        bindings.set(stats, &mut stats_dev)?;
    }

    if let Err(error) = compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace)) {
        if matches!(
            error,
            Error::Cudnn { code, .. }
                if code == Status::NotSupported
                    || code == Status::NotSupportedGraphPattern
                    || code == Status::NotSupportedShape
                    || code == Status::NotSupportedDataType
                    || code == Status::NotSupportedLayout
        ) {
            println!("frontend_fp16_sdpa_block_mask: skipped, block mask not supported");
            return Ok(());
        }
        return Err(error);
    }
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_fp16_sdpa_block_mask", run())
}
