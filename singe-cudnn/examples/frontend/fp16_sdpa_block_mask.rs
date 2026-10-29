mod common;

use std::env::args;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorId, TensorSpec},
    data_type::{DataType, bf16},
    error::Result,
    frontend::{
        composite::sdpa::{SdpaForwardOutputTensors, SdpaInputs},
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::{
            AttentionConfig, AttentionImplementation, AttentionMaskMode, AttentionScoreConfig,
        },
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
};

fn init_bf16_image(size: usize, seed: &mut u32) -> Vec<bf16> {
    common::init_f32_image(size, seed)
        .into_iter()
        .map(bf16::from_f32)
        .collect()
}

fn run() -> Result<()> {
    let example = common::ExampleContext::create()?;

    let compute_type = if args().any(|argument| argument == "--fp16-compute") {
        DataType::F16
    } else {
        DataType::F32
    };
    let implementation = if args().any(|argument| argument == "--unified") {
        AttentionImplementation::Unified
    } else {
        AttentionImplementation::Auto
    };

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

    let q = graph.insert_tensor(
        TensorId::new(1001),
        TensorSpec::new(DataType::BF16, Shape::contiguous([b, h_q, s_q, d_qk])?),
    )?;
    let k = graph.insert_tensor(
        TensorId::new(1002),
        TensorSpec::new(DataType::BF16, Shape::contiguous([b, h_k, s_kv, d_qk])?),
    )?;
    let v = graph.insert_tensor(
        TensorId::new(1003),
        TensorSpec::new(DataType::BF16, Shape::contiguous([b, h_v, s_kv, d_v])?),
    )?;
    let scale = graph.tensor(TensorSpec::scalar_f32(0.123));

    let block_rows = (s_q + tile_m - 1) / tile_m;
    let block_cols = ((s_kv + tile_n - 1) / tile_n + 7) / 8;
    let block_mask = graph.insert_tensor(
        TensorId::new(1009),
        TensorSpec::new(
            DataType::U8,
            Shape::with_strides(
                [b, h_q, block_rows, block_cols],
                vec![
                    h_q * block_rows * block_cols,
                    block_rows * block_cols,
                    block_cols,
                    1,
                ],
            )?,
        ),
    )?;

    let outputs = graph.sdpa_auto(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale,
        },
        AttentionConfig::new()
            .with_compute_type(compute_type)
            .with_implementation(implementation)
            .with_score_config(
                AttentionScoreConfig::new().with_mask_mode(AttentionMaskMode::Block(block_mask)),
            ),
        SdpaForwardOutputTensors::new(TensorOutputConfig::new().output_tensor()),
    )?;
    let o = outputs.output;
    let stats = outputs.stats;

    graph.replace_tensor(
        o,
        TensorSpec::new(
            DataType::BF16,
            Shape::with_strides([b, h_q, s_q, d_v], vec![h_q * d_v, d_v, b * h_q * d_v, 1])?,
        )
        .output_tensor(),
    )?;
    if let Some(stats) = stats {
        let tensor = graph.tensor_spec(stats)?.clone().output_tensor();
        graph.replace_tensor(stats, tensor)?;
    }

    let q_elements = graph.shape(q)?.storage_span()?;
    let k_elements = graph.shape(k)?.storage_span()?;
    let v_elements = graph.shape(v)?.storage_span()?;
    let output_elements = graph.shape(o)?.storage_span()?;
    let mask_elements = graph.shape(block_mask)?.storage_span()?;
    let stats_elements = stats
        .map(|id| graph.shape(id)?.storage_span())
        .transpose()?;
    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
    )?;

    let mut seed = 987_654_321_u32;
    let q_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_bf16_image(q_elements, &mut seed),
    )?;
    let k_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_bf16_image(k_elements, &mut seed),
    )?;
    let v_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_bf16_image(v_elements, &mut seed),
    )?;
    let mut o_dev = DeviceMemory::<bf16>::zeroed(example.cudnn.cuda_context(), output_elements)?;
    let block_mask_dev = DeviceMemory::<u8>::zeroed(example.cudnn.cuda_context(), mask_elements)?;
    let mut stats_dev = stats_elements
        .map(|elements| DeviceMemory::<f32>::zeroed(example.cudnn.cuda_context(), elements))
        .transpose()?;
    let mut workspace = common::workspace(example.cudnn.cuda_context(), compiled.workspace_size())?;

    let mut bindings = compiled.bindings()?;
    bindings.set_input(q, &q_dev.view())?;
    bindings.set_input(k, &k_dev.view())?;
    bindings.set_input(v, &v_dev.view())?;
    bindings.set_input(block_mask, &block_mask_dev.view())?;
    bindings.set(o, &mut o_dev)?;
    if let Some(stats) = stats
        && let Some(stats_dev) = &mut stats_dev
    {
        bindings.set(stats, stats_dev)?;
    }

    // SAFETY: initialized inputs, writable outputs, native plan and workspace remain live through completion.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_fp16_sdpa_block_mask", run())
}
