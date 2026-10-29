mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorId, TensorSpec},
    data_type::{DataType, bf16},
    error::Result,
    frontend::{
        composite::sdpa::{SdpaForwardOutputTensors, SdpaInputs},
        error::Error as FrontendError,
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::{
            AttentionConfig, AttentionDiagonalAlignment, AttentionDiagonalBand,
            AttentionDropoutMask, AttentionDropoutMode, AttentionMaskMode, AttentionScoreConfig,
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
    let b = 3_i64;
    let h_q = 4_i64;
    let h_k = 4_i64;
    let h_v = 4_i64;
    let s_q = 1024_i64;
    let s_kv = 1024_i64;
    let d_qk = 128_i64;
    let d_v = 128_i64;
    let dropout_scale_value = 0.1_f32;
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
    let scale = graph.tensor(TensorSpec::scalar_f32(0.123).with_byte_alignment(64)?);
    let dropout_mask = graph.insert_tensor(
        TensorId::new(1008),
        TensorSpec::new(DataType::BF16, Shape::contiguous([b, h_q, s_q, s_kv])?),
    )?;
    let dropout_scale = graph.insert_tensor(
        TensorId::new(1007),
        TensorSpec::scalar_f32(dropout_scale_value).with_byte_alignment(128)?,
    )?;
    let bias = graph.insert_tensor(
        TensorId::new(1006),
        TensorSpec::new(DataType::F32, Shape::contiguous([b, 1, s_q, s_kv])?),
    )?;
    let score_config = AttentionScoreConfig::new()
        .with_bias(bias)
        .with_mask_mode(AttentionMaskMode::DiagonalBand(
            AttentionDiagonalBand::causal(AttentionDiagonalAlignment::TopLeft),
        ))
        .with_dropout_mode(AttentionDropoutMode::CustomMask(AttentionDropoutMask::new(
            dropout_mask,
            dropout_scale,
        )));
    let config = AttentionConfig::new().with_score_config(score_config);
    let outputs = graph.sdpa_auto(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale,
        },
        config,
        SdpaForwardOutputTensors::new(TensorOutputConfig::new().output_tensor())
            .with_stats(TensorOutputConfig::new().output_tensor()),
    )?;
    let o = outputs.output;
    // Store logical [B, H, S, D] output in physical [S, B, H, D] order.
    let physical_output = Shape::contiguous([s_q, b, h_q, d_v])?;
    let strides = physical_output.strides();
    let output_shape = Shape::with_strides(
        [b, h_q, s_q, d_v],
        [strides[1], strides[2], strides[0], strides[3]],
    )?;
    graph.modify_tensor(o, |tensor| tensor.with_shape(output_shape).output_tensor())?;
    let stats = outputs.stats.ok_or_else(|| {
        FrontendError::Compile("requested attention statistics are missing".into())
    })?;
    graph.mark_as_output(stats)?;

    let q_elements = graph.shape(q)?.storage_span()?;
    let k_elements = graph.shape(k)?.storage_span()?;
    let v_elements = graph.shape(v)?.storage_span()?;
    let mask_elements = graph.shape(dropout_mask)?.storage_span()?;
    let bias_elements = graph.shape(bias)?.storage_span()?;
    let output_elements = graph.shape(o)?.storage_span()?;
    let stats_elements = graph.shape(stats)?.storage_span()?;
    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
    )?;
    let cuda = example.cudnn.cuda_context();
    let mut seed = 123_456_789_u32;
    let q_dev = DeviceMemory::from_slice(cuda, &init_bf16_image(q_elements, &mut seed))?;
    let k_dev = DeviceMemory::from_slice(cuda, &init_bf16_image(k_elements, &mut seed))?;
    let v_dev = DeviceMemory::from_slice(cuda, &init_bf16_image(v_elements, &mut seed))?;
    let dropout_mask_dev =
        DeviceMemory::from_slice(cuda, &init_bf16_image(mask_elements, &mut seed))?;
    let bias_dev =
        DeviceMemory::from_slice(cuda, &common::init_f32_image(bias_elements, &mut seed))?;
    let mut o_dev = DeviceMemory::<bf16>::zeroed(cuda, output_elements)?;
    let mut stats_dev = DeviceMemory::<f32>::zeroed(cuda, stats_elements)?;
    let mut workspace = common::workspace(cuda, compiled.workspace_size())?;
    let mut bindings = compiled.bindings()?;
    bindings.set_input(q, &q_dev.view())?;
    bindings.set_input(k, &k_dev.view())?;
    bindings.set_input(v, &v_dev.view())?;
    bindings.set_input(dropout_mask, &dropout_mask_dev.view())?;
    bindings.set_input(bias, &bias_dev.view())?;
    bindings.set_scalar(dropout_scale, dropout_scale_value)?;
    bindings.set(o, &mut o_dev)?;
    bindings.set(stats, &mut stats_dev)?;
    // SAFETY: initialized inputs, host scalars, writable outputs and workspace remain live through completion.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_fp16_sdpa_custom_dropout", run())
}
