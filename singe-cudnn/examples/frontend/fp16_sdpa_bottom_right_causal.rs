mod common;

use std::env::args;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, f16},
    error::Result,
    frontend::{
        composite::sdpa::{SdpaForwardOutputTensors, SdpaInputs},
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::{
            AttentionConfig, AttentionDiagonalAlignment, AttentionDiagonalBand,
            AttentionImplementation, AttentionMaskMode, AttentionScoreConfig,
            AttentionSequenceInputs, SoftmaxStatsBase,
        },
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
};

fn init_f16_image(size: usize, seed: &mut u32) -> Vec<f16> {
    common::init_f32_image(size, seed)
        .into_iter()
        .map(f16::from_f32)
        .collect()
}

fn run() -> Result<()> {
    let example = common::ExampleContext::create()?;
    let inference = args().any(|argument| argument == "--inference");
    let left_only = args().any(|argument| argument == "--left-only");
    let right_only = args().any(|argument| argument == "--right-only");
    let window = args().any(|argument| argument == "--window");
    let wide_window = args().any(|argument| argument == "--wide-window");
    let padding = left_only || args().any(|argument| argument == "--padding");
    let band = if left_only {
        AttentionDiagonalBand::new(AttentionDiagonalAlignment::BottomRight).with_left_window(32)
    } else if right_only {
        AttentionDiagonalBand::new(AttentionDiagonalAlignment::BottomRight).with_right_window(8)
    } else if wide_window {
        AttentionDiagonalBand::new(AttentionDiagonalAlignment::BottomRight)
            .with_left_window(i64::MAX)
            .with_right_window(i64::MAX)
    } else if window {
        AttentionDiagonalBand::new(AttentionDiagonalAlignment::BottomRight)
            .with_left_window(32)
            .with_right_window(8)
    } else {
        AttentionDiagonalBand::causal(AttentionDiagonalAlignment::BottomRight)
    }
    .with_padding(padding);
    let implementation = if args().any(|argument| argument == "--unified") {
        AttentionImplementation::Unified
    } else {
        AttentionImplementation::Composite
    };

    let b = 2_i64;
    let h = 2_i64;
    let s_q = 64_i64;
    let s_kv = 128_i64;
    let d_qk = 64_i64;
    let d_v = 64_i64;

    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F16)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );

    let query_spec = TensorSpec::new(DataType::F16, Shape::contiguous([b, h, s_q, d_qk])?);
    // Public K retains logical sequence/hidden axes in rectangular cross-attention.
    let key_spec = TensorSpec::new(DataType::F16, Shape::contiguous([b, h, s_kv, d_qk])?);
    let value_spec = TensorSpec::new(DataType::F16, Shape::contiguous([b, h, s_kv, d_v])?);
    let query_elements = query_spec.logical_element_count()?;
    let key_elements = key_spec.logical_element_count()?;
    let value_elements = value_spec.logical_element_count()?;
    let q = graph.tensor(query_spec);
    let k = graph.tensor(key_spec);
    let v = graph.tensor(value_spec);
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125));
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([b, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([b, 1, 1, 1])?,
    ));

    let outputs = graph.sdpa_auto(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale,
        },
        AttentionConfig::new()
            .with_implementation(implementation)
            .with_stats_base(if args().any(|argument| argument == "--log2-stats") {
                SoftmaxStatsBase::Log2
            } else {
                SoftmaxStatsBase::NaturalLog
            })
            .with_score_config(
                AttentionScoreConfig::new()
                    .with_sequence_inputs(AttentionSequenceInputs::per_batch(
                        sequence_length_query,
                        sequence_length_key_value,
                    ))
                    .with_mask_mode(AttentionMaskMode::DiagonalBand(band)),
            ),
        SdpaForwardOutputTensors::new(TensorOutputConfig::new().output_tensor())
            .with_stats((!inference).then_some(TensorOutputConfig::new().output_tensor())),
    )?;
    let o = outputs.output;
    let stats = outputs.stats;
    let output_elements = graph.tensor_spec(o)?.logical_element_count()?;
    let batch_elements = graph
        .tensor_spec(sequence_length_query)?
        .logical_element_count()?;

    for output in [Some(o), stats].into_iter().flatten() {
        graph.mark_as_output(output)?;
    }

    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
    )?;
    let mut seed = 123_456_789_u32;
    let q_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_f16_image(query_elements, &mut seed),
    )?;
    let k_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_f16_image(key_elements, &mut seed),
    )?;
    let v_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_f16_image(value_elements, &mut seed),
    )?;
    let seq_len_q_dev =
        DeviceMemory::from_slice(example.cudnn.cuda_context(), &vec![48_i32; batch_elements])?;
    let seq_len_kv_dev =
        DeviceMemory::from_slice(example.cudnn.cuda_context(), &vec![96_i32; batch_elements])?;
    let mut o_dev = DeviceMemory::<f16>::zeroed(example.cudnn.cuda_context(), output_elements)?;
    let mut stats_dev = stats
        .map(|stats| -> Result<_> {
            Ok(DeviceMemory::<f32>::zeroed(
                example.cudnn.cuda_context(),
                graph.tensor_spec(stats)?.logical_element_count()?,
            )?)
        })
        .transpose()?;
    let mut workspace = common::workspace(example.cudnn.cuda_context(), compiled.workspace_size())?;
    let mut bindings = compiled.bindings()?;
    bindings
        .set_input(q, &q_dev)?
        .set_input(k, &k_dev)?
        .set_input(v, &v_dev)?
        .set_input(sequence_length_query, &seq_len_q_dev)?
        .set_input(sequence_length_key_value, &seq_len_kv_dev)?
        .set(o, &mut o_dev)?;
    if let Some((stats, memory)) = stats.zip(stats_dev.as_mut()) {
        bindings.set(stats, memory)?;
    }
    // SAFETY: initialized inputs and exclusive output buffers match the native descriptors,
    // and all owners and the plan remain live until stream completion, including enqueue failures.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_fp16_sdpa_bottom_right_causal", run())
}
