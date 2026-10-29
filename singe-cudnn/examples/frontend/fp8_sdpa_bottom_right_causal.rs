mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, f8e4m3},
    error::{Error, Result},
    frontend::{
        composite::sdpa::{SdpaForwardOutputTensors, SdpaFp8Inputs},
        error::Error as FrontendError,
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::{
            AttentionConfig, AttentionDiagonalAlignment, AttentionDiagonalBand, AttentionMaskMode,
            AttentionScoreConfig, AttentionSequenceInputs,
        },
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
};

fn interleaved_shape(b: i64, h: i64, s: i64, d: i64) -> Result<Shape> {
    let physical = Shape::contiguous([b, s, h, d])?;
    let strides = physical.strides();
    Shape::with_strides(
        [b, h, s, d],
        [strides[0], strides[2], strides[1], strides[3]],
    )
}

fn run() -> Result<()> {
    let example = common::ExampleContext::create()?;
    let b = 2_i64;
    let h = 2_i64;
    let s_q = 512_i64;
    let s_kv = 1024_i64;
    let d = 128_i64;
    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F8E4M3)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );
    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        interleaved_shape(b, h, s_q, d)?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        interleaved_shape(b, h, s_kv, d)?,
    ));
    let v = graph.tensor_like(k, Some("V"))?;
    let descale_q = graph.tensor(TensorSpec::scalar_f32(1.0));
    let descale_k = graph.tensor_like(descale_q, Some("descale_k"))?;
    let descale_v = graph.tensor_like(descale_q, Some("descale_v"))?;
    let descale_s = graph.tensor_like(descale_q, Some("descale_s"))?;
    let scale_s = graph.tensor_like(descale_q, Some("scale_s"))?;
    let scale_o = graph.tensor_like(descale_q, Some("scale_o"))?;
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(0.123));
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([b, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor_like(sequence_length_query, Some("SeqLenKV"))?;
    let outputs = graph.sdpa_fp8(
        SdpaFp8Inputs::new(
            q,
            k,
            v,
            attention_scale,
            descale_q,
            descale_k,
            descale_v,
            descale_s,
            scale_s,
            scale_o,
        ),
        &AttentionConfig::new()
            .with_name("sdpa_fp8_bottom_right_causal")
            .with_score_config(
                AttentionScoreConfig::new()
                    .with_sequence_inputs(AttentionSequenceInputs::per_batch(
                        sequence_length_query,
                        sequence_length_key_value,
                    ))
                    .with_mask_mode(AttentionMaskMode::DiagonalBand(
                        AttentionDiagonalBand::causal(AttentionDiagonalAlignment::BottomRight),
                    )),
            ),
        SdpaForwardOutputTensors::new(TensorOutputConfig::new().output_tensor())
            .with_absolute_max_probabilities(TensorOutputConfig::new().output_tensor())
            .with_absolute_max_output(TensorOutputConfig::new().output_tensor()),
    )?;
    let o = outputs.output;
    let output_shape = graph.shape(q)?.clone();
    graph.modify_tensor(o, |tensor| tensor.with_shape(output_shape).output_tensor())?;
    let absolute_max_s = outputs.absolute_max_probabilities.ok_or_else(|| {
        FrontendError::Compile("requested attention probability amax is missing".into())
    })?;
    let absolute_max_o = outputs.absolute_max_output.ok_or_else(|| {
        FrontendError::Compile("requested attention output amax is missing".into())
    })?;
    graph.mark_as_output(absolute_max_s)?;
    graph.mark_as_output(absolute_max_o)?;
    let q_elements = graph.shape(q)?.storage_span()?;
    let k_elements = graph.shape(k)?.storage_span()?;
    let v_elements = graph.shape(v)?.storage_span()?;
    let output_elements = graph.shape(o)?.storage_span()?;
    let batch_elements = graph.shape(sequence_length_query)?.element_count()?;
    let score_amax_elements = graph.shape(absolute_max_s)?.storage_span()?;
    let output_amax_elements = graph.shape(absolute_max_o)?.storage_span()?;
    let query_length = i32::try_from(s_q).map_err(|_| Error::OutOfRange {
        name: "query sequence length".into(),
    })?;
    let key_value_length = i32::try_from(s_kv).map_err(|_| Error::OutOfRange {
        name: "key/value sequence length".into(),
    })?;
    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
    )?;
    let cuda = example.cudnn.cuda_context();
    let q_dev = DeviceMemory::<f8e4m3>::zeroed(cuda, q_elements)?;
    let k_dev = DeviceMemory::<f8e4m3>::zeroed(cuda, k_elements)?;
    let v_dev = DeviceMemory::<f8e4m3>::zeroed(cuda, v_elements)?;
    let mut o_dev = DeviceMemory::<f8e4m3>::zeroed(cuda, output_elements)?;
    let seq_len_q_dev = DeviceMemory::from_slice(cuda, &vec![query_length; batch_elements])?;
    let seq_len_kv_dev = DeviceMemory::from_slice(cuda, &vec![key_value_length; batch_elements])?;
    let mut absolute_max_s_dev = DeviceMemory::<f32>::zeroed(cuda, score_amax_elements)?;
    let mut absolute_max_o_dev = DeviceMemory::<f32>::zeroed(cuda, output_amax_elements)?;
    let mut workspace = common::workspace(cuda, compiled.workspace_size())?;
    let mut bindings = compiled.bindings()?;
    bindings.set_input(q, &q_dev.view())?;
    bindings.set_input(k, &k_dev.view())?;
    bindings.set_input(v, &v_dev.view())?;
    for scalar in [descale_q, descale_k, descale_v, descale_s, scale_s, scale_o] {
        bindings.set_scalar(scalar, 1.0_f32)?;
    }
    bindings.set_input(sequence_length_query, &seq_len_q_dev.view())?;
    bindings.set_input(sequence_length_key_value, &seq_len_kv_dev.view())?;
    bindings.set(o, &mut o_dev)?;
    bindings.set(absolute_max_s, &mut absolute_max_s_dev)?;
    bindings.set(absolute_max_o, &mut absolute_max_o_dev)?;
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
    common::finish("frontend_fp8_sdpa_bottom_right_causal", run())
}
