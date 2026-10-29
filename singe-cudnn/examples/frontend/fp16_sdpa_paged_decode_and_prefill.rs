mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, bf16},
    error::{Error, Result},
    frontend::{
        composite::sdpa::{SdpaForwardOutputTensors, SdpaInputs},
        error::Error as FrontendError,
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::{
            AttentionConfig, AttentionDiagonalAlignment, AttentionDiagonalBand, AttentionMaskMode,
            AttentionMaxTotalSequenceLengths, AttentionPagedCache, AttentionPagedKvCache,
            AttentionScoreConfig, AttentionSequenceInputs,
        },
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
};

#[derive(Debug, Clone, Copy)]
enum Phase {
    Prefill,
    Decode,
}

fn dimension_overflow() -> Error {
    Error::OutOfRange {
        name: "paged attention dimensions".into(),
    }
}

fn index_value(value: i64) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::OutOfRange {
        name: "paged attention int32 metadata".into(),
    })
}

fn run_phase(example: &common::ExampleContext, phase: Phase, block_size: i64) -> Result<()> {
    let is_ragged_prefill = matches!(phase, Phase::Prefill);
    let [b, h_q, h_k, h_v, s_kv, d_qk, d_v] = [8_i64, 8, 1, 1, 32 * 1024, 128, 128];
    let s_q = if is_ragged_prefill { 512 } else { 1 };
    let table_size = s_kv / block_size + i64::from(s_kv % block_size != 0);
    let num_blocks = table_size.checked_mul(b).ok_or_else(dimension_overflow)?;
    let query_lengths = (0..b)
        .map(|index| {
            if is_ragged_prefill {
                if index & 1 == 0 { 32_i32 } else { 16_i32 }
            } else {
                1_i32
            }
        })
        .collect::<Vec<_>>();
    let kv_lengths = (0..b)
        .map(|index| index_value((s_kv - index * 17).max(1)))
        .collect::<Result<Vec<_>>>()?;
    let page_indices_k = (0..num_blocks)
        .map(index_value)
        .collect::<Result<Vec<_>>>()?;
    let page_indices_v = (0..num_blocks)
        .map(|index| index_value((index * 13) % num_blocks))
        .collect::<Result<Vec<_>>>()?;

    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::BF16)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );
    let mut query_spec =
        TensorSpec::new(DataType::BF16, Shape::contiguous([b, h_q, s_q, d_qk])?).with_name("Q");
    if is_ragged_prefill {
        let capacity = query_spec.storage_byte_len(None)?;
        query_spec = query_spec.with_ragged_byte_capacity(capacity);
    }
    let q = graph.tensor(query_spec);
    let q_offsets = if is_ragged_prefill {
        let id = graph.tensor(
            TensorSpec::new(DataType::I32, Shape::contiguous([b + 1, 1, 1, 1])?)
                .with_name("query_offsets"),
        );
        graph.set_ragged_offset(q, id)?;
        Some(id)
    } else {
        None
    };
    let k = graph.tensor(
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([num_blocks, h_k, block_size, d_qk])?,
        )
        .with_name("container_K"),
    );
    let v = graph.tensor(
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([num_blocks, h_v, block_size, d_v])?,
        )
        .with_name("container_V"),
    );
    let scale = graph.tensor(TensorSpec::scalar_f32(0.123));
    let sequence_length_query = graph.tensor(
        TensorSpec::new(DataType::I32, Shape::contiguous([b, 1, 1, 1])?).with_name("seq_q"),
    );
    let sequence_length_key_value = graph.tensor_like(sequence_length_query, Some("seq_kv"))?;
    let page_table_k = graph.tensor(
        TensorSpec::new(DataType::I32, Shape::contiguous([b, 1, table_size, 1])?)
            .with_name("page_table_k"),
    );
    let page_table_v = graph.tensor_like(page_table_k, Some("page_table_v"))?;
    let max_total_query = if is_ragged_prefill {
        Some(b.checked_mul(s_q).ok_or_else(dimension_overflow)?)
    } else {
        None
    };
    let outputs = graph.sdpa_auto(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale,
        },
        AttentionConfig::new()
            .with_name("flash_attention")
            .with_compute_type(DataType::F32)
            .with_max_total_sequence_lengths(AttentionMaxTotalSequenceLengths {
                query: max_total_query,
                key_value: None,
            })
            .with_score_config(
                AttentionScoreConfig::new()
                    .with_sequence_inputs(AttentionSequenceInputs::per_batch(
                        sequence_length_query,
                        sequence_length_key_value,
                    ))
                    .with_mask_mode(AttentionMaskMode::DiagonalBand(
                        AttentionDiagonalBand::causal(AttentionDiagonalAlignment::TopLeft)
                            .with_padding(true),
                    )),
            )
            .with_paged_cache(
                AttentionPagedKvCache::new()
                    .with_k(AttentionPagedCache::new(page_table_k))
                    .with_v(AttentionPagedCache::new(page_table_v))
                    .with_max_sequence_length_key_value(s_kv),
            ),
        SdpaForwardOutputTensors::new(TensorOutputConfig::new().output_tensor())
            .with_stats(TensorOutputConfig::new().output_tensor()),
    )?;
    let o = outputs.output;
    let stats = outputs.stats.ok_or_else(|| {
        FrontendError::Compile("requested paged attention statistics are missing".into())
    })?;
    // The output is dense physical [S, B, H, D], independently of the packed query.
    let output_storage = Shape::contiguous([s_q, b, h_q, d_v])?;
    let strides = output_storage.strides();
    graph.replace_tensor(
        o,
        TensorSpec::new(
            DataType::BF16,
            Shape::with_strides(
                [b, h_q, s_q, d_v],
                [strides[1], strides[2], strides[0], strides[3]],
            )?,
        )
        .with_name("O")
        .output_tensor(),
    )?;
    graph.mark_as_output(stats)?;

    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A, HeuristicMode::Fallback]),
    )?;
    let cuda = example.cudnn.cuda_context();
    let mut seed = 123_456_789;
    let query_values = common::init_f32_image(graph.shape(q)?.storage_span()?, &mut seed)
        .into_iter()
        .map(bf16::from_f32)
        .collect::<Vec<_>>();
    let q_dev = DeviceMemory::from_slice(cuda, &query_values)?;
    let k_dev = DeviceMemory::from_slice(
        cuda,
        &vec![bf16::from_f32(0.25); graph.shape(k)?.storage_span()?],
    )?;
    let v_dev = DeviceMemory::from_slice(
        cuda,
        &vec![bf16::from_f32(0.125); graph.shape(v)?.storage_span()?],
    )?;
    let mut o_dev = DeviceMemory::<bf16>::zeroed(cuda, graph.shape(o)?.storage_span()?)?;
    let seq_len_q_dev = DeviceMemory::from_slice(cuda, &query_lengths)?;
    let seq_len_kv_dev = DeviceMemory::from_slice(cuda, &kv_lengths)?;
    let page_table_k_dev = DeviceMemory::from_slice(cuda, &page_indices_k)?;
    let page_table_v_dev = DeviceMemory::from_slice(cuda, &page_indices_v)?;
    let mut stats_dev = DeviceMemory::<f32>::zeroed(cuda, graph.shape(stats)?.storage_span()?)?;
    let q_offsets_dev = if let Some(id) = q_offsets {
        let token_elements = h_q.checked_mul(d_qk).ok_or_else(dimension_overflow)?;
        let mut offsets = vec![0_i32];
        let mut cumulative = 0_i64;
        for &length in &query_lengths {
            let elements = i64::from(length)
                .checked_mul(token_elements)
                .ok_or_else(dimension_overflow)?;
            cumulative = cumulative
                .checked_add(elements)
                .ok_or_else(dimension_overflow)?;
            offsets.push(index_value(cumulative)?);
        }
        Some((id, DeviceMemory::from_slice(cuda, &offsets)?))
    } else {
        None
    };
    let mut workspace = common::workspace(cuda, compiled.workspace_size())?;
    let mut bindings = compiled.bindings()?;
    bindings.set_input(q, &q_dev)?;
    bindings.set_input(k, &k_dev)?;
    bindings.set_input(v, &v_dev)?;
    bindings.set_input(sequence_length_query, &seq_len_q_dev)?;
    bindings.set_input(sequence_length_key_value, &seq_len_kv_dev)?;
    bindings.set_input(page_table_k, &page_table_k_dev)?;
    bindings.set_input(page_table_v, &page_table_v_dev)?;
    if let Some((id, memory)) = &q_offsets_dev {
        bindings.set_input(*id, memory)?;
    }
    bindings.set(o, &mut o_dev)?;
    bindings.set(stats, &mut stats_dev)?;
    // SAFETY: initialized cache/index inputs, writable results and workspace remain live through stream completion.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;
    println!("executed paged attention {phase:?}, block size: {block_size}");
    Ok(())
}

fn run() -> Result<()> {
    let block_size = i64::try_from(common::positive_usize_argument("--block-size", 1)?.get())
        .map_err(|_| dimension_overflow())?;
    let context = common::ExampleContext::create()?;
    run_phase(&context, Phase::Prefill, block_size)?;
    run_phase(&context, Phase::Decode, block_size)
}

fn main() -> Result<()> {
    common::finish("frontend_fp16_sdpa_paged_decode_and_prefill", run())
}
