mod common;

use std::env::args;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorId, TensorSpec},
    data_type::{DataType, bf16},
    error::{Error, Result},
    frontend::{
        composite::sdpa::{SdpaForwardOutputTensors, SdpaInputs},
        error::Error as FrontendError,
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::{
            AttentionConfig, AttentionDropoutConfig, AttentionDropoutMode, AttentionImplementation,
            AttentionMaskMode, AttentionPagedCache, AttentionPagedKvCache, AttentionScoreConfig,
            AttentionSequenceInputs, PagedCacheLayout, PagedCacheLoadConfig, PagedCacheLoadTensors,
            RandomNumberSeedSource,
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
    let save_loaded_value = args().any(|argument| argument == "--save-loaded-value");
    let explicit_cache_loads =
        save_loaded_value || args().any(|argument| argument == "--explicit-cache-loads");
    let packed_page_tables = args().any(|argument| argument == "--packed-page-tables");
    let unified = args().any(|argument| argument == "--unified");
    let rng_length = args().any(|argument| argument == "--rng-length");

    let b = 2_i64;
    let h_q = 8_i64;
    let h_k = 1_i64;
    let h_v = 1_i64;
    let s_q = 1_i64;
    // The logical sequence ends inside the final cache page.
    let s_kv = 1001_i64;
    let d_qk = 128_i64;
    let d_v = 128_i64;
    let block_size = 16_i64;
    let table_size = s_kv / block_size + i64::from(s_kv % block_size != 0);
    let sequence_lengths = [s_kv, if packed_page_tables { s_kv / 2 } else { s_kv }];
    let mut num_blocks = 0_i64;
    let mut page_offsets = vec![0_i32];
    for length in sequence_lengths {
        let pages = length / block_size + i64::from(length % block_size != 0);
        num_blocks = num_blocks.checked_add(pages).ok_or(Error::OutOfRange {
            name: "cache block count".into(),
        })?;
        page_offsets.push(i32::try_from(num_blocks).map_err(|_| Error::OutOfRange {
            name: "page table offset".into(),
        })?);
    }
    let table_elements = usize::try_from(num_blocks).map_err(|_| Error::OutOfRange {
        name: "page table storage".into(),
    })?;
    let table_bytes = table_elements
        .checked_mul(size_of::<i32>())
        .ok_or(Error::OutOfRange {
            name: "page table storage".into(),
        })?;

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
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([num_blocks, h_k, block_size, d_qk])?,
        ),
    )?;
    let v = graph.insert_tensor(
        TensorId::new(1003),
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([num_blocks, h_v, block_size, d_v])?,
        ),
    )?;
    let scale = graph.tensor(TensorSpec::scalar_f32(0.123));
    let sequence_length_query = graph.insert_tensor(
        TensorId::new(1007),
        TensorSpec::new(DataType::I32, Shape::contiguous([b])?),
    )?;
    let sequence_length_key_value = graph.insert_tensor(
        TensorId::new(1008),
        TensorSpec::new(DataType::I32, Shape::contiguous([b])?),
    )?;
    let mut page_table_spec =
        TensorSpec::new(DataType::I32, Shape::contiguous([b, 1, table_size, 1])?);
    if packed_page_tables {
        page_table_spec = page_table_spec.with_ragged_byte_capacity(table_bytes);
    }
    let page_table_k = graph.insert_tensor(TensorId::new(1009), page_table_spec.clone())?;
    let page_table_v = graph.insert_tensor(TensorId::new(1010), page_table_spec)?;
    let page_table_offsets = if packed_page_tables {
        let offsets = graph.tensor(TensorSpec::new(
            DataType::I32,
            Shape::contiguous([b + 1, 1, 1, 1])?,
        ));
        graph.set_ragged_offset(page_table_k, offsets)?;
        graph.set_ragged_offset(page_table_v, offsets)?;
        Some(offsets)
    } else {
        None
    };

    let mut config = AttentionConfig::new()
        .with_compute_type(DataType::F32)
        .with_score_config(
            AttentionScoreConfig::new()
                .with_sequence_inputs(AttentionSequenceInputs::per_batch(
                    sequence_length_query,
                    sequence_length_key_value,
                ))
                .with_mask_mode(AttentionMaskMode::Padding),
        );
    let (key, value) = if explicit_cache_loads {
        // Logical K remains [B, H, S, D], with S packed in physical memory.
        let packed = Shape::contiguous([b, h_k, d_qk, s_kv])?;
        let shape = Shape::with_strides(
            [b, h_k, s_kv, d_qk],
            [
                packed.strides()[0],
                packed.strides()[1],
                packed.strides()[3],
                packed.strides()[2],
            ],
        )?;
        let key = graph.tensor(TensorSpec::new(DataType::BF16, shape).virtual_tensor());
        graph.paged_cache_load_into(
            PagedCacheLoadTensors::new(k, key, sequence_length_key_value, page_table_k),
            PagedCacheLoadConfig::new(PagedCacheLayout::SequenceHidden)
                .with_name("key_cache")
                .with_max_sequence_length(s_kv),
        )?;
        let value_output = if save_loaded_value {
            TensorOutputConfig::new().output_tensor()
        } else {
            TensorOutputConfig::new().with_data_type(DataType::BF16)
        };
        let value = graph.paged_cache_load(
            v,
            sequence_length_key_value,
            page_table_v,
            PagedCacheLoadConfig::new(PagedCacheLayout::SequenceHidden)
                .with_name("value_cache")
                .with_max_sequence_length(s_kv),
            value_output,
        )?;
        config = config.with_implementation(AttentionImplementation::Composite);
        (key, value)
    } else {
        if unified {
            config = config.with_implementation(AttentionImplementation::Unified);
        }
        config = config.with_paged_cache(
            AttentionPagedKvCache::new()
                .with_k(AttentionPagedCache::new(page_table_k))
                .with_v(AttentionPagedCache::new(page_table_v))
                .with_max_sequence_length_key_value(
                    if rng_length || args().any(|argument| argument == "--infer-cache-length") {
                        None
                    } else {
                        Some(s_kv)
                    },
                ),
        );
        (k, v)
    };
    let inputs = SdpaInputs {
        query: q,
        key,
        value,
        scale,
    };
    let (o, stats, rng_dump) = if rng_length {
        let o = graph.tensor(
            TensorSpec::new(DataType::BF16, Shape::contiguous([b, h_q, s_q, d_v])?).output_tensor(),
        );
        let stats = graph.tensor(
            TensorSpec::new(DataType::F32, Shape::contiguous([b, h_q, s_q, 1])?).output_tensor(),
        );
        // The RNG dump retains the logical length, which is shorter than the final cache page.
        let rng_dump = graph.tensor(
            TensorSpec::new(DataType::F32, Shape::contiguous([b, h_q, s_q, s_kv])?).output_tensor(),
        );
        let score_config =
            config
                .score_config()
                .clone()
                .with_dropout_mode(AttentionDropoutMode::Philox(AttentionDropoutConfig::new(
                    0.1,
                    RandomNumberSeedSource::Host {
                        value: 1234,
                        offset: None,
                    },
                )));
        let config = config.with_score_config(score_config);
        graph.sdpa_auto_into(
            inputs,
            SdpaForwardOutputTensors {
                stats: Some(stats),
                rng_dump: Some(rng_dump),
                ..SdpaForwardOutputTensors::new(o)
            },
            &config,
        )?;
        (o, stats, Some(rng_dump))
    } else {
        let outputs = graph.sdpa_auto(
            inputs,
            config,
            SdpaForwardOutputTensors::new(TensorOutputConfig::new().output_tensor())
                .with_stats(TensorOutputConfig::new().output_tensor()),
        )?;
        let stats = outputs.stats.ok_or(FrontendError::OperationFieldMissing {
            operation: "paged sdpa",
            field: "stats",
        })?;
        (outputs.output, stats, None)
    };
    graph.replace_tensor(
        o,
        TensorSpec::new(DataType::BF16, Shape::contiguous([b, h_q, s_q, d_v])?).output_tensor(),
    )?;
    graph.mark_as_output(stats)?;

    let input_q_elements = graph.shape(q)?.storage_span()?;
    let input_k_elements = graph.shape(k)?.storage_span()?;
    let input_v_elements = graph.shape(v)?.storage_span()?;
    let output_elements = graph.shape(o)?.storage_span()?;
    let stats_elements = graph.shape(stats)?.storage_span()?;
    let rng_elements = rng_dump
        .map(|tensor| graph.shape(tensor)?.storage_span())
        .transpose()?;
    let sequence_elements = graph.shape(sequence_length_key_value)?.storage_span()?;
    let saved_value_elements = if save_loaded_value {
        Some(graph.shape(value)?.storage_span()?)
    } else {
        None
    };
    let sequence_lengths = sequence_lengths
        .into_iter()
        .map(|length| {
            i32::try_from(length).map_err(|_| Error::OutOfRange {
                name: "cache sequence length".into(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let page_table_host = (0..table_elements)
        .map(|block| {
            i32::try_from(block).map_err(|_| Error::OutOfRange {
                name: "cache block index".into(),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
    )?;
    let mut seed = 123_456_789_u32;
    let q_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_bf16_image(input_q_elements, &mut seed),
    )?;
    let k_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_bf16_image(input_k_elements, &mut seed),
    )?;
    let v_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_bf16_image(input_v_elements, &mut seed),
    )?;
    let seq_len_q_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &vec![1_i32; sequence_elements],
    )?;
    let seq_len_kv_dev = DeviceMemory::from_slice(example.cudnn.cuda_context(), &sequence_lengths)?;
    let page_table_dev = DeviceMemory::from_slice(example.cudnn.cuda_context(), &page_table_host)?;
    let page_offsets_dev = page_table_offsets
        .map(|_| DeviceMemory::from_slice(example.cudnn.cuda_context(), &page_offsets))
        .transpose()?;
    let mut o_dev = DeviceMemory::<bf16>::zeroed(example.cudnn.cuda_context(), output_elements)?;
    let mut stats_dev = DeviceMemory::<f32>::zeroed(example.cudnn.cuda_context(), stats_elements)?;
    let mut rng_dev = rng_elements
        .map(|elements| DeviceMemory::<f32>::zeroed(example.cudnn.cuda_context(), elements))
        .transpose()?;
    let mut saved_value_dev = saved_value_elements
        .map(|elements| DeviceMemory::<bf16>::zeroed(example.cudnn.cuda_context(), elements))
        .transpose()?;
    let mut workspace = common::workspace(example.cudnn.cuda_context(), compiled.workspace_size())?;

    let mut bindings = compiled.bindings()?;
    bindings.set_input(q, &q_dev.view())?;
    bindings.set_input(k, &k_dev.view())?;
    bindings.set_input(v, &v_dev.view())?;
    bindings.set_input(sequence_length_query, &seq_len_q_dev.view())?;
    bindings.set_input(sequence_length_key_value, &seq_len_kv_dev.view())?;
    bindings.set_input(page_table_k, &page_table_dev.view())?;
    bindings.set_input(page_table_v, &page_table_dev.view())?;
    if let Some(offsets) = page_table_offsets
        && let Some(offsets_dev) = &page_offsets_dev
    {
        bindings.set_input(offsets, &offsets_dev.view())?;
    }
    bindings.set(o, &mut o_dev)?;
    bindings.set(stats, &mut stats_dev)?;
    if let Some(rng_dump) = rng_dump
        && let Some(rng_dev) = &mut rng_dev
    {
        bindings.set(rng_dump, rng_dev)?;
    }
    if let Some(saved_value) = &mut saved_value_dev {
        bindings.set(value, saved_value)?;
    }
    // SAFETY: initialized inputs, writable outputs, native plans and workspace remain live through completion.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_fp16_sdpa_paged_caches", run())
}
