mod common;

use std::{
    env::args,
    sync::{Arc, Mutex},
};

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
            AttentionConfig, AttentionDiagonalAlignment, AttentionDiagonalBand, AttentionMaskMode,
            AttentionScoreConfig, AttentionSequenceInputs,
        },
        plan::{BuildPlanPolicy, CompileConfig, ExecuteArgs, HeuristicMode, PlanCache},
    },
};

fn init_bf16_image(size: usize, seed: &mut u32) -> Vec<bf16> {
    common::init_f32_image(size, seed)
        .into_iter()
        .map(bf16::from_f32)
        .collect()
}

struct SdpaGraph {
    graph: Graph,
    q: TensorId,
    k: TensorId,
    v: TensorId,
    scale: TensorId,
    sequence_length_query: TensorId,
    sequence_length_key_value: TensorId,
    o: TensorId,
    stats: TensorId,
}

fn create_sdpa_forward_graph() -> Result<SdpaGraph> {
    let b = 3_i64;
    let h = 4_i64;
    let s_q = 1024_i64;
    let s_kv = 1024_i64;
    let d_qk = 128_i64;
    let d_v = 128_i64;

    let mut graph = Graph::with_config(
        GraphConfig::new()
            .with_name("cached-sdpa-forward")
            .with_data_type_policy(
                DataTypePolicy::new()
                    .with_io(DataType::BF16)
                    .with_intermediate(DataType::F32)
                    .with_compute(DataType::F32),
            ),
    );

    let q = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([b, h, s_q, d_qk])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([b, h, s_kv, d_qk])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([b, h, s_kv, d_v])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.123));
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
        AttentionConfig::new().with_score_config(
            AttentionScoreConfig::new()
                .with_sequence_inputs(AttentionSequenceInputs::per_batch(
                    sequence_length_query,
                    sequence_length_key_value,
                ))
                .with_mask_mode(AttentionMaskMode::DiagonalBand(
                    AttentionDiagonalBand::causal(AttentionDiagonalAlignment::TopLeft)
                        .with_padding(true),
                )),
        ),
        SdpaForwardOutputTensors::new(TensorOutputConfig::new().output_tensor())
            .with_stats(TensorOutputConfig::new().output_tensor()),
    )?;
    let o = outputs.output;
    let sequence_major = Shape::contiguous([s_q, b, h, d_v])?;
    let strides = sequence_major.strides();
    graph.replace_tensor(
        o,
        TensorSpec::new(
            DataType::BF16,
            Shape::with_strides(
                [b, h, s_q, d_v],
                [strides[1], strides[2], strides[0], strides[3]],
            )?,
        )
        .output_tensor(),
    )?;
    let stats = outputs.stats.ok_or(FrontendError::OperationMissingOutput)?;

    Ok(SdpaGraph {
        graph,
        q,
        k,
        v,
        scale,
        sequence_length_query,
        sequence_length_key_value,
        o,
        stats,
    })
}

fn run() -> Result<()> {
    let example = common::ExampleContext::create()?;
    let cache_last_plan = args().any(|argument| argument == "--cache-last-plan");
    let cache = Arc::new(Mutex::new(PlanCache::new()));
    let config = CompileConfig::new()
        .with_heuristic_modes(vec![HeuristicMode::A, HeuristicMode::Fallback])
        .with_build_policy(if cache_last_plan {
            BuildPlanPolicy::AllSupported
        } else {
            BuildPlanPolicy::FirstSupported
        });
    let mut first_graph = create_sdpa_forward_graph()?;
    first_graph.graph.attach_plan_cache(Arc::clone(&cache));
    let first_key = first_graph
        .graph
        .plan_cache_key(&config, Some(&example.cudnn))?;
    let mut first = first_graph.graph.compile(Some(&example.cudnn), &config)?;
    if cache_last_plan {
        if let Some(plan) = first.plan_ids().pop() {
            first.set_default_plan(&plan)?;
        }
        cache
            .lock()
            .map_err(|_| FrontendError::PlanCacheLockPoisoned)?
            .insert(first_key.clone(), first.cached_plan_choices());
    }
    println!(
        "preferred engine: {:?}",
        first.engine_metadata().engine_index
    );

    // A fresh graph has different tensor IDs. Only its ordered topology and compilation attributes matter.
    let SdpaGraph {
        mut graph,
        q,
        k,
        v,
        scale,
        sequence_length_query,
        sequence_length_key_value,
        o,
        stats,
    } = create_sdpa_forward_graph()?;
    graph.attach_plan_cache(Arc::clone(&cache));

    // Diagnostic names and runtime defaults do not change the engine-choice cache key.
    // Fresh descriptors and bindings still use this graph's current scalar default.
    graph.set_name("renamed-cached-sdpa-forward");
    let query_spec = graph.tensor_spec(q)?.clone().with_name("query-buffer");
    graph.replace_tensor(q, query_spec)?;
    graph.replace_tensor(scale, TensorSpec::scalar_f32(0.25))?;
    let second_key = graph.plan_cache_key(&config, Some(&example.cudnn))?;
    let compiled = graph.compile(Some(&example.cudnn), &config)?;
    println!(
        "rebuilt engine: {:?}",
        compiled.engine_metadata().engine_index
    );
    drop(first);
    let cached_graphs = cache
        .lock()
        .map_err(|_| FrontendError::PlanCacheLockPoisoned)?
        .len();
    println!("first key={first_key}, second key={second_key}, cached graphs={cached_graphs}");

    let mut seed = 123_456_789_u32;
    let q_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_bf16_image(
            compiled.tensor_info(q)?.spec.shape.storage_span()?,
            &mut seed,
        ),
    )?;
    let k_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_bf16_image(
            compiled.tensor_info(k)?.spec.shape.storage_span()?,
            &mut seed,
        ),
    )?;
    let v_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_bf16_image(
            compiled.tensor_info(v)?.spec.shape.storage_span()?,
            &mut seed,
        ),
    )?;
    let seq_len_q_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &vec![
            20_i32;
            compiled
                .tensor_info(sequence_length_query)?
                .spec
                .shape
                .storage_span()?
        ],
    )?;
    let seq_len_kv_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &vec![
            20_i32;
            compiled
                .tensor_info(sequence_length_key_value)?
                .spec
                .shape
                .storage_span()?
        ],
    )?;
    let mut o_dev = DeviceMemory::<bf16>::zeroed(
        example.cudnn.cuda_context(),
        compiled.tensor_info(o)?.spec.shape.storage_span()?,
    )?;
    let mut stats_dev = DeviceMemory::<f32>::zeroed(
        example.cudnn.cuda_context(),
        compiled.tensor_info(stats)?.spec.shape.storage_span()?,
    )?;
    let mut workspace = common::workspace(example.cudnn.cuda_context(), compiled.workspace_size())?;

    let mut bindings = compiled.bindings()?;
    bindings.set_input(q, &q_dev)?;
    bindings.set_input(k, &k_dev)?;
    bindings.set_input(v, &v_dev)?;
    bindings.set_input(sequence_length_query, &seq_len_q_dev)?;
    bindings.set_input(sequence_length_key_value, &seq_len_kv_dev)?;
    bindings.set(o, &mut o_dev)?;
    bindings.set(stats, &mut stats_dev)?;

    for _ in 0..2 {
        // SAFETY: inputs and sequence metadata are initialized, and all bindings cover the retained layouts.
        // Plans, scalars, outputs and workspace remain live through completion, including enqueue failure.
        example.execute_and_wait(|context| unsafe {
            compiled.execute(
                context,
                ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
            )
        })?;
    }

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_fp16_sdpa_cached", run())
}
