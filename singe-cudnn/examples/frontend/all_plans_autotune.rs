mod common;

use std::{env::args, mem::size_of, num::NonZeroUsize};

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::{
        errata::{Errata, ErrataPlan},
        execution::ExecutionPlanCache,
        tensor::{Shape, TensorSpec},
    },
    data_type::{DataType, bf16},
    error::Result,
    frontend::{
        bindings::RuntimeOverrides,
        graph::{Graph, GraphKeyShapePolicy},
        logging::{TensorDump, TensorDumpFormat, dump_tensor},
        operation::MatmulConfig,
        plan::{
            AutotuneConfig, AutotuneSampling, AutotuneSession, BuildPlanPolicy, CompileConfig,
            HeuristicMode, NativePlanSerializationConfig, NativePlanSet, PlanTiming,
        },
    },
};

fn init_f32_image(size: usize, seed: &mut u32) -> Vec<f32> {
    let mut values = Vec::with_capacity(size);
    for _ in 0..size {
        *seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        values.push((*seed as f32) * 2.328_306_4e-10_f32);
    }
    values
}

fn init_bf16_image(size: usize, seed: &mut u32) -> Vec<bf16> {
    init_f32_image(size, seed)
        .into_iter()
        .map(bf16::from_f32)
        .collect()
}

fn trimmed_mean_autotune(session: &mut AutotuneSession<'_, '_>) -> Result<Vec<PlanTiming>> {
    let mut timings = Vec::new();
    for plan in session.plans().to_vec() {
        if !session.warmup(&plan)? {
            continue;
        }
        let mut measurements = Vec::new();
        for _ in 0..5 {
            if let Some(elapsed_ms) = session.measure(&plan)? {
                measurements.push(elapsed_ms);
            }
        }
        if measurements.len() < 3 {
            continue;
        }
        measurements.sort_unstable_by(f32::total_cmp);
        let trimmed = &measurements[1..measurements.len() - 1];
        let elapsed_ms =
            trimmed.iter().map(|&value| f64::from(value)).sum::<f64>() / trimmed.len() as f64;
        timings.push(PlanTiming {
            plan,
            elapsed_ms: elapsed_ms as f32,
            samples: measurements.len(),
        });
    }
    Ok(timings)
}

fn run() -> Result<()> {
    let worker_count = common::positive_usize_argument("--workers", 8)?;
    let mut context = common::ExampleContext::create()?;

    let b = 16;
    let m = 32;
    let n = 64;
    let k = 128;

    let mut graph = Graph::new();
    let a = graph.tensor(
        TensorSpec::new(
            DataType::BF16,
            Shape::with_strides([b, m, k], [m * k, k, 1])?,
        )
        .with_name("A"),
    );
    let b_tensor = graph.tensor(
        TensorSpec::new(
            DataType::BF16,
            Shape::with_strides([b, k, n], [k * n, n, 1])?,
        )
        .with_name("B"),
    );
    let c = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::with_strides([b, m, n], [m * n, n, 1])?,
    ));
    graph.matmul_into(
        a,
        b_tensor,
        c,
        MatmulConfig::new().with_compute_type(DataType::F32),
    )?;

    let candidates = graph.plan_candidates(
        Some(&context.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
    )?;
    let errata = Errata::load_from_config(None)?.unwrap_or_default();
    let cudnn_version = singe_cudnn::version()?.raw();
    let candidates = candidates.filter_by(|metadata| {
        !errata.is_blocked(ErrataPlan::from_engine(
            "Matmul",
            metadata.engine_index,
            &metadata.knob_choices,
            cudnn_version,
        ))
    });
    let built_candidates = candidates.build_candidates(
        Some(&context.cudnn),
        &CompileConfig::new()
            .with_build_policy(BuildPlanPolicy::AllSupportedParallel { worker_count }),
    )?;
    println!("all plans: {}", built_candidates.candidate_count());

    let mut half_seed = 123_456_789_u32;
    let mut a_dev = DeviceMemory::from_slice(
        context.cudnn.cuda_context(),
        &init_bf16_image((b * m * k) as usize, &mut half_seed),
    )?;
    let mut b_dev = DeviceMemory::from_slice(
        context.cudnn.cuda_context(),
        &init_bf16_image((b * k * n) as usize, &mut half_seed),
    )?;
    let c_init = init_bf16_image((b * m * n) as usize, &mut half_seed);

    let compiled = built_candidates.compile()?;
    println!(
        "capture warmup without application buffers: {:?}",
        compiled.warmup(&mut context.cudnn)?
    );
    for binding in compiled.required_tensor_bindings() {
        println!(
            "binding {}: {:?}, initial state={}, indexing={}, operand uses={}",
            binding.id,
            binding.access,
            binding.reads_initial_state,
            binding.indexes_memory,
            binding.bindings.len(),
        );
    }

    let mut c_dev = DeviceMemory::from_slice(context.cudnn.cuda_context(), &c_init)?;
    let mut bindings = compiled.bindings()?;
    bindings.set(a, &mut a_dev)?;
    bindings.set(b_tensor, &mut b_dev)?;
    bindings.set(c, &mut c_dev)?;
    let mut workspace =
        common::workspace(context.cudnn.cuda_context(), compiled.max_workspace_size())?;
    let overrides = RuntimeOverrides::new();
    // SAFETY: all buffers, workspace and by-value scalar owners remain live through the discarded capture.
    let warmup = unsafe {
        compiled.warmup_with_bindings(
            &mut context.cudnn,
            &bindings,
            Some(&mut workspace),
            &overrides,
        )
    }?;
    println!("capture warmup: {warmup:?}");
    // SAFETY: buffers remain live; typed bindings order their tracked producers automatically.
    // Tuning uses private output storage; execute the winner below to produce caller-visible output.
    let ranking = unsafe {
        if args().any(|argument| argument == "--custom-autotune") {
            compiled.autotune_with(
                &context.cudnn,
                &bindings,
                Some(&mut workspace),
                &overrides,
                trimmed_mean_autotune,
            )
        } else {
            compiled.autotune(
                &context.cudnn,
                &bindings,
                Some(&mut workspace),
                &overrides,
                &AutotuneConfig::new()
                    .with_sampling(AutotuneSampling::MedianOfThree)
                    .with_max_plans(
                        if args().any(|argument| argument == "--first-four-measurements") {
                            NonZeroUsize::new(4)
                        } else {
                            None
                        },
                    ),
            )
        }
    }?;
    if args().any(|argument| argument == "--ranked-plans") {
        let order = ranking
            .timings
            .iter()
            .map(|timing| timing.plan.clone())
            .collect::<Vec<_>>();
        let mut ranked = ranking.winner.compiled_graph().clone();
        ranked.reorder_plans(&order)?;
        for plan in ranked.plan_ids() {
            let metadata = ranked.engine_metadata_for(&plan)?;
            println!(
                "ranked {plan}: engine {}, workspace={} bytes, notes={:?}",
                metadata.engine_index, metadata.workspace_size, metadata.behavior_notes,
            );
        }
        let artifact = ranked.serialize(NativePlanSerializationConfig {
            plans: NativePlanSet::All,
            ..NativePlanSerializationConfig::default()
        })?;
        println!("ranked native plans: {} bytes", artifact.len());
    }
    let winner = ranking.winner;
    drop(bindings);
    drop(compiled);
    let mut bindings = winner.bindings()?;
    bindings.set(a, &mut a_dev)?;
    bindings.set(b_tensor, &mut b_dev)?;
    bindings.set(c, &mut c_dev)?;
    // This cache belongs to this context and retains native plans for the exact tensor layouts/UIDs.
    let plan_cache = ExecutionPlanCache::new("matmul-native-plans");
    let cache_key = graph.key_digest(GraphKeyShapePolicy::Include)?;
    let admission = plan_cache.observe_fastest(cache_key, winner.execution_plan())?;
    if let Some(plan) = plan_cache.get(cache_key.as_slice())? {
        println!(
            "native plan cache {}: {admission:?}, retained plan {}",
            plan_cache.name(),
            plan.tag()?,
        );
    }
    // SAFETY: the retained winner fixes the measured layouts; buffers and workspace stay live through completion.
    unsafe { winner.execute(&context.cudnn, &bindings, Some(&mut workspace)) }?;

    // SAFETY: the output allocation remains live and initialized; the helper waits for its producer stream before reading.
    unsafe {
        dump_tensor(
            &context.cudnn,
            &TensorDump {
                uid: c.into(),
                name: "matmul output prefix",
                pointer: c_dev.as_device_ptr(),
                element_count: 8,
                element_bytes: size_of::<bf16>(),
                format: TensorDumpFormat::Hexadecimal,
            },
        )?;
    }
    println!("winner={}", winner.id());

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_all_plans_autotune", run())
}
