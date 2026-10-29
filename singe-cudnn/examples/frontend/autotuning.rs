mod common;

use std::{env::args, num::NonZeroUsize};

use singe_cuda::{
    event::{EventFlags, EventRecordFlags},
    memory::DeviceMemory,
};
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, bf16},
    error::{Error, Result},
    frontend::{
        graph::{Graph, TensorOutputConfig},
        operation::MatmulConfig,
        plan::{BuildPlanPolicy, CompileConfig, EngineConfigFilter, ExecuteArgs, HeuristicMode},
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
    let mut graph = Graph::new();
    let a = graph
        .tensor(TensorSpec::new(DataType::BF16, Shape::contiguous([16, 32, 128])?).with_name("A"));
    let b = graph
        .tensor(TensorSpec::new(DataType::BF16, Shape::contiguous([16, 128, 64])?).with_name("B"));
    let c = graph.matmul(
        a,
        b,
        MatmulConfig::new()
            .with_name("projection")
            .with_compute_type(DataType::F32),
        TensorOutputConfig::new()
            .with_data_type(DataType::BF16)
            .output_tensor(),
    )?;
    let plan_limit = if args().any(|argument| argument == "--all-builds") {
        None
    } else {
        NonZeroUsize::new(8)
    };
    let workspace_limit = if args().any(|argument| argument == "--unlimited-workspace") {
        None
    } else {
        Some(1024_usize * 1024)
    };
    let built_candidates = graph
        .plan_candidates(
            Some(&example.cudnn),
            &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
        )?
        .build_candidates(
            Some(&example.cudnn),
            &CompileConfig::new()
                .with_build_policy(BuildPlanPolicy::AllSupported)
                .with_max_plans(plan_limit),
        )?
        .filter(&EngineConfigFilter::new().with_max_workspace_size(workspace_limit));
    println!("plan candidates: {}", built_candidates.candidate_count());

    let cuda = example.cudnn.cuda_context();
    let mut seed = 123_456_789_u32;
    let a_dev = DeviceMemory::from_slice(
        cuda,
        &init_bf16_image(graph.shape(a)?.storage_span()?, &mut seed),
    )?;
    let b_dev = DeviceMemory::from_slice(
        cuda,
        &init_bf16_image(graph.shape(b)?.storage_span()?, &mut seed),
    )?;
    let mut c_dev = DeviceMemory::<bf16>::zeroed(cuda, graph.shape(c)?.storage_span()?)?;
    let mut workspace = common::workspace(cuda, built_candidates.max_workspace_size())?;
    let mut bindings = built_candidates.bindings()?;
    bindings.set_input(a, &a_dev)?;
    bindings.set_input(b, &b_dev)?;
    bindings.set(c, &mut c_dev)?;
    let start = cuda.create_event_with_flags(EventFlags::DEFAULT)?;
    let stop = cuda.create_event_with_flags(EventFlags::DEFAULT)?;
    let mut execution_times = Vec::new();

    for plan in built_candidates.available_plan_ids() {
        // SAFETY: initialized inputs, writable output, workspace and events remain live through completion.
        // The helper drains the stream even after an enqueue or event-recording failure.
        let execution = example.execute_and_wait(|context| unsafe {
            built_candidates.execute(
                context,
                ExecuteArgs::new(&bindings)
                    .with_workspace(&mut workspace)
                    .with_plan(&plan),
            )?;
            start.record(&example.stream, EventRecordFlags::DEFAULT)?;
            for _ in 0..10 {
                built_candidates.execute(
                    context,
                    ExecuteArgs::new(&bindings)
                        .with_workspace(&mut workspace)
                        .with_plan(&plan),
                )?;
            }
            stop.record(&example.stream, EventRecordFlags::DEFAULT)?;
            Ok(())
        });
        if let Err(error) = execution {
            println!("{plan} failed execution: {error}");
            continue;
        }
        let elapsed_ms = stop.elapsed_time_since(&start)? / 10.0;
        println!("{plan} took {elapsed_ms:.7} ms");
        execution_times.push((plan, elapsed_ms));
    }

    let winner = execution_times
        .iter()
        .min_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1))
        .map(|(plan, _)| plan.clone())
        .ok_or(Error::NoAvailableEngines)?;
    println!("selected {}", built_candidates.name_for(&winner)?);
    drop(bindings);
    let compiled = built_candidates.compile_selected(&winner)?;
    let mut bindings = compiled.bindings()?;
    bindings.set_input(a, &a_dev)?;
    bindings.set_input(b, &b_dev)?;
    bindings.set(c, &mut c_dev)?;
    // SAFETY: all bindings and workspace remain live through execution and stream completion.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;
    println!(
        "executed {winner}, output={:?}",
        graph.shape(c)?.dimensions()
    );
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_autotuning", run())
}
