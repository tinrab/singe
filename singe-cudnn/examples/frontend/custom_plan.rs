mod common;

use std::env::args;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::{
        device::{DeviceProperties, DevicePropertiesSource},
        errata::{Errata, ErrataPlan},
        execution::{EngineChoice, EnginePlanningConfig, KnobChoiceInfo},
        heuristic::HeuristicQueryPolicy,
        tensor::{Shape, TensorId, TensorSpec},
    },
    data_type::{DataType, bf16},
    error::Result,
    frontend::{
        graph::Graph,
        operation::MatmulConfig,
        plan::{BuildPlanPolicy, CompileConfig, ExecuteArgs, HeuristicMode},
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
    let plan_limit = common::positive_usize_argument("--plans", 2)?;
    let example = common::ExampleContext::create()?;

    let b = 16;
    let m = 32;
    let n = 64;
    let k = 128;

    let mut graph = Graph::new();
    let a = graph.insert_tensor(
        TensorId::new(1000),
        TensorSpec::new(DataType::BF16, Shape::contiguous([b, m, k])?).with_name("A"),
    )?;
    let b_tensor = graph.insert_tensor(
        TensorId::new(1001),
        TensorSpec::new(DataType::BF16, Shape::contiguous([b, k, n])?).with_name("B"),
    )?;
    let c = graph.insert_tensor(
        TensorId::new(1002),
        TensorSpec::new(DataType::BF16, Shape::contiguous([b, m, n])?),
    )?;
    graph.matmul_into(
        a,
        b_tensor,
        c,
        MatmulConfig::new().with_compute_type(DataType::F32),
    )?;

    let input_a_elements = graph.shape(a)?.storage_span()?;
    let input_b_elements = graph.shape(b_tensor)?.storage_span()?;
    let output_elements = graph.shape(c)?.storage_span()?;
    let deviceless = args().any(|argument| argument == "--deviceless-preparation");
    if deviceless {
        graph.attach_device_properties(DeviceProperties::create(
            DevicePropertiesSource::Device(example.cudnn.cuda_context().device()),
        )?)?;
    }
    let target = if deviceless {
        None
    } else {
        Some(&example.cudnn)
    };
    let validated = graph.validate(target)?;
    println!(
        "validated {} expanded operations and {} declarations before native preparation",
        validated.graph().operations().len(),
        validated.tensors().len(),
    );
    let output = validated.tensor_info(c)?;
    println!(
        "validated output {}: dimensions={:?}, access={:?}",
        output.id,
        output.spec.shape.dimensions(),
        output.access,
    );
    for binding in validated.required_tensor_bindings() {
        println!(
            "binding {}: {:?}, initial state={}, indexing={}, operand uses={}",
            binding.id,
            binding.access,
            binding.reads_initial_state,
            binding.indexes_memory,
            binding.bindings.len(),
        );
    }
    let prepared = validated.prepare(target)?;
    println!(
        "prepared native operations: {}, types={:?}",
        prepared.operation_graph().operation_count(),
        prepared.operation_graph().operation_types(),
    );
    if let Some(uid) = prepared.tensor_info(c)?.backend_uid {
        let tensor = prepared.tensor_by_uid(uid)?;
        println!(
            "prepared output {}: native uid {uid}, dimensions={:?}, strides={:?}",
            tensor.id,
            tensor.spec.shape.dimensions(),
            tensor.spec.shape.strides(),
        );
    }
    let compile_config = CompileConfig::new().with_max_plans(plan_limit);
    let query_context = if args().any(|argument| argument == "--retained-target") {
        None
    } else {
        Some(&example.cudnn)
    };
    let candidates = if args().any(|argument| argument == "--query-heuristics") {
        let query_config = compile_config.clone().with_heuristic_modes(vec![
            HeuristicMode::A,
            HeuristicMode::B,
            HeuristicMode::Fallback,
        ]);
        let policy = if args().any(|argument| argument == "--stop-on-heuristic-error") {
            HeuristicQueryPolicy::StopOnError
        } else {
            HeuristicQueryPolicy::EvaluateAll
        };
        let queried = prepared.query_heuristics(query_context, &query_config, policy)?;
        for status in &queried.statuses {
            match &status.result {
                Ok(count) => println!("heuristic {}: {count} configurations", status.mode),
                Err(error) => println!("heuristic {}: {error}", status.mode),
            }
        }
        prepared.candidates_with_engine_configs(
            query_context,
            &queried.engine_configs,
            &compile_config,
        )?
    } else if args().any(|argument| argument == "--copy-configs") {
        let configs = prepared.engine_configs(query_context, &compile_config)?;
        prepared.candidates_with_engine_configs(query_context, &configs, &compile_config)?
    } else {
        let engine_indices = prepared.engine_indices()?;
        let mut explicit_configs = Vec::new();
        for engine_index in engine_indices {
            let engine = match prepared.engine(engine_index, EnginePlanningConfig::default()) {
                Ok(engine) => engine,
                Err(error) => {
                    println!("engine {engine_index} skipped: {error}");
                    continue;
                }
            };
            let knobs = match engine.knob_infos() {
                Ok(knobs) => knobs,
                Err(error) => {
                    println!("engine {engine_index} knob query failed: {error}");
                    continue;
                }
            };
            println!(
                "engine {engine_index}: numerical={:?}, behavior={:?}",
                engine.numerical_notes()?,
                engine.behavior_notes()?,
            );
            let knob_map = knobs
                .iter()
                .map(|knob| KnobChoiceInfo::new(knob.knob_type, knob.minimum_value))
                .collect::<Vec<_>>();
            explicit_configs.push(EngineChoice::new(engine_index, knob_map));
        }

        prepared.candidates_with_engines(query_context, &explicit_configs, &compile_config)?
    };
    let mut candidates = if let Some(errata) = Errata::load_from_config(None)? {
        if args().any(|argument| argument == "--print-errata") {
            println!("{}", errata.to_json()?);
        }
        let cudnn_version = version()?.raw();
        candidates.filter_by(|candidate| {
            let plan = ErrataPlan::from_engine(
                "Matmul",
                candidate.engine_index,
                &candidate.knob_choices,
                cudnn_version,
            );
            let Some(rule) = errata.matching_rules(plan).next() else {
                return true;
            };
            println!(
                "engine {} excluded by errata {:?}: knobs={:?}, cuDNN=[{}, {:?}), input={:?}, filter={:?}",
                rule.engine(),
                rule.id(),
                rule.knobs(),
                rule.version_start(),
                rule.version_end(),
                rule.input_shape(),
                rule.filter_shape(),
            );
            false
        })
    } else {
        candidates
    };
    if args().any(|argument| argument == "--prefer-small-workspace") {
        let mut order = candidates
            .support(compile_config.filter())
            .map(|support| (support.metadata.workspace_size, support.plan.clone()))
            .collect::<Vec<_>>();
        order.sort_by_key(|(workspace, _)| *workspace);
        if !order.is_empty() {
            let order = order.into_iter().map(|(_, plan)| plan).collect::<Vec<_>>();
            candidates.reorder_plans(&order)?;
        }
    }
    for support in candidates.support(compile_config.filter()) {
        println!(
            "{}: supported={}, workspace={} bytes, shared memory={:?} bytes",
            support.plan,
            support.is_supported(),
            support.metadata.workspace_size,
            support.metadata.shared_memory_size,
        );
        if let Some(reason) = &support.exclusion {
            println!("  excluded: {reason}");
        }
    }
    let mut built = candidates.into_build_candidates(&compile_config)?;
    built.build_plans(Some(&example.cudnn), BuildPlanPolicy::FirstSupported)?;
    built.build_plans(Some(&example.cudnn), BuildPlanPolicy::AllSupported)?;
    for report in built.build_reports() {
        println!("{report}");
    }

    println!("Graph has {} plan candidates.", built.candidate_count());
    let available_plans = built.available_plan_ids();

    let compiled = built.compile()?;
    println!(
        "default {}: {:?}, shared memory {:?} bytes, numerical notes {:?}",
        compiled.name(),
        compiled.engine_choice(),
        compiled.engine_metadata().shared_memory_size,
        compiled.engine_numerical_notes(),
    );
    let mut half_seed = 123_456_789_u32;
    let a_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_bf16_image(input_a_elements, &mut half_seed),
    )?;
    let b_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_bf16_image(input_b_elements, &mut half_seed),
    )?;
    let mut c_dev = DeviceMemory::<bf16>::zeroed(example.cudnn.cuda_context(), output_elements)?;
    let mut bindings = compiled.bindings()?;
    bindings.set_input(a, &a_dev)?;
    bindings.set_input(b_tensor, &b_dev)?;
    bindings.set(c, &mut c_dev)?;

    for plan in available_plans {
        let mut workspace = common::workspace(
            example.cudnn.cuda_context(),
            compiled.workspace_size_for(&plan)?,
        )?;
        // SAFETY: inputs, output, workspace and native plan remain live through stream completion.
        example.execute_and_wait(|context| unsafe {
            compiled.execute(
                context,
                ExecuteArgs::new(&bindings)
                    .with_workspace(&mut workspace)
                    .with_plan(&plan),
            )
        })?;
        println!("executed {plan}: {:?}", compiled.engine_choice_for(&plan)?);
    }

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_custom_plan", run())
}
