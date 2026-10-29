mod common;

use singe_cudnn::{
    backend::{
        behavior::BackendBehaviorNote,
        tensor::{Shape, TensorId, TensorSpec},
    },
    data_type::DataType,
    error::Result,
    frontend::{
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::MatmulConfig,
        plan::{BuildPlanPolicy, CompileConfig, EngineConfigFilter, HeuristicMode},
    },
};

fn run() -> Result<()> {
    let window_size = common::positive_usize_argument("--build-window", 8)?;
    let context = common::ExampleContext::create()?;

    let b = 16_i64;
    let m = 32_i64;
    let n = 64_i64;
    let k = 128_i64;

    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::BF16)
                .with_compute(DataType::F32),
        ),
    );
    let a = graph.insert_tensor(
        TensorId::new(1000),
        TensorSpec::new(
            DataType::BF16,
            Shape::with_strides([b, m, k], [m * k, k, 1])?,
        )
        .with_name("A"),
    )?;

    let b_tensor = graph.insert_tensor(
        TensorId::new(1001),
        TensorSpec::new(
            DataType::BF16,
            Shape::with_strides([b, k, n], [k * n, n, 1])?,
        )
        .with_name("B"),
    )?;

    let c = graph.matmul(
        a,
        b_tensor,
        MatmulConfig::new().with_compute_type(DataType::F32),
        TensorOutputConfig::new().output_tensor(),
    )?;
    graph.modify_tensor(c, |tensor| tensor.output_tensor().with_name("C"))?;
    let serial = graph.compile(
        Some(&context.cudnn),
        &CompileConfig::new()
            .with_heuristic_modes(vec![HeuristicMode::A])
            .with_build_policy(BuildPlanPolicy::FirstSupported)
            .with_filter(
                EngineConfigFilter::new()
                    .with_included_behavior_notes(vec![BackendBehaviorNote::RuntimeCompilation]),
            ),
    )?;

    println!(
        "serial: {} candidates, {} plans",
        serial.candidate_count(),
        serial.execution_plans().len(),
    );

    let compiled = graph.compile(
        Some(&context.cudnn),
        &CompileConfig::new()
            .with_heuristic_modes(vec![HeuristicMode::A])
            .with_build_policy(BuildPlanPolicy::FirstSupportedParallel { window_size })
            .with_filter(
                EngineConfigFilter::new()
                    .with_included_behavior_notes(vec![BackendBehaviorNote::RuntimeCompilation]),
            ),
    )?;

    println!(
        "parallel window {window_size}: {} candidates, {} plans",
        compiled.candidate_count(),
        compiled.execution_plans().len(),
    );

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_parallel_compilation", run())
}
