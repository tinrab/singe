mod common;

use singe_cudnn::{
    backend::behavior::BackendBehaviorNote,
    backend::tensor::{Shape, TensorSpec},
    data_type::DataType,
    error::Result,
    frontend::{
        graph::Graph,
        operation::{CompileConfig, HeuristicMode, MatmulConfig},
        plan::BuildPlanPolicy,
    },
};

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;

    let b = 16_i64;
    let m = 32_i64;
    let n = 64_i64;
    let k = 128_i64;

    let mut graph = Graph::new()
        .with_io_data_type(DataType::BF16)
        .with_compute_data_type(DataType::F32);
    let a = graph.tensor(
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([b, m, k])?.with_strides([m * k, k, 1])?,
        )
        .with_id(1000)
        .with_name("A"),
    );

    let b_tensor = graph.tensor(
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([b, k, n])?.with_strides([k * n, n, 1])?,
        )
        .with_id(1001)
        .with_name("B"),
    );

    let c = graph.matmul_infer_with_config(a, b_tensor, MatmulConfig::new(DataType::F32))?;
    graph.modify_tensor(c, |tensor| {
        tensor.output_tensor().with_id(1002).with_name("C")
    })?;
    let serial = graph.compile_with_config(
        &ctx.cudnn,
        &CompileConfig::new()
            .with_heuristic_modes(vec![HeuristicMode::A])
            .with_build_policy(BuildPlanPolicy::FirstSupported)
            .with_included_behavior_notes(vec![BackendBehaviorNote::RuntimeCompilation]),
    )?;

    assert!(!serial.execution_plans().is_empty());
    assert_eq!(serial.execution_plans().len(), 1);
    assert_eq!(serial.engine_configs().len(), 1);

    let compiled = graph.compile_with_config(
        &ctx.cudnn,
        &CompileConfig::new()
            .with_heuristic_modes(vec![HeuristicMode::A])
            .with_build_policy(BuildPlanPolicy::FirstSupportedParallel { window_size: 8 })
            .with_included_behavior_notes(vec![BackendBehaviorNote::RuntimeCompilation]),
    )?;

    assert!(!compiled.execution_plans().is_empty());
    assert_eq!(compiled.execution_plans().len(), 1);
    assert_eq!(compiled.engine_configs().len(), 1);

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_parallel_compilation", run())
}
