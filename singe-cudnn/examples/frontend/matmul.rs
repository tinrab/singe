mod common;

use singe_core::assert_close;
use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorId, TensorSpec},
    data_type::{DataType, bf16},
    error::{Error, Result},
    frontend::{
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::MatmulConfig,
        plan::{BuildPlanPolicy, CompileConfig, EngineConfigFilter, ExecuteArgs, HeuristicMode},
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

fn count_nonfinite_f32(values: &[f32]) -> usize {
    values.iter().filter(|value| !value.is_finite()).count()
}

fn run() -> Result<()> {
    let example = common::ExampleContext::create()?;

    let b = 16;
    let m = 32;
    let n = 64;
    let k = 128;

    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(DataTypePolicy::new().with_compute(DataType::F32)),
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
        MatmulConfig::new().with_name("projection"),
        TensorOutputConfig::new()
            .with_data_type(DataType::F32)
            .output_tensor(),
    )?;
    for (id, tensor) in graph.tensors_named("projection::C") {
        println!("named result {id}: {:?}", tensor.shape.dimensions());
    }
    let input_a_elements = graph.shape(a)?.storage_span()?;
    let input_b_elements = graph.shape(b_tensor)?.storage_span()?;
    let output_elements = graph.shape(c)?.storage_span()?;

    let candidates = graph
        .plan_candidates(
            Some(&example.cudnn),
            &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
        )?
        .filter(
            &EngineConfigFilter::new().with_excluded_engine_name_substrings(&["eng4_".to_owned()]),
        );
    let built_candidates = candidates.build_candidates(
        Some(&example.cudnn),
        &CompileConfig::new()
            .with_heuristic_modes(vec![HeuristicMode::A])
            .with_build_policy(BuildPlanPolicy::AllSupported),
    )?;

    let winner = built_candidates
        .available_plan_ids()
        .into_iter()
        .next()
        .ok_or(Error::NoAvailableEngines)?;
    let compiled = built_candidates.compile_selected(&winner)?;

    let mut half_seed = 123_456_789_u32;
    let a_host = init_bf16_image(input_a_elements, &mut half_seed);
    let b_host = init_bf16_image(input_b_elements, &mut half_seed);
    let a_dev = DeviceMemory::from_slice(example.cudnn.cuda_context(), &a_host)?;
    let b_dev = DeviceMemory::from_slice(example.cudnn.cuda_context(), &b_host)?;
    let mut c_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_f32_image(output_elements, &mut half_seed),
    )?;
    let mut workspace =
        common::workspace(example.cudnn.cuda_context(), compiled.max_workspace_size())?;
    let mut bindings = compiled.bindings()?;
    bindings.set_input(a, &a_dev.view())?;
    bindings.set_input(b_tensor, &b_dev.view())?;
    bindings.set(c, &mut c_dev)?;
    // SAFETY: initialized inputs, writable output, workspace and native plan remain
    // live through stream completion, including an enqueue failure.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;

    let c_host = c_dev.copy_to_host_vec()?;
    assert_eq!(count_nonfinite_f32(&c_host), 0, "c.nonfinite mismatch");

    // Reference: each batch is an [m, k] x [k, n] product accumulated in f32.
    for batch in 0..b as usize {
        for row in 0..m as usize {
            for column in 0..n as usize {
                let expected: f32 = (0..k as usize)
                    .map(|inner| {
                        a_host[(batch * m as usize + row) * k as usize + inner].to_f32()
                            * b_host[(batch * k as usize + inner) * n as usize + column].to_f32()
                    })
                    .sum();
                let actual = c_host[(batch * m as usize + row) * n as usize + column];
                assert_close!(actual, expected, 1.0e-2, "c");
            }
        }
    }

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_matmul", run())
}
