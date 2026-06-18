mod common;

use singe_core::assert_close;
use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::knob::BackendKnobType,
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, bf16, f16},
    error::{Error, Result},
    frontend::{graph::Graph, operation::CompileConfig, plan::BuildPlanPolicy},
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
        .map(f16::from_f32)
        .map(|value| bf16::from_bits(value.to_bits()))
        .collect()
}

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;

    let b = 16;
    let m = 32;
    let n = 64;
    let k = 128;

    let mut graph = Graph::new();
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
    let c = graph.tensor(
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([b, m, n])?.with_strides([m * n, n, 1])?,
        )
        .with_id(1002),
    );
    graph.matmul(a, b_tensor, c, DataType::F32)?;

    let engine_indices = graph.engine_indices(&ctx.cudnn)?;
    let mut explicit_configs = Vec::<(i64, Vec<(BackendKnobType, i64)>)>::new();
    for engine_index in engine_indices {
        let knobs = match graph.knobs_for_engine(&ctx.cudnn, engine_index) {
            Ok(knobs) => knobs,
            Err(_) => continue,
        };
        let knob_map = knobs
            .into_iter()
            .filter_map(|knob| Some((knob.knob_type, knob.minimum_value.checked_add(knob.stride)?)))
            .collect::<Vec<_>>();
        explicit_configs.push((engine_index, knob_map));
    }

    let candidates = match graph.candidates_with_engines(&ctx.cudnn, &explicit_configs) {
        Ok(candidates) => candidates,
        Err(error) => return Err(error),
    };

    let built = match candidates.build_candidates_with_config(
        &ctx.cudnn,
        &CompileConfig::new().with_build_policy(BuildPlanPolicy::AllSupported),
    ) {
        Ok(built) => built,
        Err(error) => return Err(error),
    };

    println!("Graph has {} plan candidates.", built.candidate_count());
    assert!(
        built.candidate_count() >= 1,
        "expected at least one plan candidate"
    );

    let candidate_count = built.candidate_count();
    let candidate_info = (0..candidate_count)
        .map(|plan_index| {
            Ok((
                built.check_support_at(plan_index).is_ok(),
                built.compiled_plan_index_at(plan_index)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;

    let compiled = built.compile()?;
    let mut half_seed = 123_456_789_u32;
    let mut a_dev =
        DeviceMemory::from_slice(&init_bf16_image((b * m * k) as usize, &mut half_seed))?;
    let mut b_dev =
        DeviceMemory::from_slice(&init_bf16_image((b * k * n) as usize, &mut half_seed))?;
    let c_init = init_bf16_image((b * m * n) as usize, &mut half_seed);
    let mut c_dev = DeviceMemory::from_slice(&c_init)?;
    let mut bindings = compiled.bindings();
    bindings.set(a, &mut a_dev)?;
    bindings.set(b_tensor, &mut b_dev)?;
    bindings.set(c, &mut c_dev)?;

    if candidate_info[0].0 {
        let compiled_index = candidate_info[0].1.ok_or(Error::NoAvailableEngines)?;
        let mut workspace0 = common::workspace(compiled.workspace_size_at(compiled_index)?)?;
        compiled.execute_at(&ctx.cudnn, &bindings, Some(&mut workspace0), compiled_index)?;
        println!("executed plan 0");
    }

    if candidate_info.len() > 1 && candidate_info[1].0 {
        let compiled_index = candidate_info[1].1.ok_or(Error::NoAvailableEngines)?;
        let mut workspace1 = common::workspace(compiled.workspace_size_at(compiled_index)?)?;
        compiled.execute_at(&ctx.cudnn, &bindings, Some(&mut workspace1), compiled_index)?;
        println!("executed plan 1");
    }

    let c_host = c_dev.copy_to_host_vec()?;
    let c_first = c_host[0].to_f32();
    let c_last = c_host[c_host.len() - 1].to_f32();
    let c_checksum = c_host.iter().map(|value| value.to_f32()).sum::<f32>();
    let c_nonfinite = c_host
        .iter()
        .filter(|value| !value.to_f32().is_finite())
        .count();

    println!(
        "c: first={c_first}, last={c_last}, checksum={c_checksum}, nonfinite={c_nonfinite}, len={}",
        c_host.len()
    );

    assert_eq!(c_nonfinite, 0, "c.nonfinite mismatch");
    assert_close!(c_first, 0.00009727478, 1.0e-9, "c.first");
    assert_close!(c_last, 0.000030994415, 1.0e-9, "c.last");
    assert_close!(c_checksum, 2.2549438, 1.0e-5, "c.checksum");

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_custom_plan", run())
}
