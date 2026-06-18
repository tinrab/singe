mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, bf16, f16},
    error::{Error, Result},
    frontend::{
        graph::Graph,
        operation::{CompileConfig, HeuristicMode},
        plan::BuildPlanPolicy,
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
        .with_name("A"),
    );
    let b_tensor = graph.tensor(
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([b, k, n])?.with_strides([k * n, n, 1])?,
        )
        .with_name("B"),
    );
    let c = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([b, m, n])?.with_strides([m * n, n, 1])?,
    ));
    graph.matmul(a, b_tensor, c, DataType::F32)?;

    let candidates = graph.plan_candidates(&ctx.cudnn, &[HeuristicMode::A])?;
    let built_candidates = candidates.build_candidates_with_config(
        &ctx.cudnn,
        &CompileConfig::new().with_build_policy(BuildPlanPolicy::AllSupported),
    )?;
    println!("all plans: {}", built_candidates.candidate_count());

    let mut half_seed = 123_456_789_u32;
    let mut a_dev =
        DeviceMemory::from_slice(&init_bf16_image((b * m * k) as usize, &mut half_seed))?;
    let mut b_dev =
        DeviceMemory::from_slice(&init_bf16_image((b * k * n) as usize, &mut half_seed))?;
    let c_init = init_bf16_image((b * m * n) as usize, &mut half_seed);

    let mut winner_index = None;
    for plan_index in 0..built_candidates.candidate_count() {
        if built_candidates.check_support_at(plan_index).is_ok() {
            winner_index = Some(plan_index);
            break;
        }
    }
    let winner_index = winner_index.ok_or(Error::NoAvailableEngines)?;
    let compiled = built_candidates.compile_at(winner_index)?;

    let mut c_dev = DeviceMemory::from_slice(&c_init)?;
    let mut bindings = compiled.bindings();
    bindings.set(a, &mut a_dev)?;
    bindings.set(b_tensor, &mut b_dev)?;
    bindings.set(c, &mut c_dev)?;
    let mut workspace = common::workspace(compiled.max_workspace_size()?)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    let c_host = c_dev.copy_to_host_vec()?;
    let c_first = c_host[0].to_f32();
    let c_last = c_host[c_host.len() - 1].to_f32();
    let c_checksum = c_host.iter().map(|value| value.to_f32()).sum::<f32>();
    let c_nonfinite = c_host
        .iter()
        .filter(|value| !value.to_f32().is_finite())
        .count();

    println!(
        "winner index={winner_index}, c: first={c_first}, last={c_last}, checksum={c_checksum}, nonfinite={c_nonfinite}, len={}",
        c_host.len()
    );

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_all_plans_autotune", run())
}
