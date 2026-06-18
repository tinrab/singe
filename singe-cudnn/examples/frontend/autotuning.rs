mod common;

use singe_cuda::event::{EventFlags, EventRecordFlags};
use singe_cuda::memory::DeviceMemory;
use singe_cuda::types::DevicePtr;
use singe_cudnn::{
    backend::tensor::{Shape, TensorId, TensorSpec},
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

fn create_graph(
    b: i64,
    m: i64,
    n: i64,
    k: i64,
    a_id: i64,
    b_id: i64,
    c_id: i64,
) -> Result<(Graph, TensorIdTriplet)> {
    let mut graph = Graph::new();
    let a = graph.tensor(
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([b, m, k])?.with_strides([m * k, k, 1])?,
        )
        .with_id(a_id)
        .with_name("A"),
    );
    let b_tensor = graph.tensor(
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([b, k, n])?.with_strides([k * n, n, 1])?,
        )
        .with_id(b_id)
        .with_name("B"),
    );
    let c = graph.tensor(
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([b, m, n])?.with_strides([m * n, n, 1])?,
        )
        .with_id(c_id),
    );
    graph.matmul(a, b_tensor, c, DataType::F32)?;
    Ok((graph, TensorIdTriplet { a, b: b_tensor, c }))
}

struct TensorIdTriplet {
    a: TensorId,
    b: TensorId,
    c: TensorId,
}

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;

    let b = 16_i64;
    let m = 32_i64;
    let n = 64_i64;
    let k = 128_i64;
    let a_id = 0_i64;
    let b_id = 1_i64;
    let c_id = 2_i64;

    let (graph, tensors) = create_graph(b, m, n, k, a_id, b_id, c_id)?;

    let candidates = graph.plan_candidates(&ctx.cudnn, &[HeuristicMode::A])?;

    let mut half_seed = 123_456_789_u32;
    let mut a_dev =
        DeviceMemory::from_slice(&init_bf16_image((b * m * k) as usize, &mut half_seed))?;
    let mut b_dev =
        DeviceMemory::from_slice(&init_bf16_image((b * k * n) as usize, &mut half_seed))?;
    let mut c_dev =
        DeviceMemory::from_slice(&init_bf16_image((b * m * n) as usize, &mut half_seed))?;

    let built_candidates = match candidates.build_candidates_with_config(
        &ctx.cudnn,
        &CompileConfig::new().with_build_policy(BuildPlanPolicy::AllSupported),
    ) {
        Ok(compiled) => compiled,
        Err(error) => return Err(error),
    }
    .deselect_workspace_greater_than(1024 * 1024)?;

    println!(
        "Graph has {} plan candidates.",
        built_candidates.candidate_count()
    );

    let mut bindings = built_candidates.bindings();
    unsafe {
        bindings
            .set_id(tensors.a, DevicePtr::from_raw(a_dev.as_mut_ptr().cast()))
            .set_id(tensors.b, DevicePtr::from_raw(b_dev.as_mut_ptr().cast()))
            .set_id(tensors.c, DevicePtr::from_raw(c_dev.as_mut_ptr().cast()));
    }
    let mut workspace = common::workspace(built_candidates.max_workspace_size()?)?;
    let mut execution_times = vec![10.0_f32; built_candidates.candidate_count()];

    let start = ctx
        .cudnn
        .cuda_context()
        .create_event_with_flags(EventFlags::DEFAULT)?;
    let stop = ctx
        .cudnn
        .cuda_context()
        .create_event_with_flags(EventFlags::DEFAULT)?;

    for plan_index in 0..built_candidates.candidate_count() {
        let warmup_status =
            built_candidates.execute_at(&ctx.cudnn, &bindings, Some(&mut workspace), plan_index);
        if let Err(error) = warmup_status {
            println!("Plan at index {plan_index} failed execution {error}");
            continue;
        }
        ctx.cudnn.cuda_context().synchronize()?;

        let iterations = 10;
        let mut timed_ok = true;
        start.record(&ctx.stream, EventRecordFlags::DEFAULT)?;
        for _ in 0..iterations {
            if let Err(error) =
                built_candidates.execute_at(&ctx.cudnn, &bindings, Some(&mut workspace), plan_index)
            {
                println!("Plan at index {plan_index} failed execution {error}");
                execution_times[plan_index] = f32::INFINITY;
                timed_ok = false;
                break;
            }
        }
        if !timed_ok {
            continue;
        }
        stop.record(&ctx.stream, EventRecordFlags::DEFAULT)?;
        stop.synchronize()?;
        let elapsed_ms = stop.elapsed_time_since(&start)? / iterations as f32;

        println!("Plan at index {plan_index} took {elapsed_ms:.7} ms.");
        execution_times[plan_index] = elapsed_ms;
    }

    let winner_index = execution_times
        .iter()
        .enumerate()
        .min_by(|lhs, rhs| lhs.1.total_cmp(rhs.1))
        .map(|(plan_index, _)| plan_index)
        .filter(|plan_index| execution_times[*plan_index].is_finite())
        .ok_or(Error::NoAvailableEngines)?;
    match built_candidates.name_at(winner_index) {
        Ok(winner_name) => {
            println!("Successful candidate {winner_name} is at index {winner_index}")
        }
        Err(_) => println!("Successful candidate is at index {winner_index}"),
    }

    let compiled = built_candidates.compile_at(winner_index)?;
    let mut bindings = compiled.bindings();
    bindings.set(tensors.a, &mut a_dev)?;
    bindings.set(tensors.b, &mut b_dev)?;
    bindings.set(tensors.c, &mut c_dev)?;
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
    common::finish("frontend_autotuning", run())
}
