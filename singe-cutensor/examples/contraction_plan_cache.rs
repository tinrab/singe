use std::{env, fs};

use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory};
use singe_cutensor::{
    context::Context,
    error::{Error, Result, Status},
    operation::{ComputeDescriptor, OperationDescriptor, TensorOperand},
    plan::{Plan, PlanPreference},
    tensor::TensorDescriptor,
    types::{AutotuneMode, CacheMode, WorkspacePreference},
};

fn main() -> Result<()> {
    let cuda_context = CudaContext::create()?;
    let context = Context::create(&cuda_context)?;
    let stream = cuda_context.create_stream()?;

    let cache_path = env::temp_dir().join("singe-cutensor-plan-cache.bin");
    match context.read_plan_cache_from_file(&cache_path) {
        Ok(lines) => println!("loaded {lines} cached plan entries"),
        Err(Error::Cutensor {
            code: Status::IoError,
            ..
        }) => println!("no existing plan cache at {}", cache_path.display()),
        Err(err) => return Err(err),
    }
    context.resize_plan_cache(128)?;

    let mode_c = vec!['m'.into(), 'u'.into(), 'n'.into(), 'v'.into()];
    let mode_a = vec!['m'.into(), 'h'.into(), 'k'.into(), 'n'.into()];
    let mode_b = vec!['u'.into(), 'k'.into(), 'v'.into(), 'h'.into()];

    let extent_c = vec![2, 2, 2, 2];
    let extent_a = vec![2, 2, 2, 2];
    let extent_b = vec![2, 2, 2, 2];

    let host_a = (0..extent_a.iter().product::<u64>())
        .map(|index| index as f32 + 1.0)
        .collect::<Vec<_>>();
    let host_b = (0..extent_b.iter().product::<u64>())
        .map(|index| -0.5_f32 + index as f32 * 0.25)
        .collect::<Vec<_>>();
    let host_c = vec![0.0_f32; extent_c.iter().product::<u64>() as usize];

    let device_a = DeviceMemory::from_slice(&host_a)?;
    let device_b = DeviceMemory::from_slice(&host_b)?;
    let device_c_input = DeviceMemory::from_slice(&host_c)?;
    let mut device_c = DeviceMemory::from_slice(&host_c)?;

    const ALIGNMENT: u32 = 128;

    let descriptor_a = TensorDescriptor::create_for::<f32>(&context, &extent_a, ALIGNMENT)?;
    let descriptor_b = TensorDescriptor::create_for::<f32>(&context, &extent_b, ALIGNMENT)?;
    let descriptor_c = TensorDescriptor::create_for::<f32>(&context, &extent_c, ALIGNMENT)?;

    let contraction = OperationDescriptor::contraction(
        &context,
        TensorOperand::identity(&descriptor_a, &mode_a),
        TensorOperand::identity(&descriptor_b, &mode_b),
        TensorOperand::identity(&descriptor_c, &mode_c),
        TensorOperand::identity(&descriptor_c, &mode_c),
        ComputeDescriptor::f32(),
    )?;

    let mut preference = PlanPreference::create_default(&context)?;
    preference.set_cache_mode(CacheMode::Pedantic)?;
    preference.set_autotune_mode(AutotuneMode::Incremental)?;
    preference.set_incremental_count(2)?;

    let workspace_size = Plan::estimate_workspace_size(
        &context,
        &contraction,
        &preference,
        WorkspacePreference::Default,
    )?;

    for attempt in 0..3 {
        let plan = Plan::create(&context, &contraction, &preference, workspace_size)?;
        let mut workspace = if plan.required_workspace_size() > 0 {
            Some(DeviceMemory::<u8>::create(
                plan.required_workspace_size_bytes()?,
            )?)
        } else {
            None
        };

        let alpha = 1.1_f32;
        let beta = 0.0_f32;
        plan.contract(
            &alpha,
            &device_a,
            &device_b,
            &beta,
            &device_c_input,
            &mut device_c,
            workspace.as_mut(),
            &stream,
        )?;
        stream.synchronize()?;
        println!(
            "plan-cache attempt {} used {} bytes of workspace",
            attempt + 1,
            plan.required_workspace_size()
        );
    }

    context.write_plan_cache_to_file(&cache_path)?;
    println!("wrote plan cache to {}", cache_path.display());
    let _ = fs::remove_file(&cache_path);
    Ok(())
}
