use std::{env, fs};

use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory, types::Complex32};
use singe_cutensor::{
    context::Context,
    error::{Error, Result, Status},
    operation::{ComputeDescriptor, OperationDescriptor, TensorOperand},
    plan::{Plan, PlanPreference},
    tensor::TensorDescriptor,
    types::{JitMode, WorkspacePreference},
};

fn main() -> Result<()> {
    let cuda_context = CudaContext::create()?;
    let context = Context::create(&cuda_context)?;
    let stream = cuda_context.create_stream()?;

    let cache_path = env::temp_dir().join("singe-cutensor-kernel-cache.bin");
    match context.read_kernel_cache_from_file(&cache_path) {
        Ok(()) => println!("loaded a cached JIT kernel"),
        Err(Error::Cutensor {
            code: Status::IoError,
            ..
        }) => println!("no existing kernel cache at {}", cache_path.display()),
        Err(Error::Cutensor {
            code: Status::NotSupported,
            ..
        }) => {
            println!("kernel JIT caching is not supported on this platform");
            return Ok(());
        }
        Err(err) => return Err(err),
    }

    let mode_c = vec![0.into(), 1.into(), 3.into(), 5.into()];
    let mode_a = vec![0.into(), 1.into(), 2.into(), 3.into()];
    let mode_b = vec![4.into(), 5.into(), 2.into()];

    let extent_c = vec![2, 2, 2, 2];
    let extent_a = vec![2, 2, 2, 2];
    let extent_b = vec![2, 2, 2];

    let host_a = (0..extent_a.iter().product::<u64>())
        .map(|index| Complex32::new(index as f32 * 0.25, (index % 3) as f32 * 0.5))
        .collect::<Vec<_>>();
    let host_b = (0..extent_b.iter().product::<u64>())
        .map(|index| Complex32::new(-0.25 * index as f32, 0.1 * (index as f32 + 1.0)))
        .collect::<Vec<_>>();
    let host_c = vec![Complex32::new(0.0, 0.0); extent_c.iter().product::<u64>() as usize];

    let device_a = DeviceMemory::from_slice(&host_a)?;
    let device_b = DeviceMemory::from_slice(&host_b)?;
    let device_c_input = DeviceMemory::from_slice(&host_c)?;
    let mut device_c_default = DeviceMemory::from_slice(&host_c)?;
    let mut device_c_jit = DeviceMemory::from_slice(&host_c)?;

    const ALIGNMENT: u32 = 128;

    let descriptor_a = TensorDescriptor::create_for::<Complex32>(&context, &extent_a, ALIGNMENT)?;
    let descriptor_b = TensorDescriptor::create_for::<Complex32>(&context, &extent_b, ALIGNMENT)?;
    let descriptor_c = TensorDescriptor::create_for::<Complex32>(&context, &extent_c, ALIGNMENT)?;

    let contraction = OperationDescriptor::contraction(
        &context,
        TensorOperand::identity(&descriptor_a, &mode_a),
        TensorOperand::identity(&descriptor_b, &mode_b),
        TensorOperand::identity(&descriptor_c, &mode_c),
        TensorOperand::identity(&descriptor_c, &mode_c),
        ComputeDescriptor::tf32x3(),
    )?;

    let preference_default = PlanPreference::create(
        &context,
        singe_cutensor::types::Algorithm::Default,
        JitMode::None,
    )?;
    let workspace_default = Plan::estimate_workspace_size(
        &context,
        &contraction,
        &preference_default,
        WorkspacePreference::Default,
    )?;
    let plan_default = Plan::create(
        &context,
        &contraction,
        &preference_default,
        workspace_default,
    )?;
    let mut default_workspace = if plan_default.required_workspace_size() > 0 {
        Some(DeviceMemory::<u8>::create(
            plan_default.required_workspace_size_bytes()?,
        )?)
    } else {
        None
    };

    let preference_jit = PlanPreference::create(
        &context,
        singe_cutensor::types::Algorithm::Default,
        JitMode::Default,
    )?;
    let workspace_jit = Plan::estimate_workspace_size(
        &context,
        &contraction,
        &preference_jit,
        WorkspacePreference::Default,
    )?;
    let plan_jit = Plan::create(&context, &contraction, &preference_jit, workspace_jit)?;
    let mut jit_workspace = if plan_jit.required_workspace_size() > 0 {
        Some(DeviceMemory::<u8>::create(
            plan_jit.required_workspace_size_bytes()?,
        )?)
    } else {
        None
    };

    let alpha = Complex32::new(1.1, 0.0);
    let beta = Complex32::new(0.0, 0.0);
    plan_default.contract(
        &alpha,
        &device_a,
        &device_b,
        &beta,
        &device_c_input,
        &mut device_c_default,
        default_workspace.as_mut(),
        &stream,
    )?;
    plan_jit.contract(
        &alpha,
        &device_a,
        &device_b,
        &beta,
        &device_c_input,
        &mut device_c_jit,
        jit_workspace.as_mut(),
        &stream,
    )?;
    stream.synchronize()?;

    let default_result = device_c_default.copy_to_host_vec()?;
    let jit_result = device_c_jit.copy_to_host_vec()?;
    assert_eq!(default_result.len(), jit_result.len());
    for (default_value, jit_value) in default_result.iter().zip(&jit_result) {
        assert!((default_value.re - jit_value.re).abs() < 5.0e-3);
        assert!((default_value.im - jit_value.im).abs() < 5.0e-3);
    }

    match context.write_kernel_cache_to_file(&cache_path) {
        Ok(()) => {
            println!("wrote kernel cache to {}", cache_path.display());
            let _ = fs::remove_file(&cache_path);
        }
        Err(Error::Cutensor {
            code: Status::NotSupported,
            ..
        }) => println!("kernel cache export is not supported on this platform"),
        Err(err) => return Err(err),
    }

    println!("default and JIT contractions produced matching results");
    Ok(())
}
