use std::time::Instant;

use singe_cublas::{
    blas::extension::gemm_ex,
    context::Context,
    scalar::Scalar,
    types::{ComputeType, GemmAlgorithm, Operation},
};
use singe_cuda::{context::Context as CudaContext, device::Device, memory::DeviceMemory};

fn main() -> anyhow::Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;

    let a = DeviceMemory::from_slice(&cuda, &[1.0_f64, 3.0, 2.0, 4.0])?;
    let b = DeviceMemory::from_slice(&cuda, &[5.0_f64, 7.0, 6.0, 8.0])?;
    let mut c = DeviceMemory::<f64>::zeroed(&cuda, 4)?;

    // The first invocation lets cuBLAS initialize its internal state.
    let start = Instant::now();
    gemm_ex(
        &cublas,
        Operation::NonTranspose,
        Operation::NonTranspose,
        2,
        2,
        2,
        Scalar::host(&1.0_f64),
        &a,
        2,
        &b,
        2,
        Scalar::host(&0.0_f64),
        &mut c,
        2,
        ComputeType::F64,
        GemmAlgorithm::Default,
    )?;
    let _ = c.copy_to_host_vec()?;
    println!("first call (initialization): {:?}", start.elapsed());

    let start = Instant::now();
    gemm_ex(
        &cublas,
        Operation::NonTranspose,
        Operation::NonTranspose,
        2,
        2,
        2,
        Scalar::host(&1.0_f64),
        &a,
        2,
        &b,
        2,
        Scalar::host(&0.0_f64),
        &mut c,
        2,
        ComputeType::F64,
        GemmAlgorithm::Default,
    )?;
    let _ = c.copy_to_host_vec()?;
    println!("second call (baseline): {:?}", start.elapsed());

    let start = Instant::now();
    gemm_ex(
        &cublas,
        Operation::NonTranspose,
        Operation::NonTranspose,
        2,
        2,
        2,
        Scalar::host(&1.0_f64),
        &a,
        2,
        &b,
        2,
        Scalar::host(&0.0_f64),
        &mut c,
        2,
        ComputeType::F64,
        GemmAlgorithm::Autotune,
    )?;
    let _ = c.copy_to_host_vec()?;
    println!("third call (autotune): {:?}", start.elapsed());

    let start = Instant::now();
    gemm_ex(
        &cublas,
        Operation::NonTranspose,
        Operation::NonTranspose,
        2,
        2,
        2,
        Scalar::host(&1.0_f64),
        &a,
        2,
        &b,
        2,
        Scalar::host(&0.0_f64),
        &mut c,
        2,
        ComputeType::F64,
        GemmAlgorithm::Autotune,
    )?;
    let result = c.copy_to_host_vec()?;
    println!("fourth call (cached autotune): {:?}", start.elapsed());

    assert_eq!(result, vec![19.0, 43.0, 22.0, 50.0]);
    println!("gemm_ex_autotuning result: {result:?}");
    Ok(())
}
