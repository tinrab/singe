use singe_cublas::{
    blas::level3::{dgemm, gemm_ex, zgemm},
    context::Context,
    error::Result,
    types::{ComputeType, GemmAlgorithm, Operation},
};
use singe_cuda::{
    context::Context as CudaContext, data_type::DataType, device::Device, memory::DeviceMemory,
    types::Complex64,
};

fn main() -> Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;

    let a = vec![1.0_f64, 3.0, 2.0, 4.0];
    let b = vec![5.0_f64, 7.0, 6.0, 8.0];
    let expected = cpu_gemm_f64(2, &a, &b);

    let a_device = DeviceMemory::from_slice(&a)?;
    let b_device = DeviceMemory::from_slice(&b)?;

    let mut c_dgemm = DeviceMemory::<f64>::zeroes(4)?;
    dgemm(
        &cublas,
        Operation::NonTranspose,
        Operation::NonTranspose,
        2,
        2,
        2,
        &1.0,
        &a_device,
        2,
        &b_device,
        2,
        &0.0,
        &mut c_dgemm,
        2,
    )?;

    let mut c_gemm_ex = DeviceMemory::<f64>::zeroes(4)?;
    gemm_ex(
        &cublas,
        Operation::NonTranspose,
        Operation::NonTranspose,
        2,
        2,
        2,
        &1.0,
        &a_device,
        DataType::F64,
        2,
        &b_device,
        DataType::F64,
        2,
        &0.0,
        &mut c_gemm_ex,
        DataType::F64,
        2,
        ComputeType::F64,
        GemmAlgorithm::Default,
    )?;

    let za = DeviceMemory::from_slice(&[
        Complex64::new(1.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(4.0, 0.0),
    ])?;
    let zb = DeviceMemory::from_slice(&[
        Complex64::new(5.0, 0.0),
        Complex64::new(7.0, 0.0),
        Complex64::new(6.0, 0.0),
        Complex64::new(8.0, 0.0),
    ])?;
    let mut zc = DeviceMemory::<Complex64>::zeroes(4)?;
    zgemm(
        &cublas,
        Operation::NonTranspose,
        Operation::NonTranspose,
        2,
        2,
        2,
        &Complex64::new(1.0, 0.0),
        &za,
        2,
        &zb,
        2,
        &Complex64::new(0.0, 0.0),
        &mut zc,
        2,
    )?;

    assert_eq!(c_dgemm.copy_to_host_vec()?, expected);
    assert_eq!(c_gemm_ex.copy_to_host_vec()?, expected);
    let z_result = zc.copy_to_host_vec()?;
    assert_eq!(
        z_result,
        expected
            .iter()
            .copied()
            .map(|value| Complex64::new(value, 0.0))
            .collect::<Vec<_>>()
    );

    println!("dgemm: {:?}", c_dgemm.copy_to_host_vec()?);
    println!("gemm_ex: {:?}", c_gemm_ex.copy_to_host_vec()?);
    println!("zgemm: {z_result:?}");
    Ok(())
}

fn cpu_gemm_f64(n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    for col in 0..n {
        for row in 0..n {
            let mut sum = 0.0;
            for inner in 0..n {
                sum += a[row + inner * n] * b[inner + col * n];
            }
            c[row + col * n] = sum;
        }
    }
    c
}
