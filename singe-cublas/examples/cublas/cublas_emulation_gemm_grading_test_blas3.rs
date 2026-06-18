use singe_cublas::{blas::level3::dgemm, context::Context, error::Result, types::Operation};
use singe_cuda::{context::Context as CudaContext, device::Device, memory::DeviceMemory};

fn main() -> Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;

    let n = 4;
    let a = complementary_sparse_rows(n);
    let b = complementary_sparse_cols(n);
    let gpu = run_dgemm(&cublas, n, &a, &b)?;
    let cpu = cpu_gemm(n, &a, &b);

    assert_eq!(gpu, cpu);
    assert_eq!(gpu[0], 0.0);
    assert_eq!(gpu[5], 0.0);

    println!("zero-pattern test passed: {gpu:?}");

    let scaled_a = a.iter().map(|value| value * 16.0).collect::<Vec<_>>();
    let scaled_b = b.iter().map(|value| value * 0.25).collect::<Vec<_>>();
    let scaled_gpu = run_dgemm(&cublas, n, &scaled_a, &scaled_b)?;
    let scaled_cpu = cpu_gemm(n, &scaled_a, &scaled_b);
    assert_eq!(scaled_gpu, scaled_cpu);

    println!("scaling-invariance test passed: {scaled_gpu:?}");
    Ok(())
}

fn complementary_sparse_rows(n: usize) -> Vec<f64> {
    let mut a = vec![1.0; n * n];
    for col in 0..n {
        a[col * n] = 0.0;
    }
    a
}

fn complementary_sparse_cols(n: usize) -> Vec<f64> {
    let mut b = vec![1.0; n * n];
    for row in 0..n {
        b[row + n] = 0.0;
    }
    b
}

fn run_dgemm(
    cublas: &singe_cublas::context::Context,
    n: usize,
    a: &[f64],
    b: &[f64],
) -> Result<Vec<f64>> {
    let a = DeviceMemory::from_slice(a)?;
    let b = DeviceMemory::from_slice(b)?;
    let mut c = DeviceMemory::<f64>::zeroes(n * n)?;
    dgemm(
        cublas,
        Operation::NonTranspose,
        Operation::NonTranspose,
        n,
        n,
        n,
        &1.0,
        &a,
        n,
        &b,
        n,
        &0.0,
        &mut c,
        n,
    )?;
    Ok(c.copy_to_host_vec()?)
}

fn cpu_gemm(n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
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
