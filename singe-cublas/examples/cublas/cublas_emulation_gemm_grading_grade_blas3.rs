use singe_cublas::{context::Context, error::Result, types::Operation};
use singe_cuda::{
    context::Context as CudaContext, device::Device, memory::DeviceMemory, types::Complex64,
};

fn main() -> Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;

    for &n in &[2_usize, 4, 8] {
        let a = make_real_matrix(n);
        let b = make_real_matrix(n);
        let expected = cpu_gemm_real(n, &a, &b);
        let gpu = run_dgemm(&cublas, n, &a, &b)?;
        println!(
            "dgemm n={n} max_relative_error={:.3e}",
            max_relative_error(&expected, &gpu)
        );

        let za = a
            .iter()
            .copied()
            .map(|value| Complex64::new(value, value * 0.1))
            .collect::<Vec<_>>();
        let zb = b
            .iter()
            .copied()
            .map(|value| Complex64::new(value, -value * 0.05))
            .collect::<Vec<_>>();
        let z_expected = cpu_gemm_complex(n, &za, &zb);
        let z_gpu = run_zgemm(&cublas, n, &za, &zb)?;
        println!(
            "zgemm n={n} max_relative_error={:.3e}",
            max_relative_error_complex(&z_expected, &z_gpu)
        );
    }
    Ok(())
}

fn make_real_matrix(n: usize) -> Vec<f64> {
    (0..n * n)
        .map(|index| ((index % 7) as f64 + 1.0) / ((index % 3) as f64 + 1.0))
        .collect()
}

fn cpu_gemm_real(n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
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

fn cpu_gemm_complex(n: usize, a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
    let mut c = vec![Complex64::new(0.0, 0.0); n * n];
    for col in 0..n {
        for row in 0..n {
            let mut sum = Complex64::new(0.0, 0.0);
            for inner in 0..n {
                sum += a[row + inner * n] * b[inner + col * n];
            }
            c[row + col * n] = sum;
        }
    }
    c
}

fn run_dgemm(
    cublas: &singe_cublas::context::Context,
    n: usize,
    a: &[f64],
    b: &[f64],
) -> Result<Vec<f64>> {
    use singe_cublas::blas::level3::dgemm;
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

fn run_zgemm(
    cublas: &singe_cublas::context::Context,
    n: usize,
    a: &[Complex64],
    b: &[Complex64],
) -> Result<Vec<Complex64>> {
    use singe_cublas::blas::level3::zgemm;
    let a = DeviceMemory::from_slice(a)?;
    let b = DeviceMemory::from_slice(b)?;
    let mut c = DeviceMemory::<Complex64>::zeroes(n * n)?;
    zgemm(
        cublas,
        Operation::NonTranspose,
        Operation::NonTranspose,
        n,
        n,
        n,
        &Complex64::new(1.0, 0.0),
        &a,
        n,
        &b,
        n,
        &Complex64::new(0.0, 0.0),
        &mut c,
        n,
    )?;
    Ok(c.copy_to_host_vec()?)
}

fn max_relative_error(expected: &[f64], actual: &[f64]) -> f64 {
    expected
        .iter()
        .zip(actual)
        .map(|(&expected, &actual)| {
            let scale = expected.abs().max(1.0);
            (expected - actual).abs() / scale
        })
        .fold(0.0, f64::max)
}

fn max_relative_error_complex(expected: &[Complex64], actual: &[Complex64]) -> f64 {
    expected
        .iter()
        .zip(actual)
        .map(|(&expected, &actual)| {
            let scale = expected.norm().max(1.0);
            (expected - actual).norm() / scale
        })
        .fold(0.0, f64::max)
}
