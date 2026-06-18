mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudss::{config::Config, error::Result};

use crate::common::*;

fn main() -> Result<()> {
    println!("cuDSS example: solving and computing a relative residual");

    let n = 5i64;
    let nrhs = 1i64;
    let nnz = 8i64;

    let row_offsets_h = [0_i32, 2, 4, 6, 7, 8];
    let col_indices_h = [0_i32, 2, 1, 2, 2, 4, 3, 4];
    let a_values_h = [4.0f64, 1.0, 3.0, 2.0, 5.0, 1.0, 1.0, 2.0];
    let b_values_h = [7.0f64, 12.0, 25.0, 4.0, 13.0];

    let row_offsets = DeviceMemory::from_slice(&row_offsets_h)?;
    let col_indices = DeviceMemory::from_slice(&col_indices_h)?;
    let a_values = DeviceMemory::from_slice(&a_values_h)?;
    let b_values = DeviceMemory::from_slice(&b_values_h)?;
    let x_values = DeviceMemory::<f64>::zeroes((n * nrhs) as usize)?;

    let (cuda, _stream, context) = create_context()?;
    let config = Config::create()?;
    let mut data = context.create_data()?;

    let b = dense_matrix(n, nrhs, n, &b_values)?;
    let mut x = dense_matrix(n, nrhs, n, &x_values)?;
    let a = spd_csr_matrix(n, n, nnz, &row_offsets, &col_indices, &a_values)?;

    solve(&context, &config, &mut data, &a, &mut x, &b)?;
    cuda.synchronize()?;

    let x_h = x_values.copy_to_host_vec()?;
    let residual = relative_residual(
        &row_offsets_h,
        &col_indices_h,
        &a_values_h,
        &x_h,
        &b_values_h,
    );
    println!("solution: {x_h:?}");
    println!("relative residual: {residual:.3e}");
    assert!(residual <= 1.0e-14);

    println!("Example PASSED");
    Ok(())
}

fn relative_residual(
    row_offsets: &[i32],
    col_indices: &[i32],
    values: &[f64],
    x: &[f64],
    b: &[f64],
) -> f64 {
    let n = b.len();
    let mut ax = vec![0.0; n];
    let mut frobenius_sq = 0.0;

    for row in 0..n {
        for offset in row_offsets[row] as usize..row_offsets[row + 1] as usize {
            let col = col_indices[offset] as usize;
            let value = values[offset];
            ax[row] += value * x[col];
            if col == row {
                frobenius_sq += value * value;
            } else {
                ax[col] += value * x[row];
                frobenius_sq += 2.0 * value * value;
            }
        }
    }

    let residual_norm = ax
        .iter()
        .zip(b)
        .map(|(actual, expected)| {
            let diff = actual - expected;
            diff * diff
        })
        .sum::<f64>()
        .sqrt();
    let x_norm = x.iter().map(|value| value * value).sum::<f64>().sqrt();
    let b_norm = b.iter().map(|value| value * value).sum::<f64>().sqrt();

    residual_norm / (frobenius_sq.sqrt() * x_norm + b_norm)
}
