mod common;

use singe_cuda::{memory::DeviceMemory, types::Complex32};
use singe_cudss::{config::Config, error::Result};

use crate::common::*;

fn main() -> Result<()> {
    println!("cuDSS example: solving a complex symmetric positive-definite 5x5 system");

    let n = 5i64;
    let nrhs = 1i64;
    let nnz = 8i64;

    let row_offsets = DeviceMemory::from_slice(&[0_i32, 2, 4, 6, 7, 8])?;
    let col_indices = DeviceMemory::from_slice(&[0_i32, 2, 1, 2, 2, 4, 3, 4])?;
    let a_values = DeviceMemory::from_slice(&[
        Complex32::new(4.0, 0.0),
        Complex32::new(1.0, 0.0),
        Complex32::new(3.0, 0.0),
        Complex32::new(2.0, 0.0),
        Complex32::new(5.0, 0.0),
        Complex32::new(1.0, 0.0),
        Complex32::new(1.0, 0.0),
        Complex32::new(2.0, 0.0),
    ])?;
    let b_values = DeviceMemory::from_slice(&[
        Complex32::new(7.0, 0.0),
        Complex32::new(12.0, 0.0),
        Complex32::new(25.0, 0.0),
        Complex32::new(4.0, 0.0),
        Complex32::new(13.0, 0.0),
    ])?;
    let x_values = DeviceMemory::<Complex32>::zeroes((n * nrhs) as usize)?;

    let (cuda, _stream, context) = create_context()?;
    let config = Config::create()?;
    let mut data = context.create_data()?;

    let b = dense_matrix(n, nrhs, n, &b_values)?;
    let mut x = dense_matrix(n, nrhs, n, &x_values)?;
    let a = spd_csr_matrix(n, n, nnz, &row_offsets, &col_indices, &a_values)?;

    solve(&context, &config, &mut data, &a, &mut x, &b)?;
    cuda.synchronize()?;

    let x = x_values.copy_to_host_vec()?;
    for (index, value) in x.iter().copied().enumerate() {
        let expected = (index + 1) as f32;
        println!(
            "x[{index}] = ({:.4}, {:.4}), expected ({expected:.4}, 0.0000)",
            value.re, value.im
        );
        assert!((value.re - expected).abs() + value.im.abs() <= 2.0e-6);
    }

    println!("Example PASSED");
    Ok(())
}
