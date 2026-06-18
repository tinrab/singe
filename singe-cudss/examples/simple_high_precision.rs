mod common;

use singe_cuda::{device::Device, memory::DeviceMemory};
use singe_cudss::{config::Config, error::Result, types::Fp64Mp2};

use crate::common::*;

fn main() -> Result<()> {
    let props = Device::current()?.properties()?;
    let compute_capability = props.major * 10 + props.minor;
    if compute_capability < 90 {
        println!(
            "Example SKIPPED: high precision requires compute capability 9.0, got {}.{}",
            props.major, props.minor
        );
        return Ok(());
    }

    println!("cuDSS example: solving a high-precision SPD 5x5 system");

    let n = 5i64;
    let nrhs = 1i64;
    let nnz = 8i64;

    let expected_solution = [
        Fp64Mp2::new(1.0, 1.0e-20),
        Fp64Mp2::new(2.0, -1.0e-23),
        Fp64Mp2::from_f64(3.0),
        Fp64Mp2::from_f64(4.0),
        Fp64Mp2::from_f64(5.0),
    ];

    let row_offsets = DeviceMemory::from_slice(&[0_i32, 2, 4, 6, 7, 8])?;
    let col_indices = DeviceMemory::from_slice(&[0_i32, 2, 1, 2, 2, 4, 3, 4])?;
    let a_values = DeviceMemory::from_slice(&[
        mp2(4.0),
        mp2(1.0),
        mp2(3.0),
        mp2(2.0),
        mp2(5.0),
        mp2(1.0),
        mp2(1.0),
        mp2(2.0),
    ])?;
    let b_values = DeviceMemory::from_slice(&[
        Fp64Mp2::new(7.0, 4.0e-20),
        Fp64Mp2::new(12.0, -3.0e-23),
        Fp64Mp2::new(25.0, 1.0e-20 - 2.0e-23),
        mp2(4.0),
        mp2(13.0),
    ])?;
    let x_values = DeviceMemory::<Fp64Mp2>::zeroes((n * nrhs) as usize)?;

    let (cuda, _stream, context) = create_context()?;
    let mut config = Config::create()?;
    let deterministic_mode = 0i32;
    config.set(
        singe_cudss::types::ConfigParameter::DeterministicMode,
        &deterministic_mode,
    )?;
    let mut data = context.create_data()?;

    let b = dense_matrix(n, nrhs, n, &b_values)?;
    let mut x = dense_matrix(n, nrhs, n, &x_values)?;
    let a = spd_csr_matrix(n, n, nnz, &row_offsets, &col_indices, &a_values)?;

    solve(&context, &config, &mut data, &a, &mut x, &b)?;
    cuda.synchronize()?;

    let x = x_values.copy_to_host_vec()?;
    for (index, (actual, expected)) in x.iter().copied().zip(expected_solution).enumerate() {
        let delta = (actual - expected).abs();
        println!(
            "x[{index}] = ({:+.14e}, {:+.14e}), expected ({:+.14e}, {:+.14e}), delta={delta:.14e}",
            actual.hi, actual.lo, expected.hi, expected.lo
        );
        assert!(delta <= 1.0e-30);
    }

    println!("Example PASSED");
    Ok(())
}
