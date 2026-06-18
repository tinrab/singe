mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudss::{
    config::Config,
    error::Result,
    types::{ConfigParameter, DataParameter, Phase},
};

use crate::common::*;

fn main() -> Result<()> {
    println!("cuDSS example: computing a dense Schur complement");

    let n = 5i64;
    let nrhs = 1i64;
    let nnz = 8i64;

    let row_offsets = DeviceMemory::from_slice(&[0_i32, 2, 4, 6, 7, 8])?;
    let col_indices = DeviceMemory::from_slice(&[0_i32, 2, 1, 2, 2, 4, 3, 4])?;
    let a_values = DeviceMemory::from_slice(&[4.0f64, 1.0, 3.0, 2.0, 5.0, 1.0, 1.0, 2.0])?;
    let b_values = DeviceMemory::from_slice(&[7.0f64, 12.0, 25.0, 4.0, 13.0])?;
    let x_values = DeviceMemory::<f64>::zeroes((n * nrhs) as usize)?;

    let schur_indices = [0_i32, 1, 0, 0, 1];

    let (cuda, _stream, context) = create_context()?;
    let mut config = Config::create()?;
    let schur_mode = 1i32;
    config.set(ConfigParameter::SchurMode, &schur_mode)?;
    let mut data = context.create_data()?;
    data.set_slice(DataParameter::UserSchurIndices, &schur_indices)?;

    let b = dense_matrix(n, nrhs, n, &b_values)?;
    let mut x = dense_matrix(n, nrhs, n, &x_values)?;
    let a = spd_csr_matrix(n, n, nnz, &row_offsets, &col_indices, &a_values)?;

    context.execute(Phase::ANALYSIS, &config, &mut data, &a, &mut x, &b)?;
    let shape: [i64; 3] = data.get(DataParameter::SchurShape)?;
    println!(
        "Schur shape: rows={}, cols={}, sparse_nnz={}",
        shape[0], shape[1], shape[2]
    );
    assert_eq!(shape[0], 2);
    assert_eq!(shape[1], 2);
    assert_eq!(shape[2], 4);

    context.execute(Phase::FACTORIZATION, &config, &mut data, &a, &mut x, &b)?;

    let schur_values = DeviceMemory::<f64>::zeroes((shape[0] * shape[1]) as usize)?;
    let schur = dense_matrix(shape[0], shape[1], shape[0], &schur_values)?;
    let mut schur_raw = schur.as_raw();
    data.get_into(DataParameter::SchurMatrix, &mut schur_raw)?;
    cuda.synchronize()?;

    let values = schur_values.copy_to_host_vec()?;
    println!("dense Schur complement values (column-major): {values:?}");
    let expected = [
        2.157_894_736_842_105_3,
        -0.421_052_631_578_947_3,
        -0.421_052_631_578_947_3,
        1.789_473_684_210_526_3,
    ];
    for (actual, expected) in values.iter().copied().zip(expected) {
        assert!((actual - expected).abs() <= 1.0e-14);
    }

    println!("Example PASSED");
    Ok(())
}
