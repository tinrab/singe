mod common;

use std::slice;

use singe_cuda::memory::DeviceMemory;
use singe_cudss::{
    config::Config,
    error::Result,
    types::{DataParameter, Phase},
};

use crate::common::{create_context, dense_matrix, dense_output, execute, spd_csr_matrix};

fn main() -> Result<()> {
    let (cuda, stream, context) = create_context()?;
    println!("cuDSS example: extracting reordering information after the reordering phase");

    let n = 5i64;
    let nrhs = 1i64;
    let nnz = 8i64;

    let row_perm_expected = [4_i32, 3, 1, 0, 2];
    let col_perm_expected = [4_i32, 3, 1, 0, 2];

    let row_offsets = DeviceMemory::from_slice(&cuda, &[0_i32, 2, 4, 6, 7, 8])?;
    let col_indices = DeviceMemory::from_slice(&cuda, &[0_i32, 2, 1, 2, 2, 4, 3, 4])?;
    let a_values = DeviceMemory::from_slice(&cuda, &[4.0f64, 1.0, 3.0, 2.0, 5.0, 1.0, 1.0, 2.0])?;
    let b_values = DeviceMemory::from_slice(&cuda, &[7.0f64, 12.0, 25.0, 4.0, 13.0])?;
    let mut x_values = DeviceMemory::<f64>::create_uninit(&cuda, (n * nrhs) as usize)?;

    let config = Config::create()?;
    let mut data = context.create_data()?;

    let b = dense_matrix(n, nrhs, n, &b_values)?;
    let mut x = dense_output(n, nrhs, n, x_values.output())?;
    let a = spd_csr_matrix(n, n, nnz, &row_offsets, &col_indices, &a_values)?;

    // SAFETY: the example keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        execute(
            slice::from_ref(&stream),
            &context,
            Phase::REORDERING,
            &config,
            &mut data,
            &a,
            &mut x,
            &b,
        )
    }?;

    // SAFETY: the value pointer and size describe a local of the parameter type.
    let row_perm: [i32; 5] = unsafe { data.get_raw(DataParameter::PermutationReorderRow) }?;
    // SAFETY: the value pointer and size describe a local of the parameter type.
    let col_perm: [i32; 5] = unsafe { data.get_raw(DataParameter::PermutationReorderColumn) }?;

    // SAFETY: the example keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        execute(
            slice::from_ref(&stream),
            &context,
            Phase::SYMBOLIC_FACTORIZATION | Phase::FACTORIZATION | Phase::SOLVE,
            &config,
            &mut data,
            &a,
            &mut x,
            &b,
        )
    }?;
    cuda.synchronize()?;

    drop(x);
    // SAFETY: checked completed production initialized this compact output.
    let x_values = unsafe { x_values.assume_init() };
    let x = x_values.copy_to_host_vec()?;
    for (index, value) in x.iter().copied().enumerate() {
        let expected = (index + 1) as f64;
        println!("x[{index}] = {value:.4}, expected {expected:.4}");
        assert!((value - expected).abs() <= 2.0e-15);
    }

    for (index, (actual, expected)) in row_perm.iter().zip(row_perm_expected).enumerate() {
        println!("row_perm[{index}] = {actual}, expected {expected}");
        assert_eq!(*actual, expected);
    }
    for (index, (actual, expected)) in col_perm.iter().zip(col_perm_expected).enumerate() {
        println!("col_perm[{index}] = {actual}, expected {expected}");
        assert_eq!(*actual, expected);
    }

    println!("Example PASSED");
    Ok(())
}
