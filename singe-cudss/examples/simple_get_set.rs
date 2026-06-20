mod common;

use std::mem;

use singe_cuda::{memory::DeviceMemory, types::DevicePtr};
use singe_cudss::{
    config::Config,
    error::Result,
    types::{ConfigParameter, DataParameter, MatchingAlgorithm, Phase, ReorderingAlgorithm},
};

use crate::common::*;

fn main() -> Result<()> {
    println!("cuDSS example: solver settings and extra solver data");
    println!("cuDSS version: {}", singe_cudss::version()?);

    let n = 5i64;
    let nrhs = 1i64;
    let nnz = 8i64;

    let row_offsets = DeviceMemory::from_slice(&[0_i32, 2, 4, 6, 7, 8])?;
    let col_indices = DeviceMemory::from_slice(&[0_i32, 2, 1, 2, 2, 4, 3, 4])?;
    let a_values = DeviceMemory::from_slice(&[4.0f64, 1.0, 3.0, 2.0, 5.0, 1.0, 1.0, 2.0])?;
    let b_values = DeviceMemory::from_slice(&[7.0f64, 12.0, 25.0, 4.0, 13.0])?;
    let x_values = DeviceMemory::<f64>::zeroes((n * nrhs) as usize)?;

    let diag = DeviceMemory::<f64>::zeroes(n as usize)?;
    let row_scale = DeviceMemory::<f64>::zeroes(n as usize)?;
    let col_scale = DeviceMemory::<f64>::zeroes(n as usize)?;

    let (cuda, _stream, context) = create_context()?;
    let mut config = Config::create()?;
    let mut data = context.create_data()?;

    let reordering = ReorderingAlgorithm::Default;
    let matching = MatchingAlgorithm::Auto;
    let use_superpanels = 0i32;
    config.set(ConfigParameter::ReorderingAlgorithm, &reordering)?;
    config.set(ConfigParameter::MatchingAlgorithm, &matching)?;
    config.set(ConfigParameter::UseSuperpanels, &use_superpanels)?;

    let b = dense_matrix(n, nrhs, n, &b_values)?;
    let mut x = dense_matrix(n, nrhs, n, &x_values)?;
    let a = spd_csr_matrix(n, n, nnz, &row_offsets, &col_indices, &a_values)?;

    context.execute(Phase::ANALYSIS, &config, &mut data, &a, &mut x, &b)?;

    let row_perm: [i32; 5] = data.get(DataParameter::PermutationReorderRow)?;
    let col_perm: [i32; 5] = data.get(DataParameter::PermutationReorderColumn)?;
    println!("reorder row permutation: {row_perm:?}");
    println!("reorder col permutation: {col_perm:?}");

    let matching: MatchingAlgorithm = config.get(ConfigParameter::MatchingAlgorithm)?;
    if matching != MatchingAlgorithm::None {
        let matching_perm: [i32; 5] = data.get(DataParameter::PermutationMatching)?;
        println!("matching column permutation: {matching_perm:?}");
    }

    let memory_estimates: [i64; 16] = data.get(DataParameter::MemoryEstimates)?;
    println!(
        "memory estimates: device stable={} peak={}, host stable={} peak={}",
        memory_estimates[0], memory_estimates[1], memory_estimates[2], memory_estimates[3]
    );

    context.execute(Phase::FACTORIZATION, &config, &mut data, &a, &mut x, &b)?;

    let info: i32 = data.get(DataParameter::Info)?;
    let npivots: i32 = data.get(DataParameter::NumberOfPivots)?;
    let inertia: [i32; 2] = data.get(DataParameter::Inertia)?;
    println!("info={info}, npivots={npivots}, inertia={inertia:?}");

    let lu_nnz_size = data.size_of(DataParameter::LuNnz)?;
    assert_eq!(lu_nnz_size, mem::size_of::<i64>());
    let lu_nnz: i64 = data.get(DataParameter::LuNnz)?;
    println!("LU nonzeros: {lu_nnz}");

    data.get_device(
        DataParameter::Diagonal,
        unsafe { DevicePtr::from_raw(diag.as_mut_ptr().cast()) },
        diag.byte_len(),
    )?;
    let diag = diag.copy_to_host_vec()?;
    println!("diagonal entries: {diag:?}");

    if matches!(
        matching,
        MatchingAlgorithm::Auto | MatchingAlgorithm::MaxDiagProduct
    ) {
        data.get_device(
            DataParameter::ScaleRow,
            unsafe { DevicePtr::from_raw(row_scale.as_mut_ptr().cast()) },
            row_scale.byte_len(),
        )?;
        data.get_device(
            DataParameter::ScaleColumn,
            unsafe { DevicePtr::from_raw(col_scale.as_mut_ptr().cast()) },
            col_scale.byte_len(),
        )?;
        println!("row scale: {:?}", row_scale.copy_to_host_vec()?);
        println!("col scale: {:?}", col_scale.copy_to_host_vec()?);
    }

    let iterative_refinement_steps = 2i32;
    config.set(
        ConfigParameter::IterativeRefinementSteps,
        &iterative_refinement_steps,
    )?;

    context.execute(Phase::SOLVE, &config, &mut data, &a, &mut x, &b)?;
    cuda.synchronize()?;

    let x = x_values.copy_to_host_vec()?;
    for (index, value) in x.iter().copied().enumerate() {
        let expected = (index + 1) as f64;
        println!("x[{index}] = {value:.4}, expected {expected:.4}");
        assert!((value - expected).abs() <= 2.0e-15);
    }

    println!("Example PASSED");
    Ok(())
}
