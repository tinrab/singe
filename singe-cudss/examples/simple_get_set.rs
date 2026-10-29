mod common;

use std::slice;

use singe_cuda::memory::DeviceMemory;
use singe_cudss::{
    config::Config,
    error::Result,
    types::{ConfigParameter, DataParameter, MatchingAlgorithm, Phase, ReorderingAlgorithm},
};

use crate::common::{create_context, dense_matrix, dense_output, execute, spd_csr_matrix};

fn main() -> Result<()> {
    let (cuda, stream, context) = create_context()?;
    println!("cuDSS example: solver settings and extra solver data");
    println!("cuDSS version: {}", singe_cudss::version()?);

    let n = 5i64;
    let nrhs = 1i64;
    let nnz = 8i64;

    let row_offsets = DeviceMemory::from_slice(&cuda, &[0_i32, 2, 4, 6, 7, 8])?;
    let col_indices = DeviceMemory::from_slice(&cuda, &[0_i32, 2, 1, 2, 2, 4, 3, 4])?;
    let a_values = DeviceMemory::from_slice(&cuda, &[4.0f64, 1.0, 3.0, 2.0, 5.0, 1.0, 1.0, 2.0])?;
    let b_values = DeviceMemory::from_slice(&cuda, &[7.0f64, 12.0, 25.0, 4.0, 13.0])?;
    let mut x_values = DeviceMemory::<f64>::create_uninit(&cuda, (n * nrhs) as usize)?;

    let mut diag = DeviceMemory::<f64>::create_uninit(&cuda, n as usize)?;
    let mut row_scale = DeviceMemory::<f64>::create_uninit(&cuda, n as usize)?;
    let mut col_scale = DeviceMemory::<f64>::create_uninit(&cuda, n as usize)?;

    let mut config = Config::create()?;
    let mut data = context.create_data()?;

    let reordering = ReorderingAlgorithm::Default;
    let matching = MatchingAlgorithm::Auto;
    let use_superpanels = 0i32;
    // SAFETY: the value pointer and size describe a local of the parameter type.
    unsafe { config.set_raw(ConfigParameter::ReorderingAlgorithm, &reordering) }?;
    // SAFETY: the value pointer and size describe a local of the parameter type.
    unsafe { config.set_raw(ConfigParameter::MatchingAlgorithm, &matching) }?;
    // SAFETY: the value pointer and size describe a local of the parameter type.
    unsafe { config.set_raw(ConfigParameter::UseSuperpanels, &use_superpanels) }?;

    let b = dense_matrix(n, nrhs, n, &b_values)?;
    let mut x = dense_output(n, nrhs, n, x_values.output())?;
    let a = spd_csr_matrix(n, n, nnz, &row_offsets, &col_indices, &a_values)?;

    // SAFETY: the example keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        execute(
            slice::from_ref(&stream),
            &context,
            Phase::ANALYSIS,
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
    println!("reorder row permutation: {row_perm:?}");
    println!("reorder col permutation: {col_perm:?}");

    let matching = config.matching_algorithm()?;
    if matching != MatchingAlgorithm::None {
        // SAFETY: the value pointer and size describe a local of the parameter type.
        let matching_perm: [i32; 5] = unsafe { data.get_raw(DataParameter::PermutationMatching) }?;
        println!("matching column permutation: {matching_perm:?}");
    }

    // SAFETY: the value pointer and size describe a local of the parameter type.
    let memory_estimates: [i64; 16] = unsafe { data.get_raw(DataParameter::MemoryEstimates) }?;
    println!(
        "memory estimates: device stable={} peak={}, host stable={} peak={}",
        memory_estimates[0], memory_estimates[1], memory_estimates[2], memory_estimates[3]
    );

    // SAFETY: the example keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        execute(
            slice::from_ref(&stream),
            &context,
            Phase::FACTORIZATION,
            &config,
            &mut data,
            &a,
            &mut x,
            &b,
        )
    }?;

    let info = data.info()?;
    // SAFETY: the value pointer and size describe a local of the parameter type.
    let npivots: i32 = unsafe { data.get_raw(DataParameter::NumberOfPivots) }?;
    // SAFETY: the value pointer and size describe a local of the parameter type.
    let inertia: [i32; 2] = unsafe { data.get_raw(DataParameter::Inertia) }?;
    println!("info={info}, npivots={npivots}, inertia={inertia:?}");

    let lu_nnz_size = data.size_of(DataParameter::LuNnz)?;
    assert_eq!(lu_nnz_size, size_of::<i64>());
    // SAFETY: the value pointer and size describe a local of the parameter type.
    let lu_nnz: i64 = unsafe { data.get_raw(DataParameter::LuNnz) }?;
    println!("LU nonzeros: {lu_nnz}");

    let written = stream.sync_scope(|_| {
        // SAFETY: this f64 parameter output remains exclusive through completion.
        Ok(unsafe { data.get_device::<f64>(DataParameter::Diagonal, &mut diag) })
    })??;
    assert_eq!(written, n as usize * size_of::<f64>());
    // SAFETY: the completed copy produced the full compact f64 range.
    let diag = unsafe { diag.assume_init() };
    let diag = diag.copy_to_host_vec()?;
    println!("diagonal entries: {diag:?}");

    if matches!(
        matching,
        MatchingAlgorithm::Auto | MatchingAlgorithm::MaxDiagProduct
    ) {
        let written = stream.sync_scope(|_| {
            // SAFETY: this f64 parameter output remains exclusive through completion.
            Ok(unsafe { data.get_device::<f64>(DataParameter::ScaleRow, &mut row_scale) })
        })??;
        assert_eq!(written, n as usize * size_of::<f64>());
        // SAFETY: the completed copy produced the full compact f64 range.
        let row_scale = unsafe { row_scale.assume_init() };
        let written = stream.sync_scope(|_| {
            Ok(unsafe {
                // SAFETY: this f64 parameter output remains exclusive through completion.
                data.get_device::<f64>(DataParameter::ScaleColumn, &mut col_scale)
            })
        })??;
        assert_eq!(written, n as usize * size_of::<f64>());
        // SAFETY: the completed copy produced the full compact f64 range.
        let col_scale = unsafe { col_scale.assume_init() };
        println!("row scale: {:?}", row_scale.copy_to_host_vec()?);
        println!("col scale: {:?}", col_scale.copy_to_host_vec()?);
    }

    let iterative_refinement_steps = 2i32;
    // SAFETY: the value pointer and size describe a local of the parameter type.
    unsafe {
        config.set_raw(
            ConfigParameter::IterativeRefinementSteps,
            &iterative_refinement_steps,
        )
    }?;

    // SAFETY: the example keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        execute(
            slice::from_ref(&stream),
            &context,
            Phase::SOLVE,
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

    println!("Example PASSED");
    Ok(())
}
