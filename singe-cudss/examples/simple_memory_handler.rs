mod common;

use std::{
    ffi::{c_int, c_void},
    ptr, slice,
};

use singe_cuda::memory::DeviceMemory;
use singe_cuda_sys::runtime::{self, cudaStream_t};
use singe_cudss::{config::Config, error::Result, memory::DeviceMemoryHandler};

use crate::common::{create_context, dense_matrix, dense_output, solve, spd_csr_matrix};

unsafe extern "C" fn device_alloc(
    _ctx: *mut c_void,
    ptr: *mut *mut c_void,
    size: singe_cudss_sys::size_t,
    stream: cudaStream_t,
) -> c_int {
    // SAFETY: the caller upholds this function's safety contract.
    let status = unsafe { runtime::cudaMallocAsync(ptr, size, stream) };
    println!("custom allocator: allocated {size} bytes");
    status as c_int
}

unsafe extern "C" fn device_free(
    _ctx: *mut c_void,
    ptr: *mut c_void,
    size: singe_cudss_sys::size_t,
    stream: cudaStream_t,
) -> c_int {
    // SAFETY: the caller upholds this function's safety contract.
    let status = unsafe { runtime::cudaFreeAsync(ptr, stream) };
    println!("custom allocator: freed {size} bytes");
    status as c_int
}

fn main() -> Result<()> {
    let (cuda, stream, context) = create_context()?;
    println!("cuDSS example: solving with a custom device memory handler");

    let n = 5i64;
    let nrhs = 1i64;
    let nnz = 8i64;

    let row_offsets = DeviceMemory::from_slice(&cuda, &[0_i32, 2, 4, 6, 7, 8])?;
    let col_indices = DeviceMemory::from_slice(&cuda, &[0_i32, 2, 1, 2, 2, 4, 3, 4])?;
    let a_values = DeviceMemory::from_slice(&cuda, &[4.0f64, 1.0, 3.0, 2.0, 5.0, 1.0, 1.0, 2.0])?;
    let b_values = DeviceMemory::from_slice(&cuda, &[7.0f64, 12.0, 25.0, 4.0, 13.0])?;
    let mut x_values = DeviceMemory::<f64>::create_uninit(&cuda, (n * nrhs) as usize)?;

    // SAFETY: the example keeps these buffers alive and synchronizes before reading the results.
    let handler = unsafe {
        DeviceMemoryHandler::create(
            "cuda async example memory handler",
            ptr::null_mut(),
            device_alloc,
            device_free,
        )?
    };
    // SAFETY: the example keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        context.set_device_memory_handler(&handler)?;
    }

    let config = Config::create()?;
    let mut data = context.create_data()?;

    let b = dense_matrix(n, nrhs, n, &b_values)?;
    let mut x = dense_output(n, nrhs, n, x_values.output())?;
    let a = spd_csr_matrix(n, n, nnz, &row_offsets, &col_indices, &a_values)?;

    // SAFETY: the example keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        solve(
            slice::from_ref(&stream),
            &context,
            &config,
            &mut data,
            &a,
            &mut x,
            &b,
        )
    }?;
    let returned = context.device_memory_handler()?;
    println!("attached handler: {}", returned.name());

    drop(data);
    // SAFETY: the example keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        context.clear_device_memory_handler()?;
    }
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
