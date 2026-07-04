# singe-cublas

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cublas.svg)](https://crates.io/crates/singe-cublas)
[![Documentation](https://docs.rs/singe-cublas/badge.svg)](https://docs.rs/singe-cublas)
![License](https://img.shields.io/crates/l/singe-cublas.svg)

Safe Rust wrappers for NVIDIA cuBLAS dense GPU linear algebra, including the classic cuBLAS API, cuBLASLt matmul planning, and cuBLASXt host-driven multi-GPU BLAS3 execution.

## cuBLAS Example

Classic cuBLAS uses device-resident matrices and a handle created from a CUDA context.
The flow is: allocate device memory, create a handle, call BLAS routines, then copy results back when needed.

```rust,ignore
use singe_cublas::{
    blas::level3::dgemm,
    context::Context,
    error::Result,
    types::Operation,
};
use singe_cuda::{
    context::Context as CudaContext,
    device::Device,
    memory::DeviceMemory,
};

fn main() -> Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;

    let a = DeviceMemory::from_slice(&[1.0_f64, 3.0, 2.0, 4.0])?;
    let b = DeviceMemory::from_slice(&[5.0_f64, 7.0, 6.0, 8.0])?;
    let mut c = DeviceMemory::<f64>::zeroes(4)?;

    dgemm(
        &cublas,
        Operation::NonTranspose,
        Operation::NonTranspose,
        2, 2, 2,
        &1.0, &a, 2,
        &b, 2, &0.0,
        &mut c, 2,
    )?;

    assert_eq!(c.copy_to_host_vec()?, vec![19.0, 43.0, 22.0, 50.0]);
    Ok(())
}
```

Full example: [examples/cublas/cublas_level3_gemm.rs](examples/cublas/cublas_level3_gemm.rs).

## cuBLASLt Example

cuBLASLt is the programmable GEMM interface.
The important pieces are: `MatrixLayout`, `MatmulDescriptor`, `MatmulPreference`, heuristic results, optional algorithm, and optional workspace.

```rust,ignore
use std::error::Error;

use singe_cublas::{
    lt::{
        context::Context as LtContext,
        descriptor::MatrixLayout,
        matmul::{
            matmul,
            matmul_algorithm_heuristics,
            MatmulDescriptor,
            MatmulPreference,
        },
    },
    types::{ComputeType, Operation},
};
use singe_cuda::{
    context::Context as CudaContext,
    data_type::DataType,
    device::Device,
    memory::DeviceMemory,
};

fn main() -> Result<(), Box<dyn Error>> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let lt = LtContext::create(&cuda)?;

    let a = DeviceMemory::from_slice(&[1.0_f32, 3.0, 2.0, 4.0])?;
    let b = DeviceMemory::from_slice(&[5.0_f32, 7.0, 6.0, 8.0])?;
    let c = DeviceMemory::<f32>::zeroes(4)?;
    let mut d = DeviceMemory::<f32>::zeroes(4)?;

    let a_layout = MatrixLayout::create(DataType::F32, 2, 2, 2)?;
    let b_layout = MatrixLayout::create(DataType::F32, 2, 2, 2)?;
    let c_layout = MatrixLayout::create(DataType::F32, 2, 2, 2)?;
    let d_layout = MatrixLayout::create(DataType::F32, 2, 2, 2)?;

    let mut desc = MatmulDescriptor::create(ComputeType::F32, DataType::F32)?;
    desc.set_transpose_a(Operation::NonTranspose)?;
    desc.set_transpose_b(Operation::NonTranspose)?;

    let mut preference = MatmulPreference::create()?;
    preference.set_max_workspace_bytes(1 << 20)?;
    let heuristics = matmul_algorithm_heuristics(
        &lt,
        &desc,
        &a_layout,
        &b_layout,
        &c_layout,
        &d_layout,
        &preference,
        1,
    )?;
    let mut workspace = DeviceMemory::<u8>::create(heuristics[0].workspace_size)?;

    matmul(
        &lt,
        &desc,
        &1.0_f32, &a, &a_layout,
        &b, &b_layout, &0.0_f32,
        &c, &c_layout,
        &mut d, &d_layout,
        Some(&heuristics[0].algorithm),
        Some(&mut workspace),
        None,
    )?;

    assert_eq!(d.copy_to_host_vec()?, vec![19.0, 43.0, 22.0, 50.0]);
    Ok(())
}
```

Full example: [`examples/lt/lt_sgemm.rs`](examples/lt/lt_sgemm.rs).
More cuBLASLt examples cover epilogues, custom algorithm search, strided and pointer-array batching, grouped GEMM, FP8, MXFP8, NVFP4, and emulated double precision.

## cuBLASXt Example

cuBLASXt is a host-facing BLAS3 interface for one or more selected GPUs.

```rust,ignore
use singe_cublas::{
    error::Result,
    types::Operation,
    xt::{blas::sgemm, context::Context},
};
use singe_cuda::device::Device;

fn main() -> Result<()> {
    let ctx = Context::create_for(&[Device::current()?])?;
    ctx.set_block_dim(64)?;

    let a = [1.0_f32, 2.0, 3.0, 4.0];
    let b = [5.0_f32, 6.0, 7.0, 8.0];
    let mut c = [0.0_f32; 4];

    sgemm(
        &ctx,
        Operation::NonTranspose,
        Operation::NonTranspose,
        2, 2, 2,
        &1.0, &a, 2,
        &b, 2,
        &0.0, &mut c, 2,
    )?;

    singe_core::assert_close!(&c, &[23.0, 34.0, 31.0, 46.0], 1.0e-4);
    Ok(())
}
```

Use `Context::create_for(&[...])` to select an explicit single-device or multi-device cuBLASXt configuration.

Full example: [`examples/xt/xt_sgemm.rs`](examples/xt/xt_sgemm.rs).

## Cargo features

The `singe-cublas` crate defines these Cargo features:

- `lt`: Enables cuBLASLt submodule.
- `xt`: Enables cuBLASXt submodule.
- `cublas_13_5`: cuBLAS library version that this crate targets.
