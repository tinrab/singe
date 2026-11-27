# singe-cutensor

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cutensor.svg)](https://crates.io/crates/singe-cutensor)
[![Documentation](https://docs.rs/singe-cutensor/badge.svg)](https://docs.rs/singe-cutensor)
![License](https://img.shields.io/crates/l/singe-cutensor.svg)

Safe Rust wrappers for NVIDIA cuTENSOR, cuTENSORMg, and cuTENSORMp, for tensor contractions, reductions, permutations, and elementwise operations on one or several GPUs.

This crate provides:

- **Tensor descriptors:** dense tensors with explicit extents, strides, and alignment, and block-sparse tensors.
- **Operations:** contractions of two or three tensors, reductions, permutations, elementwise binary and trinary operations, and block-sparse contractions, with modes labelled by characters.
- **Planning:** kernel-selection preferences, JIT-compiled kernels, workspace estimates, and plan and kernel caches that can be saved to files.
- **Multi-GPU and multi-process:** cuTENSORMg and cuTENSORMp, behind the `mg` and `mp` features.

## Example

Contract `D[m, n] = A[m, k] * B[k, n]`, a matrix product written as a tensor contraction:

```rust,no_run
use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory, stream::StreamOptions};
use singe_cutensor::{
    context::Context,
    operation::{ComputeDescriptor, OperationDescriptor, TensorOperand},
    plan::{Plan, PlanPreference},
    tensor::TensorDescriptor,
    types::{Mode, WorkspacePreference},
};

fn main() -> anyhow::Result<()> {
    let cuda = CudaContext::create()?;
    let context = Context::create(&cuda)?;
    let stream = cuda.create_stream(StreamOptions::default())?;

    let modes_a = [Mode::from('m'), Mode::from('k')];
    let modes_b = [Mode::from('k'), Mode::from('n')];
    let modes_d = [Mode::from('m'), Mode::from('n')];
    let descriptor_a = TensorDescriptor::create_for::<f32>(&context, &[2, 4], 128)?;
    let descriptor_b = TensorDescriptor::create_for::<f32>(&context, &[4, 3], 128)?;
    let descriptor_d = TensorDescriptor::create_for::<f32>(&context, &[2, 3], 128)?;

    let contraction = OperationDescriptor::contraction(
        &context,
        TensorOperand::identity(&descriptor_a, &modes_a),
        TensorOperand::identity(&descriptor_b, &modes_b),
        TensorOperand::identity(&descriptor_d, &modes_d),
        TensorOperand::identity(&descriptor_d, &modes_d),
        ComputeDescriptor::f32(),
    )?;
    let preference = PlanPreference::create_default(&context)?;
    let workspace_limit = Plan::estimate_workspace_size(
        &context,
        &contraction,
        &preference,
        WorkspacePreference::Default,
    )?;
    let plan = Plan::create(&context, &contraction, &preference, workspace_limit)?;
    let mut workspace = DeviceMemory::<u8>::create_uninit(&cuda, plan.workspace_size())?;

    let a = DeviceMemory::from_slice(&cuda, &[1.0_f32; 8])?;
    let b = DeviceMemory::from_slice(&cuda, &[0.5_f32; 12])?;
    let c = DeviceMemory::<f32>::zeroed(&cuda, 6)?;
    let mut d = DeviceMemory::<f32>::zeroed(&cuda, 6)?;

    // D = 1 * A * B + 0 * C
    plan.contract(
        &1.0_f32,
        &a,
        &b,
        &0.0_f32,
        &c,
        &mut d,
        Some(workspace.output()),
        &stream,
    )?;

    assert_eq!(d.copy_to_host_vec()?, [2.0; 6]);
    Ok(())
}
```

An operation is described by `TensorDescriptor`s and an `OperationDescriptor`, planned into a `Plan`, and run through methods on the plan.

`Plan::contract` checks every operand against the plan, rejects stream capture, and waits for the stream before it returns.
Each operation also has an `unsafe` `*_async` form that runs the same checks and only enqueues the work; its caller keeps the plan, operands, workspace, and scalars alive until the work finishes.

## Features

- `mg`: the `mg` module, for cuTENSORMg, which splits tensors across several GPUs in one process.
- `mp`: the `mp` module, for cuTENSORMp, which distributes tensors across processes and communicates through `singe-nccl`.

## More examples

- [Contraction](./examples/contraction.rs): plan and run `D = alpha * A * B + beta * C` on four-mode tensors.
- [Trinary contraction](./examples/contraction_trinary.rs): contract three tensors in one operation.
- [JIT contraction](./examples/contraction_jit.rs): compile a kernel at plan time and reuse the kernel cache file.
- [Plan cache](./examples/contraction_plan_cache.rs): reuse contraction planning work through the plan cache file.
- [Einsum](./examples/einsum.rs): express contractions and reductions as einsum equations.
- [Reduction](./examples/reduction.rs): reduce tensor modes.
- [Permutation](./examples/elementwise_permute.rs): reorder tensor modes.
- [Permutation with padding](./examples/elementwise_permute_padding.rs): permute into a padded output.
- [Elementwise binary](./examples/elementwise_binary.rs) and [elementwise trinary](./examples/elementwise_trinary.rs): combine two or three tensors.
- [Block-sparse contraction](./examples/blocksparse.rs): contract block-sparse tensors.
- [Multi-GPU contraction](./examples/mg_contraction_multi_gpu.rs): distribute a contraction over the listed GPUs, with `--features mg`.
- [Multi-GPU blog post](./examples/mg_blog_post.rs): the cuTENSORMg blog post's contraction over a power-of-two number of GPUs, with `--features mg`.
- [Multi-process contraction](./examples/cutensor_mp_contraction.rs): a cuTENSORMp contraction with one process per rank, configured through `SINGE_CUTENSOR_MP_*` environment variables, with `--features mp`.
