# singe-cudnn

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cudnn.svg)](https://crates.io/crates/singe-cudnn)
[![Documentation](https://docs.rs/singe-cudnn/badge.svg)](https://docs.rs/singe-cudnn)
![License](https://img.shields.io/crates/l/singe-cudnn.svg)

Safe Rust wrappers for NVIDIA cuDNN deep neural network primitives, with a Rust port of the cuDNN Frontend graph API.

This crate provides:

- **Frontend graphs:** `frontend::graph::Graph` for matmul, convolution, normalization, pointwise, reduction, resampling, `RoPE`, and mixture-of-experts operations, with shape inference.
- **Attention:** `Graph::sdpa` and `Graph::sdpa_backward` in floating point, FP8, or MXFP8, with causal, sliding-window, `ALiBi`, paged-cache, and block masks.
- **Plans:** heuristics, candidate filtering, autotuning, plan caches, and serialization of compiled plans in `frontend::plan`.
- **Owned execution:** `operation::FrontendExecution` keeps a plan, its workspace, and its buffers alive, so it can be submitted on a stream or captured into a CUDA graph.
- **Backend and legacy APIs:** the backend descriptor and engine API in `backend`, causal depthwise convolutions, graph neural network aggregation, and the legacy CTC loss, dropout, LRN, softmax, and spatial transformer APIs.

## Example

Multiply two bf16 matrices with a frontend graph:

```rust,no_run
use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory};
use singe_cudnn::{
    context::Context,
    data_type::{DataType, bf16},
    frontend::{
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::MatmulConfig,
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
    tensor::{Shape, TensorId, TensorSpec},
};

fn main() -> anyhow::Result<()> {
    let cuda = CudaContext::create()?;
    let cudnn = Context::create(&cuda)?;
    let (m, n, k): (i64, i64, i64) = (32, 64, 128);

    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(DataTypePolicy::new().with_compute(DataType::F32)),
    );
    let a = graph.insert_tensor(
        TensorId::new(1),
        TensorSpec::new(DataType::Bf16, Shape::contiguous([1, m, k])?),
    )?;
    let b = graph.insert_tensor(
        TensorId::new(2),
        TensorSpec::new(DataType::Bf16, Shape::contiguous([1, k, n])?),
    )?;
    let c = graph.matmul(
        a,
        b,
        MatmulConfig::new(),
        TensorOutputConfig::new()
            .with_data_type(DataType::F32)
            .output_tensor(),
    )?;
    let plan = graph.compile(
        Some(&cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
    )?;

    let a_device = DeviceMemory::from_slice(&cuda, &vec![bf16::ONE; usize::try_from(m * k)?])?;
    let b_device = DeviceMemory::from_slice(&cuda, &vec![bf16::ONE; usize::try_from(k * n)?])?;
    let mut c_device = DeviceMemory::<f32>::zeroed(&cuda, usize::try_from(m * n)?)?;
    let mut workspace = DeviceMemory::<u8>::create_uninit(&cuda, plan.workspace_size())?;

    let mut bindings = plan.bindings()?;
    bindings.set_input(a, &a_device.view())?;
    bindings.set_input(b, &b_device.view())?;
    bindings.set(c, &mut c_device)?;

    // SAFETY: the inputs, the output, and the workspace stay alive and unchanged
    // until `cuda.synchronize` returns.
    unsafe {
        plan.execute_async(
            &cudnn,
            ExecuteArgs::new(&bindings).with_workspace(workspace.output()),
        )?;
    }
    cuda.synchronize()?;

    assert!(c_device.copy_to_host_vec()?.iter().all(|&value| value == 128.0));
    Ok(())
}
```

A `Context` owns a cuDNN handle in a `singe-cuda` context.
Building a graph and compiling a plan are safe; running a plan is `unsafe`, because its bindings record device addresses without borrowing the buffers.

`Graph::compile` asks cuDNN's heuristics for engine configurations and builds the first one that supports the graph on this GPU.
The handle starts on the legacy default stream; `Context::set_stream` installs another one.

## Executing plans

Functions named `*_async` are `unsafe` and only enqueue work: the plan and every buffer must stay valid until the work finishes, or until the last replay of a captured graph.
Their safe counterparts without the suffix wait for the stream before returning, but plan execution has no safe counterpart.
`operation::FrontendExecution` holds those resources instead.
`FrontendExecution::create` and `TunedPlan::prepare_owned`, both `unsafe`, allocate the workspace and build the variant pack up front.
The execution retains the cuDNN context, the compiled graph, the workspace, and the buffers registered with their owners through `Bindings::set_owned_region`.
Submit it with `Stream::submit`, or record it with `Stream::capture_owned`.

## More examples

The examples are ports of the cuDNN Frontend samples, registered as `frontend_<file stem>`, as in `cargo run -p singe-cudnn --example frontend_matmul`.

- [Matmul](./examples/frontend/matmul.rs): build, compile, and run a batched matrix multiplication graph.
- [Convolution](./examples/frontend/convolution_fprop.rs): forward convolution planning and execution.
- [Attention](./examples/frontend/fp16_sdpa.rs): FP16 scaled dot-product attention.
- [Paged attention](./examples/frontend/fp16_sdpa_paged_caches.rs): attention over paged key and value caches.
- [RMS normalization](./examples/frontend/rms_norm.rs): build a normalization graph.
- [Autotuning](./examples/frontend/autotuning.rs): time plan candidates and pick the fastest.
- [CUDA graphs](./examples/frontend/cudagraphs.rs): capture and replay execution.
- [Serialization](./examples/frontend/compiled_graph_serialize.rs): save and restore compiled plans.
