mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::{
        pointwise::PointwiseMode,
        tensor::{Shape, TensorSpec},
    },
    data_type::{DataType, bf16},
    error::Result,
    frontend::{
        graph::{Graph, TensorOutputConfig},
        operation::{MatmulConfig, PointwiseOperation},
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
    math::NanPropagation,
};

fn run() -> Result<()> {
    let example = common::ExampleContext::create()?;

    let b = 16;
    let m = 32;
    let n = 64;
    let k = 128;

    let mut graph = Graph::new();
    let a =
        graph.tensor(TensorSpec::new(DataType::I8, Shape::contiguous([b, m, k])?).with_name("A"));
    let b_tensor =
        graph.tensor(TensorSpec::new(DataType::BF16, Shape::contiguous([b, k, n])?).with_name("B"));
    let a_casted = graph.pointwise(
        PointwiseOperation::unary(
            PointwiseMode::Identity,
            a,
            TensorOutputConfig::new()
                .with_data_type(DataType::BF16)
                .virtual_tensor(),
        )
        .with_compute_type(DataType::F32)
        .with_nan_propagation(NanPropagation::NotPropagate)
        .with_name("Cast_A"),
    )?;
    let c = graph.matmul(
        a_casted,
        b_tensor,
        MatmulConfig::new()
            .with_compute_type(DataType::F32)
            .with_name("GEMM"),
        TensorOutputConfig::new()
            .with_data_type(DataType::BF16)
            .output_tensor(),
    )?;

    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A, HeuristicMode::Fallback]),
    )?;
    let cuda = example.cudnn.cuda_context();
    let mut seed = 123_456_789;
    let a_dev = DeviceMemory::<i8>::from_slice(
        cuda,
        &common::init_i8_image(graph.shape(a)?.storage_span()?, &mut seed),
    )?;
    let b_values = common::init_f32_image(graph.shape(b_tensor)?.storage_span()?, &mut seed)
        .into_iter()
        .map(bf16::from_f32)
        .collect::<Vec<_>>();
    let b_dev = DeviceMemory::<bf16>::from_slice(cuda, &b_values)?;
    let mut c_dev = DeviceMemory::<bf16>::zeroed(cuda, graph.shape(c)?.storage_span()?)?;
    let mut workspace = common::workspace(cuda, compiled.workspace_size())?;
    let mut bindings = compiled.bindings()?;
    bindings.set_input(a, &a_dev)?;
    bindings.set_input(b_tensor, &b_dev)?;
    bindings.set(c, &mut c_dev)?;
    // SAFETY: initialized inputs, writable output and workspace remain live through stream completion.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;

    println!(
        "executed int8-to-bf16 mixed matmul with output shape: {:?}",
        graph.shape(c)?.dimensions()
    );

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_mixed_matmul", run())
}
