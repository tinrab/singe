mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::{
        pointwise::PointwiseMode,
        tensor::{Shape, TensorSpec},
    },
    data_type::DataType,
    error::Result,
    frontend::{
        graph::{Graph, TensorOutputConfig},
        operation::{MatmulConfig, PointwiseOperation},
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
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
    let b_storage = Shape::contiguous([b, n, k])?;
    let strides = b_storage.strides();
    let b_tensor = graph.tensor(
        TensorSpec::new(
            DataType::I8,
            Shape::with_strides([b, k, n], [strides[0], strides[2], strides[1]])?,
        )
        .with_name("B"),
    );
    let bias = graph
        .tensor(TensorSpec::new(DataType::F32, Shape::contiguous([b, m, n])?).with_name("Bias"));
    let c = graph.matmul(
        a,
        b_tensor,
        MatmulConfig::new()
            .with_compute_type(DataType::I32)
            .with_name("GEMM"),
        TensorOutputConfig::new()
            .with_data_type(DataType::F32)
            .virtual_tensor(),
    )?;
    let c_after_add = graph.pointwise(
        PointwiseOperation::binary(
            PointwiseMode::Add,
            c,
            bias,
            TensorOutputConfig::new()
                .with_data_type(DataType::F32)
                .output_tensor(),
        )
        .with_compute_type(DataType::F32)
        .with_name("pw1_add"),
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
    let b_dev = DeviceMemory::<i8>::from_slice(
        cuda,
        &common::init_i8_image(graph.shape(b_tensor)?.storage_span()?, &mut seed),
    )?;
    let bias_dev = DeviceMemory::<f32>::from_slice(
        cuda,
        &common::init_f32_image(graph.shape(bias)?.storage_span()?, &mut seed),
    )?;
    let mut c_dev = DeviceMemory::<f32>::zeroed(cuda, graph.shape(c_after_add)?.storage_span()?)?;
    let mut workspace = common::workspace(cuda, compiled.workspace_size())?;
    let mut bindings = compiled.bindings()?;
    bindings.set_input(a, &a_dev)?;
    bindings.set_input(b_tensor, &b_dev)?;
    bindings.set_input(bias, &bias_dev)?;
    bindings.set(c_after_add, &mut c_dev)?;
    // SAFETY: initialized inputs and bias, writable output and workspace remain live through stream completion.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;

    println!(
        "executed int8 matmul and fp32 bias addition with output shape: {:?}",
        graph.shape(c_after_add)?.dimensions()
    );

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_int8_matmul", run())
}
