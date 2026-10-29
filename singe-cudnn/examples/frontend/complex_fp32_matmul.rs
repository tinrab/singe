mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::{Complex32, DataType},
    error::Result,
    frontend::{
        graph::{Graph, TensorOutputConfig},
        operation::MatmulConfig,
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
};

fn init_complex_image(size: usize, seed: &mut u32) -> Vec<Complex32> {
    common::init_f32_image(size, seed)
        .into_iter()
        .map(|value| Complex32 {
            re: value,
            im: 0.5 - value,
        })
        .collect()
}

fn run() -> Result<()> {
    let example = common::ExampleContext::create()?;

    let b = 16;
    let m = 32;
    let n = 64;
    let k = 128;

    let mut graph = Graph::new();
    let a = graph.tensor(
        TensorSpec::new(DataType::ComplexF32, Shape::contiguous([b, m, k])?).with_name("A"),
    );
    let b_storage = Shape::contiguous([b, n, k])?;
    let strides = b_storage.strides();
    let b_tensor = graph.tensor(
        TensorSpec::new(
            DataType::ComplexF32,
            Shape::with_strides([b, k, n], [strides[0], strides[2], strides[1]])?,
        )
        .with_name("B"),
    );
    let c = graph.matmul(
        a,
        b_tensor,
        MatmulConfig::new()
            .with_compute_type(DataType::F32)
            .with_name("GEMM"),
        TensorOutputConfig::new()
            .with_data_type(DataType::ComplexF32)
            .output_tensor(),
    )?;
    let input_a_elements = graph.shape(a)?.storage_span()?;
    let input_b_elements = graph.shape(b_tensor)?.storage_span()?;
    let output_elements = graph.shape(c)?.storage_span()?;
    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A, HeuristicMode::Fallback]),
    )?;
    let cuda = example.cudnn.cuda_context();
    let mut seed = 123_456_789;
    let a_dev = DeviceMemory::<Complex32>::from_slice(
        cuda,
        &init_complex_image(input_a_elements, &mut seed),
    )?;
    let b_dev = DeviceMemory::<Complex32>::from_slice(
        cuda,
        &init_complex_image(input_b_elements, &mut seed),
    )?;
    let mut c_dev = DeviceMemory::<Complex32>::zeroed(cuda, output_elements)?;
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
        "executed complex fp32 matmul with output shape: {:?}",
        graph.shape(c)?.dimensions()
    );

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_complex_fp32_matmul", run())
}
