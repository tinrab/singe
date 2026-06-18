mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, bf16, f8e4m3},
    error::Result,
    frontend::{
        graph::{Graph, MatmulFp8Inputs},
        operation::HeuristicMode,
    },
};

fn named_tensor(name: &str, data_type: DataType, shape: Shape) -> TensorSpec {
    TensorSpec::new(data_type, shape).with_name(name)
}

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;

    let b = 16_i64;
    let m = 32_i64;
    let n = 64_i64;
    let k = 128_i64;

    let mut graph = Graph::new()
        .with_io_data_type(DataType::BF16)
        .with_intermediate_data_type(DataType::F32)
        .with_compute_data_type(DataType::F32);

    let a = graph.tensor(named_tensor(
        "A",
        DataType::F8E4M3,
        Shape::contiguous([b, m, k])?.with_strides([m * k, k, 1])?,
    ));
    let b_tensor = graph.tensor(named_tensor(
        "B",
        DataType::F8E4M3,
        Shape::contiguous([b, k, n])?.with_strides([k * n, 1, k])?,
    ));
    let descale_a = graph.tensor(named_tensor(
        "descale_a",
        DataType::F32,
        Shape::contiguous([1, 1, 1])?.with_strides([1, 1, 1])?,
    ));
    let descale_b = graph.tensor(named_tensor(
        "descale_b",
        DataType::F32,
        Shape::contiguous([1, 1, 1])?.with_strides([1, 1, 1])?,
    ));

    let c = graph.matmul_fp8_infer(
        MatmulFp8Inputs {
            a: a,
            b: b_tensor,
            descale_a: descale_a,
            descale_b: descale_b,
        },
        DataType::F32,
    )?;
    graph.mark_as_output(c)?;

    let compiled = graph.compile(&ctx.cudnn, &[HeuristicMode::A])?;

    let mut a_dev = DeviceMemory::<f8e4m3>::zeroes((b * m * k) as usize)?;
    let mut b_dev = DeviceMemory::<f8e4m3>::zeroes((b * k * n) as usize)?;
    let mut descale_a_dev = DeviceMemory::from_slice(&[1.0_f32])?;
    let mut descale_b_dev = DeviceMemory::from_slice(&[1.0_f32])?;
    let mut c_dev = DeviceMemory::<bf16>::zeroes((b * m * n) as usize)?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;

    let mut bindings = compiled.bindings();
    bindings.set(a, &mut a_dev)?;
    bindings.set(b_tensor, &mut b_dev)?;
    bindings.set(descale_a, &mut descale_a_dev)?;
    bindings.set(descale_b, &mut descale_b_dev)?;
    bindings.set(c, &mut c_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_fp8_matmul", run())
}
