mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::{Complex32, DataType},
    error::Result,
    frontend::{graph::Graph, operation::HeuristicMode},
};

fn named_tensor(name: &str, data_type: DataType, shape: Shape) -> TensorSpec {
    TensorSpec::new(data_type, shape).with_name(name)
}

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;

    let b = 16;
    let m = 32;
    let n = 64;
    let k = 128;

    let mut graph = Graph::new();
    let a = graph.tensor(named_tensor(
        "A",
        DataType::ComplexF32,
        Shape::contiguous([b, m, k])?.with_strides([m * k, k, 1])?,
    ));
    let b_tensor = graph.tensor(named_tensor(
        "B",
        DataType::ComplexF32,
        Shape::contiguous([b, k, n])?.with_strides([k * n, 1, k])?,
    ));
    let c = graph.tensor(named_tensor(
        "GEMM::C",
        DataType::ComplexF32,
        Shape::contiguous([b, m, n])?.with_strides([m * n, n, 1])?,
    ));
    graph.matmul(a, b_tensor, c, DataType::F32)?;

    let compiled = graph.compile(&ctx.cudnn, &[HeuristicMode::A])?;

    let mut a_dev = DeviceMemory::<Complex32>::zeroes((b * m * k) as usize)?;
    let mut b_dev = DeviceMemory::<Complex32>::zeroes((b * k * n) as usize)?;
    let mut c_dev = DeviceMemory::<Complex32>::zeroes((b * m * n) as usize)?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;
    let mut bindings = compiled.bindings();
    bindings.set(a, &mut a_dev)?;
    bindings.set(b_tensor, &mut b_dev)?;
    bindings.set(c, &mut c_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    let c_host = c_dev.copy_to_host_vec()?;
    let c_checksum = c_host.iter().map(|value| value.re + value.im).sum::<f32>();
    let c_nonzero = c_host
        .iter()
        .filter(|value| value.re != 0.0 || value.im != 0.0)
        .count();

    println!(
        "c: first={:?}, last={:?}, checksum={c_checksum}, nonzero={c_nonzero}, len={}",
        c_host[0],
        c_host[c_host.len() - 1],
        c_host.len()
    );

    println!("shape: {:?}", graph.shape(c)?.dimensions());
    println!("c_nonzero: {c_nonzero}");
    println!("c_checksum: {c_checksum}");

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_complex_fp32_matmul", run())
}
