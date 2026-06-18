mod common;

use singe_core::assert_close;
use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::pointwise::PointwiseMode,
    backend::tensor::{Shape, TensorSpec},
    data_type::DataType,
    error::Result,
    frontend::{
        graph::Graph,
        operation::{HeuristicMode, PointwiseOperation},
    },
    math::NanPropagation,
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
        DataType::I8,
        Shape::contiguous([b, m, k])?.with_strides([m * k, k, 1])?,
    ));
    let b_tensor = graph.tensor(named_tensor(
        "B",
        DataType::I8,
        Shape::contiguous([b, k, n])?.with_strides([k * n, 1, k])?,
    ));
    let bias = graph.tensor(named_tensor(
        "Bias",
        DataType::F32,
        Shape::contiguous([b, m, n])?.with_strides([m * n, n, 1])?,
    ));
    let c = graph.tensor(
        named_tensor(
            "GEMM::C",
            DataType::F32,
            Shape::contiguous([b, m, n])?.with_strides([m * n, n, 1])?,
        )
        .virtual_tensor(),
    );
    graph.matmul(a, b_tensor, c, DataType::I32)?;

    let c_after_add = graph.tensor(named_tensor(
        "pw1_add::OUT_0",
        DataType::F32,
        Shape::contiguous([b, m, n])?.with_strides([m * n, n, 1])?,
    ));
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::Add,
        lhs: c,
        rhs: bias,
        output: c_after_add,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });

    let compiled = graph.compile(&ctx.cudnn, &[HeuristicMode::A])?;

    let mut a_dev = DeviceMemory::<i8>::zeroes((b * m * k) as usize)?;
    let mut b_dev = DeviceMemory::<i8>::zeroes((b * k * n) as usize)?;
    let mut bias_dev = DeviceMemory::<f32>::zeroes((b * m * n) as usize)?;
    let mut c_dev = DeviceMemory::<f32>::zeroes((b * m * n) as usize)?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;
    let mut bindings = compiled.bindings();
    bindings.set(a, &mut a_dev)?;
    bindings.set(b_tensor, &mut b_dev)?;
    bindings.set(bias, &mut bias_dev)?;
    bindings.set(c_after_add, &mut c_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    let c_host = c_dev.copy_to_host_vec()?;
    let c_first = c_host[0];
    let c_last = c_host[c_host.len() - 1];
    let c_checksum = c_host.iter().sum::<f32>();
    let c_nonfinite = common::count_nonfinite_f32(&c_host);

    println!(
        "c: first={c_first}, last={c_last}, checksum={c_checksum}, nonfinite={c_nonfinite}, len={}",
        c_host.len()
    );

    assert_eq!(graph.shape(c_after_add)?.dimensions(), &[16, 32, 64]);
    assert_eq!(c_nonfinite, 0, "c.nonfinite mismatch");
    assert_close!(c_first, 0.0, 1.0e-6, "c.first");
    assert_close!(c_last, 0.0, 1.0e-6, "c.last");
    assert_close!(c_checksum, 0.0, 1.0e-6, "c.checksum");

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_int8_matmul", run())
}
