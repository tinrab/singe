mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::pointwise::PointwiseMode,
    backend::tensor::{Shape, TensorId, TensorSpec},
    data_type::{DataType, f16},
    error::Result,
    frontend::{
        graph::Graph,
        operation::{HeuristicMode, PointwiseOperation},
    },
    math::NanPropagation,
};

const A_ID: TensorId = TensorId::new(1001);
const B_ID: TensorId = TensorId::new(1002);
const C_ID: TensorId = TensorId::new(1003);

fn init_f16_image(size: usize, seed: &mut u32) -> Vec<f16> {
    common::init_f32_image(size, seed)
        .into_iter()
        .map(f16::from_f32)
        .collect()
}

fn build_graph() -> Result<Graph> {
    let b_start = 1_i64;
    let b = 8_i64;
    let b_end = 2_i64;
    let b_actual = b_start + b + b_end;

    let m_start = 3_i64;
    let m = 16_i64;
    let m_end = 4_i64;
    let m_actual = m_start + m + m_end;

    let n = 32_i64;
    let k = 64_i64;

    let mut graph = Graph::new()
        .with_io_data_type(DataType::F16)
        .with_compute_data_type(DataType::F32)
        .with_name("slice-gemm");

    let a = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([b_actual, m_actual, k])?.with_strides(vec![m_actual * k, k, 1])?,
        )
        .with_id(A_ID),
    );
    let a_slice = graph.slice_infer(a, &[b_start..(b_start + b), m_start..(m_start + m), 0..k])?;
    let b_tensor = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([b, k, n])?.with_strides([k * n, n, 1])?,
        )
        .with_id(B_ID),
    );
    let c0 = graph
        .tensor(TensorSpec::new(DataType::F32, Shape::contiguous([b, m, n])?).virtual_tensor());
    graph.matmul(a_slice, b_tensor, c0, DataType::F32)?;

    let c =
        graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([b, m, n])?).with_id(C_ID));
    graph.pointwise(PointwiseOperation::Unary {
        mode: PointwiseMode::ReluFwd,
        input: c0,
        output: c,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::NotPropagate,
        alpha1: 1.0,
        axis: None,
    });
    graph.mark_as_output(c)?;

    Ok(graph)
}

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;
    let graph = build_graph()?;

    let serialized = serde_json::to_vec(&graph)?;
    let restored = serde_json::from_slice::<Graph>(&serialized)?;
    assert_eq!(restored.key()?, graph.key()?);

    let compiled = restored.compile(&ctx.cudnn, &[HeuristicMode::A])?;

    let mut seed = 123_456_789_u32;
    let mut a_dev = DeviceMemory::from_slice(&init_f16_image((11 * 23 * 64) as usize, &mut seed))?;
    let mut b_dev = DeviceMemory::from_slice(&init_f16_image((8 * 64 * 32) as usize, &mut seed))?;
    let mut c_dev = DeviceMemory::<f32>::zeroes((8 * 16 * 32) as usize)?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;

    let mut bindings = compiled.bindings();
    bindings.set(A_ID, &mut a_dev)?;
    bindings.set(B_ID, &mut b_dev)?;
    bindings.set(C_ID, &mut c_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    let c_host = c_dev.copy_to_host_vec()?;
    println!(
        "c: first={}, last={}, checksum={}, nonfinite={}, len={}",
        c_host[0],
        c_host[c_host.len() - 1],
        c_host.iter().copied().sum::<f32>(),
        common::count_nonfinite_f32(&c_host),
        c_host.len()
    );

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_slice", run())
}
