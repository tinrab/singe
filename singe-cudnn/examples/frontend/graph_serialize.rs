mod common;

use singe_core::assert_close;
use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::pointwise::PointwiseMode,
    backend::tensor::{Shape, TensorId, TensorSpec},
    data_type::{DataType, f16},
    error::Result,
    frontend::{
        graph::Graph,
        operation::{ConvolutionConfig, HeuristicMode, PointwiseOperation},
    },
    math::NanPropagation,
};

const X_ID: TensorId = TensorId::new(1100);
const W_ID: TensorId = TensorId::new(1101);
const SCALE_ID: TensorId = TensorId::new(1102);
const BIAS_ID: TensorId = TensorId::new(1103);
const Y_ID: TensorId = TensorId::new(1104);

fn init_f16_image(size: usize, seed: &mut u32) -> Vec<f16> {
    common::init_f32_image(size, seed)
        .into_iter()
        .map(f16::from_f32)
        .collect()
}

fn verify_output(y_host: &[f16]) {
    let first = y_host[0].to_f32();
    let last = y_host[y_host.len() - 1].to_f32();
    let checksum = y_host.iter().map(|value| value.to_f32()).sum::<f32>();
    let nonfinite = y_host
        .iter()
        .filter(|value| !value.to_f32().is_finite())
        .count();

    println!(
        "y: first={first}, last={last}, checksum={checksum}, nonfinite={nonfinite}, len={}",
        y_host.len()
    );

    assert_eq!(nonfinite, 0, "y.nonfinite mismatch");
    assert_close!(first, 22.15625, 1.0e-5, "y.first");
    assert_close!(last, 19.203125, 1.0e-5, "y.last");
    assert_close!(checksum, 2_305_199.0, 1.0, "y.checksum");
}

fn build_graph() -> Result<Graph> {
    let n = 4_i64;
    let c = 32_i64;
    let h = 16_i64;
    let w = 16_i64;
    let k = 64_i64;
    let r = 3_i64;
    let s = 3_i64;

    let mut graph = Graph::new()
        .with_name("serialize-demo")
        .with_io_data_type(DataType::F16)
        .with_intermediate_data_type(DataType::F32)
        .with_compute_data_type(DataType::F32);

    let x = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([n, c, h, w])?.with_strides([c * h * w, 1, c * w, c])?,
        )
        .with_id(X_ID),
    );
    let w_tensor = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([k, c, r, s])?.with_strides([c * r * s, 1, c * s, c])?,
        )
        .with_id(W_ID),
    );
    let conv = graph.convolution_forward_infer(
        x,
        w_tensor,
        ConvolutionConfig::new(DataType::F32, 2)
            .with_padding(vec![1, 1])
            .with_strides([1, 1])
            .with_dilations(vec![1, 1]),
    )?;
    let conv_virtual = graph.tensor_config(conv)?.clone().virtual_tensor();
    graph.replace_tensor(conv, conv_virtual)?;
    let scale = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([1, k, 1, 1])?.with_strides([k, 1, k, k])?,
        )
        .with_id(SCALE_ID),
    );
    let scaled = graph.pointwise_binary_infer(conv, scale, PointwiseMode::Mul, DataType::F32)?;
    let scaled_virtual = graph.tensor_config(scaled)?.clone().virtual_tensor();
    graph.replace_tensor(scaled, scaled_virtual)?;
    let bias = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([1, k, 1, 1])?.with_strides([k, 1, k, k])?,
        )
        .with_id(BIAS_ID),
    );
    let biased = graph.pointwise_binary_infer(scaled, bias, PointwiseMode::Add, DataType::F32)?;
    let biased_virtual = graph.tensor_config(biased)?.clone().virtual_tensor();
    graph.replace_tensor(biased, biased_virtual)?;
    let y = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([n, k, h, w])?.with_strides([k * h * w, 1, k * w, k])?,
        )
        .with_id(Y_ID),
    );
    graph.pointwise(PointwiseOperation::Unary {
        mode: PointwiseMode::ReluFwd,
        input: biased,
        output: y,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::NotPropagate,
        alpha1: 1.0,
        axis: None,
    });

    Ok(graph)
}

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;
    let graph = build_graph()?;

    let serialized = serde_json::to_vec(&graph)?;
    let restored = serde_json::from_slice::<Graph>(&serialized)?;
    assert_eq!(restored.name(), Some("serialize-demo"));
    let json = serde_json::to_string(&restored)?;
    let restored_from_json = serde_json::from_str::<Graph>(&json)?;
    assert_eq!(restored_from_json.key()?, restored.key()?);

    let compiled = restored.compile(&ctx.cudnn, &[HeuristicMode::A])?;
    let mut seed = 123_456_789_u32;
    let mut x_dev =
        DeviceMemory::from_slice(&init_f16_image((4 * 32 * 16 * 16) as usize, &mut seed))?;
    let mut w_dev =
        DeviceMemory::from_slice(&init_f16_image((64 * 32 * 3 * 3) as usize, &mut seed))?;
    let mut scale_dev = DeviceMemory::from_slice(&init_f16_image(64, &mut seed))?;
    let mut bias_dev = DeviceMemory::from_slice(&init_f16_image(64, &mut seed))?;
    let mut y_dev = DeviceMemory::<f16>::zeroes((4 * 64 * 16 * 16) as usize)?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;

    let mut bindings = compiled.bindings();
    bindings
        .set(X_ID, &mut x_dev)?
        .set(W_ID, &mut w_dev)?
        .set(SCALE_ID, &mut scale_dev)?
        .set(BIAS_ID, &mut bias_dev)?
        .set(Y_ID, &mut y_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    let y_host = y_dev.copy_to_host_vec()?;
    verify_output(&y_host);

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_graph_serialize", run())
}
