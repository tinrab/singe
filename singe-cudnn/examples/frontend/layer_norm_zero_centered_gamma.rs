mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::pointwise::PointwiseMode,
    backend::tensor::{Shape, TensorId, TensorSpec},
    data_type::{DataType, f16},
    error::Result,
    frontend::{
        graph::Graph,
        operation::{HeuristicMode, LayerNormalizationBackwardConfig, LayerNormalizationConfig},
    },
    math::NanPropagation,
};

const BATCH_SIZE: i64 = 4;
const SEQ_LENGTH: i64 = 1024;
const HIDDEN_SIZE: i64 = 128;

fn hidden_layout() -> Result<Shape> {
    Shape::contiguous([BATCH_SIZE * SEQ_LENGTH, HIDDEN_SIZE, 1, 1])?.with_strides(vec![
        HIDDEN_SIZE,
        1,
        HIDDEN_SIZE,
        HIDDEN_SIZE,
    ])
}

fn hidden_reduction_layout() -> Result<Shape> {
    Shape::contiguous([BATCH_SIZE * SEQ_LENGTH, 1, 1, 1])?.with_strides([1, 1, 1, 1])
}

fn scale_layout() -> Result<Shape> {
    Shape::contiguous([1, HIDDEN_SIZE, 1, 1])?.with_strides(vec![
        HIDDEN_SIZE,
        1,
        HIDDEN_SIZE,
        HIDDEN_SIZE,
    ])
}

fn zero_centered_scale(graph: &mut Graph) -> Result<(TensorId, TensorId)> {
    let scale_zero_centered = graph.tensor(TensorSpec::new(DataType::F32, scale_layout()?));
    let one = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let scale = graph.pointwise_binary_infer(
        scale_zero_centered,
        one,
        PointwiseMode::Add,
        DataType::F32,
    )?;
    let scale_tensor = graph.tensor_config(scale)?.clone().virtual_tensor();
    graph.replace_tensor(scale, scale_tensor)?;
    Ok((scale_zero_centered, scale))
}

fn run_training(ctx: &common::ExampleContext) -> Result<()> {
    let mut graph = Graph::new()
        .with_io_data_type(DataType::F16)
        .with_intermediate_data_type(DataType::F32)
        .with_compute_data_type(DataType::F32);

    let x = graph.tensor(TensorSpec::new(DataType::F16, hidden_layout()?));
    let (scale_zero_centered, scale) = zero_centered_scale(&mut graph)?;
    let bias = graph.tensor(TensorSpec::new(DataType::F32, scale_layout()?));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);

    let y = graph.tensor(TensorSpec::new(DataType::F16, hidden_layout()?));
    let mean =
        graph.tensor(TensorSpec::new(DataType::F32, hidden_reduction_layout()?).virtual_tensor());
    let inv_variance =
        graph.tensor(TensorSpec::new(DataType::F32, hidden_reduction_layout()?).virtual_tensor());
    graph.layer_normalization(
        x,
        scale,
        bias,
        y,
        mean,
        inv_variance,
        LayerNormalizationConfig::training(epsilon),
    )?;

    for output in [y, mean, inv_variance] {
        graph.mark_as_output(output)?;
    }

    let compiled = graph.compile(&ctx.cudnn, &[HeuristicMode::A, HeuristicMode::Fallback])?;
    let mut x_dev = DeviceMemory::<f16>::zeroes((BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE) as usize)?;
    let mut scale_zero_centered_dev = DeviceMemory::<f32>::zeroes(HIDDEN_SIZE as usize)?;
    let mut bias_dev = DeviceMemory::<f32>::zeroes(HIDDEN_SIZE as usize)?;
    let mut y_dev = DeviceMemory::<f16>::zeroes((BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE) as usize)?;
    let mut mean_dev = DeviceMemory::<f32>::zeroes((BATCH_SIZE * SEQ_LENGTH) as usize)?;
    let mut inv_variance_dev = DeviceMemory::<f32>::zeroes((BATCH_SIZE * SEQ_LENGTH) as usize)?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;

    let mut bindings = compiled.bindings();
    bindings.set(x, &mut x_dev)?;
    bindings.set(scale_zero_centered, &mut scale_zero_centered_dev)?;
    bindings.set(bias, &mut bias_dev)?;
    bindings.set(y, &mut y_dev)?;
    bindings.set(mean, &mut mean_dev)?;
    bindings.set(inv_variance, &mut inv_variance_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))
}

#[allow(dead_code)]
fn run_inference(ctx: &common::ExampleContext) -> Result<()> {
    let mut graph = Graph::new()
        .with_io_data_type(DataType::F16)
        .with_intermediate_data_type(DataType::F32)
        .with_compute_data_type(DataType::F32);

    let x = graph.tensor(TensorSpec::new(DataType::F16, hidden_layout()?));
    let (scale_zero_centered, scale) = zero_centered_scale(&mut graph)?;
    let bias = graph.tensor(TensorSpec::new(DataType::F32, scale_layout()?));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);

    let y = graph.tensor(TensorSpec::new(DataType::F16, hidden_layout()?));
    let mean = graph.tensor(TensorSpec::new(DataType::F32, hidden_reduction_layout()?));
    let inv_variance = graph.tensor(TensorSpec::new(DataType::F32, hidden_reduction_layout()?));
    graph.layer_normalization(
        x,
        scale,
        bias,
        y,
        mean,
        inv_variance,
        LayerNormalizationConfig::inference(epsilon),
    )?;
    graph.mark_as_output(y)?;

    let compiled = graph.compile(&ctx.cudnn, &[HeuristicMode::A, HeuristicMode::Fallback])?;
    println!(
        "inference ids: x={}, scale_zero_centered={}, bias={}, y={}, mean={}, inv_variance={}",
        x.as_i64(),
        scale_zero_centered.as_i64(),
        bias.as_i64(),
        y.as_i64(),
        mean.as_i64(),
        inv_variance.as_i64(),
    );
    let mut x_dev = DeviceMemory::<f16>::zeroes((BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE) as usize)?;
    let mut scale_zero_centered_dev = DeviceMemory::<f32>::zeroes(HIDDEN_SIZE as usize)?;
    let mut bias_dev = DeviceMemory::<f32>::zeroes(HIDDEN_SIZE as usize)?;
    let mut y_dev = DeviceMemory::<f16>::zeroes((BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE) as usize)?;
    let mut mean_dev = DeviceMemory::<f32>::zeroes((BATCH_SIZE * SEQ_LENGTH) as usize)?;
    let mut inv_variance_dev = DeviceMemory::<f32>::zeroes((BATCH_SIZE * SEQ_LENGTH) as usize)?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;

    let mut bindings = compiled.bindings();
    bindings.set(x, &mut x_dev)?;
    bindings.set(scale_zero_centered, &mut scale_zero_centered_dev)?;
    bindings.set(bias, &mut bias_dev)?;
    bindings.set(y, &mut y_dev)?;
    bindings.set(mean, &mut mean_dev)?;
    bindings.set(inv_variance, &mut inv_variance_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))
}

#[allow(dead_code)]
fn run_backward(ctx: &common::ExampleContext) -> Result<()> {
    let mut graph = Graph::new()
        .with_io_data_type(DataType::F16)
        .with_intermediate_data_type(DataType::F32)
        .with_compute_data_type(DataType::F32);

    let x = graph.tensor(TensorSpec::new(DataType::F16, hidden_layout()?));
    let dy = graph.tensor(TensorSpec::new(DataType::F16, hidden_layout()?));
    let (scale_zero_centered, scale) = zero_centered_scale(&mut graph)?;
    let mean = graph.tensor(TensorSpec::new(DataType::F32, hidden_reduction_layout()?));
    let inv_variance = graph.tensor(TensorSpec::new(DataType::F32, hidden_reduction_layout()?));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);

    let outputs = graph.layer_normalization_backward_infer(
        x,
        mean,
        inv_variance,
        dy,
        scale,
        LayerNormalizationBackwardConfig::new(epsilon),
    )?;
    let dx = outputs.dx;
    let dscale = outputs.dscale;
    let bias_gradient = outputs.bias_gradient;

    for output in [dx, dscale, bias_gradient] {
        graph.mark_as_output(output)?;
    }

    let compiled = graph.compile(&ctx.cudnn, &[HeuristicMode::A, HeuristicMode::Fallback])?;
    println!(
        "backward ids: x={}, dy={}, scale_zero_centered={}, mean={}, inv_variance={}, dx={}, dscale={}, bias_gradient={}",
        x.as_i64(),
        dy.as_i64(),
        scale_zero_centered.as_i64(),
        mean.as_i64(),
        inv_variance.as_i64(),
        dx.as_i64(),
        dscale.as_i64(),
        bias_gradient.as_i64(),
    );
    let mut x_dev = DeviceMemory::<f16>::zeroes((BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE) as usize)?;
    let mut dy_dev = DeviceMemory::<f16>::zeroes((BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE) as usize)?;
    let mut scale_zero_centered_dev = DeviceMemory::<f32>::zeroes(HIDDEN_SIZE as usize)?;
    let mut mean_dev = DeviceMemory::<f32>::zeroes((BATCH_SIZE * SEQ_LENGTH) as usize)?;
    let mut inv_variance_dev = DeviceMemory::<f32>::zeroes((BATCH_SIZE * SEQ_LENGTH) as usize)?;
    let mut dx_dev = DeviceMemory::<f16>::zeroes((BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE) as usize)?;
    let mut dscale_dev = DeviceMemory::<f32>::zeroes(HIDDEN_SIZE as usize)?;
    let mut bias_gradient_dev = DeviceMemory::<f32>::zeroes(HIDDEN_SIZE as usize)?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;

    let mut bindings = compiled.bindings();
    bindings.set(x, &mut x_dev)?;
    bindings.set(dy, &mut dy_dev)?;
    bindings.set(scale_zero_centered, &mut scale_zero_centered_dev)?;
    bindings.set(mean, &mut mean_dev)?;
    bindings.set(inv_variance, &mut inv_variance_dev)?;
    bindings.set(dx, &mut dx_dev)?;
    bindings.set(dscale, &mut dscale_dev)?;
    bindings.set(bias_gradient, &mut bias_gradient_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))
}

fn run() -> Result<()> {
    let _ = NanPropagation::Propagate;
    let ctx = common::ExampleContext::create()?;
    run_training(&ctx)
}

fn main() -> Result<()> {
    common::finish("frontend_layer_norm_zero_centered_gamma", run())
}
