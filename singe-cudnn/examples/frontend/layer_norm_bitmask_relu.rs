mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::pointwise::PointwiseMode,
    backend::tensor::{Shape, TensorSpec},
    data_type::DataType,
    error::Result,
    frontend::{
        graph::Graph,
        operation::{
            CompileConfig, HeuristicMode, LayerNormalizationBackwardConfig,
            LayerNormalizationConfig, PointwiseOperation,
        },
        plan::BuildPlanPolicy,
    },
    math::NanPropagation,
    version,
};

fn run_forward(ctx: &common::ExampleContext) -> Result<()> {
    let batch_size = 4_i64;
    let seq_length = 1024_i64;
    let hidden_size = 128_i64;
    let element_count = (batch_size * seq_length * hidden_size) as usize;

    let mut graph = Graph::new()
        .with_io_data_type(DataType::F32)
        .with_intermediate_data_type(DataType::F32)
        .with_compute_data_type(DataType::F32);

    let hidden_layout = Shape::contiguous([batch_size * seq_length, hidden_size, 1, 1])?
        .with_strides([hidden_size, 1, hidden_size, hidden_size])?;
    let reduction_layout =
        Shape::contiguous([batch_size * seq_length, 1, 1, 1])?.with_strides([1, 1, 1, 1])?;
    let scale_layout = Shape::contiguous([1, hidden_size, 1, 1])?.with_strides(vec![
        hidden_size,
        1,
        hidden_size,
        hidden_size,
    ])?;

    let x = graph.tensor(TensorSpec::new(DataType::F32, hidden_layout.clone()));
    let scale = graph.tensor(TensorSpec::new(DataType::F32, scale_layout.clone()));
    let bias = graph.tensor(TensorSpec::new(DataType::F32, scale_layout));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);
    let relu_lower_bound = graph.tensor(TensorSpec::scalar_f32(0.0)?);
    let relu_upper_bound = graph.tensor(TensorSpec::scalar_f32(6.0)?);

    let ln_output =
        graph.tensor(TensorSpec::new(DataType::F32, hidden_layout.clone()).virtual_tensor());
    let mean = graph.tensor(TensorSpec::new(DataType::F32, reduction_layout.clone()));
    let inv_variance = graph.tensor(TensorSpec::new(DataType::F32, reduction_layout));
    graph.layer_normalization(
        x,
        scale,
        bias,
        ln_output,
        mean,
        inv_variance,
        LayerNormalizationConfig::training(epsilon),
    )?;

    let y = graph.tensor(TensorSpec::new(DataType::F32, hidden_layout.clone()));
    graph.pointwise(PointwiseOperation::ReluForward {
        input: ln_output,
        output: y,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        lower_clip: 0.0,
        upper_clip: 6.0,
        lower_clip_slope: 0.0,
        axis: None,
    });

    let lower_mask =
        graph.tensor(TensorSpec::new(DataType::Boolean, hidden_layout.clone()).virtual_tensor());
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::CmpGt,
        lhs: y,
        rhs: relu_lower_bound,
        output: lower_mask,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });

    let upper_mask =
        graph.tensor(TensorSpec::new(DataType::Boolean, hidden_layout.clone()).virtual_tensor());
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::CmpLt,
        lhs: y,
        rhs: relu_upper_bound,
        output: upper_mask,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });

    let bitmask = graph.tensor(TensorSpec::new(DataType::Boolean, hidden_layout));
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::LogicalAnd,
        lhs: lower_mask,
        rhs: upper_mask,
        output: bitmask,
        compute_type: DataType::Boolean,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });

    for output in [y, bitmask, mean, inv_variance] {
        graph.mark_as_output(output)?;
    }

    let compiled = graph.compile_with_config(
        &ctx.cudnn,
        &CompileConfig::new()
            .with_heuristic_modes(vec![HeuristicMode::A, HeuristicMode::Fallback])
            .with_build_policy(BuildPlanPolicy::AllSupported),
    )?;
    let mut x_dev = DeviceMemory::<f32>::zeroes(element_count)?;
    let mut scale_dev = DeviceMemory::<f32>::zeroes(hidden_size as usize)?;
    let mut bias_dev = DeviceMemory::<f32>::zeroes(hidden_size as usize)?;
    let mut y_dev = DeviceMemory::<f32>::zeroes(element_count)?;
    let mut mean_dev = DeviceMemory::<f32>::zeroes((batch_size * seq_length) as usize)?;
    let mut inv_variance_dev = DeviceMemory::<f32>::zeroes((batch_size * seq_length) as usize)?;
    let mut bitmask_dev = DeviceMemory::<bool>::zeroes(element_count)?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;

    let mut bindings = compiled.bindings();
    bindings.set(x, &mut x_dev)?;
    bindings.set(scale, &mut scale_dev)?;
    bindings.set(bias, &mut bias_dev)?;
    bindings.set(y, &mut y_dev)?;
    bindings.set(mean, &mut mean_dev)?;
    bindings.set(inv_variance, &mut inv_variance_dev)?;
    bindings.set(bitmask, &mut bitmask_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    Ok(())
}

#[allow(dead_code)]
fn run_backward(ctx: &common::ExampleContext) -> Result<()> {
    let batch_size = 4_i64;
    let seq_length = 1024_i64;
    let hidden_size = 128_i64;
    let element_count = (batch_size * seq_length * hidden_size) as usize;

    let mut graph = Graph::new()
        .with_io_data_type(DataType::F32)
        .with_intermediate_data_type(DataType::F32)
        .with_compute_data_type(DataType::F32);

    let hidden_layout = Shape::contiguous([batch_size * seq_length, hidden_size, 1, 1])?
        .with_strides([hidden_size, 1, hidden_size, hidden_size])?;
    let reduction_layout =
        Shape::contiguous([batch_size * seq_length, 1, 1, 1])?.with_strides([1, 1, 1, 1])?;
    let scale_layout = Shape::contiguous([1, hidden_size, 1, 1])?.with_strides(vec![
        hidden_size,
        1,
        hidden_size,
        hidden_size,
    ])?;

    let x = graph.tensor(TensorSpec::new(DataType::F32, hidden_layout.clone()));
    let dy = graph.tensor(TensorSpec::new(DataType::F32, hidden_layout.clone()));
    let scale = graph.tensor(TensorSpec::new(DataType::F32, scale_layout.clone()));
    let mean = graph.tensor(TensorSpec::new(DataType::F32, reduction_layout.clone()));
    let inv_variance = graph.tensor(TensorSpec::new(DataType::F32, reduction_layout));
    let mask = graph.tensor(TensorSpec::new(DataType::Boolean, hidden_layout.clone()));

    let masked_dy =
        graph.tensor(TensorSpec::new(DataType::F32, hidden_layout.clone()).virtual_tensor());
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::Mul,
        lhs: dy,
        rhs: mask,
        output: masked_dy,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });

    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);
    let dx = graph.tensor(TensorSpec::new(DataType::F32, hidden_layout.clone()));
    let dscale = graph.tensor(TensorSpec::new(DataType::F32, scale_layout.clone()));
    let bias_gradient = graph.tensor(TensorSpec::new(DataType::F32, scale_layout));
    graph.layer_normalization_backward(
        x,
        mean,
        inv_variance,
        masked_dy,
        scale,
        dscale,
        bias_gradient,
        dx,
        LayerNormalizationBackwardConfig::new(epsilon),
    )?;

    for output in [dx, dscale, bias_gradient] {
        graph.mark_as_output(output)?;
    }

    let compiled = graph.compile(&ctx.cudnn, &[HeuristicMode::A])?;
    let mut x_dev = DeviceMemory::<f32>::zeroes(element_count)?;
    let mut dy_dev = DeviceMemory::<f32>::zeroes(element_count)?;
    let mut scale_dev = DeviceMemory::<f32>::zeroes(hidden_size as usize)?;
    let mut mean_dev = DeviceMemory::<f32>::zeroes((batch_size * seq_length) as usize)?;
    let mut inv_variance_dev = DeviceMemory::<f32>::zeroes((batch_size * seq_length) as usize)?;
    let mut mask_dev = DeviceMemory::<bool>::zeroes(element_count)?;
    let mut dx_dev = DeviceMemory::<f32>::zeroes(element_count)?;
    let mut dscale_dev = DeviceMemory::<f32>::zeroes(hidden_size as usize)?;
    let mut bias_gradient_dev = DeviceMemory::<f32>::zeroes(hidden_size as usize)?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;

    let mut bindings = compiled.bindings();
    bindings.set(x, &mut x_dev)?;
    bindings.set(dy, &mut dy_dev)?;
    bindings.set(scale, &mut scale_dev)?;
    bindings.set(mean, &mut mean_dev)?;
    bindings.set(inv_variance, &mut inv_variance_dev)?;
    bindings.set(mask, &mut mask_dev)?;
    bindings.set(dx, &mut dx_dev)?;
    bindings.set(dscale, &mut dscale_dev)?;
    bindings.set(bias_gradient, &mut bias_gradient_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    Ok(())
}

fn run() -> Result<()> {
    if version()? < 91300 {
        println!("skipping layer_norm_bitmask_relu: requires cuDNN >= 9.13.0");
        return Ok(());
    }

    let ctx = common::ExampleContext::create()?;
    run_forward(&ctx)?;
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_layer_norm_bitmask_relu", run())
}
