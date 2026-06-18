mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, f8e5m2, f8ue8m0, f16},
    error::Result,
    frontend::{
        graph::Graph,
        operation::{BlockScaleQuantizeConfig, HeuristicMode, LayerNormalizationConfig},
    },
};

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;

    let batch_size = 4_i64;
    let seq_length = 1024_i64;
    let hidden_size = 128_i64;
    let block_size = 32_i64;

    let mut graph = Graph::new()
        .with_io_data_type(DataType::F16)
        .with_intermediate_data_type(DataType::F32)
        .with_compute_data_type(DataType::F32);

    let x_layout = Shape::contiguous([batch_size, seq_length, hidden_size, 1])?.with_strides([
        seq_length * hidden_size,
        hidden_size,
        1,
        hidden_size,
    ])?;
    let scale_layout = Shape::contiguous([1, 1, hidden_size, 1])?.with_strides(vec![
        hidden_size,
        hidden_size,
        1,
        hidden_size,
    ])?;
    let x = graph.tensor(TensorSpec::new(DataType::F16, x_layout.clone()));
    let scale = graph.tensor(TensorSpec::new(DataType::F32, scale_layout.clone()));
    let bias = graph.tensor(TensorSpec::new(DataType::F32, scale_layout));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);

    let y_ln = graph.tensor(TensorSpec::new(DataType::F16, x_layout).virtual_tensor());
    let mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([batch_size, seq_length, 1, 1])?
            .with_strides([seq_length, 1, seq_length, seq_length])?,
    ));
    let inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([batch_size, seq_length, 1, 1])?
            .with_strides([seq_length, 1, seq_length, seq_length])?,
    ));
    graph.layer_normalization(
        x,
        scale,
        bias,
        y_ln,
        mean,
        inv_variance,
        LayerNormalizationConfig::training(epsilon),
    )?;

    let y_ln_2d = graph.tensor(
        TensorSpec::new(
            graph.tensor_config(y_ln)?.data_type,
            Shape::contiguous([batch_size * seq_length, hidden_size])?,
        )
        .virtual_tensor(),
    );
    graph.reshape(y_ln, y_ln_2d)?;

    let row = graph.block_scale_quantize_infer(
        y_ln_2d,
        BlockScaleQuantizeConfig::new(DataType::F32, block_size).with_axis(1),
    )?;
    let col = graph.block_scale_quantize_infer(
        y_ln_2d,
        BlockScaleQuantizeConfig::new(DataType::F32, block_size)
            .with_axis(0)
            .without_transpose(),
    )?;

    graph.mark_as_output(mean)?;
    graph.mark_as_output(inv_variance)?;
    let y_row = row.output;
    let mx_row = row.scale;
    let y_col = col.output;
    let mx_col = col.scale;
    let y_row_shape = graph.tensor_config(y_row)?.shape.clone();
    let y_row_tensor = TensorSpec::new(DataType::F8E5M2, y_row_shape).output_tensor();
    graph.replace_tensor(y_row, y_row_tensor)?;
    let mx_row_shape = graph.tensor_config(mx_row)?.shape.clone();
    let mx_row_tensor = TensorSpec::new(DataType::F8E8M0, mx_row_shape).output_tensor();
    graph.replace_tensor(mx_row, mx_row_tensor)?;
    let y_col_shape = graph.tensor_config(y_col)?.shape.clone();
    let y_col_tensor = TensorSpec::new(DataType::F8E5M2, y_col_shape).output_tensor();
    graph.replace_tensor(y_col, y_col_tensor)?;
    let mx_col_shape = graph.tensor_config(mx_col)?.shape.clone();
    let mx_col_tensor = TensorSpec::new(DataType::F8E8M0, mx_col_shape).output_tensor();
    graph.replace_tensor(mx_col, mx_col_tensor)?;

    let compiled = graph.compile(&ctx.cudnn, &[HeuristicMode::A, HeuristicMode::Fallback])?;
    let mut x_dev = DeviceMemory::<f16>::zeroes((batch_size * seq_length * hidden_size) as usize)?;
    let mut scale_dev = DeviceMemory::<f32>::zeroes(hidden_size as usize)?;
    let mut bias_dev = DeviceMemory::<f32>::zeroes(hidden_size as usize)?;
    let mut mean_dev = DeviceMemory::<f32>::zeroes((batch_size * seq_length) as usize)?;
    let mut inv_variance_dev = DeviceMemory::<f32>::zeroes((batch_size * seq_length) as usize)?;
    let mut y_row_dev =
        DeviceMemory::<f8e5m2>::zeroes((batch_size * seq_length * hidden_size) as usize)?;
    let mut mx_row_dev = DeviceMemory::<f8ue8m0>::zeroes(
        (batch_size * seq_length * hidden_size / block_size) as usize,
    )?;
    let mut y_col_dev =
        DeviceMemory::<f8e5m2>::zeroes((batch_size * seq_length * hidden_size) as usize)?;
    let mut mx_col_dev = DeviceMemory::<f8ue8m0>::zeroes(
        (batch_size * seq_length * hidden_size / block_size) as usize,
    )?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;
    let mut bindings = compiled.bindings();
    bindings.set(x, &mut x_dev)?;
    bindings.set(scale, &mut scale_dev)?;
    bindings.set(bias, &mut bias_dev)?;
    bindings.set(mean, &mut mean_dev)?;
    bindings.set(inv_variance, &mut inv_variance_dev)?;
    bindings.set(y_row, &mut y_row_dev)?;
    bindings.set(mx_row, &mut mx_row_dev)?;
    bindings.set(y_col, &mut y_col_dev)?;
    bindings.set(mx_col, &mut mx_col_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_norm_block_scale", run())
}
