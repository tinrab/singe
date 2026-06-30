mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, bf16},
    error::Result,
    frontend::{
        graph::{DataTypePolicy, Graph, GraphConfig},
        operation::{AdaptiveLayerNormalizationConfig, HeuristicMode},
    },
};

#[derive(Clone, Copy)]
struct AdaptiveLayerNormShape {
    b: i64,
    s: i64,
    d: i64,
}

const ADAPTIVE_LAYER_NORM_SHAPES: [AdaptiveLayerNormShape; 4] = [
    AdaptiveLayerNormShape {
        b: 4,
        s: 1024,
        d: 128,
    },
    AdaptiveLayerNormShape {
        b: 8,
        s: 1024,
        d: 128,
    },
    AdaptiveLayerNormShape {
        b: 4,
        s: 512,
        d: 128,
    },
    AdaptiveLayerNormShape {
        b: 8,
        s: 512,
        d: 128,
    },
];

fn max_x_volume() -> usize {
    ADAPTIVE_LAYER_NORM_SHAPES
        .iter()
        .map(|shape| (shape.b * shape.s * shape.d) as usize)
        .max()
        .unwrap_or(0)
}

fn max_stats_volume() -> usize {
    ADAPTIVE_LAYER_NORM_SHAPES
        .iter()
        .map(|shape| (shape.b * shape.s) as usize)
        .max()
        .unwrap_or(0)
}

fn max_weights_volume() -> usize {
    ADAPTIVE_LAYER_NORM_SHAPES
        .iter()
        .map(|shape| (shape.b * shape.d) as usize)
        .max()
        .unwrap_or(0)
}

fn init_f32_image(size: usize, seed: &mut u32) -> Vec<f32> {
    common::init_f32_image(size, seed)
}

fn init_bf16_image(size: usize, seed: &mut u32) -> Vec<bf16> {
    init_f32_image(size, seed)
        .into_iter()
        .map(bf16::from_f32)
        .collect()
}

fn count_nonfinite_bf16(values: &[bf16]) -> usize {
    values
        .iter()
        .filter(|value| !value.to_f32().is_finite())
        .count()
}

fn count_nonfinite_f32(values: &[f32]) -> usize {
    common::count_nonfinite_f32(values)
}

fn run_shape(
    ctx: &common::ExampleContext,
    shape: AdaptiveLayerNormShape,
    max_x_volume: usize,
    max_stats_volume: usize,
    max_weights_volume: usize,
) -> Result<(Vec<bf16>, Vec<f32>, Vec<f32>)> {
    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::BF16)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );

    let x = graph.tensor(
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([shape.b, shape.s, shape.d])?,
        )
        .with_name("X"),
    );
    let scale = graph.tensor(
        TensorSpec::new(DataType::F32, Shape::contiguous([shape.b, 1, shape.d])?)
            .with_name("scale"),
    );
    let bias = graph.tensor(
        TensorSpec::new(DataType::F32, Shape::contiguous([shape.b, 1, shape.d])?).with_name("bias"),
    );
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?.with_name("epsilon"));
    let y = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([shape.b, shape.s, shape.d])?,
    ));
    let mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([shape.b, shape.s, 1])?,
    ));
    let inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([shape.b, shape.s, 1])?,
    ));
    graph.adaptive_layer_normalization(
        x,
        scale,
        y,
        mean,
        inv_variance,
        AdaptiveLayerNormalizationConfig::training(epsilon).with_bias(bias),
    )?;

    let compiled = graph.compile(&ctx.cudnn, &[HeuristicMode::A, HeuristicMode::Fallback])?;

    let mut bf16_seed = 123_456_789_u32;
    let mut float_seed = 123_456_789_u32;
    let mut x_dev = DeviceMemory::from_slice(&init_bf16_image(max_x_volume, &mut bf16_seed))?;
    let mut scale_dev =
        DeviceMemory::from_slice(&init_f32_image(max_weights_volume, &mut float_seed))?;
    let mut bias_dev =
        DeviceMemory::from_slice(&init_f32_image(max_weights_volume, &mut float_seed))?;
    let mut y_dev = DeviceMemory::from_slice(&init_bf16_image(max_x_volume, &mut bf16_seed))?;
    let mut mean_dev =
        DeviceMemory::from_slice(&init_f32_image(max_stats_volume, &mut float_seed))?;
    let mut inv_variance_dev =
        DeviceMemory::from_slice(&init_f32_image(max_stats_volume, &mut float_seed))?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;
    let mut bindings = compiled.bindings();
    bindings.set(x, &mut x_dev)?;
    bindings.set(scale, &mut scale_dev)?;
    bindings.set(bias, &mut bias_dev)?;
    bindings.set(y, &mut y_dev)?;
    bindings.set(mean, &mut mean_dev)?;
    bindings.set(inv_variance, &mut inv_variance_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    Ok((
        y_dev.copy_to_host_vec()?,
        mean_dev.copy_to_host_vec()?,
        inv_variance_dev.copy_to_host_vec()?,
    ))
}

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;
    let max_x_volume = max_x_volume();
    let max_stats_volume = max_stats_volume();
    let max_weights_volume = max_weights_volume();

    for (shape_index, shape) in ADAPTIVE_LAYER_NORM_SHAPES.iter().copied().enumerate() {
        let (y_host, mean_host, inv_variance_host) = run_shape(
            &ctx,
            shape,
            max_x_volume,
            max_stats_volume,
            max_weights_volume,
        )?;

        let y_first = y_host[0].to_f32();
        let y_last = y_host[y_host.len() - 1].to_f32();
        let y_checksum = y_host.iter().map(|value| value.to_f32()).sum::<f32>();
        let y_nonfinite = count_nonfinite_bf16(&y_host);
        let mean_first = mean_host[0];
        let mean_last = mean_host[mean_host.len() - 1];
        let mean_checksum = mean_host.iter().copied().sum::<f32>();
        let mean_nonfinite = count_nonfinite_f32(&mean_host);
        let inv_variance_first = inv_variance_host[0];
        let inv_variance_last = inv_variance_host[inv_variance_host.len() - 1];
        let inv_variance_checksum = inv_variance_host.iter().copied().sum::<f32>();
        let inv_variance_nonfinite = count_nonfinite_f32(&inv_variance_host);

        println!(
            "shape[{shape_index}]: b={}, s={}, d={}, y_nonfinite={}, mean_nonfinite={}, inv_variance_nonfinite={}",
            shape.b, shape.s, shape.d, y_nonfinite, mean_nonfinite, inv_variance_nonfinite
        );

        if shape_index == 0 {
            println!(
                "y: first={y_first}, last={y_last}, checksum={y_checksum}, nonfinite={y_nonfinite}, len={}",
                y_host.len()
            );
            println!(
                "mean: first={mean_first}, last={mean_last}, checksum={mean_checksum}, nonfinite={mean_nonfinite}, len={}",
                mean_host.len()
            );
            println!(
                "inv_variance: first={inv_variance_first}, last={inv_variance_last}, checksum={inv_variance_checksum}, nonfinite={inv_variance_nonfinite}, len={}",
                inv_variance_host.len()
            );
        }
    }

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_adaptive_layer_norm", run())
}
