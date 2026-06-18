mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::pointwise::PointwiseMode,
    backend::tensor::{Shape, TensorSpec},
    data_type::DataType,
    data_type::f16 as half_f16,
    error::Result,
    frontend::{
        graph::Graph,
        operation::{
            BatchNormalizationConfig, BatchNormalizationRunningStats, CompileConfig, HeuristicMode,
            PointwiseOperation,
        },
        plan::BuildPlanPolicy,
    },
    math::NanPropagation,
    scalar::ScalarValue,
};

fn init_f32_image(size: usize, seed: &mut u32) -> Vec<f32> {
    let mut values = Vec::with_capacity(size);
    for _ in 0..size {
        *seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        values.push((*seed as f32) * 2.328_306_4e-10_f32);
    }
    values
}

fn init_f16_image(size: usize, seed: &mut u32) -> Vec<half_f16> {
    init_f32_image(size, seed)
        .into_iter()
        .map(half_f16::from_f32)
        .collect()
}

fn init_peer_stats_0(size: usize, seed: &mut u32) -> Vec<f32> {
    let mut values = init_f32_image(size, seed);
    for index in (1..size).step_by(2) {
        values[index] = f32::from_bits(1);
    }
    values
}

fn count_nonfinite_f16(values: &[half_f16]) -> usize {
    values
        .iter()
        .filter(|value| !value.to_f32().is_finite())
        .count()
}

fn count_nonfinite_f32(values: &[f32]) -> usize {
    values.iter().filter(|value| !value.is_finite()).count()
}

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;

    let n = 4;
    let c = 32;
    let h = 16;
    let w = 16;

    let mut graph = Graph::new()
        .with_io_data_type(DataType::F16)
        .with_intermediate_data_type(DataType::F32)
        .with_compute_data_type(DataType::F32);

    let x = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([n, c, h, w])?.with_strides([c * h * w, 1, c * w, c])?,
        )
        .with_name("X"),
    );
    let prev_running_mean = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
        )
        .with_name("prev_running_mean"),
    );
    let prev_running_var = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
        )
        .with_name("prev_running_var"),
    );
    let scale = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
        )
        .with_name("scale"),
    );
    let bias = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
        )
        .with_name("bias"),
    );
    let peer_stats_0 = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, n * c, 1, 1])?.with_strides([n * c, 1, n * c, n * c])?,
    ));
    let peer_stats_1 = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, n * c, 1, 1])?.with_strides([n * c, 1, n * c, n * c])?,
    ));
    let epsilon = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([1, 1, 1, 1])?.with_strides([1, 1, 1, 1])?,
        )
        .with_scalar_value(ScalarValue::F32(1e-5))?
        .with_name("epsilon"),
    );
    let momentum = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([1, 1, 1, 1])?.with_strides([1, 1, 1, 1])?,
        )
        .with_scalar_value(ScalarValue::F32(1e-1))?
        .with_name("momentum"),
    );
    let y_bn = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([n, c, h, w])?.with_strides([c * h * w, 1, c * w, c])?,
        )
        .virtual_tensor(),
    );
    let mean = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
        )
        .virtual_tensor()
        .output_tensor(),
    );
    let inv_variance = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
        )
        .virtual_tensor()
        .output_tensor(),
    );
    let next_running_mean = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
        )
        .virtual_tensor()
        .output_tensor(),
    );
    let next_running_var = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
        )
        .virtual_tensor()
        .output_tensor(),
    );
    graph.batch_normalization(
        x,
        scale,
        bias,
        y_bn,
        mean,
        inv_variance,
        BatchNormalizationConfig::new(epsilon)
            .with_peer_stats(vec![peer_stats_0, peer_stats_1])
            .with_running_stats(BatchNormalizationRunningStats::new(
                momentum,
                prev_running_mean,
                prev_running_var,
                next_running_mean,
                next_running_var,
            )),
    )?;

    let a = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([n, c, h, w])?.with_strides([c * h * w, 1, c * w, c])?,
        )
        .with_name("A"),
    );
    let add_output = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([n, c, h, w])?.with_strides([c * h * w, 1, c * w, c])?,
        )
        .virtual_tensor(),
    );
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::Add,
        lhs: y_bn,
        rhs: a,
        output: add_output,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });
    let y = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([n, c, h, w])?.with_strides([c * h * w, 1, c * w, c])?,
        )
        .virtual_tensor()
        .output_tensor(),
    );
    graph.pointwise(PointwiseOperation::Unary {
        mode: PointwiseMode::ReluFwd,
        input: add_output,
        output: y,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        axis: None,
    });

    let compiled = match graph.compile_with_config(
        &ctx.cudnn,
        &CompileConfig::new()
            .with_heuristic_modes(vec![HeuristicMode::A, HeuristicMode::Fallback])
            .with_build_policy(BuildPlanPolicy::AllSupported),
    ) {
        Ok(compiled) => compiled,
        Err(error) => return Err(error),
    };

    let mut half_seed = 123_456_789_u32;
    let mut float_seed = 123_456_789_u32;
    let mut peer_stats_seed = 123_456_789_u32;
    let mut x_dev =
        DeviceMemory::from_slice(&init_f16_image((n * c * h * w) as usize, &mut half_seed))?;
    let mut mean_dev = DeviceMemory::from_slice(&init_f32_image(c as usize, &mut float_seed))?;
    let mut inv_variance_dev =
        DeviceMemory::from_slice(&init_f32_image(c as usize, &mut float_seed))?;
    let mut prev_running_mean_dev =
        DeviceMemory::from_slice(&init_f32_image(c as usize, &mut float_seed))?;
    let mut prev_running_var_dev =
        DeviceMemory::from_slice(&init_f32_image(c as usize, &mut float_seed))?;
    let mut next_running_mean_dev =
        DeviceMemory::from_slice(&init_f32_image(c as usize, &mut float_seed))?;
    let mut next_running_var_dev =
        DeviceMemory::from_slice(&init_f32_image(c as usize, &mut float_seed))?;
    let mut scale_dev = DeviceMemory::from_slice(&init_f32_image(c as usize, &mut float_seed))?;
    let mut bias_dev = DeviceMemory::from_slice(&init_f32_image(c as usize, &mut float_seed))?;
    let mut a_dev =
        DeviceMemory::from_slice(&init_f16_image((n * c * h * w) as usize, &mut half_seed))?;
    let mut y_dev =
        DeviceMemory::from_slice(&init_f16_image((n * c * h * w) as usize, &mut half_seed))?;
    let mut peer_stats_0_dev = DeviceMemory::from_slice(&init_peer_stats_0(
        (2 * n * c) as usize,
        &mut peer_stats_seed,
    ))?;
    let mut peer_stats_1_dev =
        DeviceMemory::from_slice(&init_f32_image((2 * n * c) as usize, &mut peer_stats_seed))?;
    let mut workspace = common::workspace(compiled.max_workspace_size()?)?;
    let mut bindings = compiled.bindings();
    bindings.set(x, &mut x_dev)?;
    bindings.set(mean, &mut mean_dev)?;
    bindings.set(inv_variance, &mut inv_variance_dev)?;
    bindings.set(scale, &mut scale_dev)?;
    bindings.set(bias, &mut bias_dev)?;
    bindings.set(a, &mut a_dev)?;
    bindings.set(y, &mut y_dev)?;
    bindings.set(peer_stats_0, &mut peer_stats_0_dev)?;
    bindings.set(peer_stats_1, &mut peer_stats_1_dev)?;
    bindings.set(prev_running_mean, &mut prev_running_mean_dev)?;
    bindings.set(prev_running_var, &mut prev_running_var_dev)?;
    bindings.set(next_running_mean, &mut next_running_mean_dev)?;
    bindings.set(next_running_var, &mut next_running_var_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    let y_host = y_dev.copy_to_host_vec()?;
    let mean_host = mean_dev.copy_to_host_vec()?;
    let inv_variance_host = inv_variance_dev.copy_to_host_vec()?;
    let next_running_mean_host = next_running_mean_dev.copy_to_host_vec()?;
    let next_running_var_host = next_running_var_dev.copy_to_host_vec()?;

    let y_first = y_host[0].to_f32();
    let y_last = y_host[y_host.len() - 1].to_f32();
    let y_checksum = y_host.iter().map(|value| value.to_f32()).sum::<f32>();
    let y_nonfinite = count_nonfinite_f16(&y_host);
    let mean_first = mean_host[0];
    let mean_last = mean_host[mean_host.len() - 1];
    let mean_checksum = mean_host.iter().copied().sum::<f32>();
    let mean_nonfinite = count_nonfinite_f32(&mean_host);
    let inv_variance_first = inv_variance_host[0];
    let inv_variance_last = inv_variance_host[inv_variance_host.len() - 1];
    let inv_variance_checksum = inv_variance_host.iter().copied().sum::<f32>();
    let inv_variance_nonfinite = count_nonfinite_f32(&inv_variance_host);
    let next_running_mean_first = next_running_mean_host[0];
    let next_running_mean_last = next_running_mean_host[next_running_mean_host.len() - 1];
    let next_running_mean_checksum = next_running_mean_host.iter().copied().sum::<f32>();
    let next_running_mean_nonfinite = count_nonfinite_f32(&next_running_mean_host);
    let next_running_var_first = next_running_var_host[0];
    let next_running_var_last = next_running_var_host[next_running_var_host.len() - 1];
    let next_running_var_checksum = next_running_var_host.iter().copied().sum::<f32>();
    let next_running_var_nonfinite = count_nonfinite_f32(&next_running_var_host);

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
    println!(
        "next_running_mean: first={next_running_mean_first}, last={next_running_mean_last}, checksum={next_running_mean_checksum}, nonfinite={next_running_mean_nonfinite}, len={}",
        next_running_mean_host.len()
    );
    println!(
        "next_running_var: first={next_running_var_first}, last={next_running_var_last}, checksum={next_running_var_checksum}, nonfinite={next_running_var_nonfinite}, len={}",
        next_running_var_host.len()
    );

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_batch_norm", run())
}
