mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::pointwise::PointwiseMode,
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, f16},
    error::Result,
    frontend::{
        graph::{DataTypePolicy, Graph, GraphConfig},
        operation::{
            BatchNormalizationBackwardConfig, BatchNormalizationInferenceConfig, CompileConfig,
            HeuristicMode, PointwiseOperation,
        },
        plan::BuildPlanPolicy,
    },
    math::NanPropagation,
};

fn init_f16_image(size: usize, seed: &mut u32) -> Vec<f16> {
    common::init_f32_image(size, seed)
        .into_iter()
        .map(f16::from_f32)
        .collect()
}

fn count_nonfinite_f16(values: &[f16]) -> usize {
    values
        .iter()
        .filter(|value| !value.to_f32().is_finite())
        .count()
}

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;

    let n = 4;
    let c = 32;
    let h = 16;
    let w = 16;

    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F16)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );

    let x = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([n, c, h, w])?.with_strides([c * h * w, 1, c * w, c])?,
        )
        .with_name("X"),
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
    let mean = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
        )
        .with_name("mean"),
    );
    let inv_variance = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
        )
        .with_name("inv_variance"),
    );
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?.with_name("epsilon"));
    let bn_y = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([n, c, h, w])?.with_strides([c * h * w, 1, c * w, c])?,
        )
        .virtual_tensor(),
    );
    graph.batch_normalization_inference(
        x,
        mean,
        inv_variance,
        scale,
        bias,
        bn_y,
        BatchNormalizationInferenceConfig::new(epsilon),
    )?;

    let dy = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([n, c, h, w])?.with_strides([c * h * w, 1, c * w, c])?,
        )
        .with_name("DY"),
    );
    let dx_drelu = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([n, c, h, w])?.with_strides([c * h * w, 1, c * w, c])?,
        )
        .virtual_tensor(),
    );
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::ReluBwd,
        lhs: dy,
        rhs: bn_y,
        output: dx_drelu,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });
    let dx = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([n, c, h, w])?.with_strides([c * h * w, 1, c * w, c])?,
    ));
    let dscale = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
    ));
    let bias_gradient = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
    ));
    graph.batch_normalization_backward(
        x,
        mean,
        inv_variance,
        dx_drelu,
        scale,
        dscale,
        bias_gradient,
        dx,
        BatchNormalizationBackwardConfig::new(epsilon),
    )?;

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
    let mut x_dev =
        DeviceMemory::from_slice(&init_f16_image((n * c * h * w) as usize, &mut half_seed))?;
    let mut dy_dev =
        DeviceMemory::from_slice(&init_f16_image((n * c * h * w) as usize, &mut half_seed))?;
    let mut mean_dev =
        DeviceMemory::from_slice(&common::init_f32_image(c as usize, &mut float_seed))?;
    let mut inv_variance_dev =
        DeviceMemory::from_slice(&common::init_f32_image(c as usize, &mut float_seed))?;
    let mut scale_dev =
        DeviceMemory::from_slice(&common::init_f32_image(c as usize, &mut float_seed))?;
    let mut bias_dev =
        DeviceMemory::from_slice(&common::init_f32_image(c as usize, &mut float_seed))?;
    let mut dscale_dev =
        DeviceMemory::from_slice(&common::init_f32_image(c as usize, &mut float_seed))?;
    let mut bias_gradient_dev =
        DeviceMemory::from_slice(&common::init_f32_image(c as usize, &mut float_seed))?;
    let mut dx_dev =
        DeviceMemory::from_slice(&init_f16_image((n * c * h * w) as usize, &mut half_seed))?;
    let mut dx_drelu_dev =
        DeviceMemory::from_slice(&init_f16_image((n * c * h * w) as usize, &mut half_seed))?;
    let mut workspace = common::workspace(compiled.max_workspace_size()?)?;
    let mut bindings = compiled.bindings();
    bindings.set(x, &mut x_dev)?;
    bindings.set(dy, &mut dy_dev)?;
    bindings.set(scale, &mut scale_dev)?;
    bindings.set(bias, &mut bias_dev)?;
    bindings.set(mean, &mut mean_dev)?;
    bindings.set(inv_variance, &mut inv_variance_dev)?;
    bindings.set(dscale, &mut dscale_dev)?;
    bindings.set(bias_gradient, &mut bias_gradient_dev)?;
    bindings.set(dx, &mut dx_dev)?;
    bindings.set(dx_drelu, &mut dx_drelu_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    let dx_host = dx_dev.copy_to_host_vec()?;
    let dscale_host = dscale_dev.copy_to_host_vec()?;
    let bias_gradient_host = bias_gradient_dev.copy_to_host_vec()?;

    let dx_first = dx_host[0].to_f32();
    let dx_last = dx_host[dx_host.len() - 1].to_f32();
    let dx_checksum = dx_host.iter().map(|value| value.to_f32()).sum::<f32>();
    let dx_nonfinite = count_nonfinite_f16(&dx_host);
    let dscale_first = dscale_host[0];
    let dscale_last = dscale_host[dscale_host.len() - 1];
    let dscale_checksum = dscale_host.iter().copied().sum::<f32>();
    let dscale_nonfinite = common::count_nonfinite_f32(&dscale_host);
    let bias_gradient_first = bias_gradient_host[0];
    let bias_gradient_last = bias_gradient_host[bias_gradient_host.len() - 1];
    let bias_gradient_checksum = bias_gradient_host.iter().copied().sum::<f32>();
    let bias_gradient_nonfinite = common::count_nonfinite_f32(&bias_gradient_host);

    println!(
        "dx: first={dx_first}, last={dx_last}, checksum={dx_checksum}, nonfinite={dx_nonfinite}, len={}",
        dx_host.len()
    );
    println!(
        "dscale: first={dscale_first}, last={dscale_last}, checksum={dscale_checksum}, nonfinite={dscale_nonfinite}, len={}",
        dscale_host.len()
    );
    println!(
        "bias_gradient: first={bias_gradient_first}, last={bias_gradient_last}, checksum={bias_gradient_checksum}, nonfinite={bias_gradient_nonfinite}, len={}",
        bias_gradient_host.len()
    );

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_batch_norm_backward", run())
}
