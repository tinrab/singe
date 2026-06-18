mod common;

use singe_core::assert_close;
use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::pointwise::PointwiseMode,
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, f16},
    error::Result,
    frontend::{
        graph::Graph,
        operation::{
            CompileConfig, ConvolutionConfig, DbnWeightConfig, HeuristicMode, PointwiseOperation,
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

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;

    let n = 4;
    let c = 32;
    let h = 16;
    let w = 16;
    let k = 64;
    let r = 3;
    let s = 3;

    let mut graph = Graph::new();
    let dy = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([n, k, h, w])?.with_strides([k * h * w, 1, k * w, k])?,
    ));
    let weight = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([k, c, r, s])?.with_strides([c * r * s, 1, c * s, c])?,
    ));
    let dgrad = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([n, c, h, w])?.with_strides([c * h * w, 1, c * w, c])?,
        )
        .virtual_tensor(),
    );
    graph.convolution_dgrad(
        weight,
        dy,
        dgrad,
        ConvolutionConfig::new(DataType::F32, 2)
            .with_padding(vec![1, 1])
            .with_strides([1, 1])
            .with_dilations(vec![1, 1]),
    )?;

    let x = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([n, c, h, w])?.with_strides([c * h * w, 1, c * w, c])?,
    ));
    let mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
    ));
    let centered = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([n, c, h, w])?.with_strides([c * h * w, 1, c * w, c])?,
        )
        .virtual_tensor(),
    );
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::Sub,
        lhs: x,
        rhs: mean,
        output: centered,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });

    let inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
    ));
    let scaled_variance = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([n, c, h, w])?.with_strides([c * h * w, 1, c * w, c])?,
        )
        .virtual_tensor(),
    );
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::Mul,
        lhs: centered,
        rhs: inv_variance,
        output: scaled_variance,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });

    let scale = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
    ));
    let scaled = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([n, c, h, w])?.with_strides([c * h * w, 1, c * w, c])?,
        )
        .virtual_tensor(),
    );
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::Mul,
        lhs: scaled_variance,
        rhs: scale,
        output: scaled,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });

    let bias = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
    ));
    let bn_output = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([n, c, h, w])?.with_strides([c * h * w, 1, c * w, c])?,
        )
        .virtual_tensor(),
    );
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::Add,
        lhs: scaled,
        rhs: bias,
        output: bn_output,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });

    let drelu = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([n, c, h, w])?.with_strides([c * h * w, 1, c * w, c])?,
        )
        .output_tensor(),
    );
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::ReluBwd,
        lhs: dgrad,
        rhs: bn_output,
        output: drelu,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });

    let channel_shape = Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?;
    let dscale = graph.tensor(TensorSpec::new(DataType::F32, channel_shape.clone()));
    let bias_gradient = graph.tensor(TensorSpec::new(DataType::F32, channel_shape.clone()));
    let eq_scale_dy = graph.tensor(TensorSpec::new(DataType::F32, channel_shape.clone()));
    let eq_scale_x = graph.tensor(TensorSpec::new(DataType::F32, channel_shape.clone()));
    let eq_bias = graph.tensor(TensorSpec::new(DataType::F32, channel_shape.clone()));
    graph.dbn_weight(
        drelu,
        x,
        scale,
        mean,
        inv_variance,
        dscale,
        bias_gradient,
        eq_bias,
        eq_scale_dy,
        eq_scale_x,
        DbnWeightConfig::new(),
    )?;

    let compiled = graph.compile_with_config(
        &ctx.cudnn,
        &CompileConfig::new()
            .with_heuristic_modes(vec![HeuristicMode::A])
            .with_build_policy(BuildPlanPolicy::AllSupported),
    )?;

    let mut half_seed = 123_456_789_u32;
    let mut float_seed = 123_456_789_u32;
    let mut dy_dev =
        DeviceMemory::from_slice(&init_f16_image((n * k * h * w) as usize, &mut half_seed))?;
    let mut weight_dev =
        DeviceMemory::from_slice(&init_f16_image((k * c * r * s) as usize, &mut half_seed))?;
    let mut x_dev =
        DeviceMemory::from_slice(&init_f16_image((n * c * h * w) as usize, &mut half_seed))?;
    let mut mean_dev =
        DeviceMemory::from_slice(&common::init_f32_image(c as usize, &mut float_seed))?;
    let mut inv_variance_dev =
        DeviceMemory::from_slice(&common::init_f32_image(c as usize, &mut float_seed))?;
    let mut scale_dev =
        DeviceMemory::from_slice(&common::init_f32_image(c as usize, &mut float_seed))?;
    let mut bias_dev =
        DeviceMemory::from_slice(&common::init_f32_image(c as usize, &mut float_seed))?;
    let mut drelu_dev = DeviceMemory::<f16>::create((n * c * h * w) as usize)?;
    let mut dscale_dev = DeviceMemory::<f32>::create(c as usize)?;
    let mut bias_gradient_dev = DeviceMemory::<f32>::create(c as usize)?;
    let mut eq_scale_dy_dev = DeviceMemory::<f32>::create(c as usize)?;
    let mut eq_scale_x_dev = DeviceMemory::<f32>::create(c as usize)?;
    let mut eq_bias_dev = DeviceMemory::<f32>::create(c as usize)?;
    let mut workspace = common::workspace(compiled.max_workspace_size()?)?;
    let mut bindings = compiled.bindings();
    bindings.set(dy, &mut dy_dev)?;
    bindings.set(weight, &mut weight_dev)?;
    bindings.set(x, &mut x_dev)?;
    bindings.set(mean, &mut mean_dev)?;
    bindings.set(inv_variance, &mut inv_variance_dev)?;
    bindings.set(scale, &mut scale_dev)?;
    bindings.set(bias, &mut bias_dev)?;
    bindings.set(drelu, &mut drelu_dev)?;
    bindings.set(dscale, &mut dscale_dev)?;
    bindings.set(bias_gradient, &mut bias_gradient_dev)?;
    bindings.set(eq_scale_dy, &mut eq_scale_dy_dev)?;
    bindings.set(eq_scale_x, &mut eq_scale_x_dev)?;
    bindings.set(eq_bias, &mut eq_bias_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    let drelu_host = drelu_dev.copy_to_host_vec()?;
    let dscale_host = dscale_dev.copy_to_host_vec()?;
    let bias_gradient_host = bias_gradient_dev.copy_to_host_vec()?;
    let eq_scale_dy_host = eq_scale_dy_dev.copy_to_host_vec()?;
    let eq_scale_x_host = eq_scale_x_dev.copy_to_host_vec()?;
    let eq_bias_host = eq_bias_dev.copy_to_host_vec()?;

    let drelu_first = drelu_host[0].to_f32();
    let drelu_last = drelu_host[drelu_host.len() - 1].to_f32();
    let drelu_checksum = drelu_host.iter().map(|value| value.to_f32()).sum::<f32>();
    let drelu_nonfinite = drelu_host
        .iter()
        .filter(|value| !value.to_f32().is_finite())
        .count();
    let dscale_first = dscale_host[0];
    let dscale_last = dscale_host[dscale_host.len() - 1];
    let dscale_checksum = dscale_host.iter().copied().sum::<f32>();
    let dscale_nonfinite = common::count_nonfinite_f32(&dscale_host);
    let bias_gradient_first = bias_gradient_host[0];
    let bias_gradient_last = bias_gradient_host[bias_gradient_host.len() - 1];
    let bias_gradient_checksum = bias_gradient_host.iter().copied().sum::<f32>();
    let bias_gradient_nonfinite = common::count_nonfinite_f32(&bias_gradient_host);
    let eq_scale_dy_first = eq_scale_dy_host[0];
    let eq_scale_dy_last = eq_scale_dy_host[eq_scale_dy_host.len() - 1];
    let eq_scale_dy_checksum = eq_scale_dy_host.iter().copied().sum::<f32>();
    let eq_scale_dy_nonfinite = common::count_nonfinite_f32(&eq_scale_dy_host);
    let eq_scale_x_first = eq_scale_x_host[0];
    let eq_scale_x_last = eq_scale_x_host[eq_scale_x_host.len() - 1];
    let eq_scale_x_checksum = eq_scale_x_host.iter().copied().sum::<f32>();
    let eq_scale_x_nonfinite = common::count_nonfinite_f32(&eq_scale_x_host);
    let eq_bias_first = eq_bias_host[0];
    let eq_bias_last = eq_bias_host[eq_bias_host.len() - 1];
    let eq_bias_checksum = eq_bias_host.iter().copied().sum::<f32>();
    let eq_bias_nonfinite = common::count_nonfinite_f32(&eq_bias_host);

    println!(
        "drelu: first={drelu_first}, last={drelu_last}, checksum={drelu_checksum}, nonfinite={drelu_nonfinite}, len={}",
        drelu_host.len()
    );
    println!(
        "dscale: first={dscale_first}, last={dscale_last}, checksum={dscale_checksum}, nonfinite={dscale_nonfinite}, len={}",
        dscale_host.len()
    );
    println!(
        "bias_gradient: first={bias_gradient_first}, last={bias_gradient_last}, checksum={bias_gradient_checksum}, nonfinite={bias_gradient_nonfinite}, len={}",
        bias_gradient_host.len()
    );
    println!(
        "eq_scale_dy: first={eq_scale_dy_first}, last={eq_scale_dy_last}, checksum={eq_scale_dy_checksum}, nonfinite={eq_scale_dy_nonfinite}, len={}",
        eq_scale_dy_host.len()
    );
    println!(
        "eq_scale_x: first={eq_scale_x_first}, last={eq_scale_x_last}, checksum={eq_scale_x_checksum}, nonfinite={eq_scale_x_nonfinite}, len={}",
        eq_scale_x_host.len()
    );
    println!(
        "eq_bias: first={eq_bias_first}, last={eq_bias_last}, checksum={eq_bias_checksum}, nonfinite={eq_bias_nonfinite}, len={}",
        eq_bias_host.len()
    );

    assert_eq!(drelu_nonfinite, 0, "drelu.nonfinite mismatch");
    assert_eq!(dscale_nonfinite, 0, "dscale.nonfinite mismatch");
    assert_eq!(
        bias_gradient_nonfinite, 0,
        "bias_gradient.nonfinite mismatch"
    );
    assert_eq!(eq_scale_dy_nonfinite, 0, "eq_scale_dy.nonfinite mismatch");
    assert_eq!(eq_scale_x_nonfinite, 0, "eq_scale_x.nonfinite mismatch");
    assert_eq!(eq_bias_nonfinite, 0, "eq_bias.nonfinite mismatch");
    assert_close!(drelu_first, 68.625, 1.0e-5, "drelu.first");
    assert_close!(drelu_last, 59.625, 1.0e-5, "drelu.last");
    assert_close!(drelu_checksum, 4_083_867.0, 1.0, "drelu.checksum");
    assert_close!(dscale_first, 12_104.965, 1.0e-3, "dscale.first");
    assert_close!(dscale_last, -1_824.679_8, 1.0e-4, "dscale.last");
    assert_close!(dscale_checksum, -107_746.86, 1.0e-2, "dscale.checksum");
    assert_close!(
        bias_gradient_first,
        136_830.88,
        1.0e-2,
        "bias_gradient.first"
    );
    assert_close!(bias_gradient_last, 135_042.03, 1.0e-2, "bias_gradient.last");
    assert_close!(
        bias_gradient_checksum,
        4_083_878.3,
        1.0,
        "bias_gradient.checksum"
    );
    assert_close!(eq_scale_dy_first, 0.03843227, 1.0e-7, "eq_scale_dy.first");
    assert_close!(eq_scale_dy_last, 0.058281034, 1.0e-7, "eq_scale_dy.last");
    assert_close!(
        eq_scale_dy_checksum,
        7.927416,
        1.0e-6,
        "eq_scale_dy.checksum",
    );
    assert_close!(eq_scale_x_first, -0.08895411, 1.0e-7, "eq_scale_x.first");
    assert_close!(eq_scale_x_last, 0.047740966, 1.0e-7, "eq_scale_x.last");
    assert_close!(
        eq_scale_x_checksum,
        22.829971,
        1.0e-6,
        "eq_scale_x.checksum",
    );
    assert_close!(eq_bias_first, -5.1306696, 1.0e-6, "eq_bias.first");
    assert_close!(eq_bias_last, -7.710958, 1.0e-6, "eq_bias.last");
    assert_close!(eq_bias_checksum, -968.6393, 1.0e-4, "eq_bias.checksum");

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_dbn_weight", run())
}
