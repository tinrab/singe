mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::pointwise::PointwiseMode,
    backend::reduction::ReduceTensorOperator,
    backend::tensor::{Shape, TensorSpec},
    cudart_version,
    data_type::{DataType, f8e4m3},
    error::Result,
    frontend::{
        graph::Graph,
        operation::{ConvolutionConfig, HeuristicMode, PointwiseOperation, ReductionOperation},
    },
    math::NanPropagation,
    version,
};

fn named_tensor(name: &str, data_type: DataType, shape: Shape) -> TensorSpec {
    TensorSpec::new(data_type, shape).with_name(name)
}

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;

    if cudart_version()? < 12_000 {
        println!("frontend_convolution_fp8_fprop: skipped, requires CUDA toolkit 12.0+");
        return Ok(());
    }
    if version()? < 8_600 {
        println!("frontend_convolution_fp8_fprop: skipped, requires cuDNN 8.6.0+");
        return Ok(());
    }
    let properties = ctx.cudnn.cuda_context().device().properties()?;
    if properties.major < 9 {
        println!("frontend_convolution_fp8_fprop: skipped, requires Hopper or newer");
        return Ok(());
    }

    let n = 16_i64;
    let c = 128_i64;
    let h = 64_i64;
    let w = 64_i64;
    let k = 256_i64;
    let r = 1_i64;
    let s = 1_i64;

    let mut graph = Graph::new()
        .with_io_data_type(DataType::F16)
        .with_intermediate_data_type(DataType::F32)
        .with_compute_data_type(DataType::F32);

    let x = graph.tensor(named_tensor(
        "image",
        DataType::F8E4M3,
        Shape::contiguous([n, c, h, w])?.with_strides([c * h * w, 1, c * w, c])?,
    ));
    let w_tensor = graph.tensor(named_tensor(
        "filter",
        DataType::F8E4M3,
        Shape::contiguous([k, c, r, s])?.with_strides([c * r * s, 1, c * s, c])?,
    ));
    let conv_output = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([n, k, h, w])?.with_strides([k * h * w, 1, k * w, k])?,
        )
        .virtual_tensor(),
    );
    graph.convolution_forward(
        x,
        w_tensor,
        conv_output,
        ConvolutionConfig::new(DataType::F32, 2)
            .with_padding(vec![0, 0])
            .with_strides([1, 1])
            .with_dilations(vec![1, 1]),
    )?;

    let descale_x = graph.tensor(named_tensor(
        "descale_x",
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?.with_strides([1, 1, 1, 1])?,
    ));
    let descale_w = graph.tensor_like_named(descale_x, "descale_w")?;
    let scale_y = graph.tensor_like_named(descale_x, "scale_y")?;

    let after_descale_x = graph
        .tensor(TensorSpec::new(DataType::F32, graph.shape(conv_output)?.clone()).virtual_tensor());
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::Mul,
        lhs: conv_output,
        rhs: descale_x,
        output: after_descale_x,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });
    let after_descale_w = graph.tensor(
        TensorSpec::new(DataType::F32, graph.shape(after_descale_x)?.clone()).virtual_tensor(),
    );
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::Mul,
        lhs: after_descale_x,
        rhs: descale_w,
        output: after_descale_w,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });

    let y = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([n, k, h, w])?.with_strides([k * h * w, 1, k * w, k])?,
    ));
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::Mul,
        lhs: after_descale_w,
        rhs: scale_y,
        output: y,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });
    graph.mark_as_output(y)?;

    let absolute_max = graph.tensor_like_named(descale_x, "absolute_max")?;
    graph.reduction(ReductionOperation::Reduce {
        op: ReduceTensorOperator::AbsoluteMax,
        input: after_descale_w,
        output: absolute_max,
        compute_type: DataType::F32,
        is_deterministic: false,
    });
    graph.mark_as_output(absolute_max)?;

    let compiled = graph.compile(&ctx.cudnn, &[HeuristicMode::A])?;

    let mut x_dev = DeviceMemory::<f8e4m3>::zeroes((n * c * h * w) as usize)?;
    let mut w_dev = DeviceMemory::<f8e4m3>::zeroes((k * c * r * s) as usize)?;
    let mut y_dev = DeviceMemory::<f8e4m3>::zeroes((n * k * h * w) as usize)?;
    let mut descale_x_dev = DeviceMemory::from_slice(&[1.0_f32])?;
    let mut descale_w_dev = DeviceMemory::from_slice(&[1.0_f32])?;
    let mut scale_y_dev = DeviceMemory::from_slice(&[1.0_f32])?;
    let mut absolute_max_dev = DeviceMemory::from_slice(&[0.0_f32])?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;

    let mut bindings = compiled.bindings();
    bindings.set(x, &mut x_dev)?;
    bindings.set(w_tensor, &mut w_dev)?;
    bindings.set(descale_x, &mut descale_x_dev)?;
    bindings.set(descale_w, &mut descale_w_dev)?;
    bindings.set(scale_y, &mut scale_y_dev)?;
    bindings.set(y, &mut y_dev)?;
    bindings.set(absolute_max, &mut absolute_max_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_convolution_fp8_fprop", run())
}
