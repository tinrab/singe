mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::pointwise::PointwiseMode,
    backend::reduction::ReduceTensorOperator,
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, f8e4m3, f16},
    error::Result,
    frontend::{
        graph::{DataTypePolicy, Graph, GraphConfig},
        operation::{HeuristicMode, PointwiseOperation, ReductionOperation},
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

    run_reduction(&ctx)?;
    run_fused_scalar(&ctx)?;
    run_fused_absolute_max_type_conversion(&ctx)?;

    Ok(())
}

fn run_reduction(ctx: &common::ExampleContext) -> Result<()> {
    let n = 64_i64;

    let mut graph = Graph::new();
    let a = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([n, n, n, n])?.with_strides([n * n * n, 1, n * n, n])?,
    ));
    let c = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?.with_strides([1, 1, 1, 1])?,
    ));
    graph.reduction(ReductionOperation::Reduce {
        op: ReduceTensorOperator::Max,
        input: a,
        output: c,
        compute_type: DataType::F32,
        is_deterministic: false,
    });
    graph.mark_as_output(c)?;

    let compiled = graph.compile(&ctx.cudnn, &[HeuristicMode::A])?;

    let mut seed = 123_456_789_u32;
    let mut a_dev =
        DeviceMemory::from_slice(&common::init_f32_image((n * n * n * n) as usize, &mut seed))?;
    let mut c_dev = DeviceMemory::<f32>::zeroes(1)?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;
    let mut bindings = compiled.bindings();
    bindings.set(a, &mut a_dev)?;
    bindings.set(c, &mut c_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    let _ = c_dev.copy_to_host_vec()?;

    Ok(())
}

fn run_fused_scalar(ctx: &common::ExampleContext) -> Result<()> {
    let n = 4_i64;

    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F16)
                .with_compute(DataType::F32),
        ),
    );

    let a = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([n, n, n])?,
    ));
    let scalar = graph.tensor(TensorSpec::scalar_f32(5.0)?);
    let c = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([n, n, n])?,
    ));
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::Add,
        lhs: a,
        rhs: scalar,
        output: c,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });
    graph.mark_as_output(c)?;

    let compiled = graph.compile(&ctx.cudnn, &[HeuristicMode::A])?;

    let mut seed = 123_456_789_u32;
    let mut a_dev = DeviceMemory::from_slice(&init_f16_image((n * n * n) as usize, &mut seed))?;
    let mut c_dev = DeviceMemory::<f16>::zeroes((n * n * n) as usize)?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;
    let mut bindings = compiled.bindings();
    bindings.set(a, &mut a_dev)?;
    bindings.set(c, &mut c_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    let _ = c_dev.copy_to_host_vec()?;

    Ok(())
}

fn run_fused_absolute_max_type_conversion(ctx: &common::ExampleContext) -> Result<()> {
    let n = 64_i64;

    let mut graph = Graph::new();
    let a = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([n, n, n, n])?.with_strides([n * n * n, 1, n * n, n])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?.with_strides([1, 1, 1, 1])?,
    ));

    let absolute_max = graph.tensor_like_named(scale, "absolute_max")?;
    graph.reduction(ReductionOperation::Reduce {
        op: ReduceTensorOperator::AbsoluteMax,
        input: a,
        output: absolute_max,
        compute_type: DataType::F32,
        is_deterministic: false,
    });
    graph.mark_as_output(absolute_max)?;

    let c = graph.pointwise_binary_infer(a, scale, PointwiseMode::Mul, DataType::F32)?;
    let c_shape = graph.shape(c)?.clone();
    graph.replace_tensor(
        c,
        TensorSpec::new(DataType::F8E4M3, c_shape).output_tensor(),
    )?;

    let compiled = graph.compile(&ctx.cudnn, &[HeuristicMode::A])?;

    let mut seed = 123_456_789_u32;
    let mut a_dev =
        DeviceMemory::from_slice(&common::init_f32_image((n * n * n * n) as usize, &mut seed))?;
    let mut scale_dev = DeviceMemory::from_slice(&[1.0_f32])?;
    let mut absolute_max_dev = DeviceMemory::<f32>::zeroes(1)?;
    let mut c_dev = DeviceMemory::<f8e4m3>::zeroes((n * n * n * n) as usize)?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;
    let mut bindings = compiled.bindings();
    bindings.set(a, &mut a_dev)?;
    bindings.set(scale, &mut scale_dev)?;
    bindings.set(absolute_max, &mut absolute_max_dev)?;
    bindings.set(c, &mut c_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    let _ = absolute_max_dev.copy_to_host_vec()?;
    let _ = c_dev.copy_to_host_vec()?;

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_pointwise", run())
}
