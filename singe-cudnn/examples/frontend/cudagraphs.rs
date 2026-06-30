mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::behavior::BackendBehaviorNote,
    backend::pointwise::PointwiseMode,
    backend::tensor::{Shape, TensorId, TensorSpec},
    data_type::{DataType, f16},
    error::{Error, Result, Status},
    frontend::{
        bindings::Bindings,
        graph::{DataTypePolicy, Graph, GraphConfig},
        operation::{HeuristicMode, MatmulConfig},
        plan::CompiledGraph,
    },
    scalar::ScalarValue,
};

fn build_graph() -> Result<(Graph, TensorId, TensorId, TensorId, TensorId)> {
    let b = 8_i64;
    let m = 32_i64;
    let n = 16_i64;
    let k = 8_i64;

    let mut graph = Graph::with_config(
        GraphConfig::new()
            .with_name("cudagraph-matmul-add")
            .with_data_type_policy(
                DataTypePolicy::new()
                    .with_io(DataType::F16)
                    .with_intermediate(DataType::F32)
                    .with_compute(DataType::F32),
            ),
    );

    let a =
        graph.tensor(TensorSpec::new(DataType::F16, Shape::contiguous([b, m, k])?).with_id(1100));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.5)?);
    let scaled = graph
        .tensor(TensorSpec::new(DataType::F16, Shape::contiguous([b, m, k])?).virtual_tensor());
    graph.pointwise(
        singe_cudnn::frontend::operation::PointwiseOperation::Binary {
            mode: PointwiseMode::Mul,
            lhs: a,
            rhs: scale,
            output: scaled,
            compute_type: DataType::F32,
            nan_propagation: singe_cudnn::math::NanPropagation::Propagate,
            alpha1: 1.0,
            alpha2: 1.0,
        },
    );
    let b_tensor =
        graph.tensor(TensorSpec::new(DataType::F16, Shape::contiguous([b, k, n])?).with_id(1101));
    let matmul = graph
        .tensor(TensorSpec::new(DataType::F32, Shape::contiguous([b, m, n])?).virtual_tensor());
    graph.matmul_with_config(scaled, b_tensor, matmul, MatmulConfig::new(DataType::F32))?;
    let bias = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([1, 1, 1])?.with_strides([1, 1, 1])?,
        )
        .pass_by_value()
        .with_id(1102),
    );
    let d =
        graph.tensor(TensorSpec::new(DataType::F16, Shape::contiguous([b, m, n])?).with_id(1103));
    graph.pointwise(
        singe_cudnn::frontend::operation::PointwiseOperation::Binary {
            mode: PointwiseMode::Add,
            lhs: matmul,
            rhs: bias,
            output: d,
            compute_type: DataType::F32,
            nan_propagation: singe_cudnn::math::NanPropagation::Propagate,
            alpha1: 1.0,
            alpha2: 1.0,
        },
    );

    Ok((graph, a, b_tensor, bias, d))
}

fn bind_graph<'a>(
    compiled: &'a CompiledGraph,
    a: TensorId,
    b_tensor: TensorId,
    bias: TensorId,
    d: TensorId,
    a_dev: &mut DeviceMemory<f16>,
    b_dev: &mut DeviceMemory<f16>,
    d_dev: &mut DeviceMemory<f16>,
    bias_value: f16,
) -> Result<Bindings<'a>> {
    let mut bindings = compiled.bindings();
    bindings.set(a, a_dev)?;
    bindings.set(b_tensor, b_dev)?;
    bindings.set_scalar(bias, ScalarValue::f16_bits(bias_value.to_bits()))?;
    bindings.set(d, d_dev)?;
    Ok(bindings)
}

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;
    let (graph, a, b_tensor, bias, d) = build_graph()?;

    let compiled = graph
        .plan_candidates(&ctx.cudnn, &[HeuristicMode::A])?
        .require_behavior_notes(vec![BackendBehaviorNote::SupportsCudaGraphNativeApi])?
        .compile(&ctx.cudnn)?;

    let mut a_dev = DeviceMemory::from_slice(&vec![f16::from_f32(1.0); 8 * 32 * 8])?;
    let mut b_dev = DeviceMemory::from_slice(&vec![f16::from_f32(1.0); 8 * 8 * 16])?;
    let mut d_dev = DeviceMemory::<f16>::zeroes(8 * 32 * 16)?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;
    let bindings = bind_graph(
        &compiled,
        a,
        b_tensor,
        bias,
        d,
        &mut a_dev,
        &mut b_dev,
        &mut d_dev,
        f16::from_f32(2.0),
    )?;

    let mut main_graph = ctx.cudnn.cuda_context().create_graph()?;
    let child_node = match compiled.append_to_cuda_graph(
        &ctx.cudnn,
        &bindings,
        Some(&mut workspace),
        &mut main_graph,
        &[],
    ) {
        Ok(node) => node,
        Err(Error::Cudnn { code, .. })
            if code == Status::NotSupported || code == Status::NotSupportedCudaGraphNativeApi =>
        {
            return Ok(());
        }
        Err(error) => return Err(error),
    };
    let mut main_exec = main_graph.instantiate()?;
    main_exec.launch(&ctx.stream)?;
    ctx.stream.synchronize()?;

    let first = d_dev.copy_to_host_vec()?;
    assert!(first.iter().all(|value| value.to_f32() == 6.0));

    let mut a_dev_new = DeviceMemory::from_slice(&vec![f16::from_f32(1.0); 8 * 32 * 8])?;
    let mut b_dev_new = DeviceMemory::from_slice(&vec![f16::from_f32(1.0); 8 * 8 * 16])?;
    let mut d_dev_new = DeviceMemory::<f16>::zeroes(8 * 32 * 16)?;
    let mut workspace_new = common::workspace(compiled.workspace_size()?)?;
    let bindings_new = bind_graph(
        &compiled,
        a,
        b_tensor,
        bias,
        d,
        &mut a_dev_new,
        &mut b_dev_new,
        &mut d_dev_new,
        f16::from_f32(1.0),
    )?;
    compiled.update_cuda_graph_node(
        &ctx.cudnn,
        &bindings_new,
        Some(&mut workspace_new),
        child_node,
        Some(&mut main_exec),
    )?;
    main_exec.launch(&ctx.stream)?;
    ctx.stream.synchronize()?;

    let second = d_dev_new.copy_to_host_vec()?;
    assert!(second.iter().all(|value| value.to_f32() == 5.0));

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_cudagraphs", run())
}
