mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::{
        pointwise::PointwiseMode,
        tensor::{Shape, TensorId, TensorSpec},
    },
    data_type::{DataType, f16},
    error::Result,
    frontend::{
        graph::{DataTypePolicy, Graph, GraphConfig, GraphKeyShapePolicy, TensorOutputConfig},
        operation::{ConvolutionConfig, PointwiseOperation},
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
    math::NanPropagation,
};

const X_ID: TensorId = TensorId::new(1100);
const W_ID: TensorId = TensorId::new(1101);
const SCALE_ID: TensorId = TensorId::new(1102);
const BIAS_ID: TensorId = TensorId::new(1103);
const Y_ID: TensorId = TensorId::new(1104);

fn init_f16_image(size: usize, seed: &mut u32) -> Vec<f16> {
    common::init_f32_image(size, seed)
        .into_iter()
        .map(f16::from_f32)
        .collect()
}

fn build_graph() -> Result<Graph> {
    let n = 4_i64;
    let c = 32_i64;
    let h = 16_i64;
    let w = 16_i64;
    let k = 64_i64;
    let r = 3_i64;
    let s = 3_i64;

    let mut graph = Graph::with_config(
        GraphConfig::new()
            .with_name("serialize-demo")
            .with_data_type_policy(
                DataTypePolicy::new()
                    .with_io(DataType::F16)
                    .with_intermediate(DataType::F32)
                    .with_compute(DataType::F32),
            ),
    );

    let x = graph.insert_tensor(
        X_ID,
        TensorSpec::new(DataType::F16, Shape::channels_last([n, c, h, w])?),
    )?;
    let w_tensor = graph.insert_tensor(
        W_ID,
        TensorSpec::new(DataType::F16, Shape::channels_last([k, c, r, s])?),
    )?;
    let conv = graph.convolution_forward(
        x,
        w_tensor,
        ConvolutionConfig::new(2)
            .with_padding(vec![1, 1])
            .with_strides([1, 1])
            .with_dilations(vec![1, 1]),
        TensorOutputConfig::new().with_data_type(DataType::F16),
    )?;
    let scale = graph.insert_tensor(
        SCALE_ID,
        TensorSpec::new(DataType::F16, Shape::channels_last([1, k, 1, 1])?),
    )?;
    let scaled = graph.pointwise(PointwiseOperation::binary(
        PointwiseMode::Mul,
        conv,
        scale,
        TensorOutputConfig::new()
            .with_data_type(DataType::F16)
            .virtual_tensor(),
    ))?;
    let bias = graph.insert_tensor(
        BIAS_ID,
        TensorSpec::new(DataType::F16, Shape::channels_last([1, k, 1, 1])?),
    )?;
    let biased = graph.pointwise(PointwiseOperation::binary(
        PointwiseMode::Add,
        scaled,
        bias,
        TensorOutputConfig::new()
            .with_data_type(DataType::F16)
            .virtual_tensor(),
    ))?;
    let y = graph.insert_tensor(
        Y_ID,
        TensorSpec::new(DataType::F16, Shape::channels_last([n, k, h, w])?),
    )?;
    graph.pointwise_into(
        PointwiseOperation::unary(PointwiseMode::ReluFwd, biased, y)
            .with_nan_propagation(NanPropagation::NotPropagate),
    )?;

    Ok(graph)
}

fn run() -> Result<()> {
    let context = common::ExampleContext::create()?;
    let graph = build_graph()?;

    let serialized = serde_json::to_vec(&graph)?;
    let restored = serde_json::from_slice::<Graph>(&serialized)?;
    println!(
        "restored graph {:?}: {} JSON bytes, key={:016x}",
        restored.name(),
        serialized.len(),
        restored.key(GraphKeyShapePolicy::Auto)?,
    );

    let compiled = restored.compile(
        Some(&context.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
    )?;
    let mut seed = 123_456_789_u32;
    let x_dev = DeviceMemory::from_slice(
        context.cudnn.cuda_context(),
        &init_f16_image(restored.shape(X_ID)?.storage_span()?, &mut seed),
    )?;
    let w_dev = DeviceMemory::from_slice(
        context.cudnn.cuda_context(),
        &init_f16_image(restored.shape(W_ID)?.storage_span()?, &mut seed),
    )?;
    let scale_dev = DeviceMemory::from_slice(
        context.cudnn.cuda_context(),
        &init_f16_image(restored.shape(SCALE_ID)?.storage_span()?, &mut seed),
    )?;
    let bias_dev = DeviceMemory::from_slice(
        context.cudnn.cuda_context(),
        &init_f16_image(restored.shape(BIAS_ID)?.storage_span()?, &mut seed),
    )?;
    let output_elements = restored.shape(Y_ID)?.storage_span()?;
    let mut y_dev = DeviceMemory::<f16>::zeroed(context.cudnn.cuda_context(), output_elements)?;
    let mut workspace = common::workspace(context.cudnn.cuda_context(), compiled.workspace_size())?;

    let mut bindings = compiled.bindings()?;
    bindings
        .set_input(X_ID, &x_dev.view())?
        .set_input(W_ID, &w_dev.view())?
        .set_input(SCALE_ID, &scale_dev.view())?
        .set_input(BIAS_ID, &bias_dev.view())?
        .set(Y_ID, &mut y_dev)?;
    // SAFETY: inputs, writable output, plan and workspace remain live until stream completion, including enqueue failure.
    let execution = unsafe {
        compiled.execute(
            &context.cudnn,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    };
    let completion = context.stream.synchronize();
    execution?;
    completion?;
    println!("executed restored graph with {output_elements} output elements");

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_graph_serialize", run())
}
