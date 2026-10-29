mod common;

use std::env::args;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::{
        pointwise::PointwiseMode,
        tensor::{Shape, TensorSpec},
    },
    data_type::{DataType, f16},
    error::Result,
    frontend::{
        graph::{DataTypePolicy, GenStatsOutputs, Graph, GraphConfig, TensorOutputConfig},
        operation::{ConvolutionConfig, GenStatsConfig, PointwiseOperation},
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
};

fn run() -> Result<()> {
    let example = common::ExampleContext::create()?;
    // Matches C++'s SBRCS sample: scale, bias, ReLU, convolution and channel statistics.
    let [n, c, h, w, k, r, s] = [4, 64, 16, 16, 32, 3, 3];
    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F16)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );
    let image = graph.tensor(
        TensorSpec::new(DataType::F16, Shape::channels_last([n, c, h, w])?).with_name("image"),
    );
    let filter = graph.tensor(
        TensorSpec::new(DataType::F16, Shape::channels_last([k, c, r, s])?).with_name("filter"),
    );
    let channel = TensorSpec::new(DataType::F16, Shape::channels_last([1, c, 1, 1])?);
    let scale = graph.tensor(channel.clone().with_name("scale"));
    let bias = graph.tensor(channel.with_name("bias"));
    let scaled = graph.pointwise(
        PointwiseOperation::binary(PointwiseMode::Mul, image, scale, TensorOutputConfig::new())
            .with_name("scale"),
    )?;
    let shifted = graph.pointwise(
        PointwiseOperation::binary(PointwiseMode::Add, scaled, bias, TensorOutputConfig::new())
            .with_name("bias"),
    )?;
    let activated = graph.pointwise(
        PointwiseOperation::unary(PointwiseMode::ReluFwd, shifted, TensorOutputConfig::new())
            .with_name("relu"),
    )?;
    let output = graph.convolution_forward(
        activated,
        filter,
        ConvolutionConfig::new(2)
            .with_name("conv")
            .with_pre_paddings([1, 1])
            .with_post_paddings([1, 1])
            .with_strides([1, 1])
            .with_dilations([1, 1]),
        TensorOutputConfig::new().output_tensor(),
    )?;
    let statistic = TensorOutputConfig::new()
        .with_data_type(DataType::F32)
        .output_tensor();
    let statistics_config = GenStatsConfig::new()
        .with_name("convolution_statistics")
        .with_compute_type(DataType::F32);
    let statistics = if args().any(|argument| argument == "--explicit-stats") {
        let channels = graph.shape(output)?.dimensions()[1];
        let shape = Shape::contiguous([1, channels, 1, 1])?;
        let sum = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()).with_name("sum"));
        let square_sum =
            graph.tensor(TensorSpec::new(DataType::F32, shape).with_name("square_sum"));
        let outputs = GenStatsOutputs::new(sum, square_sum);
        graph.genstats_into(output, outputs, statistics_config)?;
        outputs
    } else {
        graph.genstats(
            output,
            GenStatsOutputs::new(statistic, statistic),
            statistics_config,
        )?
    };
    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A, HeuristicMode::Fallback]),
    )?;
    let cuda = example.cudnn.cuda_context();
    let mut seed = 123_456_789_u32;
    let image_dev = DeviceMemory::from_slice(
        cuda,
        &common::init_f16_image(graph.shape(image)?.storage_span()?, &mut seed),
    )?;
    let filter_dev = DeviceMemory::from_slice(
        cuda,
        &common::init_f16_image(graph.shape(filter)?.storage_span()?, &mut seed),
    )?;
    let scale_dev = DeviceMemory::from_slice(
        cuda,
        &vec![f16::from_f32(1.0); graph.shape(scale)?.storage_span()?],
    )?;
    let bias_dev = DeviceMemory::from_slice(
        cuda,
        &vec![f16::from_f32(0.1); graph.shape(bias)?.storage_span()?],
    )?;
    let mut output_dev = DeviceMemory::<f16>::zeroed(cuda, graph.shape(output)?.storage_span()?)?;
    let mut sum_dev =
        DeviceMemory::<f32>::zeroed(cuda, graph.shape(statistics.sum)?.storage_span()?)?;
    let mut square_sum_dev =
        DeviceMemory::<f32>::zeroed(cuda, graph.shape(statistics.square_sum)?.storage_span()?)?;
    let mut workspace = common::workspace(cuda, compiled.workspace_size())?;
    let mut bindings = compiled.bindings()?;
    bindings.set_input(image, &image_dev)?;
    bindings.set_input(filter, &filter_dev)?;
    bindings.set_input(scale, &scale_dev)?;
    bindings.set_input(bias, &bias_dev)?;
    bindings.set(output, &mut output_dev)?;
    bindings.set(statistics.sum, &mut sum_dev)?;
    bindings.set(statistics.square_sum, &mut square_sum_dev)?;
    // SAFETY: initialized inputs, independent writable results and workspace stay live
    // through stream completion, including an enqueue failure.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;
    println!("SBRCS produced an FP16 activation and two FP32 channel-statistic tensors");
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_convolution_sbrcs", run())
}
