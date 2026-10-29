mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::{
        pointwise::PointwiseMode,
        reduction::ReduceTensorOperator,
        tensor::{Shape, TensorSpec},
    },
    data_type::{DataType, f8e4m3},
    error::Result,
    frontend::{
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::{ConvolutionConfig, PointwiseOperation, ReductionAxes, ReductionConfig},
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
};

fn run() -> Result<()> {
    let example = common::ExampleContext::create()?;

    let properties = example.cudnn.cuda_context().device().properties()?;
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

    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F16)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );

    let x = graph.tensor(
        TensorSpec::new(DataType::F8E4M3, Shape::channels_last([n, c, h, w])?).with_name("image"),
    );
    let w_tensor = graph.tensor(
        TensorSpec::new(DataType::F8E4M3, Shape::channels_last([k, c, r, s])?).with_name("filter"),
    );
    let conv_output = graph.convolution_forward(
        x,
        w_tensor,
        ConvolutionConfig::new(2)
            .with_name("conv")
            .with_padding([0, 0])
            .with_strides([1, 1])
            .with_dilations([1, 1]),
        TensorOutputConfig::new().virtual_tensor(),
    )?;

    let descale_x = graph
        .tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1; 4])?).with_name("descale_x"));
    let descale_w = graph.tensor_like(descale_x, Some("descale_w"))?;
    let scale_y = graph.tensor_like(descale_x, Some("scale_y"))?;

    let mut after_descale_w = conv_output;
    for (scale, name) in [(descale_x, "descale_x"), (descale_w, "descale_w")] {
        after_descale_w = graph.pointwise(
            PointwiseOperation::binary(
                PointwiseMode::Mul,
                after_descale_w,
                scale,
                TensorOutputConfig::new().virtual_tensor(),
            )
            .with_name(name),
        )?;
    }
    let y = graph.pointwise(
        PointwiseOperation::binary(
            PointwiseMode::Mul,
            after_descale_w,
            scale_y,
            TensorOutputConfig::new()
                .with_data_type(DataType::F8E4M3)
                .output_tensor(),
        )
        .with_name("scale_y"),
    )?;
    let absolute_max = graph.reduction(
        after_descale_w,
        &ReductionConfig::new(ReduceTensorOperator::AbsoluteMax)
            .with_axes(ReductionAxes::All)
            .with_name("absolute_max"),
        TensorOutputConfig::new()
            .with_data_type(DataType::F32)
            .output_tensor(),
    )?;

    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A, HeuristicMode::Fallback]),
    )?;
    let cuda = example.cudnn.cuda_context();
    let mut seed = 123_456_789;
    let x_dev = DeviceMemory::<f8e4m3>::from_slice(
        cuda,
        &common::init_f8e4m3_image(graph.shape(x)?.storage_span()?, &mut seed),
    )?;
    let w_dev = DeviceMemory::<f8e4m3>::from_slice(
        cuda,
        &common::init_f8e4m3_image(graph.shape(w_tensor)?.storage_span()?, &mut seed),
    )?;
    let mut y_dev = DeviceMemory::<f8e4m3>::zeroed(cuda, graph.shape(y)?.storage_span()?)?;
    let descale_x_dev = DeviceMemory::from_slice(cuda, &[0.5_f32])?;
    let descale_w_dev = DeviceMemory::from_slice(cuda, &[0.5_f32])?;
    let scale_y_dev = DeviceMemory::from_slice(cuda, &[2.0_f32])?;
    let mut absolute_max_dev =
        DeviceMemory::<f32>::zeroed(cuda, graph.shape(absolute_max)?.storage_span()?)?;
    let mut workspace = common::workspace(cuda, compiled.workspace_size())?;

    let mut bindings = compiled.bindings()?;
    bindings.set_input(x, &x_dev)?;
    bindings.set_input(w_tensor, &w_dev)?;
    bindings.set_input(descale_x, &descale_x_dev)?;
    bindings.set_input(descale_w, &descale_w_dev)?;
    bindings.set_input(scale_y, &scale_y_dev)?;
    bindings.set(y, &mut y_dev)?;
    bindings.set(absolute_max, &mut absolute_max_dev)?;
    // SAFETY: initialized inputs and scales, writable outputs and workspace remain live through stream completion.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;
    println!("executed fp8 convolution with fp32 intermediates and pre-quantization amax");

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_convolution_fp8_fprop", run())
}
