mod common;

use std::env::args;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorId, TensorSpec},
    data_type::{DataType, f16},
    error::{Error, Result},
    frontend::{
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::ConvolutionConfig,
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
};

fn init_f16_image(size: usize, seed: &mut u32) -> Vec<f16> {
    common::init_f32_image(size, seed)
        .into_iter()
        .map(f16::from_f32)
        .collect()
}

fn run() -> Result<()> {
    let depthwise = args().any(|argument| argument == "--depthwise");
    let n = 16;
    let c = 128_i64;
    let h = 64;
    let w = 64;
    let k = if depthwise { c } else { 256 };
    let r = if depthwise { 3 } else { 1 };
    let s = r;
    let padding = if depthwise { 1 } else { 0 };
    let groups = i64::try_from(
        common::positive_usize_argument("--groups", if depthwise { 128 } else { 1 })?.get(),
    )
    .map_err(|_| Error::OutOfRange {
        name: "--groups".into(),
    })?;
    if c % groups != 0 || k % groups != 0 || (depthwise && groups != c) {
        return Err(Error::OutOfRange {
            name: "--groups".into(),
        });
    }
    let filter_channels = c / groups;
    let example = common::ExampleContext::create()?;

    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F16)
                .with_compute(DataType::F32),
        ),
    );

    let x = graph.insert_tensor(
        TensorId::new(1002),
        TensorSpec::new(DataType::F16, Shape::channels_last([n, c, h, w])?).with_name("image"),
    )?;
    let w_tensor = graph.insert_tensor(
        TensorId::new(1001),
        TensorSpec::new(
            DataType::F16,
            Shape::channels_last([k, filter_channels, r, s])?,
        )
        .with_name("filter"),
    )?;
    let config = ConvolutionConfig::new(2)
        .with_name("convolution")
        .with_scaling(1.0, 1.0)
        .with_pre_paddings([padding, padding])
        .with_post_paddings([padding, padding])
        .with_strides([1, 1])
        .with_dilations([1, 1]);
    // Both forms accumulate into initialized caller storage using the native beta parameter.
    let y = if args().any(|argument| argument == "--explicit") {
        let y = graph.insert_tensor(
            TensorId::new(1003),
            TensorSpec::new(DataType::F16, Shape::channels_last([n, k, h, w])?)
                .with_name("response"),
        )?;
        graph.convolution_forward_into(x, w_tensor, y, config)?;
        y
    } else {
        graph.convolution_forward(
            x,
            w_tensor,
            config,
            TensorOutputConfig::new().output_tensor(),
        )?
    };

    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A, HeuristicMode::Fallback]),
    )?;

    let mut half_seed = 123_456_789_u32;
    let x_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_f16_image(graph.shape(x)?.storage_span()?, &mut half_seed),
    )?;
    let w_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_f16_image(graph.shape(w_tensor)?.storage_span()?, &mut half_seed),
    )?;
    let mut y_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_f16_image(graph.shape(y)?.storage_span()?, &mut half_seed),
    )?;
    let mut workspace = common::workspace(example.cudnn.cuda_context(), compiled.workspace_size())?;
    let mut bindings = compiled.bindings()?;
    bindings.set_input(x, &x_dev)?;
    bindings.set_input(w_tensor, &w_dev)?;
    bindings.set(y, &mut y_dev)?;
    // SAFETY: the plan, inputs, accumulated output and workspace remain live until
    // stream completion, including when native execution returns an enqueue error.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;

    let output = compiled.tensor_info(y)?;
    println!(
        "groups={groups}, filter channels/group={filter_channels}, {:?}: {:?}",
        output.spec.name,
        output.spec.shape.dimensions()
    );

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_convolution_fprop", run())
}
