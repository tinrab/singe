mod common;

use std::env::args;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::{
        pointwise::PointwiseMode,
        tensor::{Shape, TensorSpec},
    },
    data_type::DataType,
    error::Result,
    frontend::{
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::{ConvolutionConfig, PointwiseOperation},
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
    math::NanPropagation,
};

fn run() -> Result<()> {
    let include_identity = !args().any(|argument| argument == "--without-identity");
    let example = common::ExampleContext::create()?;

    let n = 1_i64;
    let c = 64_i64;
    let h = 32_i64;
    let w = 32_i64;
    let k = 4_i64;
    let r = 3_i64;
    let s = 3_i64;

    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::I8)
                .with_intermediate(DataType::I32)
                .with_compute(DataType::I32),
        ),
    );

    let x = graph.tensor(
        TensorSpec::new(DataType::I8, Shape::channels_last([n, c, h, w])?).with_name("image"),
    );
    let w_tensor = graph.tensor(
        TensorSpec::new(DataType::I8, Shape::channels_last([k, c, r, s])?).with_name("filter"),
    );
    // Inherit INT32 intermediate storage so the convolution result is not narrowed to INT8.
    let conv_output = graph.convolution_forward(
        x,
        w_tensor,
        ConvolutionConfig::new(2)
            .with_name("conv")
            .with_padding([1, 1])
            .with_strides([1, 1])
            .with_dilations([1, 1]),
        TensorOutputConfig::new().virtual_tensor(),
    )?;

    let y = if include_identity {
        graph.pointwise(
            PointwiseOperation::unary(
                PointwiseMode::Identity,
                conv_output,
                TensorOutputConfig::new()
                    .with_data_type(DataType::I32)
                    .output_tensor(),
            )
            .with_name("identity")
            .with_nan_propagation(NanPropagation::NotPropagate),
        )?
    } else {
        graph.mark_as_output(conv_output)?;
        conv_output
    };

    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A, HeuristicMode::Fallback]),
    )?;
    let cuda = example.cudnn.cuda_context();
    let mut seed = 123_456_789;
    let x_dev = DeviceMemory::<i8>::from_slice(
        cuda,
        &common::init_i8_image(graph.shape(x)?.storage_span()?, &mut seed),
    )?;
    let w_dev = DeviceMemory::<i8>::from_slice(
        cuda,
        &common::init_i8_image(graph.shape(w_tensor)?.storage_span()?, &mut seed),
    )?;
    let mut y_dev = DeviceMemory::<i32>::zeroed(cuda, graph.shape(y)?.storage_span()?)?;
    let mut workspace = common::workspace(cuda, compiled.workspace_size())?;

    let mut bindings = compiled.bindings()?;
    bindings.set_input(x, &x_dev)?;
    bindings.set_input(w_tensor, &w_dev)?;
    bindings.set(y, &mut y_dev)?;
    // SAFETY: initialized inputs, writable output and workspace remain live through stream completion.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;
    println!("executed int8 convolution with int32 output, identity: {include_identity}");

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_convolution_int8_fprop", run())
}
