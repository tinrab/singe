mod common;

use std::env::args;

use singe_cuda::{memory::DeviceMemory, stream::StreamBinding};
use singe_cudnn::{
    backend::{
        pointwise::PointwiseMode,
        tensor::{BackendTensorReordering, Shape, TensorSpec},
    },
    data_type::DataType,
    error::Result,
    frontend::{
        graph::{Graph, TensorOutputConfig},
        operation::{ConvolutionConfig, PointwiseOperation},
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
        reorder::{Int8x32FilterReorder, ReorderBuffers},
    },
};

fn run() -> Result<()> {
    let mut context = common::ExampleContext::create()?;
    let relu = args().any(|argument| argument == "--relu");
    // Native dimensions count 32-element vectors along the channel axis.
    let input_spec = TensorSpec::new(DataType::I8, Shape::contiguous([1, 2, 5, 5])?)
        .with_vectorization(32, -3)?;
    let filter_spec = TensorSpec::new(DataType::I8, Shape::contiguous([32, 2, 3, 3])?)
        .with_vectorization(32, -3)?
        .with_reordering(BackendTensorReordering::Int8x32);
    let reorder = if args().any(|argument| argument == "--padded-filter-rank") {
        let padded = filter_spec
            .clone()
            .with_shape(Shape::contiguous([32, 1, 2, 3, 3])?)
            .with_vectorization(32, -3)?;
        Int8x32FilterReorder::create_for_tensor(&padded, 2)?
    } else {
        Int8x32FilterReorder::create_for_tensor(&filter_spec, 2)?
    };
    let input_bytes = input_spec.storage_byte_len(None)?;
    let mut graph = Graph::new();
    let input = graph.tensor(input_spec);
    let filter = graph.tensor(filter_spec);
    let convolution_output = if relu {
        TensorOutputConfig::new()
    } else {
        TensorOutputConfig::new().output_tensor()
    };
    let convolution = graph.convolution_forward(
        input,
        filter,
        ConvolutionConfig::new(2).with_compute_type(DataType::I32),
        convolution_output,
    )?;
    let output = if relu {
        graph.pointwise(
            PointwiseOperation::unary(
                PointwiseMode::ReluFwd,
                convolution,
                TensorOutputConfig::new()
                    .with_data_type(DataType::I8)
                    .output_tensor(),
            )
            .with_compute_type(DataType::I32),
        )?
    } else {
        convolution
    };
    let compiled = graph.compile(
        Some(&context.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A, HeuristicMode::Fallback]),
    )?;

    let input_device =
        DeviceMemory::from_slice(context.cudnn.cuda_context(), &vec![1_i8; input_bytes])?;
    let filter_source = DeviceMemory::from_slice(
        context.cudnn.cuda_context(),
        &vec![1_i8; reorder.filter_byte_len()],
    )?;
    let mut reordered_filter = DeviceMemory::from_slice(
        context.cudnn.cuda_context(),
        &vec![0_i8; reorder.filter_byte_len()],
    )?;
    let mut output_device = DeviceMemory::from_slice(
        context.cudnn.cuda_context(),
        &vec![0_i8; graph.tensor_spec(output)?.storage_byte_len(None)?],
    )?;
    let mut bindings = compiled.bindings()?;
    bindings.set_input(input, &input_device.view())?;
    bindings.set_input(filter, &reordered_filter.view())?;
    bindings.set(output, &mut output_device)?;
    let mut workspace = common::workspace(context.cudnn.cuda_context(), compiled.workspace_size())?;
    StreamBinding::Borrowed(context.stream.to_borrowed()).sync_scope(|| -> Result<()> {
        // SAFETY: buffers are disjoint and initialized, and the scope retains their owners and descriptor through completion, including errors.
        unsafe {
            reorder.execute(
                &mut context.cudnn,
                &context.stream,
                Some(ReorderBuffers {
                    source: &filter_source,
                    destination: &mut reordered_filter,
                }),
                None,
            )?;
        }
        // SAFETY: the example retains its plan and buffers through the surrounding completion or replay sequence.
        unsafe {
            compiled.execute(
                &context.cudnn,
                ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
            )
        }
    })?;
    println!(
        "reordered int8x32 convolution: {:?}",
        &output_device.copy_to_host_vec()?[..8]
    );
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_convolution_int8x32_reorder", run())
}
