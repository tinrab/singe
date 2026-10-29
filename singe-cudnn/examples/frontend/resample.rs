mod common;

use std::env::args;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::{
        execution::resample::{PaddingMode, ResampleMode},
        tensor::{Shape, TensorSpec},
    },
    data_type::{DataType, f16},
    error::Result,
    frontend::{
        graph::{
            DataTypePolicy, Graph, GraphConfig, ResampleIndexOutput, ResampleOutputs,
            TensorOutputConfig,
        },
        operation::{ResampleBackwardInputs, ResampleConfig},
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
    let example = common::ExampleContext::create()?;

    let n = 8_i64;
    let h = 56_i64;
    let w = 56_i64;
    let c = 8_i64;
    let nchw = args().any(|argument| argument == "--nchw");

    let graph_config = GraphConfig::new().with_data_type_policy(
        DataTypePolicy::new()
            .with_io(DataType::F16)
            .with_compute(DataType::F32),
    );
    let mut graph = Graph::with_config(graph_config.clone());

    let input_shape = if nchw {
        Shape::contiguous([n, c, h, w])?
    } else {
        Shape::channels_last([n, c, h, w])?
    };
    let input_spec = TensorSpec::new(DataType::F16, input_shape);
    let input_elements = input_spec.shape.storage_span()?;
    let x = graph.tensor(input_spec.clone());
    let resample_config = ResampleConfig::new(ResampleMode::MaxPool)
        .with_name("pool")
        .with_padding_mode(PaddingMode::NegInf)
        .with_window_dims([2_i32, 3_i32])
        .with_strides([4_i32, 5_i32])
        .with_pre_paddings([2_i32, 3_i32])
        .with_post_paddings([4_i32, 5_i32]);
    let (y, indices) = if args().any(|argument| argument == "--explicit") {
        // These dimensions describe the same pooling geometry as the inferred path.
        // C++ preserves caller-declared output and index layouts through native finalization.
        let output_height = 16_i64;
        let output_width = 13_i64;
        let output_shape = if nchw {
            Shape::contiguous([n, c, output_height, output_width])?
        } else {
            Shape::channels_last([n, c, output_height, output_width])?
        };
        let output = graph.tensor(TensorSpec::new(DataType::F16, output_shape.clone()));
        let indices = (!nchw).then(|| graph.tensor(TensorSpec::new(DataType::I8, output_shape)));
        graph.resample_into(
            x,
            ResampleOutputs::new(output, indices),
            resample_config.clone(),
        )?;
        (output, indices)
    } else {
        let outputs = graph.resample(
            x,
            resample_config.clone(),
            TensorOutputConfig::new().output_tensor(),
            if nchw {
                ResampleIndexOutput::None
            } else {
                ResampleIndexOutput::Inferred(TensorOutputConfig::new().output_tensor())
            },
        )?;
        if nchw {
            let shape = Shape::contiguous(graph.shape(outputs.output)?.dimensions())?;
            graph.modify_tensor(outputs.output, |tensor| tensor.with_shape(shape))?;
        }
        (outputs.output, outputs.indices)
    };
    let y_elements = graph.shape(y)?.storage_span()?;
    graph.mark_as_output(y)?;
    if let Some(indices) = indices {
        graph.mark_as_output(indices)?;
    }

    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
    )?;

    let mut seed = 123_456_789_u32;
    let x_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_f16_image(input_elements, &mut seed),
    )?;
    let mut y_dev = DeviceMemory::<f16>::zeroed(example.cudnn.cuda_context(), y_elements)?;
    let mut index_buffer = indices
        .map(|id| -> Result<_> {
            let elements = graph.shape(id)?.storage_span()?;
            Ok((
                id,
                DeviceMemory::<i8>::zeroed(example.cudnn.cuda_context(), elements)?,
            ))
        })
        .transpose()?;
    let mut workspace = common::workspace(example.cudnn.cuda_context(), compiled.workspace_size())?;
    let mut bindings = compiled.bindings()?;
    bindings.set_input(x, &x_dev.view())?;
    bindings.set(y, &mut y_dev)?;
    if let Some((id, memory)) = &mut index_buffer {
        bindings.set(*id, memory)?;
    }
    // SAFETY: initialized input, writable outputs and workspace remain live through stream completion.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;
    drop(bindings);

    // NHWC backward consumes saved indices; NCHW backward requires forward X and Y.
    let mut backward = Graph::with_config(graph_config);
    let dy = backward.tensor(graph.tensor_spec(y)?.clone());
    let saved_indices = indices
        .map(|id| {
            graph
                .tensor_spec(id)
                .map(|spec| backward.tensor(spec.clone()))
        })
        .transpose()?;
    let saved_forward = if nchw {
        Some((
            backward.tensor(input_spec.clone()),
            backward.tensor(graph.tensor_spec(y)?.clone()),
        ))
    } else {
        None
    };
    let dx = backward.resample_backward(
        ResampleBackwardInputs {
            output_gradient: dy,
            input: saved_forward.map(|(input, _)| input),
            output: saved_forward.map(|(_, output)| output),
            indices: saved_indices,
        },
        input_spec.shape,
        resample_config.with_name("pool_backward"),
        TensorOutputConfig::new().output_tensor(),
    )?;
    backward.mark_as_output(dx)?;
    let compiled_backward = backward.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
    )?;
    let dy_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &vec![f16::from_f32(1.0); y_elements],
    )?;
    let mut dx_dev = DeviceMemory::<f16>::zeroed(example.cudnn.cuda_context(), input_elements)?;
    let mut backward_workspace = common::workspace(
        example.cudnn.cuda_context(),
        compiled_backward.workspace_size(),
    )?;
    let mut backward_bindings = compiled_backward.bindings()?;
    backward_bindings.set_input(dy, &dy_dev.view())?;
    if let (Some(indices), Some((_, memory))) = (saved_indices, &index_buffer) {
        backward_bindings.set_input(indices, &memory.view())?;
    }
    if let Some((input, output)) = saved_forward {
        backward_bindings.set_input(input, &x_dev.view())?;
        backward_bindings.set_input(output, &y_dev.view())?;
    }
    backward_bindings.set(dx, &mut dx_dev)?;
    // SAFETY: saved state and gradient inputs are initialized, and all buffers remain live through completion.
    example.execute_and_wait(|context| unsafe {
        compiled_backward.execute(
            context,
            ExecuteArgs::new(&backward_bindings).with_workspace(&mut backward_workspace),
        )
    })?;

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_resample", run())
}
