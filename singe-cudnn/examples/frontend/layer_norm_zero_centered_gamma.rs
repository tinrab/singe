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
        error::Error as FrontendError,
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::{
            LayerNormalizationAffine, LayerNormalizationAffineGradients,
            LayerNormalizationBackwardConfig, LayerNormalizationBackwardInputs,
            LayerNormalizationBackwardOutputs, LayerNormalizationConfig, LayerNormalizationInputs,
            LayerNormalizationOutputs, NormalizationSavedStatistics, NormalizationStatistics,
            PointwiseOperation,
        },
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
};

const BATCH_SIZE: i64 = 4;
const SEQ_LENGTH: i64 = 1024;
const HIDDEN_SIZE: i64 = 128;

fn hidden_layout() -> Result<Shape> {
    Shape::with_strides(
        [BATCH_SIZE * SEQ_LENGTH, HIDDEN_SIZE, 1, 1],
        vec![HIDDEN_SIZE, 1, HIDDEN_SIZE, HIDDEN_SIZE],
    )
}

fn hidden_reduction_layout() -> Result<Shape> {
    Shape::with_strides([BATCH_SIZE * SEQ_LENGTH, 1, 1, 1], [1, 1, 1, 1])
}

fn scale_layout() -> Result<Shape> {
    Shape::with_strides(
        [1, HIDDEN_SIZE, 1, 1],
        vec![HIDDEN_SIZE, 1, HIDDEN_SIZE, HIDDEN_SIZE],
    )
}

fn zero_centered_scale(graph: &mut Graph) -> Result<(TensorId, TensorId)> {
    let scale_zero_centered = graph.tensor(TensorSpec::new(DataType::F32, scale_layout()?));
    let one = graph.tensor(TensorSpec::scalar_f32(1.0));
    let scale = graph.pointwise(
        PointwiseOperation::binary(
            PointwiseMode::Add,
            scale_zero_centered,
            one,
            TensorOutputConfig::new()
                .with_data_type(DataType::F32)
                .virtual_tensor(),
        )
        .with_compute_type(Some(DataType::F32)),
    )?;
    Ok((scale_zero_centered, scale))
}

fn run_training(example: &common::ExampleContext) -> Result<()> {
    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F16)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );

    let x = graph.tensor(TensorSpec::new(DataType::F16, hidden_layout()?));
    let (scale_zero_centered, scale) = zero_centered_scale(&mut graph)?;
    let bias = graph.tensor(TensorSpec::new(DataType::F32, scale_layout()?));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5));

    let y = graph.tensor(TensorSpec::new(DataType::F16, hidden_layout()?));
    let mean =
        graph.tensor(TensorSpec::new(DataType::F32, hidden_reduction_layout()?).virtual_tensor());
    let inv_variance =
        graph.tensor(TensorSpec::new(DataType::F32, hidden_reduction_layout()?).virtual_tensor());
    graph.layer_normalization_into(
        LayerNormalizationInputs {
            input: x,
            affine: LayerNormalizationAffine::ScaleBias { scale, bias },
        },
        LayerNormalizationOutputs {
            output: y,
            statistics: NormalizationStatistics::Training(NormalizationSavedStatistics {
                mean,
                inv_variance,
            }),
        },
        LayerNormalizationConfig::new(epsilon).with_name("zero_centered_layer_norm"),
    )?;

    for output in [y, mean, inv_variance] {
        graph.mark_as_output(output)?;
    }

    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A, HeuristicMode::Fallback]),
    )?;
    let x_dev = DeviceMemory::<f16>::zeroed(
        example.cudnn.cuda_context(),
        graph.shape(x)?.storage_span()?,
    )?;
    let scale_zero_centered_dev = DeviceMemory::<f32>::zeroed(
        example.cudnn.cuda_context(),
        graph.shape(scale_zero_centered)?.storage_span()?,
    )?;
    let bias_dev = DeviceMemory::<f32>::zeroed(
        example.cudnn.cuda_context(),
        graph.shape(bias)?.storage_span()?,
    )?;
    let mut y_dev = DeviceMemory::<f16>::zeroed(
        example.cudnn.cuda_context(),
        graph.shape(y)?.storage_span()?,
    )?;
    let mut mean_dev = DeviceMemory::<f32>::zeroed(
        example.cudnn.cuda_context(),
        graph.shape(mean)?.storage_span()?,
    )?;
    let mut inv_variance_dev = DeviceMemory::<f32>::zeroed(
        example.cudnn.cuda_context(),
        graph.shape(inv_variance)?.storage_span()?,
    )?;
    let mut workspace = common::workspace(example.cudnn.cuda_context(), compiled.workspace_size())?;

    let mut bindings = compiled.bindings()?;
    bindings.set_input(x, &x_dev)?;
    bindings.set_input(scale_zero_centered, &scale_zero_centered_dev)?;
    bindings.set_input(bias, &bias_dev)?;
    bindings.set(y, &mut y_dev)?;
    bindings.set(mean, &mut mean_dev)?;
    bindings.set(inv_variance, &mut inv_variance_dev)?;
    // SAFETY: buffers, plan and workspace remain live until the stream completes,
    // including when execution returns an enqueue error.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })
}

fn run_inference(example: &common::ExampleContext) -> Result<()> {
    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F16)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );

    let x = graph.tensor(TensorSpec::new(DataType::F16, hidden_layout()?));
    let (scale_zero_centered, scale) = zero_centered_scale(&mut graph)?;
    let bias = graph.tensor(TensorSpec::new(DataType::F32, scale_layout()?));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5));

    let outputs = graph.layer_normalization(
        LayerNormalizationInputs {
            input: x,
            affine: LayerNormalizationAffine::ScaleBias { scale, bias },
        },
        LayerNormalizationOutputs {
            output: TensorOutputConfig::new().output_tensor(),
            statistics: NormalizationStatistics::Inference,
        },
        LayerNormalizationConfig::new(epsilon).with_name("zero_centered_layer_norm"),
    )?;
    let y = outputs.output;

    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A, HeuristicMode::Fallback]),
    )?;
    println!(
        "inference ids: x={}, scale_zero_centered={}, bias={}, y={}",
        x.as_i64(),
        scale_zero_centered.as_i64(),
        bias.as_i64(),
        y.as_i64(),
    );
    let x_dev = DeviceMemory::<f16>::zeroed(
        example.cudnn.cuda_context(),
        graph.shape(x)?.storage_span()?,
    )?;
    let scale_zero_centered_dev = DeviceMemory::<f32>::zeroed(
        example.cudnn.cuda_context(),
        graph.shape(scale_zero_centered)?.storage_span()?,
    )?;
    let bias_dev = DeviceMemory::<f32>::zeroed(
        example.cudnn.cuda_context(),
        graph.shape(bias)?.storage_span()?,
    )?;
    let mut y_dev = DeviceMemory::<f16>::zeroed(
        example.cudnn.cuda_context(),
        graph.shape(y)?.storage_span()?,
    )?;
    let mut workspace = common::workspace(example.cudnn.cuda_context(), compiled.workspace_size())?;

    let mut bindings = compiled.bindings()?;
    bindings.set_input(x, &x_dev)?;
    bindings.set_input(scale_zero_centered, &scale_zero_centered_dev)?;
    bindings.set_input(bias, &bias_dev)?;
    bindings.set(y, &mut y_dev)?;
    // SAFETY: buffers, plan and workspace remain live until the stream completes,
    // including when execution returns an enqueue error.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })
}

fn run_backward(example: &common::ExampleContext) -> Result<()> {
    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F16)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );

    let x = graph.tensor(TensorSpec::new(DataType::F16, hidden_layout()?));
    let dy = graph.tensor(TensorSpec::new(DataType::F16, hidden_layout()?));
    let (scale_zero_centered, scale) = zero_centered_scale(&mut graph)?;
    let mean = graph.tensor(TensorSpec::new(DataType::F32, hidden_reduction_layout()?));
    let inv_variance = graph.tensor(TensorSpec::new(DataType::F32, hidden_reduction_layout()?));

    let outputs = graph.layer_normalization_backward(
        LayerNormalizationBackwardInputs {
            input: x,
            statistics: NormalizationSavedStatistics { mean, inv_variance },
            output_gradient: dy,
            scale: Some(scale),
        },
        LayerNormalizationBackwardOutputs {
            input_gradient: TensorOutputConfig::new().output_tensor(),
            affine: Some(LayerNormalizationAffineGradients {
                scale_gradient: TensorOutputConfig::new()
                    .with_data_type(DataType::F32)
                    .output_tensor(),
                bias_gradient: TensorOutputConfig::new()
                    .with_data_type(DataType::F32)
                    .output_tensor(),
            }),
        },
        LayerNormalizationBackwardConfig::new().with_name("zero_centered_layer_norm_backward"),
    )?;
    let dx = outputs.input_gradient;
    let affine = outputs
        .affine
        .ok_or(FrontendError::NormalizationAffineGradients {
            operation: "layer norm zero-centered gamma backward",
        })?;
    let dscale = affine.scale_gradient;
    let bias_gradient = affine.bias_gradient;

    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A, HeuristicMode::Fallback]),
    )?;
    println!(
        "backward ids: x={}, dy={}, scale_zero_centered={}, mean={}, inv_variance={}, dx={}, dscale={}, bias_gradient={}",
        x.as_i64(),
        dy.as_i64(),
        scale_zero_centered.as_i64(),
        mean.as_i64(),
        inv_variance.as_i64(),
        dx.as_i64(),
        dscale.as_i64(),
        bias_gradient.as_i64(),
    );
    let x_dev = DeviceMemory::<f16>::zeroed(
        example.cudnn.cuda_context(),
        graph.shape(x)?.storage_span()?,
    )?;
    let dy_dev = DeviceMemory::<f16>::zeroed(
        example.cudnn.cuda_context(),
        graph.shape(dy)?.storage_span()?,
    )?;
    let scale_zero_centered_dev = DeviceMemory::<f32>::zeroed(
        example.cudnn.cuda_context(),
        graph.shape(scale_zero_centered)?.storage_span()?,
    )?;
    let mean_dev = DeviceMemory::<f32>::zeroed(
        example.cudnn.cuda_context(),
        graph.shape(mean)?.storage_span()?,
    )?;
    let inv_variance_dev = DeviceMemory::<f32>::zeroed(
        example.cudnn.cuda_context(),
        graph.shape(inv_variance)?.storage_span()?,
    )?;
    let mut dx_dev = DeviceMemory::<f16>::zeroed(
        example.cudnn.cuda_context(),
        graph.shape(dx)?.storage_span()?,
    )?;
    let mut dscale_dev = DeviceMemory::<f32>::zeroed(
        example.cudnn.cuda_context(),
        graph.shape(dscale)?.storage_span()?,
    )?;
    let mut bias_gradient_dev = DeviceMemory::<f32>::zeroed(
        example.cudnn.cuda_context(),
        graph.shape(bias_gradient)?.storage_span()?,
    )?;
    let mut workspace = common::workspace(example.cudnn.cuda_context(), compiled.workspace_size())?;

    let mut bindings = compiled.bindings()?;
    bindings.set_input(x, &x_dev)?;
    bindings.set_input(dy, &dy_dev)?;
    bindings.set_input(scale_zero_centered, &scale_zero_centered_dev)?;
    bindings.set_input(mean, &mean_dev)?;
    bindings.set_input(inv_variance, &inv_variance_dev)?;
    bindings.set(dx, &mut dx_dev)?;
    bindings.set(dscale, &mut dscale_dev)?;
    bindings.set(bias_gradient, &mut bias_gradient_dev)?;
    // SAFETY: buffers, plan and workspace remain live until the stream completes,
    // including when execution returns an enqueue error.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })
}

fn run() -> Result<()> {
    let context = common::ExampleContext::create()?;
    run_training(&context)?;
    run_inference(&context)?;
    run_backward(&context)
}

fn main() -> Result<()> {
    common::finish("frontend_layer_norm_zero_centered_gamma", run())
}
