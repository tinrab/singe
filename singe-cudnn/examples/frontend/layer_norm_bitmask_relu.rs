mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::{
        pointwise::PointwiseMode,
        tensor::{Shape, TensorSpec},
    },
    data_type::DataType,
    error::Result,
    frontend::{
        graph::{DataTypePolicy, Graph, GraphConfig},
        operation::{
            Activation, LayerNormalizationAffine, LayerNormalizationAffineGradients,
            LayerNormalizationBackwardConfig, LayerNormalizationBackwardInputs,
            LayerNormalizationBackwardOutputs, LayerNormalizationConfig, LayerNormalizationInputs,
            LayerNormalizationOutputs, NormalizationSavedStatistics, NormalizationStatistics,
            PointwiseOperation,
        },
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
    math::NanPropagation,
};

const PROBLEM_DIMENSIONS: [i64; 3] = [4, 1024, 128];

struct SavedLayerNormRelu {
    input: DeviceMemory<f32>,
    scale: DeviceMemory<f32>,
    mean: DeviceMemory<f32>,
    inv_variance: DeviceMemory<f32>,
    bitmask: DeviceMemory<u8>,
}

fn run_forward(example: &common::ExampleContext) -> Result<SavedLayerNormRelu> {
    let [batch_size, seq_length, hidden_size] = PROBLEM_DIMENSIONS;

    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F32)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );

    let hidden_layout = Shape::with_strides(
        [batch_size * seq_length, hidden_size, 1, 1],
        [hidden_size, 1, hidden_size, hidden_size],
    )?;
    let reduction_layout = Shape::with_strides([batch_size * seq_length, 1, 1, 1], [1, 1, 1, 1])?;
    let scale_layout = Shape::with_strides(
        [1, hidden_size, 1, 1],
        vec![hidden_size, 1, hidden_size, hidden_size],
    )?;

    let x = graph.tensor(TensorSpec::new(DataType::F32, hidden_layout.clone()));
    let scale = graph.tensor(TensorSpec::new(DataType::F32, scale_layout.clone()));
    let bias = graph.tensor(TensorSpec::new(DataType::F32, scale_layout));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5));
    let relu_lower_bound = graph.tensor(TensorSpec::scalar_f32(0.0));
    let relu_upper_bound = graph.tensor(TensorSpec::scalar_f32(6.0));

    let ln_output =
        graph.tensor(TensorSpec::new(DataType::F32, hidden_layout.clone()).virtual_tensor());
    let mean = graph.tensor(TensorSpec::new(DataType::F32, reduction_layout.clone()));
    let inv_variance = graph.tensor(TensorSpec::new(DataType::F32, reduction_layout));
    graph.layer_normalization_into(
        LayerNormalizationInputs {
            input: x,
            affine: LayerNormalizationAffine::ScaleBias { scale, bias },
        },
        LayerNormalizationOutputs {
            output: ln_output,
            statistics: NormalizationStatistics::Training(NormalizationSavedStatistics {
                mean,
                inv_variance,
            }),
        },
        LayerNormalizationConfig::new(epsilon),
    )?;

    let y = graph.tensor(TensorSpec::new(DataType::F32, hidden_layout.clone()));
    graph.pointwise_into(PointwiseOperation::ActivationForward {
        name: None,
        input: ln_output,
        output: y,
        compute_type: Some(DataType::F32),
        nan_propagation: NanPropagation::Propagate,
        activation: Activation::Relu {
            lower_clip: 0.0,
            upper_clip: 6.0,
            lower_clip_slope: 0.0,
        },
        alpha1: 1.0,
    })?;

    let lower_mask =
        graph.tensor(TensorSpec::new(DataType::Boolean, hidden_layout.clone()).virtual_tensor());
    graph.pointwise_into(PointwiseOperation::Binary {
        name: None,
        mode: PointwiseMode::CmpGt,
        lhs: y,
        rhs: relu_lower_bound,
        output: lower_mask,
        compute_type: Some(DataType::F32),
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    })?;

    let upper_mask =
        graph.tensor(TensorSpec::new(DataType::Boolean, hidden_layout.clone()).virtual_tensor());
    graph.pointwise_into(PointwiseOperation::Binary {
        name: None,
        mode: PointwiseMode::CmpLt,
        lhs: y,
        rhs: relu_upper_bound,
        output: upper_mask,
        compute_type: Some(DataType::F32),
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    })?;

    let bitmask = graph.tensor(TensorSpec::new(DataType::Boolean, hidden_layout));
    graph.pointwise_into(PointwiseOperation::Binary {
        name: None,
        mode: PointwiseMode::LogicalAnd,
        lhs: lower_mask,
        rhs: upper_mask,
        output: bitmask,
        compute_type: Some(DataType::Boolean),
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    })?;

    for output in [y, bitmask, mean, inv_variance] {
        graph.mark_as_output(output)?;
    }

    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A, HeuristicMode::Fallback]),
    )?;
    let input_values = (0..graph.shape(x)?.storage_span()?)
        .map(|index| ((index % 257) as f32 - 128.0) / 64.0)
        .collect::<Vec<_>>();
    let x_dev = DeviceMemory::from_slice(example.cudnn.cuda_context(), &input_values)?;
    let scale_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &vec![1.0_f32; graph.shape(scale)?.storage_span()?],
    )?;
    let bias_dev = DeviceMemory::<f32>::zeroed(
        example.cudnn.cuda_context(),
        graph.shape(bias)?.storage_span()?,
    )?;
    let mut y_dev = DeviceMemory::<f32>::zeroed(
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
    let mut bitmask_dev = DeviceMemory::<u8>::zeroed(
        example.cudnn.cuda_context(),
        graph.tensor_spec(bitmask)?.storage_byte_len(None)?,
    )?;
    let mut workspace = common::workspace(example.cudnn.cuda_context(), compiled.workspace_size())?;

    let mut bindings = compiled.bindings()?;
    bindings.set_input(x, &x_dev)?;
    bindings.set_input(scale, &scale_dev)?;
    bindings.set_input(bias, &bias_dev)?;
    bindings.set(y, &mut y_dev)?;
    bindings.set(mean, &mut mean_dev)?;
    bindings.set(inv_variance, &mut inv_variance_dev)?;
    bindings.set_encoded_bytes(bitmask, &mut bitmask_dev)?;
    // SAFETY: matching bindings, buffers and workspace remain live until the helper
    // completes the cuDNN stream, including when native execution returns an error.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;

    println!(
        "forward saved statistics and {} packed mask bytes",
        bitmask_dev.len()
    );
    Ok(SavedLayerNormRelu {
        input: x_dev,
        scale: scale_dev,
        mean: mean_dev,
        inv_variance: inv_variance_dev,
        bitmask: bitmask_dev,
    })
}

fn run_backward(example: &common::ExampleContext, saved: &SavedLayerNormRelu) -> Result<()> {
    let [batch_size, seq_length, hidden_size] = PROBLEM_DIMENSIONS;

    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F32)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );

    let hidden_layout = Shape::with_strides(
        [batch_size * seq_length, hidden_size, 1, 1],
        [hidden_size, 1, hidden_size, hidden_size],
    )?;
    let reduction_layout = Shape::with_strides([batch_size * seq_length, 1, 1, 1], [1, 1, 1, 1])?;
    let scale_layout = Shape::with_strides(
        [1, hidden_size, 1, 1],
        vec![hidden_size, 1, hidden_size, hidden_size],
    )?;

    let x = graph.tensor(TensorSpec::new(DataType::F32, hidden_layout.clone()));
    let dy = graph.tensor(TensorSpec::new(DataType::F32, hidden_layout.clone()));
    let scale = graph.tensor(TensorSpec::new(DataType::F32, scale_layout.clone()));
    let mean = graph.tensor(TensorSpec::new(DataType::F32, reduction_layout.clone()));
    let inv_variance = graph.tensor(TensorSpec::new(DataType::F32, reduction_layout));
    let mask = graph.tensor(TensorSpec::new(DataType::Boolean, hidden_layout.clone()));

    let masked_dy =
        graph.tensor(TensorSpec::new(DataType::F32, hidden_layout.clone()).virtual_tensor());
    graph.pointwise_into(PointwiseOperation::Binary {
        name: None,
        mode: PointwiseMode::Mul,
        lhs: dy,
        rhs: mask,
        output: masked_dy,
        compute_type: Some(DataType::F32),
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    })?;

    let dx = graph.tensor(TensorSpec::new(DataType::F32, hidden_layout.clone()));
    let dscale = graph.tensor(TensorSpec::new(DataType::F32, scale_layout.clone()));
    let bias_gradient = graph.tensor(TensorSpec::new(DataType::F32, scale_layout));
    graph.layer_normalization_backward_into(
        LayerNormalizationBackwardInputs {
            input: x,
            statistics: NormalizationSavedStatistics { mean, inv_variance },
            output_gradient: masked_dy,
            scale: Some(scale),
        },
        LayerNormalizationBackwardOutputs {
            input_gradient: dx,
            affine: Some(LayerNormalizationAffineGradients {
                scale_gradient: dscale,
                bias_gradient,
            }),
        },
        LayerNormalizationBackwardConfig::new(),
    )?;

    for output in [dx, dscale, bias_gradient] {
        graph.mark_as_output(output)?;
    }

    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A, HeuristicMode::Fallback]),
    )?;
    let dy_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &vec![1.0_f32; graph.shape(dy)?.storage_span()?],
    )?;
    let mut dx_dev = DeviceMemory::<f32>::zeroed(
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
    bindings.set_input(x, &saved.input)?;
    bindings.set_input(dy, &dy_dev)?;
    bindings.set_input(scale, &saved.scale)?;
    bindings.set_input(mean, &saved.mean)?;
    bindings.set_input(inv_variance, &saved.inv_variance)?;
    bindings.set_encoded_input_bytes(mask, &saved.bitmask)?;
    bindings.set(dx, &mut dx_dev)?;
    bindings.set(dscale, &mut dscale_dev)?;
    bindings.set(bias_gradient, &mut bias_gradient_dev)?;
    // SAFETY: matching bindings, buffers and workspace remain live until the helper
    // completes the cuDNN stream, including when native execution returns an error.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;

    println!("backward consumed the saved statistics and packed ReLU mask");
    Ok(())
}

fn run() -> Result<()> {
    let context = common::ExampleContext::create()?;
    let saved = run_forward(&context)?;
    run_backward(&context, &saved)
}

fn main() -> Result<()> {
    common::finish("frontend_layer_norm_bitmask_relu", run())
}
