mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, bf16},
    error::Result,
    frontend::{
        composite::sdpa::{
            SdpaBackwardInputs, SdpaForwardOutputTensors, SdpaGradientOutputConfig, SdpaInputs,
        },
        error::Error as FrontendError,
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::{
            AttentionBackwardConfig, AttentionConfig, RopeBackwardInputs, RopeConfig, RopeInputs,
        },
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
};

fn run() -> Result<()> {
    let example = common::ExampleContext::create()?;
    let cuda = example.cudnn.cuda_context();
    let sequence = 128_i64;
    let head_dimension = 128_i64;
    let rotation_dimension = 64_i64;
    let graph_config = GraphConfig::new().with_data_type_policy(
        DataTypePolicy::new()
            .with_io(DataType::BF16)
            .with_intermediate(DataType::F32)
            .with_compute(DataType::F32),
    );
    let activation = TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([2, 4, sequence, head_dimension])?,
    );
    let frequency_spec = TensorSpec::new(
        DataType::F32,
        Shape::contiguous([sequence, 1, 1, rotation_dimension])?,
    );
    // Fold the usual attention scale into Q's rotation, including its unrotated prefix.
    // K keeps unit scale, and both attention graphs use an attention scale of one.
    let query_rotation = RopeConfig {
        rotation_dimension,
        output_scale: 1.0 / (head_dimension as f32).sqrt(),
        ..RopeConfig::default()
    }
    .with_name("query_rotation");
    let key_rotation = RopeConfig {
        rotation_dimension,
        ..RopeConfig::default()
    }
    .with_name("key_rotation");
    let mut forward = Graph::with_config(graph_config.clone());
    let query = forward.tensor(activation.clone().with_name("q"));
    let key = forward.tensor(activation.clone().with_name("k"));
    let value = forward.tensor(activation.clone().with_name("v"));
    let frequencies = forward.tensor(frequency_spec.clone().with_name("raw_angles"));
    let rotated_query = forward.rope_forward(
        RopeInputs::new(query, frequencies),
        query_rotation.clone(),
        TensorOutputConfig::new().output_tensor(),
    )?;
    let rotated_key = forward.rope_forward(
        RopeInputs::new(key, frequencies),
        key_rotation.clone(),
        TensorOutputConfig::new().output_tensor(),
    )?;
    let scale = forward.tensor(TensorSpec::scalar_f32(1.0));
    let outputs = forward.sdpa_auto(
        SdpaInputs {
            query: rotated_query,
            key: rotated_key,
            value,
            scale,
        },
        AttentionConfig::new(),
        SdpaForwardOutputTensors::new(TensorOutputConfig::new().output_tensor())
            .with_stats(TensorOutputConfig::new().output_tensor()),
    )?;
    let output = outputs.output;
    let stats = outputs.stats.ok_or(FrontendError::OperationFieldMissing {
        operation: "rope sdpa training",
        field: "stats",
    })?;
    forward.mark_as_output(output)?;
    forward.mark_as_output(stats)?;
    // RoPE outputs already have device placement and need no mark_as_output call.
    let compiled_forward = forward.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
    )?;

    let elements = activation.shape.storage_span()?;
    let mut seed = 123_456_789;
    let query_device = DeviceMemory::from_slice(
        cuda,
        &common::init_f32_image(elements, &mut seed)
            .into_iter()
            .map(bf16::from_f32)
            .collect::<Vec<_>>(),
    )?;
    let key_device = DeviceMemory::from_slice(
        cuda,
        &common::init_f32_image(elements, &mut seed)
            .into_iter()
            .map(bf16::from_f32)
            .collect::<Vec<_>>(),
    )?;
    let value_device = DeviceMemory::from_slice(
        cuda,
        &common::init_f32_image(elements, &mut seed)
            .into_iter()
            .map(bf16::from_f32)
            .collect::<Vec<_>>(),
    )?;
    let angles = (0..sequence)
        .flat_map(|position| {
            (0..rotation_dimension).map(move |index| {
                let pair = index % (rotation_dimension / 2);
                position as f32 / 10_000.0_f32.powf(2.0 * pair as f32 / rotation_dimension as f32)
            })
        })
        .collect::<Vec<_>>();
    let frequency_device = DeviceMemory::from_slice(cuda, &angles)?;
    let mut rotated_query_device = DeviceMemory::<bf16>::zeroed(cuda, elements)?;
    let mut rotated_key_device = DeviceMemory::<bf16>::zeroed(cuda, elements)?;
    let mut output_device = DeviceMemory::<bf16>::zeroed(cuda, elements)?;
    let mut stats_device =
        DeviceMemory::<f32>::zeroed(cuda, forward.shape(stats)?.storage_span()?)?;
    let mut forward_workspace = common::workspace(cuda, compiled_forward.workspace_size())?;
    let mut forward_bindings = compiled_forward.bindings()?;
    for (id, memory) in [
        (query, &query_device),
        (key, &key_device),
        (value, &value_device),
    ] {
        forward_bindings.set_input(id, memory)?;
    }
    for (id, memory) in [
        (rotated_query, &mut rotated_query_device),
        (rotated_key, &mut rotated_key_device),
        (output, &mut output_device),
    ] {
        forward_bindings.set(id, memory)?;
    }
    forward_bindings.set_input(frequencies, &frequency_device)?;
    forward_bindings.set(stats, &mut stats_device)?;
    // SAFETY: bindings have the declared encodings and capacities, and all owners
    // remain live through stream completion, including when execution returns an error.
    example.execute_and_wait(|context| unsafe {
        compiled_forward.execute(
            context,
            ExecuteArgs::new(&forward_bindings).with_workspace(&mut forward_workspace),
        )
    })?;

    let mut backward = Graph::with_config(graph_config);
    let saved_query = backward.tensor(forward.tensor_spec(rotated_query)?.clone());
    let saved_key = backward.tensor(forward.tensor_spec(rotated_key)?.clone());
    let saved_value = backward.tensor(activation.clone());
    let saved_output = backward.tensor(forward.tensor_spec(output)?.clone());
    let saved_stats = backward.tensor(forward.tensor_spec(stats)?.clone());
    let saved_frequencies = backward.tensor(frequency_spec);
    let output_gradient = backward.tensor(activation);
    let backward_scale = backward.tensor(TensorSpec::scalar_f32(1.0));
    let gradients = backward.sdpa_backward(
        SdpaBackwardInputs {
            query: saved_query,
            key: saved_key,
            value: saved_value,
            output: saved_output,
            output_gradient,
            stats: saved_stats,
            scale: backward_scale,
        },
        AttentionBackwardConfig::new(),
        SdpaGradientOutputConfig {
            query_gradient: TensorOutputConfig::new().with_data_type(DataType::BF16),
            key_gradient: TensorOutputConfig::new().with_data_type(DataType::BF16),
            value_gradient: TensorOutputConfig::new().output_tensor(),
            ..Default::default()
        },
    )?;
    // Each transpose rotation uses its corresponding forward scale. Q needs the folded
    // attention factor here; K's gradient already carries that factor through saved Q.
    let query_gradient = backward.rope_backward(
        RopeBackwardInputs::new(gradients.query_gradient, saved_frequencies),
        query_rotation.with_name("query_rotation_backward"),
        TensorOutputConfig::new().output_tensor(),
    )?;
    let key_gradient = backward.rope_backward(
        RopeBackwardInputs::new(gradients.key_gradient, saved_frequencies),
        key_rotation.with_name("key_rotation_backward"),
        TensorOutputConfig::new().output_tensor(),
    )?;
    let value_gradient = gradients.value_gradient;
    let compiled_backward = backward.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
    )?;
    let output_gradient_device =
        DeviceMemory::from_slice(cuda, &vec![bf16::from_f32(1.0); elements])?;
    let mut query_gradient_device = DeviceMemory::<bf16>::zeroed(cuda, elements)?;
    let mut key_gradient_device = DeviceMemory::<bf16>::zeroed(cuda, elements)?;
    let mut value_gradient_device = DeviceMemory::<bf16>::zeroed(cuda, elements)?;
    let mut backward_workspace = common::workspace(cuda, compiled_backward.workspace_size())?;
    let mut backward_bindings = compiled_backward.bindings()?;
    for (id, memory) in [
        (saved_query, &rotated_query_device),
        (saved_key, &rotated_key_device),
        (saved_value, &value_device),
        (saved_output, &output_device),
        (output_gradient, &output_gradient_device),
    ] {
        backward_bindings.set_input(id, memory)?;
    }
    for (id, memory) in [
        (query_gradient, &mut query_gradient_device),
        (key_gradient, &mut key_gradient_device),
        (value_gradient, &mut value_gradient_device),
    ] {
        backward_bindings.set(id, memory)?;
    }
    backward_bindings.set_input(saved_stats, &stats_device)?;
    backward_bindings.set_input(saved_frequencies, &frequency_device)?;
    // SAFETY: saved tensors contain the completed forward results, gradient inputs are
    // initialized, and buffers and plans remain live through completion on success or error.
    example.execute_and_wait(|context| unsafe {
        compiled_backward.execute(
            context,
            ExecuteArgs::new(&backward_bindings).with_workspace(&mut backward_workspace),
        )
    })?;
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_rope_sdpa", run())
}
