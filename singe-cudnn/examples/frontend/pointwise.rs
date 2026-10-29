mod common;

use std::env::args;

use singe_cuda::{memory::DeviceMemory, view::OwnedDeviceRegion};
use singe_cudnn::{
    backend::{
        pointwise::PointwiseMode,
        reduction::ReduceTensorOperator,
        tensor::{Shape, TensorSpec},
    },
    data_type::{DataType, f8e4m3, f16},
    error::Result,
    frontend::{
        bindings::BindingAccess,
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::{Activation, PointwiseOperation, ReductionAxes, ReductionConfig},
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
    math::NanPropagation,
};

fn init_f16_image(size: usize, seed: &mut u32) -> Vec<f16> {
    common::init_f32_image(size, seed)
        .into_iter()
        .map(f16::from_f32)
        .collect()
}

fn run() -> Result<()> {
    let context = common::ExampleContext::create()?;
    if args().any(|argument| argument == "--index-grid") {
        return run_index_generation(&context);
    }
    if args().any(|argument| argument == "--broadcast-select") {
        return run_broadcast_select(&context);
    }
    if args().any(|argument| argument == "--inferred-activation") {
        return run_inferred_activation(&context);
    }

    run_reduction(&context)?;
    run_fused_scalar(&context)?;
    run_fused_absolute_max_type_conversion(&context)?;

    Ok(())
}

fn run_index_generation(example: &common::ExampleContext) -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 4, 8])?,
    ));
    let axis = if args().any(|argument| argument == "--default-axis") {
        None
    } else {
        Some(3)
    };
    let indices = graph.pointwise(
        PointwiseOperation::unary(
            PointwiseMode::GenIndex,
            input,
            TensorOutputConfig::new()
                .with_data_type(DataType::I32)
                .output_tensor(),
        )
        .with_axis(axis)?
        .with_compute_type(DataType::I32),
    )?;
    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
    )?;
    let cuda = example.cudnn.cuda_context();
    let input_dev = DeviceMemory::<f32>::zeroed(cuda, graph.shape(input)?.storage_span()?)?;
    let mut indices_dev = DeviceMemory::<i32>::zeroed(cuda, graph.shape(indices)?.storage_span()?)?;
    let mut workspace = common::workspace(cuda, compiled.workspace_size())?;
    let mut bindings = compiled.bindings()?;
    bindings.set_input(input, &input_dev.view())?;
    bindings.set(indices, &mut indices_dev)?;
    // SAFETY: initialized input, writable indices and workspace remain live through stream completion.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;
    Ok(())
}

fn run_inferred_activation(example: &common::ExampleContext) -> Result<()> {
    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F16)
                .with_intermediate(DataType::F16)
                .with_compute(DataType::F32),
        ),
    );
    let shape = Shape::contiguous([1, 1, 2, 32])?;
    let input = graph.tensor(TensorSpec::new(DataType::F16, shape.clone()));
    let output_gradient = graph.tensor_like(input, Some("dy"))?;
    let predicate = graph.tensor(TensorSpec::new(DataType::ByteBoolean, shape));
    let negated = graph.pointwise(
        PointwiseOperation::unary(PointwiseMode::Neg, input, TensorOutputConfig::new())
            .with_name("scaled_negation")
            .with_scaling(0.5, 1.0),
    )?;
    let activation = Activation::Swish { beta: 0.75 };
    let activated = graph.pointwise(
        PointwiseOperation::activation_forward(activation, negated, TensorOutputConfig::new())
            .with_name("swish"),
    )?;
    let activation_gradient = graph.pointwise(
        PointwiseOperation::activation_backward(
            activation,
            output_gradient,
            negated,
            TensorOutputConfig::new(),
        )
        .with_name("swish_backward"),
    )?;
    let input_gradient = graph.pointwise(
        PointwiseOperation::unary(
            PointwiseMode::Neg,
            activation_gradient,
            TensorOutputConfig::new().output_tensor(),
        )
        .with_name("input_gradient")
        .with_scaling(0.5, 1.0),
    )?;
    let selected = graph.pointwise(
        PointwiseOperation::ternary(
            PointwiseMode::BinarySelect,
            activated,
            negated,
            predicate,
            TensorOutputConfig::new().output_tensor(),
        )
        .with_name("selected"),
    )?;
    println!(
        "pointwise results: {:?}, {:?}",
        graph.tensor_spec(input_gradient)?.name,
        graph.tensor_spec(selected)?.name,
    );
    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A, HeuristicMode::Fallback]),
    )?;
    let cuda = example.cudnn.cuda_context();
    let mut seed = 123_456_789_u32;
    let input_device = DeviceMemory::from_slice(
        cuda,
        &init_f16_image(graph.shape(input)?.storage_span()?, &mut seed),
    )?;
    let gradient_device = DeviceMemory::from_slice(
        cuda,
        &vec![f16::from_f32(1.0); graph.shape(output_gradient)?.storage_span()?],
    )?;
    let predicate_values = (0..graph.shape(predicate)?.storage_span()?)
        .map(|index| index % 2 != 0)
        .collect::<Vec<_>>();
    let predicate_device = DeviceMemory::from_slice(cuda, &predicate_values)?;
    let mut bindings = compiled.bindings()?;
    bindings.set_input(input, &input_device)?;
    bindings.set_input(output_gradient, &gradient_device)?;
    bindings.set_input(predicate, &predicate_device)?;
    for output in [input_gradient, selected] {
        let bytes = graph.tensor_spec(output)?.storage_byte_len(None)?;
        let memory = DeviceMemory::<u8>::zeroed(cuda, bytes)?;
        bindings.set_owned_region(
            output,
            OwnedDeviceRegion::from_memory(&memory),
            BindingAccess::ReadWrite,
        )?;
    }
    let mut workspace = common::workspace(cuda, compiled.workspace_size())?;
    // SAFETY: initialized inputs, byte-addressed boolean predicates, retained writable outputs
    // and workspace remain live through stream completion, including enqueue failure.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;
    Ok(())
}

fn run_broadcast_select(example: &common::ExampleContext) -> Result<()> {
    let mut graph = Graph::new();
    let lhs = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 2, 1, 4])?,
    ));
    let rhs = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 1, 3, 4])?,
    ));
    let predicate = graph.tensor(TensorSpec::new(
        DataType::Boolean,
        Shape::with_strides([1, 2, 3, 4], [24, 12, 1, 3])?,
    ));
    // Only the predicate has the complete broadcast dimensions, so its strides select the output layout.
    let output = graph.pointwise(
        PointwiseOperation::ternary(
            PointwiseMode::BinarySelect,
            lhs,
            rhs,
            predicate,
            TensorOutputConfig::new().output_tensor(),
        )
        .with_name("broadcast_selection"),
    )?;
    println!("broadcast select output layout: {:?}", graph.shape(output)?);
    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A, HeuristicMode::Fallback]),
    )?;
    let cuda = example.cudnn.cuda_context();
    let lhs_device = DeviceMemory::from_slice(
        cuda,
        &vec![f16::from_f32(0.5); graph.shape(lhs)?.storage_span()?],
    )?;
    let rhs_device = DeviceMemory::from_slice(
        cuda,
        &vec![f16::from_f32(-0.5); graph.shape(rhs)?.storage_span()?],
    )?;
    let predicate_device = DeviceMemory::from_slice(
        cuda,
        &vec![0b1010_1010_u8; graph.tensor_spec(predicate)?.storage_byte_len(None)?],
    )?;
    let mut output_device =
        DeviceMemory::<f16>::zeroed(cuda, graph.shape(output)?.storage_span()?)?;
    let mut bindings = compiled.bindings()?;
    bindings.set_input(lhs, &lhs_device)?;
    bindings.set_input(rhs, &rhs_device)?;
    bindings.set_encoded_input_bytes(predicate, &predicate_device)?;
    bindings.set(output, &mut output_device)?;
    let mut workspace = common::workspace(cuda, compiled.workspace_size())?;
    // SAFETY: initialized inputs, packed predicate bytes, writable output and workspace
    // retain their declared physical spans through stream completion, including enqueue failure.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;
    Ok(())
}

fn run_reduction(example: &common::ExampleContext) -> Result<()> {
    let n = 64_i64;

    let mut graph = Graph::new();
    let a = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::with_strides([n, n, n, n], [n * n * n, 1, n * n, n])?,
    ));
    let mut config = ReductionConfig::new(ReduceTensorOperator::Max)
        .with_name("maximum")
        .with_axes(ReductionAxes::All)
        .with_deterministic(args().any(|argument| argument == "--deterministic"));
    if args().any(|argument| argument == "--unordered-reduction-axes") {
        config = config.with_axes(ReductionAxes::Dimensions(vec![3, 1]));
    }
    if args().any(|argument| argument == "--default-reduction-geometry") {
        config = config.with_axes(None);
    }
    let c = if args().any(|argument| argument == "--explicit-reduction") {
        // A row-major output reduces the last axis of the NHWC input.
        let output = graph.tensor(TensorSpec::new(
            DataType::F32,
            Shape::contiguous([n, n, n, 1])?,
        ));
        graph.reduction_into(a, output, &config)?;
        output
    } else {
        graph.reduction(a, &config, TensorOutputConfig::new().output_tensor())?
    };
    graph.mark_as_output(c)?;
    let input_elements = graph.shape(a)?.element_count()?;
    let output_elements = graph.shape(c)?.storage_span()?;

    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
    )?;

    let mut seed = 123_456_789_u32;
    let a_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &common::init_f32_image(input_elements, &mut seed),
    )?;
    let mut c_dev = DeviceMemory::<f32>::zeroed(example.cudnn.cuda_context(), output_elements)?;
    let mut workspace = common::workspace(example.cudnn.cuda_context(), compiled.workspace_size())?;
    let mut bindings = compiled.bindings()?;
    bindings.set_input(a, &a_dev.view())?;
    bindings.set(c, &mut c_dev)?;
    // SAFETY: initialized input, writable output and workspace remain live through stream completion.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;

    Ok(())
}

fn run_fused_scalar(example: &common::ExampleContext) -> Result<()> {
    let n = 4_i64;

    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F16)
                .with_compute(DataType::F32),
        ),
    );

    let a = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([n, n, n])?,
    ));
    let scalar_spec = if args().any(|argument| argument == "--constant-scalar") {
        TensorSpec::constant(5.0_f32)
    } else {
        TensorSpec::scalar(5.0_f32)
    };
    println!(
        "fused scalar: by value={}, runtime={}",
        scalar_spec.is_by_value(),
        scalar_spec.is_runtime_scalar(),
    );
    let scalar = graph.tensor(scalar_spec.with_byte_alignment(64)?);
    let c = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([n, n, n])?,
    ));
    graph.pointwise_into(PointwiseOperation::Binary {
        name: None,
        mode: PointwiseMode::Add,
        lhs: a,
        rhs: scalar,
        output: c,
        compute_type: Some(DataType::F32),
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        // The second operand contributes twice its stored scalar value.
        alpha2: 2.0,
    })?;
    graph.mark_as_output(c)?;

    let input_elements = graph.shape(a)?.storage_span()?;
    let output_elements = graph.shape(c)?.storage_span()?;
    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
    )?;

    let mut seed = 123_456_789_u32;
    let a_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_f16_image(input_elements, &mut seed),
    )?;
    let mut c_dev = DeviceMemory::<f16>::zeroed(example.cudnn.cuda_context(), output_elements)?;
    let mut workspace = common::workspace(example.cudnn.cuda_context(), compiled.workspace_size())?;
    let mut bindings = compiled.bindings()?;
    bindings.set_input(a, &a_dev.view())?;
    bindings.set(c, &mut c_dev)?;
    // SAFETY: inputs, writable output and workspace remain live through stream completion.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;

    Ok(())
}

fn run_fused_absolute_max_type_conversion(example: &common::ExampleContext) -> Result<()> {
    let n = 64_i64;

    let mut graph = Graph::new();
    let a = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::with_strides([n, n, n, n], [n * n * n, 1, n * n, n])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::with_strides([1, 1, 1, 1], [1, 1, 1, 1])?,
    ));

    let absolute_max = graph.reduction(
        a,
        &ReductionConfig::new(ReduceTensorOperator::AbsoluteMax)
            .with_compute_type(DataType::F32)
            .with_output_shape(graph.shape(scale)?.clone()),
        TensorOutputConfig::new()
            .with_data_type(DataType::F32)
            .output_tensor(),
    )?;
    graph.mark_as_output(absolute_max)?;

    let c = graph.pointwise(
        PointwiseOperation::binary(
            PointwiseMode::Mul,
            a,
            scale,
            TensorOutputConfig::new().with_data_type(DataType::F8E4M3),
        )
        .with_compute_type(None),
    )?;
    graph.mark_as_output(c)?;

    let input_elements = graph.shape(a)?.storage_span()?;
    let scale_elements = graph.shape(scale)?.storage_span()?;
    let amax_elements = graph.shape(absolute_max)?.storage_span()?;
    let output_elements = graph.shape(c)?.storage_span()?;
    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
    )?;

    let mut seed = 123_456_789_u32;
    let a_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &common::init_f32_image(input_elements, &mut seed),
    )?;
    let scale_dev =
        DeviceMemory::from_slice(example.cudnn.cuda_context(), &vec![1.0_f32; scale_elements])?;
    let mut absolute_max_dev =
        DeviceMemory::<f32>::zeroed(example.cudnn.cuda_context(), amax_elements)?;
    let mut c_dev = DeviceMemory::<f8e4m3>::zeroed(example.cudnn.cuda_context(), output_elements)?;
    let mut workspace = common::workspace(example.cudnn.cuda_context(), compiled.workspace_size())?;
    let mut bindings = compiled.bindings()?;
    bindings.set_input(a, &a_dev.view())?;
    bindings.set_input(scale, &scale_dev.view())?;
    bindings.set(absolute_max, &mut absolute_max_dev)?;
    bindings.set(c, &mut c_dev)?;
    // SAFETY: read-only inputs, writable outputs and workspace remain live through stream completion.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_pointwise", run())
}
