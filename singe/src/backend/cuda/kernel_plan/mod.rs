mod source;

use std::ptr;

use singe_cuda::{
    context::Context,
    graph::{Graph as NativeGraph, GraphNode},
    stream::StreamCaptureStatus,
};

pub use self::source::{
    KernelInput, KernelParameters, KernelPlan, KernelPlanner, QueryRotation, RotaryCacheInputs,
    RotationNorm, cudnn_paging, parameter_key, plan_cudnn_paging_indices,
    plan_cudnn_paging_results, plan_cudnn_query_lengths, value_type,
};
use crate::{
    backend::{
        cuda::{
            self,
            compiled::executable::GraphPrivateBuffers,
            context::CudaExecutionContext,
            device_views::kernel_stream_for_context,
            graph_value::CudaGraphValue,
            stream::ensure_not_stream_capturing,
            tensor_region::{CudaTensorRegion, CudaTensorSource, PendingCudaTensor},
        },
        kernel_lowering::check_status_code,
        registry::{Operation, OperationAttributes},
    },
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    shape::Shape,
    tensor::Tensor,
    utility::to_i64,
};

pub fn execute(
    operation: Operation,
    attributes: &OperationAttributes,
    inputs: &[&Tensor],
) -> Result<Tensor> {
    execute_with(&KernelPlanner::default(), operation, attributes, inputs)
}

/// Runs `operation` on `inputs` through a kernel `planner` plans.
pub fn execute_with(
    planner: &KernelPlanner,
    operation: Operation,
    attributes: &OperationAttributes,
    inputs: &[&Tensor],
) -> Result<Tensor> {
    let metadata = inputs
        .iter()
        .map(|input| KernelInput::from_source(*input))
        .collect::<Vec<_>>();
    let plan = planner.create(operation, attributes, &metadata)?;
    run(&plan, inputs)
}

/// Runs `plan` on `inputs`, which share a device, into a new output tensor.
pub fn run(plan: &KernelPlan, inputs: &[&Tensor]) -> Result<Tensor> {
    let first = inputs
        .first()
        .ok_or_else(|| Error::missing_input("kernel plan run"))?;
    let output = NewOutput::create(
        first.device(),
        plan.output().data_type(),
        plan.output().shape(),
    )?;
    if !output.is_empty() {
        launch(plan, inputs, output.context(), &[output.output()])?;
    }
    // SAFETY: the producer completed every logical element, or the output is empty.
    Ok(unsafe { output.into_tensor() })
}

// Runs `plan` on `inputs`, which share a device, into a new tensor for each output the plan declares.
pub fn run_outputs(plan: &KernelPlan, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
    let first = inputs
        .first()
        .ok_or_else(|| Error::missing_input("run outputs"))?;
    ensure_not_stream_capturing(
        first.device(),
        "cuda eager output allocation during stream capture",
    )?;
    let context = cuda::execution_context(first.device())?;
    context.bind()?;
    let outputs = plan
        .outputs
        .iter()
        .map(|spec| context.allocate_output(spec.data_type(), spec.shape()))
        .collect::<Result<Vec<_>>>()?;
    if outputs.iter().any(|output| !output.is_empty()) {
        let regions = outputs
            .iter()
            .map(PendingCudaTensor::region)
            .collect::<Vec<_>>();
        launch(plan, inputs, &context, &regions)?;
    }
    // SAFETY: the producer completed every logical element of each output, or the outputs are empty.
    Ok(outputs
        .into_iter()
        .map(|output| unsafe { output.assume_init() })
        .collect())
}

/// Runs each plan on its inputs into one new contiguous `data_type` tensor of `shape`, on the device of `device_source`.
///
/// Together the plans must write every element of the output, as the per-input plans of a concatenation do.
pub fn run_steps(
    steps: &[(KernelPlan, Vec<&Tensor>)],
    device_source: &Tensor,
    data_type: DataType,
    shape: &Shape,
) -> Result<Tensor> {
    let output = NewOutput::create(device_source.device(), data_type, shape)?;
    if !output.is_empty() {
        for (plan, inputs) in steps {
            launch(plan, inputs, output.context(), &[output.output()])?;
        }
    }
    // SAFETY: the steps together completed every logical element, or the output is empty.
    Ok(unsafe { output.into_tensor() })
}

/// Runs `plan` on `inputs`, writing into `destination`, which already holds the elements the plan does not write.
///
/// The caller has checked that `destination` does not overlap any input.
pub fn run_into(plan: &KernelPlan, inputs: &[&Tensor], destination: &Tensor) -> Result<()> {
    let context = cuda::execution_context(destination.device())?;
    launch(plan, inputs, &context, &[destination.tensor_region()?])
}

/// Copies `source` into the elements `destination` addresses and waits for the copy.
///
/// The caller has checked shape and data type equality, that `destination` has no internal overlap, and that the tensors do not overlap.
pub fn assign(destination: &Tensor, source: &Tensor) -> Result<()> {
    if destination.shape().element_count() == 0 {
        return Ok(());
    }

    let plan = KernelPlanner::default().assign(
        &KernelInput::from_source(destination),
        &KernelInput::from_source(source),
    )?;
    run_into(&plan, &[source], destination)
}

pub fn launch<S: CudaTensorSource + ?Sized>(
    plan: &KernelPlan,
    inputs: &[&S],
    context: &CudaExecutionContext,
    outputs: &[CudaTensorRegion<'_>],
) -> Result<()> {
    if plan.kernel.is_empty() {
        return Ok(());
    }

    for input in inputs {
        if input.device() != context.device() {
            return Err(Error::device_mismatch(
                "cuda kernel operands",
                context.device(),
                input.device(),
            ));
        }
    }

    let pointers = input_pointers(inputs)?;
    let destinations = outputs
        .iter()
        .map(|output| output.region().as_ptr())
        .collect::<Vec<_>>();
    let stream = kernel_stream_for_context(context, context.device())?;
    let capturing = stream.raw().capture_status()? != StreamCaptureStatus::None;
    // A captured launch would outlive scratch buffers this call frees when it returns.
    if capturing && !plan.scratch().is_empty() {
        return Err(Error::unsupported_operation(
            "cuda kernel with scratch buffers under stream capture",
        ));
    }

    let scratch = scratch_buffers(plan, context.device())?;
    let scratch_pointers = input_pointers(&scratch.iter().collect::<Vec<_>>())?;
    // Kernels report data errors through a status word that eager calls read after they synchronize; under stream capture nobody could read it, so it is null.
    let status = if plan.kernels().any(|kernel| kernel.abi.checks()) && !capturing {
        Some(Tensor::zeroed_on(
            context.device(),
            DataType::I32,
            &Shape::create([1])?,
        )?)
    } else {
        None
    };
    let status_pointer = match &status {
        Some(status) => status.tensor_region()?.region().as_ptr(),
        None => ptr::null_mut(),
    };
    let mut launches = Vec::new();
    for kernel in plan.kernels().filter(|kernel| !kernel.is_empty()) {
        let arguments =
            kernel
                .abi
                .arguments(&pointers, &destinations, &scratch_pointers, status_pointer)?;
        launches.push((
            context.runtime().kernel(kernel)?,
            kernel.launch_config()?,
            arguments,
        ));
    }

    // SAFETY: the plan's checked layouts cover every generated address.
    // The draining scope retains input/output allocations and the loaded functions, and the scratch buffers live until it has waited for the launches, which run in order on one stream.
    unsafe {
        stream.complete_or_record(inputs.iter().copied(), outputs, || {
            for (function, config, arguments) in launches {
                function.launch_on(&config, arguments, stream.raw())?;
            }
            Ok(())
        })?;
    }
    status.map_or(Ok(()), |status| check_status(&status))
}

/// Reads the status word that checking kernels of a compiled graph wrote, clearing it for the next replay, and returns the error it records.
pub fn take_status(status: &Tensor) -> Result<()> {
    let result = check_status(status);
    if result.is_err() {
        assign(
            status,
            &Tensor::zeroed_on(status.device(), DataType::I32, status.shape())?,
        )?;
    }
    result
}

/// Converts the status word a checked kernel wrote into its error.
fn check_status(status: &Tensor) -> Result<()> {
    let code = status.to_vec::<i32>()?.first().copied().unwrap_or(0);
    // The status word holds a code's bits, so a negative value is an unknown code too.
    check_status_code(code.cast_unsigned())
}

// Allocates the zeroed scratch buffers `plan`'s kernels share on `device`.
fn scratch_buffers(plan: &KernelPlan, device: Device) -> Result<Vec<Tensor>> {
    plan.scratch()
        .iter()
        .map(|&bytes| {
            let bytes = to_i64(bytes.max(1), "cuda kernel scratch bytes")?;
            Tensor::zeroed_on(device, DataType::U8, &Shape::create([bytes])?)
        })
        .collect()
}

fn input_pointers<S: CudaTensorSource + ?Sized>(inputs: &[&S]) -> Result<Vec<*mut u8>> {
    inputs
        .iter()
        .map(|input| Ok(input.tensor_region()?.region().as_ptr()))
        .collect()
}

/// One kernel launch of a native graph built by [`build_graph_steps`].
pub struct GraphStep<'a> {
    pub plan: &'a KernelPlan,
    pub inputs: Vec<&'a dyn CudaTensorSource>,
}

/// Builds a native graph that launches `steps` in order, each writing `outputs`, which the last step's plan describes.
///
/// Checking kernels report through the compiled graph's status word, and the scratch buffers kernels declare join its `private` buffers, which replays reuse one after another.
pub fn build_graph_steps(
    steps: &[GraphStep<'_>],
    outputs: &[&CudaGraphValue],
    private: &mut GraphPrivateBuffers,
) -> Result<NativeGraph> {
    let last = steps
        .last()
        .ok_or_else(|| Error::missing_node("build graph steps"))?;
    let first = outputs
        .first()
        .ok_or_else(|| Error::missing_node("build graph steps"))?;
    if outputs.len() != last.plan.outputs.len()
        || outputs
            .iter()
            .zip(&last.plan.outputs)
            .any(|(output, spec)| {
                output.data_type() != spec.data_type()
                    || output.shape() != spec.shape()
                    || !output.is_contiguous()
            })
    {
        return Err(Error::invalid_argument(
            "cuda general kernel output metadata",
        ));
    }

    let context = cuda::execution_context(first.device())?;
    let runtime = context.runtime();
    let mut graph = runtime.context().create_graph()?;
    let mut destinations = Vec::with_capacity(outputs.len());
    for output in outputs {
        validate_graph_region(output.region()?, runtime.context())?;
        destinations.push(output.region()?.region().as_ptr());
    }

    let mut previous: Option<GraphNode> = None;
    for step in steps {
        if step.plan.kernel.is_empty() {
            continue;
        }

        for input in &step.inputs {
            validate_graph_region(input.tensor_region()?, runtime.context())?;
        }

        let pointers = input_pointers(&step.inputs)?;
        let scratch = scratch_buffers(step.plan, first.device())?;
        let scratch_pointers = input_pointers(&scratch.iter().collect::<Vec<_>>())?;
        private.scratch.extend(scratch);
        for kernel in step.plan.kernels().filter(|kernel| !kernel.is_empty()) {
            let function = runtime.kernel(kernel)?;
            let config = kernel.launch_config()?;
            // The compiled graph reads the status word after each replay completes.
            let status_pointer = if kernel.abi.checks() {
                private
                    .status(first.device())?
                    .tensor_region()?
                    .region()
                    .as_ptr()
            } else {
                ptr::null_mut()
            };
            let dependencies = previous.iter().cloned().collect::<Vec<_>>();
            let arguments = kernel.abi.arguments(
                &pointers,
                &destinations,
                &scratch_pointers,
                status_pointer,
            )?;
            // SAFETY: generated offsets use validated layouts.
            // The compiled executable retains these graph values and orders each read and write.
            // `add_to_graph` retains the native module independently of the runtime's bounded compilation cache.
            let node =
                unsafe { function.add_to_graph(&mut graph, &dependencies, &config, arguments)? };
            previous = Some(node);
        }
    }
    Ok(graph)
}

fn validate_graph_region(region: CudaTensorRegion<'_>, context: &Context) -> Result<()> {
    if region.region().context().as_ref() != context {
        return Err(Error::invalid_argument("cuda general kernel context"));
    }
    Ok(())
}

// A new output tensor with the bound context that fills it.
struct NewOutput {
    context: CudaExecutionContext,
    output: PendingCudaTensor,
}

impl NewOutput {
    fn create(device: Device, data_type: DataType, shape: &Shape) -> Result<Self> {
        ensure_not_stream_capturing(device, "cuda eager output allocation during stream capture")?;
        let context = cuda::execution_context(device)?;
        context.bind()?;
        let output = context.allocate_output(data_type, shape)?;
        Ok(Self { context, output })
    }

    fn context(&self) -> &CudaExecutionContext {
        &self.context
    }

    fn output(&self) -> CudaTensorRegion<'_> {
        self.output.region()
    }

    fn is_empty(&self) -> bool {
        self.output.is_empty()
    }

    // The caller must have completed a producer that wrote every element, unless the output is empty.
    unsafe fn into_tensor(self) -> Tensor {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { self.output.assume_init() }
    }
}

#[cfg(all(test, feature = "testing"))]
mod tests;
