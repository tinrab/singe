// `rfft` and `irfft` as captured cuFFT transforms, whose plans the compiled graph keeps, since the captured kernels read their twiddle tables and work areas.
// cuFFT's inverse overwrites its input and leaves the result unnormalized, so an inverse first writes the spectrum, divided by the signal's length, into a buffer the graph keeps, and transforms that.

use singe_cuda::{
    graph::{Graph as NativeCudaGraph, GraphNode},
    stream::StreamCaptureMode,
    types::{Complex32, Complex64},
};
use singe_cufft::{
    plan::{Plan, PlanBuilder},
    types::TransformType,
};

use crate::{
    backend::{
        cuda::{
            self, CudaStream,
            compiled::{
                build::{
                    attach::attach_library_graph,
                    provider::{GraphProviderLowering, GraphProviderRequest, try_provider},
                },
                executable::GraphPrivateBuffers,
                plan::lowering::CudaGraphProvider,
            },
            device_views::{tensor_input_view, tensor_output_view_with_len},
            kernel_plan::{self, KernelInput, KernelPlanner},
            tensor_region::CudaTensorSource,
        },
        registry::{Operation, OperationAttributes},
        spec,
    },
    data_type::DataType,
    error::{Error, Result},
    graph::ir::ValueId,
    scalar::Scalar,
    shape::Shape,
    tensor::Tensor,
    utility::to_usize,
};

pub fn try_add_fft_provider_node(
    request: &GraphProviderRequest<'_>,
    operation: Operation,
    input: ValueId,
    lowering: &mut GraphProviderLowering<'_>,
) -> Result<Option<GraphNode>> {
    try_provider(
        request,
        lowering,
        CudaGraphProvider::Cufft,
        |lowering| build_fft_graph(request, operation, input, lowering.private),
        |(child_graph, plan), lowering| {
            attach_library_graph(
                lowering.native_graph,
                request.dependencies,
                lowering.realized_lowering,
                lowering.library_resources,
                request.ir,
                request.id,
                CudaGraphProvider::Cufft,
                child_graph,
                Some(plan),
            )
        },
    )
}

fn extent(value: i64) -> Result<usize> {
    to_usize(value, "cuda graph fft extent")
}

fn build_fft_graph(
    request: &GraphProviderRequest<'_>,
    operation: Operation,
    input: ValueId,
    private: &mut GraphPrivateBuffers,
) -> Result<(NativeCudaGraph, Plan)> {
    let input = request
        .cache
        .get(&input)
        .ok_or_else(|| Error::missing_input("build fft graph"))?;
    let output = request.output;
    let device = output.device();
    let data_type = output.data_type();
    let dimensions = input.shape().dimensions();
    // The signal's length, along the last axis of the forward transform's input and of the inverse's output.
    let length = match operation {
        Operation::Rfft => dimensions.last().copied().unwrap_or(0),
        Operation::Irfft => output.shape().dimensions().last().copied().unwrap_or(0),
        _ => return Err(Error::unsupported_operation("cuda graph fft")),
    };
    if !spec::transforms_with_cufft(length, data_type) || !output.is_contiguous() {
        return Err(Error::unsupported_operation(
            "cuda graph fft of these operands",
        ));
    }

    let signals = match operation {
        Operation::Rfft => input.element_count(),
        _ => output.element_count(),
    };
    let batch = signals / extent(length)?;
    let context = cuda::execution_context(device)?;
    let double = data_type == DataType::F64;
    let kind = match (operation, double) {
        (Operation::Rfft, false) => TransformType::RealToComplex,
        (Operation::Rfft, true) => TransformType::DoubleRealToDoubleComplex,
        (_, false) => TransformType::ComplexToReal,
        (_, true) => TransformType::DoubleComplexToDoubleReal,
    };
    let plan = PlanBuilder::new(context.allocator().context())
        .one_dimensional(extent(length)?, kind)
        .batch(batch)
        .create()?;
    let stream = CudaStream::create(device)?;
    let graph = if operation == Operation::Rfft {
        if !input.is_contiguous() {
            return Err(Error::ContiguousRequired);
        }

        let pairs = output.element_count() / 2;
        let (graph, ()) = stream.capture(StreamCaptureMode::ThreadLocal, || {
            // SAFETY: the plan transforms the contiguous signals into as many half spectra, which the output holds; the compiled graph keeps both and the plan alive while it replays.
            unsafe {
                if double {
                    plan.execute_d2z_f64_async(
                        &tensor_input_view::<f64>(input, DataType::F64, "cuda graph rfft input")?,
                        tensor_output_view_with_len::<Complex64>(
                            output.region()?,
                            DataType::F64,
                            pairs,
                            "cuda graph rfft output",
                        )?,
                        Some(stream.raw()),
                    )?;
                } else {
                    plan.execute_r2c_f32_async(
                        &tensor_input_view::<f32>(input, DataType::F32, "cuda graph rfft input")?,
                        tensor_output_view_with_len::<Complex32>(
                            output.region()?,
                            DataType::F32,
                            pairs,
                            "cuda graph rfft output",
                        )?,
                        Some(stream.raw()),
                    )?;
                }
            }
            Ok(())
        })?;
        graph
    } else {
        // The half spectrum a length-`length` signal has, divided by the length, in a buffer the graph keeps.
        let needed = length / 2 + 1;
        let frequencies = dimensions[dimensions.len() - 2];
        if frequencies < needed {
            return Err(Error::unsupported_operation(
                "cuda graph irfft of fewer frequencies than its signal has",
            ));
        }

        let mut spectrum = dimensions.to_vec();
        let axis = spectrum.len() - 2;
        spectrum[axis] = needed;
        let buffer = Tensor::zeroed_on(device, data_type, &Shape::create(spectrum)?)?;
        let narrowed = input.layout().slice(-2, 0, needed, 1)?;
        #[expect(
            clippy::cast_precision_loss,
            reason = "signal lengths are far below 2^53"
        )]
        let inverse = 1.0 / length as f64;
        let scale = KernelPlanner::default().create(
            Operation::Scale,
            &OperationAttributes::Scalar(Scalar::Float(inverse)),
            &[KernelInput::new(data_type, narrowed)],
        )?;
        for kernel in scale.kernels() {
            request.runtime.kernel(kernel)?;
        }

        let pairs = buffer.element_count() / 2;
        let points = output.element_count();
        let (graph, ()) = stream.capture(StreamCaptureMode::ThreadLocal, || {
            let scaling = cuda::execution_context(device)?;
            kernel_plan::launch(&scale, &[input], &scaling, &[buffer.tensor_region()?])?;
            let region = buffer.tensor_region()?;
            // SAFETY: the plan transforms the scaled half spectra, which only this node reads, into as many signals, which the output holds; the compiled graph keeps the buffer, the output, and the plan alive while it replays.
            unsafe {
                if double {
                    plan.execute_z2d_f64_async(
                        tensor_output_view_with_len::<Complex64>(
                            region,
                            DataType::F64,
                            pairs,
                            "cuda graph irfft spectrum",
                        )?,
                        tensor_output_view_with_len::<f64>(
                            output.region()?,
                            DataType::F64,
                            points,
                            "cuda graph irfft output",
                        )?,
                        Some(stream.raw()),
                    )?;
                } else {
                    plan.execute_c2r_f32_async(
                        tensor_output_view_with_len::<Complex32>(
                            region,
                            DataType::F32,
                            pairs,
                            "cuda graph irfft spectrum",
                        )?,
                        tensor_output_view_with_len::<f32>(
                            output.region()?,
                            DataType::F32,
                            points,
                            "cuda graph irfft output",
                        )?,
                        Some(stream.raw()),
                    )?;
                }
            }
            Ok(())
        })?;
        private.scratch.push(buffer);
        graph
    };
    Ok((graph.into_native(), plan))
}
