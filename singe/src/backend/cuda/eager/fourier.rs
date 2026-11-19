// Real Fourier transforms through cuFFT, whose cost grows as `n log n` where the basis matmul's grows as `n²`.
// cuFFT's interleaved complex values are the `[..., k, 2]` pairs Singe's transforms use, so its buffers are the tensors themselves.

use singe_cuda::{
    stream::Stream,
    types::{Complex32, Complex64},
};
use singe_cufft::{
    error::Result as CufftResult,
    plan::Plan,
    types::{Direction, TransformType},
};

use crate::{
    backend::{
        cuda::{
            self,
            context::CudaExecutionContext,
            device_views::{tensor_input_view, tensor_output_view_with_len},
            stream::ensure_not_stream_capturing,
            tensor_region::CudaTensorSource,
        },
        decomposition,
        registry::{Operation, OperationAttributes},
        spec,
    },
    data_type::DataType,
    error::{Error, Result},
    shape::Shape,
    tensor::Tensor,
    utility::{Locked, to_usize},
};

// Half precision transforms in `F32`.
const fn widened(data_type: DataType) -> DataType {
    match data_type {
        DataType::F16 | DataType::Bf16 => DataType::F32,
        other => other,
    }
}

// Runs `transform` with the plan of `kind` over `batch` signals of `length` points, creating it on first use, and waits for it.
#[expect(
    clippy::significant_drop_tightening,
    reason = "a plan's work area serves one transform at a time"
)]
fn with_plan(
    context: &CudaExecutionContext,
    kind: TransformType,
    length: usize,
    batch: usize,
    transform: impl FnOnce(&Plan, &Stream) -> Result<()>,
) -> Result<()> {
    let create = || -> CufftResult<Plan> {
        let mut plan = Plan::create(context.allocator().context())?;
        plan.plan_1d(length, kind, batch)?;
        Ok(plan)
    };
    let run = |plan: &Plan| context.sync_scope(|| transform(plan, context.stream().raw()));
    let runtime = context.runtime();
    // A plan's work area serves one transform at a time, so a caller holds the lock until its transform finishes.
    let mut plans = runtime.fft_plans().locked("cuda fft plans")?;
    if plans.limits().entries == 0 {
        drop(plans);
        return run(&create()?);
    }

    let key = (kind, length, batch);
    if !plans.contains_key(&key) {
        plans.insert(key, create()?);
    }

    let plan = plans
        .get(&key)
        .ok_or_else(|| Error::missing_node("cuda fft plan"))?;
    run(plan)
}

fn decomposed(
    operation: Operation,
    attributes: &OperationAttributes,
    input: &Tensor,
) -> Result<Tensor> {
    decomposition::decompose(operation, attributes, &[input])?
        .pop()
        .ok_or_else(|| Error::missing_output("cuda fft"))
}

// `rfft` along the last axis: through cuFFT where `spec::transforms_with_cufft` says, after widening half precision, and as the basis matmul otherwise.
pub fn rfft(input: &Tensor) -> Result<Tensor> {
    let dimensions = input.shape().dimensions();
    let data_type = input.data_type();
    let Some((&length, leading)) = dimensions.split_last() else {
        return decomposed(Operation::Rfft, &OperationAttributes::None, input);
    };
    if !spec::transforms_with_cufft(length, widened(data_type)) {
        return decomposed(Operation::Rfft, &OperationAttributes::None, input);
    }

    if matches!(data_type, DataType::F16 | DataType::Bf16) {
        return rfft(&input.cast(DataType::F32)?)?.cast(data_type);
    }

    ensure_not_stream_capturing(input.device(), "cuda eager rfft during stream capture")?;
    let input = input.contiguous()?;
    let frequencies = length / 2 + 1;
    let batch = input.element_count() / to_usize(length, "cuda fft extent")?;
    let mut shape = leading.to_vec();
    shape.extend([frequencies, 2]);
    let shape = Shape::create(shape)?;
    let context = cuda::execution_context(input.device())?;
    context.bind()?;
    let output = context.allocate_output(data_type, &shape)?;
    let pairs = shape.element_count() / 2;
    let kind = if data_type == DataType::F64 {
        TransformType::DoubleRealToDoubleComplex
    } else {
        TransformType::RealToComplex
    };
    with_plan(
        &context,
        kind,
        to_usize(length, "cuda fft extent")?,
        batch,
        |plan, stream| {
            // SAFETY: the plan transforms `batch` contiguous signals of `length` points into as many half spectra, which the input and output hold, and the scope waits for it.
            unsafe {
                if data_type == DataType::F64 {
                    plan.execute_async(
                        &tensor_input_view::<f64>(&input, DataType::F64, "cuda rfft input")?,
                        tensor_output_view_with_len::<Complex64>(
                            output.region(),
                            DataType::F64,
                            pairs,
                            "cuda rfft output",
                        )?,
                        Direction::Forward,
                        stream,
                    )?;
                } else {
                    plan.execute_async(
                        &tensor_input_view::<f32>(&input, DataType::F32, "cuda rfft input")?,
                        tensor_output_view_with_len::<Complex32>(
                            output.region(),
                            DataType::F32,
                            pairs,
                            "cuda rfft output",
                        )?,
                        Direction::Forward,
                        stream,
                    )?;
                }
            }
            Ok(())
        },
    )?;
    // SAFETY: cuFFT wrote every frequency.
    Ok(unsafe { output.assume_init() })
}

// `irfft` into `length` points: through cuFFT where `spec::transforms_with_cufft` says, after widening half precision, and as the basis matmul otherwise.
pub fn irfft(input: &Tensor, length: i64) -> Result<Tensor> {
    let data_type = input.data_type();
    let attributes = OperationAttributes::Length(length);
    if !spec::transforms_with_cufft(length, widened(data_type)) {
        return decomposed(Operation::Irfft, &attributes, input);
    }

    if matches!(data_type, DataType::F16 | DataType::Bf16) {
        return irfft(&input.cast(DataType::F32)?, length)?.cast(data_type);
    }

    let dimensions = input.shape().dimensions();
    let &[ref leading @ .., frequencies, 2] = dimensions else {
        return decomposed(Operation::Irfft, &attributes, input);
    };
    ensure_not_stream_capturing(input.device(), "cuda eager irfft during stream capture")?;
    // The transform reads the half spectrum a length-`length` signal has, dropping frequencies past it and zero-filling missing ones; cuFFT overwrites what it reads, so it reads a copy.
    let needed = length / 2 + 1;
    let spectrum = if frequencies >= needed {
        input.narrow(-2, 0, needed)?.copy()?
    } else {
        let mut padding = dimensions.to_vec();
        let last = padding.len() - 2;
        padding[last] = needed - frequencies;
        Tensor::concat(
            &[
                input.clone(),
                Tensor::zeroed_on(input.device(), data_type, &Shape::create(padding)?)?,
            ],
            -2,
        )?
    };
    let batch = to_usize(leading.iter().product::<i64>(), "cuda fft extent")?;
    let mut shape = leading.to_vec();
    shape.push(length);
    let shape = Shape::create(shape)?;
    let context = cuda::execution_context(input.device())?;
    context.bind()?;
    let output = context.allocate_output(data_type, &shape)?;
    let pairs = spectrum.element_count() / 2;
    let points = shape.element_count();
    let kind = if data_type == DataType::F64 {
        TransformType::DoubleComplexToDoubleReal
    } else {
        TransformType::ComplexToReal
    };
    with_plan(
        &context,
        kind,
        to_usize(length, "cuda fft extent")?,
        batch,
        |plan, stream| {
            let region = spectrum.tensor_region()?;
            // SAFETY: the plan transforms `batch` half spectra, which the copy holds and no one else reads, into as many signals of `length` points, which the output holds, and the scope waits for it.
            unsafe {
                if data_type == DataType::F64 {
                    plan.execute_async(
                        &tensor_output_view_with_len::<Complex64>(
                            region,
                            DataType::F64,
                            pairs,
                            "cuda irfft input",
                        )?,
                        tensor_output_view_with_len::<f64>(
                            output.region(),
                            DataType::F64,
                            points,
                            "cuda irfft output",
                        )?,
                        Direction::Inverse,
                        stream,
                    )?;
                } else {
                    plan.execute_async(
                        &tensor_output_view_with_len::<Complex32>(
                            region,
                            DataType::F32,
                            pairs,
                            "cuda irfft input",
                        )?,
                        tensor_output_view_with_len::<f32>(
                            output.region(),
                            DataType::F32,
                            points,
                            "cuda irfft output",
                        )?,
                        Direction::Inverse,
                        stream,
                    )?;
                }
            }
            Ok(())
        },
    )?;
    // SAFETY: cuFFT wrote every point; the transform is unnormalized, as `torch.fft.irfft` divides it by the length.
    let signal = unsafe { output.assume_init() };
    signal.div_scalar(length as f64)
}
