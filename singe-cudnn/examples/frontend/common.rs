use std::{env::args, mem::MaybeUninit, num::NonZeroUsize, result::Result as StdResult, sync::Arc};

use singe_cuda::{
    context::Context as CudaContext,
    memory::DeviceMemory,
    nvrtc,
    stream::{Stream, StreamOptions},
};
use singe_cudnn::{
    context::Context,
    data_type::{f8e4m3, f16},
    error::{Error, Result},
    tensor::Shape,
};

#[allow(dead_code)]
pub(crate) fn positive_usize_argument(flag: &str, default: usize) -> Result<NonZeroUsize> {
    let prefix = format!("{flag}=");
    let argument = args().find_map(|argument| argument.strip_prefix(&prefix).map(str::to_owned));
    let invalid = || Error::OutOfRange {
        name: flag.to_owned().into(),
    };
    argument.map_or_else(
        || NonZeroUsize::new(default).ok_or_else(invalid),
        |value| value.parse().map_err(|_| invalid()),
    )
}

#[allow(dead_code)]
fn next_random(seed: &mut u32) -> u32 {
    *seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
    *seed
}

#[allow(dead_code)]
pub(crate) fn init_f32_image(size: usize, seed: &mut u32) -> Vec<f32> {
    let mut values = Vec::with_capacity(size);
    for _ in 0..size {
        values.push((next_random(seed) as f32) * 2.328_306_4e-10_f32);
    }
    values
}

#[allow(dead_code)]
pub(crate) fn init_i8_image(size: usize, seed: &mut u32) -> Vec<i8> {
    init_f32_image(size, seed)
        .into_iter()
        .map(|value| (value * 15.0) as i8 - 7)
        .collect()
}

#[allow(dead_code)]
pub(crate) fn init_f16_image(size: usize, seed: &mut u32) -> Vec<f16> {
    init_f32_image(size, seed)
        .into_iter()
        .map(f16::from_f32)
        .collect()
}

/// Chooses finite E4M3 values from zero, signed one-half, one and two.
#[allow(dead_code)]
pub(crate) fn init_f8e4m3_image(size: usize, seed: &mut u32) -> Vec<f8e4m3> {
    const ENCODINGS: &[u8] = &[0x00, 0x30, 0xb0, 0x38, 0xb8, 0x40, 0xc0];
    (0..size)
        .map(|_| {
            let index = next_random(seed) as usize % ENCODINGS.len();
            f8e4m3::from_bits(ENCODINGS[index])
        })
        .collect()
}

/// Creates batched column-major matrices with checked physical strides.
#[allow(dead_code)]
pub(crate) fn column_major_shape([batch, rows, columns]: [i64; 3]) -> Result<Shape> {
    let physical = Shape::contiguous([batch, columns, rows])?;
    let strides = physical.strides();
    Shape::with_strides([batch, rows, columns], [strides[0], strides[2], strides[1]])
}

/// Rounds positive tensor dimensions to a positive tile multiple.
#[allow(dead_code)]
pub(crate) fn round_up(value: i64, multiple: i64) -> Result<i64> {
    let invalid = || Error::OutOfRange {
        name: "padded tensor dimension".into(),
    };
    if value <= 0 || multiple <= 0 {
        return Err(invalid());
    }

    let blocks = value / multiple + i64::from(value % multiple != 0);
    blocks.checked_mul(multiple).ok_or_else(invalid)
}

#[allow(dead_code)]
pub(crate) fn count_nonfinite_f32(values: &[f32]) -> usize {
    values.iter().filter(|value| !value.is_finite()).count()
}

pub(crate) struct ExampleContext {
    #[allow(dead_code)]
    pub cudnn: Context,
    #[allow(dead_code)]
    pub stream: Stream,
}

impl ExampleContext {
    pub(crate) fn create() -> Result<Self> {
        // Keep the nvrtc dependency linked for frontend example binaries.
        let _ = nvrtc::version()?;
        let cuda = CudaContext::create()?;
        let mut cudnn = Context::create(&cuda)?;
        let stream = cuda.create_stream(StreamOptions::default())?;
        cudnn.set_stream(Some(&stream))?;
        Ok(Self { cudnn, stream })
    }

    /// Completes work submitted on the cuDNN stream before returning its result.
    /// Retain bindings, buffers and workspace outside the callback through this call.
    /// An enqueue error still drains the stream before it propagates.
    #[allow(dead_code)]
    pub(crate) fn execute_and_wait(
        &self,
        execute: impl FnOnce(&Context) -> Result<()>,
    ) -> Result<()> {
        let execution = execute(&self.cudnn);
        let completion = self.stream.synchronize();
        execution?;
        completion?;
        Ok(())
    }

    #[allow(dead_code)]
    pub(crate) fn compute_capability(&self) -> Result<i32> {
        let properties = self.cudnn.cuda_context().device().properties()?;
        Ok(properties.major * 10 + properties.minor)
    }
}

#[allow(dead_code)]
pub(crate) fn workspace(
    allocation_context: &Arc<CudaContext>,
    size: usize,
) -> Result<DeviceMemory<MaybeUninit<u8>>> {
    Ok(DeviceMemory::<u8>::create_uninit(allocation_context, size)?)
}

#[allow(dead_code)]
pub(crate) fn finish<E>(name: &str, result: StdResult<(), E>) -> anyhow::Result<()>
where
    E: Into<anyhow::Error>,
{
    let Err(error) = result.map_err(Into::into) else {
        println!("{name}: ok");
        return Ok(());
    };
    match error.downcast_ref::<Error>() {
        // Like the upstream samples, skip configurations this cuDNN/GPU combination cannot run.
        Some(Error::Cudnn { code, .. }) if code.is_not_supported() => {
            println!("{name}: skipped, not supported on this cuDNN/GPU combination ({code})");
            Ok(())
        }
        _ => Err(error),
    }
}
