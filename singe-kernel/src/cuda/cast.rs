//! Device-to-device dtype conversion.

use ::cutile::cuda_async::device_buffer::DevicePointer;
#[cfg(feature = "dtype_bf16")]
use singe_cuda::types::bf16;
#[cfg(feature = "dtype_f16")]
use singe_cuda::types::f16;
use singe_cuda::{
    stream::Stream,
    view::{DeviceInput, DeviceOutput},
};

#[cfg(feature = "cutile")]
use crate::cuda::cutile;
#[cfg(any(
    feature = "dtype_u8",
    feature = "dtype_i8",
    feature = "dtype_u32",
    feature = "dtype_i32",
    feature = "dtype_u64",
    feature = "dtype_i64",
))]
use crate::utility::ensure_len;
use crate::{
    cuda::{
        autotune::{self, KernelCompileConfig, TuneOptions, TuneReport, TuningChoice},
        interop::input_pointer,
        utility::{
            VectorConfig, VectorTuning, resolve_vector_tuning, validate_vector_lengths,
            vector_tile_candidates, vector_tune_key,
        },
    },
    error::Result,
};

fn vector_launch_parts(tuning: Option<VectorTuning>) -> (Option<usize>, KernelCompileConfig) {
    tuning.map_or_else(
        || (None, KernelCompileConfig::default()),
        |tuning| (Some(tuning.tile_width), tuning.compile),
    )
}

fn launch_cast_vector<F>(
    stream: &Stream,
    op: &'static str,
    dtype: &'static str,
    config: VectorConfig,
    mut launch: F,
) -> Result<()>
where
    F: FnMut(Option<VectorTuning>) -> Result<()>,
{
    if config.length == 0 {
        return Ok(());
    }
    match config.tuning {
        TuningChoice::Default => return launch(None),
        TuningChoice::Explicit(tuning) => return launch(Some(tuning)),
        TuningChoice::Cached => {}
        TuningChoice::Tune(_) | TuningChoice::Retune(_) => {
            autotune::ensure_not_capturing(stream, op)?;
        }
    }

    let key = vector_tune_key(stream, op, dtype, config)?;
    let tuning = resolve_vector_tuning(
        stream,
        op,
        key.clone(),
        config.tuning,
        vector_tile_candidates,
        |options, candidates| {
            autotune::tune_candidates(stream, op, key, options, candidates, |tuning| {
                launch(Some(*tuning))
            })
        },
    )?;
    launch(tuning)
}

fn tune_cast_vector<F>(
    stream: &Stream,
    op: &'static str,
    dtype: &'static str,
    config: VectorConfig,
    options: TuneOptions,
    mut launch: F,
) -> Result<TuneReport<VectorTuning>>
where
    F: FnMut(VectorTuning) -> Result<()>,
{
    let key = vector_tune_key(stream, op, dtype, config)?;
    let report = autotune::tune_candidates(
        stream,
        op,
        key.clone(),
        options,
        vector_tile_candidates(),
        |tuning| launch(*tuning),
    )?;
    autotune::cache().insert(key, report.best);
    launch(report.best)?;
    Ok(report)
}

macro_rules! cast_tuned_fn {
    ($name:ident, $tune_name:ident, $out_ty:ty, $in_ty:ty) => {
        /// Converts every input element into the output datatype on `stream`.
        ///
        /// # Safety
        ///
        /// Inputs must be valid when read and belong to the stream's context.
        /// Order preceding producers before every read, including graph replay.
        /// Retain both ranges and exclusive output access until queued work
        /// completes, including on errors. Order previous accesses before the
        /// launch. Captured graphs must retain the ranges for every replay.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$out_ty>,
            input: &impl DeviceInput<$in_ty>,
            config: VectorConfig,
        ) -> Result<()> {
            validate_vector_lengths(out.len(), input.len(), config)?;
            let cuda_stream = stream.to_cuda_core()?;
            launch_cast_vector(
                stream,
                stringify!($name),
                stringify!($out_ty),
                config,
                |tuning| {
                    let (tile_width, compile) = vector_launch_parts(tuning);
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        cutile::cast::$name(
                            &cuda_stream,
                            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                            input_pointer(input),
                            config.length,
                            tile_width,
                            compile,
                        )
                    }
                },
            )
        }

        /// Tunes the conversion and enqueues the selected kernel.
        ///
        /// # Safety
        ///
        /// The conversion's input validity, context, ordering, and asynchronous
        /// resource lifetime requirements apply to every candidate launch.
        pub unsafe fn $tune_name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$out_ty>,
            input: &impl DeviceInput<$in_ty>,
            config: VectorConfig,
            options: TuneOptions,
        ) -> Result<TuneReport<VectorTuning>> {
            validate_vector_lengths(out.len(), input.len(), config)?;
            let cuda_stream = stream.to_cuda_core()?;
            tune_cast_vector(
                stream,
                stringify!($name),
                stringify!($out_ty),
                config,
                options,
                // SAFETY: the caller upholds this function's safety contract.
                |tuning| unsafe {
                    cutile::cast::$name(
                        &cuda_stream,
                        DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                        input_pointer(input),
                        config.length,
                        Some(tuning.tile_width),
                        tuning.compile,
                    )
                },
            )
        }
    };
}

#[cfg(any(
    feature = "dtype_u8",
    feature = "dtype_i8",
    feature = "dtype_u32",
    feature = "dtype_i32",
    feature = "dtype_u64",
    feature = "dtype_i64",
))]
macro_rules! cast_raw_fn {
    ($name:ident, $out_ty:ty, $in_ty:ty) => {
        /// Converts every input element into the output datatype on `stream`.
        ///
        /// # Safety
        ///
        /// Inputs must be valid when read and belong to the stream's context.
        /// Order preceding producers before every read, including graph replay.
        /// Retain both ranges and exclusive output access until queued work
        /// completes, including on errors. Order previous accesses before the
        /// launch. Captured graphs must retain the ranges for every replay.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$out_ty>,
            input: &impl DeviceInput<$in_ty>,
        ) -> Result<()> {
            let len = out.len();
            ensure_len(input.len(), len)?;
            if len == 0 {
                return Ok(());
            }
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::cast::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(input),
                    len,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_f32")]
#[cfg(feature = "dtype_f16")]
cast_tuned_fn!(f16_to_f32, tune_f16_to_f32, f32, f16);

#[cfg(feature = "dtype_f32")]
#[cfg(feature = "dtype_bf16")]
cast_tuned_fn!(bf16_to_f32, tune_bf16_to_f32, f32, bf16);

#[cfg(feature = "dtype_f64")]
#[cfg(feature = "dtype_f16")]
cast_tuned_fn!(f16_to_f64, tune_f16_to_f64, f64, f16);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
cast_tuned_fn!(f16_to_u8, tune_f16_to_u8, u8, f16);

#[cfg(all(feature = "dtype_f16", feature = "dtype_i8"))]
cast_tuned_fn!(f16_to_i8, tune_f16_to_i8, i8, f16);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u32"))]
cast_tuned_fn!(f16_to_u32, tune_f16_to_u32, u32, f16);

#[cfg(all(feature = "dtype_f16", feature = "dtype_i32"))]
cast_tuned_fn!(f16_to_i32, tune_f16_to_i32, i32, f16);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u64"))]
cast_tuned_fn!(f16_to_u64, tune_f16_to_u64, u64, f16);

#[cfg(all(feature = "dtype_f16", feature = "dtype_i64"))]
cast_tuned_fn!(f16_to_i64, tune_f16_to_i64, i64, f16);

#[cfg(feature = "dtype_f16")]
#[cfg(feature = "dtype_f32")]
cast_tuned_fn!(f32_to_f16, tune_f32_to_f16, f16, f32);

#[cfg(feature = "dtype_bf16")]
#[cfg(feature = "dtype_f32")]
cast_tuned_fn!(f32_to_bf16, tune_f32_to_bf16, bf16, f32);

#[cfg(feature = "dtype_f64")]
#[cfg(feature = "dtype_f32")]
cast_tuned_fn!(f32_to_f64, tune_f32_to_f64, f64, f32);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
cast_tuned_fn!(f32_to_u8, tune_f32_to_u8, u8, f32);

#[cfg(all(feature = "dtype_f32", feature = "dtype_i8"))]
cast_tuned_fn!(f32_to_i8, tune_f32_to_i8, i8, f32);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u32"))]
cast_tuned_fn!(f32_to_u32, tune_f32_to_u32, u32, f32);

#[cfg(all(feature = "dtype_f32", feature = "dtype_i32"))]
cast_tuned_fn!(f32_to_i32, tune_f32_to_i32, i32, f32);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u64"))]
cast_tuned_fn!(f32_to_u64, tune_f32_to_u64, u64, f32);

#[cfg(all(feature = "dtype_f32", feature = "dtype_i64"))]
cast_tuned_fn!(f32_to_i64, tune_f32_to_i64, i64, f32);

#[cfg(feature = "dtype_f16")]
#[cfg(feature = "dtype_f64")]
cast_tuned_fn!(f64_to_f16, tune_f64_to_f16, f16, f64);

#[cfg(feature = "dtype_f32")]
#[cfg(feature = "dtype_f64")]
cast_tuned_fn!(f64_to_f32, tune_f64_to_f32, f32, f64);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
cast_tuned_fn!(f64_to_u8, tune_f64_to_u8, u8, f64);

#[cfg(all(feature = "dtype_f64", feature = "dtype_i8"))]
cast_tuned_fn!(f64_to_i8, tune_f64_to_i8, i8, f64);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u32"))]
cast_tuned_fn!(f64_to_u32, tune_f64_to_u32, u32, f64);

#[cfg(all(feature = "dtype_f64", feature = "dtype_i32"))]
cast_tuned_fn!(f64_to_i32, tune_f64_to_i32, i32, f64);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u64"))]
cast_tuned_fn!(f64_to_u64, tune_f64_to_u64, u64, f64);

#[cfg(all(feature = "dtype_f64", feature = "dtype_i64"))]
cast_tuned_fn!(f64_to_i64, tune_f64_to_i64, i64, f64);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
cast_tuned_fn!(bool_to_f16, tune_bool_to_f16, f16, u8);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
cast_tuned_fn!(bool_to_f32, tune_bool_to_f32, f32, u8);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
cast_tuned_fn!(bool_to_f64, tune_bool_to_f64, f64, u8);

#[cfg(feature = "dtype_u8")]
cast_raw_fn!(bool_to_u8, u8, u8);

#[cfg(all(feature = "dtype_u8", feature = "dtype_i8"))]
cast_raw_fn!(bool_to_i8, i8, u8);

#[cfg(all(feature = "dtype_u8", feature = "dtype_u32"))]
cast_raw_fn!(bool_to_u32, u32, u8);

#[cfg(all(feature = "dtype_u8", feature = "dtype_i32"))]
cast_raw_fn!(bool_to_i32, i32, u8);

#[cfg(all(feature = "dtype_u8", feature = "dtype_u64"))]
cast_raw_fn!(bool_to_u64, u64, u8);

#[cfg(all(feature = "dtype_u8", feature = "dtype_i64"))]
cast_raw_fn!(bool_to_i64, i64, u8);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
cast_tuned_fn!(u8_to_f16, tune_u8_to_f16, f16, u8);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
cast_tuned_fn!(u8_to_f32, tune_u8_to_f32, f32, u8);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
cast_tuned_fn!(u8_to_f64, tune_u8_to_f64, f64, u8);

#[cfg(all(feature = "dtype_f16", feature = "dtype_i8"))]
cast_tuned_fn!(i8_to_f16, tune_i8_to_f16, f16, i8);

#[cfg(all(feature = "dtype_f32", feature = "dtype_i8"))]
cast_tuned_fn!(i8_to_f32, tune_i8_to_f32, f32, i8);

#[cfg(all(feature = "dtype_f64", feature = "dtype_i8"))]
cast_tuned_fn!(i8_to_f64, tune_i8_to_f64, f64, i8);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u32"))]
cast_tuned_fn!(u32_to_f16, tune_u32_to_f16, f16, u32);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u32"))]
cast_tuned_fn!(u32_to_f32, tune_u32_to_f32, f32, u32);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u32"))]
cast_tuned_fn!(u32_to_f64, tune_u32_to_f64, f64, u32);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u64"))]
cast_tuned_fn!(u64_to_f16, tune_u64_to_f16, f16, u64);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u64"))]
cast_tuned_fn!(u64_to_f32, tune_u64_to_f32, f32, u64);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u64"))]
cast_tuned_fn!(u64_to_f64, tune_u64_to_f64, f64, u64);

#[cfg(all(feature = "dtype_f16", feature = "dtype_i32"))]
cast_tuned_fn!(i32_to_f16, tune_i32_to_f16, f16, i32);

#[cfg(all(feature = "dtype_f32", feature = "dtype_i32"))]
cast_tuned_fn!(i32_to_f32, tune_i32_to_f32, f32, i32);

#[cfg(all(feature = "dtype_f64", feature = "dtype_i32"))]
cast_tuned_fn!(i32_to_f64, tune_i32_to_f64, f64, i32);

#[cfg(all(feature = "dtype_f16", feature = "dtype_i64"))]
cast_tuned_fn!(i64_to_f16, tune_i64_to_f16, f16, i64);

#[cfg(all(feature = "dtype_f32", feature = "dtype_i64"))]
cast_tuned_fn!(i64_to_f32, tune_i64_to_f32, f32, i64);

#[cfg(all(feature = "dtype_f64", feature = "dtype_i64"))]
cast_tuned_fn!(i64_to_f64, tune_i64_to_f64, f64, i64);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
cast_raw_fn!(f16_to_bool, u8, f16);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
cast_raw_fn!(f32_to_bool, u8, f32);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
cast_raw_fn!(f64_to_bool, u8, f64);

#[cfg(feature = "dtype_u8")]
cast_raw_fn!(u8_to_bool, u8, u8);

#[cfg(all(feature = "dtype_i8", feature = "dtype_u8"))]
cast_raw_fn!(i8_to_bool, u8, i8);

#[cfg(all(feature = "dtype_u32", feature = "dtype_u8"))]
cast_raw_fn!(u32_to_bool, u8, u32);

#[cfg(all(feature = "dtype_i32", feature = "dtype_u8"))]
cast_raw_fn!(i32_to_bool, u8, i32);

#[cfg(all(feature = "dtype_u64", feature = "dtype_u8"))]
cast_raw_fn!(u64_to_bool, u8, u64);

#[cfg(all(feature = "dtype_i64", feature = "dtype_u8"))]
cast_raw_fn!(i64_to_bool, u8, i64);

#[cfg(feature = "dtype_u8")]
cast_raw_fn!(u8_to_u8, u8, u8);

#[cfg(all(feature = "dtype_u8", feature = "dtype_i8"))]
cast_raw_fn!(u8_to_i8, i8, u8);

#[cfg(all(feature = "dtype_u8", feature = "dtype_u32"))]
cast_raw_fn!(u8_to_u32, u32, u8);

#[cfg(all(feature = "dtype_u8", feature = "dtype_i32"))]
cast_raw_fn!(u8_to_i32, i32, u8);

#[cfg(all(feature = "dtype_u8", feature = "dtype_u64"))]
cast_raw_fn!(u8_to_u64, u64, u8);

#[cfg(all(feature = "dtype_u8", feature = "dtype_i64"))]
cast_raw_fn!(u8_to_i64, i64, u8);

#[cfg(all(feature = "dtype_i8", feature = "dtype_u8"))]
cast_raw_fn!(i8_to_u8, u8, i8);

#[cfg(feature = "dtype_i8")]
cast_raw_fn!(i8_to_i8, i8, i8);

#[cfg(all(feature = "dtype_i8", feature = "dtype_u32"))]
cast_raw_fn!(i8_to_u32, u32, i8);

#[cfg(all(feature = "dtype_i8", feature = "dtype_i32"))]
cast_raw_fn!(i8_to_i32, i32, i8);

#[cfg(all(feature = "dtype_i8", feature = "dtype_u64"))]
cast_raw_fn!(i8_to_u64, u64, i8);

#[cfg(all(feature = "dtype_i8", feature = "dtype_i64"))]
cast_raw_fn!(i8_to_i64, i64, i8);

#[cfg(all(feature = "dtype_u32", feature = "dtype_u8"))]
cast_raw_fn!(u32_to_u8, u8, u32);

#[cfg(all(feature = "dtype_u32", feature = "dtype_i8"))]
cast_raw_fn!(u32_to_i8, i8, u32);

#[cfg(feature = "dtype_u32")]
cast_raw_fn!(u32_to_u32, u32, u32);

#[cfg(all(feature = "dtype_u32", feature = "dtype_i32"))]
cast_raw_fn!(u32_to_i32, i32, u32);

#[cfg(all(feature = "dtype_u32", feature = "dtype_u64"))]
cast_raw_fn!(u32_to_u64, u64, u32);

#[cfg(all(feature = "dtype_u32", feature = "dtype_i64"))]
cast_raw_fn!(u32_to_i64, i64, u32);

#[cfg(all(feature = "dtype_i32", feature = "dtype_u8"))]
cast_raw_fn!(i32_to_u8, u8, i32);

#[cfg(all(feature = "dtype_i32", feature = "dtype_i8"))]
cast_raw_fn!(i32_to_i8, i8, i32);

#[cfg(all(feature = "dtype_i32", feature = "dtype_u32"))]
cast_raw_fn!(i32_to_u32, u32, i32);

#[cfg(feature = "dtype_i32")]
cast_raw_fn!(i32_to_i32, i32, i32);

#[cfg(all(feature = "dtype_i32", feature = "dtype_u64"))]
cast_raw_fn!(i32_to_u64, u64, i32);

#[cfg(all(feature = "dtype_i32", feature = "dtype_i64"))]
cast_raw_fn!(i32_to_i64, i64, i32);

#[cfg(all(feature = "dtype_u64", feature = "dtype_u8"))]
cast_raw_fn!(u64_to_u8, u8, u64);

#[cfg(all(feature = "dtype_u64", feature = "dtype_i8"))]
cast_raw_fn!(u64_to_i8, i8, u64);

#[cfg(all(feature = "dtype_u64", feature = "dtype_u32"))]
cast_raw_fn!(u64_to_u32, u32, u64);

#[cfg(all(feature = "dtype_u64", feature = "dtype_i32"))]
cast_raw_fn!(u64_to_i32, i32, u64);

#[cfg(feature = "dtype_u64")]
cast_raw_fn!(u64_to_u64, u64, u64);

#[cfg(all(feature = "dtype_u64", feature = "dtype_i64"))]
cast_raw_fn!(u64_to_i64, i64, u64);

#[cfg(all(feature = "dtype_i64", feature = "dtype_u8"))]
cast_raw_fn!(i64_to_u8, u8, i64);

#[cfg(all(feature = "dtype_i64", feature = "dtype_i8"))]
cast_raw_fn!(i64_to_i8, i8, i64);

#[cfg(all(feature = "dtype_i64", feature = "dtype_u32"))]
cast_raw_fn!(i64_to_u32, u32, i64);

#[cfg(all(feature = "dtype_i64", feature = "dtype_i32"))]
cast_raw_fn!(i64_to_i32, i32, i64);

#[cfg(all(feature = "dtype_i64", feature = "dtype_u64"))]
cast_raw_fn!(i64_to_u64, u64, i64);

#[cfg(feature = "dtype_i64")]
cast_raw_fn!(i64_to_i64, i64, i64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cast_vector_config_builders_preserve_tuning_choice() {
        let tuning =
            VectorTuning::tile(64).with_compile(KernelCompileConfig::default().occupancy(2));
        assert_eq!(
            VectorConfig::create(17).explicit_tuning(tuning).tuning,
            TuningChoice::Explicit(tuning)
        );
        assert_eq!(
            VectorConfig::create(17).cached_tuning().tuning,
            TuningChoice::Cached
        );

        let options = TuneOptions::create(0, 1, 2);
        assert_eq!(
            VectorConfig::create(17).tune(options).tuning,
            TuningChoice::Tune(options)
        );
        assert_eq!(
            VectorConfig::create(17).retune(options).tuning,
            TuningChoice::Retune(options)
        );
    }

    #[test]
    fn test_cast_uses_shared_vector_tile_candidates() {
        let tile_widths = vector_tile_candidates()
            .into_iter()
            .map(|candidate| candidate.tile_width)
            .collect::<Vec<_>>();
        assert_eq!(tile_widths, vec![32, 64, 128, 256]);
    }

    #[test]
    fn test_cast_validation_accepts_exact_lengths() -> Result<()> {
        validate_vector_lengths(24, 24, VectorConfig::create(24))
    }

    #[test]
    fn test_cast_validation_rejects_short_input() {
        assert!(validate_vector_lengths(24, 23, VectorConfig::create(24)).is_err());
    }

    #[cfg(all(feature = "dtype_f32", feature = "dtype_bf16"))]
    #[test]
    fn test_f32_to_bf16_matches_host_conversion() -> Result<()> {
        use singe_cuda::{
            context::Context as CudaContext, device::Device, memory::DeviceMemory,
            stream::StreamFlags, types::bf16,
        };

        let Ok(context) = CudaContext::retain_primary_for_device(Device::new(0)) else {
            return Ok(());
        };
        let stream = context.create_stream_with_flags(StreamFlags::NON_BLOCKING)?;
        let input_host = vec![0.0f32, 1.0, -0.5, 3.25, -7.75, 0.001, 1024.0];
        let input = DeviceMemory::from_slice(stream.context(), &input_host)?;
        let mut out = DeviceMemory::<bf16>::create_uninit(stream.context(), input_host.len())?;

        stream.sync_scope(|_| {
            // SAFETY: the scope retains both buffers and waits for the full conversion on every path.
            Ok(unsafe {
                f32_to_bf16(
                    &stream,
                    &mut out.output(),
                    &input,
                    VectorConfig::create(input_host.len()),
                )
            })
        })??;
        // SAFETY: the completed conversion initialized every output element.
        let out = unsafe { out.assume_init() };

        let expected = input_host
            .iter()
            .map(|value| bf16::from_f32(*value).to_bits())
            .collect::<Vec<_>>();
        let actual = out
            .copy_to_host_vec()?
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        Ok(())
    }

    #[cfg(all(feature = "dtype_f32", feature = "dtype_bf16"))]
    #[test]
    fn test_bf16_to_f32_matches_host_conversion() -> Result<()> {
        use singe_cuda::{
            context::Context as CudaContext, device::Device, memory::DeviceMemory,
            stream::StreamFlags, types::bf16,
        };

        let Ok(context) = CudaContext::retain_primary_for_device(Device::new(0)) else {
            return Ok(());
        };
        let stream = context.create_stream_with_flags(StreamFlags::NON_BLOCKING)?;
        let input_host = [0.0f32, 1.0, -0.5, 3.25, -7.75, 0.001, 1024.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        let input = DeviceMemory::from_slice(stream.context(), &input_host)?;
        let mut out = DeviceMemory::<f32>::create_uninit(stream.context(), input_host.len())?;

        stream.sync_scope(|_| {
            // SAFETY: the scope retains both buffers and waits for the full conversion on every path.
            Ok(unsafe {
                bf16_to_f32(
                    &stream,
                    &mut out.output(),
                    &input,
                    VectorConfig::create(input_host.len()),
                )
            })
        })??;
        // SAFETY: the completed conversion initialized every output element.
        let out = unsafe { out.assume_init() };

        let expected = input_host
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        let actual = out.copy_to_host_vec()?;
        assert_eq!(actual, expected);
        Ok(())
    }
}
