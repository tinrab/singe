#[cfg(feature = "dtype_bf16")]
use singe_cuda::types::bf16;
#[cfg(feature = "dtype_f16")]
use singe_cuda::types::f16;
use singe_cuda::{
    stream::Stream,
    view::{DeviceInput, DeviceOutput},
};

use crate::{
    cuda::{
        cutile,
        interop::{input_pointer, write_only_pointer},
        quantization::Fp8GroupQuantConfig,
    },
    error::Result,
    utility::{checked_element_count, ensure_len_at_least},
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FusedRmsNormFp8Config {
    pub quantization: Fp8GroupQuantConfig,
    pub eps: f32,
    pub weight_offset: f32,
}

macro_rules! fused_fp8 {
    ($ty:ty, $norm:ident, $silu:ident) => {
        /// Adds and rounds the residual in the storage dtype, RMS-normalizes in f32,
        /// rounds back to storage, and emits FP8 with float32 dequantization scales.
        /// # Safety
        /// Inputs must be initialized. Writable ranges must not overlap each other or any input.
        /// Order producers and retain every allocation until completion and throughout replay.
        pub unsafe fn $norm(
            stream: &Stream,
            out: &mut impl DeviceOutput<u8>,
            scales: &mut impl DeviceOutput<f32>,
            residual_out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            residual: &impl DeviceInput<$ty>,
            weight: &impl DeviceInput<$ty>,
            config: FusedRmsNormFp8Config,
        ) -> Result<()> {
            let q = config.quantization;
            let scale_len = q.scale_len()?;
            let len = checked_element_count(q.rows, q.columns)?;
            for actual in [out.len(), residual_out.len(), input.len(), residual.len()] {
                ensure_len_at_least(actual, len)?;
            }
            ensure_len_at_least(scales.len(), scale_len)?;
            ensure_len_at_least(weight.len(), q.columns)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: checked lengths and the caller's disjointness/lifetime contract cover raw submission.
            unsafe {
                cutile::fused::rms_norm_add_quantize_fp8(
                    &stream,
                    ::cutile::cuda_async::device_buffer::DevicePointer::from_cu_deviceptr(
                        out.as_mut_ptr() as _,
                    ),
                    write_only_pointer(scales),
                    write_only_pointer(residual_out),
                    input_pointer(input),
                    input_pointer(residual),
                    input_pointer(weight),
                    q,
                    config.eps,
                    config.weight_offset,
                    stringify!($ty),
                )
            }
        }

        /// Computes `SwiGLU`, rounds to storage, and emits FP8 with float32 scales.
        /// # Safety
        /// Inputs must be initialized. Writable ranges must not overlap each other or any input.
        /// Order producers and retain every allocation until completion and throughout replay.
        pub unsafe fn $silu(
            stream: &Stream,
            out: &mut impl DeviceOutput<u8>,
            scales: &mut impl DeviceOutput<f32>,
            gate: &impl DeviceInput<$ty>,
            up: &impl DeviceInput<$ty>,
            config: Fp8GroupQuantConfig,
        ) -> Result<()> {
            let scale_len = config.scale_len()?;
            let len = checked_element_count(config.rows, config.columns)?;
            for actual in [out.len(), gate.len(), up.len()] {
                ensure_len_at_least(actual, len)?;
            }
            ensure_len_at_least(scales.len(), scale_len)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: checked lengths and the caller's disjointness/lifetime contract cover raw submission.
            unsafe {
                cutile::fused::silu_mul_quantize_fp8(
                    &stream,
                    ::cutile::cuda_async::device_buffer::DevicePointer::from_cu_deviceptr(
                        out.as_mut_ptr() as _,
                    ),
                    write_only_pointer(scales),
                    input_pointer(gate),
                    input_pointer(up),
                    config,
                    stringify!($ty),
                )
            }
        }
    };
}

#[cfg(feature = "dtype_f32")]
fused_fp8!(
    f32,
    rms_norm_add_quantize_f8e4m3_f32,
    silu_mul_quantize_f8e4m3_f32
);

#[cfg(feature = "dtype_f16")]
fused_fp8!(
    f16,
    rms_norm_add_quantize_f8e4m3_f16,
    silu_mul_quantize_f8e4m3_f16
);

#[cfg(feature = "dtype_bf16")]
fused_fp8!(
    bf16,
    rms_norm_add_quantize_f8e4m3_bf16,
    silu_mul_quantize_f8e4m3_bf16
);
