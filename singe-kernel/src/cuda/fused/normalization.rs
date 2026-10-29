use ::cutile::cuda_async::device_buffer::DevicePointer;
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
        autotune::{TuneOptions, TuneReport},
        cutile,
        fused::{
            FusedRmsNormConfig, FusedTileTuning, fused_rms_norm_add, fused_rms_norm_silu_mul,
            tune_fused_rms_norm_add, tune_fused_rms_norm_silu_mul, validate_dual_rms_norm,
            validate_rms_norm_add_wide, validate_rms_norm_gated_silu,
            validate_rms_norm_residual_add,
        },
        interop::{input_pointer, write_only_pointer},
    },
    error::Result,
};

macro_rules! autotuned_rms_norm_add_fn {
    ($name:ident, $tune_name:ident, $ty:ty, $eps_ty:ty, $dtype:literal, $backend:ident, $backend_tuned:ident) => {
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            residual_out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            residual: &impl DeviceInput<$ty>,
            weight: &impl DeviceInput<$ty>,
            config: FusedRmsNormConfig<$eps_ty>,
        ) -> Result<()> {
            fused_rms_norm_add(
                stream,
                out,
                residual_out,
                input,
                residual,
                weight,
                config,
                $dtype,
                stringify!($name),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9| unsafe {
                    cutile::fused::$backend(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9,
                    )
                },
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11| unsafe {
                    cutile::fused::$backend_tuned(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11,
                    )
                },
            )
        }

        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $tune_name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            residual_out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            residual: &impl DeviceInput<$ty>,
            weight: &impl DeviceInput<$ty>,
            config: FusedRmsNormConfig<$eps_ty>,
            options: TuneOptions,
        ) -> Result<TuneReport<FusedTileTuning>> {
            tune_fused_rms_norm_add(
                stream,
                out,
                residual_out,
                input,
                residual,
                weight,
                config,
                options,
                $dtype,
                stringify!($name),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11| unsafe {
                    cutile::fused::$backend_tuned(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11,
                    )
                },
            )
        }
    };
}

macro_rules! rms_norm_add_wide_fn {
    ($name:ident, $ty:ty, $eps_ty:ty) => {
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            residual_out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            residual: &impl DeviceInput<$ty>,
            weight: &impl DeviceInput<$ty>,
            partial_sum: &mut impl DeviceOutput<f32>,
            row_sum: &mut impl DeviceOutput<f32>,
            rows: usize,
            cols: usize,
            eps: $eps_ty,
            weight_offset: $eps_ty,
        ) -> Result<()> {
            validate_rms_norm_add_wide(
                out.len(),
                residual_out.len(),
                input.len(),
                residual.len(),
                weight.len(),
                partial_sum.len(),
                row_sum.len(),
                rows,
                cols,
            )?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::fused::$name(
                    &stream,
                    write_only_pointer(out),
                    write_only_pointer(residual_out),
                    input_pointer(input),
                    input_pointer(residual),
                    input_pointer(weight),
                    write_only_pointer(partial_sum),
                    write_only_pointer(row_sum),
                    rows,
                    cols,
                    eps,
                    weight_offset,
                )
            }
        }
    };
}

macro_rules! autotuned_rms_norm_silu_mul_fn {
    ($name:ident, $tune_name:ident, $ty:ty, $eps_ty:ty, $dtype:literal, $backend:ident, $backend_tuned:ident) => {
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            weight: &impl DeviceInput<$ty>,
            up: &impl DeviceInput<$ty>,
            config: FusedRmsNormConfig<$eps_ty>,
        ) -> Result<()> {
            fused_rms_norm_silu_mul(
                stream,
                out,
                input,
                weight,
                up,
                config,
                $dtype,
                stringify!($name),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8| unsafe {
                    cutile::fused::$backend(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
                },
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10| unsafe {
                    cutile::fused::$backend_tuned(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,
                    )
                },
            )
        }

        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $tune_name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            weight: &impl DeviceInput<$ty>,
            up: &impl DeviceInput<$ty>,
            config: FusedRmsNormConfig<$eps_ty>,
            options: TuneOptions,
        ) -> Result<TuneReport<FusedTileTuning>> {
            tune_fused_rms_norm_silu_mul(
                stream,
                out,
                input,
                weight,
                up,
                config,
                options,
                $dtype,
                stringify!($name),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10| unsafe {
                    cutile::fused::$backend_tuned(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,
                    )
                },
            )
        }
    };
}

macro_rules! rms_norm_gated_silu_fn {
    ($name:ident, $ty:ty, $eps_ty:ty) => {
        /// Computes row RMS normalization, column scaling, and a `SiLU` gate.
        ///
        /// # Safety
        ///
        /// Input values must be valid when read; order their producers before use.
        /// Retain input allocations and exclusive output access in the stream's
        /// context through completion on every path and throughout replay.
        /// Order prior accesses before launch and keep output disjoint from inputs.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            gate: &impl DeviceInput<$ty>,
            weight: &impl DeviceInput<$ty>,
            rows: usize,
            cols: usize,
            eps: $eps_ty,
            weight_offset: $eps_ty,
        ) -> Result<()> {
            validate_rms_norm_gated_silu(
                out.len(),
                input.len(),
                gate.len(),
                weight.len(),
                rows,
                cols,
            )?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::fused::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(input),
                    input_pointer(gate),
                    input_pointer(weight),
                    rows,
                    cols,
                    eps,
                    weight_offset,
                )
            }
        }
    };
}

macro_rules! dual_rms_norm_fn {
    ($name:ident, $ty:ty, $eps_ty:ty) => {
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $name(
            stream: &Stream,
            q_out: &mut impl DeviceOutput<$ty>,
            k_out: &mut impl DeviceOutput<$ty>,
            q: &impl DeviceInput<$ty>,
            k: &impl DeviceInput<$ty>,
            q_weight: &impl DeviceInput<$ty>,
            k_weight: &impl DeviceInput<$ty>,
            rows: usize,
            cols: usize,
            eps: $eps_ty,
        ) -> Result<()> {
            validate_dual_rms_norm(
                q_out.len(),
                k_out.len(),
                q.len(),
                k.len(),
                q_weight.len(),
                k_weight.len(),
                rows,
                cols,
            )?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::fused::$name(
                    &stream,
                    write_only_pointer(q_out),
                    write_only_pointer(k_out),
                    input_pointer(q),
                    input_pointer(k),
                    input_pointer(q_weight),
                    input_pointer(k_weight),
                    rows,
                    cols,
                    eps,
                )
            }
        }
    };
}

macro_rules! rms_norm_residual_add_fn {
    ($name:ident, $ty:ty, $eps_ty:ty) => {
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            residual: &impl DeviceInput<$ty>,
            weight: &impl DeviceInput<$ty>,
            rows: usize,
            cols: usize,
            eps: $eps_ty,
        ) -> Result<()> {
            validate_rms_norm_residual_add(
                out.len(),
                input.len(),
                residual.len(),
                weight.len(),
                rows,
                cols,
            )?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::fused::$name(
                    &stream,
                    write_only_pointer(out),
                    input_pointer(input),
                    input_pointer(residual),
                    input_pointer(weight),
                    rows,
                    cols,
                    eps,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_f16")]
autotuned_rms_norm_add_fn!(
    rms_norm_add_f16,
    tune_rms_norm_add_f16,
    f16,
    f32,
    "f16",
    rms_norm_add_f16,
    rms_norm_add_f16_tuned
);

#[cfg(feature = "dtype_bf16")]
autotuned_rms_norm_add_fn!(
    rms_norm_add_bf16,
    tune_rms_norm_add_bf16,
    bf16,
    f32,
    "bf16",
    rms_norm_add_bf16,
    rms_norm_add_bf16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_rms_norm_add_fn!(
    rms_norm_add_f32,
    tune_rms_norm_add_f32,
    f32,
    f32,
    "f32",
    rms_norm_add_f32,
    rms_norm_add_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_rms_norm_add_fn!(
    rms_norm_add_f64,
    tune_rms_norm_add_f64,
    f64,
    f64,
    "f64",
    rms_norm_add_f64,
    rms_norm_add_f64_tuned
);

#[cfg(feature = "dtype_f16")]
rms_norm_add_wide_fn!(rms_norm_add_wide_f16, f16, f32);

#[cfg(feature = "dtype_bf16")]
rms_norm_add_wide_fn!(rms_norm_add_wide_bf16, bf16, f32);

#[cfg(feature = "dtype_f32")]
rms_norm_add_wide_fn!(rms_norm_add_wide_f32, f32, f32);

#[cfg(feature = "dtype_f16")]
autotuned_rms_norm_silu_mul_fn!(
    rms_norm_silu_mul_f16,
    tune_rms_norm_silu_mul_f16,
    f16,
    f32,
    "f16",
    rms_norm_silu_mul_f16,
    rms_norm_silu_mul_f16_tuned
);

#[cfg(feature = "dtype_bf16")]
autotuned_rms_norm_silu_mul_fn!(
    rms_norm_silu_mul_bf16,
    tune_rms_norm_silu_mul_bf16,
    bf16,
    f32,
    "bf16",
    rms_norm_silu_mul_bf16,
    rms_norm_silu_mul_bf16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_rms_norm_silu_mul_fn!(
    rms_norm_silu_mul_f32,
    tune_rms_norm_silu_mul_f32,
    f32,
    f32,
    "f32",
    rms_norm_silu_mul_f32,
    rms_norm_silu_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_rms_norm_silu_mul_fn!(
    rms_norm_silu_mul_f64,
    tune_rms_norm_silu_mul_f64,
    f64,
    f64,
    "f64",
    rms_norm_silu_mul_f64,
    rms_norm_silu_mul_f64_tuned
);

#[cfg(feature = "dtype_f32")]
rms_norm_gated_silu_fn!(rms_norm_gated_silu_f32, f32, f32);

#[cfg(feature = "dtype_f16")]
rms_norm_gated_silu_fn!(rms_norm_gated_silu_f16, f16, f32);

#[cfg(feature = "dtype_bf16")]
rms_norm_gated_silu_fn!(rms_norm_gated_silu_bf16, bf16, f32);

#[cfg(feature = "dtype_f32")]
dual_rms_norm_fn!(dual_rms_norm_f32, f32, f32);

#[cfg(feature = "dtype_f16")]
dual_rms_norm_fn!(dual_rms_norm_f16, f16, f32);

#[cfg(feature = "dtype_bf16")]
dual_rms_norm_fn!(dual_rms_norm_bf16, bf16, f32);

#[cfg(feature = "dtype_f32")]
rms_norm_residual_add_fn!(rms_norm_residual_add_f32, f32, f32);

#[cfg(feature = "dtype_f16")]
rms_norm_residual_add_fn!(rms_norm_residual_add_f16, f16, f32);

#[cfg(feature = "dtype_bf16")]
rms_norm_residual_add_fn!(rms_norm_residual_add_bf16, bf16, f32);
