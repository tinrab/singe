#[cfg(feature = "dtype_f16")]
use singe_cuda::types::f16;
use singe_cuda::{
    stream::Stream,
    view::{DeviceInput, DeviceOutput, DeviceSliceMut},
};

#[cfg(feature = "dtype_f16")]
use crate::cuda::fused::{BshdRopeQkCacheUpdateConfig, validate_bshd_rope_qk_cache_update};
#[cfg(feature = "dtype_f32")]
use crate::cuda::fused::{
    MhcApplyResidualConfig, MhcGemmRmsScaleConfig, MhcSplitGemmRmsScaleConfig,
    validate_mhc_apply_residual, validate_mhc_finalize_scale_bias_sigmoid,
    validate_mhc_gemm_rms_scale, validate_mhc_split_gemm_rms, validate_mhc_y_buffer,
};
use crate::{
    cuda::{
        cutile,
        fused::{
            KvCacheCompactConfig, KvCacheSlidingWindowAppendConfig,
            KvCacheSlidingWindowAppendState, validate_kv_cache_compact,
            validate_kv_cache_sliding_window_append,
        },
        interop::{input_pointer, output_pointer, write_only_pointer},
    },
    error::Result,
    utility::{checked_element_count, ensure_len},
};

#[cfg(feature = "dtype_f32")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn mhc_apply_residual_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    x: &impl DeviceInput<f32>,
    f_out: &impl DeviceInput<f32>,
    y: &impl DeviceInput<f32>,
    config: MhcApplyResidualConfig,
) -> Result<()> {
    validate_mhc_apply_residual(out.len(), x.len(), f_out.len(), y.len(), config)?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::fused::mhc_apply_residual_f32(
            &stream,
            write_only_pointer(out),
            input_pointer(x),
            input_pointer(f_out),
            input_pointer(y),
            config.batch,
            config.n,
            config.channels,
        )
    }
}

#[cfg(feature = "dtype_f32")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn mhc_sinkhorn_f32(
    stream: &Stream,
    y: &mut impl DeviceSliceMut<f32>,
    batch: usize,
    n: usize,
) -> Result<()> {
    validate_mhc_y_buffer(y.len(), batch, n)?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { cutile::fused::mhc_sinkhorn_f32(&stream, output_pointer(y), batch, n) }
}

#[cfg(feature = "dtype_f32")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn mhc_gemm_rms_scale_f32(
    stream: &Stream,
    y: &mut impl DeviceOutput<f32>,
    r: &mut impl DeviceOutput<f32>,
    x: &impl DeviceInput<f32>,
    w: &impl DeviceInput<f32>,
    bias: &impl DeviceInput<f32>,
    config: MhcGemmRmsScaleConfig,
) -> Result<()> {
    validate_mhc_gemm_rms_scale(y.len(), r.len(), x.len(), w.len(), bias.len(), config)?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::fused::mhc_gemm_rms_scale_f32(
            &stream,
            write_only_pointer(y),
            write_only_pointer(r),
            input_pointer(x),
            input_pointer(w),
            input_pointer(bias),
            config.rows,
            config.columns,
            config.reduction,
            config.n,
            config.alpha_pre,
            config.alpha_post,
            config.alpha_res,
        )
    }
}

#[cfg(feature = "dtype_f32")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn mhc_split_gemm_rms_f32(
    stream: &Stream,
    y_acc: &mut impl DeviceOutput<f32>,
    r_acc: &mut impl DeviceOutput<f32>,
    x: &impl DeviceInput<f32>,
    w: &impl DeviceInput<f32>,
    config: MhcSplitGemmRmsScaleConfig,
) -> Result<()> {
    validate_mhc_split_gemm_rms(y_acc.len(), r_acc.len(), x.len(), w.len(), config)?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::fused::mhc_split_gemm_rms_f32(
            &stream,
            write_only_pointer(y_acc),
            write_only_pointer(r_acc),
            input_pointer(x),
            input_pointer(w),
            config.base.rows,
            config.base.columns,
            config.base.reduction,
            config.split_k,
        )
    }
}

#[cfg(feature = "dtype_f32")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn mhc_finalize_scale_bias_sigmoid_f32(
    stream: &Stream,
    y: &mut impl DeviceOutput<f32>,
    r: &mut impl DeviceOutput<f32>,
    y_acc: &impl DeviceInput<f32>,
    r_acc: &impl DeviceInput<f32>,
    bias: &impl DeviceInput<f32>,
    config: MhcSplitGemmRmsScaleConfig,
) -> Result<()> {
    validate_mhc_finalize_scale_bias_sigmoid(
        y.len(),
        r.len(),
        y_acc.len(),
        r_acc.len(),
        bias.len(),
        config,
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::fused::mhc_finalize_scale_bias_sigmoid_f32(
            &stream,
            write_only_pointer(y),
            write_only_pointer(r),
            input_pointer(y_acc),
            input_pointer(r_acc),
            input_pointer(bias),
            config.base.rows,
            config.base.columns,
            config.base.reduction,
            config.base.n,
            config.split_k,
            config.base.alpha_pre,
            config.base.alpha_post,
            config.base.alpha_res,
        )
    }
}

macro_rules! kv_cache_update_fn {
    ($name:ident, $ty:ty) => {
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $name(
            stream: &Stream,
            k_cache: &mut impl DeviceOutput<$ty>,
            v_cache: &mut impl DeviceOutput<$ty>,
            new_k: &impl DeviceInput<$ty>,
            new_v: &impl DeviceInput<$ty>,
            seq_len: usize,
            heads: usize,
            head_dim: usize,
            max_seq: usize,
            position_start: usize,
        ) -> Result<()> {
            let new_len = checked_element_count(checked_element_count(seq_len, heads)?, head_dim)?;
            let cache_len =
                checked_element_count(checked_element_count(heads, max_seq)?, head_dim)?;
            ensure_len(new_k.len(), new_len)?;
            ensure_len(new_v.len(), new_len)?;
            ensure_len(k_cache.len(), cache_len)?;
            ensure_len(v_cache.len(), cache_len)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::fused::$name(
                    &stream,
                    write_only_pointer(k_cache),
                    write_only_pointer(v_cache),
                    input_pointer(new_k),
                    input_pointer(new_v),
                    seq_len,
                    heads,
                    head_dim,
                    max_seq,
                    position_start,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_f16")]
kv_cache_update_fn!(kv_cache_update_f16, f16);

#[cfg(feature = "dtype_f16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn bshd_rope_qk_cache_update_f16(
    stream: &Stream,
    q_out: &mut impl DeviceOutput<f16>,
    k_out: &mut impl DeviceOutput<f16>,
    k_cache: &mut impl DeviceOutput<f16>,
    v_cache: &mut impl DeviceOutput<f16>,
    q_input: &impl DeviceInput<f16>,
    k_input: &impl DeviceInput<f16>,
    v_input: &impl DeviceInput<f16>,
    cos: &impl DeviceInput<f32>,
    sin: &impl DeviceInput<f32>,
    config: BshdRopeQkCacheUpdateConfig,
) -> Result<()> {
    validate_bshd_rope_qk_cache_update(
        q_out.len(),
        k_out.len(),
        k_cache.len(),
        v_cache.len(),
        q_input.len(),
        k_input.len(),
        v_input.len(),
        cos.len(),
        sin.len(),
        config,
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::fused::bshd_rope_qk_cache_update_f16(
            &stream,
            write_only_pointer(q_out),
            write_only_pointer(k_out),
            write_only_pointer(k_cache),
            write_only_pointer(v_cache),
            input_pointer(q_input),
            input_pointer(k_input),
            input_pointer(v_input),
            input_pointer(cos),
            input_pointer(sin),
            config.seq_len,
            config.query_heads,
            config.key_value_heads,
            config.head_dim,
            config.cache_max_seq,
            config.cache_position_start,
        )
    }
}

#[cfg(feature = "dtype_f32")]
kv_cache_update_fn!(kv_cache_update_f32, f32);

#[cfg(feature = "dtype_f64")]
kv_cache_update_fn!(kv_cache_update_f64, f64);

#[cfg(feature = "dtype_f16")]
kv_cache_update_fn!(kv_cache_update_seq_f16, f16);

#[cfg(feature = "dtype_f32")]
kv_cache_update_fn!(kv_cache_update_seq_f32, f32);

#[cfg(feature = "dtype_f64")]
kv_cache_update_fn!(kv_cache_update_seq_f64, f64);

macro_rules! kv_cache_append_sliding_window_fn {
    ($name:ident, $ty:ty, $update:ident, $compact:ident) => {
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $name(
            stream: &Stream,
            k_cache: &mut impl DeviceSliceMut<$ty>,
            v_cache: &mut impl DeviceSliceMut<$ty>,
            new_k: &impl DeviceInput<$ty>,
            new_v: &impl DeviceInput<$ty>,
            seq_len: usize,
            config: KvCacheSlidingWindowAppendConfig,
        ) -> Result<KvCacheSlidingWindowAppendState> {
            let plan = validate_kv_cache_sliding_window_append(
                k_cache.len(),
                v_cache.len(),
                new_k.len(),
                new_v.len(),
                seq_len,
                config,
            )?;
            if let Some(compact) = plan.pre_compact {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $compact(stream, k_cache, v_cache, compact) }?;
            }
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                $update(
                    stream,
                    k_cache,
                    v_cache,
                    new_k,
                    new_v,
                    seq_len,
                    config.heads,
                    config.head_dim,
                    config.max_seq,
                    plan.append_position,
                )
            }?;
            if let Some(compact) = plan.post_compact {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $compact(stream, k_cache, v_cache, compact) }?;
            }
            Ok(plan.state)
        }
    };
}

#[cfg(feature = "dtype_f16")]
kv_cache_append_sliding_window_fn!(
    kv_cache_append_sliding_window_f16,
    f16,
    kv_cache_update_f16,
    kv_cache_compact_f16
);

#[cfg(feature = "dtype_f32")]
kv_cache_append_sliding_window_fn!(
    kv_cache_append_sliding_window_f32,
    f32,
    kv_cache_update_f32,
    kv_cache_compact_f32
);

#[cfg(feature = "dtype_f64")]
kv_cache_append_sliding_window_fn!(
    kv_cache_append_sliding_window_f64,
    f64,
    kv_cache_update_f64,
    kv_cache_compact_f64
);

macro_rules! kv_cache_compact_fn {
    ($name:ident, $ty:ty) => {
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $name(
            stream: &Stream,
            k_cache: &mut impl DeviceSliceMut<$ty>,
            v_cache: &mut impl DeviceSliceMut<$ty>,
            config: KvCacheCompactConfig,
        ) -> Result<()> {
            validate_kv_cache_compact(k_cache.len(), v_cache.len(), config)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::fused::$name(
                    &stream,
                    output_pointer(k_cache),
                    output_pointer(v_cache),
                    config.heads,
                    config.head_dim,
                    config.max_seq,
                    config.source_start,
                    config.token_count,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_f16")]
kv_cache_compact_fn!(kv_cache_compact_f16, f16);

#[cfg(feature = "dtype_f32")]
kv_cache_compact_fn!(kv_cache_compact_f32, f32);

#[cfg(feature = "dtype_f64")]
kv_cache_compact_fn!(kv_cache_compact_f64, f64);
