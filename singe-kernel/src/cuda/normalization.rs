//! RMS norm, layer norm, group norm, sparsemax, and related reductions.

use std::{marker::PhantomData, sync::Arc};

#[cfg(feature = "cutile")]
use ::cutile::cuda_async::device_buffer::DevicePointer;
use singe_cuda::{
    stream::Stream,
    view::{DeviceRepr, DeviceSlice, DeviceSliceMut},
};

#[cfg(feature = "dtype-bf16")]
use singe_cuda::types::bf16;
#[cfg(feature = "dtype-f16")]
use singe_cuda::types::f16;

#[cfg(feature = "cutile")]
use crate::cuda::cutile;
use crate::{
    cuda::interop::{borrowed_stream, input_pointer, output_pointer},
    error::{Error, Result},
    utility::{checked_element_count, ensure_len},
};

macro_rules! normalization_fns {
    ($ty:ty, $zero:expr, $layer_norm:ident, $layer_norm_affine:ident, $rms_norm:ident, $rms_norm_affine:ident, $rms_norm_weight_offset:ident) => {
        pub fn $layer_norm(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            rows: usize,
            cols: usize,
            eps: $ty,
        ) -> Result<()> {
            normalization(
                stream,
                out,
                input,
                None::<&NoParameter<$ty>>,
                None::<&NoParameter<$ty>>,
                rows,
                cols,
                eps,
                $zero,
                cutile::normalization::$layer_norm,
            )
        }

        pub fn $layer_norm_affine(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            weight: &impl DeviceSlice<$ty>,
            bias: &impl DeviceSlice<$ty>,
            rows: usize,
            cols: usize,
            eps: $ty,
        ) -> Result<()> {
            normalization(
                stream,
                out,
                input,
                Some(weight),
                Some(bias),
                rows,
                cols,
                eps,
                $zero,
                cutile::normalization::$layer_norm,
            )
        }

        pub fn $rms_norm(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            rows: usize,
            cols: usize,
            eps: $ty,
        ) -> Result<()> {
            normalization(
                stream,
                out,
                input,
                None::<&NoParameter<$ty>>,
                None::<&NoParameter<$ty>>,
                rows,
                cols,
                eps,
                $zero,
                cutile::normalization::$rms_norm,
            )
        }

        pub fn $rms_norm_affine(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            weight: &impl DeviceSlice<$ty>,
            bias: &impl DeviceSlice<$ty>,
            rows: usize,
            cols: usize,
            eps: $ty,
        ) -> Result<()> {
            normalization(
                stream,
                out,
                input,
                Some(weight),
                Some(bias),
                rows,
                cols,
                eps,
                $zero,
                cutile::normalization::$rms_norm,
            )
        }

        pub fn $rms_norm_weight_offset(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            weight: &impl DeviceSlice<$ty>,
            rows: usize,
            cols: usize,
            eps: $ty,
            weight_offset: $ty,
        ) -> Result<()> {
            normalization(
                stream,
                out,
                input,
                Some(weight),
                None::<&NoParameter<$ty>>,
                rows,
                cols,
                eps,
                weight_offset,
                cutile::normalization::$rms_norm,
            )
        }
    };
}

#[cfg(feature = "dtype-f32")]
normalization_fns!(
    f32,
    0.0f32,
    layer_norm_f32,
    layer_norm_affine_f32,
    rms_norm_f32,
    rms_norm_affine_f32,
    rms_norm_weight_offset_f32
);
#[cfg(feature = "dtype-f16")]
normalization_fns!(
    f16,
    f16::from_f32(0.0),
    layer_norm_f16,
    layer_norm_affine_f16,
    rms_norm_f16,
    rms_norm_affine_f16,
    rms_norm_weight_offset_f16
);
#[cfg(feature = "dtype-bf16")]
normalization_fns!(
    bf16,
    bf16::from_f32(0.0),
    layer_norm_bf16,
    layer_norm_affine_bf16,
    rms_norm_bf16,
    rms_norm_affine_bf16,
    rms_norm_weight_offset_bf16
);
#[cfg(feature = "dtype-f64")]
normalization_fns!(
    f64,
    0.0f64,
    layer_norm_f64,
    layer_norm_affine_f64,
    rms_norm_f64,
    rms_norm_affine_f64,
    rms_norm_weight_offset_f64
);

macro_rules! sparsemax_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            validate_sparsemax(out.len(), input.len(), rows, cols)?;
            let stream = borrowed_stream(stream)?;
            cutile::normalization::$name(
                &stream,
                output_pointer(out),
                input_pointer(input),
                rows,
                cols,
            )
        }
    };
}

#[cfg(feature = "dtype-f32")]
sparsemax_fn!(sparsemax_f32, f32);
#[cfg(feature = "dtype-f16")]
sparsemax_fn!(sparsemax_f16, f16);
#[cfg(feature = "dtype-bf16")]
sparsemax_fn!(sparsemax_bf16, bf16);

pub struct BatchNormInferenceConfig<'a, T: DeviceRepr> {
    pub weight: Option<&'a dyn DeviceSlice<T>>,
    pub bias: Option<&'a dyn DeviceSlice<T>>,
    pub running_mean: Option<&'a dyn DeviceSlice<T>>,
    pub running_var: Option<&'a dyn DeviceSlice<T>>,
    pub feature_count: usize,
    pub spatial_len: usize,
    pub eps: T,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroupNormConfig {
    pub batch: usize,
    pub channels: usize,
    pub groups: usize,
    pub spatial_len: usize,
    pub eps: f32,
}

#[cfg(feature = "dtype-f32")]
pub fn group_norm_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    weight: &impl DeviceSlice<f32>,
    bias: &impl DeviceSlice<f32>,
    config: GroupNormConfig,
) -> Result<()> {
    group_norm(stream, out, input, weight, bias, config)
}

#[cfg(feature = "dtype-f16")]
pub fn group_norm_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f16>,
    input: &impl DeviceSlice<f16>,
    weight: &impl DeviceSlice<f16>,
    bias: &impl DeviceSlice<f16>,
    config: GroupNormConfig,
) -> Result<()> {
    group_norm_f16_impl(stream, out, input, weight, bias, config)
}

#[cfg(feature = "dtype-bf16")]
pub fn group_norm_bf16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<bf16>,
    input: &impl DeviceSlice<bf16>,
    weight: &impl DeviceSlice<bf16>,
    bias: &impl DeviceSlice<bf16>,
    config: GroupNormConfig,
) -> Result<()> {
    group_norm_bf16_impl(stream, out, input, weight, bias, config)
}

macro_rules! batch_norm_inference_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            config: BatchNormInferenceConfig<'_, $ty>,
        ) -> Result<()> {
            batch_norm_inference(stream, out, input, config, cutile::normalization::$name)
        }
    };
}

#[cfg(feature = "dtype-f32")]
batch_norm_inference_fn!(batch_norm_inference_f32, f32);
#[cfg(feature = "dtype-f16")]
batch_norm_inference_fn!(batch_norm_inference_f16, f16);
#[cfg(feature = "dtype-f64")]
batch_norm_inference_fn!(batch_norm_inference_f64, f64);

#[cfg(feature = "dtype-f32")]

fn group_norm(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    weight: &impl DeviceSlice<f32>,
    bias: &impl DeviceSlice<f32>,
    config: GroupNormConfig,
) -> Result<()> {
    validate_group_norm(out.len(), input.len(), weight.len(), bias.len(), config)?;

    let stream = borrowed_stream(stream)?;
    let input_ptr = input_pointer(input);
    cutile::normalization::group_norm_f32(
        &stream,
        output_pointer(out),
        input_ptr,
        input_pointer(weight),
        input_pointer(bias),
        config.batch,
        config.channels,
        config.groups,
        config.spatial_len,
        config.eps,
    )
}

#[cfg(feature = "dtype-f16")]

fn group_norm_f16_impl(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f16>,
    input: &impl DeviceSlice<f16>,
    weight: &impl DeviceSlice<f16>,
    bias: &impl DeviceSlice<f16>,
    config: GroupNormConfig,
) -> Result<()> {
    validate_group_norm(out.len(), input.len(), weight.len(), bias.len(), config)?;

    let stream = borrowed_stream(stream)?;
    let input_ptr = input_pointer(input);
    cutile::normalization::group_norm_f16(
        &stream,
        output_pointer(out),
        input_ptr,
        input_pointer(weight),
        input_pointer(bias),
        config.batch,
        config.channels,
        config.groups,
        config.spatial_len,
        config.eps,
    )
}

#[cfg(feature = "dtype-bf16")]

fn group_norm_bf16_impl(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<bf16>,
    input: &impl DeviceSlice<bf16>,
    weight: &impl DeviceSlice<bf16>,
    bias: &impl DeviceSlice<bf16>,
    config: GroupNormConfig,
) -> Result<()> {
    validate_group_norm(out.len(), input.len(), weight.len(), bias.len(), config)?;

    let stream = borrowed_stream(stream)?;
    let input_ptr = input_pointer(input);
    cutile::normalization::group_norm_bf16(
        &stream,
        output_pointer(out),
        input_ptr,
        input_pointer(weight),
        input_pointer(bias),
        config.batch,
        config.channels,
        config.groups,
        config.spatial_len,
        config.eps,
    )
}

#[cfg(feature = "cutile")]
fn batch_norm_inference<T, F>(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<T>,
    input: &impl DeviceSlice<T>,
    config: BatchNormInferenceConfig<'_, T>,
    launch: F,
) -> Result<()>
where
    T: DeviceRepr,
    F: FnOnce(
        &Arc<::cutile::cuda_core::Stream>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        usize,
        usize,
        bool,
        bool,
        bool,
        bool,
        T,
        usize,
    ) -> Result<()>,
{
    if config.feature_count == 0 || config.spatial_len == 0 {
        return Err(Error::InvalidLength);
    }
    let len = out.len();
    ensure_len(input.len(), len)?;
    if let Some(weight) = config.weight {
        ensure_len(weight.len(), config.feature_count)?;
    }
    if let Some(bias) = config.bias {
        ensure_len(bias.len(), config.feature_count)?;
    }
    if let Some(running_mean) = config.running_mean {
        ensure_len(running_mean.len(), config.feature_count)?;
    }
    if let Some(running_var) = config.running_var {
        ensure_len(running_var.len(), config.feature_count)?;
    }
    let feature_period = checked_element_count(config.feature_count, config.spatial_len)?;
    if len % feature_period != 0 {
        return Err(Error::LengthMismatch);
    }

    let stream = borrowed_stream(stream)?;
    let input_ptr = input_pointer(input);
    let weight_ptr = config
        .weight
        .map_or(input_ptr, |weight| input_pointer(weight));
    let bias_ptr = config.bias.map_or(input_ptr, |bias| input_pointer(bias));
    let running_mean_ptr = config
        .running_mean
        .map_or(input_ptr, |running_mean| input_pointer(running_mean));
    let running_var_ptr = config
        .running_var
        .map_or(input_ptr, |running_var| input_pointer(running_var));
    launch(
        &stream,
        output_pointer(out),
        input_ptr,
        weight_ptr,
        bias_ptr,
        running_mean_ptr,
        running_var_ptr,
        config.feature_count,
        config.spatial_len,
        config.weight.is_some(),
        config.bias.is_some(),
        config.running_mean.is_some(),
        config.running_var.is_some(),
        config.eps,
        len,
    )
}

fn validate_sparsemax(out_len: usize, input_len: usize, rows: usize, cols: usize) -> Result<usize> {
    if rows == 0 || cols == 0 {
        return Err(Error::InvalidLength);
    }
    let len = checked_element_count(rows, cols)?;
    ensure_len(out_len, len)?;
    ensure_len(input_len, len)?;
    Ok(len)
}

fn validate_group_norm(
    out_len: usize,
    input_len: usize,
    weight_len: usize,
    bias_len: usize,
    config: GroupNormConfig,
) -> Result<()> {
    if config.batch == 0 || config.channels == 0 || config.groups == 0 || config.spatial_len == 0 {
        return Err(Error::InvalidLength);
    }
    if config.channels % config.groups != 0 {
        return Err(Error::LengthMismatch);
    }
    let len = checked_element_count(
        checked_element_count(config.batch, config.channels)?,
        config.spatial_len,
    )?;
    ensure_len(out_len, len)?;
    ensure_len(input_len, len)?;
    ensure_len(weight_len, config.channels)?;
    ensure_len(bias_len, config.channels)?;
    Ok(())
}

#[cfg(feature = "cutile")]
fn normalization<T, W, B, F>(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<T>,
    input: &impl DeviceSlice<T>,
    weight: Option<&W>,
    bias: Option<&B>,
    rows: usize,
    cols: usize,
    eps: T,
    weight_offset: T,
    launch: F,
) -> Result<()>
where
    T: DeviceRepr,
    W: DeviceSlice<T>,
    B: DeviceSlice<T>,
    F: FnOnce(
        &Arc<::cutile::cuda_core::Stream>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        bool,
        bool,
        usize,
        usize,
        T,
        T,
    ) -> Result<()>,
{
    validate_normalization_lengths(
        out.len(),
        input.len(),
        weight.map(|weight| weight.len()),
        bias.map(|bias| bias.len()),
        rows,
        cols,
    )?;

    let stream = borrowed_stream(stream)?;
    let input = input_pointer(input);
    let weight_pointer = weight.map_or(input, input_pointer);
    let bias_pointer = bias.map_or(input, input_pointer);
    launch(
        &stream,
        output_pointer(out),
        input,
        weight_pointer,
        bias_pointer,
        weight.is_some(),
        bias.is_some(),
        rows,
        cols,
        eps,
        weight_offset,
    )
}

fn validate_normalization_lengths(
    out_len: usize,
    input_len: usize,
    weight_len: Option<usize>,
    bias_len: Option<usize>,
    rows: usize,
    cols: usize,
) -> Result<usize> {
    if cols == 0 {
        return Err(Error::InvalidLength);
    }
    let len = checked_element_count(rows, cols)?;
    ensure_len(out_len, len)?;
    ensure_len(input_len, len)?;
    if let Some(weight_len) = weight_len {
        ensure_len(weight_len, cols)?;
    }
    if let Some(bias_len) = bias_len {
        ensure_len(bias_len, cols)?;
    }
    Ok(len)
}

struct NoParameter<T>(PhantomData<T>);

impl<T: DeviceRepr> DeviceSlice<T> for NoParameter<T> {
    fn as_device_ptr(&self) -> *const T {
        std::ptr::null()
    }

    fn len(&self) -> usize {
        0
    }
}

impl<T: DeviceRepr> DeviceSliceMut<T> for NoParameter<T> {
    fn as_device_mut_ptr(&mut self) -> *mut T {
        std::ptr::null_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparsemax_validation_accepts_exact_lengths() -> Result<()> {
        assert_eq!(validate_sparsemax(15, 15, 3, 5)?, 15);
        Ok(())
    }

    #[test]
    fn sparsemax_validation_rejects_zero_rows() {
        assert!(matches!(
            validate_sparsemax(0, 0, 0, 5),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn sparsemax_validation_rejects_short_input() {
        assert!(matches!(
            validate_sparsemax(15, 14, 3, 5),
            Err(Error::LengthMismatch)
        ));
    }

    #[test]
    fn bf16_normalization_validation_accepts_affine_lengths() -> Result<()> {
        assert_eq!(
            validate_normalization_lengths(24, 24, Some(6), Some(6), 4, 6)?,
            24
        );
        Ok(())
    }

    #[test]
    fn bf16_normalization_validation_rejects_short_weight() {
        assert!(matches!(
            validate_normalization_lengths(24, 24, Some(5), None, 4, 6),
            Err(Error::LengthMismatch)
        ));
    }

    #[test]
    fn bf16_normalization_validation_rejects_zero_cols() {
        assert!(matches!(
            validate_normalization_lengths(0, 0, None, None, 4, 0),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn group_norm_validation_accepts_exact_lengths() -> Result<()> {
        let config = group_norm_validation_config();
        validate_group_norm(48, 48, 6, 6, config)?;
        Ok(())
    }

    #[test]
    fn group_norm_validation_rejects_uneven_groups() {
        let mut config = group_norm_validation_config();
        config.groups = 4;
        assert!(matches!(
            validate_group_norm(48, 48, 6, 6, config),
            Err(Error::LengthMismatch)
        ));
    }

    #[test]
    fn group_norm_validation_rejects_short_weight() {
        let config = group_norm_validation_config();
        assert!(matches!(
            validate_group_norm(48, 48, 5, 6, config),
            Err(Error::LengthMismatch)
        ));
    }

    fn group_norm_validation_config() -> GroupNormConfig {
        GroupNormConfig {
            batch: 2,
            channels: 6,
            groups: 2,
            spatial_len: 4,
            eps: 1e-5,
        }
    }
}
