use singe_cuda::types::{Complex32, Complex64};
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    signal::view::{SignalView, SignalViewMut},
    try_ffi,
    types::{ComparisonOperation, ComplexI16, DataTypeLike, IntoNpp, RoundMode},
    utility::{to_u64, validate_same_len},
};

#[macro_use]
#[path = "conversion_macros.rs"]
mod conversion_macros;

impl_convert!(convert_i8_to_f32, i8, f32, nppsConvert_8s32f_Ctx);
impl_convert!(convert_i8_to_i16, i8, i16, nppsConvert_8s16s_Ctx);
impl_convert!(convert_u8_to_f32, u8, f32, nppsConvert_8u32f_Ctx);
impl_convert!(convert_i16_to_f32, i16, f32, nppsConvert_16s32f_Ctx);
impl_convert!(convert_i16_to_i32, i16, i32, nppsConvert_16s32s_Ctx);
impl_convert!(convert_u16_to_f32, u16, f32, nppsConvert_16u32f_Ctx);
impl_convert!(convert_i32_to_i16, i32, i16, nppsConvert_32s16s_Ctx);
impl_convert!(convert_i32_to_f32, i32, f32, nppsConvert_32s32f_Ctx);
impl_convert!(convert_i32_to_f64, i32, f64, nppsConvert_32s64f_Ctx);
impl_convert!(convert_f32_to_f64, f32, f64, nppsConvert_32f64f_Ctx);
impl_convert!(convert_i64_to_f64, i64, f64, nppsConvert_64s64f_Ctx);
impl_convert!(convert_f64_to_f32, f64, f32, nppsConvert_64f32f_Ctx);

impl_convert_scaled!(
    convert_i16_to_f32_scaled,
    i16,
    f32,
    nppsConvert_16s32f_Sfs_Ctx
);
impl_convert_scaled!(
    convert_i16_to_f64_scaled,
    i16,
    f64,
    nppsConvert_16s64f_Sfs_Ctx
);
impl_convert_scaled!(
    convert_i32_to_i16_scaled,
    i32,
    i16,
    nppsConvert_32s16s_Sfs_Ctx
);
impl_convert_scaled!(
    convert_i32_to_f32_scaled,
    i32,
    f32,
    nppsConvert_32s32f_Sfs_Ctx
);
impl_convert_scaled!(
    convert_i32_to_f64_scaled,
    i32,
    f64,
    nppsConvert_32s64f_Sfs_Ctx
);

impl_convert_scaled_round!(convert_i16_to_i8_scaled, i16, i8, nppsConvert_16s8s_Sfs_Ctx);

impl_convert_scaled_round!(convert_f32_to_i8_scaled, f32, i8, nppsConvert_32f8s_Sfs_Ctx);
impl_convert_scaled_round!(convert_f32_to_u8_scaled, f32, u8, nppsConvert_32f8u_Sfs_Ctx);
impl_convert_scaled_round!(
    convert_f32_to_i16_scaled,
    f32,
    i16,
    nppsConvert_32f16s_Sfs_Ctx
);
impl_convert_scaled_round!(
    convert_f32_to_u16_scaled,
    f32,
    u16,
    nppsConvert_32f16u_Sfs_Ctx
);
impl_convert_scaled_round!(
    convert_f32_to_i32_scaled,
    f32,
    i32,
    nppsConvert_32f32s_Sfs_Ctx
);
impl_convert_scaled_round!(
    convert_i64_to_i32_scaled,
    i64,
    i32,
    nppsConvert_64s32s_Sfs_Ctx
);
impl_convert_scaled_round!(
    convert_f64_to_i16_scaled,
    f64,
    i16,
    nppsConvert_64f16s_Sfs_Ctx
);
impl_convert_scaled_round!(
    convert_f64_to_i32_scaled,
    f64,
    i32,
    nppsConvert_64f32s_Sfs_Ctx
);
impl_convert_scaled_round!(
    convert_f64_to_i64_scaled,
    f64,
    i64,
    nppsConvert_64f64s_Sfs_Ctx
);

macro_rules! impl_signal_convert_to_dispatch {
    ($source_ty:ty, $destination_ty:ty, $direct:ident) => {
        impl ConvertTo<$destination_ty> for $source_ty {
            fn convert_to(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                destination: &mut SignalViewMut<'_, $destination_ty>,
            ) -> Result<()> {
                $direct(stream_context, source, destination)
            }
        }
    };
}

macro_rules! impl_signal_convert_scaled_to_dispatch {
    ($source_ty:ty, $destination_ty:ty, $direct:ident) => {
        impl ConvertScaledTo<$destination_ty> for $source_ty {
            fn convert_scaled_to(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                destination: &mut SignalViewMut<'_, $destination_ty>,
                scale_factor: i32,
            ) -> Result<()> {
                $direct(stream_context, source, destination, scale_factor)
            }
        }
    };
}

macro_rules! impl_signal_convert_scaled_round_to_dispatch {
    ($source_ty:ty, $destination_ty:ty, $direct:ident) => {
        impl ConvertScaledRoundTo<$destination_ty> for $source_ty {
            fn convert_scaled_round_to(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                destination: &mut SignalViewMut<'_, $destination_ty>,
                round_mode: RoundMode,
                scale_factor: i32,
            ) -> Result<()> {
                $direct(
                    stream_context,
                    source,
                    destination,
                    round_mode,
                    scale_factor,
                )
            }
        }
    };
}

pub trait ConvertTo<Destination>: DataTypeLike
where
    Destination: DataTypeLike,
{
    fn convert_to(
        stream_context: &StreamContext,
        source: &SignalView<'_, Self>,
        destination: &mut SignalViewMut<'_, Destination>,
    ) -> Result<()>;
}

pub fn convert_to<Source, Destination>(
    stream_context: &StreamContext,
    source: &SignalView<'_, Source>,
    destination: &mut SignalViewMut<'_, Destination>,
) -> Result<()>
where
    Source: ConvertTo<Destination>,
    Destination: DataTypeLike,
{
    Source::convert_to(stream_context, source, destination)
}

pub trait ConvertScaledTo<Destination>: DataTypeLike
where
    Destination: DataTypeLike,
{
    fn convert_scaled_to(
        stream_context: &StreamContext,
        source: &SignalView<'_, Self>,
        destination: &mut SignalViewMut<'_, Destination>,
        scale_factor: i32,
    ) -> Result<()>;
}

pub fn convert_scaled_to<Source, Destination>(
    stream_context: &StreamContext,
    source: &SignalView<'_, Source>,
    destination: &mut SignalViewMut<'_, Destination>,
    scale_factor: i32,
) -> Result<()>
where
    Source: ConvertScaledTo<Destination>,
    Destination: DataTypeLike,
{
    Source::convert_scaled_to(stream_context, source, destination, scale_factor)
}

pub trait ConvertScaledRoundTo<Destination>: DataTypeLike
where
    Destination: DataTypeLike,
{
    fn convert_scaled_round_to(
        stream_context: &StreamContext,
        source: &SignalView<'_, Self>,
        destination: &mut SignalViewMut<'_, Destination>,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<()>;
}

pub fn convert_scaled_round_to<Source, Destination>(
    stream_context: &StreamContext,
    source: &SignalView<'_, Source>,
    destination: &mut SignalViewMut<'_, Destination>,
    round_mode: RoundMode,
    scale_factor: i32,
) -> Result<()>
where
    Source: ConvertScaledRoundTo<Destination>,
    Destination: DataTypeLike,
{
    Source::convert_scaled_round_to(
        stream_context,
        source,
        destination,
        round_mode,
        scale_factor,
    )
}

impl_signal_convert_to_dispatch!(i8, f32, convert_i8_to_f32);
impl_signal_convert_to_dispatch!(i8, i16, convert_i8_to_i16);
impl_signal_convert_to_dispatch!(u8, f32, convert_u8_to_f32);
impl_signal_convert_to_dispatch!(i16, f32, convert_i16_to_f32);
impl_signal_convert_to_dispatch!(i16, i32, convert_i16_to_i32);
impl_signal_convert_to_dispatch!(u16, f32, convert_u16_to_f32);
impl_signal_convert_to_dispatch!(i32, i16, convert_i32_to_i16);
impl_signal_convert_to_dispatch!(i32, f32, convert_i32_to_f32);
impl_signal_convert_to_dispatch!(i32, f64, convert_i32_to_f64);
impl_signal_convert_to_dispatch!(f32, f64, convert_f32_to_f64);
impl_signal_convert_to_dispatch!(i64, f64, convert_i64_to_f64);
impl_signal_convert_to_dispatch!(f64, f32, convert_f64_to_f32);

impl_signal_convert_scaled_to_dispatch!(i16, f32, convert_i16_to_f32_scaled);
impl_signal_convert_scaled_to_dispatch!(i16, f64, convert_i16_to_f64_scaled);
impl_signal_convert_scaled_to_dispatch!(i32, i16, convert_i32_to_i16_scaled);
impl_signal_convert_scaled_to_dispatch!(i32, f32, convert_i32_to_f32_scaled);
impl_signal_convert_scaled_to_dispatch!(i32, f64, convert_i32_to_f64_scaled);

impl_signal_convert_scaled_round_to_dispatch!(i16, i8, convert_i16_to_i8_scaled);
impl_signal_convert_scaled_round_to_dispatch!(f32, i8, convert_f32_to_i8_scaled);
impl_signal_convert_scaled_round_to_dispatch!(f32, u8, convert_f32_to_u8_scaled);
impl_signal_convert_scaled_round_to_dispatch!(f32, i16, convert_f32_to_i16_scaled);
impl_signal_convert_scaled_round_to_dispatch!(f32, u16, convert_f32_to_u16_scaled);
impl_signal_convert_scaled_round_to_dispatch!(f32, i32, convert_f32_to_i32_scaled);
impl_signal_convert_scaled_round_to_dispatch!(i64, i32, convert_i64_to_i32_scaled);
impl_signal_convert_scaled_round_to_dispatch!(f64, i16, convert_f64_to_i16_scaled);
impl_signal_convert_scaled_round_to_dispatch!(f64, i32, convert_f64_to_i32_scaled);
impl_signal_convert_scaled_round_to_dispatch!(f64, i64, convert_f64_to_i64_scaled);

impl_threshold!(threshold_i16, i16, nppsThreshold_16s_Ctx);
impl_threshold_in_place!(threshold_i16_in_place, i16, nppsThreshold_16s_I_Ctx);
impl_threshold_complex!(
    threshold_i16_complex,
    ComplexI16,
    i16,
    nppsThreshold_16sc_Ctx
);
impl_threshold_complex_in_place!(
    threshold_i16_complex_in_place,
    ComplexI16,
    i16,
    nppsThreshold_16sc_I_Ctx
);
impl_threshold!(threshold_f32, f32, nppsThreshold_32f_Ctx);
impl_threshold_in_place!(threshold_f32_in_place, f32, nppsThreshold_32f_I_Ctx);
impl_threshold_complex!(
    threshold_f32_complex,
    Complex32,
    f32,
    nppsThreshold_32fc_Ctx
);
impl_threshold_complex_in_place!(
    threshold_f32_complex_in_place,
    Complex32,
    f32,
    nppsThreshold_32fc_I_Ctx
);
impl_threshold!(threshold_f64, f64, nppsThreshold_64f_Ctx);
impl_threshold_in_place!(threshold_f64_in_place, f64, nppsThreshold_64f_I_Ctx);
impl_threshold_complex!(
    threshold_f64_complex,
    Complex64,
    f64,
    nppsThreshold_64fc_Ctx
);
impl_threshold_complex_in_place!(
    threshold_f64_complex_in_place,
    Complex64,
    f64,
    nppsThreshold_64fc_I_Ctx
);
impl_threshold_fixed!(threshold_less_i16, i16, nppsThreshold_LT_16s_Ctx);
impl_threshold_fixed_in_place!(threshold_less_i16_in_place, i16, nppsThreshold_LT_16s_I_Ctx);
impl_threshold_fixed_complex!(
    threshold_less_i16_complex,
    ComplexI16,
    i16,
    nppsThreshold_LT_16sc_Ctx
);
impl_threshold_fixed_complex_in_place!(
    threshold_less_i16_complex_in_place,
    ComplexI16,
    i16,
    nppsThreshold_LT_16sc_I_Ctx
);
impl_threshold_fixed!(threshold_less_f32, f32, nppsThreshold_LT_32f_Ctx);
impl_threshold_fixed_in_place!(threshold_less_f32_in_place, f32, nppsThreshold_LT_32f_I_Ctx);
impl_threshold_fixed_complex!(
    threshold_less_f32_complex,
    Complex32,
    f32,
    nppsThreshold_LT_32fc_Ctx
);
impl_threshold_fixed_complex_in_place!(
    threshold_less_f32_complex_in_place,
    Complex32,
    f32,
    nppsThreshold_LT_32fc_I_Ctx
);
impl_threshold_fixed!(threshold_less_f64, f64, nppsThreshold_LT_64f_Ctx);
impl_threshold_fixed_in_place!(threshold_less_f64_in_place, f64, nppsThreshold_LT_64f_I_Ctx);
impl_threshold_fixed_complex!(
    threshold_less_f64_complex,
    Complex64,
    f64,
    nppsThreshold_LT_64fc_Ctx
);
impl_threshold_fixed_complex_in_place!(
    threshold_less_f64_complex_in_place,
    Complex64,
    f64,
    nppsThreshold_LT_64fc_I_Ctx
);
impl_threshold_fixed!(threshold_greater_i16, i16, nppsThreshold_GT_16s_Ctx);
impl_threshold_fixed_in_place!(
    threshold_greater_i16_in_place,
    i16,
    nppsThreshold_GT_16s_I_Ctx
);
impl_threshold_fixed_complex!(
    threshold_greater_i16_complex,
    ComplexI16,
    i16,
    nppsThreshold_GT_16sc_Ctx
);
impl_threshold_fixed_complex_in_place!(
    threshold_greater_i16_complex_in_place,
    ComplexI16,
    i16,
    nppsThreshold_GT_16sc_I_Ctx
);
impl_threshold_fixed!(threshold_greater_f32, f32, nppsThreshold_GT_32f_Ctx);
impl_threshold_fixed_in_place!(
    threshold_greater_f32_in_place,
    f32,
    nppsThreshold_GT_32f_I_Ctx
);
impl_threshold_fixed_complex!(
    threshold_greater_f32_complex,
    Complex32,
    f32,
    nppsThreshold_GT_32fc_Ctx
);
impl_threshold_fixed_complex_in_place!(
    threshold_greater_f32_complex_in_place,
    Complex32,
    f32,
    nppsThreshold_GT_32fc_I_Ctx
);
impl_threshold_fixed!(threshold_greater_f64, f64, nppsThreshold_GT_64f_Ctx);
impl_threshold_fixed_in_place!(
    threshold_greater_f64_in_place,
    f64,
    nppsThreshold_GT_64f_I_Ctx
);
impl_threshold_fixed_complex!(
    threshold_greater_f64_complex,
    Complex64,
    f64,
    nppsThreshold_GT_64fc_Ctx
);
impl_threshold_fixed_complex_in_place!(
    threshold_greater_f64_complex_in_place,
    Complex64,
    f64,
    nppsThreshold_GT_64fc_I_Ctx
);
impl_threshold_value!(threshold_less_value_i16, i16, nppsThreshold_LTVal_16s_Ctx);
impl_threshold_value_in_place!(
    threshold_less_value_i16_in_place,
    i16,
    nppsThreshold_LTVal_16s_I_Ctx
);
impl_threshold_value_complex!(
    threshold_less_value_i16_complex,
    ComplexI16,
    i16,
    nppsThreshold_LTVal_16sc_Ctx
);
impl_threshold_value_complex_in_place!(
    threshold_less_value_i16_complex_in_place,
    ComplexI16,
    i16,
    nppsThreshold_LTVal_16sc_I_Ctx
);
impl_threshold_value!(threshold_less_value_f32, f32, nppsThreshold_LTVal_32f_Ctx);
impl_threshold_value_in_place!(
    threshold_less_value_f32_in_place,
    f32,
    nppsThreshold_LTVal_32f_I_Ctx
);
impl_threshold_value_complex!(
    threshold_less_value_f32_complex,
    Complex32,
    f32,
    nppsThreshold_LTVal_32fc_Ctx
);
impl_threshold_value_complex_in_place!(
    threshold_less_value_f32_complex_in_place,
    Complex32,
    f32,
    nppsThreshold_LTVal_32fc_I_Ctx
);
impl_threshold_value!(threshold_less_value_f64, f64, nppsThreshold_LTVal_64f_Ctx);
impl_threshold_value_in_place!(
    threshold_less_value_f64_in_place,
    f64,
    nppsThreshold_LTVal_64f_I_Ctx
);
impl_threshold_value_complex!(
    threshold_less_value_f64_complex,
    Complex64,
    f64,
    nppsThreshold_LTVal_64fc_Ctx
);
impl_threshold_value_complex_in_place!(
    threshold_less_value_f64_complex_in_place,
    Complex64,
    f64,
    nppsThreshold_LTVal_64fc_I_Ctx
);
impl_threshold_value!(
    threshold_greater_value_i16,
    i16,
    nppsThreshold_GTVal_16s_Ctx
);
impl_threshold_value_in_place!(
    threshold_greater_value_i16_in_place,
    i16,
    nppsThreshold_GTVal_16s_I_Ctx
);
impl_threshold_value_complex!(
    threshold_greater_value_i16_complex,
    ComplexI16,
    i16,
    nppsThreshold_GTVal_16sc_Ctx
);
impl_threshold_value_complex_in_place!(
    threshold_greater_value_i16_complex_in_place,
    ComplexI16,
    i16,
    nppsThreshold_GTVal_16sc_I_Ctx
);
impl_threshold_value!(
    threshold_greater_value_f32,
    f32,
    nppsThreshold_GTVal_32f_Ctx
);
impl_threshold_value_in_place!(
    threshold_greater_value_f32_in_place,
    f32,
    nppsThreshold_GTVal_32f_I_Ctx
);
impl_threshold_value_complex!(
    threshold_greater_value_f32_complex,
    Complex32,
    f32,
    nppsThreshold_GTVal_32fc_Ctx
);
impl_threshold_value_complex_in_place!(
    threshold_greater_value_f32_complex_in_place,
    Complex32,
    f32,
    nppsThreshold_GTVal_32fc_I_Ctx
);
impl_threshold_value!(
    threshold_greater_value_f64,
    f64,
    nppsThreshold_GTVal_64f_Ctx
);
impl_threshold_value_in_place!(
    threshold_greater_value_f64_in_place,
    f64,
    nppsThreshold_GTVal_64f_I_Ctx
);
impl_threshold_value_complex!(
    threshold_greater_value_f64_complex,
    Complex64,
    f64,
    nppsThreshold_GTVal_64fc_Ctx
);
impl_threshold_value_complex_in_place!(
    threshold_greater_value_f64_complex_in_place,
    Complex64,
    f64,
    nppsThreshold_GTVal_64fc_I_Ctx
);
#[path = "conversion_threshold_dispatch.rs"]
mod conversion_threshold_dispatch;
pub use conversion_threshold_dispatch::*;
