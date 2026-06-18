use singe_cuda::types::{Complex32, Complex64};
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    signal::view::{SignalView, SignalViewMut},
    try_ffi,
    types::{ComplexI16, ComplexI32, ComplexI64, DataTypeLike, IntoNpp},
    utility::{to_u64, validate_same_len},
};

macro_rules! impl_set {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            value: $ty,
            destination: &mut SignalViewMut<'_, $ty>,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    value.into_npp(),
                    destination.as_mut_ptr().cast(),
                    to_u64(destination.len(), "len")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_zero {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            destination: &mut SignalViewMut<'_, $ty>,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    destination.as_mut_ptr().cast(),
                    to_u64(destination.len(), "len")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_copy {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $ty>,
            destination: &mut SignalViewMut<'_, $ty>,
        ) -> Result<()> {
            validate_same_len(source.len(), destination.len(), "destination")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(source.len(), "len")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

impl_set!(set_u8, u8, nppsSet_8u_Ctx);
impl_set!(set_i8, i8, nppsSet_8s_Ctx);
impl_set!(set_u16, u16, nppsSet_16u_Ctx);
impl_set!(set_i16, i16, nppsSet_16s_Ctx);
impl_set!(set_u32, u32, nppsSet_32u_Ctx);
impl_set!(set_i32, i32, nppsSet_32s_Ctx);
impl_set!(set_i64, i64, nppsSet_64s_Ctx);
impl_set!(set_f32, f32, nppsSet_32f_Ctx);
impl_set!(set_f64, f64, nppsSet_64f_Ctx);

impl_set!(set_i16_complex, ComplexI16, nppsSet_16sc_Ctx);
impl_set!(set_i32_complex, ComplexI32, nppsSet_32sc_Ctx);
impl_set!(set_i64_complex, ComplexI64, nppsSet_64sc_Ctx);
impl_set!(set_f32_complex, Complex32, nppsSet_32fc_Ctx);
impl_set!(set_f64_complex, Complex64, nppsSet_64fc_Ctx);

impl_zero!(zero_u8, u8, nppsZero_8u_Ctx);
impl_zero!(zero_i16, i16, nppsZero_16s_Ctx);
impl_zero!(zero_i32, i32, nppsZero_32s_Ctx);
impl_zero!(zero_i64, i64, nppsZero_64s_Ctx);
impl_zero!(zero_f32, f32, nppsZero_32f_Ctx);
impl_zero!(zero_f64, f64, nppsZero_64f_Ctx);

impl_zero!(zero_i16_complex, ComplexI16, nppsZero_16sc_Ctx);
impl_zero!(zero_i32_complex, ComplexI32, nppsZero_32sc_Ctx);
impl_zero!(zero_i64_complex, ComplexI64, nppsZero_64sc_Ctx);
impl_zero!(zero_f32_complex, Complex32, nppsZero_32fc_Ctx);
impl_zero!(zero_f64_complex, Complex64, nppsZero_64fc_Ctx);

impl_copy!(copy_u8, u8, nppsCopy_8u_Ctx);
impl_copy!(copy_i16, i16, nppsCopy_16s_Ctx);
impl_copy!(copy_i32, i32, nppsCopy_32s_Ctx);
impl_copy!(copy_i64, i64, nppsCopy_64s_Ctx);
impl_copy!(copy_f32, f32, nppsCopy_32f_Ctx);

impl_copy!(copy_i16_complex, ComplexI16, nppsCopy_16sc_Ctx);
impl_copy!(copy_i32_complex, ComplexI32, nppsCopy_32sc_Ctx);
impl_copy!(copy_i64_complex, ComplexI64, nppsCopy_64sc_Ctx);
impl_copy!(copy_f32_complex, Complex32, nppsCopy_32fc_Ctx);
impl_copy!(copy_f64_complex, Complex64, nppsCopy_64fc_Ctx);

macro_rules! impl_signal_set_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            fn $method(
                stream_context: &StreamContext,
                value: Self,
                destination: &mut SignalViewMut<'_, Self>,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    value: Self,
                    destination: &mut SignalViewMut<'_, Self>,
                ) -> Result<()> {
                    $direct(stream_context, value, destination)
                }
            }
        )*

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            value: T,
            destination: &mut SignalViewMut<'_, T>,
        ) -> Result<()> {
            T::$method(stream_context, value, destination)
        }
    };
}

macro_rules! impl_signal_zero_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            fn $method(
                stream_context: &StreamContext,
                destination: &mut SignalViewMut<'_, Self>,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    destination: &mut SignalViewMut<'_, Self>,
                ) -> Result<()> {
                    $direct(stream_context, destination)
                }
            }
        )*

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            destination: &mut SignalViewMut<'_, T>,
        ) -> Result<()> {
            T::$method(stream_context, destination)
        }
    };
}

macro_rules! impl_signal_copy_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                destination: &mut SignalViewMut<'_, Self>,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    destination: &mut SignalViewMut<'_, Self>,
                ) -> Result<()> {
                    $direct(stream_context, source, destination)
                }
            }
        )*

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            destination: &mut SignalViewMut<'_, T>,
        ) -> Result<()> {
            T::$method(stream_context, source, destination)
        }
    };
}

impl_signal_set_dispatch!(Set, set, set, [
    u8 => set_u8,
    i8 => set_i8,
    u16 => set_u16,
    i16 => set_i16,
    u32 => set_u32,
    i32 => set_i32,
    i64 => set_i64,
    f32 => set_f32,
    f64 => set_f64,
    ComplexI16 => set_i16_complex,
    ComplexI32 => set_i32_complex,
    ComplexI64 => set_i64_complex,
    Complex32 => set_f32_complex,
    Complex64 => set_f64_complex
]);

impl_signal_zero_dispatch!(Zero, zero, zero, [
    u8 => zero_u8,
    i16 => zero_i16,
    i32 => zero_i32,
    i64 => zero_i64,
    f32 => zero_f32,
    f64 => zero_f64,
    ComplexI16 => zero_i16_complex,
    ComplexI32 => zero_i32_complex,
    ComplexI64 => zero_i64_complex,
    Complex32 => zero_f32_complex,
    Complex64 => zero_f64_complex
]);

impl_signal_copy_dispatch!(Copy, copy, copy, [
    u8 => copy_u8,
    i16 => copy_i16,
    i32 => copy_i32,
    i64 => copy_i64,
    f32 => copy_f32,
    ComplexI16 => copy_i16_complex,
    ComplexI32 => copy_i32_complex,
    ComplexI64 => copy_i64_complex,
    Complex32 => copy_f32_complex,
    Complex64 => copy_f64_complex
]);
