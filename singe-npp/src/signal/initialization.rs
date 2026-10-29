use singe_cuda::types::{Complex32, Complex64};
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    signal::view::{SignalOutputView, SignalView},
    try_ffi,
    types::{ComplexI16, ComplexI32, ComplexI64, DataTypeLike, IntoNpp},
    utility::{to_u64, validate_same_len},
};

macro_rules! impl_set {
    ($name:ident, $ty:ty, $ffi:ident) => {
        /// Fills every element of `destination` with `value`.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, the length cannot be represented by the native interface, or NPP fails.
        /// Success reports native submission. Destination elements become initialized only after
        /// the queued writes complete successfully.
        ///
        /// # Safety
        ///
        /// Destination addresses and capacity must satisfy the selected native routine's alignment
        /// and representation requirements. Keep the destination exclusively writable through
        /// queued completion; it may be uninitialized before the operation.
        /// Retain all participating allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        pub unsafe fn $name(
            stream_context: &StreamContext,
            value: $ty,
            destination: &mut SignalOutputView<'_, $ty>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(destination)?;

            // SAFETY: the caller upholds this function's safety contract.
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
        /// Fills every element of `destination` with zero.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, the length cannot be represented by the native interface, or NPP fails.
        /// Success reports native submission. Destination elements become initialized only after
        /// the queued writes complete successfully.
        ///
        /// # Safety
        ///
        /// Destination addresses and capacity must satisfy the selected native routine's alignment
        /// and representation requirements. Keep the destination exclusively writable through
        /// queued completion; it may be uninitialized before the operation.
        /// Retain all participating allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        pub unsafe fn $name(
            stream_context: &StreamContext,
            destination: &mut SignalOutputView<'_, $ty>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(destination)?;

            // SAFETY: the caller upholds this function's safety contract.
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
        /// Copies every element of `source` to an equally sized `destination`.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, the length cannot be represented by the native interface, or NPP fails.
        /// Source and destination lengths must match; differing lengths return an error.
        /// Success reports native submission. Destination elements become initialized only after
        /// the queued writes complete successfully.
        ///
        /// # Safety
        ///
        /// Destination addresses and capacity must satisfy the selected native routine's alignment
        /// and representation requirements. Keep the destination exclusively writable through
        /// queued completion; it may be uninitialized before the operation.
        /// Source elements must be initialized and ready when the stream reaches the copy.
        /// Keep the source read-only and disjoint from the destination through completion.
        /// Retain all participating allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $ty>,
            destination: &mut SignalOutputView<'_, $ty>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_len(source.len(), destination.len(), "destination")?;

            // SAFETY: the caller upholds this function's safety contract.
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
        #[doc = concat!("Element-type dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must match the native element representation and access roles.
        /// Successful queued completion must initialize every destination element according
        /// to the method contract, while preserving allocation lifetimes and exclusivity.
        pub unsafe trait $trait: DataTypeLike {
            /// Fills every element of `destination` with `value`.
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, a signal belongs
            /// context, the length cannot be represented by the native interface, or NPP fails.
            /// Success reports native submission. Destination elements become initialized only after
            /// the queued writes complete successfully.
            ///
            /// # Safety
            ///
            /// Destination addresses and capacity must satisfy the selected native routine's alignment
            /// and representation requirements. Keep the destination exclusively writable through
            /// queued completion; it may be uninitialized before the operation.
            /// Retain all participating allocations and the stream context through queued completion,
            /// including submission errors and every captured graph replay.
            unsafe fn $method(
                stream_context: &StreamContext,
                value: Self,
                destination: &mut SignalOutputView<'_, Self>,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    value: Self,
                    destination: &mut SignalOutputView<'_, Self>,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct(stream_context, value, destination)

        }
    }
            }
        )*

        /// Fills every element of `destination` with `value`.
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, a signal belongs
        /// context, the length cannot be represented by the native interface, or NPP fails.
        /// Success reports native submission. Destination elements become initialized only after
        /// the queued writes complete successfully.
        ///
        /// # Safety
        ///
        /// Destination addresses and capacity must satisfy the selected native routine's alignment
        /// and representation requirements. Keep the destination exclusively writable through
        /// queued completion; it may be uninitialized before the operation.
        /// Retain all participating allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        pub unsafe fn $function<T: $trait>(
            stream_context: &StreamContext,
            value: T,
            destination: &mut SignalOutputView<'_, T>,
        ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            T::$method(stream_context, value, destination)

        }
    }
    };
}

macro_rules! impl_signal_zero_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Element-type dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must match the native element representation and access roles.
        /// Successful queued completion must initialize every destination element according
        /// to the method contract, while preserving allocation lifetimes and exclusivity.
        pub unsafe trait $trait: DataTypeLike {
            /// Fills every element of `destination` with zero.
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, a signal belongs
            /// context, the length cannot be represented by the native interface, or NPP fails.
            /// Success reports native submission. Destination elements become initialized only after
            /// the queued writes complete successfully.
            ///
            /// # Safety
            ///
            /// Destination addresses and capacity must satisfy the selected native routine's alignment
            /// and representation requirements. Keep the destination exclusively writable through
            /// queued completion; it may be uninitialized before the operation.
            /// Retain all participating allocations and the stream context through queued completion,
            /// including submission errors and every captured graph replay.
            unsafe fn $method(
                stream_context: &StreamContext,
                destination: &mut SignalOutputView<'_, Self>,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    destination: &mut SignalOutputView<'_, Self>,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct(stream_context, destination)

        }
    }
            }
        )*

        /// Fills every element of `destination` with zero.
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, a signal belongs
        /// context, the length cannot be represented by the native interface, or NPP fails.
        /// Success reports native submission. Destination elements become initialized only after
        /// the queued writes complete successfully.
        ///
        /// # Safety
        ///
        /// Destination addresses and capacity must satisfy the selected native routine's alignment
        /// and representation requirements. Keep the destination exclusively writable through
        /// queued completion; it may be uninitialized before the operation.
        /// Retain all participating allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        pub unsafe fn $function<T: $trait>(
            stream_context: &StreamContext,
            destination: &mut SignalOutputView<'_, T>,
        ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            T::$method(stream_context, destination)

        }
    }
    };
}

macro_rules! impl_signal_copy_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Element-type dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must match the native element representation and access roles.
        /// Successful queued completion must initialize every destination element according
        /// to the method contract, while preserving allocation lifetimes and exclusivity.
        pub unsafe trait $trait: DataTypeLike {
            /// Copies every element of `source` to an equally sized `destination`.
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, a signal belongs
            /// context, the length cannot be represented by the native interface, or NPP fails.
            /// Source and destination lengths must match; differing lengths return an error.
            /// Success reports native submission. Destination elements become initialized only after
            /// the queued writes complete successfully.
            ///
            /// # Safety
            ///
            /// Destination addresses and capacity must satisfy the selected native routine's alignment
            /// and representation requirements. Keep the destination exclusively writable through
            /// queued completion; it may be uninitialized before the operation.
            /// Source elements must be initialized and ready when the stream reaches the copy.
            /// Keep the source read-only and disjoint from the destination through completion.
            /// Retain all participating allocations and the stream context through queued completion,
            /// including submission errors and every captured graph replay.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                destination: &mut SignalOutputView<'_, Self>,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    destination: &mut SignalOutputView<'_, Self>,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct(stream_context, source, destination)

        }
    }
            }
        )*

        /// Copies every element of `source` to an equally sized `destination`.
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, a signal belongs
        /// context, the length cannot be represented by the native interface, or NPP fails.
        /// Source and destination lengths must match; differing lengths return an error.
        /// Success reports native submission. Destination elements become initialized only after
        /// the queued writes complete successfully.
        ///
        /// # Safety
        ///
        /// Destination addresses and capacity must satisfy the selected native routine's alignment
        /// and representation requirements. Keep the destination exclusively writable through
        /// queued completion; it may be uninitialized before the operation.
        /// Source elements must be initialized and ready when the stream reaches the copy.
        /// Keep the source read-only and disjoint from the destination through completion.
        /// Retain all participating allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        pub unsafe fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            destination: &mut SignalOutputView<'_, T>,
        ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            T::$method(stream_context, source, destination)

        }
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
