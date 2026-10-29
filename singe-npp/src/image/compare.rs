//! Raw image operations queued on the selected NPP stream.
//!
//! # Safety
//! Retain every allocation until completion or the last captured replay and establish access ordering.
//! Input and read-write views require initialized pixels; outputs describe writable storage only.
//! Comparison masks write every pixel in the ROI.
//! AC4 comparisons ignore the input alpha channel.
//! Initialized results require successful completion.

use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::view::{AC4, C1, C3, C4, ImageView, MaskOutputView},
    try_ffi,
    types::{ComparisonOperation, DataTypeLike, Size},
};

macro_rules! impl_generic_compare {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native image-element and channel-layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible comparison routine.
        /// They must uphold the method's mask representation, size checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Compares source pixels with corresponding pixels in the second source using `operation`.
            /// Each mask pixel is `255` if the condition holds for every processed channel, and `0` otherwise.
            /// AC4 comparisons exclude input alpha; the destination remains a single-channel mask.
            ///
            /// Enqueues work on the selected stream.
            /// Every destination pixel in the ROI is written; row padding remains unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, any image belongs to another CUDA context, image sizes differ, or NPP rejects the comparison parameters.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep sources read-only and destination storage exclusively writable and disjoint from all sources during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                source1: &ImageView<'_, Self, Layout>,
                source2: &ImageView<'_, Self, Layout>,
                destination: &mut MaskOutputView<'_>,
                operation: ComparisonOperation,
            ) -> Result<()>;
        }

        /// Compares source pixels with corresponding pixels in the second source using `operation`.
        /// Each mask pixel is `255` if the condition holds for every processed channel, and `0` otherwise.
        /// AC4 comparisons exclude input alpha; the destination remains a single-channel mask.
        ///
        /// Enqueues work on the selected stream.
        /// Every destination pixel in the ROI is written; row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image belongs to another CUDA context, image sizes differ, or NPP rejects the comparison parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep sources read-only and destination storage exclusively writable and disjoint from all sources during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source1: &ImageView<'_, T, $layout>,
            source2: &ImageView<'_, T, $layout>,
            destination: &mut MaskOutputView<'_>,
            operation: ComparisonOperation,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source1, source2, destination, operation)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source1: &ImageView<'_, Self, $layout>,
                    source2: &ImageView<'_, Self, $layout>,
                    destination: &mut MaskOutputView<'_>,
                    operation: ComparisonOperation,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source1, source2, destination, operation)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_compare_constant_scalar {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native image-element and channel-layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible comparison routine.
        /// They must uphold the method's mask representation, size checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Compares source pixels with the supplied constant using `operation`.
            /// Each mask pixel is `255` if the condition holds for every processed channel, and `0` otherwise.
            /// AC4 comparisons exclude input alpha; the destination remains a single-channel mask.
            ///
            /// Enqueues work on the selected stream.
            /// Every destination pixel in the ROI is written; row padding remains unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, any image belongs to another CUDA context, image sizes differ, or NPP rejects the comparison parameters.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep sources read-only and destination storage exclusively writable and disjoint from all sources during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                constant: Self,
                destination: &mut MaskOutputView<'_>,
                operation: ComparisonOperation,
            ) -> Result<()>;
        }

        /// Compares source pixels with the supplied constant using `operation`.
        /// Each mask pixel is `255` if the condition holds for every processed channel, and `0` otherwise.
        /// AC4 comparisons exclude input alpha; the destination remains a single-channel mask.
        ///
        /// Enqueues work on the selected stream.
        /// Every destination pixel in the ROI is written; row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image belongs to another CUDA context, image sizes differ, or NPP rejects the comparison parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep sources read-only and destination storage exclusively writable and disjoint from all sources during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            constant: T,
            destination: &mut MaskOutputView<'_>,
            operation: ComparisonOperation,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, constant, destination, operation)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    constant: Self,
                    destination: &mut MaskOutputView<'_>,
                    operation: ComparisonOperation,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, constant, destination, operation)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_compare_constant_array {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, $channels:literal, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native image-element and channel-layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible comparison routine.
        /// They must uphold the method's mask representation, size checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Compares source pixels with the supplied constant values using `operation`.
            /// Uses one constant per processed source channel.
            /// Each mask pixel is `255` if the condition holds for every processed channel, and `0` otherwise.
            /// AC4 comparisons exclude input alpha; the destination remains a single-channel mask.
            ///
            /// Enqueues work on the selected stream.
            /// Every destination pixel in the ROI is written; row padding remains unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, any image belongs to another CUDA context, image sizes differ, or NPP rejects the comparison parameters.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep sources read-only and destination storage exclusively writable and disjoint from all sources during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                constants: [Self; $channels],
                destination: &mut MaskOutputView<'_>,
                operation: ComparisonOperation,
            ) -> Result<()>;
        }

        /// Compares source pixels with the supplied constant values using `operation`.
        /// Uses one constant per processed source channel.
        /// Each mask pixel is `255` if the condition holds for every processed channel, and `0` otherwise.
        /// AC4 comparisons exclude input alpha; the destination remains a single-channel mask.
        ///
        /// Enqueues work on the selected stream.
        /// Every destination pixel in the ROI is written; row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image belongs to another CUDA context, image sizes differ, or NPP rejects the comparison parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep sources read-only and destination storage exclusively writable and disjoint from all sources during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            constants: [T; $channels],
            destination: &mut MaskOutputView<'_>,
            operation: ComparisonOperation,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, constants, destination, operation)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    constants: [Self; $channels],
                    destination: &mut MaskOutputView<'_>,
                    operation: ComparisonOperation,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, constants, destination, operation)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_compare_equal_epsilon {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native image-element and channel-layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible comparison routine.
        /// They must uphold the method's mask representation, size checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Compares source pixels with corresponding pixels in the second source for equality within `epsilon`.
            /// Each mask pixel is `255` if the condition holds for every processed channel, and `0` otherwise.
            /// AC4 comparisons exclude input alpha; the destination remains a single-channel mask.
            /// `epsilon` is the tolerance applied to absolute pixel-value differences; the wrapper does not validate it.
            ///
            /// Enqueues work on the selected stream.
            /// Every destination pixel in the ROI is written; row padding remains unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, any image belongs to another CUDA context, image sizes differ, or NPP rejects the comparison parameters.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep sources read-only and destination storage exclusively writable and disjoint from all sources during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                source1: &ImageView<'_, Self, Layout>,
                source2: &ImageView<'_, Self, Layout>,
                destination: &mut MaskOutputView<'_>,
                epsilon: Self,
            ) -> Result<()>;
        }

        /// Compares source pixels with corresponding pixels in the second source for equality within `epsilon`.
        /// Each mask pixel is `255` if the condition holds for every processed channel, and `0` otherwise.
        /// AC4 comparisons exclude input alpha; the destination remains a single-channel mask.
        /// `epsilon` is the tolerance applied to absolute pixel-value differences; the wrapper does not validate it.
        ///
        /// Enqueues work on the selected stream.
        /// Every destination pixel in the ROI is written; row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image belongs to another CUDA context, image sizes differ, or NPP rejects the comparison parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep sources read-only and destination storage exclusively writable and disjoint from all sources during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source1: &ImageView<'_, T, $layout>,
            source2: &ImageView<'_, T, $layout>,
            destination: &mut MaskOutputView<'_>,
            epsilon: T,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source1, source2, destination, epsilon)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source1: &ImageView<'_, Self, $layout>,
                    source2: &ImageView<'_, Self, $layout>,
                    destination: &mut MaskOutputView<'_>,
                    epsilon: Self,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source1, source2, destination, epsilon)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_compare_equal_epsilon_constant_scalar {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native image-element and channel-layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible comparison routine.
        /// They must uphold the method's mask representation, size checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Compares source pixels with the supplied constant for equality within `epsilon`.
            /// Each mask pixel is `255` if the condition holds for every processed channel, and `0` otherwise.
            /// AC4 comparisons exclude input alpha; the destination remains a single-channel mask.
            /// `epsilon` is the tolerance applied to absolute pixel-value differences; the wrapper does not validate it.
            ///
            /// Enqueues work on the selected stream.
            /// Every destination pixel in the ROI is written; row padding remains unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, any image belongs to another CUDA context, image sizes differ, or NPP rejects the comparison parameters.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep sources read-only and destination storage exclusively writable and disjoint from all sources during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                constant: Self,
                destination: &mut MaskOutputView<'_>,
                epsilon: Self,
            ) -> Result<()>;
        }

        /// Compares source pixels with the supplied constant for equality within `epsilon`.
        /// Each mask pixel is `255` if the condition holds for every processed channel, and `0` otherwise.
        /// AC4 comparisons exclude input alpha; the destination remains a single-channel mask.
        /// `epsilon` is the tolerance applied to absolute pixel-value differences; the wrapper does not validate it.
        ///
        /// Enqueues work on the selected stream.
        /// Every destination pixel in the ROI is written; row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image belongs to another CUDA context, image sizes differ, or NPP rejects the comparison parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep sources read-only and destination storage exclusively writable and disjoint from all sources during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            constant: T,
            destination: &mut MaskOutputView<'_>,
            epsilon: T,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, constant, destination, epsilon)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    constant: Self,
                    destination: &mut MaskOutputView<'_>,
                    epsilon: Self,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, constant, destination, epsilon)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_compare_equal_epsilon_constant_array {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, $channels:literal, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native image-element and channel-layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible comparison routine.
        /// They must uphold the method's mask representation, size checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Compares source pixels with the supplied constant values for equality within `epsilon`.
            /// Uses one constant per processed source channel.
            /// Each mask pixel is `255` if the condition holds for every processed channel, and `0` otherwise.
            /// AC4 comparisons exclude input alpha; the destination remains a single-channel mask.
            /// `epsilon` is the tolerance applied to absolute pixel-value differences; the wrapper does not validate it.
            ///
            /// Enqueues work on the selected stream.
            /// Every destination pixel in the ROI is written; row padding remains unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, any image belongs to another CUDA context, image sizes differ, or NPP rejects the comparison parameters.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep sources read-only and destination storage exclusively writable and disjoint from all sources during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                constants: [Self; $channels],
                destination: &mut MaskOutputView<'_>,
                epsilon: Self,
            ) -> Result<()>;
        }

        /// Compares source pixels with the supplied constant values for equality within `epsilon`.
        /// Uses one constant per processed source channel.
        /// Each mask pixel is `255` if the condition holds for every processed channel, and `0` otherwise.
        /// AC4 comparisons exclude input alpha; the destination remains a single-channel mask.
        /// `epsilon` is the tolerance applied to absolute pixel-value differences; the wrapper does not validate it.
        ///
        /// Enqueues work on the selected stream.
        /// Every destination pixel in the ROI is written; row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image belongs to another CUDA context, image sizes differ, or NPP rejects the comparison parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep sources read-only and destination storage exclusively writable and disjoint from all sources during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            constants: [T; $channels],
            destination: &mut MaskOutputView<'_>,
            epsilon: T,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, constants, destination, epsilon)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    constants: [Self; $channels],
                    destination: &mut MaskOutputView<'_>,
                    epsilon: Self,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, constants, destination, epsilon)
                    }
}
            }
        )*
    };
}

macro_rules! impl_compare_c1 {
    ($name:ident, $ty:ty, $ffi:ident) => {
        /// Compares source pixels with corresponding pixels in the second source using `operation`.
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Each mask pixel is `255` if the condition holds for every processed channel, and `0` otherwise.
        /// AC4 comparisons exclude input alpha; the destination remains a single-channel mask.
        ///
        /// Enqueues work on the selected stream.
        /// Every destination pixel in the ROI is written; row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image belongs to another CUDA context, image sizes differ, or NPP rejects the comparison parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep sources read-only and destination storage exclusively writable and disjoint from all sources during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source1: &ImageView<'_, $ty, C1>,
            source2: &ImageView<'_, $ty, C1>,
            destination: &mut MaskOutputView<'_>,
            operation: ComparisonOperation,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source1)?;
            stream_context.validate_region(source2)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source1.size(), source2.size())?;
            validate_same_size(source1.size(), destination.size())?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source1.as_ptr().cast(),
                    source1.step(),
                    source2.as_ptr().cast(),
                    source2.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source1.size().into(),
                    operation.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

impl_compare_c1!(compare_u8_c1, u8, nppiCompare_8u_C1R_Ctx);

impl_compare_c1!(compare_u16_c1, u16, nppiCompare_16u_C1R_Ctx);

impl_compare_c1!(compare_i16_c1, i16, nppiCompare_16s_C1R_Ctx);

impl_compare_c1!(compare_f32_c1, f32, nppiCompare_32f_C1R_Ctx);

macro_rules! impl_compare_packed {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        /// Compares source pixels with corresponding pixels in the second source using `operation`.
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Each mask pixel is `255` if the condition holds for every processed channel, and `0` otherwise.
        /// AC4 comparisons exclude input alpha; the destination remains a single-channel mask.
        ///
        /// Enqueues work on the selected stream.
        /// Every destination pixel in the ROI is written; row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image belongs to another CUDA context, image sizes differ, or NPP rejects the comparison parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep sources read-only and destination storage exclusively writable and disjoint from all sources during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source1: &ImageView<'_, $ty, $layout>,
            source2: &ImageView<'_, $ty, $layout>,
            destination: &mut MaskOutputView<'_>,
            operation: ComparisonOperation,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source1)?;
            stream_context.validate_region(source2)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source1.size(), source2.size())?;
            validate_same_size(source1.size(), destination.size())?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source1.as_ptr().cast(),
                    source1.step(),
                    source2.as_ptr().cast(),
                    source2.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source1.size().into(),
                    operation.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

impl_compare_packed!(compare_u8_c3, u8, C3, nppiCompare_8u_C3R_Ctx);

impl_compare_packed!(compare_u8_c4, u8, C4, nppiCompare_8u_C4R_Ctx);

impl_compare_packed!(compare_u8_ac4, u8, AC4, nppiCompare_8u_AC4R_Ctx);

impl_compare_packed!(compare_u16_c3, u16, C3, nppiCompare_16u_C3R_Ctx);

impl_compare_packed!(compare_u16_c4, u16, C4, nppiCompare_16u_C4R_Ctx);

impl_compare_packed!(compare_u16_ac4, u16, AC4, nppiCompare_16u_AC4R_Ctx);

impl_compare_packed!(compare_i16_c3, i16, C3, nppiCompare_16s_C3R_Ctx);

impl_compare_packed!(compare_i16_c4, i16, C4, nppiCompare_16s_C4R_Ctx);

impl_compare_packed!(compare_i16_ac4, i16, AC4, nppiCompare_16s_AC4R_Ctx);

impl_compare_packed!(compare_f32_c3, f32, C3, nppiCompare_32f_C3R_Ctx);

impl_compare_packed!(compare_f32_c4, f32, C4, nppiCompare_32f_C4R_Ctx);

impl_compare_packed!(compare_f32_ac4, f32, AC4, nppiCompare_32f_AC4R_Ctx);

impl_generic_compare!(CompareC1, compare, compare_c1, C1, [
    u8 => compare_u8_c1,
    u16 => compare_u16_c1,
    i16 => compare_i16_c1,
    f32 => compare_f32_c1,
]);

impl_generic_compare!(CompareC3, compare, compare_c3, C3, [
    u8 => compare_u8_c3,
    u16 => compare_u16_c3,
    i16 => compare_i16_c3,
    f32 => compare_f32_c3,
]);

impl_generic_compare!(CompareC4, compare, compare_c4, C4, [
    u8 => compare_u8_c4,
    u16 => compare_u16_c4,
    i16 => compare_i16_c4,
    f32 => compare_f32_c4,
]);

impl_generic_compare!(CompareAc4, compare, compare_ac4, AC4, [
    u8 => compare_u8_ac4,
    u16 => compare_u16_ac4,
    i16 => compare_i16_ac4,
    f32 => compare_f32_ac4,
]);

macro_rules! impl_compare_constant_c1 {
    ($name:ident, $ty:ty, $ffi:ident) => {
        /// Compares source pixels with the supplied constant using `operation`.
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Each mask pixel is `255` if the condition holds for every processed channel, and `0` otherwise.
        /// AC4 comparisons exclude input alpha; the destination remains a single-channel mask.
        ///
        /// Enqueues work on the selected stream.
        /// Every destination pixel in the ROI is written; row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image belongs to another CUDA context, image sizes differ, or NPP rejects the comparison parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep sources read-only and destination storage exclusively writable and disjoint from all sources during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            constant: $ty,
            destination: &mut MaskOutputView<'_>,
            operation: ComparisonOperation,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    constant,
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    operation.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

impl_compare_constant_c1!(compare_constant_u8_c1, u8, nppiCompareC_8u_C1R_Ctx);

impl_compare_constant_c1!(compare_constant_u16_c1, u16, nppiCompareC_16u_C1R_Ctx);

impl_compare_constant_c1!(compare_constant_i16_c1, i16, nppiCompareC_16s_C1R_Ctx);

impl_compare_constant_c1!(compare_constant_f32_c1, f32, nppiCompareC_32f_C1R_Ctx);

macro_rules! impl_compare_constant_packed {
    ($name:ident, $ty:ty, $layout:ty, $channels:expr, $ffi:ident) => {
        /// Compares source pixels with the supplied constant values using `operation`.
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Uses one constant per processed source channel.
        /// Each mask pixel is `255` if the condition holds for every processed channel, and `0` otherwise.
        /// AC4 comparisons exclude input alpha; the destination remains a single-channel mask.
        ///
        /// Enqueues work on the selected stream.
        /// Every destination pixel in the ROI is written; row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image belongs to another CUDA context, image sizes differ, or NPP rejects the comparison parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep sources read-only and destination storage exclusively writable and disjoint from all sources during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            constants: [$ty; $channels],
            destination: &mut MaskOutputView<'_>,
            operation: ComparisonOperation,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;

            // SAFETY: the `ffi` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    constants.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    operation.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

impl_compare_constant_packed!(compare_constant_u8_c3, u8, C3, 3, nppiCompareC_8u_C3R_Ctx);

impl_compare_constant_packed!(compare_constant_u8_c4, u8, C4, 4, nppiCompareC_8u_C4R_Ctx);

impl_compare_constant_packed!(
    compare_constant_u8_ac4,
    u8,
    AC4,
    3,
    nppiCompareC_8u_AC4R_Ctx
);

impl_compare_constant_packed!(
    compare_constant_u16_c3,
    u16,
    C3,
    3,
    nppiCompareC_16u_C3R_Ctx
);

impl_compare_constant_packed!(
    compare_constant_u16_c4,
    u16,
    C4,
    4,
    nppiCompareC_16u_C4R_Ctx
);

impl_compare_constant_packed!(
    compare_constant_u16_ac4,
    u16,
    AC4,
    3,
    nppiCompareC_16u_AC4R_Ctx
);

impl_compare_constant_packed!(
    compare_constant_i16_c3,
    i16,
    C3,
    3,
    nppiCompareC_16s_C3R_Ctx
);

impl_compare_constant_packed!(
    compare_constant_i16_c4,
    i16,
    C4,
    4,
    nppiCompareC_16s_C4R_Ctx
);

impl_compare_constant_packed!(
    compare_constant_i16_ac4,
    i16,
    AC4,
    3,
    nppiCompareC_16s_AC4R_Ctx
);

impl_compare_constant_packed!(
    compare_constant_f32_c3,
    f32,
    C3,
    3,
    nppiCompareC_32f_C3R_Ctx
);

impl_compare_constant_packed!(
    compare_constant_f32_c4,
    f32,
    C4,
    4,
    nppiCompareC_32f_C4R_Ctx
);

impl_compare_constant_packed!(
    compare_constant_f32_ac4,
    f32,
    AC4,
    3,
    nppiCompareC_32f_AC4R_Ctx
);

impl_generic_compare_constant_scalar!(
    CompareConstantC1,
    compare_constant,
    compare_constant_c1,
    C1,
    [
        u8 => compare_constant_u8_c1,
        u16 => compare_constant_u16_c1,
        i16 => compare_constant_i16_c1,
        f32 => compare_constant_f32_c1,
    ]
);

impl_generic_compare_constant_array!(
    CompareConstantC3,
    compare_constant,
    compare_constant_c3,
    C3,
    3,
    [
        u8 => compare_constant_u8_c3,
        u16 => compare_constant_u16_c3,
        i16 => compare_constant_i16_c3,
        f32 => compare_constant_f32_c3,
    ]
);

impl_generic_compare_constant_array!(
    CompareConstantC4,
    compare_constant,
    compare_constant_c4,
    C4,
    4,
    [
        u8 => compare_constant_u8_c4,
        u16 => compare_constant_u16_c4,
        i16 => compare_constant_i16_c4,
        f32 => compare_constant_f32_c4,
    ]
);

impl_generic_compare_constant_array!(
    CompareConstantAc4,
    compare_constant,
    compare_constant_ac4,
    AC4,
    3,
    [
        u8 => compare_constant_u8_ac4,
        u16 => compare_constant_u16_ac4,
        i16 => compare_constant_i16_ac4,
        f32 => compare_constant_f32_ac4,
    ]
);

macro_rules! impl_compare_equal_epsilon {
    ($name:ident, $layout:ty, $ffi:ident) => {
        /// Compares source pixels with corresponding pixels in the second source for equality within `epsilon`.
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Each mask pixel is `255` if the condition holds for every processed channel, and `0` otherwise.
        /// AC4 comparisons exclude input alpha; the destination remains a single-channel mask.
        /// `epsilon` is the tolerance applied to absolute pixel-value differences; the wrapper does not validate it.
        ///
        /// Enqueues work on the selected stream.
        /// Every destination pixel in the ROI is written; row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image belongs to another CUDA context, image sizes differ, or NPP rejects the comparison parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep sources read-only and destination storage exclusively writable and disjoint from all sources during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source1: &ImageView<'_, f32, $layout>,
            source2: &ImageView<'_, f32, $layout>,
            destination: &mut MaskOutputView<'_>,
            epsilon: f32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source1)?;
            stream_context.validate_region(source2)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source1.size(), source2.size())?;
            validate_same_size(source1.size(), destination.size())?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source1.as_ptr().cast(),
                    source1.step(),
                    source2.as_ptr().cast(),
                    source2.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source1.size().into(),
                    epsilon,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

impl_compare_equal_epsilon!(
    compare_equal_epsilon_f32_c1,
    C1,
    nppiCompareEqualEps_32f_C1R_Ctx
);

impl_compare_equal_epsilon!(
    compare_equal_epsilon_f32_c3,
    C3,
    nppiCompareEqualEps_32f_C3R_Ctx
);

impl_compare_equal_epsilon!(
    compare_equal_epsilon_f32_c4,
    C4,
    nppiCompareEqualEps_32f_C4R_Ctx
);

impl_compare_equal_epsilon!(
    compare_equal_epsilon_f32_ac4,
    AC4,
    nppiCompareEqualEps_32f_AC4R_Ctx
);

impl_generic_compare_equal_epsilon!(
    CompareEqualEpsilonC1,
    compare_equal_epsilon,
    compare_equal_epsilon_c1,
    C1,
    [f32 => compare_equal_epsilon_f32_c1]
);

impl_generic_compare_equal_epsilon!(
    CompareEqualEpsilonC3,
    compare_equal_epsilon,
    compare_equal_epsilon_c3,
    C3,
    [f32 => compare_equal_epsilon_f32_c3]
);

impl_generic_compare_equal_epsilon!(
    CompareEqualEpsilonC4,
    compare_equal_epsilon,
    compare_equal_epsilon_c4,
    C4,
    [f32 => compare_equal_epsilon_f32_c4]
);

impl_generic_compare_equal_epsilon!(
    CompareEqualEpsilonAc4,
    compare_equal_epsilon,
    compare_equal_epsilon_ac4,
    AC4,
    [f32 => compare_equal_epsilon_f32_ac4]
);

macro_rules! impl_compare_equal_epsilon_constant_c1 {
    ($name:ident, $ffi:ident) => {
        /// Compares source pixels with the supplied constant for equality within `epsilon`.
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Each mask pixel is `255` if the condition holds for every processed channel, and `0` otherwise.
        /// AC4 comparisons exclude input alpha; the destination remains a single-channel mask.
        /// `epsilon` is the tolerance applied to absolute pixel-value differences; the wrapper does not validate it.
        ///
        /// Enqueues work on the selected stream.
        /// Every destination pixel in the ROI is written; row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image belongs to another CUDA context, image sizes differ, or NPP rejects the comparison parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep sources read-only and destination storage exclusively writable and disjoint from all sources during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, f32, C1>,
            constant: f32,
            destination: &mut MaskOutputView<'_>,
            epsilon: f32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    constant,
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    epsilon,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_compare_equal_epsilon_constant_packed {
    ($name:ident, $layout:ty, $channels:expr, $ffi:ident) => {
        /// Compares source pixels with the supplied constant values for equality within `epsilon`.
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Uses one constant per processed source channel.
        /// Each mask pixel is `255` if the condition holds for every processed channel, and `0` otherwise.
        /// AC4 comparisons exclude input alpha; the destination remains a single-channel mask.
        /// `epsilon` is the tolerance applied to absolute pixel-value differences; the wrapper does not validate it.
        ///
        /// Enqueues work on the selected stream.
        /// Every destination pixel in the ROI is written; row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image belongs to another CUDA context, image sizes differ, or NPP rejects the comparison parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep sources read-only and destination storage exclusively writable and disjoint from all sources during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, f32, $layout>,
            constants: [f32; $channels],
            destination: &mut MaskOutputView<'_>,
            epsilon: f32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;

            // SAFETY: the `ffi` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    constants.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    epsilon,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

impl_compare_equal_epsilon_constant_c1!(
    compare_equal_epsilon_constant_f32_c1,
    nppiCompareEqualEpsC_32f_C1R_Ctx
);

impl_compare_equal_epsilon_constant_packed!(
    compare_equal_epsilon_constant_f32_c3,
    C3,
    3,
    nppiCompareEqualEpsC_32f_C3R_Ctx
);

impl_compare_equal_epsilon_constant_packed!(
    compare_equal_epsilon_constant_f32_c4,
    C4,
    4,
    nppiCompareEqualEpsC_32f_C4R_Ctx
);

impl_compare_equal_epsilon_constant_packed!(
    compare_equal_epsilon_constant_f32_ac4,
    AC4,
    3,
    nppiCompareEqualEpsC_32f_AC4R_Ctx
);

impl_generic_compare_equal_epsilon_constant_scalar!(
    CompareEqualEpsilonConstantC1,
    compare_equal_epsilon_constant,
    compare_equal_epsilon_constant_c1,
    C1,
    [f32 => compare_equal_epsilon_constant_f32_c1]
);

impl_generic_compare_equal_epsilon_constant_array!(
    CompareEqualEpsilonConstantC3,
    compare_equal_epsilon_constant,
    compare_equal_epsilon_constant_c3,
    C3,
    3,
    [f32 => compare_equal_epsilon_constant_f32_c3]
);

impl_generic_compare_equal_epsilon_constant_array!(
    CompareEqualEpsilonConstantC4,
    compare_equal_epsilon_constant,
    compare_equal_epsilon_constant_c4,
    C4,
    4,
    [f32 => compare_equal_epsilon_constant_f32_c4]
);

impl_generic_compare_equal_epsilon_constant_array!(
    CompareEqualEpsilonConstantAc4,
    compare_equal_epsilon_constant,
    compare_equal_epsilon_constant_ac4,
    AC4,
    3,
    [f32 => compare_equal_epsilon_constant_f32_ac4]
);

fn validate_same_size(source: Size, destination: Size) -> Result<()> {
    if source == destination {
        return Ok(());
    }
    Err(Error::SizeMismatch {
        name: "image size".into(),
        expected: source,
        actual: destination,
    })
}
