use singe_cuda::types::{Complex32, Complex64};

use crate::{
    context::StreamContext,
    error::Result,
    signal::{
        raw::statistics::{
            dot_product_f32_buffer_size, dot_product_f32_complex_buffer_size,
            dot_product_f32_complex_to_device, dot_product_f32_complex_to_device_with_scratch,
            dot_product_f32_complex_to_f64_complex_buffer_size,
            dot_product_f32_complex_to_f64_complex_to_device,
            dot_product_f32_complex_to_f64_complex_to_device_with_scratch,
            dot_product_f32_to_device, dot_product_f32_to_device_with_scratch,
            dot_product_f32_to_f64_buffer_size, dot_product_f32_to_f64_to_device,
            dot_product_f32_to_f64_to_device_with_scratch, dot_product_f64_buffer_size,
            dot_product_f64_complex_buffer_size, dot_product_f64_complex_to_device,
            dot_product_f64_complex_to_device_with_scratch, dot_product_f64_to_device,
            dot_product_f64_to_device_with_scratch, dot_product_i16_complex_scaled_buffer_size,
            dot_product_i16_complex_scaled_to_device,
            dot_product_i16_complex_scaled_to_device_with_scratch,
            dot_product_i16_complex_to_f32_complex_buffer_size,
            dot_product_i16_complex_to_f32_complex_to_device,
            dot_product_i16_complex_to_f32_complex_to_device_with_scratch,
            dot_product_i16_complex_to_i32_complex_scaled_buffer_size,
            dot_product_i16_complex_to_i32_complex_scaled_to_device,
            dot_product_i16_complex_to_i32_complex_scaled_to_device_with_scratch,
            dot_product_i16_complex_to_i64_complex_buffer_size,
            dot_product_i16_complex_to_i64_complex_to_device,
            dot_product_i16_complex_to_i64_complex_to_device_with_scratch,
            dot_product_i16_scaled_buffer_size, dot_product_i16_scaled_to_device,
            dot_product_i16_scaled_to_device_with_scratch, dot_product_i16_to_f32_buffer_size,
            dot_product_i16_to_f32_to_device, dot_product_i16_to_f32_to_device_with_scratch,
            dot_product_i16_to_i32_scaled_buffer_size, dot_product_i16_to_i32_scaled_to_device,
            dot_product_i16_to_i32_scaled_to_device_with_scratch,
            dot_product_i16_to_i64_buffer_size, dot_product_i16_to_i64_to_device,
            dot_product_i16_to_i64_to_device_with_scratch,
            dot_product_i32_complex_scaled_buffer_size, dot_product_i32_complex_scaled_to_device,
            dot_product_i32_complex_scaled_to_device_with_scratch,
            dot_product_i32_scaled_buffer_size, dot_product_i32_scaled_to_device,
            dot_product_i32_scaled_to_device_with_scratch, max_absolute_i16_buffer_size,
            max_absolute_i16_to_device, max_absolute_i16_to_device_with_scratch,
            max_absolute_i32_buffer_size, max_absolute_i32_to_device,
            max_absolute_i32_to_device_with_scratch, max_f32_buffer_size, max_f32_to_device,
            max_f32_to_device_with_scratch, max_f64_buffer_size, max_f64_to_device,
            max_f64_to_device_with_scratch, max_i16_buffer_size, max_i16_to_device,
            max_i16_to_device_with_scratch, max_i32_buffer_size, max_i32_to_device,
            max_i32_to_device_with_scratch, min_absolute_i16_buffer_size,
            min_absolute_i16_to_device, min_absolute_i16_to_device_with_scratch,
            min_absolute_i32_buffer_size, min_absolute_i32_to_device,
            min_absolute_i32_to_device_with_scratch, min_f32_buffer_size, min_f32_to_device,
            min_f32_to_device_with_scratch, min_f64_buffer_size, min_f64_to_device,
            min_f64_to_device_with_scratch, min_i16_buffer_size, min_i16_to_device,
            min_i16_to_device_with_scratch, min_i32_buffer_size, min_i32_to_device,
            min_i32_to_device_with_scratch, norm_diff_inf_f32_buffer_size,
            norm_diff_inf_f32_complex_to_f32_buffer_size,
            norm_diff_inf_f32_complex_to_f32_to_device,
            norm_diff_inf_f32_complex_to_f32_to_device_with_scratch, norm_diff_inf_f32_to_device,
            norm_diff_inf_f32_to_device_with_scratch, norm_diff_inf_f64_buffer_size,
            norm_diff_inf_f64_complex_to_f64_buffer_size,
            norm_diff_inf_f64_complex_to_f64_to_device,
            norm_diff_inf_f64_complex_to_f64_to_device_with_scratch, norm_diff_inf_f64_to_device,
            norm_diff_inf_f64_to_device_with_scratch, norm_diff_inf_i16_to_f32_buffer_size,
            norm_diff_inf_i16_to_f32_to_device, norm_diff_inf_i16_to_f32_to_device_with_scratch,
            norm_diff_inf_i16_to_i32_scaled_buffer_size, norm_diff_inf_i16_to_i32_scaled_to_device,
            norm_diff_inf_i16_to_i32_scaled_to_device_with_scratch, norm_diff_l1_f32_buffer_size,
            norm_diff_l1_f32_complex_to_f64_buffer_size, norm_diff_l1_f32_complex_to_f64_to_device,
            norm_diff_l1_f32_complex_to_f64_to_device_with_scratch, norm_diff_l1_f32_to_device,
            norm_diff_l1_f32_to_device_with_scratch, norm_diff_l1_f64_buffer_size,
            norm_diff_l1_f64_complex_to_f64_buffer_size, norm_diff_l1_f64_complex_to_f64_to_device,
            norm_diff_l1_f64_complex_to_f64_to_device_with_scratch, norm_diff_l1_f64_to_device,
            norm_diff_l1_f64_to_device_with_scratch, norm_diff_l1_i16_to_f32_buffer_size,
            norm_diff_l1_i16_to_f32_to_device, norm_diff_l1_i16_to_f32_to_device_with_scratch,
            norm_diff_l1_i16_to_i32_scaled_buffer_size, norm_diff_l1_i16_to_i32_scaled_to_device,
            norm_diff_l1_i16_to_i32_scaled_to_device_with_scratch,
            norm_diff_l1_i16_to_i64_scaled_buffer_size, norm_diff_l1_i16_to_i64_scaled_to_device,
            norm_diff_l1_i16_to_i64_scaled_to_device_with_scratch, norm_diff_l2_f32_buffer_size,
            norm_diff_l2_f32_complex_to_f64_buffer_size, norm_diff_l2_f32_complex_to_f64_to_device,
            norm_diff_l2_f32_complex_to_f64_to_device_with_scratch, norm_diff_l2_f32_to_device,
            norm_diff_l2_f32_to_device_with_scratch, norm_diff_l2_f64_buffer_size,
            norm_diff_l2_f64_complex_to_f64_buffer_size, norm_diff_l2_f64_complex_to_f64_to_device,
            norm_diff_l2_f64_complex_to_f64_to_device_with_scratch, norm_diff_l2_f64_to_device,
            norm_diff_l2_f64_to_device_with_scratch, norm_diff_l2_i16_to_f32_buffer_size,
            norm_diff_l2_i16_to_f32_to_device, norm_diff_l2_i16_to_f32_to_device_with_scratch,
            norm_diff_l2_i16_to_i32_scaled_buffer_size, norm_diff_l2_i16_to_i32_scaled_to_device,
            norm_diff_l2_i16_to_i32_scaled_to_device_with_scratch,
            norm_diff_l2_squared_i16_to_i64_scaled_buffer_size,
            norm_diff_l2_squared_i16_to_i64_scaled_to_device,
            norm_diff_l2_squared_i16_to_i64_scaled_to_device_with_scratch,
        },
        view::{SignalOutputView, SignalView},
    },
    types::{ComplexI16, ComplexI32, ComplexI64, DataTypeLike},
    workspace::ScratchBuffer,
};

macro_rules! impl_signal_binary_statistic_to_dispatch {
    (
        $doc:literal,
        $trait:ident,
        $buffer_method:ident,
        $scratch_method:ident,
        $method:ident,
        $buffer_function:ident,
        $scratch_function:ident,
        $function:ident,
        [$($source_ty:ty => ($destination_ty:ty, $direct_buffer:ident, $direct_scratch:ident, $direct:ident)),* $(,)?]
    ) => {
        /// Native dispatch for the corresponding named NPP operation.
        ///
        /// # Safety
        /// Implementations must match the native ABI, access roles, written ranges,
        /// and status semantics. Successful completed execution must initialize its
        /// documented outputs; metadata queries must return the required capacity.
        pub unsafe trait $trait<Destination>: DataTypeLike
        where
            Destination: DataTypeLike,
        {
            /// Returns the scratch capacity, in bytes, for the selected statistic and source length.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails, a length or
            /// queried size cannot be represented, or NPP rejects the scratch-size query.
            fn $buffer_method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
            ) -> Result<usize>;

            #[doc = $doc]
            ///
            /// Writes the scalar result into the single element of `destination`.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails, a signal
            /// belongs to another CUDA context, `destination` does not contain exactly one element,
            /// a length or scratch-size conversion fails, or NPP reports an error.
            /// The source signals must have equal lengths.
            /// An undersized scratch buffer also returns an error.
            ///
            /// # Safety
            ///
            /// Follow the [raw signal execution contract](crate::signal::raw).
            /// Inputs must be initialized and ready when the selected stream reads them.
            /// Retain the source, destination, and stream context through queued completion
            /// and graph replay, without conflicting access to the written scalar.
            /// `scratch` must belong to the same CUDA context and remain exclusively writable,
            /// live, and disjoint from source and destination storage through completion.
            unsafe fn $scratch_method(
                stream_context: &StreamContext,
                source_1: &SignalView<'_, Self>,
                source_2: &SignalView<'_, Self>,
                destination: &mut SignalOutputView<'_, Destination>,
                scratch: &mut ScratchBuffer,
            ) -> Result<()>;

            #[doc = $doc]
            ///
            /// Writes the scalar result into the single element of `destination`.
            /// Allocates temporary scratch storage for this call.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails, a signal
            /// belongs to another CUDA context, `destination` does not contain exactly one element,
            /// a length or scratch-size conversion fails, or NPP reports an error.
            /// The source signals must have equal lengths.
            /// Scratch allocation errors are propagated.
            ///
            /// # Safety
            ///
            /// Follow the [raw signal execution contract](crate::signal::raw).
            /// Inputs must be initialized and ready when the selected stream reads them.
            /// Retain the source, destination, and stream context through queued completion
            /// and graph replay, without conflicting access to the written scalar.
            unsafe fn $method(
                stream_context: &StreamContext,
                source_1: &SignalView<'_, Self>,
                source_2: &SignalView<'_, Self>,
                destination: &mut SignalOutputView<'_, Destination>,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait<$destination_ty> for $source_ty {
                fn $buffer_method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                ) -> Result<usize> {
        stream_context.cuda_context().bind()?;
                    $direct_buffer(stream_context, source)
                }

                unsafe fn $scratch_method(
                    stream_context: &StreamContext,
                    source_1: &SignalView<'_, Self>,
                    source_2: &SignalView<'_, Self>,
                    destination: &mut SignalOutputView<'_, $destination_ty>,
                    scratch: &mut ScratchBuffer,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source_1)?;
        stream_context.validate_region(source_2)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct_scratch(stream_context, source_1, source_2, destination, scratch)

        }
    }

                unsafe fn $method(
                    stream_context: &StreamContext,
                    source_1: &SignalView<'_, Self>,
                    source_2: &SignalView<'_, Self>,
                    destination: &mut SignalOutputView<'_, $destination_ty>,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source_1)?;
        stream_context.validate_region(source_2)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct(stream_context, source_1, source_2, destination)

        }
    }
            }
        )*

        /// Returns the scratch capacity, in bytes, for the selected statistic and source length.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if CUDA context binding fails, a length or
        /// queried size cannot be represented, or NPP rejects the scratch-size query.
        pub fn $buffer_function<Source, Destination>(
            stream_context: &StreamContext,
            source: &SignalView<'_, Source>,
        ) -> Result<usize>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
        stream_context.cuda_context().bind()?;
            Source::$buffer_method(stream_context, source)
        }

        #[doc = $doc]
        ///
        /// Writes the scalar result into the single element of `destination`.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if CUDA context binding fails, a signal
        /// belongs to another CUDA context, `destination` does not contain exactly one element,
        /// a length or scratch-size conversion fails, or NPP reports an error.
        /// The source signals must have equal lengths.
        /// An undersized scratch buffer also returns an error.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Inputs must be initialized and ready when the selected stream reads them.
        /// Retain the source, destination, and stream context through queued completion
        /// and graph replay, without conflicting access to the written scalar.
        /// `scratch` must belong to the same CUDA context and remain exclusively writable,
        /// live, and disjoint from source and destination storage through completion.
        pub unsafe fn $scratch_function<Source, Destination>(
            stream_context: &StreamContext,
            source_1: &SignalView<'_, Source>,
            source_2: &SignalView<'_, Source>,
            destination: &mut SignalOutputView<'_, Destination>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source_1)?;
        stream_context.validate_region(source_2)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            Source::$scratch_method(stream_context, source_1, source_2, destination, scratch)

        }
    }

        #[doc = $doc]
        ///
        /// Writes the scalar result into the single element of `destination`.
        /// Allocates temporary scratch storage for this call.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if CUDA context binding fails, a signal
        /// belongs to another CUDA context, `destination` does not contain exactly one element,
        /// a length or scratch-size conversion fails, or NPP reports an error.
        /// The source signals must have equal lengths.
        /// Scratch allocation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Inputs must be initialized and ready when the selected stream reads them.
        /// Retain the source, destination, and stream context through queued completion
        /// and graph replay, without conflicting access to the written scalar.
        pub unsafe fn $function<Source, Destination>(
            stream_context: &StreamContext,
            source_1: &SignalView<'_, Source>,
            source_2: &SignalView<'_, Source>,
            destination: &mut SignalOutputView<'_, Destination>,
        ) -> Result<()>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source_1)?;
        stream_context.validate_region(source_2)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            Source::$method(stream_context, source_1, source_2, destination)

        }
    }
    };
}

macro_rules! impl_signal_scaled_binary_statistic_to_dispatch {
    (
        $doc:literal,
        $trait:ident,
        $buffer_method:ident,
        $scratch_method:ident,
        $method:ident,
        $buffer_function:ident,
        $scratch_function:ident,
        $function:ident,
        [$($source_ty:ty => ($destination_ty:ty, $direct_buffer:ident, $direct_scratch:ident, $direct:ident)),* $(,)?]
    ) => {
        /// Native dispatch for the corresponding named NPP operation.
        ///
        /// # Safety
        /// Implementations must match the native ABI, access roles, written ranges,
        /// and status semantics. Successful completed execution must initialize its
        /// documented outputs; metadata queries must return the required capacity.
        pub unsafe trait $trait<Destination>: DataTypeLike
        where
            Destination: DataTypeLike,
        {
            /// Returns the scratch capacity, in bytes, for the selected statistic and source length.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails, a length or
            /// queried size cannot be represented, or NPP rejects the scratch-size query.
            fn $buffer_method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
            ) -> Result<usize>;

            #[doc = $doc]
            ///
            /// Writes the scalar result into the single element of `destination`.
            /// Applies native fixed-point scaling using `scale_factor`.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails, a signal
            /// belongs to another CUDA context, `destination` does not contain exactly one element,
            /// a length or scratch-size conversion fails, or NPP reports an error.
            /// The source signals must have equal lengths.
            /// An undersized scratch buffer also returns an error.
            ///
            /// # Safety
            ///
            /// Follow the [raw signal execution contract](crate::signal::raw).
            /// Inputs must be initialized and ready when the selected stream reads them.
            /// Retain the source, destination, and stream context through queued completion
            /// and graph replay, without conflicting access to the written scalar.
            /// `scratch` must belong to the same CUDA context and remain exclusively writable,
            /// live, and disjoint from source and destination storage through completion.
            unsafe fn $scratch_method(
                stream_context: &StreamContext,
                source_1: &SignalView<'_, Self>,
                source_2: &SignalView<'_, Self>,
                destination: &mut SignalOutputView<'_, Destination>,
                scale_factor: i32,
                scratch: &mut ScratchBuffer,
            ) -> Result<()>;

            #[doc = $doc]
            ///
            /// Writes the scalar result into the single element of `destination`.
            /// Allocates temporary scratch storage for this call.
            /// Applies native fixed-point scaling using `scale_factor`.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails, a signal
            /// belongs to another CUDA context, `destination` does not contain exactly one element,
            /// a length or scratch-size conversion fails, or NPP reports an error.
            /// The source signals must have equal lengths.
            /// Scratch allocation errors are propagated.
            ///
            /// # Safety
            ///
            /// Follow the [raw signal execution contract](crate::signal::raw).
            /// Inputs must be initialized and ready when the selected stream reads them.
            /// Retain the source, destination, and stream context through queued completion
            /// and graph replay, without conflicting access to the written scalar.
            unsafe fn $method(
                stream_context: &StreamContext,
                source_1: &SignalView<'_, Self>,
                source_2: &SignalView<'_, Self>,
                destination: &mut SignalOutputView<'_, Destination>,
                scale_factor: i32,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait<$destination_ty> for $source_ty {
                fn $buffer_method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                ) -> Result<usize> {
        stream_context.cuda_context().bind()?;
                    $direct_buffer(stream_context, source)
                }

                unsafe fn $scratch_method(
                    stream_context: &StreamContext,
                    source_1: &SignalView<'_, Self>,
                    source_2: &SignalView<'_, Self>,
                    destination: &mut SignalOutputView<'_, $destination_ty>,
                    scale_factor: i32,
                    scratch: &mut ScratchBuffer,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source_1)?;
        stream_context.validate_region(source_2)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct_scratch(
                        stream_context,
                        source_1,
                        source_2,
                        destination,
                        scale_factor,
                        scratch,
                    )

        }
    }

                unsafe fn $method(
                    stream_context: &StreamContext,
                    source_1: &SignalView<'_, Self>,
                    source_2: &SignalView<'_, Self>,
                    destination: &mut SignalOutputView<'_, $destination_ty>,
                    scale_factor: i32,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source_1)?;
        stream_context.validate_region(source_2)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct(stream_context, source_1, source_2, destination, scale_factor)

        }
    }
            }
        )*

        /// Returns the scratch capacity, in bytes, for the selected statistic and source length.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if CUDA context binding fails, a length or
        /// queried size cannot be represented, or NPP rejects the scratch-size query.
        pub fn $buffer_function<Source, Destination>(
            stream_context: &StreamContext,
            source: &SignalView<'_, Source>,
        ) -> Result<usize>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
        stream_context.cuda_context().bind()?;
            Source::$buffer_method(stream_context, source)
        }

        #[doc = $doc]
        ///
        /// Writes the scalar result into the single element of `destination`.
        /// Applies native fixed-point scaling using `scale_factor`.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if CUDA context binding fails, a signal
        /// belongs to another CUDA context, `destination` does not contain exactly one element,
        /// a length or scratch-size conversion fails, or NPP reports an error.
        /// The source signals must have equal lengths.
        /// An undersized scratch buffer also returns an error.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Inputs must be initialized and ready when the selected stream reads them.
        /// Retain the source, destination, and stream context through queued completion
        /// and graph replay, without conflicting access to the written scalar.
        /// `scratch` must belong to the same CUDA context and remain exclusively writable,
        /// live, and disjoint from source and destination storage through completion.
        pub unsafe fn $scratch_function<Source, Destination>(
            stream_context: &StreamContext,
            source_1: &SignalView<'_, Source>,
            source_2: &SignalView<'_, Source>,
            destination: &mut SignalOutputView<'_, Destination>,
            scale_factor: i32,
            scratch: &mut ScratchBuffer,
        ) -> Result<()>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source_1)?;
        stream_context.validate_region(source_2)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            Source::$scratch_method(
                stream_context,
                source_1,
                source_2,
                destination,
                scale_factor,
                scratch,
            )

        }
    }

        #[doc = $doc]
        ///
        /// Writes the scalar result into the single element of `destination`.
        /// Allocates temporary scratch storage for this call.
        /// Applies native fixed-point scaling using `scale_factor`.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if CUDA context binding fails, a signal
        /// belongs to another CUDA context, `destination` does not contain exactly one element,
        /// a length or scratch-size conversion fails, or NPP reports an error.
        /// The source signals must have equal lengths.
        /// Scratch allocation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Inputs must be initialized and ready when the selected stream reads them.
        /// Retain the source, destination, and stream context through queued completion
        /// and graph replay, without conflicting access to the written scalar.
        pub unsafe fn $function<Source, Destination>(
            stream_context: &StreamContext,
            source_1: &SignalView<'_, Source>,
            source_2: &SignalView<'_, Source>,
            destination: &mut SignalOutputView<'_, Destination>,
            scale_factor: i32,
        ) -> Result<()>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source_1)?;
        stream_context.validate_region(source_2)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            Source::$method(stream_context, source_1, source_2, destination, scale_factor)

        }
    }
    };
}

impl_signal_binary_statistic_to_dispatch!("Computes the infinity norm of the difference between the two source signals.",

    NormDiffInfTo,
    norm_diff_inf_buffer_size,
    norm_diff_inf_to_device_with_scratch,
    norm_diff_inf_to_device,
    norm_diff_inf_buffer_size,
    norm_diff_inf_to_device_with_scratch,
    norm_diff_inf_to_device,
    [
        f32 => (f32, norm_diff_inf_f32_buffer_size, norm_diff_inf_f32_to_device_with_scratch, norm_diff_inf_f32_to_device),
        f64 => (f64, norm_diff_inf_f64_buffer_size, norm_diff_inf_f64_to_device_with_scratch, norm_diff_inf_f64_to_device),
        Complex32 => (f32, norm_diff_inf_f32_complex_to_f32_buffer_size, norm_diff_inf_f32_complex_to_f32_to_device_with_scratch, norm_diff_inf_f32_complex_to_f32_to_device),
        Complex64 => (f64, norm_diff_inf_f64_complex_to_f64_buffer_size, norm_diff_inf_f64_complex_to_f64_to_device_with_scratch, norm_diff_inf_f64_complex_to_f64_to_device),
        i16 => (f32, norm_diff_inf_i16_to_f32_buffer_size, norm_diff_inf_i16_to_f32_to_device_with_scratch, norm_diff_inf_i16_to_f32_to_device)
    ]
);

impl_signal_binary_statistic_to_dispatch!("Computes the L1 norm of the difference between the two source signals.",

    NormDiffL1To,
    norm_diff_l1_buffer_size,
    norm_diff_l1_to_device_with_scratch,
    norm_diff_l1_to_device,
    norm_diff_l1_buffer_size,
    norm_diff_l1_to_device_with_scratch,
    norm_diff_l1_to_device,
    [
        f32 => (f32, norm_diff_l1_f32_buffer_size, norm_diff_l1_f32_to_device_with_scratch, norm_diff_l1_f32_to_device),
        f64 => (f64, norm_diff_l1_f64_buffer_size, norm_diff_l1_f64_to_device_with_scratch, norm_diff_l1_f64_to_device),
        Complex32 => (f64, norm_diff_l1_f32_complex_to_f64_buffer_size, norm_diff_l1_f32_complex_to_f64_to_device_with_scratch, norm_diff_l1_f32_complex_to_f64_to_device),
        Complex64 => (f64, norm_diff_l1_f64_complex_to_f64_buffer_size, norm_diff_l1_f64_complex_to_f64_to_device_with_scratch, norm_diff_l1_f64_complex_to_f64_to_device),
        i16 => (f32, norm_diff_l1_i16_to_f32_buffer_size, norm_diff_l1_i16_to_f32_to_device_with_scratch, norm_diff_l1_i16_to_f32_to_device)
    ]
);

impl_signal_binary_statistic_to_dispatch!("Computes the L2 norm of the difference between the two source signals.",

    NormDiffL2To,
    norm_diff_l2_buffer_size,
    norm_diff_l2_to_device_with_scratch,
    norm_diff_l2_to_device,
    norm_diff_l2_buffer_size,
    norm_diff_l2_to_device_with_scratch,
    norm_diff_l2_to_device,
    [
        f32 => (f32, norm_diff_l2_f32_buffer_size, norm_diff_l2_f32_to_device_with_scratch, norm_diff_l2_f32_to_device),
        f64 => (f64, norm_diff_l2_f64_buffer_size, norm_diff_l2_f64_to_device_with_scratch, norm_diff_l2_f64_to_device),
        Complex32 => (f64, norm_diff_l2_f32_complex_to_f64_buffer_size, norm_diff_l2_f32_complex_to_f64_to_device_with_scratch, norm_diff_l2_f32_complex_to_f64_to_device),
        Complex64 => (f64, norm_diff_l2_f64_complex_to_f64_buffer_size, norm_diff_l2_f64_complex_to_f64_to_device_with_scratch, norm_diff_l2_f64_complex_to_f64_to_device),
        i16 => (f32, norm_diff_l2_i16_to_f32_buffer_size, norm_diff_l2_i16_to_f32_to_device_with_scratch, norm_diff_l2_i16_to_f32_to_device)
    ]
);

impl_signal_scaled_binary_statistic_to_dispatch!("Computes the infinity norm of the difference between the two source signals.",

    NormDiffInfScaledTo,
    norm_diff_inf_scaled_buffer_size,
    norm_diff_inf_scaled_to_device_with_scratch,
    norm_diff_inf_scaled_to_device,
    norm_diff_inf_scaled_buffer_size,
    norm_diff_inf_scaled_to_device_with_scratch,
    norm_diff_inf_scaled_to_device,
    [
        i16 => (i32, norm_diff_inf_i16_to_i32_scaled_buffer_size, norm_diff_inf_i16_to_i32_scaled_to_device_with_scratch, norm_diff_inf_i16_to_i32_scaled_to_device)
    ]
);

impl_signal_scaled_binary_statistic_to_dispatch!("Computes the L1 norm of the difference between the two source signals.",

    NormDiffL1ScaledTo,
    norm_diff_l1_scaled_buffer_size,
    norm_diff_l1_scaled_to_device_with_scratch,
    norm_diff_l1_scaled_to_device,
    norm_diff_l1_scaled_buffer_size,
    norm_diff_l1_scaled_to_device_with_scratch,
    norm_diff_l1_scaled_to_device,
    [
        i16 => (i32, norm_diff_l1_i16_to_i32_scaled_buffer_size, norm_diff_l1_i16_to_i32_scaled_to_device_with_scratch, norm_diff_l1_i16_to_i32_scaled_to_device),
        i16 => (i64, norm_diff_l1_i16_to_i64_scaled_buffer_size, norm_diff_l1_i16_to_i64_scaled_to_device_with_scratch, norm_diff_l1_i16_to_i64_scaled_to_device)
    ]
);

impl_signal_scaled_binary_statistic_to_dispatch!("Computes the L2 norm of the difference between the two source signals.",

    NormDiffL2ScaledTo,
    norm_diff_l2_scaled_buffer_size,
    norm_diff_l2_scaled_to_device_with_scratch,
    norm_diff_l2_scaled_to_device,
    norm_diff_l2_scaled_buffer_size,
    norm_diff_l2_scaled_to_device_with_scratch,
    norm_diff_l2_scaled_to_device,
    [
        i16 => (i32, norm_diff_l2_i16_to_i32_scaled_buffer_size, norm_diff_l2_i16_to_i32_scaled_to_device_with_scratch, norm_diff_l2_i16_to_i32_scaled_to_device)
    ]
);

impl_signal_scaled_binary_statistic_to_dispatch!("Computes the squared L2 norm of the difference between the two source signals.",

    NormDiffL2SquaredScaledTo,
    norm_diff_l2_squared_scaled_buffer_size,
    norm_diff_l2_squared_scaled_to_device_with_scratch,
    norm_diff_l2_squared_scaled_to_device,
    norm_diff_l2_squared_scaled_buffer_size,
    norm_diff_l2_squared_scaled_to_device_with_scratch,
    norm_diff_l2_squared_scaled_to_device,
    [
        i16 => (i64, norm_diff_l2_squared_i16_to_i64_scaled_buffer_size, norm_diff_l2_squared_i16_to_i64_scaled_to_device_with_scratch, norm_diff_l2_squared_i16_to_i64_scaled_to_device)
    ]
);

impl_signal_binary_statistic_to_dispatch!("Computes the dot product of the two source signals.",

    DotProductTo,
    dot_product_buffer_size,
    dot_product_to_device_with_scratch,
    dot_product_to_device,
    dot_product_buffer_size,
    dot_product_to_device_with_scratch,
    dot_product_to_device,
    [
        f32 => (f32, dot_product_f32_buffer_size, dot_product_f32_to_device_with_scratch, dot_product_f32_to_device),
        f32 => (f64, dot_product_f32_to_f64_buffer_size, dot_product_f32_to_f64_to_device_with_scratch, dot_product_f32_to_f64_to_device),
        f64 => (f64, dot_product_f64_buffer_size, dot_product_f64_to_device_with_scratch, dot_product_f64_to_device),
        Complex32 => (Complex32, dot_product_f32_complex_buffer_size, dot_product_f32_complex_to_device_with_scratch, dot_product_f32_complex_to_device),
        Complex32 => (Complex64, dot_product_f32_complex_to_f64_complex_buffer_size, dot_product_f32_complex_to_f64_complex_to_device_with_scratch, dot_product_f32_complex_to_f64_complex_to_device),
        Complex64 => (Complex64, dot_product_f64_complex_buffer_size, dot_product_f64_complex_to_device_with_scratch, dot_product_f64_complex_to_device),
        i16 => (i64, dot_product_i16_to_i64_buffer_size, dot_product_i16_to_i64_to_device_with_scratch, dot_product_i16_to_i64_to_device),
        i16 => (f32, dot_product_i16_to_f32_buffer_size, dot_product_i16_to_f32_to_device_with_scratch, dot_product_i16_to_f32_to_device),
        ComplexI16 => (ComplexI64, dot_product_i16_complex_to_i64_complex_buffer_size, dot_product_i16_complex_to_i64_complex_to_device_with_scratch, dot_product_i16_complex_to_i64_complex_to_device),
        ComplexI16 => (Complex32, dot_product_i16_complex_to_f32_complex_buffer_size, dot_product_i16_complex_to_f32_complex_to_device_with_scratch, dot_product_i16_complex_to_f32_complex_to_device)
    ]
);

impl_signal_scaled_binary_statistic_to_dispatch!("Computes the dot product of the two source signals.",

    DotProductScaledTo,
    dot_product_scaled_buffer_size,
    dot_product_scaled_to_device_with_scratch,
    dot_product_scaled_to_device,
    dot_product_scaled_buffer_size,
    dot_product_scaled_to_device_with_scratch,
    dot_product_scaled_to_device,
    [
        i16 => (i16, dot_product_i16_scaled_buffer_size, dot_product_i16_scaled_to_device_with_scratch, dot_product_i16_scaled_to_device),
        i16 => (i32, dot_product_i16_to_i32_scaled_buffer_size, dot_product_i16_to_i32_scaled_to_device_with_scratch, dot_product_i16_to_i32_scaled_to_device),
        i32 => (i32, dot_product_i32_scaled_buffer_size, dot_product_i32_scaled_to_device_with_scratch, dot_product_i32_scaled_to_device),
        ComplexI16 => (ComplexI16, dot_product_i16_complex_scaled_buffer_size, dot_product_i16_complex_scaled_to_device_with_scratch, dot_product_i16_complex_scaled_to_device),
        ComplexI16 => (ComplexI32, dot_product_i16_complex_to_i32_complex_scaled_buffer_size, dot_product_i16_complex_to_i32_complex_scaled_to_device_with_scratch, dot_product_i16_complex_to_i32_complex_scaled_to_device),
        ComplexI32 => (ComplexI32, dot_product_i32_complex_scaled_buffer_size, dot_product_i32_complex_scaled_to_device_with_scratch, dot_product_i32_complex_scaled_to_device)
    ]
);

impl_signal_statistic_to_dispatch!("Computes the maximum source value.",

    MaxTo,
    max_buffer_size,
    max_to_device_with_scratch,
    max_to_device,
    max_buffer_size,
    max_to_device_with_scratch,
    max_to_device,
    [
        i16 => (i16, max_i16_buffer_size, max_i16_to_device_with_scratch, max_i16_to_device),
        i32 => (i32, max_i32_buffer_size, max_i32_to_device_with_scratch, max_i32_to_device),
        f32 => (f32, max_f32_buffer_size, max_f32_to_device_with_scratch, max_f32_to_device),
        f64 => (f64, max_f64_buffer_size, max_f64_to_device_with_scratch, max_f64_to_device)
    ]
);

impl_signal_statistic_to_dispatch!("Computes the minimum source value.",

    MinTo,
    min_buffer_size,
    min_to_device_with_scratch,
    min_to_device,
    min_buffer_size,
    min_to_device_with_scratch,
    min_to_device,
    [
        i16 => (i16, min_i16_buffer_size, min_i16_to_device_with_scratch, min_i16_to_device),
        i32 => (i32, min_i32_buffer_size, min_i32_to_device_with_scratch, min_i32_to_device),
        f32 => (f32, min_f32_buffer_size, min_f32_to_device_with_scratch, min_f32_to_device),
        f64 => (f64, min_f64_buffer_size, min_f64_to_device_with_scratch, min_f64_to_device)
    ]
);

impl_signal_statistic_to_dispatch!("Computes the maximum absolute source value.",

    MaxAbsoluteTo,
    max_absolute_buffer_size,
    max_absolute_to_device_with_scratch,
    max_absolute_to_device,
    max_absolute_buffer_size,
    max_absolute_to_device_with_scratch,
    max_absolute_to_device,
    [
        i16 => (i16, max_absolute_i16_buffer_size, max_absolute_i16_to_device_with_scratch, max_absolute_i16_to_device),
        i32 => (i32, max_absolute_i32_buffer_size, max_absolute_i32_to_device_with_scratch, max_absolute_i32_to_device)
    ]
);

impl_signal_statistic_to_dispatch!("Computes the minimum absolute source value.",

    MinAbsoluteTo,
    min_absolute_buffer_size,
    min_absolute_to_device_with_scratch,
    min_absolute_to_device,
    min_absolute_buffer_size,
    min_absolute_to_device_with_scratch,
    min_absolute_to_device,
    [
        i16 => (i16, min_absolute_i16_buffer_size, min_absolute_i16_to_device_with_scratch, min_absolute_i16_to_device),
        i32 => (i32, min_absolute_i32_buffer_size, min_absolute_i32_to_device_with_scratch, min_absolute_i32_to_device)
    ]
);
