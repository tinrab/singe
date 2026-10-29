use singe_cuda::types::{Complex32, Complex64};

use crate::{
    context::StreamContext,
    error::Result,
    signal::{
        raw::statistics::{
            mean_f32_buffer_size, mean_f32_complex_buffer_size, mean_f32_complex_to_device,
            mean_f32_complex_to_device_with_scratch, mean_f32_to_device,
            mean_f32_to_device_with_scratch, mean_f64_buffer_size, mean_f64_complex_buffer_size,
            mean_f64_complex_to_device, mean_f64_complex_to_device_with_scratch,
            mean_f64_to_device, mean_f64_to_device_with_scratch,
            mean_i16_complex_scaled_buffer_size, mean_i16_complex_scaled_to_device,
            mean_i16_complex_scaled_to_device_with_scratch, mean_i16_scaled_buffer_size,
            mean_i16_scaled_to_device, mean_i16_scaled_to_device_with_scratch,
            mean_i32_scaled_buffer_size, mean_i32_scaled_to_device,
            mean_i32_scaled_to_device_with_scratch, norm_inf_f32_buffer_size,
            norm_inf_f32_complex_to_f32_buffer_size, norm_inf_f32_complex_to_f32_to_device,
            norm_inf_f32_complex_to_f32_to_device_with_scratch, norm_inf_f32_to_device,
            norm_inf_f32_to_device_with_scratch, norm_inf_f64_buffer_size,
            norm_inf_f64_complex_to_f64_buffer_size, norm_inf_f64_complex_to_f64_to_device,
            norm_inf_f64_complex_to_f64_to_device_with_scratch, norm_inf_f64_to_device,
            norm_inf_f64_to_device_with_scratch, norm_inf_i16_to_f32_buffer_size,
            norm_inf_i16_to_f32_to_device, norm_inf_i16_to_f32_to_device_with_scratch,
            norm_inf_i16_to_i32_scaled_buffer_size, norm_inf_i16_to_i32_scaled_to_device,
            norm_inf_i16_to_i32_scaled_to_device_with_scratch, norm_l1_f32_buffer_size,
            norm_l1_f32_complex_to_f64_buffer_size, norm_l1_f32_complex_to_f64_to_device,
            norm_l1_f32_complex_to_f64_to_device_with_scratch, norm_l1_f32_to_device,
            norm_l1_f32_to_device_with_scratch, norm_l1_f64_buffer_size,
            norm_l1_f64_complex_to_f64_buffer_size, norm_l1_f64_complex_to_f64_to_device,
            norm_l1_f64_complex_to_f64_to_device_with_scratch, norm_l1_f64_to_device,
            norm_l1_f64_to_device_with_scratch, norm_l1_i16_to_f32_buffer_size,
            norm_l1_i16_to_f32_to_device, norm_l1_i16_to_f32_to_device_with_scratch,
            norm_l1_i16_to_i32_scaled_buffer_size, norm_l1_i16_to_i32_scaled_to_device,
            norm_l1_i16_to_i32_scaled_to_device_with_scratch,
            norm_l1_i16_to_i64_scaled_buffer_size, norm_l1_i16_to_i64_scaled_to_device,
            norm_l1_i16_to_i64_scaled_to_device_with_scratch, norm_l2_f32_buffer_size,
            norm_l2_f32_complex_to_f64_buffer_size, norm_l2_f32_complex_to_f64_to_device,
            norm_l2_f32_complex_to_f64_to_device_with_scratch, norm_l2_f32_to_device,
            norm_l2_f32_to_device_with_scratch, norm_l2_f64_buffer_size,
            norm_l2_f64_complex_to_f64_buffer_size, norm_l2_f64_complex_to_f64_to_device,
            norm_l2_f64_complex_to_f64_to_device_with_scratch, norm_l2_f64_to_device,
            norm_l2_f64_to_device_with_scratch, norm_l2_i16_to_f32_buffer_size,
            norm_l2_i16_to_f32_to_device, norm_l2_i16_to_f32_to_device_with_scratch,
            norm_l2_i16_to_i32_scaled_buffer_size, norm_l2_i16_to_i32_scaled_to_device,
            norm_l2_i16_to_i32_scaled_to_device_with_scratch,
            norm_l2_squared_i16_to_i64_scaled_buffer_size,
            norm_l2_squared_i16_to_i64_scaled_to_device,
            norm_l2_squared_i16_to_i64_scaled_to_device_with_scratch,
            standard_deviation_f32_buffer_size, standard_deviation_f32_to_device,
            standard_deviation_f32_to_device_with_scratch, standard_deviation_f64_buffer_size,
            standard_deviation_f64_to_device, standard_deviation_f64_to_device_with_scratch,
            standard_deviation_i16_scaled_buffer_size, standard_deviation_i16_scaled_to_device,
            standard_deviation_i16_scaled_to_device_with_scratch,
            standard_deviation_i16_to_i32_scaled_buffer_size,
            standard_deviation_i16_to_i32_scaled_to_device,
            standard_deviation_i16_to_i32_scaled_to_device_with_scratch, sum_f32_buffer_size,
            sum_f32_complex_buffer_size, sum_f32_complex_to_device,
            sum_f32_complex_to_device_with_scratch, sum_f32_to_device,
            sum_f32_to_device_with_scratch, sum_f64_buffer_size, sum_f64_complex_buffer_size,
            sum_f64_complex_to_device, sum_f64_complex_to_device_with_scratch, sum_f64_to_device,
            sum_f64_to_device_with_scratch, sum_i16_complex_scaled_buffer_size,
            sum_i16_complex_scaled_to_device, sum_i16_complex_scaled_to_device_with_scratch,
            sum_i16_complex_to_i32_complex_scaled_buffer_size,
            sum_i16_complex_to_i32_complex_scaled_to_device,
            sum_i16_complex_to_i32_complex_scaled_to_device_with_scratch,
            sum_i16_scaled_buffer_size, sum_i16_scaled_to_device,
            sum_i16_scaled_to_device_with_scratch, sum_i16_to_i32_scaled_buffer_size,
            sum_i16_to_i32_scaled_to_device, sum_i16_to_i32_scaled_to_device_with_scratch,
            sum_i32_scaled_buffer_size, sum_i32_scaled_to_device,
            sum_i32_scaled_to_device_with_scratch, sum_natural_logarithm_f32_buffer_size,
            sum_natural_logarithm_f32_to_device, sum_natural_logarithm_f32_to_device_with_scratch,
            sum_natural_logarithm_f32_to_f64_buffer_size,
            sum_natural_logarithm_f32_to_f64_to_device,
            sum_natural_logarithm_f32_to_f64_to_device_with_scratch,
            sum_natural_logarithm_f64_buffer_size, sum_natural_logarithm_f64_to_device,
            sum_natural_logarithm_f64_to_device_with_scratch,
            sum_natural_logarithm_i16_to_f32_buffer_size,
            sum_natural_logarithm_i16_to_f32_to_device,
            sum_natural_logarithm_i16_to_f32_to_device_with_scratch,
        },
        view::{SignalOutputView, SignalView},
    },
    types::{ComplexI16, ComplexI32, DataTypeLike},
    workspace::ScratchBuffer,
};

macro_rules! impl_signal_statistic_to_dispatch {
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
                source: &SignalView<'_, Self>,
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
                source: &SignalView<'_, Self>,
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
                    source: &SignalView<'_, Self>,
                    destination: &mut SignalOutputView<'_, $destination_ty>,
                    scratch: &mut ScratchBuffer,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct_scratch(stream_context, source, destination, scratch)

        }
    }

                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    destination: &mut SignalOutputView<'_, $destination_ty>,
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

        /// Returns the scratch capacity, in bytes, for the selected statistic and source length.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a length or
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
        /// Returns an error if CUDA context binding fails, a signal
        /// belongs to another CUDA context, `destination` does not contain exactly one element,
        /// a length or scratch-size conversion fails, or NPP reports an error.
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
            source: &SignalView<'_, Source>,
            destination: &mut SignalOutputView<'_, Destination>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            Source::$scratch_method(stream_context, source, destination, scratch)

        }
    }

        #[doc = $doc]
        ///
        /// Writes the scalar result into the single element of `destination`.
        /// Allocates temporary scratch storage for this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal
        /// belongs to another CUDA context, `destination` does not contain exactly one element,
        /// a length or scratch-size conversion fails, or NPP reports an error.
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
            source: &SignalView<'_, Source>,
            destination: &mut SignalOutputView<'_, Destination>,
        ) -> Result<()>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            Source::$method(stream_context, source, destination)

        }
    }
    };
}

macro_rules! impl_signal_scaled_statistic_to_dispatch {
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
                source: &SignalView<'_, Self>,
                destination: &mut SignalOutputView<'_, Destination>,
                scale_factor: i32,
                scratch: &mut ScratchBuffer,
            ) -> Result<()>;

            #[doc = $doc]
            ///
            /// Writes the scalar result into the single element of `destination`.
            /// Applies native fixed-point scaling using `scale_factor`.
            /// Allocates temporary scratch storage for this call.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails, a signal
            /// belongs to another CUDA context, `destination` does not contain exactly one element,
            /// a length or scratch-size conversion fails, or NPP reports an error.
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
                source: &SignalView<'_, Self>,
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
                    source: &SignalView<'_, Self>,
                    destination: &mut SignalOutputView<'_, $destination_ty>,
                    scale_factor: i32,
                    scratch: &mut ScratchBuffer,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct_scratch(stream_context, source, destination, scale_factor, scratch)

        }
    }

                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    destination: &mut SignalOutputView<'_, $destination_ty>,
                    scale_factor: i32,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct(stream_context, source, destination, scale_factor)

        }
    }
            }
        )*

        /// Returns the scratch capacity, in bytes, for the selected statistic and source length.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a length or
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
        /// Returns an error if CUDA context binding fails, a signal
        /// belongs to another CUDA context, `destination` does not contain exactly one element,
        /// a length or scratch-size conversion fails, or NPP reports an error.
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
            source: &SignalView<'_, Source>,
            destination: &mut SignalOutputView<'_, Destination>,
            scale_factor: i32,
            scratch: &mut ScratchBuffer,
        ) -> Result<()>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            Source::$scratch_method(stream_context, source, destination, scale_factor, scratch)

        }
    }

        #[doc = $doc]
        ///
        /// Writes the scalar result into the single element of `destination`.
        /// Applies native fixed-point scaling using `scale_factor`.
        /// Allocates temporary scratch storage for this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal
        /// belongs to another CUDA context, `destination` does not contain exactly one element,
        /// a length or scratch-size conversion fails, or NPP reports an error.
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
            source: &SignalView<'_, Source>,
            destination: &mut SignalOutputView<'_, Destination>,
            scale_factor: i32,
        ) -> Result<()>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            Source::$method(stream_context, source, destination, scale_factor)

        }
    }
    };
}

impl_signal_statistic_to_dispatch!("Computes the sum of the source elements.",

    SumTo,
    sum_buffer_size,
    sum_to_device_with_scratch,
    sum_to_device,
    sum_buffer_size,
    sum_to_device_with_scratch,
    sum_to_device,
    [
        f32 => (f32, sum_f32_buffer_size, sum_f32_to_device_with_scratch, sum_f32_to_device),
        Complex32 => (Complex32, sum_f32_complex_buffer_size, sum_f32_complex_to_device_with_scratch, sum_f32_complex_to_device),
        f64 => (f64, sum_f64_buffer_size, sum_f64_to_device_with_scratch, sum_f64_to_device),
        Complex64 => (Complex64, sum_f64_complex_buffer_size, sum_f64_complex_to_device_with_scratch, sum_f64_complex_to_device)
    ]
);

impl_signal_scaled_statistic_to_dispatch!("Computes the sum of the source elements.",

    SumScaledTo,
    sum_scaled_buffer_size,
    sum_scaled_to_device_with_scratch,
    sum_scaled_to_device,
    sum_scaled_buffer_size,
    sum_scaled_to_device_with_scratch,
    sum_scaled_to_device,
    [
        i16 => (i16, sum_i16_scaled_buffer_size, sum_i16_scaled_to_device_with_scratch, sum_i16_scaled_to_device),
        i16 => (i32, sum_i16_to_i32_scaled_buffer_size, sum_i16_to_i32_scaled_to_device_with_scratch, sum_i16_to_i32_scaled_to_device),
        i32 => (i32, sum_i32_scaled_buffer_size, sum_i32_scaled_to_device_with_scratch, sum_i32_scaled_to_device),
        ComplexI16 => (ComplexI16, sum_i16_complex_scaled_buffer_size, sum_i16_complex_scaled_to_device_with_scratch, sum_i16_complex_scaled_to_device),
        ComplexI16 => (ComplexI32, sum_i16_complex_to_i32_complex_scaled_buffer_size, sum_i16_complex_to_i32_complex_scaled_to_device_with_scratch, sum_i16_complex_to_i32_complex_scaled_to_device)
    ]
);

impl_signal_statistic_to_dispatch!("Computes the sum of the natural logarithms of the source elements.",

    SumNaturalLogarithmTo,
    sum_natural_logarithm_buffer_size,
    sum_natural_logarithm_to_device_with_scratch,
    sum_natural_logarithm_to_device,
    sum_natural_logarithm_buffer_size,
    sum_natural_logarithm_to_device_with_scratch,
    sum_natural_logarithm_to_device,
    [
        f32 => (f32, sum_natural_logarithm_f32_buffer_size, sum_natural_logarithm_f32_to_device_with_scratch, sum_natural_logarithm_f32_to_device),
        f32 => (f64, sum_natural_logarithm_f32_to_f64_buffer_size, sum_natural_logarithm_f32_to_f64_to_device_with_scratch, sum_natural_logarithm_f32_to_f64_to_device),
        f64 => (f64, sum_natural_logarithm_f64_buffer_size, sum_natural_logarithm_f64_to_device_with_scratch, sum_natural_logarithm_f64_to_device),
        i16 => (f32, sum_natural_logarithm_i16_to_f32_buffer_size, sum_natural_logarithm_i16_to_f32_to_device_with_scratch, sum_natural_logarithm_i16_to_f32_to_device)
    ]
);

impl_signal_statistic_to_dispatch!("Computes the arithmetic mean of the source elements.",

    MeanTo,
    mean_buffer_size,
    mean_to_device_with_scratch,
    mean_to_device,
    mean_buffer_size,
    mean_to_device_with_scratch,
    mean_to_device,
    [
        f32 => (f32, mean_f32_buffer_size, mean_f32_to_device_with_scratch, mean_f32_to_device),
        Complex32 => (Complex32, mean_f32_complex_buffer_size, mean_f32_complex_to_device_with_scratch, mean_f32_complex_to_device),
        f64 => (f64, mean_f64_buffer_size, mean_f64_to_device_with_scratch, mean_f64_to_device),
        Complex64 => (Complex64, mean_f64_complex_buffer_size, mean_f64_complex_to_device_with_scratch, mean_f64_complex_to_device)
    ]
);

impl_signal_scaled_statistic_to_dispatch!("Computes the arithmetic mean of the source elements.",

    MeanScaledTo,
    mean_scaled_buffer_size,
    mean_scaled_to_device_with_scratch,
    mean_scaled_to_device,
    mean_scaled_buffer_size,
    mean_scaled_to_device_with_scratch,
    mean_scaled_to_device,
    [
        i16 => (i16, mean_i16_scaled_buffer_size, mean_i16_scaled_to_device_with_scratch, mean_i16_scaled_to_device),
        i32 => (i32, mean_i32_scaled_buffer_size, mean_i32_scaled_to_device_with_scratch, mean_i32_scaled_to_device),
        ComplexI16 => (ComplexI16, mean_i16_complex_scaled_buffer_size, mean_i16_complex_scaled_to_device_with_scratch, mean_i16_complex_scaled_to_device)
    ]
);

impl_signal_statistic_to_dispatch!("Computes the standard deviation of the source elements.",

    StandardDeviationTo,
    standard_deviation_buffer_size,
    standard_deviation_to_device_with_scratch,
    standard_deviation_to_device,
    standard_deviation_buffer_size,
    standard_deviation_to_device_with_scratch,
    standard_deviation_to_device,
    [
        f32 => (f32, standard_deviation_f32_buffer_size, standard_deviation_f32_to_device_with_scratch, standard_deviation_f32_to_device),
        f64 => (f64, standard_deviation_f64_buffer_size, standard_deviation_f64_to_device_with_scratch, standard_deviation_f64_to_device)
    ]
);

impl_signal_scaled_statistic_to_dispatch!("Computes the standard deviation of the source elements.",

    StandardDeviationScaledTo,
    standard_deviation_scaled_buffer_size,
    standard_deviation_scaled_to_device_with_scratch,
    standard_deviation_scaled_to_device,
    standard_deviation_scaled_buffer_size,
    standard_deviation_scaled_to_device_with_scratch,
    standard_deviation_scaled_to_device,
    [
        i16 => (i16, standard_deviation_i16_scaled_buffer_size, standard_deviation_i16_scaled_to_device_with_scratch, standard_deviation_i16_scaled_to_device),
        i16 => (i32, standard_deviation_i16_to_i32_scaled_buffer_size, standard_deviation_i16_to_i32_scaled_to_device_with_scratch, standard_deviation_i16_to_i32_scaled_to_device)
    ]
);

impl_signal_statistic_to_dispatch!("Computes the infinity norm of the source vector.",

    NormInfTo,
    norm_inf_buffer_size,
    norm_inf_to_device_with_scratch,
    norm_inf_to_device,
    norm_inf_buffer_size,
    norm_inf_to_device_with_scratch,
    norm_inf_to_device,
    [
        f32 => (f32, norm_inf_f32_buffer_size, norm_inf_f32_to_device_with_scratch, norm_inf_f32_to_device),
        f64 => (f64, norm_inf_f64_buffer_size, norm_inf_f64_to_device_with_scratch, norm_inf_f64_to_device),
        Complex32 => (f32, norm_inf_f32_complex_to_f32_buffer_size, norm_inf_f32_complex_to_f32_to_device_with_scratch, norm_inf_f32_complex_to_f32_to_device),
        Complex64 => (f64, norm_inf_f64_complex_to_f64_buffer_size, norm_inf_f64_complex_to_f64_to_device_with_scratch, norm_inf_f64_complex_to_f64_to_device),
        i16 => (f32, norm_inf_i16_to_f32_buffer_size, norm_inf_i16_to_f32_to_device_with_scratch, norm_inf_i16_to_f32_to_device)
    ]
);

impl_signal_scaled_statistic_to_dispatch!("Computes the infinity norm of the source vector.",

    NormInfScaledTo,
    norm_inf_scaled_buffer_size,
    norm_inf_scaled_to_device_with_scratch,
    norm_inf_scaled_to_device,
    norm_inf_scaled_buffer_size,
    norm_inf_scaled_to_device_with_scratch,
    norm_inf_scaled_to_device,
    [
        i16 => (i32, norm_inf_i16_to_i32_scaled_buffer_size, norm_inf_i16_to_i32_scaled_to_device_with_scratch, norm_inf_i16_to_i32_scaled_to_device)
    ]
);

impl_signal_statistic_to_dispatch!("Computes the L1 norm of the source vector.",

    NormL1To,
    norm_l1_buffer_size,
    norm_l1_to_device_with_scratch,
    norm_l1_to_device,
    norm_l1_buffer_size,
    norm_l1_to_device_with_scratch,
    norm_l1_to_device,
    [
        f32 => (f32, norm_l1_f32_buffer_size, norm_l1_f32_to_device_with_scratch, norm_l1_f32_to_device),
        f64 => (f64, norm_l1_f64_buffer_size, norm_l1_f64_to_device_with_scratch, norm_l1_f64_to_device),
        Complex32 => (f64, norm_l1_f32_complex_to_f64_buffer_size, norm_l1_f32_complex_to_f64_to_device_with_scratch, norm_l1_f32_complex_to_f64_to_device),
        Complex64 => (f64, norm_l1_f64_complex_to_f64_buffer_size, norm_l1_f64_complex_to_f64_to_device_with_scratch, norm_l1_f64_complex_to_f64_to_device),
        i16 => (f32, norm_l1_i16_to_f32_buffer_size, norm_l1_i16_to_f32_to_device_with_scratch, norm_l1_i16_to_f32_to_device)
    ]
);

impl_signal_scaled_statistic_to_dispatch!("Computes the L1 norm of the source vector.",

    NormL1ScaledTo,
    norm_l1_scaled_buffer_size,
    norm_l1_scaled_to_device_with_scratch,
    norm_l1_scaled_to_device,
    norm_l1_scaled_buffer_size,
    norm_l1_scaled_to_device_with_scratch,
    norm_l1_scaled_to_device,
    [
        i16 => (i32, norm_l1_i16_to_i32_scaled_buffer_size, norm_l1_i16_to_i32_scaled_to_device_with_scratch, norm_l1_i16_to_i32_scaled_to_device),
        i16 => (i64, norm_l1_i16_to_i64_scaled_buffer_size, norm_l1_i16_to_i64_scaled_to_device_with_scratch, norm_l1_i16_to_i64_scaled_to_device)
    ]
);

impl_signal_statistic_to_dispatch!("Computes the L2 norm of the source vector.",

    NormL2To,
    norm_l2_buffer_size,
    norm_l2_to_device_with_scratch,
    norm_l2_to_device,
    norm_l2_buffer_size,
    norm_l2_to_device_with_scratch,
    norm_l2_to_device,
    [
        f32 => (f32, norm_l2_f32_buffer_size, norm_l2_f32_to_device_with_scratch, norm_l2_f32_to_device),
        f64 => (f64, norm_l2_f64_buffer_size, norm_l2_f64_to_device_with_scratch, norm_l2_f64_to_device),
        Complex32 => (f64, norm_l2_f32_complex_to_f64_buffer_size, norm_l2_f32_complex_to_f64_to_device_with_scratch, norm_l2_f32_complex_to_f64_to_device),
        Complex64 => (f64, norm_l2_f64_complex_to_f64_buffer_size, norm_l2_f64_complex_to_f64_to_device_with_scratch, norm_l2_f64_complex_to_f64_to_device),
        i16 => (f32, norm_l2_i16_to_f32_buffer_size, norm_l2_i16_to_f32_to_device_with_scratch, norm_l2_i16_to_f32_to_device)
    ]
);

impl_signal_scaled_statistic_to_dispatch!("Computes the L2 norm of the source vector.",

    NormL2ScaledTo,
    norm_l2_scaled_buffer_size,
    norm_l2_scaled_to_device_with_scratch,
    norm_l2_scaled_to_device,
    norm_l2_scaled_buffer_size,
    norm_l2_scaled_to_device_with_scratch,
    norm_l2_scaled_to_device,
    [
        i16 => (i32, norm_l2_i16_to_i32_scaled_buffer_size, norm_l2_i16_to_i32_scaled_to_device_with_scratch, norm_l2_i16_to_i32_scaled_to_device)
    ]
);

impl_signal_scaled_statistic_to_dispatch!("Computes the squared L2 norm of the source vector.",

    NormL2SquaredScaledTo,
    norm_l2_squared_scaled_buffer_size,
    norm_l2_squared_scaled_to_device_with_scratch,
    norm_l2_squared_scaled_to_device,
    norm_l2_squared_scaled_buffer_size,
    norm_l2_squared_scaled_to_device_with_scratch,
    norm_l2_squared_scaled_to_device,
    [
        i16 => (i64, norm_l2_squared_i16_to_i64_scaled_buffer_size, norm_l2_squared_i16_to_i64_scaled_to_device_with_scratch, norm_l2_squared_i16_to_i64_scaled_to_device)
    ]
);
