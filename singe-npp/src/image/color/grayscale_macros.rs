macro_rules! impl_rgb_to_gray {
    ($doc:literal, $name:ident, $ty:ty, $source_layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Uses the CCIR601 weights `0.299 * R + 0.587 * G + 0.114 * B` for gamma-corrected RGB.
        /// Enqueues the conversion without waiting for stream completion.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, the image sizes differ, or NPP rejects the conversion parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and readable, and destination pixels must be exclusively writable and disjoint from the source during native use.
        /// Keep image allocations valid until queued work completes, including every replay of a captured call.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $source_layout>,
            destination: &mut ImageOutputView<'_, $ty, C1>,
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
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_color_to_gray {
    ($doc:literal, $name:ident, $ty:ty, $source_layout:ty, $coefficients:literal, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Passes `coefficients` as a host array with one weight per processed source channel.
        /// The gray value is the sum of each processed channel multiplied by its coefficient.
        /// Coefficient values are passed to NPP without validation.
        /// Enqueues the conversion without waiting for stream completion.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, the image sizes differ, or NPP rejects the conversion parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and readable, and destination pixels must be exclusively writable and disjoint from the source during native use.
        /// Keep image allocations valid until queued work completes, including every replay of a captured call.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $source_layout>,
            destination: &mut ImageOutputView<'_, $ty, C1>,
            coefficients: &[f32; $coefficients],
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
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    coefficients.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_rgb_to_gray {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, $source_layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        /// Dispatches this grayscale conversion for a native element type and source channel layout.
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element representation and channel layout and uphold the documented conversion contract.
        pub unsafe trait $trait<SourceLayout>: DataTypeLike + Sized {
            #[doc = $doc]
            /// Uses the CCIR601 weights `0.299 * R + 0.587 * G + 0.114 * B` for gamma-corrected RGB.
            /// Enqueues the conversion without waiting for stream completion.
            /// Row padding remains unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, the image sizes differ, or NPP rejects the conversion parameters.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source pixels must be initialized and readable, and destination pixels must be exclusively writable and disjoint from the source during native use.
            /// Keep image allocations valid until queued work completes, including every replay of a captured call.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, SourceLayout>,
                destination: &mut ImageOutputView<'_, Self, C1>,
            ) -> Result<()>;
        }

        #[doc = $doc]
        /// Uses the CCIR601 weights `0.299 * R + 0.587 * G + 0.114 * B` for gamma-corrected RGB.
        /// Enqueues the conversion without waiting for stream completion.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, the image sizes differ, or NPP rejects the conversion parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and readable, and destination pixels must be exclusively writable and disjoint from the source during native use.
        /// Keep image allocations valid until queued work completes, including every replay of a captured call.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $source_layout>,
            destination: &mut ImageOutputView<'_, T, C1>,
        ) -> Result<()>
        where
            T: $trait<$source_layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, destination)
            }
}

        $(
            unsafe impl $trait<$source_layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $source_layout>,
                    destination: &mut ImageOutputView<'_, Self, C1>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_color_to_gray {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, $source_layout:ty, $coefficients:literal, [$($ty:ty => $direct:ident),* $(,)?]) => {
        /// Dispatches this grayscale conversion for a native element type and source channel layout.
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element representation and channel layout and uphold the documented conversion contract.
        pub unsafe trait $trait<SourceLayout>: DataTypeLike + Sized {
            #[doc = $doc]
            /// Passes `coefficients` as a host array with one weight per processed source channel.
            /// The gray value is the sum of each processed channel multiplied by its coefficient.
            /// Coefficient values are passed to NPP without validation.
            /// Enqueues the conversion without waiting for stream completion.
            /// Row padding remains unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, the image sizes differ, or NPP rejects the conversion parameters.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source pixels must be initialized and readable, and destination pixels must be exclusively writable and disjoint from the source during native use.
            /// Keep image allocations valid until queued work completes, including every replay of a captured call.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, SourceLayout>,
                destination: &mut ImageOutputView<'_, Self, C1>,
                coefficients: &[f32; $coefficients],
            ) -> Result<()>;
        }

        #[doc = $doc]
        /// Passes `coefficients` as a host array with one weight per processed source channel.
        /// The gray value is the sum of each processed channel multiplied by its coefficient.
        /// Coefficient values are passed to NPP without validation.
        /// Enqueues the conversion without waiting for stream completion.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, the image sizes differ, or NPP rejects the conversion parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and readable, and destination pixels must be exclusively writable and disjoint from the source during native use.
        /// Keep image allocations valid until queued work completes, including every replay of a captured call.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $source_layout>,
            destination: &mut ImageOutputView<'_, T, C1>,
            coefficients: &[f32; $coefficients],
        ) -> Result<()>
        where
            T: $trait<$source_layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, destination, coefficients)
            }
}

        $(
            unsafe impl $trait<$source_layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $source_layout>,
                    destination: &mut ImageOutputView<'_, Self, C1>,
                    coefficients: &[f32; $coefficients],
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, coefficients)
                    }
}
            }
        )*
    };
}

macro_rules! impl_gradient_color_to_gray {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// `norm` selects the gradient distance method passed to NPP.
        /// Enqueues the conversion without waiting for stream completion.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, the image sizes differ, or NPP rejects the conversion parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and readable, and destination pixels must be exclusively writable and disjoint from the source during native use.
        /// Keep image allocations valid until queued work completes, including every replay of a captured call.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C3>,
            destination: &mut ImageOutputView<'_, $ty, C1>,
            norm: ImageNormalization,
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
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    norm.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_cfa_to_rgb {
    ($doc:literal, $name:ident, $ty:ty, $destination_layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// `source_roi` selects a rectangle in the full source image; `grid` gives the Bayer registration at that rectangle's origin.
        /// Uses the native required `InterpolationMode::Undefined` interpolation setting.
        /// Enqueues the conversion without waiting for stream completion.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the conversion parameters.
        /// The source ROI must have a nonnegative origin and positive dimensions, fit inside the source without coordinate overflow, and match the destination size.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and readable, and destination pixels must be exclusively writable and disjoint from the source during native use.
        /// Keep image allocations valid until queued work completes, including every replay of a captured call.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            source_roi: Rectangle,
            destination: &mut ImageOutputView<'_, $ty, $destination_layout>,
            grid: BayerGridPosition,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_cfa_roi(source.size(), source_roi, destination.size())?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    source_roi.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    grid.into(),
                    InterpolationMode::Undefined.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_cfa_to_rgba {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// `source_roi` selects a rectangle in the full source image; `grid` gives the Bayer registration at that rectangle's origin.
        /// Uses the native required `InterpolationMode::Undefined` interpolation setting.
        /// Writes `alpha` to the fourth channel of every destination pixel.
        /// Enqueues the conversion without waiting for stream completion.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the conversion parameters.
        /// The source ROI must have a nonnegative origin and positive dimensions, fit inside the source without coordinate overflow, and match the destination size.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and readable, and destination pixels must be exclusively writable and disjoint from the source during native use.
        /// Keep image allocations valid until queued work completes, including every replay of a captured call.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            source_roi: Rectangle,
            destination: &mut ImageOutputView<'_, $ty, AC4>,
            grid: BayerGridPosition,
            alpha: $ty,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_cfa_roi(source.size(), source_roi, destination.size())?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    source_roi.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    grid.into(),
                    InterpolationMode::Undefined.into(),
                    alpha,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}
