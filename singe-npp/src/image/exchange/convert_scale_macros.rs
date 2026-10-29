macro_rules! impl_generic_convert_to_operation {
    ($trait:ident, $function:ident, $destination_ty:ty, $layout:ty, [$($src_ty:ty => $direct:ident),+ $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the source and destination native element ABI and channel layout.
        /// They must use a compatible NPP routine and uphold the method's size and range checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Converts source samples to the destination element type.
            /// Preserves numeric values subject to native rounding and saturation; it does not scale their range.
            /// Integer results are clamped to the destination type's range.
            ///
            /// Enqueues work on the selected stream.
            /// For AC4 layouts, the destination alpha channel is not processed.
            /// Row padding is not initialized.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, source and destination sizes differ, or NPP rejects the conversion parameters.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, $destination_ty, Layout>,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait<$layout> for $src_ty {
                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, $destination_ty, $layout>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination)
                    }
}
            }
        )+

        /// Converts source samples to the destination element type.
        /// Preserves numeric values subject to native rounding and saturation; it does not scale their range.
        /// Integer results are clamped to the destination type's range.
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, the destination alpha channel is not processed.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, source and destination sizes differ, or NPP rejects the conversion parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, $destination_ty, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source, destination)
            }
}
    };
}

macro_rules! impl_generic_convert_round_to_operation {
    ($trait:ident, $function:ident, $destination_ty:ty, $layout:ty, [$($src_ty:ty => $direct:ident),+ $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the source and destination native element ABI and channel layout.
        /// They must use a compatible NPP routine and uphold the method's size and range checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Converts source samples to the destination element type.
            /// Preserves numeric values subject to native rounding and saturation; it does not scale their range.
            /// Uses `round_mode` for conversion to integer values.
            /// Integer results are clamped to the destination type's range.
            ///
            /// Enqueues work on the selected stream.
            /// For AC4 layouts, the destination alpha channel is not processed.
            /// Row padding is not initialized.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, source and destination sizes differ, or NPP rejects the conversion parameters.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, $destination_ty, Layout>,
                round_mode: RoundMode,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait<$layout> for $src_ty {
                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, $destination_ty, $layout>,
                    round_mode: RoundMode,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, round_mode)
                    }
}
            }
        )+

        /// Converts source samples to the destination element type.
        /// Preserves numeric values subject to native rounding and saturation; it does not scale their range.
        /// Uses `round_mode` for conversion to integer values.
        /// Integer results are clamped to the destination type's range.
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, the destination alpha channel is not processed.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, source and destination sizes differ, or NPP rejects the conversion parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, $destination_ty, $layout>,
            round_mode: RoundMode,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source, destination, round_mode)
            }
}
    };
}

macro_rules! impl_generic_convert_scaled_round_to_operation {
    ($trait:ident, $function:ident, $destination_ty:ty, $layout:ty, [$($src_ty:ty => $direct:ident),+ $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the source and destination native element ABI and channel layout.
        /// They must use a compatible NPP routine and uphold the method's size and range checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Converts source samples to the destination element type.
            /// Multiplies samples by `2^(-scale_factor)` before rounding and saturation.
            /// Uses `round_mode` for conversion to integer values.
            /// Integer results are clamped to the destination type's range.
            ///
            /// Enqueues work on the selected stream.
            /// For AC4 layouts, the destination alpha channel is not processed.
            /// Row padding is not initialized.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, source and destination sizes differ, or NPP rejects the conversion parameters.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, $destination_ty, Layout>,
                round_mode: RoundMode,
                scale_factor: i32,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait<$layout> for $src_ty {
                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, $destination_ty, $layout>,
                    round_mode: RoundMode,
                    scale_factor: i32,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, round_mode, scale_factor)
                    }
}
            }
        )+

        /// Converts source samples to the destination element type.
        /// Multiplies samples by `2^(-scale_factor)` before rounding and saturation.
        /// Uses `round_mode` for conversion to integer values.
        /// Integer results are clamped to the destination type's range.
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, the destination alpha channel is not processed.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, source and destination sizes differ, or NPP rejects the conversion parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, $destination_ty, $layout>,
            round_mode: RoundMode,
            scale_factor: i32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source, destination, round_mode, scale_factor)
            }
}
    };
}

macro_rules! impl_scale_to_higher {
    ($name:ident, $src_ty:ty, $dst_ty:ty, $layout:ty, $ffi:ident) => {
        /// Linearly maps the source integer type's full numeric range to the destination integer type's full numeric range.
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, the destination alpha channel is not processed.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, source and destination sizes differ, or NPP rejects the conversion parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $src_ty, $layout>,
            destination: &mut ImageOutputView<'_, $dst_ty, $layout>,
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
                    destination.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_scale_to_higher_range {
    ($name:ident, $src_ty:ty, $layout:ty, $ffi:ident) => {
        /// Linearly maps the source integer type's full numeric range to `min..=max` in the floating-point destination.
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Results are clamped to the specified destination range.
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, the destination alpha channel is not processed.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, source and destination sizes differ, or NPP rejects the conversion parameters.
        /// `min` and `max` must be finite, with `min < max`.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $src_ty, $layout>,
            destination: &mut ImageOutputView<'_, f32, $layout>,
            min: f32,
            max: f32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;
            validate_scale_range(min, max)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    min,
                    max,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_scale_to_lower_hint {
    ($name:ident, $src_ty:ty, $layout:ty, $ffi:ident) => {
        /// Linearly maps the source integer type's full numeric range to the destination integer type's full numeric range.
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Passes `hint` to NPP as the requested performance or accuracy preference.
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, the destination alpha channel is not processed.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, source and destination sizes differ, or NPP rejects the conversion parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $src_ty, $layout>,
            destination: &mut ImageOutputView<'_, u8, $layout>,
            hint: HintAlgorithm,
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
                    destination.size().into(),
                    hint.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_scale_to_lower_range {
    ($name:ident, $layout:ty, $ffi:ident) => {
        /// Linearly maps the floating-point source range `min..=max` to `0..=255` in the destination.
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Results outside the destination range are clamped.
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, the destination alpha channel is not processed.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, source and destination sizes differ, or NPP rejects the conversion parameters.
        /// `min` and `max` must be finite, with `min < max`.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, f32, $layout>,
            destination: &mut ImageOutputView<'_, u8, $layout>,
            min: f32,
            max: f32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;
            validate_scale_range(min, max)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    min,
                    max,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_scale_to_higher_operation {
    ($trait:ident, $function:ident, $layout:ty, [$($src_ty:ty => ($dst_ty:ty, $direct:ident)),+ $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the source and destination native element ABI and channel layout.
        /// They must use a compatible NPP routine and uphold the method's size and range checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Destination element type paired with this source type by the native scaling routine.
            type Destination: DataTypeLike;

            /// Linearly maps the source integer type's full numeric range to the destination integer type's full numeric range.
            ///
            /// Enqueues work on the selected stream.
            /// For AC4 layouts, the destination alpha channel is not processed.
            /// Row padding is not initialized.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, source and destination sizes differ, or NPP rejects the conversion parameters.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self::Destination, Layout>,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait<$layout> for $src_ty {
                type Destination = $dst_ty;

                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, Self::Destination, $layout>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination)
                    }
}
            }
        )+

        /// Linearly maps the source integer type's full numeric range to the destination integer type's full numeric range.
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, the destination alpha channel is not processed.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, source and destination sizes differ, or NPP rejects the conversion parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, T::Destination, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source, destination)
            }
}
    };
}

macro_rules! impl_generic_scale_to_f32_operation {
    ($trait:ident, $function:ident, $layout:ty, [$($src_ty:ty => $direct:ident),+ $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the source and destination native element ABI and channel layout.
        /// They must use a compatible NPP routine and uphold the method's size and range checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Linearly maps the source integer type's full numeric range to `min..=max` in the floating-point destination.
            /// Results are clamped to the specified destination range.
            ///
            /// Enqueues work on the selected stream.
            /// For AC4 layouts, the destination alpha channel is not processed.
            /// Row padding is not initialized.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, source and destination sizes differ, or NPP rejects the conversion parameters.
            /// `min` and `max` must be finite, with `min < max`.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, f32, Layout>,
                min: f32,
                max: f32,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait<$layout> for $src_ty {
                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, f32, $layout>,
                    min: f32,
                    max: f32,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, min, max)
                    }
}
            }
        )+

        /// Linearly maps the source integer type's full numeric range to `min..=max` in the floating-point destination.
        /// Results are clamped to the specified destination range.
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, the destination alpha channel is not processed.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, source and destination sizes differ, or NPP rejects the conversion parameters.
        /// `min` and `max` must be finite, with `min < max`.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, f32, $layout>,
            min: f32,
            max: f32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source, destination, min, max)
            }
}
    };
}

macro_rules! impl_generic_scale_to_u8_hint_operation {
    ($trait:ident, $function:ident, $layout:ty, [$($src_ty:ty => $direct:ident),+ $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the source and destination native element ABI and channel layout.
        /// They must use a compatible NPP routine and uphold the method's size and range checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Linearly maps the source integer type's full numeric range to the destination integer type's full numeric range.
            /// Passes `hint` to NPP as the requested performance or accuracy preference.
            ///
            /// Enqueues work on the selected stream.
            /// For AC4 layouts, the destination alpha channel is not processed.
            /// Row padding is not initialized.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, source and destination sizes differ, or NPP rejects the conversion parameters.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, u8, Layout>,
                hint: HintAlgorithm,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait<$layout> for $src_ty {
                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, u8, $layout>,
                    hint: HintAlgorithm,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, hint)
                    }
}
            }
        )+

        /// Linearly maps the source integer type's full numeric range to the destination integer type's full numeric range.
        /// Passes `hint` to NPP as the requested performance or accuracy preference.
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, the destination alpha channel is not processed.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, source and destination sizes differ, or NPP rejects the conversion parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, u8, $layout>,
            hint: HintAlgorithm,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source, destination, hint)
            }
}
    };
}

macro_rules! impl_generic_scale_to_u8_range_operation {
    ($trait:ident, $function:ident, $layout:ty, [$($src_ty:ty => $direct:ident),+ $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the source and destination native element ABI and channel layout.
        /// They must use a compatible NPP routine and uphold the method's size and range checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Linearly maps the floating-point source range `min..=max` to `0..=255` in the destination.
            /// Results outside the destination range are clamped.
            ///
            /// Enqueues work on the selected stream.
            /// For AC4 layouts, the destination alpha channel is not processed.
            /// Row padding is not initialized.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, source and destination sizes differ, or NPP rejects the conversion parameters.
            /// `min` and `max` must be finite, with `min < max`.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, u8, Layout>,
                min: f32,
                max: f32,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait<$layout> for $src_ty {
                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, u8, $layout>,
                    min: f32,
                    max: f32,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, min, max)
                    }
}
            }
        )+

        /// Linearly maps the floating-point source range `min..=max` to `0..=255` in the destination.
        /// Results outside the destination range are clamped.
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, the destination alpha channel is not processed.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, source and destination sizes differ, or NPP rejects the conversion parameters.
        /// `min` and `max` must be finite, with `min < max`.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, u8, $layout>,
            min: f32,
            max: f32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source, destination, min, max)
            }
}
    };
}
