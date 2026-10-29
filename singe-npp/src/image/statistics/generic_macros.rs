macro_rules! impl_generic_statistic {
    ($doc:literal, $trait_name:ident, $name:ident, $buffer_size_name:ident, $layout:ty, [$(($ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        #[doc = concat!("Native element and result dispatch for [`", stringify!($name), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must match the native element, layout, and result ABI.
        /// They must preserve the access, output-length, and completion contracts of `dispatch`.
        /// Buffer-size queries must return the required capacity in bytes.
        pub unsafe trait $trait_name<Layout>: DataTypeLike {
            /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if NPP rejects the ROI or scratch-size query,
            /// or the returned size cannot be represented as `usize`.
            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            #[doc = $doc]
            ///
            /// Provided implementations allocate temporary GPU scratch and return after successful stream completion.
            /// This scratch-owning form rejects stream capture.
            /// Each result or coordinate view must contain exactly one element per processed channel.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
            /// Result views must have the exact lengths required by the selected native operation.
            /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
            /// NPP operation errors are propagated.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
            /// Result storage must be exclusively writable and disjoint from all input and other result ranges
            /// until this call completes its native work.
            /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the selected native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                output: &mut DeviceOutputView<'_, f64>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            unsafe impl $trait_name<$layout> for $ty {
                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    output: &mut DeviceOutputView<'_, f64>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, output)
                    }
}
            }
        )+

        /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if NPP rejects the ROI or scratch-size query,
        /// or the returned size cannot be represented as `usize`.
        pub fn $buffer_size_name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        #[doc = $doc]
        ///
        /// Provided implementations allocate temporary GPU scratch and return after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// Each result or coordinate view must contain exactly one element per processed channel.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// Result views must have the exact lengths required by the selected native operation.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the selected native routine.
        pub unsafe fn $name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            output: &mut DeviceOutputView<'_, f64>,
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source, output)
            }
}
    };
}

macro_rules! impl_generic_typed_statistic {
    ($doc:literal, $trait_name:ident, $name:ident, $buffer_size_name:ident, $layout:ty, [$(($ty:ty, $output_ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        #[doc = concat!("Native element and result dispatch for [`", stringify!($name), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must match the native element, layout, and result ABI.
        /// They must preserve the access, output-length, and completion contracts of `dispatch`.
        /// Buffer-size queries must return the required capacity in bytes.
        pub unsafe trait $trait_name<Layout>: DataTypeLike {
            /// Native scalar result type selected for this input element type.
            type Output: DataTypeLike;

            /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if NPP rejects the ROI or scratch-size query,
            /// or the returned size cannot be represented as `usize`.
            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            #[doc = $doc]
            ///
            /// Provided implementations allocate temporary GPU scratch and return after successful stream completion.
            /// This scratch-owning form rejects stream capture.
            /// Each result or coordinate view must contain exactly one element per processed channel.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
            /// Result views must have the exact lengths required by the selected native operation.
            /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
            /// NPP operation errors are propagated.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
            /// Result storage must be exclusively writable and disjoint from all input and other result ranges
            /// until this call completes its native work.
            /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the selected native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                output: &mut DeviceOutputView<'_, Self::Output>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            unsafe impl $trait_name<$layout> for $ty {
                type Output = $output_ty;

                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    output: &mut DeviceOutputView<'_, Self::Output>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, output)
                    }
}
            }
        )+

        /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if NPP rejects the ROI or scratch-size query,
        /// or the returned size cannot be represented as `usize`.
        pub fn $buffer_size_name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        #[doc = $doc]
        ///
        /// Provided implementations allocate temporary GPU scratch and return after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// Each result or coordinate view must contain exactly one element per processed channel.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// Result views must have the exact lengths required by the selected native operation.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the selected native routine.
        pub unsafe fn $name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            output: &mut DeviceOutputView<'_, T::Output>,
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source, output)
            }
}
    };
}

macro_rules! impl_generic_typed_pair_statistic {
    ($doc:literal, $trait_name:ident, $name:ident, $buffer_size_name:ident, $layout:ty, [$(($ty:ty, $output_ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        #[doc = concat!("Native element and result dispatch for [`", stringify!($name), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must match the native element, layout, and result ABI.
        /// They must preserve the access, output-length, and completion contracts of `dispatch`.
        /// Buffer-size queries must return the required capacity in bytes.
        pub unsafe trait $trait_name<Layout>: DataTypeLike {
            /// Native scalar result type selected for this input element type.
            type Output: DataTypeLike;

            /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if NPP rejects the ROI or scratch-size query,
            /// or the returned size cannot be represented as `usize`.
            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            #[doc = $doc]
            ///
            /// Provided implementations allocate temporary GPU scratch and return after successful stream completion.
            /// This scratch-owning form rejects stream capture.
            /// Each result or coordinate view must contain exactly one element per processed channel.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
            /// Result views must have the exact lengths required by the selected native operation.
            /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
            /// NPP operation errors are propagated.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
            /// Result storage must be exclusively writable and disjoint from all input and other result ranges
            /// until this call completes its native work.
            /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the selected native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                minimum: &mut DeviceOutputView<'_, Self::Output>,
                maximum: &mut DeviceOutputView<'_, Self::Output>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            unsafe impl $trait_name<$layout> for $ty {
                type Output = $output_ty;

                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    minimum: &mut DeviceOutputView<'_, Self::Output>,
                    maximum: &mut DeviceOutputView<'_, Self::Output>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, minimum, maximum)
                    }
}
            }
        )+

        /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if NPP rejects the ROI or scratch-size query,
        /// or the returned size cannot be represented as `usize`.
        pub fn $buffer_size_name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        #[doc = $doc]
        ///
        /// Provided implementations allocate temporary GPU scratch and return after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// Each result or coordinate view must contain exactly one element per processed channel.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// Result views must have the exact lengths required by the selected native operation.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the selected native routine.
        pub unsafe fn $name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            minimum: &mut DeviceOutputView<'_, T::Output>,
            maximum: &mut DeviceOutputView<'_, T::Output>,
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source, minimum, maximum)
            }
}
    };
}

macro_rules! impl_generic_typed_indexed_statistic {
    ($doc:literal, $trait_name:ident, $name:ident, $buffer_size_name:ident, $layout:ty, [$(($ty:ty, $output_ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        #[doc = concat!("Native element and result dispatch for [`", stringify!($name), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must match the native element, layout, and result ABI.
        /// They must preserve the access, output-length, and completion contracts of `dispatch`.
        /// Buffer-size queries must return the required capacity in bytes.
        pub unsafe trait $trait_name<Layout>: DataTypeLike {
            /// Native scalar result type selected for this input element type.
            type Output: DataTypeLike;

            /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if NPP rejects the ROI or scratch-size query,
            /// or the returned size cannot be represented as `usize`.
            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            #[doc = $doc]
            ///
            /// Provided implementations allocate temporary GPU scratch and return after successful stream completion.
            /// This scratch-owning form rejects stream capture.
            /// Each result or coordinate view must contain exactly one element per processed channel.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
            /// Result views must have the exact lengths required by the selected native operation.
            /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
            /// NPP operation errors are propagated.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
            /// Result storage must be exclusively writable and disjoint from all input and other result ranges
            /// until this call completes its native work.
            /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the selected native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                output: &mut DeviceOutputView<'_, Self::Output>,
                index_x: &mut DeviceOutputView<'_, i32>,
                index_y: &mut DeviceOutputView<'_, i32>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            unsafe impl $trait_name<$layout> for $ty {
                type Output = $output_ty;

                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    output: &mut DeviceOutputView<'_, Self::Output>,
                    index_x: &mut DeviceOutputView<'_, i32>,
                    index_y: &mut DeviceOutputView<'_, i32>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, output, index_x, index_y)
                    }
}
            }
        )+

        /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if NPP rejects the ROI or scratch-size query,
        /// or the returned size cannot be represented as `usize`.
        pub fn $buffer_size_name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        #[doc = $doc]
        ///
        /// Provided implementations allocate temporary GPU scratch and return after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// Each result or coordinate view must contain exactly one element per processed channel.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// Result views must have the exact lengths required by the selected native operation.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the selected native routine.
        pub unsafe fn $name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            output: &mut DeviceOutputView<'_, T::Output>,
            index_x: &mut DeviceOutputView<'_, i32>,
            index_y: &mut DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source, output, index_x, index_y)
            }
}
    };
}

macro_rules! impl_generic_typed_pair_indexed_statistic_c1 {
    ($doc:literal, $trait_name:ident, $name:ident, $buffer_size_name:ident, [$(($ty:ty, $output_ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        #[doc = concat!("Native element and result dispatch for [`", stringify!($name), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must match the native element, layout, and result ABI.
        /// They must preserve the access, output-length, and completion contracts of `dispatch`.
        /// Buffer-size queries must return the required capacity in bytes.
        pub unsafe trait $trait_name: DataTypeLike {
            /// Native scalar result type selected for this input element type.
            type Output: DataTypeLike;

            /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if NPP rejects the ROI or scratch-size query,
            /// or the returned size cannot be represented as `usize`.
            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            #[doc = $doc]
            ///
            /// Provided implementations allocate temporary GPU scratch and return after successful stream completion.
            /// This scratch-owning form rejects stream capture.
            /// Each result or coordinate view must contain exactly one element per processed channel.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
            /// Result views must have the exact lengths required by the selected native operation.
            /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
            /// NPP operation errors are propagated.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
            /// Result storage must be exclusively writable and disjoint from all input and other result ranges
            /// until this call completes its native work.
            /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the selected native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                minimum: &mut DeviceOutputView<'_, Self::Output>,
                maximum: &mut DeviceOutputView<'_, Self::Output>,
                minimum_index: &mut DeviceOutputView<'_, Point>,
                maximum_index: &mut DeviceOutputView<'_, Point>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            unsafe impl $trait_name for $ty {
                type Output = $output_ty;

                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C1>,
                    minimum: &mut DeviceOutputView<'_, Self::Output>,
                    maximum: &mut DeviceOutputView<'_, Self::Output>,
                    minimum_index: &mut DeviceOutputView<'_, Point>,
                    maximum_index: &mut DeviceOutputView<'_, Point>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(
                            stream_context,
                            source,
                            minimum,
                            maximum,
                            minimum_index,
                            maximum_index,
                        )
                    }
                }
            }
        )+

        /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if NPP rejects the ROI or scratch-size query,
        /// or the returned size cannot be represented as `usize`.
        pub fn $buffer_size_name<T: $trait_name>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        #[doc = $doc]
        ///
        /// Provided implementations allocate temporary GPU scratch and return after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// Each result or coordinate view must contain exactly one element per processed channel.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// Result views must have the exact lengths required by the selected native operation.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the selected native routine.
        pub unsafe fn $name<T: $trait_name>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C1>,
            minimum: &mut DeviceOutputView<'_, T::Output>,
            maximum: &mut DeviceOutputView<'_, T::Output>,
            minimum_index: &mut DeviceOutputView<'_, Point>,
            maximum_index: &mut DeviceOutputView<'_, Point>,
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(
                    stream_context,
                    source,
                    minimum,
                    maximum,
                    minimum_index,
                    maximum_index,
                )
            }
        }
    };
}

macro_rules! impl_generic_typed_pair_indexed_statistic_c3 {
    ($doc:literal, $trait_name:ident, $name:ident, $buffer_size_name:ident, [$(($ty:ty, $output_ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        #[doc = concat!("Native element and result dispatch for [`", stringify!($name), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must match the native element, layout, and result ABI.
        /// They must preserve the access, output-length, and completion contracts of `dispatch`.
        /// Buffer-size queries must return the required capacity in bytes.
        pub unsafe trait $trait_name: DataTypeLike {
            /// Native scalar result type selected for this input element type.
            type Output: DataTypeLike;

            /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if NPP rejects the ROI or scratch-size query,
            /// or the returned size cannot be represented as `usize`.
            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            #[doc = $doc]
            ///
            /// Provided implementations allocate temporary GPU scratch and return after successful stream completion.
            /// This scratch-owning form rejects stream capture.
            /// `channel` is zero-based and must be less than 3.
            /// Each result or coordinate view must contain exactly one element per processed channel.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
            /// An out-of-range `channel` returns an error.
            /// Result views must have the exact lengths required by the selected native operation.
            /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
            /// NPP operation errors are propagated.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
            /// Result storage must be exclusively writable and disjoint from all input and other result ranges
            /// until this call completes its native work.
            /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the selected native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C3>,
                channel: usize,
                minimum: &mut DeviceOutputView<'_, Self::Output>,
                maximum: &mut DeviceOutputView<'_, Self::Output>,
                minimum_index: &mut DeviceOutputView<'_, Point>,
                maximum_index: &mut DeviceOutputView<'_, Point>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            unsafe impl $trait_name for $ty {
                type Output = $output_ty;

                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C3>,
                    channel: usize,
                    minimum: &mut DeviceOutputView<'_, Self::Output>,
                    maximum: &mut DeviceOutputView<'_, Self::Output>,
                    minimum_index: &mut DeviceOutputView<'_, Point>,
                    maximum_index: &mut DeviceOutputView<'_, Point>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(
                            stream_context,
                            source,
                            channel,
                            minimum,
                            maximum,
                            minimum_index,
                            maximum_index,
                        )
                    }
                }
            }
        )+

        /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if NPP rejects the ROI or scratch-size query,
        /// or the returned size cannot be represented as `usize`.
        pub fn $buffer_size_name<T: $trait_name>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        #[doc = $doc]
        ///
        /// Provided implementations allocate temporary GPU scratch and return after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// `channel` is zero-based and must be less than 3.
        /// Each result or coordinate view must contain exactly one element per processed channel.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// An out-of-range `channel` returns an error.
        /// Result views must have the exact lengths required by the selected native operation.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the selected native routine.
        pub unsafe fn $name<T: $trait_name>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C3>,
            channel: usize,
            minimum: &mut DeviceOutputView<'_, T::Output>,
            maximum: &mut DeviceOutputView<'_, T::Output>,
            minimum_index: &mut DeviceOutputView<'_, Point>,
            maximum_index: &mut DeviceOutputView<'_, Point>,
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(
                    stream_context,
                    source,
                    channel,
                    minimum,
                    maximum,
                    minimum_index,
                    maximum_index,
                )
            }
        }
    };
}

macro_rules! impl_generic_typed_pair_indexed_statistic_masked_c1 {
    ($doc:literal, $trait_name:ident, $name:ident, $buffer_size_name:ident, [$(($ty:ty, $output_ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        #[doc = concat!("Native element and result dispatch for [`", stringify!($name), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must match the native element, layout, and result ABI.
        /// They must preserve the access, output-length, and completion contracts of `dispatch`.
        /// Buffer-size queries must return the required capacity in bytes.
        pub unsafe trait $trait_name: DataTypeLike {
            /// Native scalar result type selected for this input element type.
            type Output: DataTypeLike;

            /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if NPP rejects the ROI or scratch-size query,
            /// or the returned size cannot be represented as `usize`.
            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            #[doc = $doc]
            ///
            /// Provided implementations allocate temporary GPU scratch and return after successful stream completion.
            /// This scratch-owning form rejects stream capture.
            /// The mask must have the same ROI size as the source image.
            /// Each result or coordinate view must contain exactly one element per processed channel.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
            /// A mask with a different ROI size returns an error.
            /// Result views must have the exact lengths required by the selected native operation.
            /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
            /// NPP operation errors are propagated.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
            /// Result storage must be exclusively writable and disjoint from all input and other result ranges
            /// until this call completes its native work.
            /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the selected native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                mask: &ImageView<'_, u8, C1>,
                minimum: &mut DeviceOutputView<'_, Self::Output>,
                maximum: &mut DeviceOutputView<'_, Self::Output>,
                minimum_index: &mut DeviceOutputView<'_, Point>,
                maximum_index: &mut DeviceOutputView<'_, Point>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            unsafe impl $trait_name for $ty {
                type Output = $output_ty;

                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C1>,
                    mask: &ImageView<'_, u8, C1>,
                    minimum: &mut DeviceOutputView<'_, Self::Output>,
                    maximum: &mut DeviceOutputView<'_, Self::Output>,
                    minimum_index: &mut DeviceOutputView<'_, Point>,
                    maximum_index: &mut DeviceOutputView<'_, Point>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(
                            stream_context,
                            source,
                            mask,
                            minimum,
                            maximum,
                            minimum_index,
                            maximum_index,
                        )
                    }
                }
            }
        )+

        /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if NPP rejects the ROI or scratch-size query,
        /// or the returned size cannot be represented as `usize`.
        pub fn $buffer_size_name<T: $trait_name>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        #[doc = $doc]
        ///
        /// Provided implementations allocate temporary GPU scratch and return after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// The mask must have the same ROI size as the source image.
        /// Each result or coordinate view must contain exactly one element per processed channel.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// A mask with a different ROI size returns an error.
        /// Result views must have the exact lengths required by the selected native operation.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the selected native routine.
        pub unsafe fn $name<T: $trait_name>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C1>,
            mask: &ImageView<'_, u8, C1>,
            minimum: &mut DeviceOutputView<'_, T::Output>,
            maximum: &mut DeviceOutputView<'_, T::Output>,
            minimum_index: &mut DeviceOutputView<'_, Point>,
            maximum_index: &mut DeviceOutputView<'_, Point>,
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(
                    stream_context,
                    source,
                    mask,
                    minimum,
                    maximum,
                    minimum_index,
                    maximum_index,
                )
            }
        }
    };
}

macro_rules! impl_generic_typed_pair_indexed_statistic_masked_c3 {
    ($doc:literal, $trait_name:ident, $name:ident, $buffer_size_name:ident, [$(($ty:ty, $output_ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        #[doc = concat!("Native element and result dispatch for [`", stringify!($name), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must match the native element, layout, and result ABI.
        /// They must preserve the access, output-length, and completion contracts of `dispatch`.
        /// Buffer-size queries must return the required capacity in bytes.
        pub unsafe trait $trait_name: DataTypeLike {
            /// Native scalar result type selected for this input element type.
            type Output: DataTypeLike;

            /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if NPP rejects the ROI or scratch-size query,
            /// or the returned size cannot be represented as `usize`.
            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            #[doc = $doc]
            ///
            /// Provided implementations allocate temporary GPU scratch and return after successful stream completion.
            /// This scratch-owning form rejects stream capture.
            /// The mask must have the same ROI size as the source image.
            /// `channel` is zero-based and must be less than 3.
            /// Each result or coordinate view must contain exactly one element per processed channel.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
            /// A mask with a different ROI size returns an error.
            /// An out-of-range `channel` returns an error.
            /// Result views must have the exact lengths required by the selected native operation.
            /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
            /// NPP operation errors are propagated.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
            /// Result storage must be exclusively writable and disjoint from all input and other result ranges
            /// until this call completes its native work.
            /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the selected native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C3>,
                mask: &ImageView<'_, u8, C1>,
                channel: usize,
                minimum: &mut DeviceOutputView<'_, Self::Output>,
                maximum: &mut DeviceOutputView<'_, Self::Output>,
                minimum_index: &mut DeviceOutputView<'_, Point>,
                maximum_index: &mut DeviceOutputView<'_, Point>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            unsafe impl $trait_name for $ty {
                type Output = $output_ty;

                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C3>,
                    mask: &ImageView<'_, u8, C1>,
                    channel: usize,
                    minimum: &mut DeviceOutputView<'_, Self::Output>,
                    maximum: &mut DeviceOutputView<'_, Self::Output>,
                    minimum_index: &mut DeviceOutputView<'_, Point>,
                    maximum_index: &mut DeviceOutputView<'_, Point>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(
                            stream_context,
                            source,
                            mask,
                            channel,
                            minimum,
                            maximum,
                            minimum_index,
                            maximum_index,
                        )
                    }
                }
            }
        )+

        /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if NPP rejects the ROI or scratch-size query,
        /// or the returned size cannot be represented as `usize`.
        pub fn $buffer_size_name<T: $trait_name>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        #[doc = $doc]
        ///
        /// Provided implementations allocate temporary GPU scratch and return after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// The mask must have the same ROI size as the source image.
        /// `channel` is zero-based and must be less than 3.
        /// Each result or coordinate view must contain exactly one element per processed channel.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// A mask with a different ROI size returns an error.
        /// An out-of-range `channel` returns an error.
        /// Result views must have the exact lengths required by the selected native operation.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the selected native routine.
        pub unsafe fn $name<T: $trait_name>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C3>,
            mask: &ImageView<'_, u8, C1>,
            channel: usize,
            minimum: &mut DeviceOutputView<'_, T::Output>,
            maximum: &mut DeviceOutputView<'_, T::Output>,
            minimum_index: &mut DeviceOutputView<'_, Point>,
            maximum_index: &mut DeviceOutputView<'_, Point>,
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(
                    stream_context,
                    source,
                    mask,
                    channel,
                    minimum,
                    maximum,
                    minimum_index,
                    maximum_index,
                )
            }
        }
    };
}

macro_rules! impl_generic_statistic_masked_c1 {
    ($doc:literal, $trait_name:ident, $name:ident, $buffer_size_name:ident, [$(($ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        #[doc = concat!("Native element and result dispatch for [`", stringify!($name), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must match the native element, layout, and result ABI.
        /// They must preserve the access, output-length, and completion contracts of `dispatch`.
        /// Buffer-size queries must return the required capacity in bytes.
        pub unsafe trait $trait_name: DataTypeLike {
            /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if NPP rejects the ROI or scratch-size query,
            /// or the returned size cannot be represented as `usize`.
            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            #[doc = $doc]
            ///
            /// Provided implementations allocate temporary GPU scratch and return after successful stream completion.
            /// This scratch-owning form rejects stream capture.
            /// The mask must have the same ROI size as the source image.
            /// Each result or coordinate view must contain exactly one element per processed channel.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
            /// A mask with a different ROI size returns an error.
            /// Result views must have the exact lengths required by the selected native operation.
            /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
            /// NPP operation errors are propagated.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
            /// Result storage must be exclusively writable and disjoint from all input and other result ranges
            /// until this call completes its native work.
            /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the selected native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                mask: &ImageView<'_, u8, C1>,
                output: &mut DeviceOutputView<'_, f64>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            unsafe impl $trait_name for $ty {
                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C1>,
                    mask: &ImageView<'_, u8, C1>,
                    output: &mut DeviceOutputView<'_, f64>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, mask, output)
                    }
}
            }
        )+

        /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if NPP rejects the ROI or scratch-size query,
        /// or the returned size cannot be represented as `usize`.
        pub fn $buffer_size_name<T: $trait_name>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        #[doc = $doc]
        ///
        /// Provided implementations allocate temporary GPU scratch and return after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// The mask must have the same ROI size as the source image.
        /// Each result or coordinate view must contain exactly one element per processed channel.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// A mask with a different ROI size returns an error.
        /// Result views must have the exact lengths required by the selected native operation.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the selected native routine.
        pub unsafe fn $name<T: $trait_name>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C1>,
            mask: &ImageView<'_, u8, C1>,
            output: &mut DeviceOutputView<'_, f64>,
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source, mask, output)
            }
}
    };
}

macro_rules! impl_generic_statistic_masked_c3 {
    ($doc:literal, $trait_name:ident, $name:ident, $buffer_size_name:ident, [$(($ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        #[doc = concat!("Native element and result dispatch for [`", stringify!($name), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must match the native element, layout, and result ABI.
        /// They must preserve the access, output-length, and completion contracts of `dispatch`.
        /// Buffer-size queries must return the required capacity in bytes.
        pub unsafe trait $trait_name: DataTypeLike {
            /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if NPP rejects the ROI or scratch-size query,
            /// or the returned size cannot be represented as `usize`.
            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            #[doc = $doc]
            ///
            /// Provided implementations allocate temporary GPU scratch and return after successful stream completion.
            /// This scratch-owning form rejects stream capture.
            /// The mask must have the same ROI size as the source image.
            /// `channel` is zero-based and must be less than 3.
            /// Each result or coordinate view must contain exactly one element per processed channel.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
            /// A mask with a different ROI size returns an error.
            /// An out-of-range `channel` returns an error.
            /// Result views must have the exact lengths required by the selected native operation.
            /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
            /// NPP operation errors are propagated.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
            /// Result storage must be exclusively writable and disjoint from all input and other result ranges
            /// until this call completes its native work.
            /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the selected native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C3>,
                mask: &ImageView<'_, u8, C1>,
                channel: usize,
                output: &mut DeviceOutputView<'_, f64>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            unsafe impl $trait_name for $ty {
                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C3>,
                    mask: &ImageView<'_, u8, C1>,
                    channel: usize,
                    output: &mut DeviceOutputView<'_, f64>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, mask, channel, output)
                    }
}
            }
        )+

        /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if NPP rejects the ROI or scratch-size query,
        /// or the returned size cannot be represented as `usize`.
        pub fn $buffer_size_name<T: $trait_name>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        #[doc = $doc]
        ///
        /// Provided implementations allocate temporary GPU scratch and return after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// The mask must have the same ROI size as the source image.
        /// `channel` is zero-based and must be less than 3.
        /// Each result or coordinate view must contain exactly one element per processed channel.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// A mask with a different ROI size returns an error.
        /// An out-of-range `channel` returns an error.
        /// Result views must have the exact lengths required by the selected native operation.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the selected native routine.
        pub unsafe fn $name<T: $trait_name>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C3>,
            mask: &ImageView<'_, u8, C1>,
            channel: usize,
            output: &mut DeviceOutputView<'_, f64>,
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source, mask, channel, output)
            }
}
    };
}

macro_rules! impl_generic_mean_stddev_c1 {
    ($doc:literal, $trait_name:ident, $name:ident, $buffer_size_name:ident, [$(($ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        #[doc = concat!("Native element and result dispatch for [`", stringify!($name), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must match the native element, layout, and result ABI.
        /// They must preserve the access, output-length, and completion contracts of `dispatch`.
        /// Buffer-size queries must return the required capacity in bytes.
        pub unsafe trait $trait_name: DataTypeLike {
            /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if NPP rejects the ROI or scratch-size query,
            /// or the returned size cannot be represented as `usize`.
            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            #[doc = $doc]
            ///
            /// Provided implementations allocate temporary GPU scratch and return after successful stream completion.
            /// This scratch-owning form rejects stream capture.
            /// Each result or coordinate view must contain exactly one element per processed channel.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
            /// Result views must have the exact lengths required by the selected native operation.
            /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
            /// NPP operation errors are propagated.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
            /// Result storage must be exclusively writable and disjoint from all input and other result ranges
            /// until this call completes its native work.
            /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the selected native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                mean: &mut DeviceOutputView<'_, f64>,
                standard_deviation: &mut DeviceOutputView<'_, f64>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            unsafe impl $trait_name for $ty {
                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C1>,
                    mean: &mut DeviceOutputView<'_, f64>,
                    standard_deviation: &mut DeviceOutputView<'_, f64>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, mean, standard_deviation)
                    }
}
            }
        )+

        /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if NPP rejects the ROI or scratch-size query,
        /// or the returned size cannot be represented as `usize`.
        pub fn $buffer_size_name<T: $trait_name>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        #[doc = $doc]
        ///
        /// Provided implementations allocate temporary GPU scratch and return after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// Each result or coordinate view must contain exactly one element per processed channel.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// Result views must have the exact lengths required by the selected native operation.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the selected native routine.
        pub unsafe fn $name<T: $trait_name>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C1>,
            mean: &mut DeviceOutputView<'_, f64>,
            standard_deviation: &mut DeviceOutputView<'_, f64>,
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source, mean, standard_deviation)
            }
}
    };
}

macro_rules! impl_generic_mean_stddev_masked_c1 {
    ($doc:literal, $trait_name:ident, $name:ident, $buffer_size_name:ident, [$(($ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        #[doc = concat!("Native element and result dispatch for [`", stringify!($name), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must match the native element, layout, and result ABI.
        /// They must preserve the access, output-length, and completion contracts of `dispatch`.
        /// Buffer-size queries must return the required capacity in bytes.
        pub unsafe trait $trait_name: DataTypeLike {
            /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if NPP rejects the ROI or scratch-size query,
            /// or the returned size cannot be represented as `usize`.
            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            #[doc = $doc]
            ///
            /// Provided implementations allocate temporary GPU scratch and return after successful stream completion.
            /// This scratch-owning form rejects stream capture.
            /// The mask must have the same ROI size as the source image.
            /// Each result or coordinate view must contain exactly one element per processed channel.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
            /// A mask with a different ROI size returns an error.
            /// Result views must have the exact lengths required by the selected native operation.
            /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
            /// NPP operation errors are propagated.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
            /// Result storage must be exclusively writable and disjoint from all input and other result ranges
            /// until this call completes its native work.
            /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the selected native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                mask: &ImageView<'_, u8, C1>,
                mean: &mut DeviceOutputView<'_, f64>,
                standard_deviation: &mut DeviceOutputView<'_, f64>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            unsafe impl $trait_name for $ty {
                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C1>,
                    mask: &ImageView<'_, u8, C1>,
                    mean: &mut DeviceOutputView<'_, f64>,
                    standard_deviation: &mut DeviceOutputView<'_, f64>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, mask, mean, standard_deviation)
                    }
}
            }
        )+

        /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if NPP rejects the ROI or scratch-size query,
        /// or the returned size cannot be represented as `usize`.
        pub fn $buffer_size_name<T: $trait_name>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        #[doc = $doc]
        ///
        /// Provided implementations allocate temporary GPU scratch and return after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// The mask must have the same ROI size as the source image.
        /// Each result or coordinate view must contain exactly one element per processed channel.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// A mask with a different ROI size returns an error.
        /// Result views must have the exact lengths required by the selected native operation.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the selected native routine.
        pub unsafe fn $name<T: $trait_name>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C1>,
            mask: &ImageView<'_, u8, C1>,
            mean: &mut DeviceOutputView<'_, f64>,
            standard_deviation: &mut DeviceOutputView<'_, f64>,
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source, mask, mean, standard_deviation)
            }
}
    };
}

macro_rules! impl_generic_dot_prod {
    ($doc:literal, $trait_name:ident, $name:ident, $buffer_size_name:ident, $layout:ty, [$(($ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        #[doc = concat!("Native element and result dispatch for [`", stringify!($name), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must match the native element, layout, and result ABI.
        /// They must preserve the access, output-length, and completion contracts of `dispatch`.
        /// Buffer-size queries must return the required capacity in bytes.
        pub unsafe trait $trait_name<Layout>: DataTypeLike {
            /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if NPP rejects the ROI or scratch-size query,
            /// or the returned size cannot be represented as `usize`.
            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            #[doc = $doc]
            ///
            /// Provided implementations allocate temporary GPU scratch and return after successful stream completion.
            /// This scratch-owning form rejects stream capture.
            /// Each result or coordinate view must contain exactly one element per processed channel.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
            /// Participating image ROI sizes must match.
            /// Result views must have the exact lengths required by the selected native operation.
            /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
            /// NPP operation errors are propagated.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
            /// Result storage must be exclusively writable and disjoint from all input and other result ranges
            /// until this call completes its native work.
            /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the selected native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, Self, Layout>,
                source_2: &ImageView<'_, Self, Layout>,
                output: &mut DeviceOutputView<'_, f64>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            unsafe impl $trait_name<$layout> for $ty {
                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source_1: &ImageView<'_, Self, $layout>,
                    source_2: &ImageView<'_, Self, $layout>,
                    output: &mut DeviceOutputView<'_, f64>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source_1, source_2, output)
                    }
}
            }
        )+

        /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if NPP rejects the ROI or scratch-size query,
        /// or the returned size cannot be represented as `usize`.
        pub fn $buffer_size_name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        #[doc = $doc]
        ///
        /// Provided implementations allocate temporary GPU scratch and return after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// Each result or coordinate view must contain exactly one element per processed channel.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// Participating image ROI sizes must match.
        /// Result views must have the exact lengths required by the selected native operation.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the selected native routine.
        pub unsafe fn $name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            source_1: &ImageView<'_, T, $layout>,
            source_2: &ImageView<'_, T, $layout>,
            output: &mut DeviceOutputView<'_, f64>,
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source_1, source_2, output)
            }
}
    };
}

macro_rules! impl_generic_quality_index {
    ($doc:literal, $trait_name:ident, $name:ident, $buffer_size_name:ident, $layout:ty, [$(($ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        #[doc = concat!("Native element and result dispatch for [`", stringify!($name), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must match the native element, layout, and result ABI.
        /// They must preserve the access, output-length, and completion contracts of `dispatch`.
        /// Buffer-size queries must return the required capacity in bytes.
        pub unsafe trait $trait_name<Layout>: DataTypeLike {
            /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if NPP rejects the ROI or scratch-size query,
            /// or the returned size cannot be represented as `usize`.
            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            #[doc = $doc]
            ///
            /// Provided implementations allocate temporary GPU scratch and return after successful stream completion.
            /// This scratch-owning form rejects stream capture.
            /// Each result or coordinate view must contain exactly one element per processed channel.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
            /// Participating image ROI sizes must match.
            /// Result views must have the exact lengths required by the selected native operation.
            /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
            /// NPP operation errors are propagated.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
            /// Result storage must be exclusively writable and disjoint from all input and other result ranges
            /// until this call completes its native work.
            /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the selected native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, Self, Layout>,
                source_2: &ImageView<'_, Self, Layout>,
                output: &mut DeviceOutputView<'_, f32>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            unsafe impl $trait_name<$layout> for $ty {
                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source_1: &ImageView<'_, Self, $layout>,
                    source_2: &ImageView<'_, Self, $layout>,
                    output: &mut DeviceOutputView<'_, f32>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source_1, source_2, output)
                    }
}
            }
        )+

        /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if NPP rejects the ROI or scratch-size query,
        /// or the returned size cannot be represented as `usize`.
        pub fn $buffer_size_name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        #[doc = $doc]
        ///
        /// Provided implementations allocate temporary GPU scratch and return after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// Each result or coordinate view must contain exactly one element per processed channel.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// Participating image ROI sizes must match.
        /// Result views must have the exact lengths required by the selected native operation.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the selected native routine.
        pub unsafe fn $name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            source_1: &ImageView<'_, T, $layout>,
            source_2: &ImageView<'_, T, $layout>,
            output: &mut DeviceOutputView<'_, f32>,
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source_1, source_2, output)
            }
}
    };
}

macro_rules! impl_generic_error_metric_masked_c1 {
    ($doc:literal, $trait_name:ident, $name:ident, $buffer_size_name:ident, [$(($ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        #[doc = concat!("Native element and result dispatch for [`", stringify!($name), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must match the native element, layout, and result ABI.
        /// They must preserve the access, output-length, and completion contracts of `dispatch`.
        /// Buffer-size queries must return the required capacity in bytes.
        pub unsafe trait $trait_name: DataTypeLike {
            /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if NPP rejects the ROI or scratch-size query,
            /// or the returned size cannot be represented as `usize`.
            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            #[doc = $doc]
            ///
            /// Provided implementations allocate temporary GPU scratch and return after successful stream completion.
            /// This scratch-owning form rejects stream capture.
            /// The mask must have the same ROI size as the source image.
            /// Each result or coordinate view must contain exactly one element per processed channel.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
            /// Participating image ROI sizes must match.
            /// A mask with a different ROI size returns an error.
            /// Result views must have the exact lengths required by the selected native operation.
            /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
            /// NPP operation errors are propagated.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
            /// Result storage must be exclusively writable and disjoint from all input and other result ranges
            /// until this call completes its native work.
            /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the selected native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, Self, C1>,
                source_2: &ImageView<'_, Self, C1>,
                mask: &ImageView<'_, u8, C1>,
                output: &mut DeviceOutputView<'_, f64>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            unsafe impl $trait_name for $ty {
                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source_1: &ImageView<'_, Self, C1>,
                    source_2: &ImageView<'_, Self, C1>,
                    mask: &ImageView<'_, u8, C1>,
                    output: &mut DeviceOutputView<'_, f64>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source_1, source_2, mask, output)
                    }
}
            }
        )+

        /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if NPP rejects the ROI or scratch-size query,
        /// or the returned size cannot be represented as `usize`.
        pub fn $buffer_size_name<T: $trait_name>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        #[doc = $doc]
        ///
        /// Provided implementations allocate temporary GPU scratch and return after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// The mask must have the same ROI size as the source image.
        /// Each result or coordinate view must contain exactly one element per processed channel.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// Participating image ROI sizes must match.
        /// A mask with a different ROI size returns an error.
        /// Result views must have the exact lengths required by the selected native operation.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the selected native routine.
        pub unsafe fn $name<T: $trait_name>(
            stream_context: &StreamContext,
            source_1: &ImageView<'_, T, C1>,
            source_2: &ImageView<'_, T, C1>,
            mask: &ImageView<'_, u8, C1>,
            output: &mut DeviceOutputView<'_, f64>,
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source_1, source_2, mask, output)
            }
}
    };
}

macro_rules! impl_generic_error_metric_masked_c3 {
    ($doc:literal, $trait_name:ident, $name:ident, $buffer_size_name:ident, [$(($ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        #[doc = concat!("Native element and result dispatch for [`", stringify!($name), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must match the native element, layout, and result ABI.
        /// They must preserve the access, output-length, and completion contracts of `dispatch`.
        /// Buffer-size queries must return the required capacity in bytes.
        pub unsafe trait $trait_name: DataTypeLike {
            /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if NPP rejects the ROI or scratch-size query,
            /// or the returned size cannot be represented as `usize`.
            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            #[doc = $doc]
            ///
            /// Provided implementations allocate temporary GPU scratch and return after successful stream completion.
            /// This scratch-owning form rejects stream capture.
            /// The mask must have the same ROI size as the source image.
            /// `channel` is zero-based and must be less than 3.
            /// Each result or coordinate view must contain exactly one element per processed channel.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
            /// Participating image ROI sizes must match.
            /// A mask with a different ROI size returns an error.
            /// An out-of-range `channel` returns an error.
            /// Result views must have the exact lengths required by the selected native operation.
            /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
            /// NPP operation errors are propagated.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
            /// Result storage must be exclusively writable and disjoint from all input and other result ranges
            /// until this call completes its native work.
            /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the selected native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, Self, C3>,
                source_2: &ImageView<'_, Self, C3>,
                mask: &ImageView<'_, u8, C1>,
                channel: usize,
                output: &mut DeviceOutputView<'_, f64>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            unsafe impl $trait_name for $ty {
                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source_1: &ImageView<'_, Self, C3>,
                    source_2: &ImageView<'_, Self, C3>,
                    mask: &ImageView<'_, u8, C1>,
                    channel: usize,
                    output: &mut DeviceOutputView<'_, f64>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source_1, source_2, mask, channel, output)
                    }
}
            }
        )+

        /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if NPP rejects the ROI or scratch-size query,
        /// or the returned size cannot be represented as `usize`.
        pub fn $buffer_size_name<T: $trait_name>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        #[doc = $doc]
        ///
        /// Provided implementations allocate temporary GPU scratch and return after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// The mask must have the same ROI size as the source image.
        /// `channel` is zero-based and must be less than 3.
        /// Each result or coordinate view must contain exactly one element per processed channel.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// Participating image ROI sizes must match.
        /// A mask with a different ROI size returns an error.
        /// An out-of-range `channel` returns an error.
        /// Result views must have the exact lengths required by the selected native operation.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the selected native routine.
        pub unsafe fn $name<T: $trait_name>(
            stream_context: &StreamContext,
            source_1: &ImageView<'_, T, C3>,
            source_2: &ImageView<'_, T, C3>,
            mask: &ImageView<'_, u8, C1>,
            channel: usize,
            output: &mut DeviceOutputView<'_, f64>,
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source_1, source_2, mask, channel, output)
            }
}
    };
}

macro_rules! impl_generic_every_statistic_in_place {
    ($doc:literal, $trait_name:ident, $name:ident, $layout:ty, [$(($ty:ty, $direct:ident)),+ $(,)?]) => {
        #[doc = concat!("Native element and result dispatch for [`", stringify!($name), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must match the native element, layout, and result ABI.
        /// They must preserve the access, output-length, and completion contracts of `dispatch`.
        pub unsafe trait $trait_name<Layout>: DataTypeLike {
            #[doc = $doc]
            ///
            /// Provided implementations enqueue work on the selected stream.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
            /// Participating image ROI sizes must match.
            /// NPP operation errors are propagated.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Both images must contain initialized values ready for the selected stream.
            /// Keep the source read-only and the destination exclusively writable and disjoint
            /// through queued completion, including submission errors and captured graph replay.
            /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the selected native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            unsafe impl $trait_name<$layout> for $ty {
                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination)
                    }
}
            }
        )+

        #[doc = $doc]
        ///
        /// Provided implementations enqueue work on the selected stream.
        ///
        /// # Errors
        ///
        /// Returns an error from the selected implementation if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// Participating image ROI sizes must match.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Both images must contain initialized values ready for the selected stream.
        /// Keep the source read-only and the destination exclusively writable and disjoint
        /// through queued completion, including submission errors and captured graph replay.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the selected native routine.
        pub unsafe fn $name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T, $layout>,
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source, destination)
            }
}
    };
}
