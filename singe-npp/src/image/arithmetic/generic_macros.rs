macro_rules! impl_generic_binary_operation {
    (
        $doc:literal,
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        #[doc = concat!("Native type and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select the NPP routine for a concrete pixel type and channel layout.
        ///
        /// # Safety
        ///
        /// Implementations must match the native pixel and layout ABI, preserve operand order,
        /// and uphold the method's initialization, capacity, lifetime, and exclusivity contract.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
            /// All participating image sizes must match; differing sizes return an error.
            /// Success reports native submission.
            ///
            /// # Safety
            ///
            /// Input pixels must be initialized and ready when the stream reaches this operation.
            /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
            /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
            /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
            /// Padding and channels left untouched by the operation are not initialized by it.
            unsafe fn $method(
                stream_context: &StreamContext,
                source1: &ImageView<'_, Self, Layout>,
                source2: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
            ) -> Result<()>;
        }

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the pixel type and channel layout.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
        /// All participating image sizes must match; differing sizes return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
        /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
        /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
        /// Padding and channels left untouched by the operation are not initialized by it.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source1: &ImageView<'_, T, $layout>,
            source2: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source1, source2, destination)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source1: &ImageView<'_, Self, $layout>,
                    source2: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source1, source2, destination)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_binary_operation_in_place {
    (
        $doc:literal,
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        #[doc = concat!("Native type and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select the NPP routine for a concrete pixel type and channel layout.
        ///
        /// # Safety
        ///
        /// Implementations must match the native pixel and layout ABI, preserve operand order,
        /// and uphold the method's initialization, capacity, lifetime, and exclusivity contract.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
            /// All participating image sizes must match; differing sizes return an error.
            /// Success reports native submission.
            ///
            /// # Safety
            ///
            /// Input pixels must be initialized and ready when the stream reaches this operation.
            /// Keep the destination image exclusively writable through completion.
            /// Any separate source image must remain read-only and disjoint from the writable image through completion.
            /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
            /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
            /// Padding and channels left untouched by the operation are not initialized by it.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                source_destination: &mut ImageViewMut<'_, Self, Layout>,
            ) -> Result<()>;
        }

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the pixel type and channel layout.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
        /// All participating image sizes must match; differing sizes return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep the destination image exclusively writable through completion.
        /// Any separate source image must remain read-only and disjoint from the writable image through completion.
        /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
        /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
        /// Padding and channels left untouched by the operation are not initialized by it.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            source_destination: &mut ImageViewMut<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, source_destination)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    source_destination: &mut ImageViewMut<'_, Self, $layout>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, source_destination)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_scaled_binary_operation {
    (
        $doc:literal,
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        #[doc = concat!("Native type and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select the NPP routine for a concrete pixel type and channel layout.
        ///
        /// # Safety
        ///
        /// Implementations must match the native pixel and layout ABI, preserve operand order,
        /// and uphold the method's initialization, capacity, lifetime, and exclusivity contract.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
            /// All participating image sizes must match; differing sizes return an error.
            /// Success reports native submission.
            ///
            /// # Safety
            ///
            /// Input pixels must be initialized and ready when the stream reaches this operation.
            /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
            /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
            /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
            /// Padding and channels left untouched by the operation are not initialized by it.
            unsafe fn $method(
                stream_context: &StreamContext,
                source1: &ImageView<'_, Self, Layout>,
                source2: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                scale_factor: i32,
            ) -> Result<()>;
        }

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the pixel type and channel layout.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
        /// All participating image sizes must match; differing sizes return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
        /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
        /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
        /// Padding and channels left untouched by the operation are not initialized by it.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source1: &ImageView<'_, T, $layout>,
            source2: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, T, $layout>,
            scale_factor: i32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source1, source2, destination, scale_factor)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source1: &ImageView<'_, Self, $layout>,
                    source2: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    scale_factor: i32,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source1, source2, destination, scale_factor)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_scaled_binary_operation_in_place {
    (
        $doc:literal,
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        #[doc = concat!("Native type and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select the NPP routine for a concrete pixel type and channel layout.
        ///
        /// # Safety
        ///
        /// Implementations must match the native pixel and layout ABI, preserve operand order,
        /// and uphold the method's initialization, capacity, lifetime, and exclusivity contract.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
            /// All participating image sizes must match; differing sizes return an error.
            /// Success reports native submission.
            ///
            /// # Safety
            ///
            /// Input pixels must be initialized and ready when the stream reaches this operation.
            /// Keep the destination image exclusively writable through completion.
            /// Any separate source image must remain read-only and disjoint from the writable image through completion.
            /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
            /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
            /// Padding and channels left untouched by the operation are not initialized by it.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                source_destination: &mut ImageViewMut<'_, Self, Layout>,
                scale_factor: i32,
            ) -> Result<()>;
        }

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the pixel type and channel layout.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
        /// All participating image sizes must match; differing sizes return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep the destination image exclusively writable through completion.
        /// Any separate source image must remain read-only and disjoint from the writable image through completion.
        /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
        /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
        /// Padding and channels left untouched by the operation are not initialized by it.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            source_destination: &mut ImageViewMut<'_, T, $layout>,
            scale_factor: i32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, source_destination, scale_factor)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    source_destination: &mut ImageViewMut<'_, Self, $layout>,
                    scale_factor: i32,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, source_destination, scale_factor)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_round_scaled_binary_operation {
    (
        $doc:literal,
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        #[doc = concat!("Native type and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select the NPP routine for a concrete pixel type and channel layout.
        ///
        /// # Safety
        ///
        /// Implementations must match the native pixel and layout ABI, preserve operand order,
        /// and uphold the method's initialization, capacity, lifetime, and exclusivity contract.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
            /// All participating image sizes must match; differing sizes return an error.
            /// Success reports native submission.
            ///
            /// # Safety
            ///
            /// Input pixels must be initialized and ready when the stream reaches this operation.
            /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
            /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
            /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
            /// Padding and channels left untouched by the operation are not initialized by it.
            unsafe fn $method(
                stream_context: &StreamContext,
                source1: &ImageView<'_, Self, Layout>,
                source2: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                scale_factor: i32,
                round_mode: RoundMode,
            ) -> Result<()>;
        }

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the pixel type and channel layout.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
        /// All participating image sizes must match; differing sizes return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
        /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
        /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
        /// Padding and channels left untouched by the operation are not initialized by it.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source1: &ImageView<'_, T, $layout>,
            source2: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, T, $layout>,
            scale_factor: i32,
            round_mode: RoundMode,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(
                    stream_context,
                    source1,
                    source2,
                    destination,
                    scale_factor,
                    round_mode,
                )
            }
        }

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source1: &ImageView<'_, Self, $layout>,
                    source2: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    scale_factor: i32,
                    round_mode: RoundMode,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(
                            stream_context,
                            source1,
                            source2,
                            destination,
                            scale_factor,
                            round_mode,
                        )
                    }
                }
            }
        )*
    };
}

macro_rules! impl_generic_round_scaled_binary_operation_in_place {
    (
        $doc:literal,
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        #[doc = concat!("Native type and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select the NPP routine for a concrete pixel type and channel layout.
        ///
        /// # Safety
        ///
        /// Implementations must match the native pixel and layout ABI, preserve operand order,
        /// and uphold the method's initialization, capacity, lifetime, and exclusivity contract.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
            /// All participating image sizes must match; differing sizes return an error.
            /// Success reports native submission.
            ///
            /// # Safety
            ///
            /// Input pixels must be initialized and ready when the stream reaches this operation.
            /// Keep the destination image exclusively writable through completion.
            /// Any separate source image must remain read-only and disjoint from the writable image through completion.
            /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
            /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
            /// Padding and channels left untouched by the operation are not initialized by it.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                source_destination: &mut ImageViewMut<'_, Self, Layout>,
                scale_factor: i32,
                round_mode: RoundMode,
            ) -> Result<()>;
        }

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the pixel type and channel layout.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
        /// All participating image sizes must match; differing sizes return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep the destination image exclusively writable through completion.
        /// Any separate source image must remain read-only and disjoint from the writable image through completion.
        /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
        /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
        /// Padding and channels left untouched by the operation are not initialized by it.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            source_destination: &mut ImageViewMut<'_, T, $layout>,
            scale_factor: i32,
            round_mode: RoundMode,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(
                    stream_context,
                    source,
                    source_destination,
                    scale_factor,
                    round_mode,
                )
            }
        }

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    source_destination: &mut ImageViewMut<'_, Self, $layout>,
                    scale_factor: i32,
                    round_mode: RoundMode,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(
                            stream_context,
                            source,
                            source_destination,
                            scale_factor,
                            round_mode,
                        )
                    }
                }
            }
        )*
    };
}

macro_rules! impl_generic_unary_operation {
    (
        $doc:literal,
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        #[doc = concat!("Native type and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select the NPP routine for a concrete pixel type and channel layout.
        ///
        /// # Safety
        ///
        /// Implementations must match the native pixel and layout ABI, preserve operand order,
        /// and uphold the method's initialization, capacity, lifetime, and exclusivity contract.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
            /// All participating image sizes must match; differing sizes return an error.
            /// Success reports native submission.
            ///
            /// # Safety
            ///
            /// Input pixels must be initialized and ready when the stream reaches this operation.
            /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
            /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
            /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
            /// Padding and channels left untouched by the operation are not initialized by it.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
            ) -> Result<()>;
        }

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the pixel type and channel layout.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
        /// All participating image sizes must match; differing sizes return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
        /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
        /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
        /// Padding and channels left untouched by the operation are not initialized by it.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, destination)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
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

macro_rules! impl_generic_unary_operation_in_place {
    (
        $doc:literal,
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        #[doc = concat!("Native type and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select the NPP routine for a concrete pixel type and channel layout.
        ///
        /// # Safety
        ///
        /// Implementations must match the native pixel and layout ABI, preserve operand order,
        /// and uphold the method's initialization, capacity, lifetime, and exclusivity contract.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
            /// Success reports native submission.
            ///
            /// # Safety
            ///
            /// Input pixels must be initialized and ready when the stream reaches this operation.
            /// Keep the destination image exclusively writable through completion.
            /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
            /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
            /// Padding and channels left untouched by the operation are not initialized by it.
            unsafe fn $method(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, Self, Layout>,
            ) -> Result<()>;
        }

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the pixel type and channel layout.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep the destination image exclusively writable through completion.
        /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
        /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
        /// Padding and channels left untouched by the operation are not initialized by it.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source_destination: &mut ImageViewMut<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source_destination)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source_destination: &mut ImageViewMut<'_, Self, $layout>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source_destination)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_scaled_unary_operation {
    (
        $doc:literal,
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        #[doc = concat!("Native type and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select the NPP routine for a concrete pixel type and channel layout.
        ///
        /// # Safety
        ///
        /// Implementations must match the native pixel and layout ABI, preserve operand order,
        /// and uphold the method's initialization, capacity, lifetime, and exclusivity contract.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
            /// All participating image sizes must match; differing sizes return an error.
            /// Success reports native submission.
            ///
            /// # Safety
            ///
            /// Input pixels must be initialized and ready when the stream reaches this operation.
            /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
            /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
            /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
            /// Padding and channels left untouched by the operation are not initialized by it.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                scale_factor: i32,
            ) -> Result<()>;
        }

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the pixel type and channel layout.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
        /// All participating image sizes must match; differing sizes return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
        /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
        /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
        /// Padding and channels left untouched by the operation are not initialized by it.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, T, $layout>,
            scale_factor: i32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, destination, scale_factor)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    scale_factor: i32,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, scale_factor)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_scaled_unary_operation_in_place {
    (
        $doc:literal,
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        #[doc = concat!("Native type and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select the NPP routine for a concrete pixel type and channel layout.
        ///
        /// # Safety
        ///
        /// Implementations must match the native pixel and layout ABI, preserve operand order,
        /// and uphold the method's initialization, capacity, lifetime, and exclusivity contract.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
            /// Success reports native submission.
            ///
            /// # Safety
            ///
            /// Input pixels must be initialized and ready when the stream reaches this operation.
            /// Keep the destination image exclusively writable through completion.
            /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
            /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
            /// Padding and channels left untouched by the operation are not initialized by it.
            unsafe fn $method(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, Self, Layout>,
                scale_factor: i32,
            ) -> Result<()>;
        }

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the pixel type and channel layout.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep the destination image exclusively writable through completion.
        /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
        /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
        /// Padding and channels left untouched by the operation are not initialized by it.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source_destination: &mut ImageViewMut<'_, T, $layout>,
            scale_factor: i32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source_destination, scale_factor)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source_destination: &mut ImageViewMut<'_, Self, $layout>,
                    scale_factor: i32,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source_destination, scale_factor)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_alpha_comp_constant {
    (
        $doc:literal,
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        #[doc = concat!("Native type and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select the NPP routine for a concrete pixel type and channel layout.
        ///
        /// # Safety
        ///
        /// Implementations must match the native pixel and layout ABI, preserve operand order,
        /// and uphold the method's initialization, capacity, lifetime, and exclusivity contract.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
            /// All participating image sizes must match; differing sizes return an error.
            /// Success reports native submission.
            ///
            /// # Safety
            ///
            /// Input pixels must be initialized and ready when the stream reaches this operation.
            /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
            /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
            /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
            /// Padding and channels left untouched by the operation are not initialized by it.
            unsafe fn $method(
                stream_context: &StreamContext,
                source1: &ImageView<'_, Self, Layout>,
                alpha1: Self,
                source2: &ImageView<'_, Self, Layout>,
                alpha2: Self,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                operation: AlphaOperation,
            ) -> Result<()>;
        }

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the pixel type and channel layout.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
        /// All participating image sizes must match; differing sizes return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
        /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
        /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
        /// Padding and channels left untouched by the operation are not initialized by it.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source1: &ImageView<'_, T, $layout>,
            alpha1: T,
            source2: &ImageView<'_, T, $layout>,
            alpha2: T,
            destination: &mut ImageOutputView<'_, T, $layout>,
            operation: AlphaOperation,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(
                    stream_context,
                    source1,
                    alpha1,
                    source2,
                    alpha2,
                    destination,
                    operation,
                )
            }
        }

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source1: &ImageView<'_, Self, $layout>,
                    alpha1: Self,
                    source2: &ImageView<'_, Self, $layout>,
                    alpha2: Self,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    operation: AlphaOperation,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(
                            stream_context,
                            source1,
                            alpha1,
                            source2,
                            alpha2,
                            destination,
                            operation,
                        )
                    }
                }
            }
        )*
    };
}

macro_rules! impl_generic_alpha_comp {
    (
        $doc:literal,
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        #[doc = concat!("Native type and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select the NPP routine for a concrete pixel type and channel layout.
        ///
        /// # Safety
        ///
        /// Implementations must match the native pixel and layout ABI, preserve operand order,
        /// and uphold the method's initialization, capacity, lifetime, and exclusivity contract.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
            /// All participating image sizes must match; differing sizes return an error.
            /// Success reports native submission.
            ///
            /// # Safety
            ///
            /// Input pixels must be initialized and ready when the stream reaches this operation.
            /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
            /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
            /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
            /// Padding and channels left untouched by the operation are not initialized by it.
            unsafe fn $method(
                stream_context: &StreamContext,
                source1: &ImageView<'_, Self, Layout>,
                source2: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                operation: AlphaOperation,
            ) -> Result<()>;
        }

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the pixel type and channel layout.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
        /// All participating image sizes must match; differing sizes return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
        /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
        /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
        /// Padding and channels left untouched by the operation are not initialized by it.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source1: &ImageView<'_, T, $layout>,
            source2: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, T, $layout>,
            operation: AlphaOperation,
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
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    operation: AlphaOperation,
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

macro_rules! impl_generic_constant_scalar_operation {
    (
        $doc:literal,
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty, $constant_ty:ty => $direct:ident),* $(,)?]
    ) => {
        #[doc = concat!("Native type and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select the NPP routine for a concrete pixel type and channel layout.
        ///
        /// # Safety
        ///
        /// Implementations must match the native pixel and layout ABI, preserve operand order,
        /// and uphold the method's initialization, capacity, lifetime, and exclusivity contract.
        /// Associated constant types must match the operand representation expected by the native routine.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Scalar representation of the native constant operand for this pixel type and layout.
            type Constant: Copy;

            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
            /// All participating image sizes must match; differing sizes return an error.
            /// Success reports native submission.
            ///
            /// # Safety
            ///
            /// Input pixels must be initialized and ready when the stream reaches this operation.
            /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
            /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
            /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
            /// Padding and channels left untouched by the operation are not initialized by it.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                constant: Self::Constant,
                destination: &mut ImageOutputView<'_, Self, Layout>,
            ) -> Result<()>;
        }

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the pixel type and channel layout.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
        /// All participating image sizes must match; differing sizes return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
        /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
        /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
        /// Padding and channels left untouched by the operation are not initialized by it.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            constant: T::Constant,
            destination: &mut ImageOutputView<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, constant, destination)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                type Constant = $constant_ty;

                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    constant: Self::Constant,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, constant, destination)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_constant_scalar_operation_in_place {
    (
        $doc:literal,
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty, $constant_ty:ty => $direct:ident),* $(,)?]
    ) => {
        #[doc = concat!("Native type and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select the NPP routine for a concrete pixel type and channel layout.
        ///
        /// # Safety
        ///
        /// Implementations must match the native pixel and layout ABI, preserve operand order,
        /// and uphold the method's initialization, capacity, lifetime, and exclusivity contract.
        /// Associated constant types must match the operand representation expected by the native routine.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Scalar representation of the native constant operand for this pixel type and layout.
            type Constant: Copy;

            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
            /// Success reports native submission.
            ///
            /// # Safety
            ///
            /// Input pixels must be initialized and ready when the stream reaches this operation.
            /// Keep the destination image exclusively writable through completion.
            /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
            /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
            /// Padding and channels left untouched by the operation are not initialized by it.
            unsafe fn $method(
                stream_context: &StreamContext,
                constant: Self::Constant,
                source_destination: &mut ImageViewMut<'_, Self, Layout>,
            ) -> Result<()>;
        }

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the pixel type and channel layout.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep the destination image exclusively writable through completion.
        /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
        /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
        /// Padding and channels left untouched by the operation are not initialized by it.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            constant: T::Constant,
            source_destination: &mut ImageViewMut<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, constant, source_destination)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                type Constant = $constant_ty;

                unsafe fn $method(
                    stream_context: &StreamContext,
                    constant: Self::Constant,
                    source_destination: &mut ImageViewMut<'_, Self, $layout>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, constant, source_destination)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_constant_array_operation {
    (
        $doc:literal,
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        $channels:literal,
        [$($ty:ty, $constant_ty:ty => $direct:ident),* $(,)?]
    ) => {
        #[doc = concat!("Native type and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select the NPP routine for a concrete pixel type and channel layout.
        ///
        /// # Safety
        ///
        /// Implementations must match the native pixel and layout ABI, preserve operand order,
        /// and uphold the method's initialization, capacity, lifetime, and exclusivity contract.
        /// Associated constant types must match the operand representation expected by the native routine.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Scalar representation of the native constant operand for this pixel type and layout.
            type Constant: Copy;

            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
            /// All participating image sizes must match; differing sizes return an error.
            /// Success reports native submission.
            ///
            /// # Safety
            ///
            /// Input pixels must be initialized and ready when the stream reaches this operation.
            /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
            /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
            /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
            /// Padding and channels left untouched by the operation are not initialized by it.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                constants: [Self::Constant; $channels],
                destination: &mut ImageOutputView<'_, Self, Layout>,
            ) -> Result<()>;
        }

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the pixel type and channel layout.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
        /// All participating image sizes must match; differing sizes return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
        /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
        /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
        /// Padding and channels left untouched by the operation are not initialized by it.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            constants: [T::Constant; $channels],
            destination: &mut ImageOutputView<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, constants, destination)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                type Constant = $constant_ty;

                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    constants: [Self::Constant; $channels],
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, constants, destination)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_constant_array_operation_in_place {
    (
        $doc:literal,
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        $channels:literal,
        [$($ty:ty, $constant_ty:ty => $direct:ident),* $(,)?]
    ) => {
        #[doc = concat!("Native type and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select the NPP routine for a concrete pixel type and channel layout.
        ///
        /// # Safety
        ///
        /// Implementations must match the native pixel and layout ABI, preserve operand order,
        /// and uphold the method's initialization, capacity, lifetime, and exclusivity contract.
        /// Associated constant types must match the operand representation expected by the native routine.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Scalar representation of the native constant operand for this pixel type and layout.
            type Constant: Copy;

            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
            /// Success reports native submission.
            ///
            /// # Safety
            ///
            /// Input pixels must be initialized and ready when the stream reaches this operation.
            /// Keep the destination image exclusively writable through completion.
            /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
            /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
            /// Padding and channels left untouched by the operation are not initialized by it.
            unsafe fn $method(
                stream_context: &StreamContext,
                constants: [Self::Constant; $channels],
                source_destination: &mut ImageViewMut<'_, Self, Layout>,
            ) -> Result<()>;
        }

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the pixel type and channel layout.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep the destination image exclusively writable through completion.
        /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
        /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
        /// Padding and channels left untouched by the operation are not initialized by it.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            constants: [T::Constant; $channels],
            source_destination: &mut ImageViewMut<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, constants, source_destination)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                type Constant = $constant_ty;

                unsafe fn $method(
                    stream_context: &StreamContext,
                    constants: [Self::Constant; $channels],
                    source_destination: &mut ImageViewMut<'_, Self, $layout>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, constants, source_destination)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_scaled_constant_scalar_operation {
    (
        $doc:literal,
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        #[doc = concat!("Native type and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select the NPP routine for a concrete pixel type and channel layout.
        ///
        /// # Safety
        ///
        /// Implementations must match the native pixel and layout ABI, preserve operand order,
        /// and uphold the method's initialization, capacity, lifetime, and exclusivity contract.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
            /// All participating image sizes must match; differing sizes return an error.
            /// Success reports native submission.
            ///
            /// # Safety
            ///
            /// Input pixels must be initialized and ready when the stream reaches this operation.
            /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
            /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
            /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
            /// Padding and channels left untouched by the operation are not initialized by it.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                constant: Self,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                scale_factor: i32,
            ) -> Result<()>;
        }

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the pixel type and channel layout.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
        /// All participating image sizes must match; differing sizes return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
        /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
        /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
        /// Padding and channels left untouched by the operation are not initialized by it.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            constant: T,
            destination: &mut ImageOutputView<'_, T, $layout>,
            scale_factor: i32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, constant, destination, scale_factor)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    constant: Self,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    scale_factor: i32,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, constant, destination, scale_factor)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_scaled_constant_scalar_operation_in_place {
    (
        $doc:literal,
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        #[doc = concat!("Native type and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select the NPP routine for a concrete pixel type and channel layout.
        ///
        /// # Safety
        ///
        /// Implementations must match the native pixel and layout ABI, preserve operand order,
        /// and uphold the method's initialization, capacity, lifetime, and exclusivity contract.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
            /// Success reports native submission.
            ///
            /// # Safety
            ///
            /// Input pixels must be initialized and ready when the stream reaches this operation.
            /// Keep the destination image exclusively writable through completion.
            /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
            /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
            /// Padding and channels left untouched by the operation are not initialized by it.
            unsafe fn $method(
                stream_context: &StreamContext,
                constant: Self,
                source_destination: &mut ImageViewMut<'_, Self, Layout>,
                scale_factor: i32,
            ) -> Result<()>;
        }

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the pixel type and channel layout.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep the destination image exclusively writable through completion.
        /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
        /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
        /// Padding and channels left untouched by the operation are not initialized by it.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            constant: T,
            source_destination: &mut ImageViewMut<'_, T, $layout>,
            scale_factor: i32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, constant, source_destination, scale_factor)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    constant: Self,
                    source_destination: &mut ImageViewMut<'_, Self, $layout>,
                    scale_factor: i32,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, constant, source_destination, scale_factor)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_scaled_constant_array_operation {
    (
        $doc:literal,
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        $channels:literal,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        #[doc = concat!("Native type and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select the NPP routine for a concrete pixel type and channel layout.
        ///
        /// # Safety
        ///
        /// Implementations must match the native pixel and layout ABI, preserve operand order,
        /// and uphold the method's initialization, capacity, lifetime, and exclusivity contract.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
            /// All participating image sizes must match; differing sizes return an error.
            /// Success reports native submission.
            ///
            /// # Safety
            ///
            /// Input pixels must be initialized and ready when the stream reaches this operation.
            /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
            /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
            /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
            /// Padding and channels left untouched by the operation are not initialized by it.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                constants: [Self; $channels],
                destination: &mut ImageOutputView<'_, Self, Layout>,
                scale_factor: i32,
            ) -> Result<()>;
        }

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the pixel type and channel layout.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
        /// All participating image sizes must match; differing sizes return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
        /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
        /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
        /// Padding and channels left untouched by the operation are not initialized by it.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            constants: [T; $channels],
            destination: &mut ImageOutputView<'_, T, $layout>,
            scale_factor: i32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, constants, destination, scale_factor)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    constants: [Self; $channels],
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    scale_factor: i32,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, constants, destination, scale_factor)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_scaled_constant_array_operation_in_place {
    (
        $doc:literal,
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        $channels:literal,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        #[doc = concat!("Native type and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select the NPP routine for a concrete pixel type and channel layout.
        ///
        /// # Safety
        ///
        /// Implementations must match the native pixel and layout ABI, preserve operand order,
        /// and uphold the method's initialization, capacity, lifetime, and exclusivity contract.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
            /// Success reports native submission.
            ///
            /// # Safety
            ///
            /// Input pixels must be initialized and ready when the stream reaches this operation.
            /// Keep the destination image exclusively writable through completion.
            /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
            /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
            /// Padding and channels left untouched by the operation are not initialized by it.
            unsafe fn $method(
                stream_context: &StreamContext,
                constants: [Self; $channels],
                source_destination: &mut ImageViewMut<'_, Self, Layout>,
                scale_factor: i32,
            ) -> Result<()>;
        }

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the pixel type and channel layout.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, an image belongs to another CUDA context, or NPP reports a failure.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep the destination image exclusively writable through completion.
        /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
        /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
        /// Padding and channels left untouched by the operation are not initialized by it.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            constants: [T; $channels],
            source_destination: &mut ImageViewMut<'_, T, $layout>,
            scale_factor: i32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, constants, source_destination, scale_factor)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    constants: [Self; $channels],
                    source_destination: &mut ImageViewMut<'_, Self, $layout>,
                    scale_factor: i32,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, constants, source_destination, scale_factor)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_device_constant_operation {
    (
        $doc:literal,
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty, $constant_ty:ty => $direct:ident),* $(,)?]
    ) => {
        #[doc = concat!("Native type and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select the NPP routine for a concrete pixel type and channel layout.
        ///
        /// # Safety
        ///
        /// Implementations must match the native pixel and layout ABI, preserve operand order,
        /// and uphold the method's initialization, capacity, lifetime, and exclusivity contract.
        /// Associated constant types must match the operand representation expected by the native routine.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Scalar representation of the native constant operand for this pixel type and layout.
            type Constant: Copy + 'static;

            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, an image or constant buffer belongs to another CUDA context, the device-constant buffer is too short, or NPP reports a failure.
            /// All participating image sizes must match; differing sizes return an error.
            /// Success reports native submission.
            ///
            /// # Safety
            ///
            /// Input pixels must be initialized and ready when the stream reaches this operation.
            /// Device constants must also be initialized and ready, and remain read-only through completion.
            /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
            /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
            /// Retain images, device constants, and the stream context through queued completion, including submission errors and every captured graph replay.
            /// Padding and channels left untouched by the operation are not initialized by it.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                constants: &DeviceMemory<Self::Constant>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
            ) -> Result<()>;
        }

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the pixel type and channel layout.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, an image or constant buffer belongs to another CUDA context, the device-constant buffer is too short, or NPP reports a failure.
        /// All participating image sizes must match; differing sizes return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Device constants must also be initialized and ready, and remain read-only through completion.
        /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
        /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
        /// Retain images, device constants, and the stream context through queued completion, including submission errors and every captured graph replay.
        /// Padding and channels left untouched by the operation are not initialized by it.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            constants: &DeviceMemory<T::Constant>,
            destination: &mut ImageOutputView<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, constants, destination)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                type Constant = $constant_ty;

                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    constants: &DeviceMemory<Self::Constant>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, constants, destination)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_device_constant_operation_in_place {
    (
        $doc:literal,
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty, $constant_ty:ty => $direct:ident),* $(,)?]
    ) => {
        #[doc = concat!("Native type and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select the NPP routine for a concrete pixel type and channel layout.
        ///
        /// # Safety
        ///
        /// Implementations must match the native pixel and layout ABI, preserve operand order,
        /// and uphold the method's initialization, capacity, lifetime, and exclusivity contract.
        /// Associated constant types must match the operand representation expected by the native routine.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Scalar representation of the native constant operand for this pixel type and layout.
            type Constant: Copy + 'static;

            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, an image or constant buffer belongs to another CUDA context, the device-constant buffer is too short, or NPP reports a failure.
            /// Success reports native submission.
            ///
            /// # Safety
            ///
            /// Input pixels must be initialized and ready when the stream reaches this operation.
            /// Device constants must also be initialized and ready, and remain read-only through completion.
            /// Keep the destination image exclusively writable through completion.
            /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
            /// Retain images, device constants, and the stream context through queued completion, including submission errors and every captured graph replay.
            /// Padding and channels left untouched by the operation are not initialized by it.
            unsafe fn $method(
                stream_context: &StreamContext,
                constants: &DeviceMemory<Self::Constant>,
                source_destination: &mut ImageViewMut<'_, Self, Layout>,
            ) -> Result<()>;
        }

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the pixel type and channel layout.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, an image or constant buffer belongs to another CUDA context, the device-constant buffer is too short, or NPP reports a failure.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Device constants must also be initialized and ready, and remain read-only through completion.
        /// Keep the destination image exclusively writable through completion.
        /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
        /// Retain images, device constants, and the stream context through queued completion, including submission errors and every captured graph replay.
        /// Padding and channels left untouched by the operation are not initialized by it.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            constants: &DeviceMemory<T::Constant>,
            source_destination: &mut ImageViewMut<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, constants, source_destination)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                type Constant = $constant_ty;

                unsafe fn $method(
                    stream_context: &StreamContext,
                    constants: &DeviceMemory<Self::Constant>,
                    source_destination: &mut ImageViewMut<'_, Self, $layout>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, constants, source_destination)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_scaled_device_constant_operation {
    (
        $doc:literal,
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        #[doc = concat!("Native type and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select the NPP routine for a concrete pixel type and channel layout.
        ///
        /// # Safety
        ///
        /// Implementations must match the native pixel and layout ABI, preserve operand order,
        /// and uphold the method's initialization, capacity, lifetime, and exclusivity contract.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, an image or constant buffer belongs to another CUDA context, the device-constant buffer is too short, or NPP reports a failure.
            /// All participating image sizes must match; differing sizes return an error.
            /// Success reports native submission.
            ///
            /// # Safety
            ///
            /// Input pixels must be initialized and ready when the stream reaches this operation.
            /// Device constants must also be initialized and ready, and remain read-only through completion.
            /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
            /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
            /// Retain images, device constants, and the stream context through queued completion, including submission errors and every captured graph replay.
            /// Padding and channels left untouched by the operation are not initialized by it.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                constants: &DeviceMemory<Self>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                scale_factor: i32,
            ) -> Result<()>;
        }

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the pixel type and channel layout.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, an image or constant buffer belongs to another CUDA context, the device-constant buffer is too short, or NPP reports a failure.
        /// All participating image sizes must match; differing sizes return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Device constants must also be initialized and ready, and remain read-only through completion.
        /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
        /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
        /// Retain images, device constants, and the stream context through queued completion, including submission errors and every captured graph replay.
        /// Padding and channels left untouched by the operation are not initialized by it.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            constants: &DeviceMemory<T>,
            destination: &mut ImageOutputView<'_, T, $layout>,
            scale_factor: i32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, constants, destination, scale_factor)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    constants: &DeviceMemory<Self>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    scale_factor: i32,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, constants, destination, scale_factor)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_scaled_device_constant_operation_in_place {
    (
        $doc:literal,
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        #[doc = concat!("Native type and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select the NPP routine for a concrete pixel type and channel layout.
        ///
        /// # Safety
        ///
        /// Implementations must match the native pixel and layout ABI, preserve operand order,
        /// and uphold the method's initialization, capacity, lifetime, and exclusivity contract.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, an image or constant buffer belongs to another CUDA context, the device-constant buffer is too short, or NPP reports a failure.
            /// Success reports native submission.
            ///
            /// # Safety
            ///
            /// Input pixels must be initialized and ready when the stream reaches this operation.
            /// Device constants must also be initialized and ready, and remain read-only through completion.
            /// Keep the destination image exclusively writable through completion.
            /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
            /// Retain images, device constants, and the stream context through queued completion, including submission errors and every captured graph replay.
            /// Padding and channels left untouched by the operation are not initialized by it.
            unsafe fn $method(
                stream_context: &StreamContext,
                constants: &DeviceMemory<Self>,
                source_destination: &mut ImageViewMut<'_, Self, Layout>,
                scale_factor: i32,
            ) -> Result<()>;
        }

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the pixel type and channel layout.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, an image or constant buffer belongs to another CUDA context, the device-constant buffer is too short, or NPP reports a failure.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Device constants must also be initialized and ready, and remain read-only through completion.
        /// Keep the destination image exclusively writable through completion.
        /// Addresses, row strides, image regions, and constant operands must satisfy the selected native function's alignment, capacity, and representation requirements.
        /// Retain images, device constants, and the stream context through queued completion, including submission errors and every captured graph replay.
        /// Padding and channels left untouched by the operation are not initialized by it.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            constants: &DeviceMemory<T>,
            source_destination: &mut ImageViewMut<'_, T, $layout>,
            scale_factor: i32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, constants, source_destination, scale_factor)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    constants: &DeviceMemory<Self>,
                    source_destination: &mut ImageViewMut<'_, Self, $layout>,
                    scale_factor: i32,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, constants, source_destination, scale_factor)
                    }
}
            }
        )*
    };
}
