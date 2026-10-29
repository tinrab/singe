macro_rules! impl_add_constant_scaled_scalar {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image belongs to another CUDA context.
        /// The source and destination image sizes must match; differing sizes return an error.
        /// Native NPP failures are returned as errors. Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
        /// All addresses, row strides, and image regions must satisfy the native function's alignment and capacity requirements.
        /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
        /// The operation does not initialize padding or channels it leaves untouched.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            constant: $ty,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
            scale_factor: i32,
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
                    constant.into_npp(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_add_constant_scaled_scalar_in_place {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image belongs to another CUDA context.
        /// Native NPP failures are returned as errors. Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep the image exclusively writable through completion.
        /// All addresses, row strides, and image regions must satisfy the native function's alignment and capacity requirements.
        /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
        /// The operation does not initialize padding or channels it leaves untouched.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            constant: $ty,
            source_destination: &mut ImageViewMut<'_, $ty, $layout>,
            scale_factor: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_destination)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    constant.into_npp(),
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source_destination.size().into(),
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_add_constant_scaled_array {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $channels:literal, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image belongs to another CUDA context.
        /// The source and destination image sizes must match; differing sizes return an error.
        /// Native NPP failures are returned as errors. Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
        /// All addresses, row strides, and image regions must satisfy the native function's alignment and capacity requirements.
        /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
        /// The operation does not initialize padding or channels it leaves untouched.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            constants: [$ty; $channels],
            destination: &mut ImageOutputView<'_, $ty, $layout>,
            scale_factor: i32,
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
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_add_constant_scaled_array_in_place {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $channels:literal, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image belongs to another CUDA context.
        /// Native NPP failures are returned as errors. Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep the image exclusively writable through completion.
        /// All addresses, row strides, and image regions must satisfy the native function's alignment and capacity requirements.
        /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
        /// The operation does not initialize padding or channels it leaves untouched.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            constants: [$ty; $channels],
            source_destination: &mut ImageViewMut<'_, $ty, $layout>,
            scale_factor: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_destination)?;

            // SAFETY: the `ffi` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
            unsafe {
                try_ffi!(sys::$ffi(
                    constants.as_ptr().cast(),
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source_destination.size().into(),
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_add_constant_scalar {
    ($doc:literal, $name:ident, $pixel_ty:ty, $constant_ty:ty, $layout:ty, $ffi:ident) => {
        impl_add_constant_scalar!($doc, $name, $pixel_ty, $constant_ty, $layout, $ffi, |x| x);
    };
    ($doc:literal, $name:ident, $pixel_ty:ty, $constant_ty:ty, $layout:ty, $ffi:ident, $into:expr) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image belongs to another CUDA context.
        /// The source and destination image sizes must match; differing sizes return an error.
        /// Native NPP failures are returned as errors. Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
        /// All addresses, row strides, and image regions must satisfy the native function's alignment and capacity requirements.
        /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
        /// The operation does not initialize padding or channels it leaves untouched.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            constant: $constant_ty,
            destination: &mut ImageOutputView<'_, $pixel_ty, $layout>,
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
                    $into(constant).into_npp(),
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

macro_rules! impl_add_constant_scalar_in_place {
    ($doc:literal, $name:ident, $pixel_ty:ty, $constant_ty:ty, $layout:ty, $ffi:ident) => {
        impl_add_constant_scalar_in_place!(
            $doc,
            $name,
            $pixel_ty,
            $constant_ty,
            $layout,
            $ffi,
            |x| x
        );
    };
    ($doc:literal, $name:ident, $pixel_ty:ty, $constant_ty:ty, $layout:ty, $ffi:ident, $into:expr) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image belongs to another CUDA context.
        /// Native NPP failures are returned as errors. Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep the image exclusively writable through completion.
        /// All addresses, row strides, and image regions must satisfy the native function's alignment and capacity requirements.
        /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
        /// The operation does not initialize padding or channels it leaves untouched.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            constant: $constant_ty,
            source_destination: &mut ImageViewMut<'_, $pixel_ty, $layout>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_destination)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    $into(constant).into_npp(),
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source_destination.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_add_constant_array {
    ($doc:literal, $name:ident, $pixel_ty:ty, $constant_ty:ty, $layout:ty, $channels:literal, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image belongs to another CUDA context.
        /// The source and destination image sizes must match; differing sizes return an error.
        /// Native NPP failures are returned as errors. Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
        /// All addresses, row strides, and image regions must satisfy the native function's alignment and capacity requirements.
        /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
        /// The operation does not initialize padding or channels it leaves untouched.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            constants: [$constant_ty; $channels],
            destination: &mut ImageOutputView<'_, $pixel_ty, $layout>,
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
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_add_constant_array_in_place {
    ($doc:literal, $name:ident, $pixel_ty:ty, $constant_ty:ty, $layout:ty, $channels:literal, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image belongs to another CUDA context.
        /// Native NPP failures are returned as errors. Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep the image exclusively writable through completion.
        /// All addresses, row strides, and image regions must satisfy the native function's alignment and capacity requirements.
        /// Retain images and the stream context through queued completion, including submission errors and every captured graph replay.
        /// The operation does not initialize padding or channels it leaves untouched.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            constants: [$constant_ty; $channels],
            source_destination: &mut ImageViewMut<'_, $pixel_ty, $layout>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_destination)?;

            // SAFETY: the `ffi` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
            unsafe {
                try_ffi!(sys::$ffi(
                    constants.as_ptr().cast(),
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source_destination.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_add_device_constant_scaled {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $constant_len:literal, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image or device-constant buffer belongs to another CUDA context.
        /// Also returns an error if the device-constant buffer is shorter than the required channel count.
        /// The source and destination image sizes must match; differing sizes return an error.
        /// Native NPP failures are returned as errors. Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Device constants must also be initialized and ready, and remain read-only through completion.
        /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
        /// All addresses, row strides, and image regions must satisfy the native function's alignment and capacity requirements.
        /// Retain images, device constants, and the stream context through queued completion, including submission errors and every captured graph replay.
        /// The operation does not initialize padding or channels it leaves untouched.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            constants: &DeviceMemory<$ty>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
            scale_factor: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(constants)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;
            let constants = device_constant_ptr(constants, $constant_len)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    constants.cast(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_add_device_constant_scaled_in_place {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $constant_len:literal, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image or device-constant buffer belongs to another CUDA context.
        /// Also returns an error if the device-constant buffer is shorter than the required channel count.
        /// Native NPP failures are returned as errors. Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Device constants must also be initialized and ready, and remain read-only through completion.
        /// Keep the image exclusively writable through completion.
        /// All addresses, row strides, and image regions must satisfy the native function's alignment and capacity requirements.
        /// Retain images, device constants, and the stream context through queued completion, including submission errors and every captured graph replay.
        /// The operation does not initialize padding or channels it leaves untouched.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            constants: &DeviceMemory<$ty>,
            source_destination: &mut ImageViewMut<'_, $ty, $layout>,
            scale_factor: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(constants)?;
            stream_context.validate_region(source_destination)?;

            let constants = device_constant_ptr(constants, $constant_len)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    constants.cast(),
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source_destination.size().into(),
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_add_device_constant {
    ($doc:literal, $name:ident, $pixel_ty:ty, $constant_ty:ty, $layout:ty, $constant_len:literal, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image or device-constant buffer belongs to another CUDA context.
        /// Also returns an error if the device-constant buffer is shorter than the required channel count.
        /// The source and destination image sizes must match; differing sizes return an error.
        /// Native NPP failures are returned as errors. Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Device constants must also be initialized and ready, and remain read-only through completion.
        /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
        /// All addresses, row strides, and image regions must satisfy the native function's alignment and capacity requirements.
        /// Retain images, device constants, and the stream context through queued completion, including submission errors and every captured graph replay.
        /// The operation does not initialize padding or channels it leaves untouched.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            constants: &DeviceMemory<$constant_ty>,
            destination: &mut ImageOutputView<'_, $pixel_ty, $layout>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(constants)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;
            let constants = device_constant_ptr(constants, $constant_len)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    constants.cast(),
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

macro_rules! impl_add_device_constant_in_place {
    ($doc:literal, $name:ident, $pixel_ty:ty, $constant_ty:ty, $layout:ty, $constant_len:literal, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image or device-constant buffer belongs to another CUDA context.
        /// Also returns an error if the device-constant buffer is shorter than the required channel count.
        /// Native NPP failures are returned as errors. Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Device constants must also be initialized and ready, and remain read-only through completion.
        /// Keep the image exclusively writable through completion.
        /// All addresses, row strides, and image regions must satisfy the native function's alignment and capacity requirements.
        /// Retain images, device constants, and the stream context through queued completion, including submission errors and every captured graph replay.
        /// The operation does not initialize padding or channels it leaves untouched.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            constants: &DeviceMemory<$constant_ty>,
            source_destination: &mut ImageViewMut<'_, $pixel_ty, $layout>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(constants)?;
            stream_context.validate_region(source_destination)?;

            let constants = device_constant_ptr(constants, $constant_len)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    constants.cast(),
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source_destination.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_binary_scaled {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image belongs to another CUDA context, or NPP reports a failure.
        /// All participating image sizes must match; differing sizes return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
        /// All addresses, row strides, and image regions must satisfy the native function's alignment and capacity requirements.
        /// Retain the images and stream context through queued completion, including submission errors and every captured graph replay.
        /// The operation does not initialize padding or channels it leaves untouched.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source1: &ImageView<'_, $ty, $layout>,
            source2: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
            scale_factor: i32,
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
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_binary_scaled_in_place {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image belongs to another CUDA context, or NPP reports a failure.
        /// All participating image sizes must match; differing sizes return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep the destination image exclusively writable through completion.
        /// The separate source image must remain read-only and disjoint from the writable image through completion.
        /// All addresses, row strides, and image regions must satisfy the native function's alignment and capacity requirements.
        /// Retain the images and stream context through queued completion, including submission errors and every captured graph replay.
        /// The operation does not initialize padding or channels it leaves untouched.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            source_destination: &mut ImageViewMut<'_, $ty, $layout>,
            scale_factor: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(source_destination)?;

            validate_same_size(source.size(), source_destination.size())?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source.size().into(),
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_binary_round_scaled {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image belongs to another CUDA context, or NPP reports a failure.
        /// All participating image sizes must match; differing sizes return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
        /// All addresses, row strides, and image regions must satisfy the native function's alignment and capacity requirements.
        /// Retain the images and stream context through queued completion, including submission errors and every captured graph replay.
        /// The operation does not initialize padding or channels it leaves untouched.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source1: &ImageView<'_, $ty, $layout>,
            source2: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
            scale_factor: i32,
            round_mode: RoundMode,
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
                    round_mode.into(),
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_binary_round_scaled_in_place {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image belongs to another CUDA context, or NPP reports a failure.
        /// All participating image sizes must match; differing sizes return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep the destination image exclusively writable through completion.
        /// The separate source image must remain read-only and disjoint from the writable image through completion.
        /// All addresses, row strides, and image regions must satisfy the native function's alignment and capacity requirements.
        /// Retain the images and stream context through queued completion, including submission errors and every captured graph replay.
        /// The operation does not initialize padding or channels it leaves untouched.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            source_destination: &mut ImageViewMut<'_, $ty, $layout>,
            scale_factor: i32,
            round_mode: RoundMode,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(source_destination)?;

            validate_same_size(source.size(), source_destination.size())?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source.size().into(),
                    round_mode.into(),
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_binary {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image belongs to another CUDA context, or NPP reports a failure.
        /// All participating image sizes must match; differing sizes return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
        /// All addresses, row strides, and image regions must satisfy the native function's alignment and capacity requirements.
        /// Retain the images and stream context through queued completion, including submission errors and every captured graph replay.
        /// The operation does not initialize padding or channels it leaves untouched.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source1: &ImageView<'_, $ty, $layout>,
            source2: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
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
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_binary_in_place {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image belongs to another CUDA context, or NPP reports a failure.
        /// All participating image sizes must match; differing sizes return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep the destination image exclusively writable through completion.
        /// The separate source image must remain read-only and disjoint from the writable image through completion.
        /// All addresses, row strides, and image regions must satisfy the native function's alignment and capacity requirements.
        /// Retain the images and stream context through queued completion, including submission errors and every captured graph replay.
        /// The operation does not initialize padding or channels it leaves untouched.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            source_destination: &mut ImageViewMut<'_, $ty, $layout>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(source_destination)?;

            validate_same_size(source.size(), source_destination.size())?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_unary_operation {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image belongs to another CUDA context, or NPP reports a failure.
        /// All participating image sizes must match; differing sizes return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
        /// All addresses, row strides, and image regions must satisfy the native function's alignment and capacity requirements.
        /// Retain the images and stream context through queued completion, including submission errors and every captured graph replay.
        /// The operation does not initialize padding or channels it leaves untouched.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
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

macro_rules! impl_unary_operation_in_place {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image belongs to another CUDA context, or NPP reports a failure.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep the destination image exclusively writable through completion.
        /// All addresses, row strides, and image regions must satisfy the native function's alignment and capacity requirements.
        /// Retain the images and stream context through queued completion, including submission errors and every captured graph replay.
        /// The operation does not initialize padding or channels it leaves untouched.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_destination: &mut ImageViewMut<'_, $ty, $layout>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_destination)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source_destination.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_unary_operation_scaled {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image belongs to another CUDA context, or NPP reports a failure.
        /// All participating image sizes must match; differing sizes return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep source images read-only and destination pixels exclusively writable and disjoint from every input through completion.
        /// All addresses, row strides, and image regions must satisfy the native function's alignment and capacity requirements.
        /// Retain the images and stream context through queued completion, including submission errors and every captured graph replay.
        /// The operation does not initialize padding or channels it leaves untouched.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
            scale_factor: i32,
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
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_unary_operation_scaled_in_place {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image belongs to another CUDA context, or NPP reports a failure.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep the destination image exclusively writable through completion.
        /// All addresses, row strides, and image regions must satisfy the native function's alignment and capacity requirements.
        /// Retain the images and stream context through queued completion, including submission errors and every captured graph replay.
        /// The operation does not initialize padding or channels it leaves untouched.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_destination: &mut ImageViewMut<'_, $ty, $layout>,
            scale_factor: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_destination)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source_destination.size().into(),
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}
