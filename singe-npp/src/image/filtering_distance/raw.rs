use singe_cuda::memory::DeviceMemory;
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering_validation::{
            optional_image_output, validate_distance_transform_outputs, validate_positive_size,
            validate_same_size,
        },
        view::{C1, C2, ImageOutputView, ImageView},
    },
    try_ffi,
    types::Size,
    utility::to_usize,
};

macro_rules! impl_distance_transform_pba_u16 {
    (
        $doc:literal,
        $name:ident,
        $source_ty:ty,
        $manhattan_ty:ty,
        $transform_ty:ty,
        $ffi_name:ident
    ) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        ///
        /// `None` disables an output; at least one output must be supplied.
        /// Owns temporary device scratch, rejects stream capture, and returns after successful stream completion.
        /// Source values in the inclusive range `minimum_site_value..=maximum_site_value` are sites.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the transform parameters.
        /// At least one output must have positive dimensions; all supplied outputs must have the source image size.
        /// Workspace-size queries, device allocation, capture checks, and stream synchronization can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready for native execution.
        /// NPP receives a mutable source pointer; provide exclusive access to source storage during native use.
        /// Keep each destination exclusively writable and disjoint from the source and all other destinations during native use.
        /// Keep image allocations valid until this call completes its native work.
        /// Native PBA requires ROI width and height in `64..=32767`; this wrapper checks positive dimensions but does not enforce that range.
        /// The source must contain at least one site pixel; the wrapper does not inspect pixel values or validate the site interval.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, C1>,
            minimum_site_value: $source_ty,
            maximum_site_value: $source_ty,
            destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
            destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
            destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, $manhattan_ty, C2>>,
            destination_transform: Option<&mut ImageOutputView<'_, $transform_ty, C1>>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            if let Some(image) = destination_voronoi.as_ref() {
                stream_context.validate_region(*image)?;
            }
            if let Some(image) = destination_voronoi_indices.as_ref() {
                stream_context.validate_region(*image)?;
            }
            if let Some(image) = destination_voronoi_manhattan.as_ref() {
                stream_context.validate_region(*image)?;
            }
            if let Some(image) = destination_transform.as_ref() {
                stream_context.validate_region(*image)?;
            }
            stream_context.stream().ensure_not_capturing()?;


                let (destination_voronoi_ptr, destination_voronoi_step, destination_voronoi_size) =
                    optional_image_output(destination_voronoi);
                let (
                    destination_voronoi_indices_ptr,
                    destination_voronoi_indices_step,
                    destination_voronoi_indices_size,
                ) = optional_image_output(destination_voronoi_indices);
                let (
                    destination_voronoi_manhattan_ptr,
                    destination_voronoi_manhattan_step,
                    destination_voronoi_manhattan_size,
                ) = optional_image_output(destination_voronoi_manhattan);
                let (
                    destination_transform_ptr,
                    destination_transform_step,
                    destination_transform_size,
                ) = optional_image_output(destination_transform);
                let roi = validate_distance_transform_outputs(
                    destination_voronoi_size,
                    destination_voronoi_indices_size,
                    destination_voronoi_manhattan_size,
                    destination_transform_size,
                )?;
                validate_same_size(source.size(), roi)?;
                let required_bytes = distance_transform_pba_buffer_size(roi)?;
                let scratch = DeviceMemory::<u8>::create_uninit(
                    stream_context.cuda_context(),
                    required_bytes,
                )?;


                    stream_context.sync_scope(|| {
                        // SAFETY: the caller upholds this function's safety contract.
                        unsafe { try_ffi!(sys::$ffi_name(
                            source.as_ptr().cast_mut().cast(),
                            source.step(),
                            minimum_site_value,
                            maximum_site_value,
                            destination_voronoi_ptr.cast(),
                            destination_voronoi_step,
                            destination_voronoi_indices_ptr.cast(),
                            destination_voronoi_indices_step,
                            destination_voronoi_manhattan_ptr.cast(),
                            destination_voronoi_manhattan_step,
                            destination_transform_ptr.cast(),
                            destination_transform_step,
                            roi.into(),
                            scratch.as_mut_ptr().cast(),
                            stream_context.as_raw(),
                        )) }
                    })?;

                Ok(())

        }
    };
}

macro_rules! impl_distance_transform_pba_f64 {
    (
        $doc:literal,
        $name:ident,
        $source_ty:ty,
        $manhattan_ty:ty,
        $ffi_name:ident
    ) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        ///
        /// `None` disables an output; at least one output must be supplied.
        /// Owns temporary device scratch, rejects stream capture, and returns after successful stream completion.
        /// Passes no antialiasing workspace, leaving antialiasing disabled.
        /// Source values in the inclusive range `minimum_site_value..=maximum_site_value` are sites.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the transform parameters.
        /// At least one output must have positive dimensions; all supplied outputs must have the source image size.
        /// Workspace-size queries, device allocation, capture checks, and stream synchronization can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready for native execution.
        /// NPP receives a mutable source pointer; provide exclusive access to source storage during native use.
        /// Keep each destination exclusively writable and disjoint from the source and all other destinations during native use.
        /// Keep image allocations valid until this call completes its native work.
        /// Native PBA requires ROI width and height in `64..=32767`; this wrapper checks positive dimensions but does not enforce that range.
        /// The source must contain at least one site pixel; the wrapper does not inspect pixel values or validate the site interval.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, C1>,
            minimum_site_value: $source_ty,
            maximum_site_value: $source_ty,
            destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
            destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
            destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, $manhattan_ty, C2>>,
            destination_transform: Option<&mut ImageOutputView<'_, f64, C1>>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            if let Some(image) = destination_voronoi.as_ref() {
                stream_context.validate_region(*image)?;
            }
            if let Some(image) = destination_voronoi_indices.as_ref() {
                stream_context.validate_region(*image)?;
            }
            if let Some(image) = destination_voronoi_manhattan.as_ref() {
                stream_context.validate_region(*image)?;
            }
            if let Some(image) = destination_transform.as_ref() {
                stream_context.validate_region(*image)?;
            }
            stream_context.stream().ensure_not_capturing()?;


                let (destination_voronoi_ptr, destination_voronoi_step, destination_voronoi_size) =
                    optional_image_output(destination_voronoi);
                let (
                    destination_voronoi_indices_ptr,
                    destination_voronoi_indices_step,
                    destination_voronoi_indices_size,
                ) = optional_image_output(destination_voronoi_indices);
                let (
                    destination_voronoi_manhattan_ptr,
                    destination_voronoi_manhattan_step,
                    destination_voronoi_manhattan_size,
                ) = optional_image_output(destination_voronoi_manhattan);
                let (
                    destination_transform_ptr,
                    destination_transform_step,
                    destination_transform_size,
                ) = optional_image_output(destination_transform);
                let roi = validate_distance_transform_outputs(
                    destination_voronoi_size,
                    destination_voronoi_indices_size,
                    destination_voronoi_manhattan_size,
                    destination_transform_size,
                )?;
                validate_same_size(source.size(), roi)?;
                let required_bytes = distance_transform_pba_buffer_size(roi)?;
                let scratch = DeviceMemory::<u8>::create_uninit(
                    stream_context.cuda_context(),
                    required_bytes,
                )?;


                    stream_context.sync_scope(|| {
                        // SAFETY: the caller upholds this function's safety contract.
                        unsafe { try_ffi!(sys::$ffi_name(
                            source.as_ptr().cast_mut().cast(),
                            source.step(),
                            minimum_site_value,
                            maximum_site_value,
                            destination_voronoi_ptr.cast(),
                            destination_voronoi_step,
                            destination_voronoi_indices_ptr.cast(),
                            destination_voronoi_indices_step,
                            destination_voronoi_manhattan_ptr.cast(),
                            destination_voronoi_manhattan_step,
                            destination_transform_ptr.cast(),
                            destination_transform_step,
                            roi.into(),
                            scratch.as_mut_ptr().cast(),
                            std::ptr::null_mut(),
                            stream_context.as_raw(),
                        )) }
                    })?;

                Ok(())

        }
    };
}

macro_rules! impl_distance_transform_pba_f64_antialiasing {
    (
        $doc:literal,
        $name:ident,
        $source_ty:ty,
        $manhattan_ty:ty,
        $ffi_name:ident
    ) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        ///
        /// `None` disables an output; at least one output must be supplied.
        /// Owns temporary device scratch, rejects stream capture, and returns after successful stream completion.
        /// Allocates the additional antialiasing workspace and enables antialiasing for the double-precision transform.
        /// Source values in the inclusive range `minimum_site_value..=maximum_site_value` are sites.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the transform parameters.
        /// At least one output must have positive dimensions; all supplied outputs must have the source image size.
        /// Workspace-size queries, device allocation, capture checks, and stream synchronization can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready for native execution.
        /// NPP receives a mutable source pointer; provide exclusive access to source storage during native use.
        /// Keep each destination exclusively writable and disjoint from the source and all other destinations during native use.
        /// Keep image allocations valid until this call completes its native work.
        /// Native PBA requires ROI width and height in `64..=32767`; this wrapper checks positive dimensions but does not enforce that range.
        /// The source must contain at least one site pixel; the wrapper does not inspect pixel values or validate the site interval.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, C1>,
            minimum_site_value: $source_ty,
            maximum_site_value: $source_ty,
            destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
            destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
            destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, $manhattan_ty, C2>>,
            destination_transform: Option<&mut ImageOutputView<'_, f64, C1>>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            if let Some(image) = destination_voronoi.as_ref() {
                stream_context.validate_region(*image)?;
            }
            if let Some(image) = destination_voronoi_indices.as_ref() {
                stream_context.validate_region(*image)?;
            }
            if let Some(image) = destination_voronoi_manhattan.as_ref() {
                stream_context.validate_region(*image)?;
            }
            if let Some(image) = destination_transform.as_ref() {
                stream_context.validate_region(*image)?;
            }
            stream_context.stream().ensure_not_capturing()?;


                let (destination_voronoi_ptr, destination_voronoi_step, destination_voronoi_size) =
                    optional_image_output(destination_voronoi);
                let (
                    destination_voronoi_indices_ptr,
                    destination_voronoi_indices_step,
                    destination_voronoi_indices_size,
                ) = optional_image_output(destination_voronoi_indices);
                let (
                    destination_voronoi_manhattan_ptr,
                    destination_voronoi_manhattan_step,
                    destination_voronoi_manhattan_size,
                ) = optional_image_output(destination_voronoi_manhattan);
                let (
                    destination_transform_ptr,
                    destination_transform_step,
                    destination_transform_size,
                ) = optional_image_output(destination_transform);
                let roi = validate_distance_transform_outputs(
                    destination_voronoi_size,
                    destination_voronoi_indices_size,
                    destination_voronoi_manhattan_size,
                    destination_transform_size,
                )?;
                validate_same_size(source.size(), roi)?;
                let required_bytes = distance_transform_pba_buffer_size(roi)?;
                let antialiasing_required_bytes =
                    distance_transform_pba_antialiasing_buffer_size(roi)?;
                let scratch = DeviceMemory::<u8>::create_uninit(
                    stream_context.cuda_context(),
                    required_bytes,
                )?;
                let antialiasing_scratch = DeviceMemory::<u8>::create_uninit(
                    stream_context.cuda_context(),
                    antialiasing_required_bytes,
                )?;


                    stream_context.sync_scope(|| {
                        // SAFETY: the caller upholds this function's safety contract.
                        unsafe { try_ffi!(sys::$ffi_name(
                            source.as_ptr().cast_mut().cast(),
                            source.step(),
                            minimum_site_value,
                            maximum_site_value,
                            destination_voronoi_ptr.cast(),
                            destination_voronoi_step,
                            destination_voronoi_indices_ptr.cast(),
                            destination_voronoi_indices_step,
                            destination_voronoi_manhattan_ptr.cast(),
                            destination_voronoi_manhattan_step,
                            destination_transform_ptr.cast(),
                            destination_transform_step,
                            roi.into(),
                            scratch.as_mut_ptr().cast(),
                            antialiasing_scratch.as_mut_ptr().cast(),
                            stream_context.as_raw(),
                        )) }
                    })?;

                Ok(())

        }
    };
}

macro_rules! impl_signed_distance_transform_pba_f32 {
    (
        $doc:literal,
        $name:ident,
        $manhattan_ty:ty,
        $ffi_name:ident
    ) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        ///
        /// `None` disables an output; at least one output must be supplied.
        /// Owns temporary device scratch, rejects stream capture, and returns after successful stream completion.
        /// `cutoff_value` classifies source pixels; the horizontal and vertical offsets shift the final transform by the supplied subpixel fractions.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the transform parameters.
        /// At least one output must have positive dimensions; all supplied outputs must have the source image size.
        /// Workspace-size queries, device allocation, capture checks, and stream synchronization can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready for native execution.
        /// NPP receives a mutable source pointer; provide exclusive access to source storage during native use.
        /// Keep each destination exclusively writable and disjoint from the source and all other destinations during native use.
        /// Keep image allocations valid until this call completes its native work.
        /// Native PBA requires ROI width and height in `64..=32767`; this wrapper checks positive dimensions but does not enforce that range.
        /// Source classification and subpixel parameters must satisfy the native signed-distance input requirements; this wrapper does not validate their values.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, f32, C1>,
            cutoff_value: f32,
            horizontal_subpixel_offset: f32,
            vertical_subpixel_offset: f32,
            destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
            destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
            destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, $manhattan_ty, C2>>,
            destination_transform: Option<&mut ImageOutputView<'_, f32, C1>>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            if let Some(image) = destination_voronoi.as_ref() {
                stream_context.validate_region(*image)?;
            }
            if let Some(image) = destination_voronoi_indices.as_ref() {
                stream_context.validate_region(*image)?;
            }
            if let Some(image) = destination_voronoi_manhattan.as_ref() {
                stream_context.validate_region(*image)?;
            }
            if let Some(image) = destination_transform.as_ref() {
                stream_context.validate_region(*image)?;
            }
            stream_context.stream().ensure_not_capturing()?;


                let (destination_voronoi_ptr, destination_voronoi_step, destination_voronoi_size) =
                    optional_image_output(destination_voronoi);
                let (
                    destination_voronoi_indices_ptr,
                    destination_voronoi_indices_step,
                    destination_voronoi_indices_size,
                ) = optional_image_output(destination_voronoi_indices);
                let (
                    destination_voronoi_manhattan_ptr,
                    destination_voronoi_manhattan_step,
                    destination_voronoi_manhattan_size,
                ) = optional_image_output(destination_voronoi_manhattan);
                let (
                    destination_transform_ptr,
                    destination_transform_step,
                    destination_transform_size,
                ) = optional_image_output(destination_transform);
                let roi = validate_distance_transform_outputs(
                    destination_voronoi_size,
                    destination_voronoi_indices_size,
                    destination_voronoi_manhattan_size,
                    destination_transform_size,
                )?;
                validate_same_size(source.size(), roi)?;
                let required_bytes = signed_distance_transform_pba_buffer_size(roi)?;
                let scratch = DeviceMemory::<u8>::create_uninit(
                    stream_context.cuda_context(),
                    required_bytes,
                )?;


                    stream_context.sync_scope(|| {
                        // SAFETY: the caller upholds this function's safety contract.
                        unsafe { try_ffi!(sys::$ffi_name(
                            source.as_ptr().cast_mut().cast(),
                            source.step(),
                            cutoff_value,
                            horizontal_subpixel_offset,
                            vertical_subpixel_offset,
                            destination_voronoi_ptr.cast(),
                            destination_voronoi_step,
                            destination_voronoi_indices_ptr.cast(),
                            destination_voronoi_indices_step,
                            destination_voronoi_manhattan_ptr.cast(),
                            destination_voronoi_manhattan_step,
                            destination_transform_ptr.cast(),
                            destination_transform_step,
                            roi.into(),
                            scratch.as_mut_ptr().cast(),
                            stream_context.as_raw(),
                        )) }
                    })?;

                Ok(())

        }
    };
}

macro_rules! impl_signed_distance_transform_pba_f64 {
    (
        $doc:literal,
        $name:ident,
        $source_ty:ty,
        $manhattan_ty:ty,
        $ffi_name:ident
    ) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        ///
        /// `None` disables an output; at least one output must be supplied.
        /// Owns temporary device scratch, rejects stream capture, and returns after successful stream completion.
        /// Passes no antialiasing workspace, leaving antialiasing disabled.
        /// `cutoff_value` classifies source pixels; the horizontal and vertical offsets shift the final transform by the supplied subpixel fractions.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the transform parameters.
        /// At least one output must have positive dimensions; all supplied outputs must have the source image size.
        /// Workspace-size queries, device allocation, capture checks, and stream synchronization can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready for native execution.
        /// NPP receives a mutable source pointer; provide exclusive access to source storage during native use.
        /// Keep each destination exclusively writable and disjoint from the source and all other destinations during native use.
        /// Keep image allocations valid until this call completes its native work.
        /// Native PBA requires ROI width and height in `64..=32767`; this wrapper checks positive dimensions but does not enforce that range.
        /// Source classification and subpixel parameters must satisfy the native signed-distance input requirements; this wrapper does not validate their values.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, C1>,
            cutoff_value: $source_ty,
            horizontal_subpixel_offset: f64,
            vertical_subpixel_offset: f64,
            destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
            destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
            destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, $manhattan_ty, C2>>,
            destination_transform: Option<&mut ImageOutputView<'_, f64, C1>>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            if let Some(image) = destination_voronoi.as_ref() {
                stream_context.validate_region(*image)?;
            }
            if let Some(image) = destination_voronoi_indices.as_ref() {
                stream_context.validate_region(*image)?;
            }
            if let Some(image) = destination_voronoi_manhattan.as_ref() {
                stream_context.validate_region(*image)?;
            }
            if let Some(image) = destination_transform.as_ref() {
                stream_context.validate_region(*image)?;
            }
            stream_context.stream().ensure_not_capturing()?;


                let (destination_voronoi_ptr, destination_voronoi_step, destination_voronoi_size) =
                    optional_image_output(destination_voronoi);
                let (
                    destination_voronoi_indices_ptr,
                    destination_voronoi_indices_step,
                    destination_voronoi_indices_size,
                ) = optional_image_output(destination_voronoi_indices);
                let (
                    destination_voronoi_manhattan_ptr,
                    destination_voronoi_manhattan_step,
                    destination_voronoi_manhattan_size,
                ) = optional_image_output(destination_voronoi_manhattan);
                let (
                    destination_transform_ptr,
                    destination_transform_step,
                    destination_transform_size,
                ) = optional_image_output(destination_transform);
                let roi = validate_distance_transform_outputs(
                    destination_voronoi_size,
                    destination_voronoi_indices_size,
                    destination_voronoi_manhattan_size,
                    destination_transform_size,
                )?;
                validate_same_size(source.size(), roi)?;
                let required_bytes = signed_distance_transform_pba_64f_buffer_size(roi)?;
                let scratch = DeviceMemory::<u8>::create_uninit(
                    stream_context.cuda_context(),
                    required_bytes,
                )?;


                    stream_context.sync_scope(|| {
                        // SAFETY: the caller upholds this function's safety contract.
                        unsafe { try_ffi!(sys::$ffi_name(
                            source.as_ptr().cast_mut().cast(),
                            source.step(),
                            cutoff_value,
                            horizontal_subpixel_offset,
                            vertical_subpixel_offset,
                            destination_voronoi_ptr.cast(),
                            destination_voronoi_step,
                            destination_voronoi_indices_ptr.cast(),
                            destination_voronoi_indices_step,
                            destination_voronoi_manhattan_ptr.cast(),
                            destination_voronoi_manhattan_step,
                            destination_transform_ptr.cast(),
                            destination_transform_step,
                            roi.into(),
                            scratch.as_mut_ptr().cast(),
                            std::ptr::null_mut(),
                            stream_context.as_raw(),
                        )) }
                    })?;

                Ok(())

        }
    };
}

macro_rules! impl_signed_distance_transform_pba_f64_antialiasing {
    (
        $doc:literal,
        $name:ident,
        $source_ty:ty,
        $manhattan_ty:ty,
        $ffi_name:ident
    ) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        ///
        /// `None` disables an output; at least one output must be supplied.
        /// Owns temporary device scratch, rejects stream capture, and returns after successful stream completion.
        /// Allocates the additional antialiasing workspace and enables antialiasing for the double-precision transform.
        /// `cutoff_value` classifies source pixels; the horizontal and vertical offsets shift the final transform by the supplied subpixel fractions.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the transform parameters.
        /// At least one output must have positive dimensions; all supplied outputs must have the source image size.
        /// Workspace-size queries, device allocation, capture checks, and stream synchronization can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready for native execution.
        /// NPP receives a mutable source pointer; provide exclusive access to source storage during native use.
        /// Keep each destination exclusively writable and disjoint from the source and all other destinations during native use.
        /// Keep image allocations valid until this call completes its native work.
        /// Native PBA requires ROI width and height in `64..=32767`; this wrapper checks positive dimensions but does not enforce that range.
        /// Source classification and subpixel parameters must satisfy the native signed-distance input requirements; this wrapper does not validate their values.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, C1>,
            cutoff_value: $source_ty,
            horizontal_subpixel_offset: f64,
            vertical_subpixel_offset: f64,
            destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
            destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
            destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, $manhattan_ty, C2>>,
            destination_transform: Option<&mut ImageOutputView<'_, f64, C1>>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            if let Some(image) = destination_voronoi.as_ref() {
                stream_context.validate_region(*image)?;
            }
            if let Some(image) = destination_voronoi_indices.as_ref() {
                stream_context.validate_region(*image)?;
            }
            if let Some(image) = destination_voronoi_manhattan.as_ref() {
                stream_context.validate_region(*image)?;
            }
            if let Some(image) = destination_transform.as_ref() {
                stream_context.validate_region(*image)?;
            }
            stream_context.stream().ensure_not_capturing()?;


                let (destination_voronoi_ptr, destination_voronoi_step, destination_voronoi_size) =
                    optional_image_output(destination_voronoi);
                let (
                    destination_voronoi_indices_ptr,
                    destination_voronoi_indices_step,
                    destination_voronoi_indices_size,
                ) = optional_image_output(destination_voronoi_indices);
                let (
                    destination_voronoi_manhattan_ptr,
                    destination_voronoi_manhattan_step,
                    destination_voronoi_manhattan_size,
                ) = optional_image_output(destination_voronoi_manhattan);
                let (
                    destination_transform_ptr,
                    destination_transform_step,
                    destination_transform_size,
                ) = optional_image_output(destination_transform);
                let roi = validate_distance_transform_outputs(
                    destination_voronoi_size,
                    destination_voronoi_indices_size,
                    destination_voronoi_manhattan_size,
                    destination_transform_size,
                )?;
                validate_same_size(source.size(), roi)?;
                let required_bytes = signed_distance_transform_pba_64f_buffer_size(roi)?;
                let antialiasing_required_bytes =
                    signed_distance_transform_pba_antialiasing_buffer_size(roi)?;
                let scratch = DeviceMemory::<u8>::create_uninit(
                    stream_context.cuda_context(),
                    required_bytes,
                )?;
                let antialiasing_scratch = DeviceMemory::<u8>::create_uninit(
                    stream_context.cuda_context(),
                    antialiasing_required_bytes,
                )?;


                    stream_context.sync_scope(|| {
                        // SAFETY: the caller upholds this function's safety contract.
                        unsafe { try_ffi!(sys::$ffi_name(
                            source.as_ptr().cast_mut().cast(),
                            source.step(),
                            cutoff_value,
                            horizontal_subpixel_offset,
                            vertical_subpixel_offset,
                            destination_voronoi_ptr.cast(),
                            destination_voronoi_step,
                            destination_voronoi_indices_ptr.cast(),
                            destination_voronoi_indices_step,
                            destination_voronoi_manhattan_ptr.cast(),
                            destination_voronoi_manhattan_step,
                            destination_transform_ptr.cast(),
                            destination_transform_step,
                            roi.into(),
                            scratch.as_mut_ptr().cast(),
                            antialiasing_scratch.as_mut_ptr().cast(),
                            stream_context.as_raw(),
                        )) }
                    })?;

                Ok(())

        }
    };
}

/// Returns scratch size in bytes for PBA distance transformation on `roi`.
///
/// # Errors
///
/// Returns an error if either ROI dimension is nonpositive, NPP rejects the size query, or the returned size cannot be represented as `usize`.
pub fn distance_transform_pba_buffer_size(roi: Size) -> Result<usize> {
    validate_positive_size(roi)?;
    let mut bytes = 0_u64;
    // SAFETY: `nppiDistanceTransformPBAGetBufferSize` receives valid output locations.
    unsafe {
        try_ffi!(sys::nppiDistanceTransformPBAGetBufferSize(
            roi.into(),
            &raw mut bytes,
        ))?;
    }
    to_usize(bytes, "buffer size")
}

/// Returns antialiasing scratch size in bytes for PBA distance transformation on `roi`.
///
/// # Errors
///
/// Returns an error if either ROI dimension is nonpositive, NPP rejects the size query, or the returned size cannot be represented as `usize`.
pub fn distance_transform_pba_antialiasing_buffer_size(roi: Size) -> Result<usize> {
    validate_positive_size(roi)?;
    let mut bytes = 0_u64;
    // SAFETY: `nppiDistanceTransformPBAGetAntialiasingBufferSize` receives valid output locations.
    unsafe {
        try_ffi!(sys::nppiDistanceTransformPBAGetAntialiasingBufferSize(
            roi.into(),
            &raw mut bytes,
        ))?;
    }
    to_usize(bytes, "buffer size")
}

/// Returns scratch size in bytes for signed PBA distance transformation on `roi`.
///
/// # Errors
///
/// Returns an error if either ROI dimension is nonpositive, NPP rejects the size query, or the returned size cannot be represented as `usize`.
pub fn signed_distance_transform_pba_buffer_size(roi: Size) -> Result<usize> {
    validate_positive_size(roi)?;
    let mut bytes = 0_u64;
    // SAFETY: `nppiSignedDistanceTransformPBAGetBufferSize` receives valid output locations.
    unsafe {
        try_ffi!(sys::nppiSignedDistanceTransformPBAGetBufferSize(
            roi.into(),
            &raw mut bytes,
        ))?;
    }
    to_usize(bytes, "buffer size")
}

/// Returns scratch size in bytes for signed double-precision PBA distance transformation on `roi`.
///
/// # Errors
///
/// Returns an error if either ROI dimension is nonpositive, NPP rejects the size query, or the returned size cannot be represented as `usize`.
pub fn signed_distance_transform_pba_64f_buffer_size(roi: Size) -> Result<usize> {
    validate_positive_size(roi)?;
    let mut bytes = 0_u64;
    // SAFETY: `nppiSignedDistanceTransformPBAGet64fBufferSize` receives valid output locations.
    unsafe {
        try_ffi!(sys::nppiSignedDistanceTransformPBAGet64fBufferSize(
            roi.into(),
            &raw mut bytes,
        ))?;
    }
    to_usize(bytes, "buffer size")
}

/// Returns antialiasing scratch size in bytes for signed PBA distance transformation on `roi`.
///
/// # Errors
///
/// Returns an error if either ROI dimension is nonpositive, NPP rejects the size query, or the returned size cannot be represented as `usize`.
pub fn signed_distance_transform_pba_antialiasing_buffer_size(roi: Size) -> Result<usize> {
    validate_positive_size(roi)?;
    let mut bytes = 0_u64;
    // SAFETY: `nppiSignedDistanceTransformPBAGetAntialiasingBufferSize` receives valid output locations.
    unsafe {
        try_ffi!(
            sys::nppiSignedDistanceTransformPBAGetAntialiasingBufferSize(
                roi.into(),
                &raw mut bytes,
            )
        )?;
    }
    to_usize(bytes, "buffer size")
}

impl_distance_transform_pba_u16!(
    "1 channel 8-bit unsigned grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional unsigned 16-bit truncated integer transform with optional relative Manhattan distances.",
    distance_transform_pba_u8_to_u16_c1,
    u8,
    i16,
    u16,
    nppiDistanceTransformPBA_8u16u_C1R_Ctx
);

impl_distance_transform_pba_u16!(
    "1 channel 8-bit unsigned grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional unsigned 16-bit truncated integer transform with optional absolute Manhattan distances.",
    distance_transform_abs_pba_u8_to_u16_c1,
    u8,
    u16,
    u16,
    nppiDistanceTransformAbsPBA_8u16u_C1R_Ctx
);

impl_distance_transform_pba_u16!(
    "1 channel 8-bit signed grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional unsigned 16-bit truncated integer transform with optional relative Manhattan distances.",
    distance_transform_pba_i8_to_u16_c1,
    i8,
    i16,
    u16,
    nppiDistanceTransformPBA_8s16u_C1R_Ctx
);

impl_distance_transform_pba_u16!(
    "1 channel 8-bit signed grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional unsigned 16-bit truncated integer transform with optional absolute Manhattan distances.",
    distance_transform_abs_pba_i8_to_u16_c1,
    i8,
    u16,
    u16,
    nppiDistanceTransformAbsPBA_8s16u_C1R_Ctx
);

impl_distance_transform_pba_u16!(
    "1 channel 16-bit unsigned grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional unsigned 16-bit truncated integer transform with optional relative Manhattan distances.",
    distance_transform_pba_u16_to_u16_c1,
    u16,
    i16,
    u16,
    nppiDistanceTransformPBA_16u16u_C1R_Ctx
);

impl_distance_transform_pba_u16!(
    "1 channel 16-bit unsigned grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional unsigned 16-bit truncated integer transform with optional absolute Manhattan distances.",
    distance_transform_abs_pba_u16_to_u16_c1,
    u16,
    u16,
    u16,
    nppiDistanceTransformAbsPBA_16u16u_C1R_Ctx
);

impl_distance_transform_pba_u16!(
    "1 channel 16-bit signed grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional unsigned 16-bit truncated integer transform with optional relative Manhattan distances.",
    distance_transform_pba_i16_to_u16_c1,
    i16,
    i16,
    u16,
    nppiDistanceTransformPBA_16s16u_C1R_Ctx
);

impl_distance_transform_pba_u16!(
    "1 channel 16-bit signed grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional unsigned 16-bit truncated integer transform with absolute Manhattan distances.",
    distance_transform_abs_pba_i16_to_u16_c1,
    i16,
    u16,
    u16,
    nppiDistanceTransformAbsPBA_16s16u_C1R_Ctx
);

impl_distance_transform_pba_u16!(
    "1 channel 8-bit unsigned grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional 32-bit floating point transform with optional relative Manhattan distances.",
    distance_transform_pba_u8_to_f32_c1,
    u8,
    i16,
    f32,
    nppiDistanceTransformPBA_8u32f_C1R_Ctx
);

impl_distance_transform_pba_u16!(
    "1 channel 8-bit unsigned grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional 32-bit floating point transform with optional absolute Manhattan distances.",
    distance_transform_abs_pba_u8_to_f32_c1,
    u8,
    u16,
    f32,
    nppiDistanceTransformAbsPBA_8u32f_C1R_Ctx
);

impl_distance_transform_pba_u16!(
    "1 channel 8-bit signed grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional 32-bit floating point transform with optional relative Manhattan distances.",
    distance_transform_pba_i8_to_f32_c1,
    i8,
    i16,
    f32,
    nppiDistanceTransformPBA_8s32f_C1R_Ctx
);

impl_distance_transform_pba_u16!(
    "1 channel 8-bit signed grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional 32-bit floating point transform with optional absolute Manhattan distances.",
    distance_transform_abs_pba_i8_to_f32_c1,
    i8,
    u16,
    f32,
    nppiDistanceTransformAbsPBA_8s32f_C1R_Ctx
);

impl_distance_transform_pba_u16!(
    "1 channel 16-bit unsigned grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional 32-bit floating point transform with optional relative Manhattan distances.",
    distance_transform_pba_u16_to_f32_c1,
    u16,
    i16,
    f32,
    nppiDistanceTransformPBA_16u32f_C1R_Ctx
);

impl_distance_transform_pba_u16!(
    "1 channel 16-bit unsigned grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional 32-bit floating point transform with optional absolute Manhattan distances.",
    distance_transform_abs_pba_u16_to_f32_c1,
    u16,
    u16,
    f32,
    nppiDistanceTransformAbsPBA_16u32f_C1R_Ctx
);

impl_distance_transform_pba_u16!(
    "1 channel 16-bit signed grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional 32-bit floating point transform with optional relative Manhattan distances.",
    distance_transform_pba_i16_to_f32_c1,
    i16,
    i16,
    f32,
    nppiDistanceTransformPBA_16s32f_C1R_Ctx
);

impl_distance_transform_pba_u16!(
    "1 channel 16-bit signed grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional 32-bit floating point transform with optional absolute Manhattan distances.",
    distance_transform_abs_pba_i16_to_f32_c1,
    i16,
    u16,
    f32,
    nppiDistanceTransformAbsPBA_16s32f_C1R_Ctx
);

impl_distance_transform_pba_f64!(
    "1 channel 8-bit unsigned grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional 64-bit floating point transform with optional relative Manhattan distances.",
    distance_transform_pba_u8_to_f64_c1,
    u8,
    i16,
    nppiDistanceTransformPBA_8u64f_C1R_Ctx
);

impl_distance_transform_pba_f64!(
    "1 channel 8-bit unsigned grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional 64-bit floating point transform with optional absolute Manhattan distances.",
    distance_transform_abs_pba_u8_to_f64_c1,
    u8,
    u16,
    nppiDistanceTransformAbsPBA_8u64f_C1R_Ctx
);

impl_distance_transform_pba_f64!(
    "1 channel 8-bit signed grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional 64-bit floating point transform with optional relative Manhattan distances.",
    distance_transform_pba_i8_to_f64_c1,
    i8,
    i16,
    nppiDistanceTransformPBA_8s64f_C1R_Ctx
);

impl_distance_transform_pba_f64!(
    "1 channel 8-bit signed grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional 64-bit floating point transform with optional absolute Manhattan distances.",
    distance_transform_abs_pba_i8_to_f64_c1,
    i8,
    u16,
    nppiDistanceTransformAbsPBA_8s64f_C1R_Ctx
);

impl_distance_transform_pba_f64!(
    "1 channel 16-bit unsigned grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional 64-bit floating point transform with optional relative Manhattan distances.",
    distance_transform_pba_u16_to_f64_c1,
    u16,
    i16,
    nppiDistanceTransformPBA_16u64f_C1R_Ctx
);

impl_distance_transform_pba_f64!(
    "1 channel 16-bit unsigned grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional 64-bit floating point transform with optional absolute Manhattan distances.",
    distance_transform_abs_pba_u16_to_f64_c1,
    u16,
    u16,
    nppiDistanceTransformAbsPBA_16u64f_C1R_Ctx
);

impl_distance_transform_pba_f64!(
    "1 channel 16-bit signed grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional 64-bit floating point transform with optional relative Manhattan distances.",
    distance_transform_pba_i16_to_f64_c1,
    i16,
    i16,
    nppiDistanceTransformPBA_16s64f_C1R_Ctx
);

impl_distance_transform_pba_f64!(
    "1 channel 16-bit signed grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional 64-bit floating point transform with optional absolute Manhattan distances.",
    distance_transform_abs_pba_i16_to_f64_c1,
    i16,
    u16,
    nppiDistanceTransformAbsPBA_16s64f_C1R_Ctx
);

impl_distance_transform_pba_f64!(
    "1 channel 32-bit floating point grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional 64-bit floating point transform with optional relative Manhattan distances.",
    distance_transform_pba_f32_to_f64_c1,
    f32,
    i16,
    nppiDistanceTransformPBA_32f64f_C1R_Ctx
);

impl_distance_transform_pba_f64!(
    "1 channel 32-bit floating point grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional 64-bit floating point transform with optional absolute Manhattan distances.",
    distance_transform_abs_pba_f32_to_f64_c1,
    f32,
    u16,
    nppiDistanceTransformAbsPBA_32f64f_C1R_Ctx
);

impl_distance_transform_pba_f64!(
    "1 channel 64-bit floating point grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional 64-bit floating point transform with optional relative Manhattan distances.",
    distance_transform_pba_f64_c1,
    f64,
    i16,
    nppiDistanceTransformPBA_64f_C1R_Ctx
);

impl_distance_transform_pba_f64!(
    "1 channel 64-bit floating point grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional 64-bit floating point transform with optional absolute Manhattan distances.",
    distance_transform_abs_pba_f64_c1,
    f64,
    u16,
    nppiDistanceTransformAbsPBA_64f_C1R_Ctx
);

impl_distance_transform_pba_f64_antialiasing!(
    "1 channel 8-bit unsigned grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional 64-bit floating point transform with optional relative Manhattan distances.",
    distance_transform_pba_u8_to_f64_c1_antialiasing,
    u8,
    i16,
    nppiDistanceTransformPBA_8u64f_C1R_Ctx
);

impl_distance_transform_pba_f64_antialiasing!(
    "1 channel 8-bit unsigned grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional 64-bit floating point transform with optional absolute Manhattan distances.",
    distance_transform_abs_pba_u8_to_f64_c1_antialiasing,
    u8,
    u16,
    nppiDistanceTransformAbsPBA_8u64f_C1R_Ctx
);

impl_distance_transform_pba_f64_antialiasing!(
    "1 channel 8-bit signed grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional 64-bit floating point transform with optional relative Manhattan distances.",
    distance_transform_pba_i8_to_f64_c1_antialiasing,
    i8,
    i16,
    nppiDistanceTransformPBA_8s64f_C1R_Ctx
);

impl_distance_transform_pba_f64_antialiasing!(
    "1 channel 8-bit signed grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional 64-bit floating point transform with optional absolute Manhattan distances.",
    distance_transform_abs_pba_i8_to_f64_c1_antialiasing,
    i8,
    u16,
    nppiDistanceTransformAbsPBA_8s64f_C1R_Ctx
);

impl_distance_transform_pba_f64_antialiasing!(
    "1 channel 16-bit unsigned grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional 64-bit floating point transform with optional relative Manhattan distances.",
    distance_transform_pba_u16_to_f64_c1_antialiasing,
    u16,
    i16,
    nppiDistanceTransformPBA_16u64f_C1R_Ctx
);

impl_distance_transform_pba_f64_antialiasing!(
    "1 channel 16-bit unsigned grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional 64-bit floating point transform with optional absolute Manhattan distances.",
    distance_transform_abs_pba_u16_to_f64_c1_antialiasing,
    u16,
    u16,
    nppiDistanceTransformAbsPBA_16u64f_C1R_Ctx
);

impl_distance_transform_pba_f64_antialiasing!(
    "1 channel 16-bit signed grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional 64-bit floating point transform with optional relative Manhattan distances.",
    distance_transform_pba_i16_to_f64_c1_antialiasing,
    i16,
    i16,
    nppiDistanceTransformPBA_16s64f_C1R_Ctx
);

impl_distance_transform_pba_f64_antialiasing!(
    "1 channel 16-bit signed grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional 64-bit floating point transform with optional absolute Manhattan distances.",
    distance_transform_abs_pba_i16_to_f64_c1_antialiasing,
    i16,
    u16,
    nppiDistanceTransformAbsPBA_16s64f_C1R_Ctx
);

impl_distance_transform_pba_f64_antialiasing!(
    "1 channel 32-bit floating point grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional 64-bit floating point transform with optional relative Manhattan distances.",
    distance_transform_pba_f32_to_f64_c1_antialiasing,
    f32,
    i16,
    nppiDistanceTransformPBA_32f64f_C1R_Ctx
);

impl_distance_transform_pba_f64_antialiasing!(
    "1 channel 32-bit floating point grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional 64-bit floating point transform with optional absolute Manhattan distances.",
    distance_transform_abs_pba_f32_to_f64_c1_antialiasing,
    f32,
    u16,
    nppiDistanceTransformAbsPBA_32f64f_C1R_Ctx
);

impl_distance_transform_pba_f64_antialiasing!(
    "1 channel 64-bit floating point grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional 64-bit floating point transform with optional relative Manhattan distances.",
    distance_transform_pba_f64_c1_antialiasing,
    f64,
    i16,
    nppiDistanceTransformPBA_64f_C1R_Ctx
);

impl_distance_transform_pba_f64_antialiasing!(
    "1 channel 64-bit floating point grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or optional 64-bit floating point transform with optional absolute Manhattan distances.",
    distance_transform_abs_pba_f64_c1_antialiasing,
    f64,
    u16,
    nppiDistanceTransformAbsPBA_64f_C1R_Ctx
);

impl_signed_distance_transform_pba_f32!(
    "1 channel 32-bit floating point grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram and 32-bit floating point transform with optional sub-pixel shifts.",
    signed_distance_transform_pba_f32_c1,
    i16,
    nppiSignedDistanceTransformPBA_32f_C1R_Ctx
);

impl_signed_distance_transform_pba_f32!(
    "1 channel 32-bit floating point grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram and 32-bit floating point transform with optional sub-pixel shifts.",
    signed_distance_transform_abs_pba_f32_c1,
    u16,
    nppiSignedDistanceTransformAbsPBA_32f_C1R_Ctx
);

impl_signed_distance_transform_pba_f64!(
    "1 channel 32-bit floating point grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram and 64-bit floating point transform with optional sub-pixel shifts.",
    signed_distance_transform_pba_f32_to_f64_c1,
    f32,
    i16,
    nppiSignedDistanceTransformPBA_32f64f_C1R_Ctx
);

impl_signed_distance_transform_pba_f64!(
    "1 channel 32-bit floating point grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram and 64-bit floating point transform with optional sub-pixel shifts.",
    signed_distance_transform_abs_pba_f32_to_f64_c1,
    f32,
    u16,
    nppiSignedDistanceTransformAbsPBA_32f64f_C1R_Ctx
);

impl_signed_distance_transform_pba_f64!(
    "1 channel 64-bit floating point grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram and 64-bit floating point transform with optional sub-pixel shifts.",
    signed_distance_transform_pba_f64_c1,
    f64,
    i16,
    nppiSignedDistanceTransformPBA_64f_C1R_Ctx
);

impl_signed_distance_transform_pba_f64!(
    "1 channel 64-bit floating point grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram and 64-bit floating point transform with optional sub-pixel shifts.",
    signed_distance_transform_abs_pba_f64_c1,
    f64,
    u16,
    nppiSignedDistanceTransformAbsPBA_64f_C1R_Ctx
);

impl_signed_distance_transform_pba_f64_antialiasing!(
    "1 channel 32-bit floating point grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram and 64-bit floating point transform with optional sub-pixel shifts.",
    signed_distance_transform_pba_f32_to_f64_c1_antialiasing,
    f32,
    i16,
    nppiSignedDistanceTransformPBA_32f64f_C1R_Ctx
);

impl_signed_distance_transform_pba_f64_antialiasing!(
    "1 channel 32-bit floating point grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram and 64-bit floating point transform with optional sub-pixel shifts.",
    signed_distance_transform_abs_pba_f32_to_f64_c1_antialiasing,
    f32,
    u16,
    nppiSignedDistanceTransformAbsPBA_32f64f_C1R_Ctx
);

impl_signed_distance_transform_pba_f64_antialiasing!(
    "1 channel 64-bit floating point grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram and 64-bit floating point transform with optional sub-pixel shifts.",
    signed_distance_transform_pba_f64_c1_antialiasing,
    f64,
    i16,
    nppiSignedDistanceTransformPBA_64f_C1R_Ctx
);

impl_signed_distance_transform_pba_f64_antialiasing!(
    "1 channel 64-bit floating point grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram and 64-bit floating point transform with optional sub-pixel shifts.",
    signed_distance_transform_abs_pba_f64_c1_antialiasing,
    f64,
    u16,
    nppiSignedDistanceTransformAbsPBA_64f_C1R_Ctx
);
