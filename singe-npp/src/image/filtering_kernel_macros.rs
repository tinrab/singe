macro_rules! impl_filter_wiener_border {
    ($name:ident, $pixel_ty:ty, $layout:ty, $noise_len:expr, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            source_offset: Point,
            destination: &mut ImageViewMut<'_, $pixel_ty, $layout>,
            mask_size: Size,
            anchor: Point,
            noise: &mut [f32; $noise_len],
            border_type: BorderType,
        ) -> Result<()> {
            validate_border_roi(source.size(), source_offset, destination.size())?;
            validate_filter_mask(mask_size, anchor)?;
            if border_type != BorderType::Replicate {
                return Err(Error::OutOfRange {
                    name: "border type".into(),
                });
            }
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    source_offset.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    mask_size.into(),
                    anchor.into(),
                    noise.as_mut_ptr().cast(),
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_masked_kernel {
    ($name:ident, $pixel_ty:ty, $layout:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            destination: &mut ImageViewMut<'_, $pixel_ty, $layout>,
            mask_size: MaskSize,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    mask_size.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_masked_kernel_border {
    ($name:ident, $pixel_ty:ty, $layout:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            source_offset: Point,
            destination: &mut ImageViewMut<'_, $pixel_ty, $layout>,
            mask_size: MaskSize,
            border_type: BorderType,
        ) -> Result<()> {
            validate_border_roi(source.size(), source_offset, destination.size())?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    source_offset.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    mask_size.into(),
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_masked_kernel_typed {
    ($name:ident, $source_ty:ty, $source_layout:ty, $destination_ty:ty, $destination_layout:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $source_layout>,
            destination: &mut ImageViewMut<'_, $destination_ty, $destination_layout>,
            mask_size: MaskSize,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    mask_size.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_masked_kernel_typed_border {
    ($name:ident, $source_ty:ty, $source_layout:ty, $destination_ty:ty, $destination_layout:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $source_layout>,
            source_offset: Point,
            destination: &mut ImageViewMut<'_, $destination_ty, $destination_layout>,
            mask_size: MaskSize,
            border_type: BorderType,
        ) -> Result<()> {
            validate_border_roi(source.size(), source_offset, destination.size())?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    source_offset.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    mask_size.into(),
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_gradient_vector_border {
    (
        $name:ident,
        $source_ty:ty,
        $source_layout:ty,
        $gradient_ty:ty,
        $ffi_name:ident
    ) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $source_layout>,
            source_offset: Point,
            destination_x: Option<&mut ImageViewMut<'_, $gradient_ty, C1>>,
            destination_y: Option<&mut ImageViewMut<'_, $gradient_ty, C1>>,
            destination_magnitude: Option<&mut ImageViewMut<'_, $gradient_ty, C1>>,
            destination_angle: Option<&mut ImageViewMut<'_, f32, C1>>,
            mask_size: MaskSize,
            norm: ImageNormalization,
            border_type: BorderType,
        ) -> Result<()> {
            let (destination_x_ptr, destination_x_step, destination_x_size) =
                optional_image_output(destination_x);
            let (destination_y_ptr, destination_y_step, destination_y_size) =
                optional_image_output(destination_y);
            let (
                destination_magnitude_ptr,
                destination_magnitude_step,
                destination_magnitude_size,
            ) = optional_image_output(destination_magnitude);
            let (destination_angle_ptr, destination_angle_step, destination_angle_size) =
                optional_image_output(destination_angle);
            let roi = validate_gradient_vector_outputs(
                destination_x_size,
                destination_y_size,
                destination_magnitude_size,
                destination_angle_size,
            )?;
            validate_border_roi(source.size(), source_offset, roi)?;

            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    source_offset.into(),
                    destination_x_ptr.cast(),
                    destination_x_step,
                    destination_y_ptr.cast(),
                    destination_y_step,
                    destination_magnitude_ptr.cast(),
                    destination_magnitude_step,
                    destination_angle_ptr.cast(),
                    destination_angle_step,
                    roi.into(),
                    mask_size.into(),
                    norm.into(),
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}
