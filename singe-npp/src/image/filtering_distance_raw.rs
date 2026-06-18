use super::*;

macro_rules! impl_distance_transform_pba_u16 {
    (
        $name:ident,
        $source_ty:ty,
        $manhattan_ty:ty,
        $transform_ty:ty,
        $ffi_name:ident
    ) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, C1>,
            minimum_site_value: $source_ty,
            maximum_site_value: $source_ty,
            destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
            destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
            destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, $manhattan_ty, C2>>,
            destination_transform: Option<&mut ImageViewMut<'_, $transform_ty, C1>>,
        ) -> Result<()> {
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
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;

            unsafe {
                try_ffi!(sys::$ffi_name(
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
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_distance_transform_pba_f64 {
    (
        $name:ident,
        $source_ty:ty,
        $manhattan_ty:ty,
        $ffi_name:ident
    ) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, C1>,
            minimum_site_value: $source_ty,
            maximum_site_value: $source_ty,
            destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
            destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
            destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, $manhattan_ty, C2>>,
            destination_transform: Option<&mut ImageViewMut<'_, f64, C1>>,
        ) -> Result<()> {
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
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;

            unsafe {
                try_ffi!(sys::$ffi_name(
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
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_distance_transform_pba_f64_antialiasing {
    (
        $name:ident,
        $source_ty:ty,
        $manhattan_ty:ty,
        $ffi_name:ident
    ) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, C1>,
            minimum_site_value: $source_ty,
            maximum_site_value: $source_ty,
            destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
            destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
            destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, $manhattan_ty, C2>>,
            destination_transform: Option<&mut ImageViewMut<'_, f64, C1>>,
        ) -> Result<()> {
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
            let antialiasing_required_bytes = distance_transform_pba_antialiasing_buffer_size(roi)?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;
            let antialiasing_scratch = DeviceMemory::<u8>::create(antialiasing_required_bytes)?;

            unsafe {
                try_ffi!(sys::$ffi_name(
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
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_signed_distance_transform_pba_f32 {
    (
        $name:ident,
        $manhattan_ty:ty,
        $ffi_name:ident
    ) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, f32, C1>,
            cutoff_value: f32,
            subpixel_x_shift: f32,
            subpixel_y_shift: f32,
            destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
            destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
            destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, $manhattan_ty, C2>>,
            destination_transform: Option<&mut ImageViewMut<'_, f32, C1>>,
        ) -> Result<()> {
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
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;

            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast_mut().cast(),
                    source.step(),
                    cutoff_value,
                    subpixel_x_shift,
                    subpixel_y_shift,
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
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_signed_distance_transform_pba_f64 {
    (
        $name:ident,
        $source_ty:ty,
        $manhattan_ty:ty,
        $ffi_name:ident
    ) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, C1>,
            cutoff_value: $source_ty,
            subpixel_x_shift: f64,
            subpixel_y_shift: f64,
            destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
            destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
            destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, $manhattan_ty, C2>>,
            destination_transform: Option<&mut ImageViewMut<'_, f64, C1>>,
        ) -> Result<()> {
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
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;

            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast_mut().cast(),
                    source.step(),
                    cutoff_value,
                    subpixel_x_shift,
                    subpixel_y_shift,
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
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_signed_distance_transform_pba_f64_antialiasing {
    (
        $name:ident,
        $source_ty:ty,
        $manhattan_ty:ty,
        $ffi_name:ident
    ) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, C1>,
            cutoff_value: $source_ty,
            subpixel_x_shift: f64,
            subpixel_y_shift: f64,
            destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
            destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
            destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, $manhattan_ty, C2>>,
            destination_transform: Option<&mut ImageViewMut<'_, f64, C1>>,
        ) -> Result<()> {
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
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;
            let antialiasing_scratch = DeviceMemory::<u8>::create(antialiasing_required_bytes)?;

            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast_mut().cast(),
                    source.step(),
                    cutoff_value,
                    subpixel_x_shift,
                    subpixel_y_shift,
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
                ))?;
            }
            Ok(())
        }
    };
}

pub fn distance_transform_pba_buffer_size(roi: Size) -> Result<usize> {
    validate_positive_size(roi)?;
    let mut bytes = 0_u64;
    unsafe {
        try_ffi!(sys::nppiDistanceTransformPBAGetBufferSize(
            roi.into(),
            &raw mut bytes,
        ))?;
    }
    to_usize(bytes, "buffer size")
}

pub fn distance_transform_pba_antialiasing_buffer_size(roi: Size) -> Result<usize> {
    validate_positive_size(roi)?;
    let mut bytes = 0_u64;
    unsafe {
        try_ffi!(sys::nppiDistanceTransformPBAGetAntialiasingBufferSize(
            roi.into(),
            &raw mut bytes,
        ))?;
    }
    to_usize(bytes, "buffer size")
}

pub fn signed_distance_transform_pba_buffer_size(roi: Size) -> Result<usize> {
    validate_positive_size(roi)?;
    let mut bytes = 0_u64;
    unsafe {
        try_ffi!(sys::nppiSignedDistanceTransformPBAGetBufferSize(
            roi.into(),
            &raw mut bytes,
        ))?;
    }
    to_usize(bytes, "buffer size")
}

pub fn signed_distance_transform_pba_64f_buffer_size(roi: Size) -> Result<usize> {
    validate_positive_size(roi)?;
    let mut bytes = 0_u64;
    unsafe {
        try_ffi!(sys::nppiSignedDistanceTransformPBAGet64fBufferSize(
            roi.into(),
            &raw mut bytes,
        ))?;
    }
    to_usize(bytes, "buffer size")
}

pub fn signed_distance_transform_pba_antialiasing_buffer_size(roi: Size) -> Result<usize> {
    validate_positive_size(roi)?;
    let mut bytes = 0_u64;
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
    distance_transform_pba_u8_to_u16_c1,
    u8,
    i16,
    u16,
    nppiDistanceTransformPBA_8u16u_C1R_Ctx
);
impl_distance_transform_pba_u16!(
    distance_transform_abs_pba_u8_to_u16_c1,
    u8,
    u16,
    u16,
    nppiDistanceTransformAbsPBA_8u16u_C1R_Ctx
);
impl_distance_transform_pba_u16!(
    distance_transform_pba_i8_to_u16_c1,
    i8,
    i16,
    u16,
    nppiDistanceTransformPBA_8s16u_C1R_Ctx
);
impl_distance_transform_pba_u16!(
    distance_transform_abs_pba_i8_to_u16_c1,
    i8,
    u16,
    u16,
    nppiDistanceTransformAbsPBA_8s16u_C1R_Ctx
);
impl_distance_transform_pba_u16!(
    distance_transform_pba_u16_to_u16_c1,
    u16,
    i16,
    u16,
    nppiDistanceTransformPBA_16u16u_C1R_Ctx
);
impl_distance_transform_pba_u16!(
    distance_transform_abs_pba_u16_to_u16_c1,
    u16,
    u16,
    u16,
    nppiDistanceTransformAbsPBA_16u16u_C1R_Ctx
);
impl_distance_transform_pba_u16!(
    distance_transform_pba_i16_to_u16_c1,
    i16,
    i16,
    u16,
    nppiDistanceTransformPBA_16s16u_C1R_Ctx
);
impl_distance_transform_pba_u16!(
    distance_transform_abs_pba_i16_to_u16_c1,
    i16,
    u16,
    u16,
    nppiDistanceTransformAbsPBA_16s16u_C1R_Ctx
);
impl_distance_transform_pba_u16!(
    distance_transform_pba_u8_to_f32_c1,
    u8,
    i16,
    f32,
    nppiDistanceTransformPBA_8u32f_C1R_Ctx
);
impl_distance_transform_pba_u16!(
    distance_transform_abs_pba_u8_to_f32_c1,
    u8,
    u16,
    f32,
    nppiDistanceTransformAbsPBA_8u32f_C1R_Ctx
);
impl_distance_transform_pba_u16!(
    distance_transform_pba_i8_to_f32_c1,
    i8,
    i16,
    f32,
    nppiDistanceTransformPBA_8s32f_C1R_Ctx
);
impl_distance_transform_pba_u16!(
    distance_transform_abs_pba_i8_to_f32_c1,
    i8,
    u16,
    f32,
    nppiDistanceTransformAbsPBA_8s32f_C1R_Ctx
);
impl_distance_transform_pba_u16!(
    distance_transform_pba_u16_to_f32_c1,
    u16,
    i16,
    f32,
    nppiDistanceTransformPBA_16u32f_C1R_Ctx
);
impl_distance_transform_pba_u16!(
    distance_transform_abs_pba_u16_to_f32_c1,
    u16,
    u16,
    f32,
    nppiDistanceTransformAbsPBA_16u32f_C1R_Ctx
);
impl_distance_transform_pba_u16!(
    distance_transform_pba_i16_to_f32_c1,
    i16,
    i16,
    f32,
    nppiDistanceTransformPBA_16s32f_C1R_Ctx
);
impl_distance_transform_pba_u16!(
    distance_transform_abs_pba_i16_to_f32_c1,
    i16,
    u16,
    f32,
    nppiDistanceTransformAbsPBA_16s32f_C1R_Ctx
);
impl_distance_transform_pba_f64!(
    distance_transform_pba_u8_to_f64_c1,
    u8,
    i16,
    nppiDistanceTransformPBA_8u64f_C1R_Ctx
);
impl_distance_transform_pba_f64!(
    distance_transform_abs_pba_u8_to_f64_c1,
    u8,
    u16,
    nppiDistanceTransformAbsPBA_8u64f_C1R_Ctx
);
impl_distance_transform_pba_f64!(
    distance_transform_pba_i8_to_f64_c1,
    i8,
    i16,
    nppiDistanceTransformPBA_8s64f_C1R_Ctx
);
impl_distance_transform_pba_f64!(
    distance_transform_abs_pba_i8_to_f64_c1,
    i8,
    u16,
    nppiDistanceTransformAbsPBA_8s64f_C1R_Ctx
);
impl_distance_transform_pba_f64!(
    distance_transform_pba_u16_to_f64_c1,
    u16,
    i16,
    nppiDistanceTransformPBA_16u64f_C1R_Ctx
);
impl_distance_transform_pba_f64!(
    distance_transform_abs_pba_u16_to_f64_c1,
    u16,
    u16,
    nppiDistanceTransformAbsPBA_16u64f_C1R_Ctx
);
impl_distance_transform_pba_f64!(
    distance_transform_pba_i16_to_f64_c1,
    i16,
    i16,
    nppiDistanceTransformPBA_16s64f_C1R_Ctx
);
impl_distance_transform_pba_f64!(
    distance_transform_abs_pba_i16_to_f64_c1,
    i16,
    u16,
    nppiDistanceTransformAbsPBA_16s64f_C1R_Ctx
);
impl_distance_transform_pba_f64!(
    distance_transform_pba_f32_to_f64_c1,
    f32,
    i16,
    nppiDistanceTransformPBA_32f64f_C1R_Ctx
);
impl_distance_transform_pba_f64!(
    distance_transform_abs_pba_f32_to_f64_c1,
    f32,
    u16,
    nppiDistanceTransformAbsPBA_32f64f_C1R_Ctx
);
impl_distance_transform_pba_f64!(
    distance_transform_pba_f64_c1,
    f64,
    i16,
    nppiDistanceTransformPBA_64f_C1R_Ctx
);
impl_distance_transform_pba_f64!(
    distance_transform_abs_pba_f64_c1,
    f64,
    u16,
    nppiDistanceTransformAbsPBA_64f_C1R_Ctx
);
impl_distance_transform_pba_f64_antialiasing!(
    distance_transform_pba_u8_to_f64_c1_antialiasing,
    u8,
    i16,
    nppiDistanceTransformPBA_8u64f_C1R_Ctx
);
impl_distance_transform_pba_f64_antialiasing!(
    distance_transform_abs_pba_u8_to_f64_c1_antialiasing,
    u8,
    u16,
    nppiDistanceTransformAbsPBA_8u64f_C1R_Ctx
);
impl_distance_transform_pba_f64_antialiasing!(
    distance_transform_pba_i8_to_f64_c1_antialiasing,
    i8,
    i16,
    nppiDistanceTransformPBA_8s64f_C1R_Ctx
);
impl_distance_transform_pba_f64_antialiasing!(
    distance_transform_abs_pba_i8_to_f64_c1_antialiasing,
    i8,
    u16,
    nppiDistanceTransformAbsPBA_8s64f_C1R_Ctx
);
impl_distance_transform_pba_f64_antialiasing!(
    distance_transform_pba_u16_to_f64_c1_antialiasing,
    u16,
    i16,
    nppiDistanceTransformPBA_16u64f_C1R_Ctx
);
impl_distance_transform_pba_f64_antialiasing!(
    distance_transform_abs_pba_u16_to_f64_c1_antialiasing,
    u16,
    u16,
    nppiDistanceTransformAbsPBA_16u64f_C1R_Ctx
);
impl_distance_transform_pba_f64_antialiasing!(
    distance_transform_pba_i16_to_f64_c1_antialiasing,
    i16,
    i16,
    nppiDistanceTransformPBA_16s64f_C1R_Ctx
);
impl_distance_transform_pba_f64_antialiasing!(
    distance_transform_abs_pba_i16_to_f64_c1_antialiasing,
    i16,
    u16,
    nppiDistanceTransformAbsPBA_16s64f_C1R_Ctx
);
impl_distance_transform_pba_f64_antialiasing!(
    distance_transform_pba_f32_to_f64_c1_antialiasing,
    f32,
    i16,
    nppiDistanceTransformPBA_32f64f_C1R_Ctx
);
impl_distance_transform_pba_f64_antialiasing!(
    distance_transform_abs_pba_f32_to_f64_c1_antialiasing,
    f32,
    u16,
    nppiDistanceTransformAbsPBA_32f64f_C1R_Ctx
);
impl_distance_transform_pba_f64_antialiasing!(
    distance_transform_pba_f64_c1_antialiasing,
    f64,
    i16,
    nppiDistanceTransformPBA_64f_C1R_Ctx
);
impl_distance_transform_pba_f64_antialiasing!(
    distance_transform_abs_pba_f64_c1_antialiasing,
    f64,
    u16,
    nppiDistanceTransformAbsPBA_64f_C1R_Ctx
);
impl_signed_distance_transform_pba_f32!(
    signed_distance_transform_pba_f32_c1,
    i16,
    nppiSignedDistanceTransformPBA_32f_C1R_Ctx
);
impl_signed_distance_transform_pba_f32!(
    signed_distance_transform_abs_pba_f32_c1,
    u16,
    nppiSignedDistanceTransformAbsPBA_32f_C1R_Ctx
);
impl_signed_distance_transform_pba_f64!(
    signed_distance_transform_pba_f32_to_f64_c1,
    f32,
    i16,
    nppiSignedDistanceTransformPBA_32f64f_C1R_Ctx
);
impl_signed_distance_transform_pba_f64!(
    signed_distance_transform_abs_pba_f32_to_f64_c1,
    f32,
    u16,
    nppiSignedDistanceTransformAbsPBA_32f64f_C1R_Ctx
);
impl_signed_distance_transform_pba_f64!(
    signed_distance_transform_pba_f64_c1,
    f64,
    i16,
    nppiSignedDistanceTransformPBA_64f_C1R_Ctx
);
impl_signed_distance_transform_pba_f64!(
    signed_distance_transform_abs_pba_f64_c1,
    f64,
    u16,
    nppiSignedDistanceTransformAbsPBA_64f_C1R_Ctx
);
impl_signed_distance_transform_pba_f64_antialiasing!(
    signed_distance_transform_pba_f32_to_f64_c1_antialiasing,
    f32,
    i16,
    nppiSignedDistanceTransformPBA_32f64f_C1R_Ctx
);
impl_signed_distance_transform_pba_f64_antialiasing!(
    signed_distance_transform_abs_pba_f32_to_f64_c1_antialiasing,
    f32,
    u16,
    nppiSignedDistanceTransformAbsPBA_32f64f_C1R_Ctx
);
impl_signed_distance_transform_pba_f64_antialiasing!(
    signed_distance_transform_pba_f64_c1_antialiasing,
    f64,
    i16,
    nppiSignedDistanceTransformPBA_64f_C1R_Ctx
);
impl_signed_distance_transform_pba_f64_antialiasing!(
    signed_distance_transform_abs_pba_f64_c1_antialiasing,
    f64,
    u16,
    nppiSignedDistanceTransformAbsPBA_64f_C1R_Ctx
);
