use super::*;

pub fn flood_fill_buffer_size(roi: Size) -> Result<usize> {
    validate_positive_size(roi)?;
    let mut bytes = 0;
    unsafe {
        try_ffi!(sys::nppiFloodFillGetBufferSize(roi.into(), &raw mut bytes,))?;
    }
    to_usize(bytes, "buffer size")
}

macro_rules! flood_fill_region_ptr {
    ($region:expr) => {
        match $region {
            Some(region) => region as *mut ConnectedRegion,
            None => std::ptr::null_mut(),
        }
    };
}

macro_rules! impl_flood_fill_c1 {
    ($name:ident, $pixel_ty:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source_destination: &mut ImageViewMut<'_, $pixel_ty, C1>,
            seed: Point,
            new_value: $pixel_ty,
            norm: ImageNormalization,
            connected_region: Option<&mut ConnectedRegion>,
        ) -> Result<()> {
            validate_seed(source_destination.size(), seed)?;
            let scratch =
                DeviceMemory::<u8>::create(flood_fill_buffer_size(source_destination.size())?)?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    seed.into(),
                    new_value,
                    norm.into(),
                    source_destination.size().into(),
                    flood_fill_region_ptr!(connected_region).cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_flood_fill_c3 {
    ($name:ident, $pixel_ty:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source_destination: &mut ImageViewMut<'_, $pixel_ty, C3>,
            seed: Point,
            new_values: [$pixel_ty; 3],
            norm: ImageNormalization,
            connected_region: Option<&mut ConnectedRegion>,
        ) -> Result<()> {
            validate_seed(source_destination.size(), seed)?;
            let scratch =
                DeviceMemory::<u8>::create(flood_fill_buffer_size(source_destination.size())?)?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    seed.into(),
                    new_values.as_ptr().cast(),
                    norm.into(),
                    source_destination.size().into(),
                    flood_fill_region_ptr!(connected_region).cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_flood_fill_boundary_c1 {
    ($name:ident, $pixel_ty:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source_destination: &mut ImageViewMut<'_, $pixel_ty, C1>,
            seed: Point,
            new_value: $pixel_ty,
            boundary_value: $pixel_ty,
            norm: ImageNormalization,
            connected_region: Option<&mut ConnectedRegion>,
        ) -> Result<()> {
            validate_seed(source_destination.size(), seed)?;
            let scratch =
                DeviceMemory::<u8>::create(flood_fill_buffer_size(source_destination.size())?)?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    seed.into(),
                    new_value,
                    boundary_value,
                    norm.into(),
                    source_destination.size().into(),
                    flood_fill_region_ptr!(connected_region).cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_flood_fill_boundary_c3 {
    ($name:ident, $pixel_ty:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source_destination: &mut ImageViewMut<'_, $pixel_ty, C3>,
            seed: Point,
            new_values: [$pixel_ty; 3],
            boundary_values: [$pixel_ty; 3],
            norm: ImageNormalization,
            connected_region: Option<&mut ConnectedRegion>,
        ) -> Result<()> {
            validate_seed(source_destination.size(), seed)?;
            let scratch =
                DeviceMemory::<u8>::create(flood_fill_buffer_size(source_destination.size())?)?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    seed.into(),
                    new_values.as_ptr().cast(),
                    boundary_values.as_ptr().cast(),
                    norm.into(),
                    source_destination.size().into(),
                    flood_fill_region_ptr!(connected_region).cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_flood_fill_range_c1 {
    ($name:ident, $pixel_ty:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source_destination: &mut ImageViewMut<'_, $pixel_ty, C1>,
            seed: Point,
            min: $pixel_ty,
            max: $pixel_ty,
            new_value: $pixel_ty,
            norm: ImageNormalization,
            connected_region: Option<&mut ConnectedRegion>,
        ) -> Result<()> {
            validate_seed(source_destination.size(), seed)?;
            let scratch =
                DeviceMemory::<u8>::create(flood_fill_buffer_size(source_destination.size())?)?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    seed.into(),
                    min,
                    max,
                    new_value,
                    norm.into(),
                    source_destination.size().into(),
                    flood_fill_region_ptr!(connected_region).cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_flood_fill_range_c3 {
    ($name:ident, $pixel_ty:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source_destination: &mut ImageViewMut<'_, $pixel_ty, C3>,
            seed: Point,
            min: [$pixel_ty; 3],
            max: [$pixel_ty; 3],
            new_values: [$pixel_ty; 3],
            norm: ImageNormalization,
            connected_region: Option<&mut ConnectedRegion>,
        ) -> Result<()> {
            validate_seed(source_destination.size(), seed)?;
            let mut min = min;
            let mut max = max;
            let scratch =
                DeviceMemory::<u8>::create(flood_fill_buffer_size(source_destination.size())?)?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    seed.into(),
                    min.as_mut_ptr().cast(),
                    max.as_mut_ptr().cast(),
                    new_values.as_ptr().cast(),
                    norm.into(),
                    source_destination.size().into(),
                    flood_fill_region_ptr!(connected_region).cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_flood_fill_range_boundary_c1 {
    ($name:ident, $pixel_ty:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source_destination: &mut ImageViewMut<'_, $pixel_ty, C1>,
            seed: Point,
            min: $pixel_ty,
            max: $pixel_ty,
            new_value: $pixel_ty,
            boundary_value: $pixel_ty,
            norm: ImageNormalization,
            connected_region: Option<&mut ConnectedRegion>,
        ) -> Result<()> {
            validate_seed(source_destination.size(), seed)?;
            let scratch =
                DeviceMemory::<u8>::create(flood_fill_buffer_size(source_destination.size())?)?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    seed.into(),
                    min,
                    max,
                    new_value,
                    boundary_value,
                    norm.into(),
                    source_destination.size().into(),
                    flood_fill_region_ptr!(connected_region).cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_flood_fill_range_boundary_c3 {
    ($name:ident, $pixel_ty:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source_destination: &mut ImageViewMut<'_, $pixel_ty, C3>,
            seed: Point,
            min: [$pixel_ty; 3],
            max: [$pixel_ty; 3],
            new_values: [$pixel_ty; 3],
            boundary_values: [$pixel_ty; 3],
            norm: ImageNormalization,
            connected_region: Option<&mut ConnectedRegion>,
        ) -> Result<()> {
            validate_seed(source_destination.size(), seed)?;
            let mut min = min;
            let mut max = max;
            let scratch =
                DeviceMemory::<u8>::create(flood_fill_buffer_size(source_destination.size())?)?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    seed.into(),
                    min.as_mut_ptr().cast(),
                    max.as_mut_ptr().cast(),
                    new_values.as_ptr().cast(),
                    boundary_values.as_ptr().cast(),
                    norm.into(),
                    source_destination.size().into(),
                    flood_fill_region_ptr!(connected_region).cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

impl_flood_fill_c1!(flood_fill_u8_c1_in_place, u8, nppiFloodFill_8u_C1IR_Ctx);
impl_flood_fill_c3!(flood_fill_u8_c3_in_place, u8, nppiFloodFill_8u_C3IR_Ctx);
impl_flood_fill_c1!(flood_fill_u16_c1_in_place, u16, nppiFloodFill_16u_C1IR_Ctx);
impl_flood_fill_c3!(flood_fill_u16_c3_in_place, u16, nppiFloodFill_16u_C3IR_Ctx);
impl_flood_fill_c1!(flood_fill_u32_c1_in_place, u32, nppiFloodFill_32u_C1IR_Ctx);
impl_flood_fill_c3!(flood_fill_u32_c3_in_place, u32, nppiFloodFill_32u_C3IR_Ctx);
impl_flood_fill_boundary_c1!(
    flood_fill_boundary_u8_c1_in_place,
    u8,
    nppiFloodFillBoundary_8u_C1IR_Ctx
);
impl_flood_fill_boundary_c3!(
    flood_fill_boundary_u8_c3_in_place,
    u8,
    nppiFloodFillBoundary_8u_C3IR_Ctx
);
impl_flood_fill_boundary_c1!(
    flood_fill_boundary_u16_c1_in_place,
    u16,
    nppiFloodFillBoundary_16u_C1IR_Ctx
);
impl_flood_fill_boundary_c3!(
    flood_fill_boundary_u16_c3_in_place,
    u16,
    nppiFloodFillBoundary_16u_C3IR_Ctx
);
impl_flood_fill_boundary_c1!(
    flood_fill_boundary_u32_c1_in_place,
    u32,
    nppiFloodFillBoundary_32u_C1IR_Ctx
);
impl_flood_fill_boundary_c3!(
    flood_fill_boundary_u32_c3_in_place,
    u32,
    nppiFloodFillBoundary_32u_C3IR_Ctx
);
impl_flood_fill_range_c1!(
    flood_fill_range_u8_c1_in_place,
    u8,
    nppiFloodFillRange_8u_C1IR_Ctx
);
impl_flood_fill_range_c3!(
    flood_fill_range_u8_c3_in_place,
    u8,
    nppiFloodFillRange_8u_C3IR_Ctx
);
impl_flood_fill_range_c1!(
    flood_fill_range_u16_c1_in_place,
    u16,
    nppiFloodFillRange_16u_C1IR_Ctx
);
impl_flood_fill_range_c3!(
    flood_fill_range_u16_c3_in_place,
    u16,
    nppiFloodFillRange_16u_C3IR_Ctx
);
impl_flood_fill_range_c1!(
    flood_fill_range_u32_c1_in_place,
    u32,
    nppiFloodFillRange_32u_C1IR_Ctx
);
impl_flood_fill_range_c3!(
    flood_fill_range_u32_c3_in_place,
    u32,
    nppiFloodFillRange_32u_C3IR_Ctx
);
impl_flood_fill_range_boundary_c1!(
    flood_fill_range_boundary_u8_c1_in_place,
    u8,
    nppiFloodFillRangeBoundary_8u_C1IR_Ctx
);
impl_flood_fill_range_boundary_c3!(
    flood_fill_range_boundary_u8_c3_in_place,
    u8,
    nppiFloodFillRangeBoundary_8u_C3IR_Ctx
);
impl_flood_fill_range_boundary_c1!(
    flood_fill_range_boundary_u16_c1_in_place,
    u16,
    nppiFloodFillRangeBoundary_16u_C1IR_Ctx
);
impl_flood_fill_range_boundary_c3!(
    flood_fill_range_boundary_u16_c3_in_place,
    u16,
    nppiFloodFillRangeBoundary_16u_C3IR_Ctx
);
impl_flood_fill_range_boundary_c1!(
    flood_fill_range_boundary_u32_c1_in_place,
    u32,
    nppiFloodFillRangeBoundary_32u_C1IR_Ctx
);
impl_flood_fill_range_boundary_c3!(
    flood_fill_range_boundary_u32_c3_in_place,
    u32,
    nppiFloodFillRangeBoundary_32u_C3IR_Ctx
);
impl_flood_fill_range_c1!(
    flood_fill_gradient_u8_c1_in_place,
    u8,
    nppiFloodFillGradient_8u_C1IR_Ctx
);
impl_flood_fill_range_c3!(
    flood_fill_gradient_u8_c3_in_place,
    u8,
    nppiFloodFillGradient_8u_C3IR_Ctx
);
impl_flood_fill_range_c1!(
    flood_fill_gradient_u16_c1_in_place,
    u16,
    nppiFloodFillGradient_16u_C1IR_Ctx
);
impl_flood_fill_range_c3!(
    flood_fill_gradient_u16_c3_in_place,
    u16,
    nppiFloodFillGradient_16u_C3IR_Ctx
);
impl_flood_fill_range_c1!(
    flood_fill_gradient_u32_c1_in_place,
    u32,
    nppiFloodFillGradient_32u_C1IR_Ctx
);
impl_flood_fill_range_c3!(
    flood_fill_gradient_u32_c3_in_place,
    u32,
    nppiFloodFillGradient_32u_C3IR_Ctx
);
impl_flood_fill_range_boundary_c1!(
    flood_fill_gradient_boundary_u8_c1_in_place,
    u8,
    nppiFloodFillGradientBoundary_8u_C1IR_Ctx
);
impl_flood_fill_range_boundary_c3!(
    flood_fill_gradient_boundary_u8_c3_in_place,
    u8,
    nppiFloodFillGradientBoundary_8u_C3IR_Ctx
);
impl_flood_fill_range_boundary_c1!(
    flood_fill_gradient_boundary_u16_c1_in_place,
    u16,
    nppiFloodFillGradientBoundary_16u_C1IR_Ctx
);
impl_flood_fill_range_boundary_c3!(
    flood_fill_gradient_boundary_u16_c3_in_place,
    u16,
    nppiFloodFillGradientBoundary_16u_C3IR_Ctx
);
impl_flood_fill_range_boundary_c1!(
    flood_fill_gradient_boundary_u32_c1_in_place,
    u32,
    nppiFloodFillGradientBoundary_32u_C1IR_Ctx
);
impl_flood_fill_range_boundary_c3!(
    flood_fill_gradient_boundary_u32_c3_in_place,
    u32,
    nppiFloodFillGradientBoundary_32u_C3IR_Ctx
);
