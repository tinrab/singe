use super::*;

impl_gradient_vector_border!(
    gradient_vector_prewitt_border_u8_to_i16_f32_c1,
    u8,
    C1,
    i16,
    nppiGradientVectorPrewittBorder_8u16s_C1R_Ctx
);
impl_gradient_vector_border!(
    gradient_vector_prewitt_border_u8_c3_to_i16_f32_c1,
    u8,
    C3,
    i16,
    nppiGradientVectorPrewittBorder_8u16s_C3C1R_Ctx
);
impl_gradient_vector_border!(
    gradient_vector_prewitt_border_i16_to_f32_c1,
    i16,
    C1,
    f32,
    nppiGradientVectorPrewittBorder_16s32f_C1R_Ctx
);
impl_gradient_vector_border!(
    gradient_vector_prewitt_border_i16_c3_to_f32_c1,
    i16,
    C3,
    f32,
    nppiGradientVectorPrewittBorder_16s32f_C3C1R_Ctx
);
impl_gradient_vector_border!(
    gradient_vector_prewitt_border_u16_to_f32_c1,
    u16,
    C1,
    f32,
    nppiGradientVectorPrewittBorder_16u32f_C1R_Ctx
);
impl_gradient_vector_border!(
    gradient_vector_prewitt_border_u16_c3_to_f32_c1,
    u16,
    C3,
    f32,
    nppiGradientVectorPrewittBorder_16u32f_C3C1R_Ctx
);
impl_gradient_vector_border!(
    gradient_vector_prewitt_border_f32_to_f32_c1,
    f32,
    C1,
    f32,
    nppiGradientVectorPrewittBorder_32f_C1R_Ctx
);
impl_gradient_vector_border!(
    gradient_vector_prewitt_border_f32_c3_to_f32_c1,
    f32,
    C3,
    f32,
    nppiGradientVectorPrewittBorder_32f_C3C1R_Ctx
);
impl_gradient_vector_border!(
    gradient_vector_scharr_border_u8_to_i16_f32_c1,
    u8,
    C1,
    i16,
    nppiGradientVectorScharrBorder_8u16s_C1R_Ctx
);
impl_gradient_vector_border!(
    gradient_vector_scharr_border_u8_c3_to_i16_f32_c1,
    u8,
    C3,
    i16,
    nppiGradientVectorScharrBorder_8u16s_C3C1R_Ctx
);
impl_gradient_vector_border!(
    gradient_vector_scharr_border_i16_to_f32_c1,
    i16,
    C1,
    f32,
    nppiGradientVectorScharrBorder_16s32f_C1R_Ctx
);
impl_gradient_vector_border!(
    gradient_vector_scharr_border_i16_c3_to_f32_c1,
    i16,
    C3,
    f32,
    nppiGradientVectorScharrBorder_16s32f_C3C1R_Ctx
);
impl_gradient_vector_border!(
    gradient_vector_scharr_border_u16_to_f32_c1,
    u16,
    C1,
    f32,
    nppiGradientVectorScharrBorder_16u32f_C1R_Ctx
);
impl_gradient_vector_border!(
    gradient_vector_scharr_border_u16_c3_to_f32_c1,
    u16,
    C3,
    f32,
    nppiGradientVectorScharrBorder_16u32f_C3C1R_Ctx
);
impl_gradient_vector_border!(
    gradient_vector_scharr_border_f32_to_f32_c1,
    f32,
    C1,
    f32,
    nppiGradientVectorScharrBorder_32f_C1R_Ctx
);
impl_gradient_vector_border!(
    gradient_vector_scharr_border_f32_c3_to_f32_c1,
    f32,
    C3,
    f32,
    nppiGradientVectorScharrBorder_32f_C3C1R_Ctx
);
impl_gradient_vector_border!(
    gradient_vector_sobel_border_u8_to_i16_f32_c1,
    u8,
    C1,
    i16,
    nppiGradientVectorSobelBorder_8u16s_C1R_Ctx
);
impl_gradient_vector_border!(
    gradient_vector_sobel_border_u8_c3_to_i16_f32_c1,
    u8,
    C3,
    i16,
    nppiGradientVectorSobelBorder_8u16s_C3C1R_Ctx
);
impl_gradient_vector_border!(
    gradient_vector_sobel_border_i16_to_f32_c1,
    i16,
    C1,
    f32,
    nppiGradientVectorSobelBorder_16s32f_C1R_Ctx
);
impl_gradient_vector_border!(
    gradient_vector_sobel_border_i16_c3_to_f32_c1,
    i16,
    C3,
    f32,
    nppiGradientVectorSobelBorder_16s32f_C3C1R_Ctx
);
impl_gradient_vector_border!(
    gradient_vector_sobel_border_u16_to_f32_c1,
    u16,
    C1,
    f32,
    nppiGradientVectorSobelBorder_16u32f_C1R_Ctx
);
impl_gradient_vector_border!(
    gradient_vector_sobel_border_u16_c3_to_f32_c1,
    u16,
    C3,
    f32,
    nppiGradientVectorSobelBorder_16u32f_C3C1R_Ctx
);
impl_gradient_vector_border!(
    gradient_vector_sobel_border_f32_to_f32_c1,
    f32,
    C1,
    f32,
    nppiGradientVectorSobelBorder_32f_C1R_Ctx
);
impl_gradient_vector_border!(
    gradient_vector_sobel_border_f32_c3_to_f32_c1,
    f32,
    C3,
    f32,
    nppiGradientVectorSobelBorder_32f_C3C1R_Ctx
);

macro_rules! impl_gradient_vector_border_c1_dispatch {
    ($trait_name:ident, $method_name:ident, $source_ty:ty, $gradient_ty:ty, $direct_name:ident) => {
        impl $trait_name for $source_ty {
            type Gradient = $gradient_ty;

            fn $method_name(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                source_offset: Point,
                destination_x: Option<&mut ImageViewMut<'_, Self::Gradient, C1>>,
                destination_y: Option<&mut ImageViewMut<'_, Self::Gradient, C1>>,
                destination_magnitude: Option<&mut ImageViewMut<'_, Self::Gradient, C1>>,
                destination_angle: Option<&mut ImageViewMut<'_, f32, C1>>,
                mask_size: MaskSize,
                norm: ImageNormalization,
                border_type: BorderType,
            ) -> Result<()> {
                $direct_name(
                    stream_context,
                    source,
                    source_offset,
                    destination_x,
                    destination_y,
                    destination_magnitude,
                    destination_angle,
                    mask_size,
                    norm,
                    border_type,
                )
            }
        }
    };
}

macro_rules! impl_gradient_vector_border_c3_to_c1_dispatch {
    ($trait_name:ident, $method_name:ident, $source_ty:ty, $gradient_ty:ty, $direct_name:ident) => {
        impl $trait_name for $source_ty {
            type Gradient = $gradient_ty;

            fn $method_name(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C3>,
                source_offset: Point,
                destination_x: Option<&mut ImageViewMut<'_, Self::Gradient, C1>>,
                destination_y: Option<&mut ImageViewMut<'_, Self::Gradient, C1>>,
                destination_magnitude: Option<&mut ImageViewMut<'_, Self::Gradient, C1>>,
                destination_angle: Option<&mut ImageViewMut<'_, f32, C1>>,
                mask_size: MaskSize,
                norm: ImageNormalization,
                border_type: BorderType,
            ) -> Result<()> {
                $direct_name(
                    stream_context,
                    source,
                    source_offset,
                    destination_x,
                    destination_y,
                    destination_magnitude,
                    destination_angle,
                    mask_size,
                    norm,
                    border_type,
                )
            }
        }
    };
}

pub trait GradientVectorPrewittBorderC1: DataTypeLike {
    type Gradient: DataTypeLike;

    fn gradient_vector_prewitt_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination_x: Option<&mut ImageViewMut<'_, Self::Gradient, C1>>,
        destination_y: Option<&mut ImageViewMut<'_, Self::Gradient, C1>>,
        destination_magnitude: Option<&mut ImageViewMut<'_, Self::Gradient, C1>>,
        destination_angle: Option<&mut ImageViewMut<'_, f32, C1>>,
        mask_size: MaskSize,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_gradient_vector_border_c1_dispatch!(
    GradientVectorPrewittBorderC1,
    gradient_vector_prewitt_border,
    u8,
    i16,
    gradient_vector_prewitt_border_u8_to_i16_f32_c1
);
impl_gradient_vector_border_c1_dispatch!(
    GradientVectorPrewittBorderC1,
    gradient_vector_prewitt_border,
    i16,
    f32,
    gradient_vector_prewitt_border_i16_to_f32_c1
);
impl_gradient_vector_border_c1_dispatch!(
    GradientVectorPrewittBorderC1,
    gradient_vector_prewitt_border,
    u16,
    f32,
    gradient_vector_prewitt_border_u16_to_f32_c1
);
impl_gradient_vector_border_c1_dispatch!(
    GradientVectorPrewittBorderC1,
    gradient_vector_prewitt_border,
    f32,
    f32,
    gradient_vector_prewitt_border_f32_to_f32_c1
);

pub fn gradient_vector_prewitt_border<T: GradientVectorPrewittBorderC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    source_offset: Point,
    destination_x: Option<&mut ImageViewMut<'_, T::Gradient, C1>>,
    destination_y: Option<&mut ImageViewMut<'_, T::Gradient, C1>>,
    destination_magnitude: Option<&mut ImageViewMut<'_, T::Gradient, C1>>,
    destination_angle: Option<&mut ImageViewMut<'_, f32, C1>>,
    mask_size: MaskSize,
    norm: ImageNormalization,
    border_type: BorderType,
) -> Result<()> {
    T::gradient_vector_prewitt_border(
        stream_context,
        source,
        source_offset,
        destination_x,
        destination_y,
        destination_magnitude,
        destination_angle,
        mask_size,
        norm,
        border_type,
    )
}

pub trait GradientVectorPrewittBorderC3ToC1: DataTypeLike {
    type Gradient: DataTypeLike;

    fn gradient_vector_prewitt_border_color(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C3>,
        source_offset: Point,
        destination_x: Option<&mut ImageViewMut<'_, Self::Gradient, C1>>,
        destination_y: Option<&mut ImageViewMut<'_, Self::Gradient, C1>>,
        destination_magnitude: Option<&mut ImageViewMut<'_, Self::Gradient, C1>>,
        destination_angle: Option<&mut ImageViewMut<'_, f32, C1>>,
        mask_size: MaskSize,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_gradient_vector_border_c3_to_c1_dispatch!(
    GradientVectorPrewittBorderC3ToC1,
    gradient_vector_prewitt_border_color,
    u8,
    i16,
    gradient_vector_prewitt_border_u8_c3_to_i16_f32_c1
);
impl_gradient_vector_border_c3_to_c1_dispatch!(
    GradientVectorPrewittBorderC3ToC1,
    gradient_vector_prewitt_border_color,
    i16,
    f32,
    gradient_vector_prewitt_border_i16_c3_to_f32_c1
);
impl_gradient_vector_border_c3_to_c1_dispatch!(
    GradientVectorPrewittBorderC3ToC1,
    gradient_vector_prewitt_border_color,
    u16,
    f32,
    gradient_vector_prewitt_border_u16_c3_to_f32_c1
);
impl_gradient_vector_border_c3_to_c1_dispatch!(
    GradientVectorPrewittBorderC3ToC1,
    gradient_vector_prewitt_border_color,
    f32,
    f32,
    gradient_vector_prewitt_border_f32_c3_to_f32_c1
);

pub fn gradient_vector_prewitt_border_color<T: GradientVectorPrewittBorderC3ToC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C3>,
    source_offset: Point,
    destination_x: Option<&mut ImageViewMut<'_, T::Gradient, C1>>,
    destination_y: Option<&mut ImageViewMut<'_, T::Gradient, C1>>,
    destination_magnitude: Option<&mut ImageViewMut<'_, T::Gradient, C1>>,
    destination_angle: Option<&mut ImageViewMut<'_, f32, C1>>,
    mask_size: MaskSize,
    norm: ImageNormalization,
    border_type: BorderType,
) -> Result<()> {
    T::gradient_vector_prewitt_border_color(
        stream_context,
        source,
        source_offset,
        destination_x,
        destination_y,
        destination_magnitude,
        destination_angle,
        mask_size,
        norm,
        border_type,
    )
}

pub trait GradientVectorScharrBorderC1: DataTypeLike {
    type Gradient: DataTypeLike;

    fn gradient_vector_scharr_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination_x: Option<&mut ImageViewMut<'_, Self::Gradient, C1>>,
        destination_y: Option<&mut ImageViewMut<'_, Self::Gradient, C1>>,
        destination_magnitude: Option<&mut ImageViewMut<'_, Self::Gradient, C1>>,
        destination_angle: Option<&mut ImageViewMut<'_, f32, C1>>,
        mask_size: MaskSize,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_gradient_vector_border_c1_dispatch!(
    GradientVectorScharrBorderC1,
    gradient_vector_scharr_border,
    u8,
    i16,
    gradient_vector_scharr_border_u8_to_i16_f32_c1
);
impl_gradient_vector_border_c1_dispatch!(
    GradientVectorScharrBorderC1,
    gradient_vector_scharr_border,
    i16,
    f32,
    gradient_vector_scharr_border_i16_to_f32_c1
);
impl_gradient_vector_border_c1_dispatch!(
    GradientVectorScharrBorderC1,
    gradient_vector_scharr_border,
    u16,
    f32,
    gradient_vector_scharr_border_u16_to_f32_c1
);
impl_gradient_vector_border_c1_dispatch!(
    GradientVectorScharrBorderC1,
    gradient_vector_scharr_border,
    f32,
    f32,
    gradient_vector_scharr_border_f32_to_f32_c1
);

pub fn gradient_vector_scharr_border<T: GradientVectorScharrBorderC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    source_offset: Point,
    destination_x: Option<&mut ImageViewMut<'_, T::Gradient, C1>>,
    destination_y: Option<&mut ImageViewMut<'_, T::Gradient, C1>>,
    destination_magnitude: Option<&mut ImageViewMut<'_, T::Gradient, C1>>,
    destination_angle: Option<&mut ImageViewMut<'_, f32, C1>>,
    mask_size: MaskSize,
    norm: ImageNormalization,
    border_type: BorderType,
) -> Result<()> {
    T::gradient_vector_scharr_border(
        stream_context,
        source,
        source_offset,
        destination_x,
        destination_y,
        destination_magnitude,
        destination_angle,
        mask_size,
        norm,
        border_type,
    )
}

pub trait GradientVectorScharrBorderC3ToC1: DataTypeLike {
    type Gradient: DataTypeLike;

    fn gradient_vector_scharr_border_color(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C3>,
        source_offset: Point,
        destination_x: Option<&mut ImageViewMut<'_, Self::Gradient, C1>>,
        destination_y: Option<&mut ImageViewMut<'_, Self::Gradient, C1>>,
        destination_magnitude: Option<&mut ImageViewMut<'_, Self::Gradient, C1>>,
        destination_angle: Option<&mut ImageViewMut<'_, f32, C1>>,
        mask_size: MaskSize,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_gradient_vector_border_c3_to_c1_dispatch!(
    GradientVectorScharrBorderC3ToC1,
    gradient_vector_scharr_border_color,
    u8,
    i16,
    gradient_vector_scharr_border_u8_c3_to_i16_f32_c1
);
impl_gradient_vector_border_c3_to_c1_dispatch!(
    GradientVectorScharrBorderC3ToC1,
    gradient_vector_scharr_border_color,
    i16,
    f32,
    gradient_vector_scharr_border_i16_c3_to_f32_c1
);
impl_gradient_vector_border_c3_to_c1_dispatch!(
    GradientVectorScharrBorderC3ToC1,
    gradient_vector_scharr_border_color,
    u16,
    f32,
    gradient_vector_scharr_border_u16_c3_to_f32_c1
);
impl_gradient_vector_border_c3_to_c1_dispatch!(
    GradientVectorScharrBorderC3ToC1,
    gradient_vector_scharr_border_color,
    f32,
    f32,
    gradient_vector_scharr_border_f32_c3_to_f32_c1
);

pub fn gradient_vector_scharr_border_color<T: GradientVectorScharrBorderC3ToC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C3>,
    source_offset: Point,
    destination_x: Option<&mut ImageViewMut<'_, T::Gradient, C1>>,
    destination_y: Option<&mut ImageViewMut<'_, T::Gradient, C1>>,
    destination_magnitude: Option<&mut ImageViewMut<'_, T::Gradient, C1>>,
    destination_angle: Option<&mut ImageViewMut<'_, f32, C1>>,
    mask_size: MaskSize,
    norm: ImageNormalization,
    border_type: BorderType,
) -> Result<()> {
    T::gradient_vector_scharr_border_color(
        stream_context,
        source,
        source_offset,
        destination_x,
        destination_y,
        destination_magnitude,
        destination_angle,
        mask_size,
        norm,
        border_type,
    )
}

pub trait GradientVectorSobelBorderC1: DataTypeLike {
    type Gradient: DataTypeLike;

    fn gradient_vector_sobel_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination_x: Option<&mut ImageViewMut<'_, Self::Gradient, C1>>,
        destination_y: Option<&mut ImageViewMut<'_, Self::Gradient, C1>>,
        destination_magnitude: Option<&mut ImageViewMut<'_, Self::Gradient, C1>>,
        destination_angle: Option<&mut ImageViewMut<'_, f32, C1>>,
        mask_size: MaskSize,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_gradient_vector_border_c1_dispatch!(
    GradientVectorSobelBorderC1,
    gradient_vector_sobel_border,
    u8,
    i16,
    gradient_vector_sobel_border_u8_to_i16_f32_c1
);
impl_gradient_vector_border_c1_dispatch!(
    GradientVectorSobelBorderC1,
    gradient_vector_sobel_border,
    i16,
    f32,
    gradient_vector_sobel_border_i16_to_f32_c1
);
impl_gradient_vector_border_c1_dispatch!(
    GradientVectorSobelBorderC1,
    gradient_vector_sobel_border,
    u16,
    f32,
    gradient_vector_sobel_border_u16_to_f32_c1
);
impl_gradient_vector_border_c1_dispatch!(
    GradientVectorSobelBorderC1,
    gradient_vector_sobel_border,
    f32,
    f32,
    gradient_vector_sobel_border_f32_to_f32_c1
);

pub fn gradient_vector_sobel_border<T: GradientVectorSobelBorderC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    source_offset: Point,
    destination_x: Option<&mut ImageViewMut<'_, T::Gradient, C1>>,
    destination_y: Option<&mut ImageViewMut<'_, T::Gradient, C1>>,
    destination_magnitude: Option<&mut ImageViewMut<'_, T::Gradient, C1>>,
    destination_angle: Option<&mut ImageViewMut<'_, f32, C1>>,
    mask_size: MaskSize,
    norm: ImageNormalization,
    border_type: BorderType,
) -> Result<()> {
    T::gradient_vector_sobel_border(
        stream_context,
        source,
        source_offset,
        destination_x,
        destination_y,
        destination_magnitude,
        destination_angle,
        mask_size,
        norm,
        border_type,
    )
}

pub trait GradientVectorSobelBorderC3ToC1: DataTypeLike {
    type Gradient: DataTypeLike;

    fn gradient_vector_sobel_border_color(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C3>,
        source_offset: Point,
        destination_x: Option<&mut ImageViewMut<'_, Self::Gradient, C1>>,
        destination_y: Option<&mut ImageViewMut<'_, Self::Gradient, C1>>,
        destination_magnitude: Option<&mut ImageViewMut<'_, Self::Gradient, C1>>,
        destination_angle: Option<&mut ImageViewMut<'_, f32, C1>>,
        mask_size: MaskSize,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_gradient_vector_border_c3_to_c1_dispatch!(
    GradientVectorSobelBorderC3ToC1,
    gradient_vector_sobel_border_color,
    u8,
    i16,
    gradient_vector_sobel_border_u8_c3_to_i16_f32_c1
);
impl_gradient_vector_border_c3_to_c1_dispatch!(
    GradientVectorSobelBorderC3ToC1,
    gradient_vector_sobel_border_color,
    i16,
    f32,
    gradient_vector_sobel_border_i16_c3_to_f32_c1
);
impl_gradient_vector_border_c3_to_c1_dispatch!(
    GradientVectorSobelBorderC3ToC1,
    gradient_vector_sobel_border_color,
    u16,
    f32,
    gradient_vector_sobel_border_u16_c3_to_f32_c1
);
impl_gradient_vector_border_c3_to_c1_dispatch!(
    GradientVectorSobelBorderC3ToC1,
    gradient_vector_sobel_border_color,
    f32,
    f32,
    gradient_vector_sobel_border_f32_c3_to_f32_c1
);

pub fn gradient_vector_sobel_border_color<T: GradientVectorSobelBorderC3ToC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C3>,
    source_offset: Point,
    destination_x: Option<&mut ImageViewMut<'_, T::Gradient, C1>>,
    destination_y: Option<&mut ImageViewMut<'_, T::Gradient, C1>>,
    destination_magnitude: Option<&mut ImageViewMut<'_, T::Gradient, C1>>,
    destination_angle: Option<&mut ImageViewMut<'_, f32, C1>>,
    mask_size: MaskSize,
    norm: ImageNormalization,
    border_type: BorderType,
) -> Result<()> {
    T::gradient_vector_sobel_border_color(
        stream_context,
        source,
        source_offset,
        destination_x,
        destination_y,
        destination_magnitude,
        destination_angle,
        mask_size,
        norm,
        border_type,
    )
}
