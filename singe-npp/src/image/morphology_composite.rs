use super::*;

macro_rules! impl_morph_buffer_size {
    ($name:ident, $ffi:ident) => {
        pub fn $name(roi: Size) -> Result<usize> {
            validate_positive_size(roi, "roi")?;

            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$ffi(roi.into(), &raw mut bytes))?;
            }

            to_usize(bytes, "buffer size")
        }
    };
}

macro_rules! impl_generic_morph_buffer_size {
    ($trait:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn morph_buffer_size(roi: Size) -> Result<usize>;
        }

        pub fn $function<T>(roi: Size) -> Result<usize>
        where
            T: $trait<$layout>,
        {
            T::morph_buffer_size(roi)
        }

        $(
            impl $trait<$layout> for $ty {
                fn morph_buffer_size(roi: Size) -> Result<usize> {
                    $direct(roi)
                }
            }
        )*
    };
}

impl_morph_buffer_size!(morph_buffer_size_u8_c1, nppiMorphGetBufferSize_8u_C1R);
impl_morph_buffer_size!(morph_buffer_size_u8_c3, nppiMorphGetBufferSize_8u_C3R);
impl_morph_buffer_size!(morph_buffer_size_u8_c4, nppiMorphGetBufferSize_8u_C4R);
impl_morph_buffer_size!(morph_buffer_size_u16_c1, nppiMorphGetBufferSize_16u_C1R);
impl_morph_buffer_size!(morph_buffer_size_i16_c1, nppiMorphGetBufferSize_16s_C1R);
impl_morph_buffer_size!(morph_buffer_size_f32_c1, nppiMorphGetBufferSize_32f_C1R);
impl_morph_buffer_size!(morph_buffer_size_f32_c3, nppiMorphGetBufferSize_32f_C3R);
impl_morph_buffer_size!(morph_buffer_size_f32_c4, nppiMorphGetBufferSize_32f_C4R);

impl_generic_morph_buffer_size!(MorphBufferSizeC1, morph_buffer_size_c1, C1, [
    u8 => morph_buffer_size_u8_c1,
    u16 => morph_buffer_size_u16_c1,
    i16 => morph_buffer_size_i16_c1,
    f32 => morph_buffer_size_f32_c1,
]);
impl_generic_morph_buffer_size!(MorphBufferSizeC3, morph_buffer_size_c3, C3, [
    u8 => morph_buffer_size_u8_c3,
    f32 => morph_buffer_size_f32_c3,
]);
impl_generic_morph_buffer_size!(MorphBufferSizeC4, morph_buffer_size_c4, C4, [
    u8 => morph_buffer_size_u8_c4,
    f32 => morph_buffer_size_f32_c4,
]);

macro_rules! impl_morphology_composite_border {
    ($with_scratch_name:ident, $name:ident, $ty:ty, $layout:ty, $buffer_size:ident, $ffi:ident) => {
        pub fn $with_scratch_name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            source_offset: Point,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            mask: &[u8],
            mask_size: Size,
            anchor: Point,
            scratch: &mut ScratchBuffer,
            border_type: BorderType,
        ) -> Result<()> {
            validate_border_roi(source.size(), source_offset, destination.size())?;
            validate_mask(mask, mask_size, anchor)?;
            scratch.require($buffer_size(destination.size())?)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    source_offset.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    mask.as_ptr().cast(),
                    mask_size.into(),
                    anchor.into(),
                    scratch.as_mut_ptr().cast(),
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            source_offset: Point,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            mask: &[u8],
            mask_size: Size,
            anchor: Point,
            border_type: BorderType,
        ) -> Result<()> {
            let mut scratch = ScratchBuffer::create($buffer_size(destination.size())?)?;
            $with_scratch_name(
                stream_context,
                source,
                source_offset,
                destination,
                mask,
                mask_size,
                anchor,
                &mut scratch,
                border_type,
            )
        }
    };
}

macro_rules! impl_generic_morphology_composite_border {
    (
        $trait:ident,
        $method_with_scratch:ident,
        $method:ident,
        $function_with_scratch:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct_with_scratch:ident, $direct:ident),* $(,)?]
    ) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method_with_scratch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                mask: &[u8],
                mask_size: Size,
                anchor: Point,
                scratch: &mut ScratchBuffer,
                border_type: BorderType,
            ) -> Result<()>;

            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                mask: &[u8],
                mask_size: Size,
                anchor: Point,
                border_type: BorderType,
            ) -> Result<()>;
        }

        pub fn $function_with_scratch<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            source_offset: Point,
            destination: &mut ImageViewMut<'_, T, $layout>,
            mask: &[u8],
            mask_size: Size,
            anchor: Point,
            scratch: &mut ScratchBuffer,
            border_type: BorderType,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method_with_scratch(
                stream_context,
                source,
                source_offset,
                destination,
                mask,
                mask_size,
                anchor,
                scratch,
                border_type,
            )
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            source_offset: Point,
            destination: &mut ImageViewMut<'_, T, $layout>,
            mask: &[u8],
            mask_size: Size,
            anchor: Point,
            border_type: BorderType,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(
                stream_context,
                source,
                source_offset,
                destination,
                mask,
                mask_size,
                anchor,
                border_type,
            )
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method_with_scratch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    source_offset: Point,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    mask: &[u8],
                    mask_size: Size,
                    anchor: Point,
                    scratch: &mut ScratchBuffer,
                    border_type: BorderType,
                ) -> Result<()> {
                    $direct_with_scratch(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        mask,
                        mask_size,
                        anchor,
                        scratch,
                        border_type,
                    )
                }

                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    source_offset: Point,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    mask: &[u8],
                    mask_size: Size,
                    anchor: Point,
                    border_type: BorderType,
                ) -> Result<()> {
                    $direct(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        mask,
                        mask_size,
                        anchor,
                        border_type,
                    )
                }
            }
        )*
    };
}

impl_morphology_composite_border!(
    morph_close_border_u8_c1_with_scratch,
    morph_close_border_u8_c1,
    u8,
    C1,
    morph_buffer_size_u8_c1,
    nppiMorphCloseBorder_8u_C1R_Ctx
);
impl_morphology_composite_border!(
    morph_close_border_u8_c3_with_scratch,
    morph_close_border_u8_c3,
    u8,
    C3,
    morph_buffer_size_u8_c3,
    nppiMorphCloseBorder_8u_C3R_Ctx
);
impl_morphology_composite_border!(
    morph_close_border_u8_c4_with_scratch,
    morph_close_border_u8_c4,
    u8,
    C4,
    morph_buffer_size_u8_c4,
    nppiMorphCloseBorder_8u_C4R_Ctx
);
impl_morphology_composite_border!(
    morph_close_border_u16_c1_with_scratch,
    morph_close_border_u16_c1,
    u16,
    C1,
    morph_buffer_size_u16_c1,
    nppiMorphCloseBorder_16u_C1R_Ctx
);
impl_morphology_composite_border!(
    morph_close_border_i16_c1_with_scratch,
    morph_close_border_i16_c1,
    i16,
    C1,
    morph_buffer_size_i16_c1,
    nppiMorphCloseBorder_16s_C1R_Ctx
);
impl_morphology_composite_border!(
    morph_close_border_f32_c1_with_scratch,
    morph_close_border_f32_c1,
    f32,
    C1,
    morph_buffer_size_f32_c1,
    nppiMorphCloseBorder_32f_C1R_Ctx
);
impl_morphology_composite_border!(
    morph_close_border_f32_c3_with_scratch,
    morph_close_border_f32_c3,
    f32,
    C3,
    morph_buffer_size_f32_c3,
    nppiMorphCloseBorder_32f_C3R_Ctx
);
impl_morphology_composite_border!(
    morph_close_border_f32_c4_with_scratch,
    morph_close_border_f32_c4,
    f32,
    C4,
    morph_buffer_size_f32_c4,
    nppiMorphCloseBorder_32f_C4R_Ctx
);

impl_morphology_composite_border!(
    morph_open_border_u8_c1_with_scratch,
    morph_open_border_u8_c1,
    u8,
    C1,
    morph_buffer_size_u8_c1,
    nppiMorphOpenBorder_8u_C1R_Ctx
);
impl_morphology_composite_border!(
    morph_open_border_u8_c3_with_scratch,
    morph_open_border_u8_c3,
    u8,
    C3,
    morph_buffer_size_u8_c3,
    nppiMorphOpenBorder_8u_C3R_Ctx
);
impl_morphology_composite_border!(
    morph_open_border_u8_c4_with_scratch,
    morph_open_border_u8_c4,
    u8,
    C4,
    morph_buffer_size_u8_c4,
    nppiMorphOpenBorder_8u_C4R_Ctx
);
impl_morphology_composite_border!(
    morph_open_border_u16_c1_with_scratch,
    morph_open_border_u16_c1,
    u16,
    C1,
    morph_buffer_size_u16_c1,
    nppiMorphOpenBorder_16u_C1R_Ctx
);
impl_morphology_composite_border!(
    morph_open_border_i16_c1_with_scratch,
    morph_open_border_i16_c1,
    i16,
    C1,
    morph_buffer_size_i16_c1,
    nppiMorphOpenBorder_16s_C1R_Ctx
);
impl_morphology_composite_border!(
    morph_open_border_f32_c1_with_scratch,
    morph_open_border_f32_c1,
    f32,
    C1,
    morph_buffer_size_f32_c1,
    nppiMorphOpenBorder_32f_C1R_Ctx
);
impl_morphology_composite_border!(
    morph_open_border_f32_c3_with_scratch,
    morph_open_border_f32_c3,
    f32,
    C3,
    morph_buffer_size_f32_c3,
    nppiMorphOpenBorder_32f_C3R_Ctx
);
impl_morphology_composite_border!(
    morph_open_border_f32_c4_with_scratch,
    morph_open_border_f32_c4,
    f32,
    C4,
    morph_buffer_size_f32_c4,
    nppiMorphOpenBorder_32f_C4R_Ctx
);

impl_morphology_composite_border!(
    morph_top_hat_border_u8_c1_with_scratch,
    morph_top_hat_border_u8_c1,
    u8,
    C1,
    morph_buffer_size_u8_c1,
    nppiMorphTopHatBorder_8u_C1R_Ctx
);
impl_morphology_composite_border!(
    morph_top_hat_border_u8_c3_with_scratch,
    morph_top_hat_border_u8_c3,
    u8,
    C3,
    morph_buffer_size_u8_c3,
    nppiMorphTopHatBorder_8u_C3R_Ctx
);
impl_morphology_composite_border!(
    morph_top_hat_border_u8_c4_with_scratch,
    morph_top_hat_border_u8_c4,
    u8,
    C4,
    morph_buffer_size_u8_c4,
    nppiMorphTopHatBorder_8u_C4R_Ctx
);
impl_morphology_composite_border!(
    morph_top_hat_border_u16_c1_with_scratch,
    morph_top_hat_border_u16_c1,
    u16,
    C1,
    morph_buffer_size_u16_c1,
    nppiMorphTopHatBorder_16u_C1R_Ctx
);
impl_morphology_composite_border!(
    morph_top_hat_border_i16_c1_with_scratch,
    morph_top_hat_border_i16_c1,
    i16,
    C1,
    morph_buffer_size_i16_c1,
    nppiMorphTopHatBorder_16s_C1R_Ctx
);
impl_morphology_composite_border!(
    morph_top_hat_border_f32_c1_with_scratch,
    morph_top_hat_border_f32_c1,
    f32,
    C1,
    morph_buffer_size_f32_c1,
    nppiMorphTopHatBorder_32f_C1R_Ctx
);
impl_morphology_composite_border!(
    morph_top_hat_border_f32_c3_with_scratch,
    morph_top_hat_border_f32_c3,
    f32,
    C3,
    morph_buffer_size_f32_c3,
    nppiMorphTopHatBorder_32f_C3R_Ctx
);
impl_morphology_composite_border!(
    morph_top_hat_border_f32_c4_with_scratch,
    morph_top_hat_border_f32_c4,
    f32,
    C4,
    morph_buffer_size_f32_c4,
    nppiMorphTopHatBorder_32f_C4R_Ctx
);

impl_morphology_composite_border!(
    morph_black_hat_border_u8_c1_with_scratch,
    morph_black_hat_border_u8_c1,
    u8,
    C1,
    morph_buffer_size_u8_c1,
    nppiMorphBlackHatBorder_8u_C1R_Ctx
);
impl_morphology_composite_border!(
    morph_black_hat_border_u8_c3_with_scratch,
    morph_black_hat_border_u8_c3,
    u8,
    C3,
    morph_buffer_size_u8_c3,
    nppiMorphBlackHatBorder_8u_C3R_Ctx
);
impl_morphology_composite_border!(
    morph_black_hat_border_u8_c4_with_scratch,
    morph_black_hat_border_u8_c4,
    u8,
    C4,
    morph_buffer_size_u8_c4,
    nppiMorphBlackHatBorder_8u_C4R_Ctx
);
impl_morphology_composite_border!(
    morph_black_hat_border_u16_c1_with_scratch,
    morph_black_hat_border_u16_c1,
    u16,
    C1,
    morph_buffer_size_u16_c1,
    nppiMorphBlackHatBorder_16u_C1R_Ctx
);
impl_morphology_composite_border!(
    morph_black_hat_border_i16_c1_with_scratch,
    morph_black_hat_border_i16_c1,
    i16,
    C1,
    morph_buffer_size_i16_c1,
    nppiMorphBlackHatBorder_16s_C1R_Ctx
);
impl_morphology_composite_border!(
    morph_black_hat_border_f32_c1_with_scratch,
    morph_black_hat_border_f32_c1,
    f32,
    C1,
    morph_buffer_size_f32_c1,
    nppiMorphBlackHatBorder_32f_C1R_Ctx
);
impl_morphology_composite_border!(
    morph_black_hat_border_f32_c3_with_scratch,
    morph_black_hat_border_f32_c3,
    f32,
    C3,
    morph_buffer_size_f32_c3,
    nppiMorphBlackHatBorder_32f_C3R_Ctx
);
impl_morphology_composite_border!(
    morph_black_hat_border_f32_c4_with_scratch,
    morph_black_hat_border_f32_c4,
    f32,
    C4,
    morph_buffer_size_f32_c4,
    nppiMorphBlackHatBorder_32f_C4R_Ctx
);

impl_morphology_composite_border!(
    morph_gradient_border_u8_c1_with_scratch,
    morph_gradient_border_u8_c1,
    u8,
    C1,
    morph_buffer_size_u8_c1,
    nppiMorphGradientBorder_8u_C1R_Ctx
);
impl_morphology_composite_border!(
    morph_gradient_border_u8_c3_with_scratch,
    morph_gradient_border_u8_c3,
    u8,
    C3,
    morph_buffer_size_u8_c3,
    nppiMorphGradientBorder_8u_C3R_Ctx
);
impl_morphology_composite_border!(
    morph_gradient_border_u8_c4_with_scratch,
    morph_gradient_border_u8_c4,
    u8,
    C4,
    morph_buffer_size_u8_c4,
    nppiMorphGradientBorder_8u_C4R_Ctx
);
impl_morphology_composite_border!(
    morph_gradient_border_u16_c1_with_scratch,
    morph_gradient_border_u16_c1,
    u16,
    C1,
    morph_buffer_size_u16_c1,
    nppiMorphGradientBorder_16u_C1R_Ctx
);
impl_morphology_composite_border!(
    morph_gradient_border_i16_c1_with_scratch,
    morph_gradient_border_i16_c1,
    i16,
    C1,
    morph_buffer_size_i16_c1,
    nppiMorphGradientBorder_16s_C1R_Ctx
);
impl_morphology_composite_border!(
    morph_gradient_border_f32_c1_with_scratch,
    morph_gradient_border_f32_c1,
    f32,
    C1,
    morph_buffer_size_f32_c1,
    nppiMorphGradientBorder_32f_C1R_Ctx
);
impl_morphology_composite_border!(
    morph_gradient_border_f32_c3_with_scratch,
    morph_gradient_border_f32_c3,
    f32,
    C3,
    morph_buffer_size_f32_c3,
    nppiMorphGradientBorder_32f_C3R_Ctx
);
impl_morphology_composite_border!(
    morph_gradient_border_f32_c4_with_scratch,
    morph_gradient_border_f32_c4,
    f32,
    C4,
    morph_buffer_size_f32_c4,
    nppiMorphGradientBorder_32f_C4R_Ctx
);

impl_generic_morphology_composite_border!(
    MorphCloseBorderC1,
    morph_close_border_with_scratch,
    morph_close_border,
    morph_close_border_c1_with_scratch,
    morph_close_border_c1,
    C1,
    [
        u8 => morph_close_border_u8_c1_with_scratch, morph_close_border_u8_c1,
        u16 => morph_close_border_u16_c1_with_scratch, morph_close_border_u16_c1,
        i16 => morph_close_border_i16_c1_with_scratch, morph_close_border_i16_c1,
        f32 => morph_close_border_f32_c1_with_scratch, morph_close_border_f32_c1,
    ]
);
impl_generic_morphology_composite_border!(
    MorphCloseBorderC3,
    morph_close_border_with_scratch,
    morph_close_border,
    morph_close_border_c3_with_scratch,
    morph_close_border_c3,
    C3,
    [
        u8 => morph_close_border_u8_c3_with_scratch, morph_close_border_u8_c3,
        f32 => morph_close_border_f32_c3_with_scratch, morph_close_border_f32_c3,
    ]
);
impl_generic_morphology_composite_border!(
    MorphCloseBorderC4,
    morph_close_border_with_scratch,
    morph_close_border,
    morph_close_border_c4_with_scratch,
    morph_close_border_c4,
    C4,
    [
        u8 => morph_close_border_u8_c4_with_scratch, morph_close_border_u8_c4,
        f32 => morph_close_border_f32_c4_with_scratch, morph_close_border_f32_c4,
    ]
);
impl_generic_morphology_composite_border!(
    MorphOpenBorderC1,
    morph_open_border_with_scratch,
    morph_open_border,
    morph_open_border_c1_with_scratch,
    morph_open_border_c1,
    C1,
    [
        u8 => morph_open_border_u8_c1_with_scratch, morph_open_border_u8_c1,
        u16 => morph_open_border_u16_c1_with_scratch, morph_open_border_u16_c1,
        i16 => morph_open_border_i16_c1_with_scratch, morph_open_border_i16_c1,
        f32 => morph_open_border_f32_c1_with_scratch, morph_open_border_f32_c1,
    ]
);
impl_generic_morphology_composite_border!(
    MorphOpenBorderC3,
    morph_open_border_with_scratch,
    morph_open_border,
    morph_open_border_c3_with_scratch,
    morph_open_border_c3,
    C3,
    [
        u8 => morph_open_border_u8_c3_with_scratch, morph_open_border_u8_c3,
        f32 => morph_open_border_f32_c3_with_scratch, morph_open_border_f32_c3,
    ]
);
impl_generic_morphology_composite_border!(
    MorphOpenBorderC4,
    morph_open_border_with_scratch,
    morph_open_border,
    morph_open_border_c4_with_scratch,
    morph_open_border_c4,
    C4,
    [
        u8 => morph_open_border_u8_c4_with_scratch, morph_open_border_u8_c4,
        f32 => morph_open_border_f32_c4_with_scratch, morph_open_border_f32_c4,
    ]
);
impl_generic_morphology_composite_border!(
    MorphTopHatBorderC1,
    morph_top_hat_border_with_scratch,
    morph_top_hat_border,
    morph_top_hat_border_c1_with_scratch,
    morph_top_hat_border_c1,
    C1,
    [
        u8 => morph_top_hat_border_u8_c1_with_scratch, morph_top_hat_border_u8_c1,
        u16 => morph_top_hat_border_u16_c1_with_scratch, morph_top_hat_border_u16_c1,
        i16 => morph_top_hat_border_i16_c1_with_scratch, morph_top_hat_border_i16_c1,
        f32 => morph_top_hat_border_f32_c1_with_scratch, morph_top_hat_border_f32_c1,
    ]
);
impl_generic_morphology_composite_border!(
    MorphTopHatBorderC3,
    morph_top_hat_border_with_scratch,
    morph_top_hat_border,
    morph_top_hat_border_c3_with_scratch,
    morph_top_hat_border_c3,
    C3,
    [
        u8 => morph_top_hat_border_u8_c3_with_scratch, morph_top_hat_border_u8_c3,
        f32 => morph_top_hat_border_f32_c3_with_scratch, morph_top_hat_border_f32_c3,
    ]
);
impl_generic_morphology_composite_border!(
    MorphTopHatBorderC4,
    morph_top_hat_border_with_scratch,
    morph_top_hat_border,
    morph_top_hat_border_c4_with_scratch,
    morph_top_hat_border_c4,
    C4,
    [
        u8 => morph_top_hat_border_u8_c4_with_scratch, morph_top_hat_border_u8_c4,
        f32 => morph_top_hat_border_f32_c4_with_scratch, morph_top_hat_border_f32_c4,
    ]
);
impl_generic_morphology_composite_border!(
    MorphBlackHatBorderC1,
    morph_black_hat_border_with_scratch,
    morph_black_hat_border,
    morph_black_hat_border_c1_with_scratch,
    morph_black_hat_border_c1,
    C1,
    [
        u8 => morph_black_hat_border_u8_c1_with_scratch, morph_black_hat_border_u8_c1,
        u16 => morph_black_hat_border_u16_c1_with_scratch, morph_black_hat_border_u16_c1,
        i16 => morph_black_hat_border_i16_c1_with_scratch, morph_black_hat_border_i16_c1,
        f32 => morph_black_hat_border_f32_c1_with_scratch, morph_black_hat_border_f32_c1,
    ]
);
impl_generic_morphology_composite_border!(
    MorphBlackHatBorderC3,
    morph_black_hat_border_with_scratch,
    morph_black_hat_border,
    morph_black_hat_border_c3_with_scratch,
    morph_black_hat_border_c3,
    C3,
    [
        u8 => morph_black_hat_border_u8_c3_with_scratch, morph_black_hat_border_u8_c3,
        f32 => morph_black_hat_border_f32_c3_with_scratch, morph_black_hat_border_f32_c3,
    ]
);
impl_generic_morphology_composite_border!(
    MorphBlackHatBorderC4,
    morph_black_hat_border_with_scratch,
    morph_black_hat_border,
    morph_black_hat_border_c4_with_scratch,
    morph_black_hat_border_c4,
    C4,
    [
        u8 => morph_black_hat_border_u8_c4_with_scratch, morph_black_hat_border_u8_c4,
        f32 => morph_black_hat_border_f32_c4_with_scratch, morph_black_hat_border_f32_c4,
    ]
);
impl_generic_morphology_composite_border!(
    MorphGradientBorderC1,
    morph_gradient_border_with_scratch,
    morph_gradient_border,
    morph_gradient_border_c1_with_scratch,
    morph_gradient_border_c1,
    C1,
    [
        u8 => morph_gradient_border_u8_c1_with_scratch, morph_gradient_border_u8_c1,
        u16 => morph_gradient_border_u16_c1_with_scratch, morph_gradient_border_u16_c1,
        i16 => morph_gradient_border_i16_c1_with_scratch, morph_gradient_border_i16_c1,
        f32 => morph_gradient_border_f32_c1_with_scratch, morph_gradient_border_f32_c1,
    ]
);
impl_generic_morphology_composite_border!(
    MorphGradientBorderC3,
    morph_gradient_border_with_scratch,
    morph_gradient_border,
    morph_gradient_border_c3_with_scratch,
    morph_gradient_border_c3,
    C3,
    [
        u8 => morph_gradient_border_u8_c3_with_scratch, morph_gradient_border_u8_c3,
        f32 => morph_gradient_border_f32_c3_with_scratch, morph_gradient_border_f32_c3,
    ]
);
impl_generic_morphology_composite_border!(
    MorphGradientBorderC4,
    morph_gradient_border_with_scratch,
    morph_gradient_border,
    morph_gradient_border_c4_with_scratch,
    morph_gradient_border_c4,
    C4,
    [
        u8 => morph_gradient_border_u8_c4_with_scratch, morph_gradient_border_u8_c4,
        f32 => morph_gradient_border_f32_c4_with_scratch, morph_gradient_border_f32_c4,
    ]
);
