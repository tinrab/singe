macro_rules! impl_mask_morphology_border_image {
    ($ty:ty, $layout:ty, $dilate:path, $erode:path) => {
        impl<'a> MaskMorphologyBorderImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn dilate_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask: &[u8],
                mask_size: Size,
                anchor: Point,
                border_type: BorderType,
            ) -> Result<()> {
                $dilate(
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

            fn erode_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask: &[u8],
                mask_size: Size,
                anchor: Point,
                border_type: BorderType,
            ) -> Result<()> {
                $erode(
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
    };
}

macro_rules! impl_morphology_3x3_border_image {
    ($ty:ty, $layout:ty, $dilate:path, $erode:path) => {
        impl<'a> Morphology3x3BorderImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn dilate_3x3_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                border_type: BorderType,
            ) -> Result<()> {
                $dilate(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    border_type,
                )
            }

            fn erode_3x3_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                border_type: BorderType,
            ) -> Result<()> {
                $erode(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    border_type,
                )
            }
        }
    };
}

macro_rules! impl_gray_mask_morphology_border_image {
    ($ty:ty, $mask_ty:ty, $dilate:path, $erode:path) => {
        impl<'a> GrayMaskMorphologyBorderImage<$ty, $mask_ty> for ImagePipeline<'a, $ty, C1> {
            fn gray_dilate_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $ty, C1>,
                mask: &[$mask_ty],
                mask_size: Size,
                anchor: Point,
                border_type: BorderType,
            ) -> Result<()> {
                $dilate(
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

            fn gray_erode_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $ty, C1>,
                mask: &[$mask_ty],
                mask_size: Size,
                anchor: Point,
                border_type: BorderType,
            ) -> Result<()> {
                $erode(
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
    };
}

#[path = "morphology_3x3_border_dispatch_impls.rs"]
mod morphology_3x3_border_dispatch_impls;

#[path = "morphology_mask_border_dispatch_impls.rs"]
mod morphology_mask_border_dispatch_impls;

#[path = "morphology_gray_mask_border_dispatch_impls.rs"]
mod morphology_gray_mask_border_dispatch_impls;
