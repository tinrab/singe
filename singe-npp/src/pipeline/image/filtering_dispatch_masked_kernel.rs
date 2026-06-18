use super::ImagePipeline;

macro_rules! impl_masked_kernel_filter_image {
    (
        $ty:ty,
        $layout:ty,
        $high_pass:path,
        $low_pass:path,
        $gauss:path,
        $laplace:path
    ) => {
        impl<'a> MaskedKernelFilterImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn filter_high_pass_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask_size: MaskSize,
            ) -> Result<()> {
                $high_pass(stream_context, source, destination, mask_size)
            }

            fn filter_low_pass_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask_size: MaskSize,
            ) -> Result<()> {
                $low_pass(stream_context, source, destination, mask_size)
            }

            fn filter_gauss_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask_size: MaskSize,
            ) -> Result<()> {
                $gauss(stream_context, source, destination, mask_size)
            }

            fn filter_laplace_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask_size: MaskSize,
            ) -> Result<()> {
                $laplace(stream_context, source, destination, mask_size)
            }
        }
    };
}

macro_rules! impl_masked_kernel_border_filter_image {
    (
        $ty:ty,
        $layout:ty,
        $high_pass:path,
        $low_pass:path,
        $gauss:path,
        $laplace:path
    ) => {
        impl<'a> MaskedKernelBorderFilterImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn filter_high_pass_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask_size: MaskSize,
                border_type: BorderType,
            ) -> Result<()> {
                $high_pass(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    mask_size,
                    border_type,
                )
            }

            fn filter_low_pass_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask_size: MaskSize,
                border_type: BorderType,
            ) -> Result<()> {
                $low_pass(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    mask_size,
                    border_type,
                )
            }

            fn filter_gauss_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask_size: MaskSize,
                border_type: BorderType,
            ) -> Result<()> {
                $gauss(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    mask_size,
                    border_type,
                )
            }

            fn filter_laplace_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask_size: MaskSize,
                border_type: BorderType,
            ) -> Result<()> {
                $laplace(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    mask_size,
                    border_type,
                )
            }
        }
    };
}

#[path = "filtering_dispatch_masked_kernel_impls.rs"]
mod masked_kernel_impls;

#[path = "filtering_dispatch_masked_kernel_border.rs"]
mod masked_kernel_border;
