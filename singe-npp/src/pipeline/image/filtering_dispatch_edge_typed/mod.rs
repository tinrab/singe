mod border;
mod directional;

macro_rules! impl_sobel_extended_filter_image {
    (
        $ty:ty,
        $layout:ty,
        $horizontal_mask:path,
        $vertical_mask:path,
        $horizontal_second:path,
        $vertical_second:path,
        $cross:path
    ) => {
        unsafe impl<'a> SobelExtendedFilterImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn filter_sobel_horizontal_mask_image(
                stream_context: &StreamContext,
                source: &$crate::image::view::ImageWindow<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                mask_size: MaskSize,
            ) -> Result<()> {
                let (mask, anchor) =
                    $crate::pipeline::image::filtering::centered_filter_geometry(mask_size)?;
                source.require_halo(mask, anchor)?;
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $horizontal_mask(stream_context, &source.view(), destination, mask_size)
                })
            }

            fn filter_sobel_vertical_mask_image(
                stream_context: &StreamContext,
                source: &$crate::image::view::ImageWindow<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                mask_size: MaskSize,
            ) -> Result<()> {
                let (mask, anchor) =
                    $crate::pipeline::image::filtering::centered_filter_geometry(mask_size)?;
                source.require_halo(mask, anchor)?;
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $vertical_mask(stream_context, &source.view(), destination, mask_size)
                })
            }

            fn filter_sobel_horizontal_second_image(
                stream_context: &StreamContext,
                source: &$crate::image::view::ImageWindow<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                mask_size: MaskSize,
            ) -> Result<()> {
                let (mask, anchor) =
                    $crate::pipeline::image::filtering::centered_filter_geometry(mask_size)?;
                source.require_halo(mask, anchor)?;
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $horizontal_second(stream_context, &source.view(), destination, mask_size)
                })
            }

            fn filter_sobel_vertical_second_image(
                stream_context: &StreamContext,
                source: &$crate::image::view::ImageWindow<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                mask_size: MaskSize,
            ) -> Result<()> {
                let (mask, anchor) =
                    $crate::pipeline::image::filtering::centered_filter_geometry(mask_size)?;
                source.require_halo(mask, anchor)?;
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $vertical_second(stream_context, &source.view(), destination, mask_size)
                })
            }

            fn filter_sobel_cross_image(
                stream_context: &StreamContext,
                source: &$crate::image::view::ImageWindow<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                mask_size: MaskSize,
            ) -> Result<()> {
                let (mask, anchor) =
                    $crate::pipeline::image::filtering::centered_filter_geometry(mask_size)?;
                source.require_halo(mask, anchor)?;
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $cross(stream_context, &source.view(), destination, mask_size)
                })
            }
        }
    };
}

macro_rules! impl_scharr_filter_image {
    ($ty:ty, $layout:ty, $horizontal:path, $vertical:path) => {
        unsafe impl<'a> ScharrFilterImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn filter_scharr_horizontal_image(
                stream_context: &StreamContext,
                source: &ImageWindow<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
            ) -> Result<()> {
                source.require_halo(
                    $crate::types::Size::new(3, 3),
                    $crate::types::Point { x: 1, y: 1 },
                )?;
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $horizontal(stream_context, &source.view(), destination)
                })
            }

            fn filter_scharr_vertical_image(
                stream_context: &StreamContext,
                source: &ImageWindow<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
            ) -> Result<()> {
                source.require_halo(
                    $crate::types::Size::new(3, 3),
                    $crate::types::Point { x: 1, y: 1 },
                )?;
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $vertical(stream_context, &source.view(), destination)
                })
            }
        }
    };
}

mod f32_impls;
