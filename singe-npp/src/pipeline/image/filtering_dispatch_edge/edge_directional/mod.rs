macro_rules! impl_edge_directional_filter_image {
    (
        $ty:ty,
        $layout:ty,
        $prewitt_horizontal:path,
        $prewitt_vertical:path,
        $roberts_down:path,
        $roberts_up:path,
        $sobel_horizontal:path,
        $sobel_vertical:path
    ) => {
        unsafe impl<'a> EdgeDirectionalFilterImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn filter_prewitt_horizontal_image(
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
                    $prewitt_horizontal(stream_context, &source.view(), destination)
                })
            }

            fn filter_prewitt_vertical_image(
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
                    $prewitt_vertical(stream_context, &source.view(), destination)
                })
            }

            fn filter_roberts_down_image(
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
                    $roberts_down(stream_context, &source.view(), destination)
                })
            }

            fn filter_roberts_up_image(
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
                    $roberts_up(stream_context, &source.view(), destination)
                })
            }

            fn filter_sobel_horizontal_image(
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
                    $sobel_horizontal(stream_context, &source.view(), destination)
                })
            }

            fn filter_sobel_vertical_image(
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
                    $sobel_vertical(stream_context, &source.view(), destination)
                })
            }
        }
    };
}

mod float_impls;
mod integer_impls;
