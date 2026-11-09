use crate::pipeline::image::ImagePipeline;

macro_rules! impl_edge_directional_border_filter_image {
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
        unsafe impl<'a> EdgeDirectionalBorderFilterImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn filter_prewitt_horizontal_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: $crate::types::Point,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                border_type: $crate::types::BorderType,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $prewitt_horizontal(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        border_type,
                    )
                })
            }

            fn filter_prewitt_vertical_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: $crate::types::Point,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                border_type: $crate::types::BorderType,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $prewitt_vertical(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        border_type,
                    )
                })
            }

            fn filter_roberts_down_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: $crate::types::Point,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                border_type: $crate::types::BorderType,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $roberts_down(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        border_type,
                    )
                })
            }

            fn filter_roberts_up_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: $crate::types::Point,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                border_type: $crate::types::BorderType,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $roberts_up(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        border_type,
                    )
                })
            }

            fn filter_sobel_horizontal_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: $crate::types::Point,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                border_type: $crate::types::BorderType,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $sobel_horizontal(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        border_type,
                    )
                })
            }

            fn filter_sobel_vertical_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: $crate::types::Point,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                border_type: $crate::types::BorderType,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $sobel_vertical(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        border_type,
                    )
                })
            }
        }
    };
}

mod edge_border;
mod edge_directional;
mod edge_sharpen;
mod sharpen_border;
