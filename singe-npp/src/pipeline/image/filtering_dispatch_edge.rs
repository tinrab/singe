use super::ImagePipeline;

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
        impl<'a> EdgeDirectionalBorderFilterImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn filter_prewitt_horizontal_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: crate::types::Point,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                border_type: crate::types::BorderType,
            ) -> Result<()> {
                $prewitt_horizontal(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    border_type,
                )
            }

            fn filter_prewitt_vertical_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: crate::types::Point,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                border_type: crate::types::BorderType,
            ) -> Result<()> {
                $prewitt_vertical(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    border_type,
                )
            }

            fn filter_roberts_down_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: crate::types::Point,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                border_type: crate::types::BorderType,
            ) -> Result<()> {
                $roberts_down(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    border_type,
                )
            }

            fn filter_roberts_up_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: crate::types::Point,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                border_type: crate::types::BorderType,
            ) -> Result<()> {
                $roberts_up(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    border_type,
                )
            }

            fn filter_sobel_horizontal_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: crate::types::Point,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                border_type: crate::types::BorderType,
            ) -> Result<()> {
                $sobel_horizontal(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    border_type,
                )
            }

            fn filter_sobel_vertical_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: crate::types::Point,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                border_type: crate::types::BorderType,
            ) -> Result<()> {
                $sobel_vertical(
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

#[path = "filtering_dispatch_edge_border.rs"]
mod edge_border;
#[path = "filtering_dispatch_edge_directional.rs"]
mod edge_directional;
#[path = "filtering_dispatch_edge_sharpen.rs"]
mod edge_sharpen;
#[path = "filtering_dispatch_sharpen_border.rs"]
mod sharpen_border;
