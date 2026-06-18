macro_rules! impl_mirror_batch_image {
    ($ty:ty, $layout:ty, $mirror:path, $mirror_in_place:path) => {
        impl<'a> ImagePipeline<'a, $ty, $layout> {
            pub fn mirror_batch(
                stream_context: &StreamContext,
                axis: crate::types::Axis,
                sources: &[ImageView<'_, $ty, $layout>],
                destinations: &mut [ImageViewMut<'_, $ty, $layout>],
            ) -> Result<()> {
                $mirror(stream_context, axis, sources, destinations)
            }

            pub fn mirror_batch_in_place(
                stream_context: &StreamContext,
                axis: crate::types::Axis,
                images: &mut [ImageViewMut<'_, $ty, $layout>],
            ) -> Result<()> {
                $mirror_in_place(stream_context, axis, images)
            }
        }
    };
}

macro_rules! impl_resize_batch_image {
    ($ty:ty, $layout:ty, $resize:path, $resize_advanced:path) => {
        impl<'a> ImagePipeline<'a, $ty, $layout> {
            pub fn resize_batch(
                stream_context: &StreamContext,
                resize: &geometry::Resize,
                sources: &[ImageView<'_, $ty, $layout>],
                destinations: &mut [ImageViewMut<'_, $ty, $layout>],
            ) -> Result<()> {
                $resize(stream_context, resize, sources, destinations)
            }

            pub fn resize_batch_advanced(
                stream_context: &StreamContext,
                rois: &[geometry::ResizeBatchAdvanced],
                interpolation: crate::types::InterpolationMode,
                sources: &[ImageView<'_, $ty, $layout>],
                destinations: &mut [ImageViewMut<'_, $ty, $layout>],
            ) -> Result<()> {
                $resize_advanced(stream_context, rois, interpolation, sources, destinations)
            }
        }
    };
}

macro_rules! impl_resize_batch_advanced_image {
    ($ty:ty, $layout:ty, $resize_advanced:path) => {
        impl<'a> ImagePipeline<'a, $ty, $layout> {
            pub fn resize_batch_advanced(
                stream_context: &StreamContext,
                rois: &[geometry::ResizeBatchAdvanced],
                interpolation: crate::types::InterpolationMode,
                sources: &[ImageView<'_, $ty, $layout>],
                destinations: &mut [ImageViewMut<'_, $ty, $layout>],
            ) -> Result<()> {
                $resize_advanced(stream_context, rois, interpolation, sources, destinations)
            }
        }
    };
}

macro_rules! impl_warp_batch_image {
    ($ty:ty, $layout:ty, $affine:path, $perspective:path) => {
        impl<'a> ImagePipeline<'a, $ty, $layout> {
            pub fn warp_affine_batch(
                stream_context: &StreamContext,
                warp: &geometry::WarpAffineBatch,
                coefficients: &[crate::types::AffineCoefficients],
                sources: &[ImageView<'_, $ty, $layout>],
                destinations: &mut [ImageViewMut<'_, $ty, $layout>],
            ) -> Result<()> {
                $affine(stream_context, warp, coefficients, sources, destinations)
            }

            pub fn warp_perspective_batch(
                stream_context: &StreamContext,
                warp: &geometry::WarpPerspectiveBatch,
                coefficients: &[crate::types::PerspectiveCoefficients],
                sources: &[ImageView<'_, $ty, $layout>],
                destinations: &mut [ImageViewMut<'_, $ty, $layout>],
            ) -> Result<()> {
                $perspective(stream_context, warp, coefficients, sources, destinations)
            }
        }
    };
}
