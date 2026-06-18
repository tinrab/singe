macro_rules! impl_rotate_image {
    ($ty:ty, $layout:ty, $rotate:path) => {
        impl<'a> RotateImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn rotate_image(
                stream_context: &StreamContext,
                rotate: &geometry::Rotate,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $rotate(stream_context, rotate, source, destination)
            }
        }
    };
}

macro_rules! impl_resize_sqr_pixel_image {
    ($ty:ty, $layout:ty, $resize:path) => {
        impl<'a> ResizeSqrPixelImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn resize_sqr_pixel_image(
                stream_context: &StreamContext,
                resize: &geometry::ResizeSqrPixel,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $resize(stream_context, resize, source, destination)
            }
        }
    };
}

macro_rules! impl_resize_sqr_pixel_advanced_image {
    ($ty:ty, $layout:ty, $resize:path) => {
        impl<'a> ResizeSqrPixelAdvancedImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn resize_sqr_pixel_advanced_image(
                stream_context: &StreamContext,
                resize: &geometry::ResizeSqrPixelAdvanced,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $resize(stream_context, resize, source, destination)
            }
        }
    };
}

macro_rules! impl_remap_image {
    ($ty:ty, $map_ty:ty, $layout:ty, $remap:path) => {
        impl<'a> RemapImage<$ty, $map_ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn remap_image(
                stream_context: &StreamContext,
                remap: &geometry::Remap,
                source: &ImageView<'_, $ty, $layout>,
                x_map: &ImageView<'_, $map_ty, C1>,
                y_map: &ImageView<'_, $map_ty, C1>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $remap(stream_context, remap, source, x_map, y_map, destination)
            }
        }
    };
}

macro_rules! impl_warp_affine_image {
    ($ty:ty, $layout:ty, $warp:path) => {
        impl<'a> WarpAffineImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn warp_affine_image(
                stream_context: &StreamContext,
                warp: &geometry::WarpAffine,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $warp(stream_context, warp, source, destination)
            }
        }
    };
}

macro_rules! impl_warp_affine_back_image {
    ($ty:ty, $layout:ty, $warp:path) => {
        impl<'a> WarpAffineBackImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn warp_affine_back_image(
                stream_context: &StreamContext,
                warp: &geometry::WarpAffine,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $warp(stream_context, warp, source, destination)
            }
        }
    };
}

macro_rules! impl_warp_affine_quad_image {
    ($ty:ty, $layout:ty, $warp:path) => {
        impl<'a> WarpAffineQuadImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn warp_affine_quad_image(
                stream_context: &StreamContext,
                warp: &geometry::WarpQuad,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $warp(stream_context, warp, source, destination)
            }
        }
    };
}

macro_rules! impl_warp_perspective_image {
    ($ty:ty, $layout:ty, $warp:path) => {
        impl<'a> WarpPerspectiveImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn warp_perspective_image(
                stream_context: &StreamContext,
                warp: &geometry::WarpPerspective,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $warp(stream_context, warp, source, destination)
            }
        }
    };
}

macro_rules! impl_warp_perspective_back_image {
    ($ty:ty, $layout:ty, $warp:path) => {
        impl<'a> WarpPerspectiveBackImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn warp_perspective_back_image(
                stream_context: &StreamContext,
                warp: &geometry::WarpPerspective,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $warp(stream_context, warp, source, destination)
            }
        }
    };
}

macro_rules! impl_warp_perspective_quad_image {
    ($ty:ty, $layout:ty, $warp:path) => {
        impl<'a> WarpPerspectiveQuadImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn warp_perspective_quad_image(
                stream_context: &StreamContext,
                warp: &geometry::WarpQuad,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $warp(stream_context, warp, source, destination)
            }
        }
    };
}

macro_rules! impl_geometry_for_layout {
    ($ty:ty, $layout:ty, $rotate:path, $resize_sqr_pixel:path, $remap:path, $affine:path, $affine_back:path, $affine_quad:path, $perspective:path, $perspective_back:path, $perspective_quad:path) => {
        impl_rotate_image!($ty, $layout, $rotate);
        impl_resize_sqr_pixel_image!($ty, $layout, $resize_sqr_pixel);
        impl_remap_image!($ty, f32, $layout, $remap);
        impl_warp_affine_image!($ty, $layout, $affine);
        impl_warp_affine_back_image!($ty, $layout, $affine_back);
        impl_warp_affine_quad_image!($ty, $layout, $affine_quad);
        impl_warp_perspective_image!($ty, $layout, $perspective);
        impl_warp_perspective_back_image!($ty, $layout, $perspective_back);
        impl_warp_perspective_quad_image!($ty, $layout, $perspective_quad);
    };
}

macro_rules! impl_warp_for_layout {
    ($ty:ty, $layout:ty, $affine:path, $affine_back:path, $affine_quad:path, $perspective:path, $perspective_back:path, $perspective_quad:path) => {
        impl_warp_affine_image!($ty, $layout, $affine);
        impl_warp_affine_back_image!($ty, $layout, $affine_back);
        impl_warp_affine_quad_image!($ty, $layout, $affine_quad);
        impl_warp_perspective_image!($ty, $layout, $perspective);
        impl_warp_perspective_back_image!($ty, $layout, $perspective_back);
        impl_warp_perspective_quad_image!($ty, $layout, $perspective_quad);
    };
}

macro_rules! impl_affine_perspective_for_layout {
    ($ty:ty, $layout:ty, $affine:path, $perspective:path) => {
        impl_warp_affine_image!($ty, $layout, $affine);
        impl_warp_perspective_image!($ty, $layout, $perspective);
    };
}
