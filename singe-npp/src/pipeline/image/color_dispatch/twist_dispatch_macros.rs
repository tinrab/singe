macro_rules! impl_color_twist_image {
    ($ty:ty, $layout:ty, $twist:path, $twist_in_place:path) => {
        unsafe impl<'a> ColorTwistImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn color_twist_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                twist: $crate::pipeline::ColorTwistMatrix,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $twist(stream_context, source, destination, twist) })
            }

            fn color_twist_image_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, $ty, $layout>,
                twist: $crate::pipeline::ColorTwistMatrix,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(image)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $twist_in_place(stream_context, image, twist) })
            }
        }
    };
}

macro_rules! impl_planar_color_twist_image {
    ($ty:ty, $twist:path, $twist_in_place:path) => {
        unsafe impl PlanarColorTwistImage<$ty> for ImagePipeline<'_, $ty, C1> {
            fn color_twist_planar_image(
                stream_context: &StreamContext,
                source: &PlanarImageView<'_, $ty, 3>,
                destination: &mut PlanarImageOutputView<'_, $ty, 3>,
                twist: $crate::pipeline::ColorTwistMatrix,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                for plane in source.planes() {
                    stream_context.prepare_region(plane)?;
                }
                for plane in destination.planes() {
                    stream_context.prepare_region(plane)?;
                }
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $twist(stream_context, source, destination, twist) })
            }

            fn color_twist_planar_image_in_place(
                stream_context: &StreamContext,
                image: &mut PlanarImageViewMut<'_, $ty, 3>,
                twist: $crate::pipeline::ColorTwistMatrix,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                for plane in image.planes() {
                    stream_context.prepare_region(plane)?;
                }
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $twist_in_place(stream_context, image, twist) })
            }
        }
    };
}

macro_rules! impl_color_twist_with_constants_image {
    ($ty:ty, $twist:path, $twist_in_place:path) => {
        unsafe impl<'a> ColorTwistWithConstantsImage<$ty> for ImagePipeline<'a, $ty, C4> {
            fn color_twist_with_constants_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C4>,
                destination: &mut ImageOutputView<'_, $ty, C4>,
                twist: $crate::pipeline::ColorTwistMatrix4,
                constants: $crate::pipeline::ColorTwistConstants4,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $twist(stream_context, source, destination, twist, constants)
                })
            }

            fn color_twist_with_constants_image_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, $ty, C4>,
                twist: $crate::pipeline::ColorTwistMatrix4,
                constants: $crate::pipeline::ColorTwistConstants4,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(image)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $twist_in_place(stream_context, image, twist, constants)
                })
            }
        }
    };
}

macro_rules! impl_color_twist_batch_image {
    ($ty:ty, $layout:ty, $twist:path, $twist_in_place:path) => {
        unsafe impl<'a> ColorTwistBatchImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn color_twist_batch_image(
                stream_context: &StreamContext,
                sources: &[ImageView<'_, $ty, $layout>],
                twists: &[ColorTwistMatrix],
                destinations: &mut [ImageOutputView<'_, $ty, $layout>],
                min: f32,
                max: f32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                for image in sources.iter() {
                    stream_context.prepare_region(image)?;
                }
                for image in destinations.iter() {
                    stream_context.prepare_region(image)?;
                }
                // SAFETY: the native helper completes borrowed use on every path.
                unsafe { $twist(stream_context, sources, twists, destinations, min, max) }
            }

            fn color_twist_batch_image_in_place(
                stream_context: &StreamContext,
                images: &mut [ImageViewMut<'_, $ty, $layout>],
                twists: &[ColorTwistMatrix],
                min: f32,
                max: f32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                for image in images.iter() {
                    stream_context.prepare_region(image)?;
                }
                // SAFETY: the native helper completes borrowed use on every path.
                unsafe { $twist_in_place(stream_context, images, twists, min, max) }
            }
        }
    };
}

macro_rules! impl_color_twist_batch_with_constants_image {
    ($ty:ty, $twist:path, $twist_in_place:path) => {
        unsafe impl<'a> ColorTwistBatchWithConstantsImage<$ty> for ImagePipeline<'a, $ty, C4> {
            fn color_twist_batch_with_constants_image(
                stream_context: &StreamContext,
                sources: &[ImageView<'_, $ty, C4>],
                twists: &[ColorTwistBatchConstantsMatrix4],
                destinations: &mut [ImageOutputView<'_, $ty, C4>],
                min: f32,
                max: f32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                for image in sources.iter() {
                    stream_context.prepare_region(image)?;
                }
                for image in destinations.iter() {
                    stream_context.prepare_region(image)?;
                }
                // SAFETY: the native helper completes borrowed use on every path.
                unsafe { $twist(stream_context, sources, twists, destinations, min, max) }
            }

            fn color_twist_batch_with_constants_image_in_place(
                stream_context: &StreamContext,
                images: &mut [ImageViewMut<'_, $ty, C4>],
                twists: &[ColorTwistBatchConstantsMatrix4],
                min: f32,
                max: f32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                for image in images.iter() {
                    stream_context.prepare_region(image)?;
                }
                // SAFETY: the native helper completes borrowed use on every path.
                unsafe { $twist_in_place(stream_context, images, twists, min, max) }
            }
        }
    };
}
