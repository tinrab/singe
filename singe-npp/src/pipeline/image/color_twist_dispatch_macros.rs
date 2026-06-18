macro_rules! impl_color_twist_image {
    ($ty:ty, $layout:ty, $twist:path, $twist_in_place:path) => {
        impl<'a> ColorTwistImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn color_twist_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                twist: crate::pipeline::ColorTwistMatrix,
            ) -> Result<()> {
                $twist(stream_context, source, destination, twist)
            }

            fn color_twist_image_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, $ty, $layout>,
                twist: crate::pipeline::ColorTwistMatrix,
            ) -> Result<()> {
                $twist_in_place(stream_context, image, twist)
            }
        }
    };
}

macro_rules! impl_planar_color_twist_image {
    ($ty:ty, $twist:path, $twist_in_place:path) => {
        impl PlanarColorTwistImage<$ty> for ImagePipeline<'_, $ty, C1> {
            fn color_twist_planar_image(
                stream_context: &StreamContext,
                source: &PlanarImageView<'_, $ty, 3>,
                destination: &mut PlanarImageViewMut<'_, $ty, 3>,
                twist: crate::pipeline::ColorTwistMatrix,
            ) -> Result<()> {
                $twist(stream_context, source, destination, twist)
            }

            fn color_twist_planar_image_in_place(
                stream_context: &StreamContext,
                image: &mut PlanarImageViewMut<'_, $ty, 3>,
                twist: crate::pipeline::ColorTwistMatrix,
            ) -> Result<()> {
                $twist_in_place(stream_context, image, twist)
            }
        }
    };
}

macro_rules! impl_color_twist_with_constants_image {
    ($ty:ty, $twist:path, $twist_in_place:path) => {
        impl<'a> ColorTwistWithConstantsImage<$ty> for ImagePipeline<'a, $ty, C4> {
            fn color_twist_with_constants_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C4>,
                destination: &mut ImageViewMut<'_, $ty, C4>,
                twist: crate::pipeline::ColorTwistMatrix4,
                constants: crate::pipeline::ColorTwistConstants4,
            ) -> Result<()> {
                $twist(stream_context, source, destination, twist, constants)
            }

            fn color_twist_with_constants_image_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, $ty, C4>,
                twist: crate::pipeline::ColorTwistMatrix4,
                constants: crate::pipeline::ColorTwistConstants4,
            ) -> Result<()> {
                $twist_in_place(stream_context, image, twist, constants)
            }
        }
    };
}

macro_rules! impl_color_twist_batch_image {
    ($ty:ty, $layout:ty, $twist:path, $twist_in_place:path) => {
        impl<'a> ColorTwistBatchImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn color_twist_batch_image(
                stream_context: &StreamContext,
                sources: &[ImageView<'_, $ty, $layout>],
                twists: &[ColorTwistMatrix],
                destinations: &mut [ImageViewMut<'_, $ty, $layout>],
                min: f32,
                max: f32,
            ) -> Result<()> {
                $twist(stream_context, sources, twists, destinations, min, max)
            }

            fn color_twist_batch_image_in_place(
                stream_context: &StreamContext,
                images: &mut [ImageViewMut<'_, $ty, $layout>],
                twists: &[ColorTwistMatrix],
                min: f32,
                max: f32,
            ) -> Result<()> {
                $twist_in_place(stream_context, images, twists, min, max)
            }
        }
    };
}

macro_rules! impl_color_twist_batch_with_constants_image {
    ($ty:ty, $twist:path, $twist_in_place:path) => {
        impl<'a> ColorTwistBatchWithConstantsImage<$ty> for ImagePipeline<'a, $ty, C4> {
            fn color_twist_batch_with_constants_image(
                stream_context: &StreamContext,
                sources: &[ImageView<'_, $ty, C4>],
                twists: &[ColorTwistBatchConstantsMatrix4],
                destinations: &mut [ImageViewMut<'_, $ty, C4>],
                min: f32,
                max: f32,
            ) -> Result<()> {
                $twist(stream_context, sources, twists, destinations, min, max)
            }

            fn color_twist_batch_with_constants_image_in_place(
                stream_context: &StreamContext,
                images: &mut [ImageViewMut<'_, $ty, C4>],
                twists: &[ColorTwistBatchConstantsMatrix4],
                min: f32,
                max: f32,
            ) -> Result<()> {
                $twist_in_place(stream_context, images, twists, min, max)
            }
        }
    };
}
