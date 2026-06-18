use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageView, ImageViewMut, PlanarImageView, PlanarImageViewMut},
};

pub trait GammaImage<L> {
    fn gamma_forward_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, L>,
        destination: &mut ImageViewMut<'_, u8, L>,
    ) -> Result<()>;

    fn gamma_forward_image_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, u8, L>,
    ) -> Result<()>;

    fn gamma_inverse_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, L>,
        destination: &mut ImageViewMut<'_, u8, L>,
    ) -> Result<()>;

    fn gamma_inverse_image_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, u8, L>,
    ) -> Result<()>;
}

pub trait PlanarGammaImage {
    fn gamma_forward_planar_image(
        stream_context: &StreamContext,
        source: &PlanarImageView<'_, u8, 3>,
        destination: &mut PlanarImageViewMut<'_, u8, 3>,
    ) -> Result<()>;

    fn gamma_forward_planar_image_in_place(
        stream_context: &StreamContext,
        image: &mut PlanarImageViewMut<'_, u8, 3>,
    ) -> Result<()>;

    fn gamma_inverse_planar_image(
        stream_context: &StreamContext,
        source: &PlanarImageView<'_, u8, 3>,
        destination: &mut PlanarImageViewMut<'_, u8, 3>,
    ) -> Result<()>;

    fn gamma_inverse_planar_image_in_place(
        stream_context: &StreamContext,
        image: &mut PlanarImageViewMut<'_, u8, 3>,
    ) -> Result<()>;
}

macro_rules! impl_gamma_image {
    ($layout:ty, $forward:path, $forward_in_place:path, $inverse:path, $inverse_in_place:path) => {
        impl<'a> GammaImage<$layout> for ImagePipeline<'a, u8, $layout> {
            fn gamma_forward_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, u8, $layout>,
                destination: &mut ImageViewMut<'_, u8, $layout>,
            ) -> Result<()> {
                $forward(stream_context, source, destination)
            }

            fn gamma_forward_image_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, u8, $layout>,
            ) -> Result<()> {
                $forward_in_place(stream_context, image)
            }

            fn gamma_inverse_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, u8, $layout>,
                destination: &mut ImageViewMut<'_, u8, $layout>,
            ) -> Result<()> {
                $inverse(stream_context, source, destination)
            }

            fn gamma_inverse_image_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, u8, $layout>,
            ) -> Result<()> {
                $inverse_in_place(stream_context, image)
            }
        }
    };
}

macro_rules! impl_planar_gamma_image {
    ($forward:path, $forward_in_place:path, $inverse:path, $inverse_in_place:path) => {
        impl PlanarGammaImage for ImagePipeline<'_, u8, C1> {
            fn gamma_forward_planar_image(
                stream_context: &StreamContext,
                source: &PlanarImageView<'_, u8, 3>,
                destination: &mut PlanarImageViewMut<'_, u8, 3>,
            ) -> Result<()> {
                $forward(stream_context, source, destination)
            }

            fn gamma_forward_planar_image_in_place(
                stream_context: &StreamContext,
                image: &mut PlanarImageViewMut<'_, u8, 3>,
            ) -> Result<()> {
                $forward_in_place(stream_context, image)
            }

            fn gamma_inverse_planar_image(
                stream_context: &StreamContext,
                source: &PlanarImageView<'_, u8, 3>,
                destination: &mut PlanarImageViewMut<'_, u8, 3>,
            ) -> Result<()> {
                $inverse(stream_context, source, destination)
            }

            fn gamma_inverse_planar_image_in_place(
                stream_context: &StreamContext,
                image: &mut PlanarImageViewMut<'_, u8, 3>,
            ) -> Result<()> {
                $inverse_in_place(stream_context, image)
            }
        }
    };
}
