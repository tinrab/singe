use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageView, ImageViewMut},
};

pub trait BorderImage<T, L> {
    fn copy_replicate_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        top: usize,
        left: usize,
    ) -> Result<()>;

    fn copy_wrap_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        top: usize,
        left: usize,
    ) -> Result<()>;
}

macro_rules! impl_border_image {
    ($ty:ty, $layout:ty, $replicate_border:path, $wrap_border:path) => {
        impl<'a> BorderImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn copy_replicate_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                top: usize,
                left: usize,
            ) -> Result<()> {
                $replicate_border(stream_context, source, destination, top, left)
            }

            fn copy_wrap_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                top: usize,
                left: usize,
            ) -> Result<()> {
                $wrap_border(stream_context, source, destination, top, left)
            }
        }
    };
}

#[path = "border_replicate_wrap_dispatch_impls.rs"]
mod impls;
