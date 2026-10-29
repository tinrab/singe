use crate::{
    context::StreamContext,
    error::Result,
    image::view::{
        ImageOutputView, ImageView, ImageViewMut, PlanarImageOutputView, PlanarImageView,
        PlanarImageViewMut,
    },
};

/// # Safety
/// Complete borrowed native work on every path and uphold the operation's declared write coverage.
pub unsafe trait GammaImage<L> {
    fn gamma_forward_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, L>,
        destination: &mut ImageOutputView<'_, u8, L>,
    ) -> Result<()>;

    fn gamma_forward_image_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, u8, L>,
    ) -> Result<()>;

    fn gamma_inverse_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, L>,
        destination: &mut ImageOutputView<'_, u8, L>,
    ) -> Result<()>;

    fn gamma_inverse_image_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, u8, L>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and uphold the operation's declared write coverage.
pub unsafe trait PlanarGammaImage {
    fn gamma_forward_planar_image(
        stream_context: &StreamContext,
        source: &PlanarImageView<'_, u8, 3>,
        destination: &mut PlanarImageOutputView<'_, u8, 3>,
    ) -> Result<()>;

    fn gamma_forward_planar_image_in_place(
        stream_context: &StreamContext,
        image: &mut PlanarImageViewMut<'_, u8, 3>,
    ) -> Result<()>;

    fn gamma_inverse_planar_image(
        stream_context: &StreamContext,
        source: &PlanarImageView<'_, u8, 3>,
        destination: &mut PlanarImageOutputView<'_, u8, 3>,
    ) -> Result<()>;

    fn gamma_inverse_planar_image_in_place(
        stream_context: &StreamContext,
        image: &mut PlanarImageViewMut<'_, u8, 3>,
    ) -> Result<()>;
}

macro_rules! impl_gamma_image {
    ($layout:ty, $forward:path, $forward_in_place:path, $inverse:path, $inverse_in_place:path) => {
        unsafe impl<'a> GammaImage<$layout> for ImagePipeline<'a, u8, $layout> {
            fn gamma_forward_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, u8, $layout>,
                destination: &mut ImageOutputView<'_, u8, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $forward(stream_context, source, destination) })
            }

            fn gamma_forward_image_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, u8, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(image)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe { $forward_in_place(stream_context, image) })
            }

            fn gamma_inverse_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, u8, $layout>,
                destination: &mut ImageOutputView<'_, u8, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $inverse(stream_context, source, destination) })
            }

            fn gamma_inverse_image_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, u8, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(image)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe { $inverse_in_place(stream_context, image) })
            }
        }
    };
}

macro_rules! impl_planar_gamma_image {
    ($forward:path, $forward_in_place:path, $inverse:path, $inverse_in_place:path) => {
        unsafe impl PlanarGammaImage for ImagePipeline<'_, u8, C1> {
            fn gamma_forward_planar_image(
                stream_context: &StreamContext,
                source: &PlanarImageView<'_, u8, 3>,
                destination: &mut PlanarImageOutputView<'_, u8, 3>,
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
                    .sync_scope(|| unsafe { $forward(stream_context, source, destination) })
            }

            fn gamma_forward_planar_image_in_place(
                stream_context: &StreamContext,
                image: &mut PlanarImageViewMut<'_, u8, 3>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                for plane in image.planes() {
                    stream_context.prepare_region(plane)?;
                }
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe { $forward_in_place(stream_context, image) })
            }

            fn gamma_inverse_planar_image(
                stream_context: &StreamContext,
                source: &PlanarImageView<'_, u8, 3>,
                destination: &mut PlanarImageOutputView<'_, u8, 3>,
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
                    .sync_scope(|| unsafe { $inverse(stream_context, source, destination) })
            }

            fn gamma_inverse_planar_image_in_place(
                stream_context: &StreamContext,
                image: &mut PlanarImageViewMut<'_, u8, 3>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                for plane in image.planes() {
                    stream_context.prepare_region(plane)?;
                }
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe { $inverse_in_place(stream_context, image) })
            }
        }
    };
}
