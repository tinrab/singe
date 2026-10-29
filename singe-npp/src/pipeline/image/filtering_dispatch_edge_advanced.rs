macro_rules! impl_unsharp_border_filter_image {
    ($ty:ty, $layout:ty, $unsharp:path) => {
        unsafe impl<'a> UnsharpBorderFilterImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn filter_unsharp_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                radius: f32,
                sigma: f32,
                weight: f32,
                threshold: f32,
                border_type: BorderType,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the native helper completes borrowed use on every path.
                unsafe {
                    $unsharp(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        radius,
                        sigma,
                        weight,
                        threshold,
                        border_type,
                    )
                }
            }
        }
    };
}

macro_rules! impl_advanced_gauss_filter_image {
    ($ty:ty, $layout:ty, $gauss:path, $gauss_border:path) => {
        unsafe impl<'a> AdvancedGaussFilterImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn filter_gauss_advanced_image(
                stream_context: &StreamContext,
                source: &$crate::image::view::ImageWindow<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                kernel: &[f32],
            ) -> Result<()> {
                let (mask, origin) =
                    $crate::pipeline::image::filtering::gaussian_geometry(kernel.len())?;
                source.require_halo(mask, origin)?;
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $gauss(stream_context, &source.view(), destination, kernel)
                })
            }

            fn filter_gauss_advanced_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                kernel: &[f32],
                border_type: BorderType,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $gauss_border(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        kernel,
                        border_type,
                    )
                })
            }
        }
    };
}

mod gauss_advanced_impls;
mod gauss_pyramid_bilateral;
mod unsharp_border_impls;
