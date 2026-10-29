use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{AC4, C1, C3, C4, ImageOutputView},
    },
    pipeline::image::{ImagePipeline, filtering_traits::HighLowGaussFilterImage},
    types::MaskSize,
};

macro_rules! impl_high_low_gauss_filter_image {
    ($ty:ty, $layout:ty, $high_pass:path, $low_pass:path, $gauss:path) => {
        unsafe impl<'a> HighLowGaussFilterImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn filter_high_pass_image(
                stream_context: &StreamContext,
                source: &$crate::image::view::ImageWindow<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                mask_size: MaskSize,
            ) -> Result<()> {
                let (mask, anchor) =
                    $crate::pipeline::image::filtering::centered_filter_geometry(mask_size)?;
                source.require_halo(mask, anchor)?;
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $high_pass(stream_context, &source.view(), destination, mask_size)
                })
            }

            fn filter_low_pass_image(
                stream_context: &StreamContext,
                source: &$crate::image::view::ImageWindow<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                mask_size: MaskSize,
            ) -> Result<()> {
                let (mask, anchor) =
                    $crate::pipeline::image::filtering::centered_filter_geometry(mask_size)?;
                source.require_halo(mask, anchor)?;
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $low_pass(stream_context, &source.view(), destination, mask_size)
                })
            }

            fn filter_gauss_image(
                stream_context: &StreamContext,
                source: &$crate::image::view::ImageWindow<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                mask_size: MaskSize,
            ) -> Result<()> {
                let (mask, anchor) =
                    $crate::pipeline::image::filtering::centered_filter_geometry(mask_size)?;
                source.require_halo(mask, anchor)?;
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $gauss(stream_context, &source.view(), destination, mask_size)
                })
            }
        }
    };
}

impl_high_low_gauss_filter_image!(
    u16,
    C1,
    filtering::filter_high_pass_u16_c1,
    filtering::filter_low_pass_u16_c1,
    filtering::filter_gauss_u16_c1
);

impl_high_low_gauss_filter_image!(
    u16,
    C3,
    filtering::filter_high_pass_u16_c3,
    filtering::filter_low_pass_u16_c3,
    filtering::filter_gauss_u16_c3
);

impl_high_low_gauss_filter_image!(
    u16,
    C4,
    filtering::filter_high_pass_u16_c4,
    filtering::filter_low_pass_u16_c4,
    filtering::filter_gauss_u16_c4
);

impl_high_low_gauss_filter_image!(
    u16,
    AC4,
    filtering::filter_high_pass_u16_ac4,
    filtering::filter_low_pass_u16_ac4,
    filtering::filter_gauss_u16_ac4
);
