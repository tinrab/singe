use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{C1, C3, C4, ImageOutputView},
    },
    pipeline::image::{ImagePipeline, filtering_traits::SumWindowFilterImage},
};

macro_rules! impl_sum_window_filter_image {
    ($ty:ty, $layout:ty, $column:path, $row:path) => {
        unsafe impl<'a> SumWindowFilterImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn sum_window_column_image(
                stream_context: &StreamContext,
                source: &$crate::image::view::ImageWindow<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, f32, $layout>,
                mask_size: i32,
                anchor: i32,
            ) -> Result<()> {
                let (mask, origin) = $crate::pipeline::image::filtering::FilterAxis::Column
                    .geometry(
                        usize::try_from(mask_size)
                            .map_err(|_| $crate::error::Error::out_of_range("filter taps"))?,
                        anchor,
                    )?;
                source.require_halo(mask, origin)?;
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $column(
                        stream_context,
                        &source.view(),
                        destination,
                        mask_size,
                        anchor,
                    )
                })
            }

            fn sum_window_row_image(
                stream_context: &StreamContext,
                source: &$crate::image::view::ImageWindow<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, f32, $layout>,
                mask_size: i32,
                anchor: i32,
            ) -> Result<()> {
                let (mask, origin) = $crate::pipeline::image::filtering::FilterAxis::Row.geometry(
                    usize::try_from(mask_size)
                        .map_err(|_| $crate::error::Error::out_of_range("filter taps"))?,
                    anchor,
                )?;
                source.require_halo(mask, origin)?;
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $row(
                        stream_context,
                        &source.view(),
                        destination,
                        mask_size,
                        anchor,
                    )
                })
            }
        }
    };
}

impl_sum_window_filter_image!(
    u8,
    C1,
    filtering::sum_window_column_u8_c1,
    filtering::sum_window_row_u8_c1
);

impl_sum_window_filter_image!(
    u8,
    C3,
    filtering::sum_window_column_u8_c3,
    filtering::sum_window_row_u8_c3
);

impl_sum_window_filter_image!(
    u8,
    C4,
    filtering::sum_window_column_u8_c4,
    filtering::sum_window_row_u8_c4
);

impl_sum_window_filter_image!(
    u16,
    C1,
    filtering::sum_window_column_u16_c1,
    filtering::sum_window_row_u16_c1
);

impl_sum_window_filter_image!(
    u16,
    C3,
    filtering::sum_window_column_u16_c3,
    filtering::sum_window_row_u16_c3
);

impl_sum_window_filter_image!(
    u16,
    C4,
    filtering::sum_window_column_u16_c4,
    filtering::sum_window_row_u16_c4
);

impl_sum_window_filter_image!(
    i16,
    C1,
    filtering::sum_window_column_i16_c1,
    filtering::sum_window_row_i16_c1
);

impl_sum_window_filter_image!(
    i16,
    C3,
    filtering::sum_window_column_i16_c3,
    filtering::sum_window_row_i16_c3
);

impl_sum_window_filter_image!(
    i16,
    C4,
    filtering::sum_window_column_i16_c4,
    filtering::sum_window_row_i16_c4
);
