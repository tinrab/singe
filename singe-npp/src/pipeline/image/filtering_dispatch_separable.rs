use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{AC4, C1, C3, C4, ImageOutputView},
    },
    pipeline::image::{
        ImagePipeline,
        filtering_traits::{FloatSeparableFilterImage, IntegerSeparableFilterImage},
    },
};

macro_rules! impl_integer_separable_filter_image {
    ($ty:ty, $layout:ty, $column:path, $row:path) => {
        unsafe impl<'a> IntegerSeparableFilterImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn filter_column_image(
                stream_context: &StreamContext,
                source: &$crate::image::view::ImageWindow<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                kernel: &[i32],
                anchor: i32,
                divisor: i32,
            ) -> Result<()> {
                let (mask, origin) = $crate::pipeline::image::filtering::FilterAxis::Column
                    .geometry(kernel.len(), anchor)?;
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
                        kernel,
                        anchor,
                        divisor,
                    )
                })
            }

            fn filter_row_image(
                stream_context: &StreamContext,
                source: &$crate::image::view::ImageWindow<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                kernel: &[i32],
                anchor: i32,
                divisor: i32,
            ) -> Result<()> {
                let (mask, origin) = $crate::pipeline::image::filtering::FilterAxis::Row
                    .geometry(kernel.len(), anchor)?;
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
                        kernel,
                        anchor,
                        divisor,
                    )
                })
            }
        }
    };
}

macro_rules! impl_float_separable_filter_image {
    ($ty:ty, $layout:ty, $column:path, $row:path) => {
        unsafe impl<'a> FloatSeparableFilterImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn filter_column32f_image(
                stream_context: &StreamContext,
                source: &$crate::image::view::ImageWindow<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                kernel: &[f32],
                anchor: i32,
            ) -> Result<()> {
                let (mask, origin) = $crate::pipeline::image::filtering::FilterAxis::Column
                    .geometry(kernel.len(), anchor)?;
                source.require_halo(mask, origin)?;
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $column(stream_context, &source.view(), destination, kernel, anchor)
                })
            }

            fn filter_row32f_image(
                stream_context: &StreamContext,
                source: &$crate::image::view::ImageWindow<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                kernel: &[f32],
                anchor: i32,
            ) -> Result<()> {
                let (mask, origin) = $crate::pipeline::image::filtering::FilterAxis::Row
                    .geometry(kernel.len(), anchor)?;
                source.require_halo(mask, origin)?;
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $row(stream_context, &source.view(), destination, kernel, anchor)
                })
            }
        }
    };
}

mod float_separable;
mod i16_float_separable;
mod integer_separable;

impl_float_separable_filter_image!(
    u8,
    C1,
    filtering::filter_column32f_u8_c1,
    filtering::filter_row32f_u8_c1
);

impl_float_separable_filter_image!(
    u8,
    C3,
    filtering::filter_column32f_u8_c3,
    filtering::filter_row32f_u8_c3
);

impl_float_separable_filter_image!(
    u8,
    C4,
    filtering::filter_column32f_u8_c4,
    filtering::filter_row32f_u8_c4
);

impl_float_separable_filter_image!(
    u8,
    AC4,
    filtering::filter_column32f_u8_ac4,
    filtering::filter_row32f_u8_ac4
);

impl_float_separable_filter_image!(
    u16,
    C1,
    filtering::filter_column32f_u16_c1,
    filtering::filter_row32f_u16_c1
);

impl_float_separable_filter_image!(
    u16,
    C3,
    filtering::filter_column32f_u16_c3,
    filtering::filter_row32f_u16_c3
);

impl_float_separable_filter_image!(
    u16,
    C4,
    filtering::filter_column32f_u16_c4,
    filtering::filter_row32f_u16_c4
);

impl_float_separable_filter_image!(
    u16,
    AC4,
    filtering::filter_column32f_u16_ac4,
    filtering::filter_row32f_u16_ac4
);
