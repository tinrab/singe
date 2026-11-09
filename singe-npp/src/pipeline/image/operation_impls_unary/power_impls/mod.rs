use crate::pipeline::image::{
    operation_impls_unary::ImagePipeline,
    operation_traits::{SquareImage, SquareRootImage},
};

macro_rules! impl_scaled_unary_power_image {
    (
        $ty:ty,
        $layout:ty,
        $square:path,
        $square_in_place:path,
        $square_root:path,
        $square_root_in_place:path
    ) => {
        unsafe impl<'a> SquareImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn square_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $square(stream_context, source, destination, scale_factor)
                })
            }

            fn square_image_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $square_in_place(stream_context, source_destination, scale_factor)
                })
            }
        }

        unsafe impl<'a> SquareRootImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn square_root_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $square_root(stream_context, source, destination, scale_factor)
                })
            }

            fn square_root_image_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $square_root_in_place(stream_context, source_destination, scale_factor)
                })
            }
        }
    };
}

macro_rules! impl_scaled_square_image {
    ($ty:ty, $layout:ty, $square:path, $square_in_place:path) => {
        unsafe impl<'a> SquareImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn square_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $square(stream_context, source, destination, scale_factor)
                })
            }

            fn square_image_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $square_in_place(stream_context, source_destination, scale_factor)
                })
            }
        }
    };
}

macro_rules! impl_float_unary_power_image {
    (
        $layout:ty,
        $square:path,
        $square_in_place:path,
        $square_root:path,
        $square_root_in_place:path
    ) => {
        unsafe impl<'a> SquareImage<f32, $layout> for ImagePipeline<'a, f32, $layout> {
            fn square_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, f32, $layout>,
                destination: &mut ImageOutputView<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $square(stream_context, source, destination) })
            }

            fn square_image_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $square_in_place(stream_context, source_destination) })
            }
        }

        unsafe impl<'a> SquareRootImage<f32, $layout> for ImagePipeline<'a, f32, $layout> {
            fn square_root_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, f32, $layout>,
                destination: &mut ImageOutputView<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $square_root(stream_context, source, destination) })
            }

            fn square_root_image_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $square_root_in_place(stream_context, source_destination)
                })
            }
        }
    };
}

mod ac4_impls;
mod float_impls;
mod i16_impls;
mod square_only_impls;
mod u16_impls;
mod u8_impls;
