use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageOutputView, ImageView},
};

/// # Safety
/// Complete borrowed native work on every path and preserve the native ABI.
/// Successful execution initializes every physical channel throughout the destination, including its border.
pub unsafe trait BorderImage<T, L> {
    fn copy_replicate_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        top: usize,
        left: usize,
    ) -> Result<()>;

    fn copy_wrap_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        top: usize,
        left: usize,
    ) -> Result<()>;
}

macro_rules! impl_border_image {
    ($ty:ty, $layout:ty, $physical:ty, $replicate_border:path, $wrap_border:path) => {
        unsafe impl<'a> BorderImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn copy_replicate_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                top: usize,
                left: usize,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context.sync_scope(|| {
                    let source = source.with_layout::<$physical>()?;
                    let mut destination = destination.output().with_layout::<$physical>()?;
                    // SAFETY: native dispatch writes the full border and source area, including alpha.
                    // The scope retains the input and output through completion.
                    unsafe {
                        $replicate_border(stream_context, &source, &mut destination, top, left)
                    }
                })
            }

            fn copy_wrap_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                top: usize,
                left: usize,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context.sync_scope(|| {
                    let source = source.with_layout::<$physical>()?;
                    let mut destination = destination.output().with_layout::<$physical>()?;
                    // SAFETY: native dispatch writes the full border and source area, including alpha.
                    // The scope retains the input and output through completion.
                    unsafe { $wrap_border(stream_context, &source, &mut destination, top, left) }
                })
            }
        }
    };
}

mod impls;
