use crate::{
    context::StreamContext,
    error::Result,
    image::{
        exchange,
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView},
    },
    pipeline::image::border::ImagePipeline,
};

mod replicate_wrap;

pub use replicate_wrap::BorderImage;

/// # Safety
/// Complete borrowed native work on every path and preserve the native ABI.
/// Successful execution initializes every destination channel, including source and border alpha.
/// The border value includes every physical channel.
pub unsafe trait ConstantBorderImage<T, L> {
    type Value;

    fn copy_constant_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        top: usize,
        left: usize,
        value: Self::Value,
    ) -> Result<()>;
}

macro_rules! impl_constant_border_image {
    ($ty:ty, $layout:ty, $physical:ty, $value_ty:ty, $constant_border:path) => {
        unsafe impl<'a> ConstantBorderImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            type Value = $value_ty;

            fn copy_constant_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                top: usize,
                left: usize,
                value: Self::Value,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context.sync_scope(|| {
                    let source = source.with_layout::<$physical>()?;
                    let mut destination = destination.output().with_layout::<$physical>()?;
                    // SAFETY: native dispatch writes every source and border channel, including alpha.
                    // The completion scope retains every participant.
                    unsafe {
                        $constant_border(
                            stream_context,
                            &source,
                            &mut destination,
                            top,
                            left,
                            value,
                        )
                    }
                })
            }
        }
    };
}

impl_constant_border_image!(u8, C1, C1, u8, exchange::copy_constant_border_c1);

impl_constant_border_image!(u8, C3, C3, [u8; 3], exchange::copy_constant_border_c3);

impl_constant_border_image!(u8, C4, C4, [u8; 4], exchange::copy_constant_border_c4);

impl_constant_border_image!(u8, AC4, C4, [u8; 4], exchange::copy_constant_border_c4);

impl_constant_border_image!(u16, C1, C1, u16, exchange::copy_constant_border_c1);

impl_constant_border_image!(u16, C3, C3, [u16; 3], exchange::copy_constant_border_c3);

impl_constant_border_image!(u16, C4, C4, [u16; 4], exchange::copy_constant_border_c4);

impl_constant_border_image!(u16, AC4, C4, [u16; 4], exchange::copy_constant_border_c4);

impl_constant_border_image!(i16, C1, C1, i16, exchange::copy_constant_border_c1);

impl_constant_border_image!(i16, C3, C3, [i16; 3], exchange::copy_constant_border_c3);

impl_constant_border_image!(i16, C4, C4, [i16; 4], exchange::copy_constant_border_c4);

impl_constant_border_image!(i16, AC4, C4, [i16; 4], exchange::copy_constant_border_c4);

impl_constant_border_image!(i32, C1, C1, i32, exchange::copy_constant_border_c1);

impl_constant_border_image!(i32, C3, C3, [i32; 3], exchange::copy_constant_border_c3);

impl_constant_border_image!(i32, C4, C4, [i32; 4], exchange::copy_constant_border_c4);

impl_constant_border_image!(i32, AC4, C4, [i32; 4], exchange::copy_constant_border_c4);

impl_constant_border_image!(f32, C1, C1, f32, exchange::copy_constant_border_c1);

impl_constant_border_image!(f32, C3, C3, [f32; 3], exchange::copy_constant_border_c3);

impl_constant_border_image!(f32, C4, C4, [f32; 4], exchange::copy_constant_border_c4);

impl_constant_border_image!(f32, AC4, C4, [f32; 4], exchange::copy_constant_border_c4);
