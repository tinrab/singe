use std::{mem::MaybeUninit, sync::Arc};

use singe_cuda::context::Context as CudaContext;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        SupportedImage, color,
        memory::Image,
        view::{C1, C2, ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::image::planar::{SemiplanarImage, subsampled_size},
    types::Size,
};

type Nv12PackedInverse<T, L> = for<'y, 'uv, 'destination> unsafe fn(
    &StreamContext,
    &ImageView<'y, T, C1>,
    &ImageView<'uv, T, C2>,
    &mut ImageOutputView<'destination, T, L>,
) -> Result<()>;

type Nv12PackedInverseColorTwist<T, L> = for<'y, 'uv, 'destination> unsafe fn(
    &StreamContext,
    &ImageView<'y, T, C1>,
    &ImageView<'uv, T, C2>,
    &mut ImageOutputView<'destination, T, L>,
    color::ColorTwistMatrix,
) -> Result<()>;

mod semiplanar_packed_color;

impl<T> SemiplanarImage<T>
where
    T: Copy,
{
    /// Allocates uninitialized NV12-shaped storage with full-resolution Y and half-resolution UV.
    ///
    /// The interleaved chroma plane has half the width and half the height of `size`.
    /// No color conversion or initialization is performed.
    ///
    /// # Errors
    ///
    /// Returns an error if either dimension is nonpositive or odd, allocation-size arithmetic
    /// fails, or allocating either plane fails.
    pub fn create_nv12_uninit(
        allocation_context: &Arc<CudaContext>,
        size: Size,
    ) -> Result<SemiplanarImage<MaybeUninit<T>>>
    where
        T: SupportedImage<C1> + SupportedImage<C2>,
    {
        Self::create_uninit(allocation_context, size, 2, 2)
    }

    /// Allocates an uninitialized full-resolution first plane and one interleaved chroma plane.
    /// Chroma dimensions are the full-resolution dimensions divided by the supplied horizontal
    /// and vertical factors. Does not initialize any samples.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions or subsampling factors are nonpositive, a dimension is not
    /// divisible by its factor, allocation-size arithmetic fails, or allocating a plane fails.
    pub fn create_uninit(
        allocation_context: &Arc<CudaContext>,
        size: Size,
        horizontal_subsampling: i32,
        vertical_subsampling: i32,
    ) -> Result<SemiplanarImage<MaybeUninit<T>>>
    where
        T: SupportedImage<C1> + SupportedImage<C2>,
    {
        let chroma_size = subsampled_size(size, horizontal_subsampling, vertical_subsampling)?;
        Ok(SemiplanarImage {
            y: Image::create_uninit(allocation_context, size)?,
            uv: Image::create_uninit(allocation_context, chroma_size)?,
        })
    }

    fn color_convert_to_packed<L>(
        self,
        stream_context: &StreamContext,
        operation: Nv12PackedInverse<T, L>,
    ) -> Result<Image<T, L>>
    where
        L: ChannelLayout,
        T: SupportedImage<L>,
    {
        self.prepare_eager(stream_context)?;

        let mut destination = Image::create_uninit(stream_context.cuda_context(), self.y.size())?;

        {
            let y = self.y.view()?;
            let uv = self.uv.view()?;
            let mut destination_view = destination.output()?;
            stream_context.sync_scope(|| {
                // SAFETY: this private helper selects a matching native color producer and retains all participants.
                unsafe { operation(stream_context, &y, &uv, &mut destination_view) }
            })?;
        }

        // SAFETY: completed color conversion established all logical output channels.
        Ok(unsafe { destination.assume_init() })
    }

    fn color_twist_to_packed<L>(
        self,
        stream_context: &StreamContext,
        twist: color::ColorTwistMatrix,
        operation: Nv12PackedInverseColorTwist<T, L>,
    ) -> Result<Image<T, L>>
    where
        L: ChannelLayout,
        T: SupportedImage<L>,
    {
        self.prepare_eager(stream_context)?;

        let mut destination = Image::create_uninit(stream_context.cuda_context(), self.y.size())?;

        {
            let y = self.y.view()?;
            let uv = self.uv.view()?;
            let mut destination_view = destination.output()?;
            stream_context.sync_scope(|| {
                // SAFETY: this private helper selects a matching native color producer and retains all participants.
                unsafe { operation(stream_context, &y, &uv, &mut destination_view, twist) }
            })?;
        }

        // SAFETY: completed color conversion established all logical output channels.
        Ok(unsafe { destination.assume_init() })
    }
}
