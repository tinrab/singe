use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageViewMut},
    },
    pipeline::image::{CopyImage, ImageBacking, ImagePipeline, filtering::FloodFillImage},
    types::{ConnectedRegion, ImageNormalization, Point},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: CopyImage<T, L> + FloodFillImage<T, L>,
    <Self as FloodFillImage<T, L>>::Value: Copy,
{
    pub(crate) fn flood_fill_operation(
        self,
        seed: Point,
        new_value: <Self as FloodFillImage<T, L>>::Value,
        min: Option<<Self as FloodFillImage<T, L>>::Value>,
        max: Option<<Self as FloodFillImage<T, L>>::Value>,
        boundary_value: Option<<Self as FloodFillImage<T, L>>::Value>,
        norm: ImageNormalization,
        connected_region: Option<&mut ConnectedRegion>,
        operation: FloodFillOperationCallback<L, T, <Self as FloodFillImage<T, L>>::Value>,
    ) -> Result<Self> {
        self.ensure_eager()?;
        let ImagePipeline {
            stream_context,
            workspace,
            backing,
        } = self;
        let mut image = match backing {
            ImageBacking::Owned(image) => image,
            ImageBacking::Borrowed(source) => Image::<T, L>::from_view(&source)?,
        };

        {
            let mut image_view = image.view_mut()?;
            operation(
                stream_context,
                &mut image_view,
                seed,
                new_value,
                min,
                max,
                boundary_value,
                norm,
                connected_region,
            )?;
        }

        Ok(Self {
            stream_context,
            workspace,
            backing: ImageBacking::Owned(image),
        })
    }
}

type FloodFillOperationCallback<L, T, V> = fn(
    &StreamContext,
    &mut ImageViewMut<'_, T, L>,
    Point,
    V,
    Option<V>,
    Option<V>,
    Option<V>,
    ImageNormalization,
    Option<&mut ConnectedRegion>,
) -> Result<()>;
