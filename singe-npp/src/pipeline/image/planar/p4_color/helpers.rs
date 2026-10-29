use crate::{
    context::StreamContext,
    error::Result,
    image::{
        SupportedImage, exchange,
        memory::Image,
        view::{AC4, C3, C4, ImageOutputView, PlanarImageOutputView, PlanarImageView},
    },
    pipeline::image::planar::p4_color::PlanarImage,
};

type Planar4ToPackedColorConvert = for<'source, 'destination> unsafe fn(
    &StreamContext,
    &PlanarImageView<'source, u8, 4>,
    &mut ImageOutputView<'destination, u8, AC4>,
) -> Result<()>;

type Planar4ToC3ColorConvert = for<'source, 'destination> unsafe fn(
    &StreamContext,
    &PlanarImageView<'source, u8, 4>,
    &mut ImageOutputView<'destination, u8, C3>,
) -> Result<()>;

type Planar4ToPlanar4ColorConvert = for<'source, 'destination> unsafe fn(
    &StreamContext,
    &PlanarImageView<'source, u8, 4>,
    &mut PlanarImageOutputView<'destination, u8, 4>,
) -> Result<()>;

type Planar4ToPlanar3ColorConvert = for<'source, 'destination> unsafe fn(
    &StreamContext,
    &PlanarImageView<'source, u8, 4>,
    &mut PlanarImageOutputView<'destination, u8, 3>,
) -> Result<()>;

pub trait PlanarP4ColorConversionExt {
    fn color_convert(
        self,
        stream_context: &StreamContext,
        operation: Planar4ToPlanar4ColorConvert,
    ) -> Result<Self>
    where
        Self: Sized;

    fn color_convert_to_ac4(
        self,
        stream_context: &StreamContext,
        operation: Planar4ToPackedColorConvert,
    ) -> Result<Image<u8, AC4>>
    where
        u8: SupportedImage<AC4>;

    fn color_convert_to_p3(
        self,
        stream_context: &StreamContext,
        operation: Planar4ToPlanar3ColorConvert,
    ) -> Result<PlanarImage<u8, 3>>;

    fn color_convert_to_c3(
        self,
        stream_context: &StreamContext,
        operation: Planar4ToC3ColorConvert,
    ) -> Result<Image<u8, C3>>
    where
        u8: SupportedImage<C3>;
}

impl PlanarP4ColorConversionExt for PlanarImage<u8, 4> {
    fn color_convert(
        self,
        stream_context: &StreamContext,
        operation: Planar4ToPlanar4ColorConvert,
    ) -> Result<Self> {
        self.prepare_eager(stream_context)?;

        let mut destination =
            Self::create_uninit(stream_context.cuda_context(), self.planes[0].size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            stream_context.sync_scope(|| {
                // SAFETY: this private helper selects a matching native color producer and retains all participants.
                unsafe {
                    operation(stream_context, &source, &mut destination_view)?;
                    exchange::copy_c1(
                        stream_context,
                        &source.planes()[3],
                        &mut destination_view.planes_mut()[3],
                    )
                }
            })?;
        }

        // SAFETY: completed color conversion established all logical output channels.
        Ok(unsafe { destination.assume_init() })
    }

    fn color_convert_to_ac4(
        self,
        stream_context: &StreamContext,
        operation: Planar4ToPackedColorConvert,
    ) -> Result<Image<u8, AC4>>
    where
        u8: SupportedImage<AC4>,
    {
        self.prepare_eager(stream_context)?;

        let mut destination =
            Image::create_uninit(stream_context.cuda_context(), self.planes[0].size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            stream_context.sync_scope(|| {
                // SAFETY: this private helper selects a matching native color producer and retains all participants.
                unsafe {
                    operation(stream_context, &source, &mut destination_view)?;
                    let mut rgba = destination_view.output().with_layout::<C4>()?;
                    exchange::insert_channel_c4(stream_context, &source.planes()[3], &mut rgba, 3)
                }
            })?;
        }

        // SAFETY: completed color conversion established all logical output channels.
        Ok(unsafe { destination.assume_init() })
    }

    fn color_convert_to_p3(
        self,
        stream_context: &StreamContext,
        operation: Planar4ToPlanar3ColorConvert,
    ) -> Result<PlanarImage<u8, 3>> {
        self.prepare_eager(stream_context)?;

        let mut destination = PlanarImage::<u8, 3>::create_uninit(
            stream_context.cuda_context(),
            self.planes[0].size(),
        )?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            stream_context.sync_scope(|| {
                // SAFETY: this private helper selects a matching native color producer and retains all participants.
                unsafe { operation(stream_context, &source, &mut destination_view) }
            })?;
        }

        // SAFETY: completed color conversion established all logical output channels.
        Ok(unsafe { destination.assume_init() })
    }

    fn color_convert_to_c3(
        self,
        stream_context: &StreamContext,
        operation: Planar4ToC3ColorConvert,
    ) -> Result<Image<u8, C3>>
    where
        u8: SupportedImage<C3>,
    {
        self.prepare_eager(stream_context)?;

        let mut destination =
            Image::create_uninit(stream_context.cuda_context(), self.planes[0].size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            stream_context.sync_scope(|| {
                // SAFETY: this private helper selects a matching native color producer and retains all participants.
                unsafe { operation(stream_context, &source, &mut destination_view) }
            })?;
        }

        // SAFETY: completed color conversion established all logical output channels.
        Ok(unsafe { destination.assume_init() })
    }
}
