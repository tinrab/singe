use singe_cuda::{
    stream::Stream as CudaStream,
    view::{AnyBitPatternDeviceRepr, DeviceRegion, DeviceRegionProvider},
};

use crate::{
    buffer::DeviceBuffer,
    context::{Context, JpegState},
    decode::{
        DecodeOptions, DecodeParams, DecodedImage, DecodedOutput,
        image_descriptor_from_decoded_output,
    },
    decoder::JpegDecoder,
    error::Result,
    image::{Image, ImageViewMut, OutputSpec, PlaneMut},
    stream::JpegStream,
    types::JpegInfo,
};

/// Decoder for one image at a time.
///
/// It keeps its [`Context`], decoder state, and the staged decoder that options such as a region need, and reuses them across calls.
#[derive(Debug)]
pub struct Decoder {
    context: Context,
    state: JpegState,
    options: DecodeOptions,
    staged: Option<StagedDecoder>,
}

// Created on the first decode whose options need nvJPEG's staged decoding, and dropped when the options change.
#[derive(Debug)]
struct StagedDecoder {
    params: DecodeParams,
    jpeg_stream: JpegStream,
    state: JpegState,
    decoder: JpegDecoder,
}

impl StagedDecoder {
    fn create(context: &Context, options: DecodeOptions) -> Result<Self> {
        let decoder = JpegDecoder::create(context, context.backend())?;
        let mut state = decoder.create_state()?;
        state.attach_device_buffer(DeviceBuffer::create(context)?)?;
        let mut params = DecodeParams::create(context)?;
        options.apply_to(&mut params)?;
        Ok(Self {
            params,
            jpeg_stream: JpegStream::create(context)?,
            state,
            decoder,
        })
    }
}

impl Decoder {
    /// Creates a decoder for `context` with `options`.
    ///
    /// Options that need staged decoding, such as a region, use the backend `context` was created with.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG fails to create the state.
    pub fn create(context: &Context, options: DecodeOptions) -> Result<Self> {
        Ok(Self {
            context: context.clone(),
            state: context.create_jpeg_state()?,
            options,
            staged: None,
        })
    }

    /// Returns the nvJPEG handle.
    #[must_use]
    pub const fn context(&self) -> &Context {
        &self.context
    }

    /// Returns the decode options.
    #[must_use]
    pub const fn options(&self) -> DecodeOptions {
        self.options
    }

    /// Replaces the decode options.
    pub fn set_options(&mut self, options: DecodeOptions) {
        if options != self.options {
            self.staged = None;
        }

        self.options = options;
    }

    /// Reads the JPEG header in `data`.
    ///
    /// # Errors
    ///
    /// Returns an error if `data` is empty or too long for nvJPEG, the CUDA context cannot be bound, or nvJPEG cannot read the header.
    pub fn read_info(&self, data: &[u8]) -> Result<JpegInfo> {
        self.context.image_info(data)
    }

    /// Returns the size and layout of each output plane for an image described by `info`, with the current options.
    ///
    /// # Errors
    ///
    /// Returns an error if [`DecodeOptions::output_spec`] fails.
    pub fn output_spec(&self, info: &JpegInfo) -> Result<OutputSpec> {
        self.options.output_spec(info)
    }

    /// Reads the JPEG header in `data` and returns the size and layout of each output plane.
    ///
    /// # Errors
    ///
    /// Returns an error if [`read_info`](Self::read_info) or [`DecodeOptions::output_spec`] fails.
    pub fn read_output_spec(&self, data: &[u8]) -> Result<OutputSpec> {
        let info = self.read_info(data)?;
        self.output_spec(&info)
    }

    /// Allocates one output plane, decodes into it, and waits for `stream`.
    ///
    /// Use it with single-plane output formats, such as [`OutputFormat::RgbInterleaved`](crate::types::OutputFormat::RgbInterleaved), [`OutputFormat::BgrInterleaved`](crate::types::OutputFormat::BgrInterleaved), [`OutputFormat::Y`](crate::types::OutputFormat::Y), and [`OutputFormat::Yuy2`](crate::types::OutputFormat::Yuy2).
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` is being captured or belongs to another CUDA context, the header cannot be read or the output size cannot be computed, the output format has more than one plane, the allocation fails, or nvJPEG or CUDA fails.
    pub fn decode_single_plane_to_device(
        &mut self,
        data: &[u8],
        stream: &CudaStream,
    ) -> Result<DecodedImage> {
        stream.ensure_not_capturing()?;
        let spec = self.read_output_spec(data)?;
        let plane = spec.single_plane()?;
        let mut output = plane.create_uninit_buffer(stream.context())?;
        {
            let mut view = ImageViewMut::create([PlaneMut::create(
                output.output(),
                plane.size(),
                plane.row_bytes(),
            )?])?;
            self.decode_into(data, &mut view, stream)?;
        }
        Ok(DecodedImage {
            // SAFETY: the decode above finished and wrote every byte of this tightly packed output.
            data: unsafe { output.assume_init() },
            plane,
        })
    }

    /// Creates [`DecodeParams`] that match the current options.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, or nvJPEG fails to create the parameters or rejects an option.
    pub fn create_params(&self) -> Result<DecodeParams> {
        let mut params = DecodeParams::create(&self.context)?;
        self.options.apply_to(&mut params)?;
        Ok(params)
    }

    /// Decodes one JPEG into `destination` and waits for `stream`.
    ///
    /// Options that only set the output format use nvJPEG's single-image decoding.
    /// Options that need [`DecodeParams`], such as a region or EXIF orientation, use its staged decoding.
    /// Stream capture is not allowed, and the call waits for the GPU work even if it fails.
    /// Only the image's pixels are written; padding and unused memory stay uninitialized.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` is being captured or belongs to another CUDA context; the header cannot be read or the output size cannot be computed; `destination` has the wrong plane count, context, alignment, size, or pitch; or nvJPEG or CUDA fails.
    pub fn decode_into<T: AnyBitPatternDeviceRepr, const C: usize>(
        &mut self,
        data: &[u8],
        destination: &mut ImageViewMut<'_, T, C>,
        stream: &CudaStream,
    ) -> Result<()> {
        stream.ensure_not_capturing()?;
        let () = T::LAYOUT;
        let spec = self.read_output_spec(data)?;
        destination.validate_output(&spec, stream.context())?;
        let mut native_destination = destination.descriptor()?;
        let regions = destination
            .planes()
            .iter()
            .map(|plane| plane.region())
            .collect::<Vec<_>>();
        self.decode_descriptor(data, &mut native_destination, &regions, stream)
    }

    /// Allocates the output planes, decodes into them, and waits for `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` is being captured or belongs to another CUDA context, the header cannot be read or the output size cannot be computed, an allocation fails, or nvJPEG or CUDA fails.
    pub fn decode_to_device(&mut self, data: &[u8], stream: &CudaStream) -> Result<DecodedOutput> {
        stream.ensure_not_capturing()?;
        let spec = self.read_output_spec(data)?;
        let mut output = DecodedOutput::create_uninit(stream.context(), spec)?;
        let mut destination = image_descriptor_from_decoded_output(&mut output)?;
        let regions = output
            .planes
            .iter()
            .map(|plane| plane.data.region())
            .collect::<Vec<_>>();
        self.decode_descriptor(data, &mut destination, &regions, stream)?;
        // SAFETY: the decode above finished and wrote every byte of this tightly packed output.
        Ok(unsafe { output.assume_init() })
    }

    fn decode_descriptor(
        &mut self,
        data: &[u8],
        destination: &mut Image,
        regions: &[DeviceRegion<'_>],
        stream: &CudaStream,
    ) -> Result<()> {
        if self.options.requires_parameterized_decode() {
            return self.decode_staged_descriptor(data, destination, regions, stream);
        }
        stream.complete(|| {
            stream.join_regions(regions)?;
            // SAFETY: callers validated the destination's geometry against the header, `data` and the destination stay borrowed, and `complete` waits for the work on every path.
            unsafe {
                self.context.decode_async(
                    &mut self.state,
                    data,
                    self.options.output_format,
                    destination,
                    stream,
                )
            }
        })
    }

    fn decode_staged_descriptor(
        &mut self,
        data: &[u8],
        destination: &mut Image,
        regions: &[DeviceRegion<'_>],
        stream: &CudaStream,
    ) -> Result<()> {
        let staged = match self.staged.take() {
            Some(staged) => staged,
            None => StagedDecoder::create(&self.context, self.options)?,
        };
        let staged = self.staged.insert(staged);
        staged.jpeg_stream.parse(data, false)?;
        stream.complete(|| {
            stream.join_regions(regions)?;
            // SAFETY: callers validated the destination's geometry against the header, the cached decoder objects and the destination stay borrowed, and `complete` waits for the work on every path.
            unsafe {
                self.context.decode_jpeg_async(
                    &staged.decoder,
                    &mut staged.state,
                    &staged.jpeg_stream,
                    destination,
                    &staged.params,
                    stream,
                )
            }
        })
    }
}
