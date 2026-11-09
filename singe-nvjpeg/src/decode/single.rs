use std::{fs, path::Path, sync::Arc};

use singe_cuda::{
    context::Context as CudaContext,
    stream::Stream as CudaStream,
    view::{AnyBitPatternDeviceRepr, DeviceRegion, DeviceRegionProvider},
};

use crate::{
    allocator::AllocatorSet,
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
    types::{Backend, CreateFlags, JpegInfo},
    utility::complete,
};

/// Decoder for one image at a time.
///
/// It owns its [`Context`] and decoder state, and reuses them across calls.
#[derive(Debug)]
pub struct Decoder {
    pub(crate) handle: Context,
    state: JpegState,
    backend: Backend,
    options: DecodeOptions,
}

/// Builder for a [`Decoder`] with a custom backend, flags, options, or allocators.
#[derive(Debug, Clone, Copy)]
pub struct DecoderBuilder {
    backend: Backend,
    flags: CreateFlags,
    options: DecodeOptions,
    allocators: Option<AllocatorSet>,
}

impl Decoder {
    /// Returns a builder for a decoder with a custom backend, flags, options, or allocators.
    #[must_use]
    pub const fn builder() -> DecoderBuilder {
        DecoderBuilder::new()
    }

    /// Creates a decoder in `cuda_context` that uses `backend` and the default options.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, or nvJPEG rejects the backend or fails to create the handle or state.
    pub fn create(cuda_context: &Arc<CudaContext>, backend: Backend) -> Result<Self> {
        Self::builder().backend(backend).create(cuda_context)
    }

    /// Creates a decoder in `cuda_context` that uses `backend` and `options`.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, or nvJPEG rejects the backend or fails to create the handle or state.
    pub fn create_with_options(
        cuda_context: &Arc<CudaContext>,
        backend: Backend,
        options: DecodeOptions,
    ) -> Result<Self> {
        Self::builder()
            .backend(backend)
            .options(options)
            .create(cuda_context)
    }

    /// Creates a decoder that uses an existing `handle` and `options`.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG fails to create the state.
    pub fn from_handle(handle: Context, options: DecodeOptions) -> Result<Self> {
        Self::from_handle_with_backend(handle, Backend::Default, options)
    }

    /// Creates a decoder that uses an existing `handle`, `options`, and `backend`.
    ///
    /// The backend applies when the options need nvJPEG's staged decoding, such as for a region.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG fails to create the state.
    pub fn from_handle_with_backend(
        handle: Context,
        backend: Backend,
        options: DecodeOptions,
    ) -> Result<Self> {
        let state = handle.create_jpeg_state()?;
        Ok(Self {
            handle,
            state,
            backend,
            options,
        })
    }

    /// Returns the nvJPEG handle.
    #[must_use]
    pub const fn handle(&self) -> &Context {
        &self.handle
    }

    /// Returns the decode options.
    #[must_use]
    pub const fn options(&self) -> DecodeOptions {
        self.options
    }

    /// Returns the backend used for staged decoding.
    #[must_use]
    pub const fn backend(&self) -> Backend {
        self.backend
    }

    /// Replaces the decode options.
    pub fn set_options(&mut self, options: DecodeOptions) {
        self.options = options;
    }

    /// Reads the JPEG header in `data`.
    ///
    /// # Errors
    ///
    /// Returns an error if `data` is empty or too long for nvJPEG, the CUDA context cannot be bound, or nvJPEG cannot read the header.
    pub fn read_info(&self, data: &[u8]) -> Result<JpegInfo> {
        self.handle.image_info(data)
    }

    /// Reads the header of the JPEG file at `path`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or [`read_info`](Self::read_info) fails.
    pub fn read_file_info(&self, path: impl AsRef<Path>) -> Result<JpegInfo> {
        let data = fs::read(path)?;
        self.read_info(&data)
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

    /// Reads the header of the JPEG file at `path` and returns the size and layout of each output plane.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or [`read_output_spec`](Self::read_output_spec) fails.
    pub fn read_file_output_spec(&self, path: impl AsRef<Path>) -> Result<OutputSpec> {
        let data = fs::read(path)?;
        self.read_output_spec(&data)
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
            // SAFETY: the completed producer wrote every element of this tightly packed output.
            data: unsafe { output.assume_init() },
            plane,
        })
    }

    /// Reads the JPEG file at `path`, allocates one output plane, decodes into it, and waits for `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or [`decode_single_plane_to_device`](Self::decode_single_plane_to_device) fails.
    pub fn decode_file_single_plane_to_device(
        &mut self,
        path: impl AsRef<Path>,
        stream: &CudaStream,
    ) -> Result<DecodedImage> {
        let data = fs::read(path)?;
        self.decode_single_plane_to_device(&data, stream)
    }

    /// Creates [`DecodeParams`] that match the current options.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, or nvJPEG fails to create the parameters or rejects an option.
    pub fn create_params(&self) -> Result<DecodeParams> {
        let mut params = DecodeParams::create(&self.handle)?;
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

    /// Reads the JPEG file at `path`, decodes it into `destination`, and waits for `stream`.
    ///
    /// Stream capture is not allowed, and the call waits for the GPU work even if it fails.
    /// Only the image's pixels are written; padding and unused memory stay uninitialized.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or [`decode_into`](Self::decode_into) fails.
    pub fn decode_file_into<T: AnyBitPatternDeviceRepr, const C: usize>(
        &mut self,
        path: impl AsRef<Path>,
        destination: &mut ImageViewMut<'_, T, C>,
        stream: &CudaStream,
    ) -> Result<()> {
        let data = fs::read(path)?;
        self.decode_into(&data, destination, stream)
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
        // SAFETY: the completed producer wrote every element of this tightly packed output.
        Ok(unsafe { output.assume_init() })
    }

    /// Reads the JPEG file at `path`, allocates the output planes, decodes into them, and waits for `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or [`decode_to_device`](Self::decode_to_device) fails.
    pub fn decode_file_to_device(
        &mut self,
        path: impl AsRef<Path>,
        stream: &CudaStream,
    ) -> Result<DecodedOutput> {
        let data = fs::read(path)?;
        self.decode_to_device(&data, stream)
    }

    fn decode_descriptor(
        &mut self,
        data: &[u8],
        destination: &mut Image,
        regions: &[DeviceRegion<'_>],
        stream: &CudaStream,
    ) -> Result<()> {
        if self.options.requires_parameterized_decode() {
            return self.decode_with_active_params_descriptor(data, destination, regions, stream);
        }
        complete(stream, || {
            stream.join_regions(regions)?;
            // SAFETY: callers validated the image geometry and retain the inputs,
            // output storage and native descriptors through completion on every path.
            unsafe {
                self.handle.decode_async(
                    &mut self.state,
                    data,
                    self.options.output_format,
                    destination,
                    stream,
                )
            }
        })
    }

    fn decode_with_active_params_descriptor(
        &self,
        data: &[u8],
        destination: &mut Image,
        regions: &[DeviceRegion<'_>],
        stream: &CudaStream,
    ) -> Result<()> {
        let jpeg_stream = JpegStream::create_from_bitstream(&self.handle, data, false)?;
        let params = self.create_params()?;
        let decoder = JpegDecoder::create(&self.handle, self.backend)?;
        let mut state = decoder.create_state()?;
        state.attach_device_buffer(DeviceBuffer::create(&self.handle)?)?;
        complete(stream, || {
            stream.join_regions(regions)?;
            // SAFETY: checked image views establish output capacity and context;
            // all native owners and buffers outlive completion on every path.
            unsafe {
                self.handle.decode_jpeg_async(
                    &decoder,
                    &mut state,
                    &jpeg_stream,
                    destination,
                    &params,
                    stream,
                )
            }
        })
    }
}

impl DecoderBuilder {
    /// Creates a builder with nvJPEG's defaults.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            backend: Backend::Default,
            flags: CreateFlags::DEFAULT,
            options: DecodeOptions::new(),
            allocators: None,
        }
    }

    /// Sets the backend.
    #[must_use]
    pub const fn backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    /// Sets the handle creation flags.
    #[must_use]
    pub const fn flags(mut self, flags: CreateFlags) -> Self {
        self.flags = flags;
        self
    }

    /// Sets the decode options.
    #[must_use]
    pub const fn options(mut self, options: DecodeOptions) -> Self {
        self.options = options;
        self
    }

    /// Sets custom device and pinned host allocators.
    #[must_use]
    pub const fn allocators(mut self, allocators: AllocatorSet) -> Self {
        self.allocators = Some(allocators);
        self
    }

    /// Creates the decoder in `cuda_context`.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, nvJPEG rejects the backend, flags, or allocators, or nvJPEG fails to create the handle or state.
    pub fn create(self, cuda_context: &Arc<CudaContext>) -> Result<Decoder> {
        let mut builder = Context::builder().backend(self.backend).flags(self.flags);
        if let Some(allocators) = self.allocators {
            builder = builder.allocators(allocators);
        }

        let handle = builder.create(cuda_context)?;
        Decoder::from_handle_with_backend(handle, self.backend, self.options)
    }
}

impl Default for DecoderBuilder {
    fn default() -> Self {
        Self::new()
    }
}
