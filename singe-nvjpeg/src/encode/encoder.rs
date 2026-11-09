use std::{fs, path::Path};

use singe_cuda::{memory::DeviceMemory, stream::Stream as CudaStream};

use crate::{
    allocator::AllocatorSet,
    context::Context,
    encode::{
        EncodeInput, EncodeOptions, EncodeParams, EncodedDeviceBitstream, EncoderState, Quality,
    },
    error::{Error, Result},
    types::{Backend, ChromaSubsampling, CreateFlags, EncoderBackend, JpegEncoding},
    utility::complete,
};

/// Encoder for one image at a time.
///
/// It owns its [`Context`], state, and settings, and reuses them across calls.
#[derive(Debug)]
pub struct Encoder {
    pub(crate) handle: Context,
    state: EncoderState,
    params: EncodeParams,
    options: EncodeOptions,
}

/// Builder for an [`Encoder`] with a custom backend, flags, options, or allocators.
#[derive(Debug, Clone, Copy)]
pub struct EncoderBuilder {
    handle_backend: Backend,
    flags: CreateFlags,
    encoder_backend: EncoderBackend,
    options: EncodeOptions,
    allocators: Option<AllocatorSet>,
}

impl Encoder {
    /// Returns a builder for an encoder with a custom backend, flags, options, or allocators.
    #[must_use]
    pub const fn builder() -> EncoderBuilder {
        EncoderBuilder::new()
    }

    /// Creates an encoder in `stream`'s CUDA context that uses `backend` and the default options.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context or is being captured, the CUDA context cannot be bound, nvJPEG rejects the backend or options or fails to create the handle, state, or settings, or waiting for `stream` fails.
    pub fn create(backend: EncoderBackend, stream: &CudaStream) -> Result<Self> {
        Self::builder().encoder_backend(backend).create(stream)
    }

    /// Creates an encoder in `stream`'s CUDA context that uses `backend` and `options`.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context or is being captured, the CUDA context cannot be bound, nvJPEG rejects the backend or options or fails to create the handle, state, or settings, or waiting for `stream` fails.
    pub fn create_with_options(
        backend: EncoderBackend,
        options: EncodeOptions,
        stream: &CudaStream,
    ) -> Result<Self> {
        Self::builder()
            .encoder_backend(backend)
            .options(options)
            .create(stream)
    }

    /// Creates an encoder that uses an existing `handle`.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context or is being captured, the CUDA context cannot be bound, nvJPEG rejects the backend or options or fails to create the handle, state, or settings, or waiting for `stream` fails.
    pub fn from_handle(
        handle: Context,
        backend: EncoderBackend,
        options: EncodeOptions,
        stream: &CudaStream,
    ) -> Result<Self> {
        let state = EncoderState::create_with_backend(&handle, backend, stream)?;
        let mut params = EncodeParams::create(&handle, stream)?;
        options.apply_to(&mut params, stream)?;
        Ok(Self {
            handle,
            state,
            params,
            options,
        })
    }

    /// Returns the nvJPEG handle.
    #[must_use]
    pub const fn handle(&self) -> &Context {
        &self.handle
    }

    /// Returns the encode options.
    #[must_use]
    pub const fn options(&self) -> EncodeOptions {
        self.options
    }

    /// Sets the quality.
    ///
    /// The options change only if nvJPEG accepts the value.
    /// The call waits for `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context or is being captured, the CUDA context cannot be bound, nvJPEG rejects the setting, or waiting for `stream` fails.
    /// It also fails if `quality` is not a valid [`Quality`].
    pub fn set_quality(
        &mut self,
        quality: impl TryInto<Quality>,
        stream: &CudaStream,
    ) -> Result<&mut Self> {
        let quality = quality.try_into().map_err(|_| Error::OutOfRange {
            name: "quality".into(),
        })?;
        self.params.set_quality(quality, stream)?;
        self.options.quality = Some(quality);
        Ok(self)
    }

    /// Sets the JPEG encoding process.
    ///
    /// The options change only if nvJPEG accepts the value.
    /// The call waits for `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context or is being captured, the CUDA context cannot be bound, nvJPEG rejects the setting, or waiting for `stream` fails.
    pub fn set_encoding(
        &mut self,
        encoding: JpegEncoding,
        stream: &CudaStream,
    ) -> Result<&mut Self> {
        self.params.set_encoding(encoding, stream)?;
        self.options = self.options.with_encoding(encoding);
        Ok(self)
    }

    /// Sets whether to use optimized Huffman tables.
    ///
    /// The options change only if nvJPEG accepts the value.
    /// The call waits for `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context or is being captured, the CUDA context cannot be bound, nvJPEG rejects the setting, or waiting for `stream` fails.
    pub fn set_optimized_huffman(
        &mut self,
        optimized: bool,
        stream: &CudaStream,
    ) -> Result<&mut Self> {
        self.params.set_optimized_huffman(optimized, stream)?;
        self.options = self.options.with_optimized_huffman(optimized);
        Ok(self)
    }

    /// Sets the chroma subsampling of the output.
    ///
    /// The options change only if nvJPEG accepts the value.
    /// The call waits for `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context or is being captured, the CUDA context cannot be bound, nvJPEG rejects the setting, or waiting for `stream` fails.
    pub fn set_sampling(
        &mut self,
        sampling: ChromaSubsampling,
        stream: &CudaStream,
    ) -> Result<&mut Self> {
        self.params.set_sampling_factors(sampling, stream)?;
        self.options = self.options.with_sampling(sampling);
        Ok(self)
    }

    /// Sets the number of minimum coded units between restart markers.
    ///
    /// The options change only if nvJPEG accepts the value.
    /// The call waits for `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context or is being captured, the CUDA context cannot be bound, nvJPEG rejects the setting, or waiting for `stream` fails.
    pub fn set_restart_interval(
        &mut self,
        interval: u32,
        stream: &CudaStream,
    ) -> Result<&mut Self> {
        self.params.set_restart_interval(interval, stream)?;
        self.options = self.options.with_restart_interval(interval);
        Ok(self)
    }

    /// Returns the largest size, in bytes, that `input` can encode to.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, a dimension does not fit a positive `i32`, nvJPEG fails the query, or the size does not fit `usize`.
    pub fn buffer_size(&self, input: &EncodeInput<'_>) -> Result<usize> {
        self.handle
            .encode_buffer_size(&self.params, input.width_i32()?, input.height_i32()?)
    }

    /// Encodes `input` and returns the JPEG data in host memory, waiting for `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context or is being captured, the CUDA context cannot be bound, nvJPEG fails to encode or return the data, a size does not fit, or waiting for `stream` fails.
    pub fn encode_to_vec(
        &mut self,
        input: &EncodeInput<'_>,
        stream: &CudaStream,
    ) -> Result<Vec<u8>> {
        // SAFETY: `complete` drains the stream before returning, so every borrowed buffer outlives the queued work.
        complete(stream, || unsafe {
            self.handle
                .encode_async(&mut self.state, &self.params, input, stream)
        })?;

        let mut output = Vec::with_capacity(self.buffer_size(input)?);
        let length = self
            .handle
            .retrieve_bitstream(&mut self.state, output.spare_capacity_mut(), stream)?
            .len();
        // SAFETY: retrieval returned the completed initialized prefix of spare capacity.
        unsafe {
            output.set_len(length);
        }
        Ok(output)
    }

    /// Encodes `input` and writes the JPEG data to the file at `path`.
    ///
    /// # Errors
    ///
    /// Returns an error if [`encode_to_vec`](Self::encode_to_vec) fails or the file cannot be written.
    pub fn encode_to_file(
        &mut self,
        input: &EncodeInput<'_>,
        path: impl AsRef<Path>,
        stream: &CudaStream,
    ) -> Result<()> {
        let output = self.encode_to_vec(input, stream)?;
        fs::write(path, output)?;
        Ok(())
    }

    /// Encodes `input` and returns the JPEG data in device memory, waiting for `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context or is being captured, the CUDA context cannot be bound, an allocation fails, nvJPEG fails to encode or return the data, a size does not fit, nvJPEG reports more data than the buffer holds, or waiting for `stream` fails.
    pub fn encode_to_device(
        &mut self,
        input: &EncodeInput<'_>,
        stream: &CudaStream,
    ) -> Result<EncodedDeviceBitstream> {
        // SAFETY: `complete` drains the stream before returning, so every borrowed buffer outlives the queued work.
        complete(stream, || unsafe {
            self.handle
                .encode_async(&mut self.state, &self.params, input, stream)
        })?;

        let mut output =
            DeviceMemory::<u8>::create_uninit(stream.context(), self.buffer_size(input)?)?;
        // SAFETY: `complete` drains the stream before returning, so every borrowed buffer outlives the queued work.
        let length = complete(stream, || unsafe {
            self.handle.retrieve_bitstream_device_async(
                &mut self.state,
                Some(output.output()),
                stream,
            )
        })?;
        // SAFETY: the completed retrieval wrote the first `length` bytes of `output`.
        unsafe { EncodedDeviceBitstream::create(output, length) }
    }
}

impl EncoderBuilder {
    /// Creates a builder with nvJPEG's defaults.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            handle_backend: Backend::Default,
            flags: CreateFlags::DEFAULT,
            encoder_backend: EncoderBackend::Default,
            options: EncodeOptions::new(),
            allocators: None,
        }
    }

    /// Sets the backend of the nvJPEG handle.
    #[must_use]
    pub const fn handle_backend(mut self, backend: Backend) -> Self {
        self.handle_backend = backend;
        self
    }

    /// Sets the handle creation flags.
    #[must_use]
    pub const fn flags(mut self, flags: CreateFlags) -> Self {
        self.flags = flags;
        self
    }

    /// Sets the encoder backend.
    #[must_use]
    pub const fn encoder_backend(mut self, backend: EncoderBackend) -> Self {
        self.encoder_backend = backend;
        self
    }

    /// Sets the encode options.
    #[must_use]
    pub const fn options(mut self, options: EncodeOptions) -> Self {
        self.options = options;
        self
    }

    /// Sets custom device and pinned host allocators.
    #[must_use]
    pub const fn allocators(mut self, allocators: AllocatorSet) -> Self {
        self.allocators = Some(allocators);
        self
    }

    /// Creates the encoder in `stream`'s CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context or is being captured, the CUDA context cannot be bound, nvJPEG rejects the backend or options or fails to create the handle, state, or settings, or waiting for `stream` fails.
    pub fn create(self, stream: &CudaStream) -> Result<Encoder> {
        let mut builder = Context::builder()
            .backend(self.handle_backend)
            .flags(self.flags);
        if let Some(allocators) = self.allocators {
            builder = builder.allocators(allocators);
        }

        let handle = builder.create(stream.context())?;
        Encoder::from_handle(handle, self.encoder_backend, self.options, stream)
    }
}

impl Default for EncoderBuilder {
    fn default() -> Self {
        Self::new()
    }
}
