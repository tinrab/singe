use std::sync::Arc;

use singe_cuda::{
    context::Context as CudaContext, stream::Stream as CudaStream, view::AnyBitPatternDeviceRepr,
};

use crate::{
    allocator::AllocatorSet,
    context::{Context, JpegState},
    decode::{
        DecodeParams, DecodedImage, DecodedOutput, image_descriptors_from_decoded_images,
        image_descriptors_from_decoded_outputs, image_descriptors_from_views, validate_batch,
    },
    error::{Error, Result},
    image::{Image, ImageViewMut, OutputSpec, Size},
    types::{Backend, ChromaSubsampling, CreateFlags, OutputFormat},
    utility::{complete, to_i32},
};

/// Batch size, CPU thread limit, and output format for a [`BatchedDecoder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BatchConfig {
    batch_size: usize,
    max_cpu_threads: usize,
    output_format: OutputFormat,
}

/// Largest image size and chroma subsampling to allocate for ahead of batched decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BatchPrealloc {
    size: Size,
    subsampling: ChromaSubsampling,
}

/// Decoder for batches of a fixed size.
///
/// It owns its [`Context`] and decoder state, and reuses them across batches.
#[derive(Debug)]
pub struct BatchedDecoder {
    pub(crate) handle: Context,
    state: JpegState,
    config: BatchConfig,
}

/// Builder for a [`BatchedDecoder`] with a custom backend, flags, or allocators.
#[derive(Debug, Clone, Copy)]
pub struct BatchedDecoderBuilder {
    backend: Backend,
    flags: CreateFlags,
    config: BatchConfig,
    allocators: Option<AllocatorSet>,
}

impl BatchConfig {
    /// Creates a configuration for batches of `batch_size` images, decoded with up to `max_cpu_threads` CPU threads into `output_format`.
    ///
    /// # Errors
    ///
    /// Returns an error if `batch_size` or `max_cpu_threads` is zero or does not fit an `i32`.
    pub fn create(
        batch_size: usize,
        max_cpu_threads: usize,
        output_format: OutputFormat,
    ) -> Result<Self> {
        if batch_size == 0 {
            return Err(Error::Empty {
                name: "batch_size".into(),
            });
        }

        if max_cpu_threads == 0 {
            return Err(Error::Empty {
                name: "max_cpu_threads".into(),
            });
        }

        to_i32(batch_size, "batch_size")?;
        to_i32(max_cpu_threads, "max_cpu_threads")?;
        Ok(Self {
            batch_size,
            max_cpu_threads,
            output_format,
        })
    }

    /// Returns the number of images per batch.
    #[must_use]
    pub const fn batch_size(self) -> usize {
        self.batch_size
    }

    /// Returns the maximum number of CPU threads nvJPEG may use.
    #[must_use]
    pub const fn max_cpu_threads(self) -> usize {
        self.max_cpu_threads
    }

    /// Returns the output format.
    #[must_use]
    pub const fn output_format(self) -> OutputFormat {
        self.output_format
    }

    fn validate(self) -> Result<()> {
        if self.batch_size == 0 {
            return Err(Error::Empty {
                name: "batch_size".into(),
            });
        }

        if self.max_cpu_threads == 0 {
            return Err(Error::Empty {
                name: "max_cpu_threads".into(),
            });
        }

        to_i32(self.batch_size, "batch_size")?;
        to_i32(self.max_cpu_threads, "max_cpu_threads")?;
        Ok(())
    }
}

impl BatchPrealloc {
    /// Creates a description of the largest images, of `size` with `subsampling`, to allocate for.
    ///
    /// # Errors
    ///
    /// Returns an error if a dimension is zero.
    pub fn create(size: Size, subsampling: ChromaSubsampling) -> Result<Self> {
        let prealloc = Self { size, subsampling };
        prealloc.validate()?;
        Ok(prealloc)
    }

    /// Returns the largest image size.
    #[must_use]
    pub const fn size(self) -> Size {
        self.size
    }

    /// Returns the chroma subsampling.
    #[must_use]
    pub const fn subsampling(self) -> ChromaSubsampling {
        self.subsampling
    }

    fn validate(self) -> Result<()> {
        if self.size.is_empty() {
            return Err(Error::Empty {
                name: "size".into(),
            });
        }
        Ok(())
    }
}

impl BatchedDecoder {
    /// Returns a builder for a decoder with `config` and a custom backend, flags, or allocators.
    #[must_use]
    pub const fn builder(config: BatchConfig) -> BatchedDecoderBuilder {
        BatchedDecoderBuilder::new(config)
    }

    /// Creates a decoder in `cuda_context` with `config` and the given backend.
    ///
    /// # Errors
    ///
    /// Returns an error if `config` is invalid, the CUDA context cannot be bound, or nvJPEG rejects the settings or fails to create the handle or state.
    pub fn create(
        cuda_context: &Arc<CudaContext>,
        backend: Backend,
        config: BatchConfig,
    ) -> Result<Self> {
        Self::builder(config).backend(backend).create(cuda_context)
    }

    /// Creates a decoder that uses an existing `handle`.
    ///
    /// # Errors
    ///
    /// Returns an error if `config` is invalid, the CUDA context cannot be bound, or nvJPEG fails to create or initialize the state.
    pub fn from_handle(handle: Context, config: BatchConfig) -> Result<Self> {
        config.validate()?;
        let mut state = handle.create_jpeg_state()?;
        handle.decode_batched_initialize(
            &mut state,
            config.batch_size,
            config.max_cpu_threads,
            config.output_format,
        )?;
        Ok(Self {
            handle,
            state,
            config,
        })
    }

    /// Returns the nvJPEG handle.
    #[must_use]
    pub const fn handle(&self) -> &Context {
        &self.handle
    }

    /// Returns the batch configuration.
    #[must_use]
    pub const fn config(&self) -> BatchConfig {
        self.config
    }

    /// Allocates device memory ahead of decoding, for images up to the size in `prealloc`.
    ///
    /// This avoids repeated allocations when images differ in size or subsampling.
    /// Only the hardware backends use it; nvJPEG resizes the memory later if an image is larger.
    ///
    /// # Errors
    ///
    /// Returns an error if a dimension is zero or does not fit a positive `i32`, the CUDA context cannot be bound, or nvJPEG fails to allocate.
    pub fn preallocate(&mut self, prealloc: BatchPrealloc) -> Result<()> {
        prealloc.validate()?;
        self.handle.decode_batched_preallocate(
            &mut self.state,
            self.config.batch_size,
            prealloc.size.width(),
            prealloc.size.height(),
            prealloc.subsampling,
            self.config.output_format,
        )
    }

    /// Sets external Huffman and quantization tables for later batches, or clears them with `None`.
    ///
    /// Use it for JPEG data from TIFF files, whose tables are stored separately.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, the tables are too long for nvJPEG, or nvJPEG rejects them.
    pub fn parse_jpeg_tables(&mut self, tables: Option<&[u8]>) -> Result<()> {
        self.handle
            .decode_batched_parse_jpeg_tables(&mut self.state, tables)
    }

    /// Enqueues decoding of a batch into `destinations` on `stream`.
    ///
    /// # Safety
    ///
    /// Each destination must describe device-accessible buffers large enough for the batch's output format, valid until the work on `stream` finishes.
    /// The [raw execution rules](crate#raw-execution) apply, including on failure and for every replay of a captured graph.
    ///
    /// # Errors
    ///
    /// Returns an error if `inputs` does not have one image per batch slot, `destinations` has a different length, an input is empty or too long for nvJPEG, `stream` belongs to another CUDA context, the context cannot be bound, or nvJPEG fails.
    pub unsafe fn decode_async(
        &mut self,
        inputs: &[&[u8]],
        destinations: &mut [Image],
        stream: &CudaStream,
    ) -> Result<()> {
        validate_batch(self.config.batch_size, inputs.len(), "inputs")?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            self.handle
                .decode_batched_async(&mut self.state, inputs, destinations, stream)
        }
    }

    /// Decodes a full batch into `destinations` and waits for `stream`.
    ///
    /// Stream capture is not allowed, and the call waits for the GPU work even if it fails.
    /// Only the image's pixels are written; padding and unused memory stay uninitialized.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` is being captured or belongs to another CUDA context; the inputs or destinations do not match the batch size; a header cannot be read or the output size cannot be computed; a destination has the wrong plane count, context, alignment, size, or pitch; or nvJPEG or CUDA fails.
    pub fn decode_into<T: AnyBitPatternDeviceRepr, const C: usize>(
        &mut self,
        inputs: &[&[u8]],
        destinations: &mut [&mut ImageViewMut<'_, T, C>],
        stream: &CudaStream,
    ) -> Result<()> {
        stream.ensure_not_capturing()?;
        validate_batch(self.config.batch_size, inputs.len(), "inputs")?;
        let () = T::LAYOUT;
        for (input, destination) in inputs.iter().zip(destinations.iter()) {
            let info = self.handle.image_info(input)?;
            let spec = OutputSpec::for_jpeg(&info, self.config.output_format)?;
            destination.validate_output(&spec, stream.context())?;
        }

        let mut images = image_descriptors_from_views(destinations)?;
        let regions = destinations
            .iter()
            .flat_map(|image| image.planes().iter().map(|plane| plane.region()))
            .collect::<Vec<_>>();
        complete(stream, || {
            stream.join_regions(&regions)?;
            // SAFETY: `complete` drains the stream before returning, so every borrowed buffer outlives the queued work.
            unsafe {
                self.handle
                    .decode_batched_async(&mut self.state, inputs, &mut images, stream)
            }
        })?;
        Ok(())
    }

    /// Allocates one output plane per image, decodes the batch into them, and waits for `stream`.
    ///
    /// Use it with single-plane output formats, such as [`OutputFormat::RgbInterleaved`], [`OutputFormat::BgrInterleaved`], [`OutputFormat::Y`], and [`OutputFormat::Yuy2`].
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` is being captured or belongs to another CUDA context, the inputs do not match the batch size, a header cannot be read or the output size cannot be computed, the output format has more than one plane, an allocation fails, or nvJPEG or CUDA fails.
    pub fn decode_single_planes_to_device(
        &mut self,
        inputs: &[&[u8]],
        stream: &CudaStream,
    ) -> Result<Vec<DecodedImage>> {
        stream.ensure_not_capturing()?;
        validate_batch(self.config.batch_size, inputs.len(), "inputs")?;
        let mut outputs = inputs
            .iter()
            .map(|input| {
                let info = self.handle.image_info(input)?;
                let spec = OutputSpec::for_jpeg(&info, self.config.output_format)?;
                let plane = spec.single_plane()?;
                Ok(DecodedImage {
                    data: plane.create_uninit_buffer(stream.context())?,
                    plane,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut images = image_descriptors_from_decoded_images(&mut outputs)?;
        // SAFETY: `complete` drains the stream before returning, so every borrowed buffer outlives the queued work.
        complete(stream, || unsafe {
            self.handle
                .decode_batched_async(&mut self.state, inputs, &mut images, stream)
        })?;
        // Each tightly packed plane was fully written by the completed decode.
        Ok(outputs
            .into_iter()
            // SAFETY: the completed producer wrote every element of this tightly packed output.
            .map(|output| unsafe { output.assume_init() })
            .collect())
    }

    /// Allocates the output planes for every image, decodes the batch into them, and waits for `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` is being captured or belongs to another CUDA context, the inputs do not match the batch size, a header cannot be read or the output size cannot be computed, an allocation fails, or nvJPEG or CUDA fails.
    pub fn decode_to_device(
        &mut self,
        inputs: &[&[u8]],
        stream: &CudaStream,
    ) -> Result<Vec<DecodedOutput>> {
        stream.ensure_not_capturing()?;
        validate_batch(self.config.batch_size, inputs.len(), "inputs")?;
        let mut outputs = inputs
            .iter()
            .map(|input| {
                let info = self.handle.image_info(input)?;
                let spec = OutputSpec::for_jpeg(&info, self.config.output_format)?;
                DecodedOutput::create_uninit(stream.context(), spec)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut images = image_descriptors_from_decoded_outputs(&mut outputs)?;
        // SAFETY: `complete` drains the stream before returning, so every borrowed buffer outlives the queued work.
        complete(stream, || unsafe {
            self.handle
                .decode_batched_async(&mut self.state, inputs, &mut images, stream)
        })?;
        // Each tightly packed plane was fully written by the completed decode.
        Ok(outputs
            .into_iter()
            // SAFETY: the completed producer wrote every element of this tightly packed output.
            .map(|output| unsafe { output.assume_init() })
            .collect())
    }

    /// Enqueues decoding of a batch into `destinations` on `stream`, with separate decode settings for each image.
    ///
    /// The settings can select a region of each image.
    ///
    /// # Safety
    ///
    /// Each destination must describe device-accessible buffers large enough for its image's settings, valid until the work on `stream` finishes.
    /// The [raw execution rules](crate#raw-execution) apply, including on failure and for every replay of a captured graph.
    ///
    /// # Errors
    ///
    /// Returns an error if `inputs` does not have one image per batch slot, `destinations` or `params` has a different length, a [`DecodeParams`] was created with another [`Context`], an input is empty or too long for nvJPEG, `stream` belongs to another CUDA context, the context cannot be bound, or nvJPEG fails.
    pub unsafe fn decode_with_params_async(
        &mut self,
        inputs: &[&[u8]],
        destinations: &mut [Image],
        params: &mut [&mut DecodeParams],
        stream: &CudaStream,
    ) -> Result<()> {
        validate_batch(self.config.batch_size, inputs.len(), "inputs")?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            self.handle.decode_batched_with_params_async(
                &mut self.state,
                inputs,
                destinations,
                params,
                stream,
            )
        }
    }

    /// Decodes a full batch into `destinations`, with separate decode settings for each image, and waits for `stream`.
    ///
    /// Stream capture is not allowed, and the call waits for the GPU work even if it fails.
    /// Only the image's pixels are written; padding and unused memory stay uninitialized.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` is being captured or belongs to another CUDA context; the inputs, destinations, or `params` do not match the batch size; a [`DecodeParams`] was created with another [`Context`]; a header cannot be read or the output size cannot be computed; a destination has the wrong plane count, context, alignment, size, or pitch; or nvJPEG or CUDA fails.
    pub fn decode_with_params_into<T: AnyBitPatternDeviceRepr, const C: usize>(
        &mut self,
        inputs: &[&[u8]],
        destinations: &mut [&mut ImageViewMut<'_, T, C>],
        params: &mut [&mut DecodeParams],
        stream: &CudaStream,
    ) -> Result<()> {
        stream.ensure_not_capturing()?;
        validate_batch(self.config.batch_size, inputs.len(), "inputs")?;
        validate_batch(self.config.batch_size, params.len(), "params")?;
        let () = T::LAYOUT;
        for ((input, destination), params) in
            inputs.iter().zip(destinations.iter()).zip(params.iter())
        {
            let info = self.handle.image_info(input)?;
            let spec = params.options.output_spec(&info)?;
            destination.validate_output(&spec, stream.context())?;
        }

        let mut images = image_descriptors_from_views(destinations)?;
        let regions = destinations
            .iter()
            .flat_map(|image| image.planes().iter().map(|plane| plane.region()))
            .collect::<Vec<_>>();
        complete(stream, || {
            stream.join_regions(&regions)?;
            // SAFETY: `complete` drains the stream before returning, so every borrowed buffer outlives the queued work.
            unsafe {
                self.handle.decode_batched_with_params_async(
                    &mut self.state,
                    inputs,
                    &mut images,
                    params,
                    stream,
                )
            }
        })?;
        Ok(())
    }
}

impl BatchedDecoderBuilder {
    /// Creates a builder with nvJPEG's defaults.
    #[must_use]
    pub const fn new(config: BatchConfig) -> Self {
        Self {
            backend: Backend::Default,
            flags: CreateFlags::DEFAULT,
            config,
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
    /// Returns an error if the batch configuration is invalid, the CUDA context cannot be bound, nvJPEG rejects the backend, flags, or allocators, or nvJPEG fails to create the handle or state.
    pub fn create(self, cuda_context: &Arc<CudaContext>) -> Result<BatchedDecoder> {
        let mut builder = Context::builder().backend(self.backend).flags(self.flags);
        if let Some(allocators) = self.allocators {
            builder = builder.allocators(allocators);
        }

        let handle = builder.create(cuda_context)?;
        BatchedDecoder::from_handle(handle, self.config)
    }
}
