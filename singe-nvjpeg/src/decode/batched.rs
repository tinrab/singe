use singe_cuda::{stream::Stream as CudaStream, view::AnyBitPatternDeviceRepr};

use crate::{
    context::{Context, JpegState},
    decode::{
        DecodeParams, DecodedImage, DecodedOutput, image_descriptors_from_decoded_images,
        image_descriptors_from_decoded_outputs, image_descriptors_from_views,
    },
    error::{Error, Result},
    image::{Image, ImageViewMut, OutputSpec, Size},
    types::{ChromaSubsampling, OutputFormat},
    utility::{to_i32, validate_same_len},
};

/// Batch size and output format for a [`BatchedDecoder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BatchedDecoderOptions {
    batch_size: usize,
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
/// It keeps its [`Context`] and decoder state, and reuses them across batches.
#[derive(Debug)]
pub struct BatchedDecoder {
    context: Context,
    state: JpegState,
    options: BatchedDecoderOptions,
}

impl BatchedDecoderOptions {
    /// Creates options for batches of `batch_size` images decoded into `output_format`.
    ///
    /// # Errors
    ///
    /// Returns an error if `batch_size` is zero or does not fit an `i32`.
    pub fn create(batch_size: usize, output_format: OutputFormat) -> Result<Self> {
        let options = Self {
            batch_size,
            output_format,
        };
        options.validate()?;
        Ok(options)
    }

    /// Returns a copy with `output_format` as the output format.
    #[must_use]
    pub const fn with_output_format(mut self, output_format: OutputFormat) -> Self {
        self.output_format = output_format;
        self
    }

    /// Returns the number of images per batch.
    #[must_use]
    pub const fn batch_size(self) -> usize {
        self.batch_size
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

        to_i32(self.batch_size, "batch_size")?;
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
    /// Creates a decoder for `context` with `options`.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, or nvJPEG fails to create or initialize the state.
    pub fn create(context: &Context, options: BatchedDecoderOptions) -> Result<Self> {
        options.validate()?;
        let mut state = context.create_jpeg_state()?;
        context.decode_batched_initialize(&mut state, options.batch_size, options.output_format)?;
        Ok(Self {
            context: context.clone(),
            state,
            options,
        })
    }

    /// Returns the nvJPEG handle.
    #[must_use]
    pub const fn context(&self) -> &Context {
        &self.context
    }

    /// Returns the batch options.
    #[must_use]
    pub const fn options(&self) -> BatchedDecoderOptions {
        self.options
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
        self.context.decode_batched_preallocate(
            &mut self.state,
            self.options.batch_size,
            prealloc.size.width(),
            prealloc.size.height(),
            prealloc.subsampling,
            self.options.output_format,
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
        self.context
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
        validate_same_len(self.options.batch_size, inputs.len(), "inputs")?;
        // SAFETY: this function's caller guarantees the destinations are large enough and outlive the queued work, which is the callee's contract.
        unsafe {
            self.context
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
        validate_same_len(self.options.batch_size, inputs.len(), "inputs")?;
        let () = T::LAYOUT;
        for (input, destination) in inputs.iter().zip(destinations.iter()) {
            let info = self.context.image_info(input)?;
            let spec = OutputSpec::for_jpeg(&info, self.options.output_format)?;
            destination.validate_output(&spec, stream.context())?;
        }

        let mut images = image_descriptors_from_views(destinations)?;
        let regions = destinations
            .iter()
            .flat_map(|image| image.planes().iter().map(|plane| plane.region()))
            .collect::<Vec<_>>();
        stream.complete(|| {
            stream.join_regions(&regions)?;
            // SAFETY: each destination was validated against its header above, and `complete` waits for `stream` on every path, so the borrowed inputs and planes outlive the work.
            unsafe {
                self.context
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
        validate_same_len(self.options.batch_size, inputs.len(), "inputs")?;
        let mut outputs = inputs
            .iter()
            .map(|input| {
                let info = self.context.image_info(input)?;
                let spec = OutputSpec::for_jpeg(&info, self.options.output_format)?;
                let plane = spec.single_plane()?;
                Ok(DecodedImage {
                    data: plane.create_uninit_buffer(stream.context())?,
                    plane,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut images = image_descriptors_from_decoded_images(&mut outputs)?;
        // SAFETY: each output was allocated from its header's plane size, and `complete` waits for `stream` on every path, so the inputs and outputs outlive the work.
        stream.complete(|| unsafe {
            self.context
                .decode_batched_async(&mut self.state, inputs, &mut images, stream)
        })?;
        // Each tightly packed plane was fully written by the completed decode.
        Ok(outputs
            .into_iter()
            // SAFETY: the decode above finished and wrote every byte of this tightly packed output.
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
        validate_same_len(self.options.batch_size, inputs.len(), "inputs")?;
        let mut outputs = inputs
            .iter()
            .map(|input| {
                let info = self.context.image_info(input)?;
                let spec = OutputSpec::for_jpeg(&info, self.options.output_format)?;
                DecodedOutput::create_uninit(stream.context(), spec)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut images = image_descriptors_from_decoded_outputs(&mut outputs)?;
        // SAFETY: each output was allocated from its header's output spec, and `complete` waits for `stream` on every path, so the inputs and outputs outlive the work.
        stream.complete(|| unsafe {
            self.context
                .decode_batched_async(&mut self.state, inputs, &mut images, stream)
        })?;
        // Each tightly packed plane was fully written by the completed decode.
        Ok(outputs
            .into_iter()
            // SAFETY: the decode above finished and wrote every byte of this tightly packed output.
            .map(|output| unsafe { output.assume_init() })
            .collect())
    }

    /// Enqueues decoding of a batch into `destinations` on `stream`, with separate decode settings for each image.
    ///
    /// The settings can select a region of each image.
    ///
    /// # Safety
    ///
    /// Each destination must describe device-accessible buffers large enough for the batch's output format and its image's settings, valid until the work on `stream` finishes.
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
        validate_same_len(self.options.batch_size, inputs.len(), "inputs")?;
        // SAFETY: this function's caller guarantees the destinations are large enough and outlive the queued work, which is the callee's contract.
        unsafe {
            self.context.decode_batched_with_params_async(
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
        validate_same_len(self.options.batch_size, inputs.len(), "inputs")?;
        validate_same_len(self.options.batch_size, params.len(), "params")?;
        let () = T::LAYOUT;
        for ((input, destination), params) in
            inputs.iter().zip(destinations.iter()).zip(params.iter())
        {
            let info = self.context.image_info(input)?;
            // nvJPEG writes batched output in the batch's format, ignoring the params' format.
            let spec = params
                .options
                .with_output_format(self.options.output_format)
                .output_spec(&info)?;
            destination.validate_output(&spec, stream.context())?;
        }

        let mut images = image_descriptors_from_views(destinations)?;
        let regions = destinations
            .iter()
            .flat_map(|image| image.planes().iter().map(|plane| plane.region()))
            .collect::<Vec<_>>();
        stream.complete(|| {
            stream.join_regions(&regions)?;
            // SAFETY: each destination was validated against its header and settings above, and `complete` waits for `stream` on every path, so the borrowed inputs and planes outlive the work.
            unsafe {
                self.context.decode_batched_with_params_async(
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
