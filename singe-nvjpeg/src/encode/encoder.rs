use singe_cuda::{memory::DeviceMemory, stream::Stream as CudaStream};

use crate::{
    context::Context,
    encode::{EncodeInput, EncodeOptions, EncodeParams, EncodedDeviceBitstream, EncoderState},
    error::Result,
    types::EncoderBackend,
};

/// Encoder for one image at a time.
///
/// It keeps its [`Context`], state, and settings, and reuses them across calls.
#[derive(Debug)]
pub struct Encoder {
    context: Context,
    state: EncoderState,
    params: EncodeParams,
    options: EncodeOptions,
}

impl Encoder {
    /// Creates an encoder for `context` that uses `backend` and `options`, and waits for `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context or is being captured, the CUDA context cannot be bound, nvJPEG rejects the backend or options or fails to create the state or settings, or waiting for `stream` fails.
    pub fn create(
        context: &Context,
        backend: EncoderBackend,
        options: EncodeOptions,
        stream: &CudaStream,
    ) -> Result<Self> {
        let state = EncoderState::create(context, backend, stream)?;
        let params = options.create_params(context, stream)?;
        Ok(Self {
            context: context.clone(),
            state,
            params,
            options,
        })
    }

    /// Returns the nvJPEG handle.
    #[must_use]
    pub const fn context(&self) -> &Context {
        &self.context
    }

    /// Returns the encode options.
    #[must_use]
    pub const fn options(&self) -> EncodeOptions {
        self.options
    }

    /// Replaces the encode options, and waits for `stream`.
    ///
    /// The options change only if nvJPEG accepts every value; settings that `options` leaves unset return to nvJPEG's defaults.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context or is being captured, the CUDA context cannot be bound, nvJPEG fails to create the settings or rejects a value, or waiting for `stream` fails.
    pub fn set_options(&mut self, options: EncodeOptions, stream: &CudaStream) -> Result<()> {
        self.params = options.create_params(&self.context, stream)?;
        self.options = options;
        Ok(())
    }

    /// Returns the largest size, in bytes, that `input` can encode to.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, a dimension does not fit a positive `i32`, nvJPEG fails the query, or the size does not fit `usize`.
    pub fn buffer_size(&self, input: &EncodeInput<'_>) -> Result<usize> {
        self.context
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
        // SAFETY: an `EncodeInput` holds initialized pixels checked against its format and stays borrowed, and `complete` waits for `stream` on every path.
        stream.complete(|| unsafe {
            self.context
                .encode_async(&mut self.state, &self.params, input, stream)
        })?;

        let mut output = Vec::with_capacity(self.buffer_size(input)?);
        let length = self
            .context
            .retrieve_bitstream(&mut self.state, output.spare_capacity_mut(), stream)?
            .len();
        // SAFETY: the finished copy initialized the first `length` bytes of the spare capacity, which `retrieve_bitstream` checked against its size.
        unsafe {
            output.set_len(length);
        }
        Ok(output)
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
        // SAFETY: an `EncodeInput` holds initialized pixels checked against its format and stays borrowed, and `complete` waits for `stream` on every path.
        stream.complete(|| unsafe {
            self.context
                .encode_async(&mut self.state, &self.params, input, stream)
        })?;

        let mut output =
            DeviceMemory::<u8>::create_uninit(stream.context(), self.buffer_size(input)?)?;
        // SAFETY: `output` is an allocation owned here of the size nvJPEG reported, its region is passed for ordering, and `complete` waits for `stream` on every path.
        let length = stream.complete(|| unsafe {
            self.context.retrieve_bitstream_device_async(
                &mut self.state,
                Some(output.output()),
                stream,
            )
        })?;
        // SAFETY: the completed retrieval wrote the first `length` bytes of `output`.
        unsafe { EncodedDeviceBitstream::create(output, length) }
    }
}
