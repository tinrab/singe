use std::{ffi::CString, mem::ManuallyDrop, ptr, slice, sync::Arc};

use singe_core::report_cleanup_failure;
use singe_cuda::{
    context::Context as CudaContext,
    stream::{Stream, StreamOptions},
};
use singe_nvimagecodec_sys as sys;

use crate::{
    code_stream::CodeStream,
    codec::{ExecutionParams, RawBatch, complete_submission, prepare_images, submit_async},
    error::{Error, Result},
    future::Future,
    image::Image,
    instance::{Instance, InstanceRef},
    options::DecoderOptions,
    try_ffi,
    types::{
        CodecFallback, MetadataFormat, MetadataKind, MetadataValueType, NativeStruct,
        ProcessingStatus,
    },
    utility::{collect_raw, to_i32, to_usize, validate_batch, validate_processing_statuses},
};

/// Settings for one decode call.
#[derive(Debug, Clone, Copy)]
pub struct DecodeParams {
    raw: sys::nvimgcodecDecodeParams_t,
}

impl Default for DecodeParams {
    fn default() -> Self {
        Self {
            raw: sys::nvimgcodecDecodeParams_t::init(),
        }
    }
}

impl DecodeParams {
    /// Creates decode settings that ignore the EXIF orientation.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets whether decoding rotates and flips the image as its EXIF orientation says.
    pub fn set_apply_exif_orientation(&mut self, apply: bool) -> &mut Self {
        self.raw.apply_exif_orientation = i32::from(apply);
        self
    }

    /// Returns a pointer to the native parameters, valid while `self` is.
    #[must_use]
    pub fn as_raw(&self) -> *const sys::nvimgcodecDecodeParams_t {
        &raw const self.raw
    }
}

/// Metadata entry read from a code stream, such as EXIF or TIFF tag data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Metadata {
    kind: MetadataKind,
    format: MetadataFormat,
    value_type: MetadataValueType,
    value_count: u32,
    id: u16,
    buffer: Vec<u8>,
}

impl Metadata {
    pub(crate) fn from_raw(raw: sys::nvimgcodecMetadata_t) -> Result<Self> {
        let buffer_size = to_usize(raw.buffer_size, "metadata.buffer_size")?;
        let buffer = if buffer_size == 0 {
            Vec::new()
        } else {
            if raw.buffer.is_null() {
                return Err(Error::NullPointer {
                    name: "metadata.buffer".into(),
                });
            }
            // SAFETY: the metadata buffer holds `buffer_size` bytes owned by the borrowed code stream.
            unsafe { slice::from_raw_parts(raw.buffer.cast::<u8>(), buffer_size) }.to_vec()
        };

        Ok(Self {
            kind: raw.kind.try_into()?,
            format: raw.format.try_into()?,
            value_type: raw.value_type.try_into()?,
            value_count: raw.value_count,
            id: raw.id,
            buffer,
        })
    }

    /// Returns the kind of metadata, such as EXIF or a TIFF tag.
    #[must_use]
    pub const fn kind(&self) -> MetadataKind {
        self.kind
    }

    /// Returns how the metadata bytes are encoded.
    #[must_use]
    pub const fn format(&self) -> MetadataFormat {
        self.format
    }

    /// Returns the type of each metadata value.
    #[must_use]
    pub const fn value_type(&self) -> MetadataValueType {
        self.value_type
    }

    /// Returns the number of metadata values.
    ///
    /// For ASCII values, this counts bytes, including the terminating NUL byte.
    #[must_use]
    pub const fn value_count(&self) -> u32 {
        self.value_count
    }

    /// Returns the TIFF tag ID; other kinds of metadata do not use it.
    #[must_use]
    pub const fn id(&self) -> u16 {
        self.id
    }

    /// Returns the metadata bytes as stored.
    #[must_use]
    pub fn buffer(&self) -> &[u8] {
        &self.buffer
    }

    /// Consumes the entry and returns its bytes.
    #[must_use]
    pub fn into_buffer(self) -> Vec<u8> {
        self.buffer
    }
}

/// Decoder that can be reused across calls.
///
/// It keeps its [`Instance`] and CUDA context alive, and owns the CUDA stream it runs on.
#[derive(Debug)]
pub struct Decoder {
    raw: sys::nvimgcodecDecoder_t,
    instance: InstanceRef,
    execution: Box<ExecutionParams>,
    stream: Option<Stream>,
}

/// Native decoder handle with the values that keep it valid, from [`Decoder::into_raw`].
///
/// Destroy the handle before dropping this value, or pass it back to [`Decoder::from_raw`].
#[derive(Debug)]
pub struct RawDecoder {
    /// Native decoder handle, owned exclusively.
    pub handle: sys::nvimgcodecDecoder_t,
    /// Instance the handle was created with.
    pub instance: Instance,
    /// Execution parameters the handle was created with, at the address nvImageCodec was given.
    pub execution: Box<ExecutionParams>,
    /// Stream in the execution context, or `None` for CPU-only execution.
    pub stream: Option<Stream>,
}

/// Code stream to decode and the image to decode it into.
#[derive(Debug)]
pub struct DecodeItem<'a, 'stream, 'image> {
    /// Compressed input.
    pub stream: &'a CodeStream<'stream>,
    /// Image that receives the decoded pixels.
    pub output: &'a mut Image<'image>,
}

#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "native owners transfer their handles and storage borrows under the adjacent safety contract"
)]
// SAFETY: nvImageCodec does not bind a decoder to its creating thread, and without `Sync` one thread uses it at a time; its stream, boxed execution parameters, and instance owner move with it.
unsafe impl Send for Decoder {}

impl<'a, 'stream, 'image> DecodeItem<'a, 'stream, 'image> {
    /// Pairs `stream` with the image it decodes into.
    pub const fn new(stream: &'a CodeStream<'stream>, output: &'a mut Image<'image>) -> Self {
        Self { stream, output }
    }
}

impl Decoder {
    /// Creates a decoder for `instance` that runs as `execution_params` describes.
    ///
    /// `options` passes settings to the decoders as space-separated `name=value` pairs: `<decoder_id>:<name>=<value>` for one decoder, and `:<name>=<value>` for all decoders that support it, such as `:num_cuda_streams=4` or `:fancy_upsampling=1`.
    /// nvImageCodec's documentation lists the options for each decoder.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, a CUDA stream cannot be created, `options` contains a NUL byte, or nvImageCodec fails to create the decoder.
    pub fn create(
        instance: &Instance,
        execution_params: &ExecutionParams,
        options: Option<&DecoderOptions>,
    ) -> Result<Self> {
        let execution = Box::new(execution_params.clone());
        let stream = execution
            .context()
            .map(|context| {
                context.bind()?;
                context.create_stream(StreamOptions::default())
            })
            .transpose()?;
        let options = options
            .and_then(DecoderOptions::render)
            .map(CString::new)
            .transpose()?;
        let mut raw = ptr::null_mut();
        // SAFETY: `instance` is live, `execution` is boxed so the address nvImageCodec keeps outlives the decoder, `options` is a NUL-terminated string alive through the call, and `raw` is a writable local.
        unsafe {
            try_ffi!(sys::nvimgcodecDecoderCreate(
                instance.as_raw(),
                &raw mut raw,
                execution.as_raw(),
                options
                    .as_ref()
                    .map_or(ptr::null(), |options| options.as_ptr()),
            ))?;
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            raw,
            instance: instance.as_instance_ref(),
            execution,
            stream,
        })
    }

    /// Checks whether the decoder can decode each item with `params`, and returns a status for each.
    ///
    /// `fallback` selects whether the check stops at the preferred codec or also examines lower-priority codecs; see [`CodecFallback`].
    ///
    /// # Errors
    ///
    /// Returns an error if the batch is empty or larger than `i32::MAX`, a stream or image belongs to another instance, the CUDA context cannot be bound, or nvImageCodec fails the check.
    pub fn can_decode(
        &self,
        items: &[DecodeItem<'_, '_, '_>],
        params: &DecodeParams,
        fallback: CodecFallback,
    ) -> Result<Vec<ProcessingStatus>> {
        validate_batch(items.len(), items.len())?;
        for item in items {
            self.instance
                .require_same(item.stream.instance(), "stream")?;
            self.instance
                .require_same(item.output.instance(), "output")?;
        }

        let raw_streams = collect_raw(items, |item| item.stream.as_raw());
        let raw_images = collect_raw(items, |item| item.output.as_raw());
        self.check_decode(&raw_streams, &raw_images, params, fallback)
    }

    /// Checks whether the decoder can decode `stream` into `output` with `params`, and returns the status.
    ///
    /// See [`can_decode`](Self::can_decode) for `fallback`.
    ///
    /// # Errors
    ///
    /// Returns an error if [`can_decode`](Self::can_decode) would fail for this item.
    pub fn can_decode_one(
        &self,
        stream: &CodeStream<'_>,
        output: &Image<'_>,
        params: &DecodeParams,
        fallback: CodecFallback,
    ) -> Result<ProcessingStatus> {
        self.instance.require_same(stream.instance(), "stream")?;
        self.instance.require_same(output.instance(), "output")?;
        let statuses =
            self.check_decode(&[stream.as_raw()], &[output.as_raw()], params, fallback)?;
        Ok(statuses[0])
    }

    fn check_decode(
        &self,
        raw_streams: &[sys::nvimgcodecCodeStream_t],
        raw_images: &[sys::nvimgcodecImage_t],
        params: &DecodeParams,
        fallback: CodecFallback,
    ) -> Result<Vec<ProcessingStatus>> {
        if let Some(context) = self.context() {
            context.bind()?;
        }

        let batch_size = to_i32(raw_streams.len(), "batch_size")?;
        let mut statuses = vec![0; raw_streams.len()];
        // SAFETY: both handle arrays hold `batch_size` live handles of this decoder's instance, and `statuses` has room for one status per item.
        unsafe {
            try_ffi!(sys::nvimgcodecDecoderCanDecode(
                self.raw,
                raw_streams.as_ptr(),
                raw_images.as_ptr(),
                batch_size,
                params.as_raw(),
                statuses.as_mut_ptr(),
                fallback.force_format(),
            ))?;
        }

        Ok(statuses.into_iter().map(ProcessingStatus::from).collect())
    }

    /// Returns the metadata entries in `stream`, such as EXIF data.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another instance, the CUDA context cannot be bound, nvImageCodec fails the query or reports inconsistent counts, or an entry's size does not fit `usize`.
    pub fn metadata(&self, stream: &CodeStream<'_>) -> Result<Vec<Metadata>> {
        self.instance.require_same(stream.instance(), "stream")?;
        if let Some(context) = self.context() {
            context.bind()?;
        }

        let mut count = 0;
        // SAFETY: the decoder and `stream` are live handles of one instance, a null array asks only for the count, and `count` is a writable local.
        unsafe {
            try_ffi!(sys::nvimgcodecDecoderGetMetadata(
                self.raw,
                stream.as_raw(),
                ptr::null_mut(),
                &raw mut count,
            ))?;
        }

        let capacity = to_usize(count, "metadata_count")?;
        if capacity == 0 {
            return Ok(Vec::new());
        }

        let mut metadata = vec![sys::nvimgcodecMetadata_t::init(); capacity];
        let mut metadata_ptr = metadata.as_mut_ptr();
        // SAFETY: the decoder and `stream` are live handles of one instance, `metadata` holds the `count` initialized entries the first call reported, and `metadata_ptr` and `count` are writable locals.
        unsafe {
            try_ffi!(sys::nvimgcodecDecoderGetMetadata(
                self.raw,
                stream.as_raw(),
                &raw mut metadata_ptr,
                &raw mut count,
            ))?;
        }

        let len = to_usize(count, "metadata_count")?;
        if len == 0 {
            return Ok(Vec::new());
        }

        if metadata_ptr.is_null() {
            return Err(Error::NullPointer {
                name: "metadata".into(),
            });
        }

        if metadata_ptr == metadata.as_mut_ptr() && len > metadata.len() {
            return Err(Error::out_of_range("metadata_count"));
        }
        // Plugins may fill the supplied array or return their own metadata array.
        // Copy payloads while both the decoder and code stream remain borrowed.
        // SAFETY: the plugin filled `count` metadata entries that stay valid while the decoder and code stream are borrowed.
        unsafe { slice::from_raw_parts(metadata_ptr, len) }
            .iter()
            .copied()
            .map(Metadata::from_raw)
            .collect()
    }

    fn prepare_decode(
        &self,
        items: &mut [DecodeItem<'_, '_, '_>],
    ) -> Result<(Vec<Stream>, RawBatch)> {
        validate_batch(items.len(), items.len())?;
        for item in items.iter() {
            self.instance
                .require_same(item.stream.instance(), "stream")?;
        }

        let streams = {
            let mut images = items
                .iter_mut()
                .map(|item| &mut *item.output)
                .collect::<Vec<_>>();
            prepare_images(
                &self.instance,
                self.stream.as_ref(),
                &mut images,
                true,
                self.execution.raw.skip_pre_sync != 0,
            )?
        };
        let batch = RawBatch {
            code_streams: collect_raw(items, |item| item.stream.as_raw()),
            images: collect_raw(items, |item| item.output.as_raw()),
        };
        Ok((streams, batch))
    }

    /// Decodes each item with `params` and waits for the work.
    ///
    /// The call waits on the CPU and the GPU before returning, so the images are ready to use.
    ///
    /// # Errors
    ///
    /// Returns an error if the batch is empty or too large, a stream or image belongs to another instance or CUDA context, stream capture is active, a CUDA call or nvImageCodec fails, or an image's processing status is not success.
    pub fn decode(
        &self,
        items: &mut [DecodeItem<'_, '_, '_>],
        params: &DecodeParams,
    ) -> Result<()> {
        let (streams, batch) = self.prepare_decode(items)?;
        let statuses = complete_submission(
            streams,
            batch.len(),
            // SAFETY: the batch holds live handles of this decoder's instance, and `complete_submission` waits for the CPU and GPU work before the items are released.
            |future, batch_size| unsafe {
                sys::nvimgcodecDecoderDecode(
                    self.raw,
                    batch.code_streams.as_ptr(),
                    batch.images.as_ptr(),
                    batch_size,
                    params.as_raw(),
                    future,
                )
            },
        )?;
        for (item, status) in items.iter_mut().zip(&statuses) {
            if *status == ProcessingStatus::SUCCESS {
                item.output.decoded();
            }
        }

        validate_processing_statuses(&statuses)
    }

    /// Decodes `stream` into `output` with `params` and waits for the work.
    ///
    /// # Errors
    ///
    /// Returns an error if [`decode`](Self::decode) would fail for this item.
    pub fn decode_one(
        &self,
        stream: &CodeStream<'_>,
        output: &mut Image<'_>,
        params: &DecodeParams,
    ) -> Result<()> {
        self.decode(&mut [DecodeItem::new(stream, output)], params)
    }

    /// Submits decoding of each item with `params`, and returns a future to wait on.
    ///
    /// The images and streams are checked and prepared as in [`decode`](Self::decode), and work that produces device inputs is waited for before submission.
    /// Output images keep their uninitialized state; after [`Future::wait`] succeeds, read the pixels through the buffers the images borrow.
    ///
    /// # Safety
    ///
    /// The crate's raw execution contract applies: every code stream, image, buffer, and `params` must stay valid, and the output buffers unused by anything else, until the returned future and the GPU work have finished, even if submission fails.
    /// Dropping the future does not wait for the GPU work.
    ///
    /// # Errors
    ///
    /// Returns an error if [`decode`](Self::decode) would fail before submitting, or nvImageCodec fails the submission.
    pub unsafe fn decode_async<'a>(
        &'a self,
        items: &mut [DecodeItem<'_, '_, '_>],
        params: &DecodeParams,
    ) -> Result<Future<'a>> {
        let (streams, batch) = self.prepare_decode(items)?;
        let submit = |batch: &RawBatch, future, batch_size| {
            // SAFETY: the batch holds `batch_size` live handles of this decoder's instance, and the future retains the handle arrays.
            unsafe {
                sys::nvimgcodecDecoderDecode(
                    self.raw,
                    batch.code_streams.as_ptr(),
                    batch.images.as_ptr(),
                    batch_size,
                    params.as_raw(),
                    future,
                )
            }
        };
        // SAFETY: the closure passes the batch's handles to one decode call, and the caller keeps everything the submission uses alive until it finishes.
        unsafe { submit_async(&self.instance, streams, batch, submit) }
    }

    /// Returns the native handle, which this decoder still owns.
    #[must_use]
    pub fn as_raw(&self) -> sys::nvimgcodecDecoder_t {
        self.raw
    }

    /// Takes ownership of a native decoder, from [`into_raw`](Self::into_raw) or created outside this crate.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be valid, owned by nobody else, created with `raw.instance`, and configured with `raw.execution` at its current address.
    /// All work submitted with it outside this crate must have finished.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NullHandle`] if `raw.handle` is null.
    /// Returns [`Error::StreamContextMismatch`] if `raw.stream` is not in the execution context, or is missing for GPU execution.
    pub unsafe fn from_raw(raw: RawDecoder) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }

        if raw.stream.as_ref().map(Stream::context) != raw.execution.context() {
            return Err(Error::StreamContextMismatch);
        }

        Ok(Self {
            raw: raw.handle,
            instance: raw.instance.into_instance_ref(),
            execution: raw.execution,
            stream: raw.stream,
        })
    }

    /// Returns the CUDA context, or `None` for CPU-only execution.
    #[must_use]
    pub fn context(&self) -> Option<&Arc<CudaContext>> {
        self.execution.context()
    }

    /// Releases the native handle with the values that keep it valid.
    ///
    /// Destroy the handle or pass it back to [`from_raw`](Self::from_raw).
    #[must_use]
    pub fn into_raw(self) -> RawDecoder {
        let this = ManuallyDrop::new(self);
        // SAFETY: `this` is never dropped, so each owning field moves out exactly once.
        let (instance, execution, stream) = unsafe {
            (
                ptr::read(&raw const this.instance),
                ptr::read(&raw const this.execution),
                ptr::read(&raw const this.stream),
            )
        };
        RawDecoder {
            handle: this.raw,
            instance: instance.into_instance(),
            execution,
            stream,
        }
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            if let Some(context) = self.execution.context()
                && let Err(error) = context.bind()
            {
                report_cleanup_failure(
                    "bind the cuda context to destroy",
                    "nvimagecodec decoder",
                    &error,
                );
                return;
            }

            // SAFETY: `self.raw` is the live decoder this value owns, its CUDA context is bound, and no `Future` that borrows the decoder remains.
            if let Err(error) = unsafe { try_ffi!(sys::nvimgcodecDecoderDestroy(self.raw)) } {
                report_cleanup_failure("destroy", "nvimagecodec decoder", &error);
            }
        }
    }
}
