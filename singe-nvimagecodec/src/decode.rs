use std::{ffi::CString, ptr, slice, sync::Arc};

use singe_cuda::{context::Context as CudaContext, stream::Stream};
use singe_nvimagecodec_sys as sys;

use crate::{
    code_stream::CodeStream,
    codec::{ExecutionParams, complete_submission, prepare_images},
    error::{Error, Result},
    future::Future,
    image::Image,
    instance::{Instance, InstanceRef},
    options::DecoderOptions,
    try_ffi,
    types::{MetadataFormat, MetadataKind, MetadataValueType, ProcessingStatus},
    utility::{NativeStruct, collect_raw, to_i32, validate_batch, validate_processing_statuses},
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
        let buffer_size = usize::try_from(raw.buffer_size)
            .map_err(|_| Error::out_of_range("metadata.buffer_size"))?;
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
            kind: raw.kind.into(),
            format: raw.format.into(),
            value_type: raw.value_type.into(),
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
    /// Native decoder handle.
    pub handle: sys::nvimgcodecDecoder_t,
    owner: Decoder,
}

// SAFETY: each transfer carries exclusive codec ownership and all retained execution resources.
unsafe impl Send for RawDecoder {}

/// Code stream to decode and the image to decode it into.
#[derive(Debug)]
pub struct DecodeItem<'a, 'stream, 'image> {
    /// Compressed input.
    pub stream: &'a CodeStream<'stream>,
    /// Image that receives the decoded pixels.
    pub output: &'a mut Image<'image>,
}

/// Arguments for [`Decoder::decode_raw_async`].
#[derive(Debug, Clone, Copy)]
pub struct DecodeRawAsync<'a> {
    /// Array of `batch_size` native code-stream handles.
    pub streams: *const sys::nvimgcodecCodeStream_t,
    /// Array of `batch_size` native image handles.
    pub images: *const sys::nvimgcodecImage_t,
    /// Number of images and code streams.
    pub batch_size: usize,
    /// Decode settings.
    pub params: &'a DecodeParams,
}

// SAFETY: moving a codec transfers exclusive ownership of its native workers and retained context.
#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "native owners transfer their handles and storage borrows under the adjacent safety contract"
)]
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
                context.create_stream()
            })
            .transpose()?;
        let options = options
            .and_then(DecoderOptions::render)
            .map(CString::new)
            .transpose()?;
        let mut raw = ptr::null_mut();
        // SAFETY: `nvimgcodecDecoderCreate` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
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

    /// Checks whether the decoder can decode each of `streams` into the matching image of `images` with `params`, and returns a status for each.
    ///
    /// With `force_format` set, nvImageCodec falls back to lower-priority codecs when the preferred one does not support the format, and checks them too.
    /// Without it, when the preferred codec can handle the input but not the requested format or parameters, the check stops there, and the status flags show what to change to avoid falling back.
    ///
    /// # Errors
    ///
    /// Returns an error if the batch is empty or larger than `i32::MAX`, the slices have different lengths, a stream or image belongs to another instance, the CUDA context cannot be bound, or nvImageCodec fails the check.
    pub fn can_decode(
        &self,
        streams: &[&CodeStream<'_>],
        images: &[&Image<'_>],
        params: &DecodeParams,
        force_format: bool,
    ) -> Result<Vec<ProcessingStatus>> {
        validate_batch(streams.len(), images.len())?;
        if let Some(context) = self.context() {
            context.bind()?;
        }

        for stream in streams {
            self.instance.require_same(stream.instance())?;
        }

        for image in images {
            self.instance.require_same(image.instance())?;
        }

        let batch_size = to_i32(streams.len(), "batch_size")?;
        let raw_streams = collect_raw(streams, |stream| stream.as_raw());
        let raw_images = collect_raw(images, |image| image.as_raw());
        let mut statuses = vec![0; streams.len()];

        // SAFETY: `nvimgcodecDecoderCanDecode` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::nvimgcodecDecoderCanDecode(
                self.raw,
                raw_streams.as_ptr(),
                raw_images.as_ptr(),
                batch_size,
                params.as_raw(),
                statuses.as_mut_ptr(),
                i32::from(force_format),
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
        self.instance.require_same(stream.instance())?;
        if let Some(context) = self.context() {
            context.bind()?;
        }

        let mut count = 0;
        // SAFETY: `nvimgcodecDecoderGetMetadata` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::nvimgcodecDecoderGetMetadata(
                self.raw,
                stream.as_raw(),
                ptr::null_mut(),
                &raw mut count,
            ))?;
        }

        if count < 0 {
            return Err(Error::out_of_range("metadata_count"));
        }

        if count == 0 {
            return Ok(Vec::new());
        }

        let mut metadata = vec![sys::nvimgcodecMetadata_t::init(); count as usize];
        let mut metadata_ptr = metadata.as_mut_ptr();
        // SAFETY: `nvimgcodecDecoderGetMetadata` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::nvimgcodecDecoderGetMetadata(
                self.raw,
                stream.as_raw(),
                &raw mut metadata_ptr,
                &raw mut count,
            ))?;
        }

        if count < 0 {
            return Err(Error::out_of_range("metadata_count"));
        }

        if count == 0 {
            return Ok(Vec::new());
        }

        if metadata_ptr.is_null() {
            return Err(Error::NullPointer {
                name: "metadata".into(),
            });
        }

        if metadata_ptr == metadata.as_mut_ptr() && count as usize > metadata.len() {
            return Err(Error::LengthMismatch {
                name: "metadata_count".into(),
                expected: metadata.len(),
                actual: count as usize,
            });
        }
        // Plugins may fill the supplied array or return their own metadata array.
        // Copy payloads while both the decoder and code stream remain borrowed.
        // SAFETY: the plugin filled `count` metadata entries that stay valid while the decoder and code stream are borrowed.
        unsafe { slice::from_raw_parts(metadata_ptr, count as usize) }
            .iter()
            .copied()
            .map(Metadata::from_raw)
            .collect()
    }

    /// Checks whether the decoder can decode each item with `params`, and returns a status for each.
    ///
    /// See [`can_decode`](Self::can_decode) for `force_format`.
    ///
    /// # Errors
    ///
    /// Returns an error if the batch is empty or larger than `i32::MAX`, the slices have different lengths, a stream or image belongs to another instance, the CUDA context cannot be bound, or nvImageCodec fails the check.
    pub fn can_decode_batch(
        &self,
        items: &[DecodeItem<'_, '_, '_>],
        params: &DecodeParams,
        force_format: bool,
    ) -> Result<Vec<ProcessingStatus>> {
        validate_batch(items.len(), items.len())?;
        if let Some(context) = self.context() {
            context.bind()?;
        }

        for item in items {
            self.instance.require_same(item.stream.instance())?;
            self.instance.require_same(item.output.instance())?;
        }

        let batch_size = to_i32(items.len(), "batch_size")?;
        let raw_streams = collect_raw(items, |item| item.stream.as_raw());
        let raw_images = collect_raw(items, |item| item.output.as_raw());
        let mut statuses = vec![0; items.len()];

        // SAFETY: `nvimgcodecDecoderCanDecode` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::nvimgcodecDecoderCanDecode(
                self.raw,
                raw_streams.as_ptr(),
                raw_images.as_ptr(),
                batch_size,
                params.as_raw(),
                statuses.as_mut_ptr(),
                i32::from(force_format),
            ))?;
        }

        Ok(statuses.into_iter().map(ProcessingStatus::from).collect())
    }

    fn submit_decode(
        &mut self,
        streams: &[&CodeStream<'_>],
        images: &mut [&mut Image<'_>],
        params: &DecodeParams,
    ) -> Result<Vec<ProcessingStatus>> {
        validate_batch(streams.len(), images.len())?;
        for stream in streams {
            self.instance.require_same(stream.instance())?;
        }

        let completion_streams = prepare_images(
            &self.instance,
            self.stream.as_ref(),
            images,
            true,
            self.execution.raw.skip_pre_sync != 0,
        )?;
        let raw_streams = collect_raw(streams, |stream| stream.as_raw());
        let raw_images = collect_raw(images, |image| image.as_raw());
        let statuses = complete_submission(
            completion_streams,
            images.len(),
            // SAFETY: `nvimgcodecDecoderDecode` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
            |future, batch_size| unsafe {
                sys::nvimgcodecDecoderDecode(
                    self.raw,
                    raw_streams.as_ptr(),
                    raw_images.as_ptr(),
                    batch_size,
                    params.as_raw(),
                    future,
                )
            },
        )?;
        for (image, status) in images.iter_mut().zip(&statuses) {
            if *status == ProcessingStatus::SUCCESS {
                image.decoded();
            }
        }

        validate_processing_statuses(&statuses)?;
        Ok(statuses)
    }

    /// Decodes each item with `params`, waits for the work, and returns a status for each.
    ///
    /// The call waits on the CPU and the GPU before returning, so the images are ready to use.
    ///
    /// # Errors
    ///
    /// Returns an error if the batch is empty or too large, a stream or image belongs to another instance or CUDA context, stream capture is active, a CUDA call or nvImageCodec fails, or an image's processing status is not success.
    pub fn decode(
        &mut self,
        items: &mut [DecodeItem<'_, '_, '_>],
        params: &DecodeParams,
    ) -> Result<Vec<ProcessingStatus>> {
        let streams = items.iter().map(|item| item.stream).collect::<Vec<_>>();
        let mut images = items
            .iter_mut()
            .map(|item| &mut *item.output)
            .collect::<Vec<_>>();
        self.submit_decode(&streams, &mut images, params)
    }

    /// Decodes `stream` into `output` with `params`, waits for the work, and returns the processing status.
    ///
    /// # Errors
    ///
    /// Returns an error if [`decode`](Self::decode) would fail for this item.
    pub fn decode_one(
        &mut self,
        stream: &CodeStream<'_>,
        output: &mut Image<'_>,
        params: &DecodeParams,
    ) -> Result<ProcessingStatus> {
        let statuses = self.submit_decode(&[stream], &mut [output], params)?;
        Ok(statuses[0])
    }

    /// Submits decoding of each native stream into the matching native image, and returns a future to wait on.
    ///
    /// # Safety
    ///
    /// `params.streams` and `params.images` must point to arrays of `params.batch_size` valid handles created with this decoder's instance and CUDA context.
    /// The handles, arrays, parameters, and all image and bitstream storage they use must stay valid until the returned future and the GPU work have finished, even if submission fails.
    /// The output images and their storage must not be used by anything else until then.
    /// This crate skips nvImageCodec's own wait for input streams, so work that produces the inputs must finish before this call.
    /// Dropping the future does not wait for the GPU work.
    ///
    /// # Errors
    ///
    /// Returns an error if the batch is empty or larger than `i32::MAX`, a handle array is null, the CUDA context cannot be bound, or nvImageCodec fails the submission.
    pub unsafe fn decode_raw_async<'a>(&'a self, params: DecodeRawAsync<'_>) -> Result<Future<'a>> {
        if params.batch_size == 0 {
            return Err(Error::Empty {
                name: "batch".into(),
            });
        }

        if params.streams.is_null() {
            return Err(Error::NullPointer {
                name: "streams".into(),
            });
        }

        if params.images.is_null() {
            return Err(Error::NullPointer {
                name: "images".into(),
            });
        }

        let batch_size = to_i32(params.batch_size, "batch_size")?;
        let mut future = ptr::null_mut();
        if let Some(context) = self.context() {
            context.bind()?;
        }

        // SAFETY: `nvimgcodecDecoderDecode` receives live handles owned by the wrappers and valid output locations.
        let status = unsafe {
            sys::nvimgcodecDecoderDecode(
                self.raw,
                params.streams,
                params.images,
                batch_size,
                params.params.as_raw(),
                &raw mut future,
            )
        };
        if let Err(error) = try_ffi!(status) {
            if !future.is_null() {
                drop(Future::create(
                    future,
                    self.instance.clone(),
                    params.batch_size,
                )?);
            }
            return Err(error);
        }

        Future::create(future, self.instance.clone(), params.batch_size)
    }

    /// Returns the native handle, which this decoder still owns.
    #[must_use]
    pub fn as_raw(&self) -> sys::nvimgcodecDecoder_t {
        self.raw
    }

    /// Takes ownership of a native decoder created outside this crate.
    ///
    /// # Safety
    ///
    /// `raw` must be owned by nobody else, created with `instance`, and configured as `execution_params` describes.
    /// All work that used `raw` must have finished.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw` is null or the CUDA stream for `execution_params`'s context cannot be created.
    pub unsafe fn from_raw_parts(
        instance: &Instance,
        raw: sys::nvimgcodecDecoder_t,
        execution_params: &ExecutionParams,
    ) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        let execution = Box::new(execution_params.clone());
        let stream = execution
            .context()
            .map(CudaContext::create_stream)
            .transpose()?;
        Ok(Self {
            raw,
            instance: instance.as_instance_ref(),
            execution,
            stream,
        })
    }

    /// Takes back ownership of a handle from [`into_raw`](Self::into_raw).
    ///
    /// # Safety
    ///
    /// The handle must still be valid and owned by nobody else, and all work submitted with it outside this crate must have finished.
    #[must_use]
    pub unsafe fn from_raw(mut raw: RawDecoder) -> Self {
        raw.owner.raw = raw.handle;
        raw.owner
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
    pub fn into_raw(mut self) -> RawDecoder {
        let handle = self.raw;
        self.raw = ptr::null_mut();
        RawDecoder {
            handle,
            owner: self,
        }
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            if let Some(context) = self.execution.context() {
                let _ = context.bind();
            }
            // SAFETY: `nvimgcodecDecoderDestroy` receives live handles owned by the wrappers.
            unsafe {
                if let Err(error) = try_ffi!(sys::nvimgcodecDecoderDestroy(self.raw)) {
                    eprintln!(
                        "failed to free nvimagecodec decoder: {}",
                        singe_core::ErrorChain(&error)
                    );
                }
            }
        }
    }
}
