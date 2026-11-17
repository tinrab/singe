use std::{ffi::CString, mem::ManuallyDrop, ptr, sync::Arc};

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
    options::EncoderOptions,
    try_ffi,
    types::{
        CodecFallback, Jpeg2kBitstreamType, Jpeg2kProgressionOrder, NativeStruct, ProcessingStatus,
        QualityType,
    },
    utility::{collect_raw, to_i32, validate_batch, validate_processing_statuses},
};

/// Settings for one encode call: quality and JPEG or JPEG 2000 options.
#[derive(Debug)]
pub struct EncodeParams {
    raw: sys::nvimgcodecEncodeParams_t,
    jpeg: Option<Box<sys::nvimgcodecJpegEncodeParams_t>>,
    jpeg2k: Option<Box<sys::nvimgcodecJpeg2kEncodeParams_t>>,
}

impl Default for EncodeParams {
    fn default() -> Self {
        Self {
            raw: sys::nvimgcodecEncodeParams_t::init(),
            jpeg: None,
            jpeg2k: None,
        }
    }
}

impl Clone for EncodeParams {
    fn clone(&self) -> Self {
        let mut cloned = Self {
            raw: self.raw,
            jpeg: self.jpeg.clone(),
            jpeg2k: self.jpeg2k.clone(),
        };
        cloned.rebuild_extension_chain();
        cloned
    }
}

impl EncodeParams {
    /// Creates encode settings with the default quality metric and no JPEG or JPEG 2000 options.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the quality metric and its value.
    ///
    /// The value is passed to nvImageCodec unchecked.
    /// The accepted range depends on the codec; [`Encoder::can_encode`] or encoding reports values it rejects.
    pub fn set_quality(&mut self, quality_type: QualityType, value: f32) -> &mut Self {
        self.raw.quality_type = quality_type.into();
        self.raw.quality_value = value;
        self
    }

    /// Sets the JPEG options.
    pub fn set_jpeg(&mut self, params: JpegEncodeParams) -> &mut Self {
        self.jpeg = Some(Box::new(params.raw));
        self.rebuild_extension_chain();
        self
    }

    /// Sets the JPEG 2000 options.
    pub fn set_jpeg2k(&mut self, params: Jpeg2kEncodeParams) -> &mut Self {
        self.jpeg2k = Some(Box::new(params.raw));
        self.rebuild_extension_chain();
        self
    }

    /// Returns a pointer to the native parameters and their codec options, valid while `self` is.
    #[must_use]
    pub fn as_raw(&self) -> *const sys::nvimgcodecEncodeParams_t {
        &raw const self.raw
    }

    fn rebuild_extension_chain(&mut self) {
        self.raw.struct_next = ptr::null_mut();
        if let Some(jpeg) = self.jpeg.as_mut() {
            jpeg.struct_next = ptr::null_mut();
        }

        if let Some(jpeg2k) = self.jpeg2k.as_mut() {
            jpeg2k.struct_next = ptr::null_mut();
        }

        let mut next = &mut self.raw.struct_next;
        if let Some(jpeg) = self.jpeg.as_mut() {
            *next = (&raw mut **jpeg).cast();
            next = &mut jpeg.struct_next;
        }
        if let Some(jpeg2k) = self.jpeg2k.as_mut() {
            *next = (&raw mut **jpeg2k).cast();
        }
    }
}

/// JPEG encoding options.
#[derive(Debug, Clone, Copy)]
pub struct JpegEncodeParams {
    raw: sys::nvimgcodecJpegEncodeParams_t,
}

impl Default for JpegEncodeParams {
    fn default() -> Self {
        Self {
            raw: sys::nvimgcodecJpegEncodeParams_t::init(),
        }
    }
}

impl JpegEncodeParams {
    /// Creates JPEG options without optimized Huffman tables.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets whether to use optimized Huffman tables, which give smaller files of the same quality but encode more slowly.
    pub fn set_optimized_huffman(&mut self, optimized: bool) -> &mut Self {
        self.raw.optimized_huffman = i32::from(optimized);
        self
    }
}

/// JPEG 2000 encoding options.
#[derive(Debug, Clone, Copy)]
pub struct Jpeg2kEncodeParams {
    raw: sys::nvimgcodecJpeg2kEncodeParams_t,
}

impl Default for Jpeg2kEncodeParams {
    fn default() -> Self {
        Self {
            raw: sys::nvimgcodecJpeg2kEncodeParams_t::init(),
        }
    }
}

impl Jpeg2kEncodeParams {
    /// Creates JPEG 2000 options with every field zeroed.
    ///
    /// Set the fields the chosen backend needs.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets whether to write a bare codestream or a JP2 file.
    pub fn set_stream_type(&mut self, stream_type: Jpeg2kBitstreamType) -> &mut Self {
        self.raw.stream_type = stream_type.into();
        self
    }

    /// Sets the progression order of packets in the codestream.
    pub fn set_progression_order(&mut self, order: Jpeg2kProgressionOrder) -> &mut Self {
        self.raw.prog_order = order.into();
        self
    }

    /// Sets the number of resolution levels.
    ///
    /// The value is passed to nvImageCodec unchecked; the valid range depends on the codec and image size.
    pub fn set_num_resolutions(&mut self, resolutions: u32) -> &mut Self {
        self.raw.num_resolutions = resolutions;
        self
    }

    /// Sets the code block width and height.
    ///
    /// # Errors
    ///
    /// Returns an error if either dimension is not 32 or 64.
    pub fn set_code_block_size(&mut self, width: u32, height: u32) -> Result<&mut Self> {
        validate_jpeg2k_code_block_size(width, "code_block_w")?;
        validate_jpeg2k_code_block_size(height, "code_block_h")?;
        self.raw.code_block_w = width;
        self.raw.code_block_h = height;
        Ok(self)
    }

    /// Sets whether to apply the multiple-component transform.
    pub fn set_multiple_component_transform(&mut self, enabled: bool) -> &mut Self {
        self.raw.mct_mode = i32::from(enabled);
        self
    }

    /// Sets whether to use High-Throughput JPEG 2000 (HTJ2K) block coding.
    pub fn set_high_throughput(&mut self, enabled: bool) -> &mut Self {
        self.raw.ht = i32::from(enabled);
        self
    }
}

/// Encoder that can be reused across calls.
///
/// It keeps its [`Instance`] and CUDA context alive, and owns the CUDA stream it runs on.
#[derive(Debug)]
pub struct Encoder {
    raw: sys::nvimgcodecEncoder_t,
    instance: InstanceRef,
    execution: Box<ExecutionParams>,
    stream: Option<Stream>,
}

/// Native encoder handle with the values that keep it valid, from [`Encoder::into_raw`].
///
/// Destroy the handle before dropping this value, or pass it back to [`Encoder::from_raw`].
#[derive(Debug)]
pub struct RawEncoder {
    /// Native encoder handle, owned exclusively.
    pub handle: sys::nvimgcodecEncoder_t,
    /// Instance the handle was created with.
    pub instance: Instance,
    /// Execution parameters the handle was created with, at the address nvImageCodec was given.
    pub execution: Box<ExecutionParams>,
    /// Stream in the execution context, or `None` for CPU-only execution.
    pub stream: Option<Stream>,
}

/// Image to encode and the code stream to encode it into.
#[derive(Debug)]
pub struct EncodeItem<'a, 'image, 'stream> {
    /// Image to encode.
    ///
    /// Its pixels must be initialized.
    pub input: &'a mut Image<'image>,
    /// Code stream that receives the compressed data.
    ///
    /// It must have been created for encoding, such as with [`CodeStream::create_file_output`].
    pub stream: &'a mut CodeStream<'stream>,
}

#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "native owners transfer their handles and storage borrows under the adjacent safety contract"
)]
// SAFETY: nvImageCodec does not bind an encoder to its creating thread, and without `Sync` one thread uses it at a time; its stream, boxed execution parameters, and instance owner move with it.
unsafe impl Send for Encoder {}

impl<'a, 'image, 'stream> EncodeItem<'a, 'image, 'stream> {
    /// Pairs `input` with the code stream it encodes into.
    pub const fn new(input: &'a mut Image<'image>, stream: &'a mut CodeStream<'stream>) -> Self {
        Self { input, stream }
    }
}

impl Encoder {
    /// Creates an encoder for `instance` that runs as `execution_params` describes.
    ///
    /// `options` passes settings to the encoders as space-separated `name=value` pairs: `<encoder_id>:<name>=<value>` for one encoder, and `:<name>=<value>` for global options, such as `:num_cuda_streams=4`.
    /// nvImageCodec's documentation lists the options for each encoder.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, a CUDA stream cannot be created, `options` contains a NUL byte, or nvImageCodec fails to create the encoder.
    pub fn create(
        instance: &Instance,
        execution_params: &ExecutionParams,
        options: Option<&EncoderOptions>,
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
            .and_then(EncoderOptions::render)
            .map(CString::new)
            .transpose()?;
        let mut raw = ptr::null_mut();
        // SAFETY: `instance` is live, `execution` is boxed so the address nvImageCodec keeps outlives the encoder, `options` is a NUL-terminated string alive through the call, and `raw` is a writable local.
        unsafe {
            try_ffi!(sys::nvimgcodecEncoderCreate(
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

    /// Checks whether the encoder can encode each item with `params`, and returns a status for each.
    ///
    /// `fallback` selects whether the check stops at the preferred codec or also examines lower-priority codecs; see [`CodecFallback`].
    ///
    /// # Errors
    ///
    /// Returns an error if the batch is empty or larger than `i32::MAX`, a stream or image belongs to another instance, the CUDA context cannot be bound, or nvImageCodec fails the check.
    pub fn can_encode(
        &self,
        items: &[EncodeItem<'_, '_, '_>],
        params: &EncodeParams,
        fallback: CodecFallback,
    ) -> Result<Vec<ProcessingStatus>> {
        validate_batch(items.len(), items.len())?;
        for item in items {
            self.instance
                .require_same(item.stream.instance(), "stream")?;
            self.instance.require_same(item.input.instance(), "input")?;
        }

        let raw_images = collect_raw(items, |item| item.input.as_raw());
        let raw_streams = collect_raw(items, |item| item.stream.as_raw());
        self.check_encode(&raw_images, &raw_streams, params, fallback)
    }

    /// Checks whether the encoder can encode `input` into `stream` with `params`, and returns the status.
    ///
    /// See [`can_encode`](Self::can_encode) for `fallback`.
    ///
    /// # Errors
    ///
    /// Returns an error if [`can_encode`](Self::can_encode) would fail for this item.
    pub fn can_encode_one(
        &self,
        input: &Image<'_>,
        stream: &CodeStream<'_>,
        params: &EncodeParams,
        fallback: CodecFallback,
    ) -> Result<ProcessingStatus> {
        self.instance.require_same(stream.instance(), "stream")?;
        self.instance.require_same(input.instance(), "input")?;
        let statuses =
            self.check_encode(&[input.as_raw()], &[stream.as_raw()], params, fallback)?;
        Ok(statuses[0])
    }

    fn check_encode(
        &self,
        raw_images: &[sys::nvimgcodecImage_t],
        raw_streams: &[sys::nvimgcodecCodeStream_t],
        params: &EncodeParams,
        fallback: CodecFallback,
    ) -> Result<Vec<ProcessingStatus>> {
        if let Some(context) = self.context() {
            context.bind()?;
        }

        let batch_size = to_i32(raw_images.len(), "batch_size")?;
        let mut statuses = vec![0; raw_images.len()];
        // SAFETY: both handle arrays hold `batch_size` live handles of this encoder's instance, and `statuses` has room for one status per item.
        unsafe {
            try_ffi!(sys::nvimgcodecEncoderCanEncode(
                self.raw,
                raw_images.as_ptr(),
                raw_streams.as_ptr(),
                batch_size,
                params.as_raw(),
                statuses.as_mut_ptr(),
                fallback.force_format(),
            ))?;
        }

        Ok(statuses.into_iter().map(ProcessingStatus::from).collect())
    }

    fn prepare_encode(
        &self,
        items: &mut [EncodeItem<'_, '_, '_>],
    ) -> Result<(Vec<Stream>, RawBatch)> {
        validate_batch(items.len(), items.len())?;
        for item in items.iter() {
            self.instance
                .require_same(item.stream.instance(), "stream")?;
            item.stream.require_output()?;
        }

        let streams = {
            let mut images = items
                .iter_mut()
                .map(|item| &mut *item.input)
                .collect::<Vec<_>>();
            prepare_images(
                &self.instance,
                self.stream.as_ref(),
                &mut images,
                false,
                self.execution.raw.skip_pre_sync != 0,
            )?
        };
        let batch = RawBatch {
            code_streams: collect_raw(items, |item| item.stream.as_raw()),
            images: collect_raw(items, |item| item.input.as_raw()),
        };
        Ok((streams, batch))
    }

    /// Encodes each item with `params` and waits for the work.
    ///
    /// The call waits on the CPU and the GPU before returning, so the code streams hold the compressed data.
    ///
    /// # Errors
    ///
    /// Returns an error if the batch is empty or too large, a stream or image belongs to another instance or CUDA context, stream capture is active, a CUDA call or nvImageCodec fails, an image's processing status is not success, or a code stream was not created for encoding.
    pub fn encode(
        &self,
        items: &mut [EncodeItem<'_, '_, '_>],
        params: &EncodeParams,
    ) -> Result<()> {
        let (streams, batch) = self.prepare_encode(items)?;
        let statuses = complete_submission(
            streams,
            batch.len(),
            // SAFETY: the batch holds live handles of this encoder's instance, and `complete_submission` waits for the CPU and GPU work before the items are released.
            |future, batch_size| unsafe {
                sys::nvimgcodecEncoderEncode(
                    self.raw,
                    batch.images.as_ptr(),
                    batch.code_streams.as_ptr(),
                    batch_size,
                    params.as_raw(),
                    future,
                )
            },
        )?;
        validate_processing_statuses(&statuses)
    }

    /// Encodes `input` into `stream` with `params` and waits for the work.
    ///
    /// # Errors
    ///
    /// Returns an error if [`encode`](Self::encode) would fail for this item.
    pub fn encode_one(
        &self,
        input: &mut Image<'_>,
        stream: &mut CodeStream<'_>,
        params: &EncodeParams,
    ) -> Result<()> {
        self.encode(&mut [EncodeItem::new(input, stream)], params)
    }

    /// Submits encoding of each item with `params`, and returns a future to wait on.
    ///
    /// The images and streams are checked and prepared as in [`encode`](Self::encode), and work that produces device inputs is waited for before submission.
    ///
    /// # Safety
    ///
    /// The crate's raw execution contract applies: every image, code stream, buffer, and `params` must stay valid, and the code streams unused by anything else, until the returned future and the GPU work have finished, even if submission fails.
    /// Dropping the future does not wait for the GPU work.
    ///
    /// # Errors
    ///
    /// Returns an error if [`encode`](Self::encode) would fail before submitting, or nvImageCodec fails the submission.
    pub unsafe fn encode_async<'a>(
        &'a self,
        items: &mut [EncodeItem<'_, '_, '_>],
        params: &EncodeParams,
    ) -> Result<Future<'a>> {
        let (streams, batch) = self.prepare_encode(items)?;
        let submit = |batch: &RawBatch, future, batch_size| {
            // SAFETY: the batch holds `batch_size` live handles of this encoder's instance, and the future retains the handle arrays.
            unsafe {
                sys::nvimgcodecEncoderEncode(
                    self.raw,
                    batch.images.as_ptr(),
                    batch.code_streams.as_ptr(),
                    batch_size,
                    params.as_raw(),
                    future,
                )
            }
        };
        // SAFETY: the closure passes the batch's handles to one encode call, and the caller keeps everything the submission uses alive until it finishes.
        unsafe { submit_async(&self.instance, streams, batch, submit) }
    }

    /// Returns the native handle, which this encoder still owns.
    #[must_use]
    pub fn as_raw(&self) -> sys::nvimgcodecEncoder_t {
        self.raw
    }

    /// Takes ownership of a native encoder, from [`into_raw`](Self::into_raw) or created outside this crate.
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
    pub unsafe fn from_raw(raw: RawEncoder) -> Result<Self> {
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
    pub fn into_raw(self) -> RawEncoder {
        let this = ManuallyDrop::new(self);
        // SAFETY: `this` is never dropped, so each owning field moves out exactly once.
        let (instance, execution, stream) = unsafe {
            (
                ptr::read(&raw const this.instance),
                ptr::read(&raw const this.execution),
                ptr::read(&raw const this.stream),
            )
        };
        RawEncoder {
            handle: this.raw,
            instance: instance.into_instance(),
            execution,
            stream,
        }
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            if let Some(context) = self.execution.context()
                && let Err(error) = context.bind()
            {
                report_cleanup_failure(
                    "bind the cuda context to destroy",
                    "nvimagecodec encoder",
                    &error,
                );
                return;
            }

            // SAFETY: `self.raw` is the live encoder this value owns, its CUDA context is bound, and no `Future` that borrows the encoder remains.
            if let Err(error) = unsafe { try_ffi!(sys::nvimgcodecEncoderDestroy(self.raw)) } {
                report_cleanup_failure("destroy", "nvimagecodec encoder", &error);
            }
        }
    }
}

fn validate_jpeg2k_code_block_size(value: u32, name: &str) -> Result<()> {
    match value {
        32 | 64 => Ok(()),
        _ => Err(Error::Unsupported {
            name: name.to_string().into(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_params_clone_rebuilds_extension_chain() -> Result<()> {
        let mut jpeg = JpegEncodeParams::new();
        jpeg.set_optimized_huffman(true);
        let mut jpeg2k = Jpeg2kEncodeParams::new();
        jpeg2k.set_code_block_size(32, 64)?;

        let mut params = EncodeParams::new();
        params.set_jpeg(jpeg).set_jpeg2k(jpeg2k);
        let cloned = params.clone();

        assert!(!params.raw.struct_next.is_null());
        assert!(!cloned.raw.struct_next.is_null());
        assert_ne!(params.raw.struct_next, cloned.raw.struct_next);
        assert_eq!(
            cloned.raw.struct_next,
            cloned
                .jpeg
                .as_ref()
                .map(|jpeg| (&raw const **jpeg).cast_mut().cast())
                .ok_or_else(|| Error::Empty {
                    name: "jpeg".into(),
                })?
        );
        Ok(())
    }
}
