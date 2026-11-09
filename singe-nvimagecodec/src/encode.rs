use std::{ffi::CString, ptr, sync::Arc};

use singe_cuda::{context::Context as CudaContext, stream::Stream};
use singe_nvimagecodec_sys as sys;

use crate::{
    code_stream::CodeStream,
    codec::{ExecutionParams, complete_submission, prepare_images},
    error::{Error, Result},
    future::Future,
    image::Image,
    instance::{Instance, InstanceRef},
    options::EncoderOptions,
    try_ffi,
    types::{Jpeg2kBitstreamType, Jpeg2kProgressionOrder, ProcessingStatus, QualityType},
    utility::{
        NativeStruct, collect_raw, to_i32, validate_batch, validate_jpeg2k_code_block_size,
        validate_processing_statuses,
    },
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
    /// Native encoder handle.
    pub handle: sys::nvimgcodecEncoder_t,
    owner: Encoder,
}

// SAFETY: each transfer carries exclusive codec ownership and all retained execution resources.
unsafe impl Send for RawEncoder {}

/// Image to encode and the code stream to encode it into.
#[derive(Debug)]
pub struct EncodeItem<'a, 'image, 'stream> {
    /// Image to encode.
    ///
    /// Its pixels must be initialized.
    pub input: &'a mut Image<'image>,
    /// Code stream that receives the compressed data.
    ///
    /// It must have been created for encoding, such as with [`CodeStream::to_file`].
    pub stream: &'a mut CodeStream<'stream>,
}

/// Arguments for [`Encoder::encode_raw_async`].
#[derive(Debug, Clone, Copy)]
pub struct EncodeRawAsync<'a> {
    /// Array of `batch_size` native image handles.
    pub images: *const sys::nvimgcodecImage_t,
    /// Array of `batch_size` native code-stream handles.
    pub streams: *const sys::nvimgcodecCodeStream_t,
    /// Number of images and code streams.
    pub batch_size: usize,
    /// Encode settings.
    pub params: &'a EncodeParams,
}

// Moving a codec transfers exclusive ownership of its native workers and retained context.
#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "native owners transfer their handles and storage borrows under the adjacent safety contract"
)]
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
                context.create_stream()
            })
            .transpose()?;
        let options = options
            .and_then(EncoderOptions::render)
            .map(CString::new)
            .transpose()?;
        let mut raw = ptr::null_mut();
        // SAFETY: `nvimgcodecEncoderCreate` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
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

    /// Checks whether the encoder can encode each of `images` into the matching code stream of `streams` with `params`, and returns a status for each.
    ///
    /// With `force_format` set, nvImageCodec falls back to lower-priority codecs when the preferred one does not support the format, and checks them too.
    /// Without it, when the preferred codec can handle the input but not the requested format or parameters, the check stops there, and the status flags show what to change to avoid falling back.
    ///
    /// # Errors
    ///
    /// Returns an error if the batch is empty or larger than `i32::MAX`, the slices have different lengths, a stream or image belongs to another instance, the CUDA context cannot be bound, or nvImageCodec fails the check.
    pub fn can_encode(
        &self,
        images: &[&Image<'_>],
        streams: &[&CodeStream<'_>],
        params: &EncodeParams,
        force_format: bool,
    ) -> Result<Vec<ProcessingStatus>> {
        validate_batch(images.len(), streams.len())?;
        if let Some(context) = self.context() {
            context.bind()?;
        }

        for stream in streams {
            self.instance.require_same(stream.instance())?;
        }

        for image in images {
            self.instance.require_same(image.instance())?;
        }

        let batch_size = to_i32(images.len(), "batch_size")?;
        let raw_images = collect_raw(images, |image| image.as_raw());
        let raw_streams = collect_raw(streams, |stream| stream.as_raw());
        let mut statuses = vec![0; images.len()];

        // SAFETY: `nvimgcodecEncoderCanEncode` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::nvimgcodecEncoderCanEncode(
                self.raw,
                raw_images.as_ptr(),
                raw_streams.as_ptr(),
                batch_size,
                params.as_raw(),
                statuses.as_mut_ptr(),
                i32::from(force_format),
            ))?;
        }

        Ok(statuses.into_iter().map(ProcessingStatus::from).collect())
    }

    /// Checks whether the encoder can encode each item with `params`, and returns a status for each.
    ///
    /// See [`can_encode`](Self::can_encode) for `force_format`.
    ///
    /// # Errors
    ///
    /// Returns an error if the batch is empty or larger than `i32::MAX`, the slices have different lengths, a stream or image belongs to another instance, the CUDA context cannot be bound, or nvImageCodec fails the check.
    pub fn can_encode_batch(
        &self,
        items: &[EncodeItem<'_, '_, '_>],
        params: &EncodeParams,
        force_format: bool,
    ) -> Result<Vec<ProcessingStatus>> {
        validate_batch(items.len(), items.len())?;
        if let Some(context) = self.context() {
            context.bind()?;
        }

        for item in items {
            self.instance.require_same(item.stream.instance())?;
            self.instance.require_same(item.input.instance())?;
        }

        let batch_size = to_i32(items.len(), "batch_size")?;
        let raw_images = collect_raw(items, |item| item.input.as_raw());
        let raw_streams = collect_raw(items, |item| item.stream.as_raw());
        let mut statuses = vec![0; items.len()];

        // SAFETY: `nvimgcodecEncoderCanEncode` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::nvimgcodecEncoderCanEncode(
                self.raw,
                raw_images.as_ptr(),
                raw_streams.as_ptr(),
                batch_size,
                params.as_raw(),
                statuses.as_mut_ptr(),
                i32::from(force_format),
            ))?;
        }

        Ok(statuses.into_iter().map(ProcessingStatus::from).collect())
    }

    fn submit_encode(
        &mut self,
        images: &mut [&mut Image<'_>],
        streams: &[&mut CodeStream<'_>],
        params: &EncodeParams,
    ) -> Result<Vec<ProcessingStatus>> {
        validate_batch(images.len(), streams.len())?;
        for stream in streams {
            self.instance.require_same(stream.instance())?;
            stream.require_output()?;
        }

        let completion_streams = prepare_images(
            &self.instance,
            self.stream.as_ref(),
            images,
            false,
            self.execution.raw.skip_pre_sync != 0,
        )?;
        let raw_images = collect_raw(images, |image| image.as_raw());
        let raw_streams = collect_raw(streams, |stream| stream.as_raw());
        let statuses = complete_submission(
            completion_streams,
            images.len(),
            // SAFETY: `nvimgcodecEncoderEncode` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
            |future, batch_size| unsafe {
                sys::nvimgcodecEncoderEncode(
                    self.raw,
                    raw_images.as_ptr(),
                    raw_streams.as_ptr(),
                    batch_size,
                    params.as_raw(),
                    future,
                )
            },
        )?;
        validate_processing_statuses(&statuses)?;
        Ok(statuses)
    }

    /// Encodes each item with `params`, waits for the work, and returns a status for each.
    ///
    /// The call waits on the CPU and the GPU before returning, so the code streams hold the compressed data.
    ///
    /// # Errors
    ///
    /// Returns an error if the batch is empty or too large, a stream or image belongs to another instance or CUDA context, stream capture is active, a CUDA call or nvImageCodec fails, an image's processing status is not success, or a code stream was not created for encoding.
    pub fn encode(
        &mut self,
        items: &mut [EncodeItem<'_, '_, '_>],
        params: &EncodeParams,
    ) -> Result<Vec<ProcessingStatus>> {
        let (mut images, streams): (Vec<_>, Vec<_>) = items
            .iter_mut()
            .map(|item| (&mut *item.input, &mut *item.stream))
            .unzip();
        self.submit_encode(&mut images, &streams, params)
    }

    /// Encodes `input` into `stream` with `params`, waits for the work, and returns the processing status.
    ///
    /// # Errors
    ///
    /// Returns an error if [`encode`](Self::encode) would fail for this item.
    pub fn encode_one(
        &mut self,
        input: &mut Image<'_>,
        stream: &mut CodeStream<'_>,
        params: &EncodeParams,
    ) -> Result<ProcessingStatus> {
        let statuses = self.submit_encode(&mut [input], &[stream], params)?;
        Ok(statuses[0])
    }

    /// Submits encoding of each native image into the matching native code stream, and returns a future to wait on.
    ///
    /// # Safety
    ///
    /// `params.images` and `params.streams` must point to arrays of `params.batch_size` valid handles created with this encoder's instance and CUDA context.
    /// The handles, arrays, parameters, and all image and bitstream storage they use must stay valid until the returned future and the GPU work have finished, even if submission fails.
    /// The output code streams and their buffers must not be used by anything else until then.
    /// This crate skips nvImageCodec's own wait for input streams, so work that produces the inputs must finish before this call.
    /// Dropping the future does not wait for the GPU work.
    ///
    /// # Errors
    ///
    /// Returns an error if the batch is empty or larger than `i32::MAX`, a handle array is null, the CUDA context cannot be bound, or nvImageCodec fails the submission.
    pub unsafe fn encode_raw_async<'a>(&'a self, params: EncodeRawAsync<'_>) -> Result<Future<'a>> {
        if params.batch_size == 0 {
            return Err(Error::Empty {
                name: "batch".into(),
            });
        }

        if params.images.is_null() {
            return Err(Error::NullPointer {
                name: "images".into(),
            });
        }

        if params.streams.is_null() {
            return Err(Error::NullPointer {
                name: "streams".into(),
            });
        }

        let batch_size = to_i32(params.batch_size, "batch_size")?;
        let mut future = ptr::null_mut();
        if let Some(context) = self.context() {
            context.bind()?;
        }

        // SAFETY: `nvimgcodecEncoderEncode` receives live handles owned by the wrappers and valid output locations.
        let status = unsafe {
            sys::nvimgcodecEncoderEncode(
                self.raw,
                params.images,
                params.streams,
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

    /// Returns the native handle, which this encoder still owns.
    #[must_use]
    pub fn as_raw(&self) -> sys::nvimgcodecEncoder_t {
        self.raw
    }

    /// Takes ownership of a native encoder created outside this crate.
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
        raw: sys::nvimgcodecEncoder_t,
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
    pub unsafe fn from_raw(mut raw: RawEncoder) -> Self {
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
    pub fn into_raw(mut self) -> RawEncoder {
        let handle = self.raw;
        self.raw = ptr::null_mut();
        RawEncoder {
            handle,
            owner: self,
        }
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            if let Some(context) = self.execution.context() {
                let _ = context.bind();
            }
            // SAFETY: `nvimgcodecEncoderDestroy` receives live handles owned by the wrappers.
            unsafe {
                if let Err(error) = try_ffi!(sys::nvimgcodecEncoderDestroy(self.raw)) {
                    eprintln!(
                        "failed to free nvimagecodec encoder: {}",
                        singe_core::ErrorChain(&error)
                    );
                }
            }
        }
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
