//! TIFF encoding from device memory.

use std::{
    ffi::{CString, c_void},
    fs,
    mem::ManuallyDrop,
    path::Path,
    ptr,
    result::Result as StdResult,
    slice,
    sync::Arc,
};

use singe_core::{abort_after_cleanup_failure, report_cleanup_failure};
use singe_cuda::{
    context::Context,
    error::Error as CudaError,
    stream::{AsStream, BorrowedStream, Stream as CudaStream},
    view::{DeviceRange, DeviceRegion, DeviceSlice},
};
use singe_cuda_sys::runtime::cudaStream_t;
use singe_nvtiff_sys as sys;

use crate::{
    allocator::AllocatorSet,
    error::{Error, Result},
    image::{EncodeImageInfo, ImageGeometry, ImageInfo},
    try_ffi,
    types::{GeoKey, GeoKeyDataType, OutputFormat, TagDataType, Variant},
    utility::{to_size_t, to_u32, to_usize},
};

/// TIFF encoder.
///
/// It keeps its CUDA context and stream alive, and creates, runs, and destroys its native state on that stream.
#[derive(Debug)]
pub struct Encoder {
    owner: Arc<EncoderOwner>,
    stream: CudaStream,
}

/// Image information, inputs, and tags for encoding one image.
#[derive(Debug)]
pub struct EncodeParams {
    owner: Arc<EncodeParamsOwner>,
    stream: CudaStream,
    image_info: Option<EncodeImageInfo>,
}

/// Device inputs set on [`EncodeParams`], borrowed for `'a`.
#[derive(Debug)]
pub struct EncodeInputs<'a> {
    pointers: Vec<*mut u8>,
    regions: Vec<DeviceRegion<'a>>,
}

/// Sizes of the parts of an encoded TIFF.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BitstreamSize {
    /// Size of the TIFF metadata in bytes.
    pub metadata: usize,
    /// Size of the encoded image data in bytes.
    pub bitstream: usize,
}

/// Encoded TIFF file in host memory.
///
/// It uses no CUDA resources and can outlive its encoder and settings.
#[derive(Debug, Clone)]
pub struct EncodedTiff {
    bytes: Vec<u8>,
    size: BitstreamSize,
}

/// Native encoder handle with the stream it is destroyed on, from [`Encoder::into_raw`].
///
/// Dropping it leaks the native encoder.
#[derive(Debug)]
pub struct RawEncoder {
    /// Native encoder handle, owned exclusively.
    pub handle: sys::nvtiffEncoder_t,
    /// Stream that creates and destroys the encoder, retaining its CUDA context.
    pub stream: CudaStream,
}

// Private cleanup owners retain context but not the stream wrapper, allowing
// capture cleanup to retain them without creating a stream reference cycle.
#[derive(Debug)]
struct EncoderOwner {
    raw: sys::nvtiffEncoder_t,
    stream: BorrowedStream,
}

#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "the borrowed cleanup stream is only used to destroy the native owner"
)]
// SAFETY: an nvTIFF encoder has no thread affinity and is used through `&mut Encoder` only.
// The destroying thread binds the retained context first, and the stream wrapper or capture cleanup keeps the borrowed stream live.
unsafe impl Send for EncoderOwner {}

// SAFETY: shared references reach only the handle value; the native object is used through the owning wrapper's exclusive borrows and destroyed in `Drop`, which needs ownership.
unsafe impl Sync for EncoderOwner {}

impl Drop for EncoderOwner {
    fn drop(&mut self) {
        if let Err(error) = self.stream.context().bind() {
            report_cleanup_failure("bind the cuda context to destroy", "nvtiff encoder", &error);
            return;
        }

        // SAFETY: the last owner destroys the encoder once, with its context bound; the safe calls waited for their work, raw callers must finish theirs before dropping, and capture cleanup releases this owner only after capture ends.
        let destroyed =
            unsafe { try_ffi!(sys::nvtiffEncoderDestroy(self.raw, self.stream.as_raw())) };
        if let Err(error) = destroyed {
            report_cleanup_failure("destroy", "nvtiff encoder", &error);
        }
    }
}

#[derive(Debug)]
struct EncodeParamsOwner {
    raw: sys::nvtiffEncodeParams_t,
    stream: BorrowedStream,
}

#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "the borrowed cleanup stream is only used to destroy the native owner"
)]
// SAFETY: nvTIFF encode settings have no thread affinity and are changed through `&mut EncodeParams` only.
// The destroying thread binds the retained context first, and the stream wrapper or capture cleanup keeps the borrowed stream live.
unsafe impl Send for EncodeParamsOwner {}

// SAFETY: shared references reach only the handle value; the native object is used through the owning wrapper's exclusive borrows and destroyed in `Drop`, which needs ownership.
unsafe impl Sync for EncodeParamsOwner {}

impl Drop for EncodeParamsOwner {
    fn drop(&mut self) {
        if let Err(error) = self.stream.context().bind() {
            report_cleanup_failure(
                "bind the cuda context to destroy",
                "nvtiff encode params",
                &error,
            );
            return;
        }

        // SAFETY: the last owner destroys the settings once, with their context bound; the safe calls waited for their work, raw callers must finish theirs before dropping, and capture cleanup releases this owner only after capture ends.
        if let Err(error) = unsafe {
            try_ffi!(sys::nvtiffEncodeParamsDestroy(
                self.raw,
                self.stream.as_raw()
            ))
        } {
            report_cleanup_failure("destroy", "nvtiff encode params", &error);
        }
    }
}

impl Encoder {
    /// Creates an encoder on `stream` with `allocators`.
    ///
    /// [`AllocatorSet::new`] selects nvTIFF's default allocators.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active or invalidated, the CUDA context cannot be bound, nvTIFF fails to create the encoder, or waiting for `stream` fails.
    pub fn create(mut allocators: AllocatorSet, stream: &CudaStream) -> Result<Self> {
        let mut raw = ptr::null_mut();
        stream.complete(|| {
            stream.context().bind()?;
            // SAFETY: `nvtiffEncoderCreate` receives a live stream of the bound context, a writable output location, and allocators that are null for nvTIFF's defaults or come from unsafe constructors that guarantee callbacks following nvTIFF's contract and outliving the encoder.
            unsafe {
                try_ffi!(sys::nvtiffEncoderCreate(
                    &raw mut raw,
                    allocators.device_raw(),
                    allocators.pinned_raw(),
                    stream.as_raw()
                ))
            }
        })?;
        Self::from_created_raw(raw, stream)
    }

    fn from_created_raw(raw: sys::nvtiffEncoder_t, stream: &CudaStream) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            owner: Arc::new(EncoderOwner {
                raw,
                // SAFETY: the wrapper retains `stream` until this owner drops, directly or through the stream's capture cleanup.
                stream: unsafe { stream.to_unretained() },
            }),
            stream: stream.clone(),
        })
    }

    /// Takes ownership of a native encoder, which is destroyed on `raw.stream`.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be valid, owned by nobody else, and created in `raw.stream`'s CUDA context.
    /// All work submitted with it must have finished.
    /// Custom allocator data must stay valid until the encoder has been destroyed.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NullHandle`] if `raw.handle` is null.
    pub unsafe fn from_raw(raw: RawEncoder) -> Result<Self> {
        let RawEncoder { handle, stream } = raw;
        Self::from_created_raw(handle, &stream)
    }

    /// Releases the native handle with its stream, without destroying it.
    ///
    /// After the submitted work finishes, destroy the handle on `stream` or pass it back to [`from_raw`](Self::from_raw).
    ///
    /// # Errors
    ///
    /// Returns the encoder unchanged if something else still shares the native handle, such as cleanup for a captured graph.
    pub fn into_raw(self) -> StdResult<RawEncoder, Self> {
        let mut this = ManuallyDrop::new(self);
        // SAFETY: `this` is never dropped, so the owner moves out once, and the `Err` arm writes it back before `this` is used again.
        let owner = unsafe { ptr::read(&raw const this.owner) };
        match Arc::try_unwrap(owner) {
            Ok(owner) => {
                let owner = ManuallyDrop::new(owner);
                // SAFETY: `this` is never dropped, so its stream moves out exactly once.
                let stream = unsafe { ptr::read(&raw const this.stream) };
                // SAFETY: `owner` is never dropped, so its borrowed stream moves out exactly once.
                drop(unsafe { ptr::read(&raw const owner.stream) });
                Ok(RawEncoder {
                    handle: owner.raw,
                    stream,
                })
            }
            Err(owner) => {
                // SAFETY: `this.owner` was moved out above, so writing it back restores the only copy.
                unsafe {
                    ptr::write(&raw mut this.owner, owner);
                }
                Err(ManuallyDrop::into_inner(this))
            }
        }
    }

    /// Returns the native handle, which this encoder still owns.
    #[must_use]
    pub fn as_raw(&self) -> sys::nvtiffEncoder_t {
        self.owner.raw
    }

    /// Returns the CUDA context of the encoder.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.stream.context()
    }

    /// Returns the stream the encoder runs on.
    #[must_use]
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    /// Encodes `inputs` as one TIFF file in host memory, with the image information in `params`, and waits for the work.
    ///
    /// If compression would make the data larger than the raw image, nvTIFF stores it uncompressed.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, `params` or an input is in another CUDA context, the image information is missing or invalid, an input is too small, a size or count overflows, or encoding, finalizing, writing, or waiting for the stream fails.
    pub fn encode<'input, B: DeviceSlice<u8> + ?Sized + 'input>(
        &mut self,
        params: &mut EncodeParams,
        inputs: impl IntoIterator<Item = &'input B>,
    ) -> Result<EncodedTiff> {
        self.stream.ensure_not_capturing()?;
        let mut raw = self.raw_params(slice::from_ref(params))?;
        let inputs = params.set_inputs(inputs)?;
        let stream = self.stream.clone();
        stream.complete(|| {
            self.context().bind()?;
            self.stream.join_regions(inputs.regions())?;
            // SAFETY: the initialized inputs were checked against the image size and context, the work is ordered after their writers, and the settings, inputs, and pointer arrays outlive `complete`, which waits for the stream on every path.
            unsafe {
                self.submit(sys::nvtiffEncode, &mut raw)?;
                self.submit(sys::nvtiffEncodeFinalize, &mut raw)
            }
        })?;
        let size = self.raw_bitstream_size(&mut raw)?;
        let mut bytes = vec![0; size.total()?];
        // Serialize exactly once. Native GeoTIFF serialization may mutate its
        // metadata; all later reads and writes use this independent snapshot.
        self.write_raw_buffer(&mut raw, &mut bytes)?;
        Ok(EncodedTiff { bytes, size })
    }

    fn raw_params(&self, params: &[EncodeParams]) -> Result<Vec<sys::nvtiffEncodeParams_t>> {
        params
            .iter()
            .map(|params| {
                if params.context() == self.context() {
                    Ok(params.as_raw())
                } else {
                    Err(Error::ContextMismatch {
                        name: "encode params".into(),
                    })
                }
            })
            .collect()
    }

    // Enqueues `operation` for `params` on the encoder's stream.
    // Safety: the contract of `encode_async` applies.
    unsafe fn submit(
        &mut self,
        operation: unsafe extern "C" fn(
            sys::nvtiffEncoder_t,
            *mut sys::nvtiffEncodeParams_t,
            u32,
            cudaStream_t,
        ) -> sys::nvtiffStatus_t,
        params: &mut [sys::nvtiffEncodeParams_t],
    ) -> Result<()> {
        let count = to_u32(params.len(), "params")?;
        self.context().bind()?;
        // SAFETY: the encoder and `params` are live in the bound context, `params` holds `count` handles, and the caller keeps every participant alive until the work finishes.
        unsafe {
            try_ffi!(operation(
                self.as_raw(),
                params.as_mut_ptr(),
                count,
                self.stream.as_raw()
            ))
        }
    }

    // Writes the finalized TIFF for `params` into `buffer` and waits for the stream.
    fn write_raw_buffer(
        &mut self,
        params: &mut [sys::nvtiffEncodeParams_t],
        buffer: &mut [u8],
    ) -> Result<()> {
        let capacity = to_size_t(buffer.len(), "buffer")?;
        let num_params = to_u32(params.len(), "params")?;
        let stream = self.stream.clone();
        stream.complete(|| {
            self.context().bind()?;
            // SAFETY: the encoder and `params` are live in the bound context, and `buffer` is writable for `capacity` bytes until `complete` waits for the stream.
            unsafe {
                try_ffi!(sys::nvtiffWriteTiffBuffer(
                    self.as_raw(),
                    params.as_mut_ptr(),
                    num_params,
                    buffer.as_mut_ptr(),
                    capacity,
                    self.stream.as_raw()
                ))
            }
        })
    }

    fn raw_bitstream_size(
        &mut self,
        params: &mut [sys::nvtiffEncodeParams_t],
    ) -> Result<BitstreamSize> {
        self.stream.ensure_not_capturing()?;
        let mut metadata = 0;
        let mut bitstream = 0;
        self.context().bind()?;
        // SAFETY: the encoder and `params` are live in the bound context, and both sizes are writable output locations.
        unsafe {
            try_ffi!(sys::nvtiffGetBitstreamSize(
                self.as_raw(),
                params.as_mut_ptr(),
                to_u32(params.len(), "params")?,
                &raw mut metadata,
                &raw mut bitstream
            ))?;
        }
        Ok(BitstreamSize {
            metadata: to_usize(metadata, "metadata size")?,
            bitstream: to_usize(bitstream, "bitstream size")?,
        })
    }

    /// Enqueues encoding of the images described by `params` on the encoder's stream.
    ///
    /// Set the image information and inputs on each [`EncodeParams`] first, and pass the [`EncodeInputs`] returned for them as `inputs`; the work is ordered after the queued writers of their memory.
    /// If compression, LZW by default, would make the data larger than the raw image, nvTIFF stores it uncompressed.
    /// Call [`finalize_async`](Self::finalize_async) next.
    ///
    /// # Safety
    ///
    /// `params` must be valid encode settings in this encoder's CUDA context, with initialized inputs of the right size.
    /// The settings, inputs, and pointer arrays must stay alive and unchanged until the work finishes, even on failure, and for every replay of a captured graph.
    /// Each request must finish before the encoder is reused.
    ///
    /// # Errors
    ///
    /// Returns an error if a value in `params` or an input belongs to another CUDA context, there are more settings than fit a `u32`, the context cannot be bound, the ordering after the input writers cannot be enqueued, or nvTIFF rejects the request.
    pub unsafe fn encode_async(
        &mut self,
        params: &mut [EncodeParams],
        inputs: &[EncodeInputs<'_>],
    ) -> Result<()> {
        let mut raw = self.raw_params(params)?;
        let regions = inputs
            .iter()
            .flat_map(|inputs| inputs.regions().iter().copied())
            .collect::<Vec<_>>();
        self.context().bind()?;
        self.stream.join_regions(&regions)?;
        // SAFETY: `raw` holds the checked same-context handles of `params`, the work is ordered after the input writers, and the caller guarantees initialized, sized inputs that outlive the work.
        unsafe { self.submit(sys::nvtiffEncode, &mut raw) }
    }

    /// Enqueues the end of the encoding started by [`encode_async`](Self::encode_async), on the encoder's stream.
    ///
    /// Call it before [`write_buffer`](Self::write_buffer).
    ///
    /// # Safety
    ///
    /// `params` must be the settings passed to [`encode_async`](Self::encode_async).
    /// They, their inputs and pointer arrays, and this encoder must stay alive until the work finishes, even on failure.
    ///
    /// # Errors
    ///
    /// Returns an error if a value in `params` belongs to another CUDA context, there are more settings than fit a `u32`, the context cannot be bound, or nvTIFF rejects the request.
    pub unsafe fn finalize_async(&mut self, params: &mut [EncodeParams]) -> Result<()> {
        let mut raw = self.raw_params(params)?;
        // SAFETY: `raw` holds the checked same-context handles of `params`, which the caller guarantees were passed to `encode_async` and outlive the work with their inputs.
        unsafe { self.submit(sys::nvtiffEncodeFinalize, &mut raw) }
    }

    /// Returns the sizes of the TIFF metadata and encoded data for `params`.
    ///
    /// The metadata size is known before encoding, but the data size only after [`finalize_async`](Self::finalize_async).
    ///
    /// # Safety
    ///
    /// `params` must be valid encode settings in this encoder's CUDA context, and must have been finalized successfully by this encoder.
    /// Do not change them before using the sizes.
    ///
    /// # Errors
    ///
    /// Returns an error if a value in `params` belongs to another CUDA context, stream capture is active, the CUDA context cannot be bound, a count or size does not fit, or nvTIFF cannot report the sizes.
    pub unsafe fn bitstream_size(&mut self, params: &mut [EncodeParams]) -> Result<BitstreamSize> {
        let mut raw = self.raw_params(params)?;
        self.raw_bitstream_size(&mut raw)
    }

    /// Writes the whole TIFF file, metadata included, into `buffer` and returns the number of bytes written.
    ///
    /// It waits for the stream before returning.
    ///
    /// # Safety
    ///
    /// `params` must be valid encode settings in this encoder's CUDA context, finalized by this encoder, and must stay alive and unchanged until this returns.
    ///
    /// # Errors
    ///
    /// Returns an error if `buffer` is too small, a count or size does not fit, the sizes overflow when added, stream capture is active, the CUDA context cannot be bound, or nvTIFF fails to report the size, write, or wait.
    pub unsafe fn write_buffer(
        &mut self,
        params: &mut [EncodeParams],
        buffer: &mut [u8],
    ) -> Result<usize> {
        let mut params = self.raw_params(params)?;
        let size = self.raw_bitstream_size(&mut params)?.total()?;
        if buffer.len() < size {
            return Err(Error::BufferTooSmall {
                name: "encoded".into(),
                required: size,
                actual: buffer.len(),
            });
        }

        self.write_raw_buffer(&mut params, buffer)?;
        Ok(size)
    }
}

impl BitstreamSize {
    /// Returns the size of the whole TIFF file in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the sum overflows `usize`.
    pub fn total(self) -> Result<usize> {
        self.metadata
            .checked_add(self.bitstream)
            .ok_or_else(|| Error::OutOfRange {
                name: "encoded bitstream size".into(),
            })
    }
}

impl EncodedTiff {
    /// Returns the sizes of the metadata and encoded data.
    #[must_use]
    pub const fn size(&self) -> BitstreamSize {
        self.size
    }

    /// Returns the TIFF file's bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consumes the file and returns its bytes.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Copies the TIFF file to the start of `buffer` and returns the number of bytes written.
    ///
    /// # Errors
    ///
    /// Returns an error if `buffer` is too small.
    pub fn write_buffer(&self, buffer: &mut [u8]) -> Result<usize> {
        let len = self.bytes.len();
        if buffer.len() < len {
            return Err(Error::BufferTooSmall {
                name: "encoded".into(),
                required: len,
                actual: buffer.len(),
            });
        }

        buffer[..len].copy_from_slice(&self.bytes);
        Ok(len)
    }

    /// Writes the TIFF file to `path`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created or written.
    pub fn write_file(&self, path: impl AsRef<Path>) -> Result<()> {
        fs::write(path, &self.bytes)?;
        Ok(())
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        if let Err(error) = self
            .stream
            .defer_drop_until_capture_end(Arc::clone(&self.owner))
        {
            abort_after_cleanup_failure("retain until capture ends", "nvtiff encoder", &error);
        }
    }
}

impl EncodeParams {
    /// Creates encode settings on `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active or invalidated, the CUDA context cannot be bound, or nvTIFF fails to create the settings.
    pub fn create(stream: &CudaStream) -> Result<Self> {
        stream.ensure_not_capturing()?;
        stream.context().bind()?;
        let mut raw = ptr::null_mut();
        // SAFETY: `raw` is a writable handle location that nvTIFF fills on success.
        unsafe {
            try_ffi!(sys::nvtiffEncodeParamsCreate(&raw mut raw))?;
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            owner: Arc::new(EncodeParamsOwner {
                raw,
                // SAFETY: the wrapper retains `stream` until this owner drops, directly or through the stream's capture cleanup.
                stream: unsafe { stream.to_unretained() },
            }),
            stream: stream.clone(),
            image_info: None,
        })
    }

    /// Returns the native handle, which this value still owns.
    ///
    /// Changes made through it must keep the pixel layout set with [`set_image_info`](Self::set_image_info).
    #[must_use]
    pub fn as_raw(&self) -> sys::nvtiffEncodeParams_t {
        self.owner.raw
    }

    /// Returns the CUDA context of the settings.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.stream.context()
    }

    /// Returns the stream the settings were created on.
    #[must_use]
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    /// Sets whether to write a regular TIFF or a `BigTIFF`; regular by default.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvTIFF rejects the variant.
    pub fn set_tiff_variant(&mut self, variant: Variant) -> Result<&mut Self> {
        self.context().bind()?;
        // SAFETY: the settings handle is live and owned by this value, and `variant` converts to a valid native variant.
        unsafe {
            try_ffi!(sys::nvtiffEncodeParamsSetTiffVariant(
                self.as_raw(),
                variant.into(),
            ))?;
        }
        Ok(self)
    }

    /// Sets the image size, samples, and format to encode.
    ///
    /// The width, height, samples per pixel, bits per sample, and sample format are required, and nvTIFF accepts:
    ///
    /// - 1 or 3 samples per pixel;
    /// - [`PhotometricInterpretation::MinisBlack`](crate::types::PhotometricInterpretation::MinisBlack) for 1 sample and [`PhotometricInterpretation::Rgb`](crate::types::PhotometricInterpretation::Rgb) for 3, set automatically when [`PhotometricInterpretation::Unknown`](crate::types::PhotometricInterpretation::Unknown);
    /// - [`SampleFormat::Unknown`](crate::types::SampleFormat::Unknown), [`SampleFormat::Uint`](crate::types::SampleFormat::Uint), or [`SampleFormat::IeeeFloat`](crate::types::SampleFormat::IeeeFloat);
    /// - contiguous planar configuration;
    /// - [`Compression::Lzw`](crate::types::Compression::Lzw) or [`Compression::None`](crate::types::Compression::None).
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvTIFF rejects the information.
    pub fn set_image_info(&mut self, image_info: &EncodeImageInfo) -> Result<&mut Self> {
        self.context().bind()?;
        // Failed native configuration must not leave a stale layout proof.
        self.image_info = None;
        // SAFETY: the settings handle is live and owned by this value, and `image_info` points to a checked description borrowed for the call.
        unsafe {
            try_ffi!(sys::nvtiffEncodeParamsSetImageInfo(
                self.as_raw(),
                image_info.as_raw(),
            ))?;
        }

        self.image_info = Some(*image_info);
        Ok(self)
    }

    /// Sets the device memory of each image to encode.
    ///
    /// The returned value owns the pointer array given to nvTIFF and borrows the inputs.
    /// Keep it alive until raw encoding finishes; it does not mark the inputs as initialized, and [`Encoder::encode`] needs initialized inputs.
    ///
    /// # Errors
    ///
    /// Returns an error if the image information has not been set or is invalid, an input is in another CUDA context or smaller than its image, a size overflows, the CUDA context cannot be bound, there are more inputs than fit a `u32`, or nvTIFF rejects the inputs.
    pub fn set_inputs<'a, B: DeviceRange<u8> + ?Sized + 'a>(
        &mut self,
        inputs: impl IntoIterator<Item = &'a B>,
    ) -> Result<EncodeInputs<'a>> {
        self.context().bind()?;
        let info = self.image_info.ok_or_else(|| Error::OutOfRange {
            name: "encoder image metadata".into(),
        })?;
        let layout = ImageInfo::from_raw(*info.as_raw())
            .decode_layout(OutputFormat::UnchangedInterleaved)?;
        let mut regions = Vec::new();
        let mut pointers = Vec::new();
        for input in inputs {
            if input.context() != self.context() {
                return Err(CudaError::MemoryContextMismatch.into());
            }

            if input.len() < layout.size_bytes {
                return Err(Error::BufferTooSmall {
                    name: "encoder input bytes".into(),
                    required: layout.size_bytes,
                    actual: input.len(),
                });
            }

            regions.push(input.region());
            pointers.push(input.as_device_ptr().cast_mut());
        }

        let mut inputs = EncodeInputs { pointers, regions };
        // SAFETY: each pointer starts an input in this context with at least the image's bytes, and the returned `EncodeInputs` owns the pointer array and borrows the inputs for `'a`.
        // Encoding through these settings is ordered after the input writers by `Encoder::encode` or required of `encode_async` callers.
        unsafe {
            self.set_inputs_raw(&mut inputs.pointers)?;
        }
        Ok(inputs)
    }

    /// Sets the number of rows per strip; strips of 8 KB are used by default.
    ///
    /// Only strips, not tiles, are supported, so only the strip height matters.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvTIFF rejects the layout.
    pub fn set_image_geometry(&mut self, geometry: &ImageGeometry) -> Result<&mut Self> {
        self.context().bind()?;
        // SAFETY: the settings handle is live and owned by this value, and `geometry` points to a native layout borrowed for the call.
        unsafe {
            try_ffi!(sys::nvtiffEncodeParamsSetImageGeometry(
                self.as_raw(),
                geometry.as_raw(),
            ))?;
        }
        Ok(self)
    }

    /// Sets the device pointer of each image to encode.
    ///
    /// # Safety
    ///
    /// Each pointer in `inputs` must point to device-accessible memory large enough for the image information set on these settings, valid and unchanged until encoding and finalizing finish.
    /// The `inputs` array itself must stay alive and unchanged for as long.
    ///
    /// # Errors
    ///
    /// Returns an error if there are more images than fit a `u32`, the CUDA context cannot be bound, or nvTIFF rejects the pointers.
    pub unsafe fn set_inputs_raw(&mut self, inputs: &mut [*mut u8]) -> Result<&mut Self> {
        self.context().bind()?;
        // SAFETY: the settings handle is live and owned by this value, `inputs` holds the passed count of pointers, and the caller keeps the pointers and the array valid until the encoding finishes.
        unsafe {
            try_ffi!(sys::nvtiffEncodeParamsSetInputs(
                self.as_raw(),
                inputs.as_mut_ptr(),
                to_u32(inputs.len(), "inputs")?,
            ))?;
        }
        Ok(self)
    }

    /// Sets TIFF tag `tag` to `count` values of `data_type` read from `value`.
    ///
    /// # Safety
    ///
    /// `value` must point to `count` values of `data_type`, valid for as long as nvTIFF needs them to write the file.
    /// The tag must not change the pixel layout set with [`set_image_info`](Self::set_image_info).
    ///
    /// # Errors
    ///
    /// Returns an error if `count` does not fit a `u32`, the CUDA context cannot be bound, or nvTIFF rejects the tag, type, or value.
    pub unsafe fn set_tag_raw(
        &mut self,
        tag: u16,
        data_type: TagDataType,
        value: *mut c_void,
        count: impl TryInto<u32>,
    ) -> Result<&mut Self> {
        self.context().bind()?;
        // SAFETY: the settings handle is live and owned by this value, and the caller guarantees `value` holds `count` values of `data_type` for as long as nvTIFF reads them.
        unsafe {
            try_ffi!(sys::nvtiffEncodeParamsSetTag(
                self.as_raw(),
                tag,
                data_type.into(),
                value,
                to_u32(count, "count")?
            ))?;
        }
        Ok(self)
    }

    /// Sets `GeoTIFF` key `key` to `count` values of `value_size` bytes each, read from `value`.
    ///
    /// # Safety
    ///
    /// `value` must point to `count` values of `value_size` bytes, valid for as long as nvTIFF needs them to write the file.
    ///
    /// # Errors
    ///
    /// Returns an error if `value_size` or `count` does not fit a `u32`, the CUDA context cannot be bound, or nvTIFF rejects the key, type, or value.
    pub unsafe fn set_geo_key_raw(
        &mut self,
        key: GeoKey,
        data_type: GeoKeyDataType,
        value: *mut c_void,
        value_size: impl TryInto<u32>,
        count: impl TryInto<u32>,
    ) -> Result<&mut Self> {
        self.context().bind()?;
        // SAFETY: the settings handle is live and owned by this value, and the caller guarantees `value` holds `count` values of `value_size` bytes for as long as nvTIFF reads them.
        unsafe {
            try_ffi!(sys::nvtiffEncodeParamsSetGeoKey(
                self.as_raw(),
                key.into(),
                data_type.into(),
                value,
                to_u32(value_size, "value_size")?,
                to_u32(count, "count")?
            ))?;
        }
        Ok(self)
    }

    /// Sets `GeoTIFF` key `key` to a SHORT value.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvTIFF rejects the key or value.
    pub fn set_geo_key_short(&mut self, key: GeoKey, value: u16) -> Result<&mut Self> {
        self.context().bind()?;
        // SAFETY: the settings handle is live and owned by this value, and the value is passed by value with a count of 1.
        unsafe {
            try_ffi!(sys::nvtiffEncodeParamsSetGeoKeySHORT(
                self.as_raw(),
                key.into(),
                value,
                1
            ))?;
        }
        Ok(self)
    }

    /// Sets `GeoTIFF` key `key` to a DOUBLE value.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvTIFF rejects the key or value.
    pub fn set_geo_key_double(&mut self, key: GeoKey, value: f64) -> Result<&mut Self> {
        self.context().bind()?;
        // SAFETY: the settings handle is live and owned by this value, and the value is passed by value with a count of 1.
        unsafe {
            try_ffi!(sys::nvtiffEncodeParamsSetGeoKeyDOUBLE(
                self.as_raw(),
                key.into(),
                value,
                1
            ))?;
        }
        Ok(self)
    }

    /// Sets `GeoTIFF` key `key` to an ASCII string.
    ///
    /// # Errors
    ///
    /// Returns an error if `value` contains a NUL byte or is too long for nvTIFF, the CUDA context cannot be bound, or nvTIFF rejects the key or value.
    pub fn set_geo_key_ascii(&mut self, key: GeoKey, value: impl AsRef<str>) -> Result<&mut Self> {
        let value = CString::new(value.as_ref())?;
        self.context().bind()?;
        // SAFETY: the settings handle is live and owned by this value, and `value` is a NUL-terminated string whose size, terminator included, is passed with it.
        unsafe {
            try_ffi!(sys::nvtiffEncodeParamsSetGeoKeyASCII(
                self.as_raw(),
                key.into(),
                value.as_ptr(),
                to_u32(value.as_bytes_with_nul().len(), "value")?
            ))?;
        }
        Ok(self)
    }
}

impl Drop for EncodeParams {
    fn drop(&mut self) {
        if let Err(error) = self
            .stream
            .defer_drop_until_capture_end(Arc::clone(&self.owner))
        {
            abort_after_cleanup_failure(
                "retain until capture ends",
                "nvtiff encode params",
                &error,
            );
        }
    }
}

impl<'a> EncodeInputs<'a> {
    /// Returns the memory regions of the inputs.
    #[must_use]
    pub fn regions(&self) -> &[DeviceRegion<'a>] {
        &self.regions
    }

    /// Returns the pointer array given to nvTIFF.
    #[must_use]
    pub fn as_raw(&self) -> &[*mut u8] {
        &self.pointers
    }
}
