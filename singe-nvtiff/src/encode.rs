//! TIFF encoding from device memory.

use std::{
    ffi::c_void, fs, mem::ManuallyDrop, path::Path, process, ptr, result::Result as StdResult,
    slice, sync::Arc,
};

use singe_cuda::{
    context::Context,
    error::Error as CudaError,
    stream::{BorrowedStream, Stream as CudaStream},
    view::{DeviceInput, DeviceRegion, DeviceSlice},
};
use singe_cuda_sys::runtime::cudaStream_t;
use singe_nvtiff_sys as sys;

use crate::{
    allocator::AllocatorSet,
    error::{Error, Result},
    image::{EncodeImageInfo, ImageGeometry, ImageInfo},
    try_ffi,
    types::{GeoKey, GeoKeyDataType, OutputFormat, TagDataType, Variant},
    utility::{c_string, cleanup_stream, path_c_string, to_size_t, to_u32, to_usize},
};

/// TIFF encoder.
///
/// It keeps its CUDA context and stream alive, and creates and destroys its native state on that stream.
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

// Private cleanup owners retain context but not the stream wrapper, allowing
// capture cleanup to retain them without creating a stream reference cycle.
#[derive(Debug)]
struct EncoderOwner {
    raw: sys::nvtiffEncoder_t,
    stream: BorrowedStream,
}

// SAFETY: native mutation requires exclusive Encoder access; final cleanup binds its context.
#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "the borrowed cleanup stream is only used to destroy the native owner"
)]
unsafe impl Send for EncoderOwner {}

unsafe impl Sync for EncoderOwner {}

impl Drop for EncoderOwner {
    fn drop(&mut self) {
        if self.stream.context().bind().is_err() {
            return;
        }
        // SAFETY: `nvtiffEncoderDestroy` receives live handles owned by the wrappers, and no queued work uses them.
        if let Err(error) =
            unsafe { try_ffi!(sys::nvtiffEncoderDestroy(self.raw, self.stream.as_raw())) }
        {
            eprintln!(
                "failed to destroy nvtiff encoder: {}",
                singe_core::ErrorChain(&error)
            );
        }
    }
}

#[derive(Debug)]
struct EncodeParamsOwner {
    raw: sys::nvtiffEncodeParams_t,
    stream: BorrowedStream,
}

// SAFETY: mutation requires exclusive wrapper borrows; final cleanup binds its context.
#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "the borrowed cleanup stream is only used to destroy the native owner"
)]
unsafe impl Send for EncodeParamsOwner {}

unsafe impl Sync for EncodeParamsOwner {}

impl Drop for EncodeParamsOwner {
    fn drop(&mut self) {
        if self.stream.context().bind().is_err() {
            return;
        }
        // SAFETY: `nvtiffEncodeParamsDestroy` receives live handles owned by the wrappers, and no queued work uses them.
        if let Err(error) = unsafe {
            try_ffi!(sys::nvtiffEncodeParamsDestroy(
                self.raw,
                self.stream.as_raw()
            ))
        } {
            eprintln!(
                "failed to destroy nvtiff encode params: {}",
                singe_core::ErrorChain(&error)
            );
        }
    }
}

impl Encoder {
    /// Creates an encoder on `stream` with the default allocators.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active or invalidated, the CUDA context cannot be bound, nvTIFF fails to create the encoder, or waiting for `stream` fails.
    pub fn create(stream: &CudaStream) -> Result<Self> {
        stream.ensure_not_capturing()?;
        let mut raw = ptr::null_mut();
        stream.sync_scope(|_| {
            // SAFETY: `nvtiffEncoderCreate` receives live handles owned by the wrappers and valid output locations.
            Ok(unsafe {
                try_ffi!(sys::nvtiffEncoderCreate(
                    &raw mut raw,
                    ptr::null_mut(),
                    ptr::null_mut(),
                    stream.as_raw()
                ))
            })
        })??;
        Self::from_created_raw(raw, stream)
    }

    /// Creates an encoder on `stream` with custom allocators.
    ///
    /// # Safety
    ///
    /// The callbacks must allocate and free memory in stream order, as nvTIFF expects.
    /// The callbacks and their context data must stay valid until the encoder has been destroyed.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active or invalidated, the CUDA context cannot be bound, nvTIFF fails to create the encoder, or waiting for `stream` fails.
    pub unsafe fn create_with_allocators(
        mut allocators: AllocatorSet,
        stream: &CudaStream,
    ) -> Result<Self> {
        stream.ensure_not_capturing()?;
        let mut raw = ptr::null_mut();
        stream.sync_scope(|_| {
            // SAFETY: `nvtiffEncoderCreate` receives live handles owned by the wrappers and valid output locations.
            Ok(unsafe {
                try_ffi!(sys::nvtiffEncoderCreate(
                    &raw mut raw,
                    allocators.device_raw(),
                    allocators.pinned_raw(),
                    stream.as_raw()
                ))
            })
        })??;
        Self::from_created_raw(raw, stream)
    }

    fn from_created_raw(raw: sys::nvtiffEncoder_t, stream: &CudaStream) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            owner: Arc::new(EncoderOwner {
                raw,
                stream: cleanup_stream(stream),
            }),
            stream: stream.clone(),
        })
    }

    /// Takes ownership of a native encoder, which is destroyed on `stream`.
    ///
    /// # Safety
    ///
    /// `raw` must be valid, owned by nobody else, and created in `stream`'s CUDA context.
    /// All work submitted with it must have finished.
    /// Custom allocator data must stay valid until the encoder has been destroyed.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw` is null.
    pub unsafe fn from_raw(raw: sys::nvtiffEncoder_t, stream: &CudaStream) -> Result<Self> {
        Self::from_created_raw(raw, stream)
    }

    /// Releases the native handle without destroying it.
    ///
    /// After the submitted work finishes, destroy the handle on [`stream`](Self::stream) or pass it back to [`from_raw`](Self::from_raw).
    /// The handle does not keep the CUDA context or stream alive, so both must outlive it.
    ///
    /// # Errors
    ///
    /// Returns an error containing the encoder unchanged if something else still shares the native handle, such as cleanup for a captured graph.
    ///
    /// # Panics
    ///
    /// Panics if the handle becomes shared after the check, which indicates a bug in this crate.
    pub fn into_raw(self) -> StdResult<sys::nvtiffEncoder_t, Self> {
        if Arc::strong_count(&self.owner) != 1 {
            return Err(self);
        }

        let this = ManuallyDrop::new(self);
        // SAFETY: `this` is never dropped, so each field moves out exactly once.
        let (owner, stream) = unsafe {
            (
                ptr::read(&raw const this.owner),
                ptr::read(&raw const this.stream),
            )
        };
        drop(stream);
        let owner =
            ManuallyDrop::new(Arc::into_inner(owner).expect("the native owner is uniquely held"));
        // SAFETY: the owner is never dropped, so its stream moves out exactly once.
        drop(unsafe { ptr::read(&raw const owner.stream) });
        Ok(owner.raw)
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
        stream.sync_scope(|_| {
            Ok((|| -> Result<()> {
                stream.join_regions(inputs.regions())?;
                // SAFETY: initialized input extents and contexts were validated;
                // owners, descriptors, parameters, and pointer arrays remain borrowed.
                unsafe {
                    self.submit(sys::nvtiffEncode, &mut raw, &stream)?;
                    self.submit(sys::nvtiffEncodeFinalize, &mut raw, &stream)
                }
            })())
        })??;
        let size = self.raw_bitstream_size(&mut raw)?;
        let mut bytes = vec![0; size.total()?];
        let capacity = to_size_t(bytes.len(), "encoded buffer bytes")?;
        // Serialize exactly once. Native GeoTIFF serialization may mutate its
        // metadata; all later reads and writes use this independent snapshot.
        stream.sync_scope(|_| {
            // SAFETY: `nvtiffWriteTiffBuffer` receives live handles owned by the wrappers and valid output locations.
            Ok(unsafe {
                try_ffi!(sys::nvtiffWriteTiffBuffer(
                    self.as_raw(),
                    raw.as_mut_ptr(),
                    1,
                    bytes.as_mut_ptr(),
                    capacity,
                    stream.as_raw()
                ))
            })
        })??;
        Ok(EncodedTiff { bytes, size })
    }

    fn raw_params(&self, params: &[EncodeParams]) -> Result<Vec<sys::nvtiffEncodeParams_t>> {
        params
            .iter()
            .map(|params| {
                if params.context() == self.context() {
                    Ok(params.as_raw())
                } else {
                    Err(CudaError::MemoryContextMismatch.into())
                }
            })
            .collect()
    }

    /// Enqueues `operation` for `params` on `stream`.
    ///
    /// # Safety
    ///
    /// The contract of [`encode_async`](Self::encode_async) applies.
    unsafe fn submit(
        &mut self,
        operation: unsafe extern "C" fn(
            sys::nvtiffEncoder_t,
            *mut sys::nvtiffEncodeParams_t,
            u32,
            cudaStream_t,
        ) -> sys::nvtiffStatus_t,
        params: &mut [sys::nvtiffEncodeParams_t],
        stream: &CudaStream,
    ) -> Result<()> {
        if stream.context() != self.context() {
            return Err(Error::StreamContextMismatch);
        }

        let count = to_u32(params.len(), "params")?;
        self.context().bind()?;
        // SAFETY: `operation` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(operation(
                self.as_raw(),
                params.as_mut_ptr(),
                count,
                stream.as_raw()
            ))
        }
    }

    fn raw_bitstream_size(
        &mut self,
        params: &mut [sys::nvtiffEncodeParams_t],
    ) -> Result<BitstreamSize> {
        self.stream.ensure_not_capturing()?;
        let mut metadata = 0;
        let mut bitstream = 0;
        // SAFETY: `nvtiffGetBitstreamSize` receives live handles owned by the wrappers and valid output locations.
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

    /// Enqueues encoding of the images described by `params` on `stream`.
    ///
    /// Set the image information and inputs on each [`EncodeParams`] first.
    /// If compression, LZW by default, would make the data larger than the raw image, nvTIFF stores it uncompressed.
    /// Call [`finalize_async`](Self::finalize_async) next.
    ///
    /// # Safety
    ///
    /// `params` must be valid encode settings in this encoder's CUDA context, with initialized inputs of the right size whose writers are ordered before this work.
    /// The settings, inputs, and pointer arrays must stay alive and unchanged until the work finishes, even on failure, and for every replay of a captured graph.
    /// Each request must finish before the encoder is reused.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` or a value in `params` belongs to another CUDA context, there are more settings than fit a `u32`, the context cannot be bound, or nvTIFF rejects the request.
    pub unsafe fn encode_async(
        &mut self,
        params: &mut [EncodeParams],
        stream: &CudaStream,
    ) -> Result<()> {
        let mut raw = self.raw_params(params)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { self.submit(sys::nvtiffEncode, &mut raw, stream) }
    }

    /// Enqueues the end of the encoding started by [`encode_async`](Self::encode_async), on `stream`.
    ///
    /// Call it before [`write_buffer`](Self::write_buffer) or [`write_file`](Self::write_file).
    ///
    /// # Safety
    ///
    /// `params` must be the settings passed to [`encode_async`](Self::encode_async).
    /// They, their inputs and pointer arrays, and this encoder must stay alive until the work finishes, even on failure.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` or a value in `params` belongs to another CUDA context, there are more settings than fit a `u32`, the context cannot be bound, or nvTIFF rejects the request.
    pub unsafe fn finalize_async(
        &mut self,
        params: &mut [EncodeParams],
        stream: &CudaStream,
    ) -> Result<()> {
        let mut raw = self.raw_params(params)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { self.submit(sys::nvtiffEncodeFinalize, &mut raw, stream) }
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
    /// It waits for the stream before writing.
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
        let capacity = to_size_t(buffer.len(), "buffer")?;
        let num_params = to_u32(params.len(), "params")?;
        let size = self.raw_bitstream_size(&mut params)?.total()?;
        if buffer.len() < size {
            return Err(Error::BufferTooSmall {
                name: "encoded".into(),
                required: size,
                actual: buffer.len(),
            });
        }

        let stream = self.stream.clone();
        stream.sync_scope(|_| {
            // SAFETY: `nvtiffWriteTiffBuffer` receives live handles owned by the wrappers and valid output locations.
            Ok(unsafe {
                try_ffi!(sys::nvtiffWriteTiffBuffer(
                    self.as_raw(),
                    params.as_mut_ptr(),
                    num_params,
                    buffer.as_mut_ptr(),
                    capacity,
                    stream.as_raw()
                ))
            })
        })??;
        Ok(size)
    }

    /// Writes the whole TIFF file, metadata included, to `path`.
    ///
    /// It waits for the stream before writing.
    ///
    /// # Safety
    ///
    /// `params` must be valid encode settings in this encoder's CUDA context, finalized by this encoder, and must stay alive and unchanged until this returns.
    ///
    /// # Errors
    ///
    /// Returns an error if `path` contains a NUL byte, there are more settings than fit a `u32`, stream capture is active, the CUDA context cannot be bound, or nvTIFF fails to write the file or wait.
    pub unsafe fn write_file(
        &mut self,
        params: &mut [EncodeParams],
        path: impl AsRef<Path>,
    ) -> Result<()> {
        self.stream.ensure_not_capturing()?;
        let mut params = self.raw_params(params)?;
        let num_params = to_u32(params.len(), "params")?;
        let path = path_c_string(path)?;
        let stream = self.stream.clone();
        stream.sync_scope(|_| {
            // SAFETY: `nvtiffWriteTiffFile` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
            Ok(unsafe {
                try_ffi!(sys::nvtiffWriteTiffFile(
                    self.as_raw(),
                    params.as_mut_ptr(),
                    num_params,
                    path.as_ptr(),
                    stream.as_raw()
                ))
            })
        })??;
        Ok(())
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
        if self
            .stream
            .defer_drop_until_capture_end(Arc::clone(&self.owner))
            .is_err()
        {
            process::abort();
        }
    }
}

impl EncodeParams {
    /// Creates encode settings on `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active or invalidated, the CUDA context cannot be bound, nvTIFF fails to create the settings, or waiting for `stream` fails.
    pub fn create(stream: &CudaStream) -> Result<Self> {
        stream.ensure_not_capturing()?;
        let mut raw = ptr::null_mut();
        stream.sync_scope(|_| {
            // SAFETY: `nvtiffEncodeParamsCreate` receives valid output locations.
            Ok(unsafe { try_ffi!(sys::nvtiffEncodeParamsCreate(&raw mut raw)) })
        })??;
        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            owner: Arc::new(EncodeParamsOwner {
                raw,
                stream: cleanup_stream(stream),
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
        // SAFETY: `nvtiffEncodeParamsSetTiffVariant` receives live handles owned by the wrappers.
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
        // SAFETY: `nvtiffEncodeParamsSetImageInfo` receives live handles owned by the wrappers.
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
    pub fn set_inputs<'a, B: DeviceInput<u8> + ?Sized + 'a>(
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
                return Err(Error::LengthMismatch {
                    name: "encoder input bytes".into(),
                    expected: layout.size_bytes,
                    actual: input.len(),
                });
            }

            regions.push(input.region());
            pointers.push(input.as_device_ptr().cast_mut());
        }

        let mut inputs = EncodeInputs { pointers, regions };
        // SAFETY: this only installs checked pointer metadata. Raw submission
        // requires the caller to retain the guard and establish input readiness.
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
        // SAFETY: `nvtiffEncodeParamsSetImageGeometry` receives live handles owned by the wrappers.
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
        // SAFETY: `nvtiffEncodeParamsSetInputs` receives live handles owned by the wrappers and valid output locations.
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
        // SAFETY: `nvtiffEncodeParamsSetTag` receives live handles owned by the wrappers.
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
        // SAFETY: `nvtiffEncodeParamsSetGeoKey` receives live handles owned by the wrappers.
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
        // SAFETY: `nvtiffEncodeParamsSetGeoKeySHORT` receives live handles owned by the wrappers.
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
        // SAFETY: `nvtiffEncodeParamsSetGeoKeyDOUBLE` receives live handles owned by the wrappers.
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
        let value = c_string(value)?;
        self.context().bind()?;
        // SAFETY: `nvtiffEncodeParamsSetGeoKeyASCII` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
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
        if self
            .stream
            .defer_drop_until_capture_end(Arc::clone(&self.owner))
            .is_err()
        {
            process::abort();
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
