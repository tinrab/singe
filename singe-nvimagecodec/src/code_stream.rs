use std::{
    ffi::c_void,
    fmt::{self, Debug, Formatter},
    marker::PhantomData,
    path::Path,
    ptr,
};

use singe_core::{path_to_cstring, string_from_c_chars};
use singe_nvimagecodec_sys as sys;

use crate::{
    error::{Error, Result},
    image::ImageInfo,
    instance::{Instance, InstanceRef},
    try_ffi,
    types::OutOfBoundsPolicy,
    utility::{
        HostOutputBuffer, NativeStruct, resize_host_output_buffer, to_i32, to_u64, to_usize,
    },
};

/// Rectangular region of an image, with the fill used for pixels outside the image.
#[derive(Clone, Copy)]
pub struct Region {
    raw: sys::nvimgcodecRegion_t,
}

impl Debug for Region {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Region").finish_non_exhaustive()
    }
}

impl Region {
    /// Creates a region from `start` to `end`, one coordinate per dimension, with zero as the fill value.
    ///
    /// nvImageCodec checks the coordinates against the image and the codec when the region is used.
    ///
    /// # Errors
    ///
    /// Returns an error if the region has no dimensions, `start` and `end` have different lengths, or there are more dimensions than nvImageCodec supports.
    pub fn create(start: &[i32], end: &[i32]) -> Result<Self> {
        if start.is_empty() {
            return Err(Error::Empty {
                name: "region".into(),
            });
        }

        if start.len() != end.len() {
            return Err(Error::LengthMismatch {
                name: "region".into(),
                expected: start.len(),
                actual: end.len(),
            });
        }

        if start.len() > sys::NVIMGCODEC_MAX_NUM_DIM as usize {
            return Err(Error::LengthMismatch {
                name: "region".into(),
                expected: sys::NVIMGCODEC_MAX_NUM_DIM as usize,
                actual: start.len(),
            });
        }

        let mut raw = sys::nvimgcodecRegion_t::init();
        raw.ndim = to_i32(start.len(), "region.ndim")?;
        raw.start[..start.len()].copy_from_slice(start);
        raw.end[..end.len()].copy_from_slice(end);
        Ok(Self { raw })
    }

    /// Sets how pixels outside the image are filled.
    pub fn set_out_of_bounds_policy(&mut self, policy: OutOfBoundsPolicy) -> &mut Self {
        self.raw.out_of_bounds_policy = policy.into();
        self
    }

    /// Fills pixels outside the image with constant unsigned integer samples, one per channel.
    ///
    /// This selects [`OutOfBoundsPolicy::Constant`] and replaces only the first `samples.len()` fill values; the rest keep their earlier values, which start at zero.
    /// Channels past the last fill value reuse it.
    ///
    /// # Errors
    ///
    /// Returns an error if `samples` has more values than nvImageCodec supports.
    pub fn set_constant_fill_u32(&mut self, samples: &[u32]) -> Result<&mut Self> {
        self.set_constant_fill_samples(samples, |value| sys::nvimgcodecOutOfBoundsSample_t {
            value: sys::nvimgcodecOutOfBoundsSample_t__bindgen_ty_1 { as_uint: *value },
            type_: sys::nvimgcodecSampleDataType_t::NVIMGCODEC_SAMPLE_DATA_TYPE_UINT32,
        })
    }

    /// Fills pixels outside the image with constant signed integer samples, one per channel.
    ///
    /// This selects [`OutOfBoundsPolicy::Constant`] and replaces only the first `samples.len()` fill values; the rest keep their earlier values, which start at zero.
    /// Channels past the last fill value reuse it.
    ///
    /// # Errors
    ///
    /// Returns an error if `samples` has more values than nvImageCodec supports.
    pub fn set_constant_fill_i32(&mut self, samples: &[i32]) -> Result<&mut Self> {
        self.set_constant_fill_samples(samples, |value| sys::nvimgcodecOutOfBoundsSample_t {
            value: sys::nvimgcodecOutOfBoundsSample_t__bindgen_ty_1 { as_int: *value },
            type_: sys::nvimgcodecSampleDataType_t::NVIMGCODEC_SAMPLE_DATA_TYPE_INT32,
        })
    }

    /// Fills pixels outside the image with constant floating-point samples, one per channel.
    ///
    /// This selects [`OutOfBoundsPolicy::Constant`] and replaces only the first `samples.len()` fill values; the rest keep their earlier values, which start at zero.
    /// Channels past the last fill value reuse it.
    ///
    /// # Errors
    ///
    /// Returns an error if `samples` has more values than nvImageCodec supports.
    pub fn set_constant_fill_f32(&mut self, samples: &[f32]) -> Result<&mut Self> {
        self.set_constant_fill_samples(samples, |value| sys::nvimgcodecOutOfBoundsSample_t {
            value: sys::nvimgcodecOutOfBoundsSample_t__bindgen_ty_1 { as_float: *value },
            type_: sys::nvimgcodecSampleDataType_t::NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32,
        })
    }

    fn set_constant_fill_samples<T>(
        &mut self,
        samples: &[T],
        convert: impl Fn(&T) -> sys::nvimgcodecOutOfBoundsSample_t,
    ) -> Result<&mut Self> {
        if samples.len() > sys::NVIMGCODEC_MAX_ROI_FILL_CHANNELS as usize {
            return Err(Error::LengthMismatch {
                name: "out_of_bounds_samples".into(),
                expected: sys::NVIMGCODEC_MAX_ROI_FILL_CHANNELS as usize,
                actual: samples.len(),
            });
        }

        self.raw.out_of_bounds_policy = OutOfBoundsPolicy::Constant.into();
        for (index, sample) in samples.iter().enumerate() {
            self.raw.out_of_bounds_samples[index] = convert(sample);
        }
        Ok(self)
    }

    /// Returns a copy of the native region.
    #[must_use]
    pub const fn as_raw(self) -> sys::nvimgcodecRegion_t {
        self.raw
    }
}

/// Codec name, image count, and size of a code stream.
#[derive(Clone)]
pub struct CodeStreamInfo {
    raw: sys::nvimgcodecCodeStreamInfo_t,
}

impl Debug for CodeStreamInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("CodeStreamInfo").finish_non_exhaustive()
    }
}

impl CodeStreamInfo {
    pub(crate) const fn from_raw(raw: &sys::nvimgcodecCodeStreamInfo_t) -> Self {
        Self { raw: *raw }
    }

    /// Returns the name of the codec, such as `"jpeg"`.
    #[must_use]
    pub fn codec_name(&self) -> String {
        string_from_c_chars(&self.raw.codec_name)
    }

    /// Returns the number of images in the stream.
    #[must_use]
    pub const fn num_images(&self) -> u64 {
        self.raw.num_images
    }

    /// Returns the size of the compressed data in bytes.
    #[must_use]
    pub const fn size(&self) -> u64 {
        self.raw.size
    }
}

/// Image, region, and starting offset to read from a code stream.
pub struct CodeStreamView {
    raw: sys::nvimgcodecCodeStreamView_t,
}

impl Debug for CodeStreamView {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("CodeStreamView").finish_non_exhaustive()
    }
}

impl Default for CodeStreamView {
    fn default() -> Self {
        Self {
            raw: sys::nvimgcodecCodeStreamView_t::init(),
        }
    }
}

impl CodeStreamView {
    /// Creates a view of the first image, from the default offset, with no region.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Selects the image at `index`, counting from zero.
    ///
    /// The parser checks that the image exists.
    ///
    /// # Errors
    ///
    /// Returns an error if `index` does not fit a `u64`.
    pub fn set_image_index(&mut self, index: usize) -> Result<&mut Self> {
        self.raw.image_idx = to_u64(index, "index")?;
        Ok(self)
    }

    /// Sets the byte offset at which parsing starts, from the beginning of the file or buffer.
    ///
    /// Zero lets nvImageCodec choose the starting position.
    ///
    /// # Errors
    ///
    /// Returns an error if `offset` does not fit a `u64`.
    pub fn set_bitstream_offset(&mut self, offset: usize) -> Result<&mut Self> {
        self.raw.bitstream_offset = to_u64(offset, "offset")?;
        Ok(self)
    }

    /// Selects a region of the image.
    pub fn set_region(&mut self, region: Region) -> &mut Self {
        self.raw.region = region.as_raw();
        self
    }

    /// Returns a pointer to the native view, valid while `self` is.
    #[must_use]
    pub fn as_raw(&self) -> *const sys::nvimgcodecCodeStreamView_t {
        &raw const self.raw
    }
}

impl Clone for CodeStreamView {
    fn clone(&self) -> Self {
        let mut raw = sys::nvimgcodecCodeStreamView_t::init();
        raw.image_idx = self.raw.image_idx;
        raw.region = self.raw.region;
        raw.bitstream_offset = self.raw.bitstream_offset;
        Self { raw }
    }
}

/// Compressed image data to decode from, or a destination to encode into.
///
/// A code stream keeps its [`Instance`] alive.
/// One created from host memory borrows that memory, and one that encodes to host memory owns the output buffer.
#[derive(Debug)]
pub struct CodeStream<'a> {
    raw: sys::nvimgcodecCodeStream_t,
    instance: InstanceRef,
    host_output: Option<Box<HostOutputBuffer>>,
    writable: bool,
    _metadata: Option<Box<ImageInfo<'static>>>,
    _source: PhantomData<&'a [u8]>,
}

// SAFETY: moving a code stream moves its source borrow and the stable callback allocation.
// Safe encoding borrows the sink exclusively and completes callbacks before returning.
#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "native owners transfer their handles and storage borrows under the adjacent safety contract"
)]
unsafe impl Send for CodeStream<'_> {}

/// Native code-stream handle with the values that keep it valid, from [`CodeStream::into_raw`].
///
/// Destroy the handle before dropping this value, or pass it back to [`CodeStream::from_raw`].
#[derive(Debug)]
pub struct RawCodeStream<'a> {
    /// Native code-stream handle.
    pub handle: sys::nvimgcodecCodeStream_t,
    owner: CodeStream<'a>,
}

// SAFETY: the transfer carries the same source borrow and exclusive callback storage as CodeStream.
unsafe impl Send for RawCodeStream<'_> {}

impl<'a> CodeStream<'a> {
    pub(crate) fn instance(&self) -> &InstanceRef {
        &self.instance
    }

    pub(crate) fn require_output(&self) -> Result<()> {
        if self.writable {
            Ok(())
        } else {
            Err(Error::Unsupported {
                name: "read_only_code_stream".into(),
            })
        }
    }

    /// Opens the compressed image file at `path` for decoding.
    ///
    /// Only the view's bitstream offset applies when opening a stream; its image index and region are ignored.
    /// Use [`sub_stream`](Self::sub_stream) to select an image or region.
    ///
    /// # Errors
    ///
    /// Returns an error if `path` contains a NUL byte or nvImageCodec fails to create the stream.
    pub fn from_file(
        instance: &Instance,
        path: impl AsRef<Path>,
        view: Option<&CodeStreamView>,
    ) -> Result<CodeStream<'static>> {
        let path = path_to_cstring(path.as_ref())?;
        let mut raw = ptr::null_mut();
        // SAFETY: `nvimgcodecCodeStreamCreateFromFile` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::nvimgcodecCodeStreamCreateFromFile(
                instance.as_raw(),
                &raw mut raw,
                path.as_ptr(),
                view.map_or(ptr::null(), CodeStreamView::as_raw),
            ))?;
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(CodeStream {
            raw,
            instance: instance.as_instance_ref(),
            host_output: None,
            writable: false,
            _metadata: None,
            _source: PhantomData,
        })
    }

    /// Wraps the compressed image in `data` for decoding.
    ///
    /// Only the view's bitstream offset applies when opening a stream; its image index and region are ignored.
    /// Use [`sub_stream`](Self::sub_stream) to select an image or region.
    ///
    /// # Errors
    ///
    /// Returns an error if `data` is longer than nvImageCodec supports or nvImageCodec fails to create the stream.
    pub fn from_host_memory(
        instance: &Instance,
        data: &'a [u8],
        view: Option<&CodeStreamView>,
    ) -> Result<Self> {
        let mut raw = ptr::null_mut();
        // SAFETY: `nvimgcodecCodeStreamCreateFromHostMem` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::nvimgcodecCodeStreamCreateFromHostMem(
                instance.as_raw(),
                &raw mut raw,
                data.as_ptr(),
                to_u64(data.len(), "data")?,
                view.map_or(ptr::null(), CodeStreamView::as_raw),
            ))?;
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            raw,
            instance: instance.as_instance_ref(),
            host_output: None,
            writable: false,
            _metadata: None,
            _source: PhantomData,
        })
    }

    /// Creates a destination that encodes into the file at `path`, in the format and codec described by `image_info`.
    ///
    /// # Errors
    ///
    /// Returns an error if `path` contains a NUL byte or nvImageCodec fails to create the stream.
    pub fn to_file(
        instance: &Instance,
        path: impl AsRef<Path>,
        image_info: &ImageInfo<'_>,
    ) -> Result<CodeStream<'static>> {
        let path = path_to_cstring(path.as_ref())?;
        let metadata = Box::new(image_info.metadata());
        let mut raw = ptr::null_mut();
        // SAFETY: `nvimgcodecCodeStreamCreateToFile` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::nvimgcodecCodeStreamCreateToFile(
                instance.as_raw(),
                &raw mut raw,
                path.as_ptr(),
                metadata.as_raw(),
            ))?;
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(CodeStream {
            raw,
            instance: instance.as_instance_ref(),
            host_output: None,
            writable: true,
            _metadata: Some(metadata),
            _source: PhantomData,
        })
    }

    /// Creates a destination that encodes into a growable host buffer, in the format and codec described by `image_info`.
    ///
    /// Read the result with [`host_memory`](Self::host_memory) or [`into_host_memory`](Self::into_host_memory) after encoding.
    ///
    /// # Errors
    ///
    /// Returns an error if nvImageCodec fails to create the stream.
    pub fn to_host_memory(
        instance: &Instance,
        image_info: &ImageInfo<'_>,
    ) -> Result<CodeStream<'static>> {
        let mut output = Box::<HostOutputBuffer>::default();
        let metadata = Box::new(image_info.metadata());
        let mut raw = ptr::null_mut();
        // SAFETY: `nvimgcodecCodeStreamCreateToHostMem` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::nvimgcodecCodeStreamCreateToHostMem(
                instance.as_raw(),
                &raw mut raw,
                ptr::from_mut::<HostOutputBuffer>(output.as_mut()).cast::<c_void>(),
                Some(resize_host_output_buffer),
                metadata.as_raw(),
            ))?;
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(CodeStream {
            raw,
            instance: instance.as_instance_ref(),
            host_output: Some(output),
            writable: true,
            _metadata: Some(metadata),
            _source: PhantomData,
        })
    }

    /// Creates a code stream for the image, region, and offset selected by `view`.
    ///
    /// # Errors
    ///
    /// Returns an error if nvImageCodec cannot create the substream, for example because the image does not exist.
    pub fn sub_stream<'parent>(
        &'parent self,
        view: &CodeStreamView,
    ) -> Result<CodeStream<'parent>> {
        let mut raw = ptr::null_mut();
        // SAFETY: `nvimgcodecCodeStreamGetSubCodeStream` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::nvimgcodecCodeStreamGetSubCodeStream(
                self.raw,
                &raw mut raw,
                view.as_raw(),
            ))?;
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(CodeStream {
            raw,
            instance: self.instance.clone(),
            host_output: None,
            writable: false,
            _metadata: None,
            _source: PhantomData,
        })
    }

    /// Returns the codec name, image count, and size of the stream.
    ///
    /// # Errors
    ///
    /// Returns an error if nvImageCodec fails the query.
    pub fn info(&self) -> Result<CodeStreamInfo> {
        let mut info = sys::nvimgcodecCodeStreamInfo_t::init();
        // SAFETY: `nvimgcodecCodeStreamGetCodeStreamInfo` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::nvimgcodecCodeStreamGetCodeStreamInfo(
                self.raw,
                &raw mut info,
            ))?;
        }
        Ok(CodeStreamInfo::from_raw(&info))
    }

    /// Returns the format and size of the image in the stream.
    ///
    /// # Errors
    ///
    /// Returns an error if nvImageCodec cannot read the image information.
    pub fn image_info(&self) -> Result<ImageInfo<'static>> {
        let mut image_info = ImageInfo::empty();
        // SAFETY: `nvimgcodecCodeStreamGetImageInfo` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::nvimgcodecCodeStreamGetImageInfo(
                self.raw,
                image_info.as_raw_mut(),
            ))?;
        }
        Ok(image_info.metadata())
    }

    /// Returns the native handle, which this value still owns.
    #[must_use]
    pub fn as_raw(&self) -> sys::nvimgcodecCodeStream_t {
        self.raw
    }

    /// Returns the encoded bytes of a stream from [`to_host_memory`](Self::to_host_memory).
    ///
    /// Call it after encoding finishes.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream does not encode to host memory, the size query fails, or the reported size does not fit the buffer.
    pub fn host_memory(&self) -> Result<&[u8]> {
        let output = self
            .host_output
            .as_ref()
            .ok_or_else(|| Error::Unsupported {
                name: "host_output".into(),
            })?;
        let length = to_usize(self.info()?.size(), "bitstream_size")?;
        output
            .data
            .get(..length)
            .ok_or_else(|| Error::LengthMismatch {
                name: "bitstream_size".into(),
                expected: length,
                actual: output.data.len(),
            })
    }

    /// Destroys the stream and returns the encoded bytes of a stream from [`to_host_memory`](Self::to_host_memory).
    ///
    /// Call it after encoding finishes.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream does not encode to host memory, the size query fails, the reported size does not fit the buffer, or destroying the stream fails.
    ///
    /// # Panics
    ///
    /// Panics if the output buffer is missing after the checks passed, which indicates a bug in this crate.
    pub fn into_host_memory(mut self) -> Result<Vec<u8>> {
        let length = self.host_memory()?.len();
        self.destroy()?;
        let mut output = self
            .host_output
            .take()
            .expect("host memory was validated before destroying the code stream")
            .data;
        output.truncate(length);
        Ok(output)
    }

    /// Takes ownership of a native code stream created outside this crate.
    ///
    /// # Safety
    ///
    /// `raw` must be owned by nobody else and created with `instance`.
    /// The buffers, callbacks, and callback data it uses must stay valid for `'a`.
    /// `writable` must be `true` only for a stream that can be encoded into.
    /// All work that used `raw` must have finished.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw` is null.
    pub unsafe fn from_raw_parts(
        instance: &Instance,
        raw: sys::nvimgcodecCodeStream_t,
        writable: bool,
    ) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            raw,
            instance: instance.as_instance_ref(),
            host_output: None,
            writable,
            _metadata: None,
            _source: PhantomData,
        })
    }

    /// Takes back ownership of a handle from [`into_raw`](Self::into_raw).
    ///
    /// # Safety
    ///
    /// The handle must still be valid and owned by nobody else, with its metadata unchanged.
    /// All work that used it outside this crate must have finished.
    #[must_use]
    pub unsafe fn from_raw(mut raw: RawCodeStream<'a>) -> Self {
        raw.owner.raw = raw.handle;
        raw.owner
    }

    /// Releases the native handle with the values that keep it valid.
    ///
    /// Destroy the handle or pass it back to [`from_raw`](Self::from_raw).
    #[must_use]
    pub fn into_raw(mut self) -> RawCodeStream<'a> {
        let handle = self.raw;
        self.raw = ptr::null_mut();
        RawCodeStream {
            handle,
            owner: self,
        }
    }

    fn destroy(&mut self) -> Result<()> {
        if !self.raw.is_null() {
            // SAFETY: `nvimgcodecCodeStreamDestroy` receives live handles owned by the wrappers.
            unsafe {
                try_ffi!(sys::nvimgcodecCodeStreamDestroy(self.raw))?;
            }

            self.raw = ptr::null_mut();
        }
        Ok(())
    }
}

impl Drop for CodeStream<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.destroy() {
            eprintln!(
                "failed to free nvimagecodec code stream: {}",
                singe_core::ErrorChain(&error)
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_region_rejects_too_many_dimensions() {
        let start = [0; sys::NVIMGCODEC_MAX_NUM_DIM as usize + 1];
        let end = [1; sys::NVIMGCODEC_MAX_NUM_DIM as usize + 1];

        let result = Region::create(&start, &end);

        assert!(matches!(
            result,
            Err(Error::LengthMismatch { name, .. }) if name == "region"
        ));
    }
}
