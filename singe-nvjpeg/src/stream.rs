//! Parsing JPEG data and reading its header.

use std::{mem::ManuallyDrop, ptr, sync::Arc};

use singe_core::report_cleanup_failure;
use singe_cuda::context::Context as CudaContext;
use singe_nvjpeg_sys as sys;

use crate::{
    context::Context,
    error::{Error, Result},
    image::Size,
    try_ffi,
    types::{ChromaSubsampling, ExifOrientation, JpegEncoding},
    utility::{to_non_zero_u64, to_u64},
};

/// Parsed JPEG data, from which the header information can be read.
///
/// Parsing copies the data, so the stream does not borrow it.
#[derive(Debug)]
pub struct JpegStream {
    raw: sys::nvjpegJpegStream_t,
    pub(crate) handle: Context,
}

/// Native JPEG stream with its library handle, from [`JpegStream::into_raw`].
///
/// Dropping it leaks the native handle.
#[derive(Debug)]
pub struct RawJpegStream {
    /// Native handle, owned exclusively.
    pub handle: sys::nvjpegJpegStream_t,
    /// nvJPEG library handle the native handle was created with.
    pub context: Context,
}

/// One component of a parsed JPEG.
#[derive(Debug, Clone, Copy)]
pub struct JpegComponent<'a> {
    stream: &'a JpegStream,
    index: u32,
}

/// Header information read from a parsed JPEG.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct JpegStreamInfo {
    encoding: JpegEncoding,
    frame: Size,
    components: Vec<Size>,
    exif_orientation: ExifOrientation,
    sample_precision: u32,
    chroma_subsampling: ChromaSubsampling,
}

// SAFETY: nvJPEG requires a bitstream handle to be used by one thread at a time, not by a fixed thread, and every parse takes `&mut JpegStream`.
unsafe impl Send for JpegStream {}

impl JpegStreamInfo {
    /// Returns the JPEG encoding process.
    #[must_use]
    pub const fn encoding(&self) -> JpegEncoding {
        self.encoding
    }

    /// Returns the size of the whole image.
    #[must_use]
    pub const fn frame(&self) -> Size {
        self.frame
    }

    /// Returns the size of each component.
    #[must_use]
    pub fn components(&self) -> &[Size] {
        &self.components
    }

    /// Returns the EXIF orientation, or [`ExifOrientation::Unknown`] if the image has none.
    #[must_use]
    pub const fn exif_orientation(&self) -> ExifOrientation {
        self.exif_orientation
    }

    /// Returns the bits per sample.
    #[must_use]
    pub const fn sample_precision(&self) -> u32 {
        self.sample_precision
    }

    /// Returns the chroma subsampling.
    #[must_use]
    pub const fn chroma_subsampling(&self) -> ChromaSubsampling {
        self.chroma_subsampling
    }
}

impl JpegStream {
    /// Takes ownership of a native stream created with `raw.context`.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be valid, created with `raw.context`, and owned by nobody else.
    /// It is destroyed when the returned value drops, so all work that used it must have finished.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NullHandle`] if `raw.handle` is null.
    pub unsafe fn from_raw(raw: RawJpegStream) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            raw: raw.handle,
            handle: raw.context,
        })
    }

    /// Releases the native handle with its [`Context`], without destroying it.
    ///
    /// Destroy the handle, or pass it back to [`from_raw`](Self::from_raw), while the context is alive.
    #[must_use]
    pub fn into_raw(self) -> RawJpegStream {
        let this = ManuallyDrop::new(self);
        RawJpegStream {
            handle: this.raw,
            // SAFETY: `this` is never dropped, so its context moves out exactly once.
            context: unsafe { ptr::read(&raw const this.handle) },
        }
    }

    /// Returns the CUDA context of the stream.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        self.handle.cuda_context()
    }

    /// Creates an empty stream for `handle`.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG fails to create the stream.
    pub fn create(handle: &Context) -> Result<Self> {
        handle.bind()?;

        let mut raw = ptr::null_mut();
        // SAFETY: `handle` owns a live library handle whose context was bound above, and `raw` is a local that receives the stream.
        unsafe {
            try_ffi!(sys::nvjpegJpegStreamCreate(handle.as_raw(), &raw mut raw))?;
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            raw,
            handle: handle.clone(),
        })
    }

    /// Creates a stream and parses the JPEG in `data` into it.
    ///
    /// See [`parse`](Self::parse) for `save_metadata`.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream cannot be created or [`parse`](Self::parse) fails.
    pub fn create_from_bitstream(
        handle: &Context,
        data: &[u8],
        save_metadata: bool,
    ) -> Result<Self> {
        let mut stream = Self::create(handle)?;
        stream.parse(data, save_metadata)?;
        Ok(stream)
    }

    /// Creates a stream and parses only the header of the JPEG in `data` into it.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream cannot be created or [`parse_header`](Self::parse_header) fails.
    pub fn create_from_header(handle: &Context, data: &[u8]) -> Result<Self> {
        let mut stream = Self::create(handle)?;
        stream.parse_header(data)?;
        Ok(stream)
    }

    /// Creates a stream and parses external JPEG tables into it.
    ///
    /// See [`parse_tables`](Self::parse_tables).
    ///
    /// # Errors
    ///
    /// Returns an error if the stream cannot be created or [`parse_tables`](Self::parse_tables) fails.
    pub fn create_from_tables(handle: &Context, tables: Option<&[u8]>) -> Result<Self> {
        let mut stream = Self::create(handle)?;
        stream.parse_tables(tables)?;
        Ok(stream)
    }

    /// Parses the JPEG in `data`, replacing what the stream held.
    ///
    /// The stream copies `data`, so it can be dropped afterwards.
    /// `save_metadata` asks nvJPEG to keep the headers and APP markers; nvJPEG marks it for future use and does not act on it yet.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, `data` is empty or too long for nvJPEG, or nvJPEG cannot parse it.
    pub fn parse(&mut self, data: &[u8], save_metadata: bool) -> Result<()> {
        self.handle.bind()?;

        // Saving the stream makes nvJPEG copy `data`, so the stream never points into it after this call.
        let save_stream = true;
        // SAFETY: `&mut self` owns the live stream handle exclusively, and `data` is borrowed for the call with its checked nonzero length and copied by nvJPEG.
        unsafe {
            try_ffi!(sys::nvjpegJpegStreamParse(
                self.handle.as_raw(),
                data.as_ptr(),
                to_non_zero_u64(data.len(), "data length")?,
                i32::from(save_metadata),
                i32::from(save_stream),
                self.raw
            ))
        }
    }

    /// Parses only the header of the JPEG in `data`.
    ///
    /// This is enough to read the image information and to check support with [`Context::batched_decode_supported`].
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, `data` is empty or too long for nvJPEG, or nvJPEG cannot parse it.
    pub fn parse_header(&mut self, data: &[u8]) -> Result<()> {
        self.handle.bind()?;

        // SAFETY: `&mut self` owns the live stream handle exclusively, and `data` is borrowed for the synchronous call with its checked nonzero length.
        unsafe {
            try_ffi!(sys::nvjpegJpegStreamParseHeader(
                self.handle.as_raw(),
                data.as_ptr(),
                to_non_zero_u64(data.len(), "data length")?,
                self.raw
            ))
        }
    }

    /// Parses external Huffman and quantization tables, or clears them with `None`.
    ///
    /// Use it for JPEG data from TIFF files, whose tables are stored separately.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, the tables are too long for nvJPEG, or nvJPEG rejects them.
    pub fn parse_tables(&mut self, tables: Option<&[u8]>) -> Result<()> {
        self.handle.bind()?;

        let (ptr, len) = tables.map_or((ptr::null(), 0), |tables| (tables.as_ptr(), tables.len()));
        // SAFETY: `&mut self` owns the live stream handle exclusively, and `ptr` is null with length 0 or points to the tables borrowed for the synchronous call.
        unsafe {
            try_ffi!(sys::nvjpegJpegStreamParseTables(
                self.handle.as_raw(),
                ptr,
                to_u64(len, "tables length")?,
                self.raw
            ))
        }
    }

    /// Returns the JPEG encoding process, such as [`JpegEncoding::BaselineDct`] or [`JpegEncoding::ProgressiveDctHuffman`].
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG fails the query.
    pub fn encoding(&self) -> Result<JpegEncoding> {
        self.handle.bind()?;

        let mut encoding = sys::nvjpegJpegEncoding_t::NVJPEG_ENCODING_UNKNOWN;
        // SAFETY: `self` owns the live stream handle, and the outputs are locals that nvJPEG writes.
        unsafe {
            try_ffi!(sys::nvjpegJpegStreamGetJpegEncoding(
                self.raw,
                &raw mut encoding
            ))?;
        }
        Ok(encoding.try_into()?)
    }

    /// Returns the size of the whole image.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG fails the query.
    pub fn frame_dimensions(&self) -> Result<Size> {
        self.handle.bind()?;

        let mut width = 0;
        let mut height = 0;
        // SAFETY: `self` owns the live stream handle, and the outputs are locals that nvJPEG writes.
        unsafe {
            try_ffi!(sys::nvjpegJpegStreamGetFrameDimensions(
                self.raw,
                &raw mut width,
                &raw mut height
            ))?;
        }
        Ok(Size::new(width, height))
    }

    /// Returns the number of components.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG fails the query.
    pub fn component_count(&self) -> Result<u32> {
        self.handle.bind()?;

        let mut components = 0;
        // SAFETY: `self` owns the live stream handle, and the outputs are locals that nvJPEG writes.
        unsafe {
            try_ffi!(sys::nvjpegJpegStreamGetComponentsNum(
                self.raw,
                &raw mut components
            ))?;
        }
        Ok(components)
    }

    /// Returns an iterator over the components.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG fails the query.
    pub fn components(&self) -> Result<impl ExactSizeIterator<Item = JpegComponent<'_>> + '_> {
        let count = self.component_count()?;
        Ok((0..count).map(|index| JpegComponent {
            stream: self,
            index,
        }))
    }

    /// Returns the size of component `component`.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, `component` is out of range, or nvJPEG fails the query.
    pub fn component_dimensions(&self, component: u32) -> Result<Size> {
        self.handle.bind()?;

        let mut width = 0;
        let mut height = 0;
        // SAFETY: `self` owns the live stream handle, and the outputs are locals that nvJPEG writes.
        unsafe {
            try_ffi!(sys::nvjpegJpegStreamGetComponentDimensions(
                self.raw,
                component,
                &raw mut width,
                &raw mut height
            ))?;
        }
        Ok(Size::new(width, height))
    }

    /// Returns the size of every component.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG fails the query.
    pub fn all_component_dimensions(&self) -> Result<Vec<Size>> {
        self.components()?.map(JpegComponent::dimensions).collect()
    }

    /// Returns the EXIF orientation, or [`ExifOrientation::Unknown`] if the image has none.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG fails the query.
    pub fn exif_orientation(&self) -> Result<ExifOrientation> {
        self.handle.bind()?;

        let mut orientation = sys::nvjpegExifOrientation_t::NVJPEG_ORIENTATION_UNKNOWN;
        // SAFETY: `self` owns the live stream handle, and the outputs are locals that nvJPEG writes.
        unsafe {
            try_ffi!(sys::nvjpegJpegStreamGetExifOrientation(
                self.raw,
                &raw mut orientation
            ))?;
        }
        Ok(orientation.try_into()?)
    }

    /// Returns the bits per sample.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG fails the query.
    pub fn sample_precision(&self) -> Result<u32> {
        self.handle.bind()?;

        let mut precision = 0;
        // SAFETY: `self` owns the live stream handle, and the outputs are locals that nvJPEG writes.
        unsafe {
            try_ffi!(sys::nvjpegJpegStreamGetSamplePrecision(
                self.raw,
                &raw mut precision
            ))?;
        }
        Ok(precision)
    }

    /// Returns the chroma subsampling.
    ///
    /// Grayscale images report [`ChromaSubsampling::Gray`].
    /// Three-component images report the matching subsampling, or [`ChromaSubsampling::Unknown`] if none matches; two- and four-component images always report [`ChromaSubsampling::Unknown`].
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG fails the query.
    pub fn chroma_subsampling(&self) -> Result<ChromaSubsampling> {
        self.handle.bind()?;

        let mut subsampling = sys::nvjpegChromaSubsampling_t::NVJPEG_CSS_UNKNOWN;
        // SAFETY: `self` owns the live stream handle, and the outputs are locals that nvJPEG writes.
        unsafe {
            try_ffi!(sys::nvjpegJpegStreamGetChromaSubsampling(
                self.raw,
                &raw mut subsampling
            ))?;
        }
        Ok(subsampling.try_into()?)
    }

    /// Returns the header information.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG fails the query.
    pub fn info(&self) -> Result<JpegStreamInfo> {
        Ok(JpegStreamInfo {
            encoding: self.encoding()?,
            frame: self.frame_dimensions()?,
            components: self.all_component_dimensions()?,
            exif_orientation: self.exif_orientation()?,
            sample_precision: self.sample_precision()?,
            chroma_subsampling: self.chroma_subsampling()?,
        })
    }

    /// Returns the native handle, which this stream still owns.
    #[must_use]
    pub const fn as_raw(&self) -> sys::nvjpegJpegStream_t {
        self.raw
    }
}

impl JpegComponent<'_> {
    /// Returns the component's index.
    #[must_use]
    pub const fn index(self) -> u32 {
        self.index
    }

    /// Returns the component's size.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG fails the query.
    pub fn dimensions(self) -> Result<Size> {
        self.stream.component_dimensions(self.index)
    }
}

impl Drop for JpegStream {
    fn drop(&mut self) {
        if let Err(error) = self.handle.bind() {
            report_cleanup_failure(
                "bind the cuda context to destroy",
                "nvjpeg jpeg stream",
                &error,
            );
            return;
        }

        if !self.raw.is_null() {
            // SAFETY: the stream owns its handle exclusively and is destroyed once; safe decodes wait for their work, and raw callers keep the stream alive until theirs finishes.
            if let Err(error) = unsafe { try_ffi!(sys::nvjpegJpegStreamDestroy(self.raw)) } {
                report_cleanup_failure("destroy", "nvjpeg jpeg stream", &error);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimensions_reports_empty_axes() {
        let size = Size::new(4, 2);
        assert_eq!(size.width, 4);
        assert_eq!(size.height, 2);
        assert!(!size.is_empty());
        assert!(Size::new(0, 2).is_empty());
        assert!(Size::new(4, 0).is_empty());
    }
}
