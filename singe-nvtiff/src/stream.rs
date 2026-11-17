//! Parsed TIFF files: images, tags, and `GeoTIFF` keys.

use std::{ffi::c_void, marker::PhantomData, mem::MaybeUninit, path::Path, ptr};

use singe_core::{path_to_cstring, report_cleanup_failure};
use singe_nvtiff_sys as sys;

use crate::{
    error::{Error, Result},
    image::{IfdOffset, ImageGeometry, ImageInfo},
    try_ffi,
    types::{
        GeoKey, GeoKeyDataType, GeoKeyInfo, GeoKeyValue, TagDataType, TagInfo, TagValue, Variant,
    },
    utility::{bytes_to_values, to_size_t, to_u32, to_usize},
};

/// Parsed TIFF file.
///
/// nvTIFF reads the header when opening and each image directory when first used.
/// It is `Send` and not `Sync`.
#[derive(Debug)]
pub struct TiffStream<'data> {
    raw: sys::nvtiffStream_t,
    _data: PhantomData<&'data [u8]>,
}

// SAFETY: a parsed stream has no thread affinity, and the borrowed data is `Sync`; without `Sync`, calls on one stream never overlap.
unsafe impl Send for TiffStream<'_> {}

/// Header of a TIFF file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamHeader {
    /// Whether the file is a regular TIFF or a `BigTIFF`.
    pub variant: Variant,
    /// Offset of the first image's directory.
    pub first_ifd_offset: IfdOffset,
}

/// Image in a TIFF file.
#[derive(Debug, Clone, Copy)]
pub struct TiffImage<'stream, 'data> {
    stream: &'stream TiffStream<'data>,
    offset: IfdOffset,
}

/// Iterator over the images in a TIFF file.
#[derive(Debug)]
pub struct Images<'stream, 'data> {
    stream: &'stream TiffStream<'data>,
    next: Option<IfdOffset>,
}

impl TiffStream<'static> {
    /// Opens the TIFF file at `path` and reads its header.
    ///
    /// Image directories are parsed on first use, so a corrupt one is reported by the call that reads it.
    ///
    /// # Errors
    ///
    /// Returns an error if `path` contains a NUL byte, the file cannot be read, the header is invalid, or nvTIFF returns a null handle.
    pub fn open_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path_to_cstring(path.as_ref())?;
        let mut raw = ptr::null_mut();
        // SAFETY: `path` is a NUL-terminated string that outlives the call, and `raw` is a writable handle location.
        unsafe {
            try_ffi!(sys::nvtiffStreamOpenFromFile(path.as_ptr(), &raw mut raw))?;
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            raw,
            _data: PhantomData,
        })
    }
}

impl<'data> TiffStream<'data> {
    /// Opens a TIFF file in host memory and reads its header.
    ///
    /// Image directories and tags are read from `data` on first use, so a corrupt one is reported by the call that reads it.
    /// Decoding is faster from pinned memory.
    ///
    /// # Errors
    ///
    /// Returns an error if `data` is too long for nvTIFF, the header is invalid, or nvTIFF returns a null handle.
    pub fn open(data: &'data [u8]) -> Result<Self> {
        let mut raw = ptr::null_mut();
        // SAFETY: `data` holds the passed length and is borrowed for `'data`, which the returned stream carries, so nvTIFF's later reads of it stay valid; `raw` is a writable handle location.
        unsafe {
            try_ffi!(sys::nvtiffStreamOpen(
                data.as_ptr(),
                to_size_t(data.len(), "data")?,
                &raw mut raw,
            ))?;
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            raw,
            _data: PhantomData,
        })
    }

    /// Returns the file header.
    ///
    /// # Errors
    ///
    /// Returns an error if nvTIFF cannot read the header or the first offset does not fit `usize`.
    pub fn header(&self) -> Result<StreamHeader> {
        let mut header = MaybeUninit::<sys::nvtiffStreamHeader_t>::uninit();
        // SAFETY: `self.raw` is a live stream owned by this value, and `header` is writable for one header.
        unsafe {
            try_ffi!(sys::nvtiffStreamGetHeader(self.raw, header.as_mut_ptr()))?;
        }

        // SAFETY: successful header parsing writes both fields with valid values.
        let header = unsafe { header.assume_init() };
        Ok(StreamHeader {
            variant: header.variant.try_into()?,
            first_ifd_offset: IfdOffset::new(to_usize(
                header.first_ifd_offset,
                "first IFD offset",
            )?),
        })
    }

    /// Returns the directory offset of the image after the one at `offset`, or `None` for the last image.
    ///
    /// It reads only the directory headers, not the tags.
    /// Starting from a `SubIFD` offset walks that chain instead of the main one.
    ///
    /// # Errors
    ///
    /// Returns an error if nvTIFF cannot read the directory at `offset`, or the next offset does not fit `usize`.
    pub fn next_ifd_offset(&self, offset: IfdOffset) -> Result<Option<IfdOffset>> {
        let mut next = 0;
        // SAFETY: `self.raw` is a live stream owned by this value, and `next` is a writable offset.
        unsafe {
            try_ffi!(sys::nvtiffStreamGetNextIFDOffset(
                self.raw,
                to_size_t(offset.raw(), "ifd_offset")?,
                &raw mut next,
            ))?;
        }

        if next == sys::size_t::MAX {
            return Ok(None);
        }
        Ok(Some(IfdOffset::new(to_usize(next, "next IFD offset")?)))
    }

    /// Returns an iterator over the images of the main directory chain.
    ///
    /// # Errors
    ///
    /// Returns an error if [`header`](Self::header) fails or the first offset does not fit the native type.
    pub fn images(&self) -> Result<Images<'_, 'data>> {
        let first = self.header()?.first_ifd_offset;
        let is_last = to_size_t(first.raw(), "first IFD offset")? == sys::size_t::MAX;
        Ok(Images {
            stream: self,
            next: (!is_last).then_some(first),
        })
    }

    /// Returns information about the image at `offset`.
    ///
    /// # Errors
    ///
    /// Returns an error if `offset` is not an image directory or nvTIFF cannot read it.
    pub fn image_info(&self, offset: IfdOffset) -> Result<ImageInfo> {
        let mut image_info = sys::nvtiffImageInfo_t::default();
        // SAFETY: `self.raw` is a live stream owned by this value, and `image_info` is writable for one description.
        unsafe {
            try_ffi!(sys::nvtiffStreamGetImageInfo(
                self.raw,
                to_size_t(offset.raw(), "ifd_offset")?,
                &raw mut image_info,
            ))?;
        }
        Ok(ImageInfo::from_raw(image_info))
    }

    /// Returns the strip or tile layout of the image at `offset`.
    ///
    /// # Errors
    ///
    /// Returns an error if `offset` is not an image directory or nvTIFF cannot read it.
    pub fn image_geometry(&self, offset: IfdOffset) -> Result<ImageGeometry> {
        let mut geometry = sys::nvtiffImageGeometry_t::default();
        // SAFETY: `self.raw` is a live stream owned by this value, and `geometry` is writable for one layout.
        unsafe {
            try_ffi!(sys::nvtiffStreamGetImageGeometry(
                self.raw,
                to_size_t(offset.raw(), "ifd_offset")?,
                &raw mut geometry,
            ))?;
        }
        Ok(ImageGeometry::from_raw(geometry))
    }

    /// Returns the IDs of the tags in the image at `offset`.
    ///
    /// # Errors
    ///
    /// Returns an error if `offset` is not an image directory or nvTIFF cannot read its tags.
    pub fn tags(&self, offset: IfdOffset) -> Result<Vec<u16>> {
        let offset = to_size_t(offset.raw(), "ifd_offset")?;
        let mut num_tags = 0;
        // SAFETY: `self.raw` is a live stream owned by this value, a null tag array asks only for the count, and `num_tags` is writable.
        unsafe {
            try_ffi!(sys::nvtiffStreamGetNumberOfTags(
                self.raw,
                offset,
                ptr::null_mut(),
                &raw mut num_tags,
            ))?;
        }

        let mut tags = vec![0_u16; to_usize(num_tags, "tags")?];
        // SAFETY: `self.raw` is a live stream owned by this value, and `tags` has room for the `num_tags` IDs reported for this image.
        unsafe {
            try_ffi!(sys::nvtiffStreamGetNumberOfTags(
                self.raw,
                offset,
                tags.as_mut_ptr(),
                &raw mut num_tags,
            ))?;
        }

        tags.truncate(to_usize(num_tags, "tags")?);
        Ok(tags)
    }

    /// Returns whether the image at `offset` has `tag`, a [`Tag`](crate::types::Tag) or a tag ID.
    ///
    /// # Errors
    ///
    /// Returns an error if `offset` is not an image directory or nvTIFF cannot read its tags.
    pub fn has_tag(&self, offset: IfdOffset, tag: impl Into<u16>) -> Result<bool> {
        let tag = tag.into();
        let mut has_tag = 0;
        // SAFETY: `self.raw` is a live stream owned by this value, and `has_tag` is writable.
        unsafe {
            try_ffi!(sys::nvtiffStreamHasTag(
                self.raw,
                to_size_t(offset.raw(), "ifd_offset")?,
                tag,
                &raw mut has_tag,
            ))?;
        }
        Ok(has_tag != 0)
    }

    /// Returns the type and size of `tag`, a [`Tag`](crate::types::Tag) or a tag ID, in the image at `offset`.
    ///
    /// # Errors
    ///
    /// Returns an error if the tag is absent, `offset` is not an image directory, or nvTIFF cannot read the tag.
    pub fn tag_info(&self, offset: IfdOffset, tag: impl Into<u16>) -> Result<TagInfo> {
        let tag = tag.into();
        let mut data_type = sys::nvtiffTagDataType_t::NVTIFF_TAG_TYPE_UNDEFINED;
        let mut size = 0;
        let mut count = 0;
        // SAFETY: `self.raw` is a live stream owned by this value, and the type, size, and count are writable locals.
        unsafe {
            try_ffi!(sys::nvtiffStreamGetTagInfo(
                self.raw,
                to_size_t(offset.raw(), "ifd_offset")?,
                tag,
                &raw mut data_type,
                &raw mut size,
                &raw mut count,
            ))?;
        }
        Ok(TagInfo {
            data_type: data_type.try_into()?,
            size,
            count,
        })
    }

    /// Copies the value of `tag`, a [`Tag`](crate::types::Tag) or a tag ID, in the image at `offset` into `output`.
    ///
    /// `output` must hold at least `size * count` bytes, as returned by [`tag_info`](Self::tag_info).
    ///
    /// # Errors
    ///
    /// Returns an error if `output` is too small or nvTIFF cannot read the tag.
    pub fn tag_value_into(
        &self,
        offset: IfdOffset,
        tag: impl Into<u16>,
        output: &mut [u8],
    ) -> Result<()> {
        let tag = tag.into();
        let info = self.tag_info(offset, tag)?;
        let required = info.byte_len()?;
        if output.len() < required {
            return Err(Error::BufferTooSmall {
                name: "output".into(),
                required,
                actual: output.len(),
            });
        }
        // SAFETY: `self.raw` is a live stream owned by this value, and `output` was checked to hold the `size * count` bytes of `info.count` values.
        unsafe {
            try_ffi!(sys::nvtiffStreamGetTagValue(
                self.raw,
                to_size_t(offset.raw(), "ifd_offset")?,
                tag,
                output.as_mut_ptr().cast::<c_void>(),
                info.count,
            ))
        }
    }

    /// Returns the value of `tag`, a [`Tag`](crate::types::Tag) or a tag ID, in the image at `offset`.
    ///
    /// # Errors
    ///
    /// Returns an error if nvTIFF cannot read the tag, its size overflows `usize`, or its size is not a multiple of its element size.
    pub fn tag_value(&self, offset: IfdOffset, tag: impl Into<u16>) -> Result<TagValue> {
        let tag = tag.into();
        let info = self.tag_info(offset, tag)?;
        let mut bytes = vec![0_u8; info.byte_len()?];
        self.tag_value_into(offset, tag, &mut bytes)?;
        tag_value_from_bytes(info.data_type, bytes)
    }

    /// Returns the type and size of `GeoTIFF` key `key`.
    ///
    /// # Errors
    ///
    /// Returns an error if the key is absent or nvTIFF cannot read it.
    pub fn geo_key_info(&self, key: GeoKey) -> Result<GeoKeyInfo> {
        let mut size = 0;
        let mut count = 0;
        let mut data_type = sys::nvtiffGeoKeyDataType_t::NVTIFF_GEOKEY_TYPE_UNKNOWN;
        // SAFETY: `self.raw` is a live stream owned by this value, and the size, count, and type are writable locals.
        unsafe {
            try_ffi!(sys::nvtiffStreamGetGeoKeyInfo(
                self.raw,
                key.into(),
                &raw mut size,
                &raw mut count,
                &raw mut data_type
            ))?;
        }
        Ok(GeoKeyInfo {
            size,
            count,
            data_type: data_type.try_into()?,
        })
    }

    /// Returns the `GeoTIFF` keys in the file.
    ///
    /// # Errors
    ///
    /// Returns an error if nvTIFF cannot read the keys.
    pub fn geo_keys(&self) -> Result<Vec<GeoKey>> {
        let mut num_keys = 0;
        // SAFETY: `self.raw` is a live stream owned by this value, a null key array asks only for the count, and `num_keys` is writable.
        unsafe {
            try_ffi!(sys::nvtiffStreamGetNumberOfGeoKeys(
                self.raw,
                ptr::null_mut(),
                &raw mut num_keys
            ))?;
        }

        let mut keys =
            vec![sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_END; to_usize(num_keys, "geo keys")?];
        // SAFETY: `self.raw` is a live stream owned by this value, and `keys` has room for the `num_keys` keys reported for the file.
        unsafe {
            try_ffi!(sys::nvtiffStreamGetNumberOfGeoKeys(
                self.raw,
                keys.as_mut_ptr(),
                &raw mut num_keys
            ))?;
        }

        keys.truncate(to_usize(num_keys, "geo keys")?);
        keys.into_iter()
            .map(|key| Ok(GeoKey::try_from(key)?))
            .collect()
    }

    /// Copies the value of `GeoTIFF` key `key` into `output` and returns its type and size.
    ///
    /// # Errors
    ///
    /// Returns an error if `output` is too small or nvTIFF cannot read the key.
    pub fn geo_key_into(&self, key: GeoKey, output: &mut [u8]) -> Result<GeoKeyInfo> {
        let info = self.geo_key_info(key)?;
        let required = info.byte_len()?;
        if output.len() < required {
            return Err(Error::BufferTooSmall {
                name: "output".into(),
                required,
                actual: output.len(),
            });
        }

        // SAFETY: `self.raw` is a live stream owned by this value, and `output` was checked to hold the `size * count` bytes of `info.count` values.
        unsafe {
            try_ffi!(sys::nvtiffStreamGetGeoKey(
                self.raw,
                key.into(),
                output.as_mut_ptr().cast::<c_void>(),
                info.count
            ))?;
        }
        Ok(info)
    }

    /// Returns the value of `GeoTIFF` key `key`.
    ///
    /// # Errors
    ///
    /// Returns an error if nvTIFF cannot read the key.
    pub fn geo_key_value(&self, key: GeoKey) -> Result<GeoKeyValue> {
        let info = self.geo_key_info(key)?;
        match info.data_type {
            GeoKeyDataType::Short => {
                let mut values = vec![0_u16; to_usize(info.count, "geo key count")?];
                self.geo_key_short(key, 0, &mut values)?;
                Ok(GeoKeyValue::Short(values))
            }
            GeoKeyDataType::Double => {
                let mut values = vec![0_f64; to_usize(info.count, "geo key count")?];
                self.geo_key_double(key, 0, &mut values)?;
                Ok(GeoKeyValue::Double(values))
            }
            GeoKeyDataType::Ascii => Ok(GeoKeyValue::Ascii(self.geo_key_ascii(key, info.count)?)),
            GeoKeyDataType::Unknown => {
                let mut bytes = vec![0_u8; info.byte_len()?];
                self.geo_key_into(key, &mut bytes)?;
                Ok(GeoKeyValue::Unknown(bytes))
            }
        }
    }

    /// Returns the ASCII value of `GeoTIFF` key `key`, reading at most `max_len` bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if `max_len` does not fit a `u32`, or nvTIFF rejects the key, its type, or the length.
    pub fn geo_key_ascii(&self, key: GeoKey, max_len: impl TryInto<u32>) -> Result<String> {
        let max_len = to_u32(max_len, "max_len")?;
        let mut buffer = vec![0_u8; to_usize(max_len, "max_len")?];
        // SAFETY: `self.raw` is a live stream owned by this value, and `buffer` holds the `max_len` bytes passed as its size.
        unsafe {
            try_ffi!(sys::nvtiffStreamGetGeoKeyASCII(
                self.raw,
                key.into(),
                buffer.as_mut_ptr().cast(),
                max_len
            ))?;
        }

        let len = buffer
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(buffer.len());
        Ok(String::from_utf8_lossy(&buffer[..len]).into_owned())
    }

    /// Reads SHORT values of `GeoTIFF` key `key`, starting at element `index`, into `output`.
    ///
    /// # Errors
    ///
    /// Returns an error if the start or count does not fit a `u32`, or nvTIFF rejects the key, its type, or the range.
    pub fn geo_key_short(
        &self,
        key: GeoKey,
        index: impl TryInto<u32>,
        output: &mut [u16],
    ) -> Result<()> {
        // SAFETY: `self.raw` is a live stream owned by this value, and `output` holds the count of values passed with it.
        unsafe {
            try_ffi!(sys::nvtiffStreamGetGeoKeySHORT(
                self.raw,
                key.into(),
                output.as_mut_ptr(),
                to_u32(index, "index")?,
                to_u32(output.len(), "output")?,
            ))
        }
    }

    /// Reads DOUBLE values of `GeoTIFF` key `key`, starting at element `index`, into `output`.
    ///
    /// # Errors
    ///
    /// Returns an error if the start or count does not fit a `u32`, or nvTIFF rejects the key, its type, or the range.
    pub fn geo_key_double(
        &self,
        key: GeoKey,
        index: impl TryInto<u32>,
        output: &mut [f64],
    ) -> Result<()> {
        // SAFETY: `self.raw` is a live stream owned by this value, and `output` holds the count of values passed with it.
        unsafe {
            try_ffi!(sys::nvtiffStreamGetGeoKeyDOUBLE(
                self.raw,
                key.into(),
                output.as_mut_ptr(),
                to_u32(index, "index")?,
                to_u32(output.len(), "output")?,
            ))
        }
    }

    /// Returns the native handle, which this value still owns.
    #[must_use]
    pub const fn as_raw(&self) -> sys::nvtiffStream_t {
        self.raw
    }
}

impl Drop for TiffStream<'_> {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: `self.raw` is owned by this value and closed once; safe decodes have finished, and raw decode callers keep the stream alive until their work finishes.
            if let Err(error) = unsafe { try_ffi!(sys::nvtiffStreamClose(self.raw)) } {
                report_cleanup_failure("close", "nvtiff stream", &error);
            }
        }
    }
}

impl TiffImage<'_, '_> {
    /// Returns the offset of the image's directory.
    #[must_use]
    pub const fn ifd_offset(&self) -> IfdOffset {
        self.offset
    }

    /// Returns information about the image.
    ///
    /// # Errors
    ///
    /// Returns an error if [`TiffStream::image_info`] fails.
    pub fn info(&self) -> Result<ImageInfo> {
        self.stream.image_info(self.offset)
    }

    /// Returns the strip or tile layout of the image.
    ///
    /// # Errors
    ///
    /// Returns an error if [`TiffStream::image_geometry`] fails.
    pub fn geometry(&self) -> Result<ImageGeometry> {
        self.stream.image_geometry(self.offset)
    }

    /// Returns the IDs of the image's tags.
    ///
    /// # Errors
    ///
    /// Returns an error if [`TiffStream::tags`] fails.
    pub fn tags(&self) -> Result<Vec<u16>> {
        self.stream.tags(self.offset)
    }

    /// Returns the next image in the main directory chain, or `None` for the last image.
    ///
    /// # Errors
    ///
    /// Returns an error if [`TiffStream::next_ifd_offset`] fails.
    pub fn next(&self) -> Result<Option<Self>> {
        Ok(self
            .stream
            .next_ifd_offset(self.offset)?
            .map(|offset| TiffImage {
                stream: self.stream,
                offset,
            }))
    }
}

impl<'stream, 'data> Iterator for Images<'stream, 'data> {
    type Item = Result<TiffImage<'stream, 'data>>;

    fn next(&mut self) -> Option<Self::Item> {
        let offset = self.next?;
        match self.stream.next_ifd_offset(offset) {
            Ok(next) => {
                self.next = next;
                Some(Ok(TiffImage {
                    stream: self.stream,
                    offset,
                }))
            }
            Err(error) => {
                self.next = None;
                Some(Err(error))
            }
        }
    }
}

fn tag_value_from_bytes(data_type: TagDataType, bytes: Vec<u8>) -> Result<TagValue> {
    match data_type {
        TagDataType::Byte => Ok(TagValue::Byte(bytes)),
        TagDataType::Ascii => Ok(TagValue::Ascii(string_from_nul_terminated(&bytes))),
        TagDataType::Short => Ok(TagValue::Short(bytes_to_values(&bytes)?)),
        TagDataType::Long => Ok(TagValue::Long(bytes_to_values(&bytes)?)),
        TagDataType::Double => Ok(TagValue::Double(bytes_to_values(&bytes)?)),
        _ => Ok(TagValue::Raw { data_type, bytes }),
    }
}

fn string_from_nul_terminated(bytes: &[u8]) -> String {
    let len = bytes
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..len]).into_owned()
}
