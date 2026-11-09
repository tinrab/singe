use std::{fmt, marker::PhantomData, mem::MaybeUninit, ptr, slice};

use singe_core::string_from_c_chars;
use singe_cuda::{
    stream::Stream,
    view::{DeviceOutputView, DeviceRegion, DeviceSlice, DeviceSliceMut},
};
use singe_nvimagecodec_sys as sys;

use crate::{
    error::{Error, Result},
    instance::{Instance, InstanceRef},
    try_ffi,
    types::{
        ChromaSubsampling, ColorSpec, ImageBufferKind, JpegEncoding, SampleDataType, SampleFormat,
    },
    utility::{NativeStruct, to_u32, to_u64, to_usize, validate_host_buffer_len},
};

/// Rotation in quarter turns clockwise, and horizontal and vertical flips.
#[derive(Debug, Clone, Copy)]
pub struct Orientation {
    raw: sys::nvimgcodecOrientation_t,
}

/// Clockwise rotation in quarter turns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rotation {
    /// No rotation.
    Degrees0,
    /// 90 degrees clockwise.
    Degrees90,
    /// 180 degrees.
    Degrees180,
    /// 270 degrees clockwise.
    Degrees270,
}

impl Rotation {
    /// Returns the clockwise angle in degrees.
    #[must_use]
    pub const fn degrees_clockwise(self) -> i32 {
        match self {
            Self::Degrees0 => 0,
            Self::Degrees90 => 90,
            Self::Degrees180 => 180,
            Self::Degrees270 => 270,
        }
    }
}

impl Default for Orientation {
    fn default() -> Self {
        Self {
            raw: sys::nvimgcodecOrientation_t::init(),
        }
    }
}

impl Orientation {
    /// Creates an orientation with no rotation or flips.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the rotation.
    pub fn set_rotation(&mut self, rotation: Rotation) -> &mut Self {
        self.raw.rotated = rotation.degrees_clockwise();
        self
    }

    /// Sets the clockwise rotation in degrees, without checking it.
    ///
    /// nvImageCodec accepts only multiples of 90.
    pub fn set_rotation_degrees(&mut self, degrees_clockwise: i32) -> &mut Self {
        self.raw.rotated = degrees_clockwise;
        self
    }

    /// Sets whether to flip the image horizontally.
    pub fn set_flip_x(&mut self, flip: bool) -> &mut Self {
        self.raw.flip_x = i32::from(flip);
        self
    }

    /// Sets whether to flip the image vertically.
    pub fn set_flip_y(&mut self, flip: bool) -> &mut Self {
        self.raw.flip_y = i32::from(flip);
        self
    }

    /// Returns a copy of the native orientation.
    #[must_use]
    pub fn as_raw(self) -> sys::nvimgcodecOrientation_t {
        self.raw
    }
}

/// Size, row stride, channel count, and sample type of one image plane.
#[derive(Debug, Clone, Copy)]
pub struct ImagePlaneInfo {
    raw: sys::nvimgcodecImagePlaneInfo_t,
}

impl ImagePlaneInfo {
    /// Creates a plane description with a row stride in bytes.
    ///
    /// Nothing else is checked here; the image constructors check the plane against the buffer.
    /// The precision starts at the full bit depth of the sample type.
    ///
    /// # Errors
    ///
    /// Returns an error if `row_stride` does not fit a `u64`.
    pub fn create(
        width: u32,
        height: u32,
        row_stride: usize,
        num_channels: u32,
        sample_type: SampleDataType,
    ) -> Result<Self> {
        Ok(Self {
            raw: sys::nvimgcodecImagePlaneInfo_t {
                width,
                height,
                row_stride: to_u64(row_stride, "row_stride")?,
                num_channels,
                sample_type: sample_type.into(),
                ..sys::nvimgcodecImagePlaneInfo_t::init()
            },
        })
    }

    /// Sets the number of significant bits per sample, without checking it.
    ///
    /// Zero means the full bit depth of the sample type.
    pub fn set_precision(&mut self, precision: u8) -> &mut Self {
        self.raw.precision = precision;
        self
    }

    /// Returns the width in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.raw.width
    }

    /// Returns the height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.raw.height
    }

    /// Returns the distance between the starts of consecutive rows, in bytes.
    #[must_use]
    pub const fn row_stride(&self) -> u64 {
        self.raw.row_stride
    }

    /// Returns the number of channels per pixel in this plane.
    #[must_use]
    pub const fn num_channels(&self) -> u32 {
        self.raw.num_channels
    }

    /// Returns the type of each sample.
    #[must_use]
    pub fn sample_type(&self) -> SampleDataType {
        self.raw.sample_type.into()
    }

    /// Returns the number of significant bits per sample, or 0 for the full bit depth of the sample type.
    #[must_use]
    pub const fn precision(&self) -> u8 {
        self.raw.precision
    }

    /// Returns a copy of the native plane description.
    #[must_use]
    pub fn as_raw(self) -> sys::nvimgcodecImagePlaneInfo_t {
        self.raw
    }
}

/// How an image may use the memory that holds its pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleAccess {
    /// The pixels are initialized and read-only, for encoding.
    Input,
    /// The memory is writable and not yet initialized, for decoding.
    Output,
    /// The pixels are initialized and writable, for encoding or decoding.
    InputOutput,
}

/// Memory that holds an image's pixels, for [`ImageInfo::from_buffer`].
#[derive(Debug)]
pub enum ImageBuffer<'a> {
    /// Host memory.
    Host {
        /// Address of the first byte.
        pointer: *mut (),
        /// Size in bytes, including row padding.
        byte_len: usize,
    },
    /// Region of a device allocation.
    Device(DeviceRegion<'a>),
}

/// Pixel layout of an image, and the memory that holds its pixels.
///
/// The planes are stored one after another in the buffer.
/// Each takes `row_stride * height` bytes, including the padding after its last row.
/// Each plane's start and row stride must be multiples of its sample size.
/// Creating image information does not initialize any pixels or padding.
pub struct ImageInfo<'a> {
    pub(crate) raw: sys::nvimgcodecImageInfo_t,
    jpeg: Option<Box<sys::nvimgcodecJpegImageInfo_t>>,
    binding: Option<ImageBuffer<'a>>,
    access: SampleAccess,
    stream: Option<Stream>,
    _buffer: PhantomData<&'a mut [MaybeUninit<u8>]>,
}

impl fmt::Debug for ImageInfo<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImageInfo")
            .field("sample_format", &self.sample_format())
            .field("num_planes", &self.num_planes())
            .field("binding", &self.binding)
            .field("access", &self.access)
            .field("stream", &self.stream)
            .finish_non_exhaustive()
    }
}

impl ImageInfo<'static> {
    pub(crate) fn empty() -> Self {
        Self::from_raw(sys::nvimgcodecImageInfo_t::init())
    }
}

impl<'a> ImageInfo<'a> {
    pub(crate) const fn from_raw(raw: sys::nvimgcodecImageInfo_t) -> Self {
        Self {
            raw,
            jpeg: None,
            binding: None,
            access: SampleAccess::Output,
            stream: None,
            _buffer: PhantomData,
        }
    }

    /// Describes initialized, writable host pixels in `buffer`, for encoding or decoding.
    ///
    /// # Errors
    ///
    /// Returns an error if `planes` is empty or has more planes than nvImageCodec supports, a plane has a zero dimension or channel count, a sample type is not supported, a row stride is too small or misaligned, the buffer is misaligned or too small, or a size overflows.
    pub fn from_host_buffer(
        buffer: &'a mut [u8],
        sample_format: SampleFormat,
        planes: &[ImagePlaneInfo],
    ) -> Result<Self> {
        // SAFETY: the host buffer stays borrowed for the image lifetime and covers `byte_len` bytes.
        unsafe {
            Self::from_buffer(
                ImageBuffer::Host {
                    pointer: buffer.as_mut_ptr().cast(),
                    byte_len: buffer.len(),
                },
                SampleAccess::InputOutput,
                sample_format,
                planes,
            )
        }
    }

    /// Describes initialized, read-only host pixels in `buffer`, for encoding.
    ///
    /// # Errors
    ///
    /// Returns an error if `planes` is empty or has more planes than nvImageCodec supports, a plane has a zero dimension or channel count, a sample type is not supported, a row stride is too small or misaligned, the buffer is misaligned or too small, or a size overflows.
    pub fn from_host_input(
        buffer: &'a [u8],
        sample_format: SampleFormat,
        planes: &[ImagePlaneInfo],
    ) -> Result<Self> {
        // SAFETY: the host buffer stays borrowed for the image lifetime and covers `byte_len` bytes.
        unsafe {
            Self::from_buffer(
                ImageBuffer::Host {
                    pointer: buffer.as_ptr().cast_mut().cast(),
                    byte_len: buffer.len(),
                },
                SampleAccess::Input,
                sample_format,
                planes,
            )
        }
    }

    /// Describes uninitialized host memory in `buffer`, for decoding into.
    ///
    /// # Errors
    ///
    /// Returns an error if `planes` is empty or has more planes than nvImageCodec supports, a plane has a zero dimension or channel count, a sample type is not supported, a row stride is too small or misaligned, the buffer is misaligned or too small, or a size overflows.
    pub fn from_host_output(
        buffer: &'a mut [MaybeUninit<u8>],
        sample_format: SampleFormat,
        planes: &[ImagePlaneInfo],
    ) -> Result<Self> {
        // SAFETY: the host buffer stays borrowed for the image lifetime and covers `byte_len` bytes.
        unsafe {
            Self::from_buffer(
                ImageBuffer::Host {
                    pointer: buffer.as_mut_ptr().cast(),
                    byte_len: buffer.len(),
                },
                SampleAccess::Output,
                sample_format,
                planes,
            )
        }
    }

    /// Describes initialized, writable device pixels in `buffer`, for encoding or decoding.
    ///
    /// A CUDA stream is created in the buffer's CUDA context and kept with the image information.
    ///
    /// # Errors
    ///
    /// Returns an error if `planes` is empty or has more planes than nvImageCodec supports, a plane has a zero dimension or channel count, a sample type is not supported, a row stride is too small or misaligned, the buffer is misaligned or too small, a size overflows, or a CUDA stream cannot be created in the buffer's context.
    pub fn from_device_buffer_mut<B: DeviceSliceMut<u8> + ?Sized>(
        buffer: &'a mut B,
        sample_format: SampleFormat,
        planes: &[ImagePlaneInfo],
    ) -> Result<Self> {
        // SAFETY: the device region stays borrowed for the image lifetime with the access mode declared here.
        unsafe {
            Self::from_buffer(
                ImageBuffer::Device(buffer.region()),
                SampleAccess::InputOutput,
                sample_format,
                planes,
            )
        }
    }

    /// Describes initialized, read-only device pixels in `buffer`, for encoding.
    ///
    /// A CUDA stream is created in the buffer's CUDA context and kept with the image information.
    ///
    /// # Errors
    ///
    /// Returns an error if `planes` is empty or has more planes than nvImageCodec supports, a plane has a zero dimension or channel count, a sample type is not supported, a row stride is too small or misaligned, the buffer is misaligned or too small, a size overflows, or a CUDA stream cannot be created in the buffer's context.
    pub fn from_device_buffer<B: DeviceSlice<u8> + ?Sized>(
        buffer: &'a B,
        sample_format: SampleFormat,
        planes: &[ImagePlaneInfo],
    ) -> Result<Self> {
        // SAFETY: the device region stays borrowed for the image lifetime with the access mode declared here.
        unsafe {
            Self::from_buffer(
                ImageBuffer::Device(buffer.region()),
                SampleAccess::Input,
                sample_format,
                planes,
            )
        }
    }

    /// Describes uninitialized device memory in `buffer`, for decoding into.
    ///
    /// A successful decode initializes the pixels but not the row padding.
    ///
    /// A CUDA stream is created in the buffer's CUDA context and kept with the image information.
    ///
    /// # Errors
    ///
    /// Returns an error if `planes` is empty or has more planes than nvImageCodec supports, a plane has a zero dimension or channel count, a sample type is not supported, a row stride is too small or misaligned, the buffer is misaligned or too small, a size overflows, or a CUDA stream cannot be created in the buffer's context.
    pub fn from_device_output(
        buffer: DeviceOutputView<'a, u8>,
        sample_format: SampleFormat,
        planes: &[ImagePlaneInfo],
    ) -> Result<Self> {
        // SAFETY: the device region stays borrowed for the image lifetime with the access mode declared here.
        unsafe {
            Self::from_buffer(
                ImageBuffer::Device(buffer.into_region()),
                SampleAccess::Output,
                sample_format,
                planes,
            )
        }
    }

    /// Describes pixel memory that this crate cannot check, such as memory from another library.
    ///
    /// # Safety
    ///
    /// The whole buffer must stay valid for `'a`, in the CUDA context it reports.
    /// `access` must match what the memory holds and what the caller may do with it: initialized pixels for [`SampleAccess::Input`] and [`SampleAccess::InputOutput`], and exclusive access for writable memory.
    /// The pixels must match `sample_format` and `planes`.
    ///
    /// # Errors
    ///
    /// Returns an error if `planes` is empty or has more planes than nvImageCodec supports, a plane has a zero dimension or channel count, a sample type is not supported, a row stride is too small or misaligned, the buffer is misaligned or too small, a size overflows, a CUDA stream cannot be created for device memory, or the buffer's CUDA context does not match.
    pub unsafe fn from_buffer(
        buffer: ImageBuffer<'a>,
        access: SampleAccess,
        sample_format: SampleFormat,
        planes: &[ImagePlaneInfo],
    ) -> Result<Self> {
        let (pointer, byte_len, buffer_kind) = match &buffer {
            ImageBuffer::Host { pointer, byte_len } => {
                (*pointer, *byte_len, ImageBufferKind::StridedHost)
            }
            ImageBuffer::Device(region) => (
                region.as_ptr().cast(),
                region.byte_len(),
                ImageBufferKind::StridedDevice,
            ),
        };
        if pointer.is_null() {
            return Err(Error::NullPointer {
                name: "buffer".into(),
            });
        }

        validate_host_buffer_len(byte_len, planes)?;
        let mut offset = 0_usize;
        for plane in planes {
            let alignment = plane
                .sample_type()
                .byte_size()
                .ok_or_else(|| Error::Unsupported {
                    name: "sample_type".into(),
                })?;
            if !(pointer as usize)
                .wrapping_add(offset)
                .is_multiple_of(alignment)
                || !plane.row_stride().is_multiple_of(alignment as u64)
            {
                return Err(Error::out_of_range("sample_alignment"));
            }

            offset = offset
                .checked_add(
                    (to_usize(plane.row_stride(), "plane row stride")?)
                        .checked_mul(plane.height() as usize)
                        .ok_or_else(|| Error::out_of_range("plane_size"))?,
                )
                .ok_or_else(|| Error::out_of_range("buffer"))?;
        }

        let mut info = Self::from_raw(sys::nvimgcodecImageInfo_t::init());
        info.raw.buffer = pointer.cast();
        info.raw.buffer_kind = buffer_kind.into();
        info.raw.sample_format = sample_format.into();
        info.raw.num_planes = to_u32(planes.len(), "plane count")?;
        for (index, plane) in planes.iter().enumerate() {
            info.raw.plane_info[index] = plane.as_raw();
        }

        if let ImageBuffer::Device(region) = &buffer {
            let stream = region.context().create_stream()?;
            info.set_stream(&stream)?;
        }

        info.binding = Some(buffer);
        info.access = access;
        Ok(info)
    }

    /// Returns a copy of the layout and codec information, without the pixel memory.
    #[must_use]
    pub fn metadata(&self) -> ImageInfo<'static> {
        let mut info = ImageInfo::from_raw(sys::nvimgcodecImageInfo_t {
            codec_name: self.raw.codec_name,
            color_spec: self.raw.color_spec,
            chroma_subsampling: self.raw.chroma_subsampling,
            sample_format: self.raw.sample_format,
            orientation: self.raw.orientation,
            num_planes: self.raw.num_planes,
            plane_info: self.raw.plane_info,
            buffer_kind: self.raw.buffer_kind,
            ..sys::nvimgcodecImageInfo_t::init()
        });
        for plane in &mut info.raw.plane_info {
            plane.struct_next = ptr::null_mut();
        }

        info.raw.orientation.struct_next = ptr::null_mut();
        info.raw.buffer = ptr::null_mut();
        info.raw.cuda_stream = ptr::null_mut();
        info.jpeg.clone_from(&self.jpeg);
        info.rebuild_extension_chain();
        info
    }

    /// Sets how the codec interprets the color channels.
    pub fn set_color_spec(&mut self, color_spec: ColorSpec) -> &mut Self {
        self.raw.color_spec = color_spec.into();
        self
    }

    /// Sets the chroma subsampling.
    pub fn set_chroma_subsampling(&mut self, chroma_subsampling: ChromaSubsampling) -> &mut Self {
        self.raw.chroma_subsampling = chroma_subsampling.into();
        self
    }

    /// Sets the rotation and flips.
    pub fn set_orientation(&mut self, orientation: Orientation) -> &mut Self {
        self.raw.orientation = orientation.as_raw();
        self
    }

    /// Sets the CUDA stream that GPU work on the image runs on.
    ///
    /// # Errors
    ///
    /// Returns an error if the image is in device memory of a different CUDA context than `stream`.
    pub fn set_stream(&mut self, stream: &Stream) -> Result<&mut Self> {
        if let Some(ImageBuffer::Device(region)) = &self.binding
            && stream.context() != region.context()
        {
            return Err(Error::StreamContextMismatch);
        }

        self.raw.cuda_stream = stream.as_raw();
        self.stream = Some(stream.clone());
        Ok(self)
    }

    /// Sets the JPEG-specific information.
    pub fn set_jpeg(&mut self, jpeg: JpegImageInfo) -> &mut Self {
        self.jpeg = Some(Box::new(jpeg.raw));
        self.rebuild_extension_chain();
        self
    }

    /// Returns a pointer to the native image information, valid while `self` is.
    #[must_use]
    pub fn as_raw(&self) -> *const sys::nvimgcodecImageInfo_t {
        &raw const self.raw
    }

    /// Returns a mutable pointer to the native image information.
    ///
    /// Writes through it must keep the buffer, access, CUDA context, plane sizes, and extension pointers consistent before safe methods use this value again.
    pub fn as_raw_mut(&mut self) -> *mut sys::nvimgcodecImageInfo_t {
        &raw mut self.raw
    }

    /// Returns the name of the codec, such as `"jpeg"`.
    #[must_use]
    pub fn codec_name(&self) -> String {
        string_from_c_chars(&self.raw.codec_name)
    }

    /// Returns how the codec interprets the color channels.
    #[must_use]
    pub fn color_spec(&self) -> ColorSpec {
        self.raw.color_spec.into()
    }

    /// Returns the chroma subsampling.
    #[must_use]
    pub fn chroma_subsampling(&self) -> ChromaSubsampling {
        self.raw.chroma_subsampling.into()
    }

    /// Returns how color channels are arranged in channels and planes.
    #[must_use]
    pub fn sample_format(&self) -> SampleFormat {
        self.raw.sample_format.into()
    }

    /// Returns the number of planes.
    #[must_use]
    pub const fn num_planes(&self) -> u32 {
        self.raw.num_planes
    }

    /// Returns plane `index`, or `None` if there is no such plane.
    #[must_use]
    pub fn plane(&self, index: usize) -> Option<ImagePlaneInfo> {
        self.raw
            .plane_info
            .get(index)
            .filter(|_| index < self.raw.num_planes as usize)
            .copied()
            .map(|raw| ImagePlaneInfo { raw })
    }

    /// Returns an iterator over the planes.
    pub fn planes(&self) -> impl Iterator<Item = ImagePlaneInfo> + '_ {
        self.raw.plane_info[..(self.raw.num_planes as usize).min(self.raw.plane_info.len())]
            .iter()
            .copied()
            .map(|raw| ImagePlaneInfo { raw })
    }

    /// Returns whether the pixels are in host or device memory.
    #[must_use]
    pub fn buffer_kind(&self) -> ImageBufferKind {
        self.raw.buffer_kind.into()
    }

    fn rebuild_extension_chain(&mut self) {
        self.raw.struct_next = ptr::null_mut();
        if let Some(jpeg) = self.jpeg.as_mut() {
            jpeg.struct_next = ptr::null_mut();
            self.raw.struct_next = (&raw mut **jpeg).cast();
        }
    }
}

/// JPEG-specific image information.
#[derive(Debug, Clone, Copy)]
pub struct JpegImageInfo {
    raw: sys::nvimgcodecJpegImageInfo_t,
}

impl Default for JpegImageInfo {
    fn default() -> Self {
        Self {
            raw: sys::nvimgcodecJpegImageInfo_t::init(),
        }
    }
}

impl JpegImageInfo {
    /// Creates JPEG information with every field zeroed.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the JPEG coding process, such as baseline or progressive.
    pub fn set_encoding(&mut self, encoding: JpegEncoding) -> &mut Self {
        self.raw.encoding = encoding.into();
        self
    }
}

/// Image that nvImageCodec can decode into or encode from.
///
/// It keeps its [`Instance`] alive and borrows the memory that holds its pixels.
#[derive(Debug)]
pub struct Image<'a> {
    raw: sys::nvimgcodecImage_t,
    instance: InstanceRef,
    info: ImageInfo<'a>,
}

/// Native image handle with the values that keep it valid, from [`Image::into_raw`].
///
/// Destroy the handle before dropping this value, or pass it back to [`Image::from_raw`].
#[derive(Debug)]
pub struct RawImage<'a> {
    /// Native image handle.
    pub handle: sys::nvimgcodecImage_t,
    owner: Image<'a>,
}

// SAFETY: native images are exclusively owned, and safe submissions finish before returning.
// Moving the owner also moves its backing-storage borrow and retained stream.
#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "native owners transfer their handles and storage borrows under the adjacent safety contract"
)]
unsafe impl Send for Image<'_> {}

// SAFETY: the transfer carries the same exclusive image ownership and sample borrow.
unsafe impl Send for RawImage<'_> {}

impl<'a> Image<'a> {
    pub(crate) fn instance(&self) -> &InstanceRef {
        &self.instance
    }

    /// Creates an image for `instance` from `image_info`.
    ///
    /// # Errors
    ///
    /// Returns an error if `image_info` has no pixel memory or nvImageCodec fails to create the image.
    pub fn create(instance: &Instance, image_info: ImageInfo<'a>) -> Result<Self> {
        if image_info.binding.is_none() {
            return Err(Error::Unsupported {
                name: "image_without_sample_storage".into(),
            });
        }

        let mut raw = ptr::null_mut();
        // SAFETY: `nvimgcodecImageCreate` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::nvimgcodecImageCreate(
                instance.as_raw(),
                &raw mut raw,
                image_info.as_raw(),
            ))?;
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            raw,
            instance: instance.as_instance_ref(),
            info: image_info,
        })
    }

    /// Creates an image for `instance` from initialized, writable host pixels in `buffer`.
    ///
    /// # Errors
    ///
    /// Returns an error if [`ImageInfo::from_host_buffer`] fails or nvImageCodec fails to create the image.
    pub fn from_host_buffer(
        instance: &Instance,
        buffer: &'a mut [u8],
        sample_format: SampleFormat,
        planes: &[ImagePlaneInfo],
    ) -> Result<Self> {
        Self::create(
            instance,
            ImageInfo::from_host_buffer(buffer, sample_format, planes)?,
        )
    }

    /// Creates an image for `instance` from initialized, read-only host pixels in `buffer`, for encoding.
    ///
    /// # Errors
    ///
    /// Returns an error if [`ImageInfo::from_host_input`] fails or nvImageCodec fails to create the image.
    pub fn from_host_input(
        instance: &Instance,
        buffer: &'a [u8],
        sample_format: SampleFormat,
        planes: &[ImagePlaneInfo],
    ) -> Result<Self> {
        Self::create(
            instance,
            ImageInfo::from_host_input(buffer, sample_format, planes)?,
        )
    }

    /// Creates an image for `instance` that decodes into uninitialized host memory in `buffer`.
    ///
    /// # Errors
    ///
    /// Returns an error if [`ImageInfo::from_host_output`] fails or nvImageCodec fails to create the image.
    pub fn from_host_output(
        instance: &Instance,
        buffer: &'a mut [MaybeUninit<u8>],
        sample_format: SampleFormat,
        planes: &[ImagePlaneInfo],
    ) -> Result<Self> {
        Self::create(
            instance,
            ImageInfo::from_host_output(buffer, sample_format, planes)?,
        )
    }

    /// Creates an image for `instance` from initialized, writable device pixels in `buffer`.
    ///
    /// # Errors
    ///
    /// Returns an error if [`ImageInfo::from_device_buffer_mut`] fails or nvImageCodec fails to create the image.
    pub fn from_device_buffer_mut<B>(
        instance: &Instance,
        buffer: &'a mut B,
        sample_format: SampleFormat,
        planes: &[ImagePlaneInfo],
    ) -> Result<Self>
    where
        B: DeviceSliceMut<u8> + ?Sized,
    {
        Self::create(
            instance,
            ImageInfo::from_device_buffer_mut(buffer, sample_format, planes)?,
        )
    }

    /// Creates an image for `instance` from initialized, read-only device pixels in `buffer`, for encoding.
    ///
    /// # Errors
    ///
    /// Returns an error if [`ImageInfo::from_device_buffer`] fails or nvImageCodec fails to create the image.
    pub fn from_device_buffer<B: DeviceSlice<u8> + ?Sized>(
        instance: &Instance,
        buffer: &'a B,
        sample_format: SampleFormat,
        planes: &[ImagePlaneInfo],
    ) -> Result<Self> {
        Self::create(
            instance,
            ImageInfo::from_device_buffer(buffer, sample_format, planes)?,
        )
    }

    /// Creates an image for `instance` that decodes into uninitialized device memory in `buffer`.
    ///
    /// # Errors
    ///
    /// Returns an error if [`ImageInfo::from_device_output`] fails or nvImageCodec fails to create the image.
    pub fn from_device_output(
        instance: &Instance,
        buffer: DeviceOutputView<'a, u8>,
        sample_format: SampleFormat,
        planes: &[ImagePlaneInfo],
    ) -> Result<Self> {
        Self::create(
            instance,
            ImageInfo::from_device_output(buffer, sample_format, planes)?,
        )
    }

    /// Returns how the image may use its pixel memory.
    ///
    /// After a successful decode, an output image's pixels are initialized; its row padding is not.
    #[must_use]
    pub fn sample_access(&self) -> SampleAccess {
        self.info.access
    }

    /// Returns the device memory region of the pixels, or `None` for host memory.
    ///
    /// The region does not claim that the pixels are initialized.
    #[must_use]
    pub fn region(&self) -> Option<DeviceRegion<'_>> {
        match &self.info.binding {
            Some(ImageBuffer::Device(region)) => Some(*region),
            _ => None,
        }
    }

    /// Returns the CUDA stream of the image, if it has one.
    #[must_use]
    pub fn stream(&self) -> Option<&Stream> {
        self.info.stream.as_ref()
    }

    /// Returns row `row` of plane `plane` in host memory, without padding.
    ///
    /// # Errors
    ///
    /// Returns an error if the pixels are not initialized, the image is not in host memory, or `plane` or `row` is out of bounds.
    /// It also fails if a row stride does not fit `usize` or the sample type has no size, which can only happen after writes through [`ImageInfo::as_raw_mut`] or a raw handle.
    pub fn host_row(&self, plane: usize, row: usize) -> Result<&[u8]> {
        if self.info.access == SampleAccess::Output {
            return Err(Error::UninitializedSamples);
        }

        let Some(ImageBuffer::Host { pointer, .. }) = &self.info.binding else {
            return Err(Error::Unsupported {
                name: "host_samples".into(),
            });
        };
        let planes = self.info.planes().collect::<Vec<_>>();
        let current = planes
            .get(plane)
            .ok_or_else(|| Error::out_of_range("plane"))?;
        if row >= current.height() as usize {
            return Err(Error::out_of_range("row"));
        }

        // Construction validated the full plane extents and sample alignment.
        let mut offset = row * to_usize(current.row_stride(), "row stride")?;
        for previous in &planes[..plane] {
            offset += to_usize(previous.row_stride(), "row stride")? * previous.height() as usize;
        }

        let sample_bytes = current
            .sample_type()
            .byte_size()
            .ok_or_else(|| Error::Unsupported {
                name: "sample type".into(),
            })?;
        let bytes = current.width() as usize * current.num_channels() as usize * sample_bytes;
        // SAFETY: the plane lies inside the host buffer that this image borrows.
        Ok(unsafe { slice::from_raw_parts(pointer.cast::<u8>().add(offset), bytes) })
    }

    pub(crate) fn prepare(&mut self, stream: Option<&Stream>, output: bool) -> Result<()> {
        if output && self.info.access == SampleAccess::Input {
            return Err(Error::ReadOnlySamples);
        }

        if !output && self.info.access == SampleAccess::Output {
            return Err(Error::UninitializedSamples);
        }

        if let Some(default) = stream {
            if let Some(actual) = self.info.stream.as_ref()
                && actual.context() != default.context()
            {
                return Err(Error::StreamContextMismatch);
            }
            if self.info.stream.is_none() {
                self.info.set_stream(default)?;
                // nvImageCodec explicitly supports reusing an existing image handle.
                // SAFETY: `nvimgcodecImageCreate` receives live handles owned by the wrappers and valid output locations.
                unsafe {
                    try_ffi!(sys::nvimgcodecImageCreate(
                        self.instance.as_raw(),
                        &raw mut self.raw,
                        self.info.as_raw()
                    ))?;
                }
            }
        } else if self.region().is_some() {
            return Err(Error::Unsupported {
                name: "device_samples_in_cpu_only_codec".into(),
            });
        }

        if let Some(stream) = &self.info.stream {
            stream.ensure_not_capturing()?;
            if let Some(region) = self.region() {
                stream.join_regions(&[region])?;
            }
        }
        Ok(())
    }

    pub(crate) fn decoded(&mut self) {
        if self.info.access == SampleAccess::Output {
            self.info.access = SampleAccess::InputOutput;
        }
    }

    /// Returns the image information.
    #[must_use]
    pub fn image_info(&self) -> &ImageInfo<'a> {
        &self.info
    }

    /// Returns the native handle, which this image still owns.
    #[must_use]
    pub fn as_raw(&self) -> sys::nvimgcodecImage_t {
        self.raw
    }

    /// Takes ownership of a native image created outside this crate, described by `info`.
    ///
    /// # Safety
    ///
    /// `raw` must be owned by nobody else, created with `instance`, and match `info` exactly.
    /// All CPU and GPU work that used it must have finished.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw` is null or `info` has no pixel memory.
    pub unsafe fn from_raw_parts(
        instance: &Instance,
        raw: sys::nvimgcodecImage_t,
        info: ImageInfo<'a>,
    ) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        if info.binding.is_none() {
            return Err(Error::Unsupported {
                name: "image_without_sample_storage".into(),
            });
        }
        Ok(Self {
            raw,
            instance: instance.as_instance_ref(),
            info,
        })
    }

    /// Takes back ownership of a handle from [`into_raw`](Self::into_raw).
    ///
    /// # Safety
    ///
    /// The handle must still be valid and owned by nobody else, with its metadata unchanged.
    /// All work that used it outside this crate must have finished.
    #[must_use]
    pub unsafe fn from_raw(mut raw: RawImage<'a>) -> Self {
        raw.owner.raw = raw.handle;
        raw.owner
    }

    /// Releases the native handle with the values that keep it valid.
    ///
    /// Destroy the handle or pass it back to [`from_raw`](Self::from_raw).
    #[must_use]
    pub fn into_raw(mut self) -> RawImage<'a> {
        let handle = self.raw;
        self.raw = ptr::null_mut();
        RawImage {
            handle,
            owner: self,
        }
    }
}

impl Drop for Image<'_> {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: `nvimgcodecImageDestroy` receives live handles owned by the wrappers.
            unsafe {
                if let Err(error) = try_ffi!(sys::nvimgcodecImageDestroy(self.raw)) {
                    eprintln!(
                        "failed to free nvimagecodec image: {}",
                        singe_core::ErrorChain(&error)
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use singe_cuda::{memory::DeviceMemory, testing::bootstrap_for_device};

    use super::*;
    use crate::types::JpegEncoding;

    #[test]
    fn test_image_info_from_host_buffer_rejects_short_buffer() -> Result<()> {
        let planes = [ImagePlaneInfo::create(4, 2, 4, 1, SampleDataType::U8)?];
        let mut buffer = [0_u8; 7];

        let result = ImageInfo::from_host_buffer(&mut buffer, SampleFormat::PlanarY, &planes);

        assert!(matches!(
            result,
            Err(Error::LengthMismatch { name, expected, actual })
                if name == "buffer" && expected == 8 && actual == 7
        ));
        Ok(())
    }

    #[test]
    fn test_image_info_from_host_buffer_accepts_declared_layout() -> Result<()> {
        let planes = [ImagePlaneInfo::create(4, 2, 4, 1, SampleDataType::U8)?];
        let mut buffer = [0_u8; 8];

        let image_info = ImageInfo::from_host_buffer(&mut buffer, SampleFormat::PlanarY, &planes)?;

        assert_eq!(image_info.buffer_kind(), ImageBufferKind::StridedHost);
        assert_eq!(image_info.sample_format(), SampleFormat::PlanarY);
        assert_eq!(image_info.num_planes(), 1);
        Ok(())
    }

    #[cfg(feature = "testing")]
    #[test]
    fn test_image_info_from_device_buffer_accepts_declared_layout() -> Result<()> {
        let planes = [ImagePlaneInfo::create(4, 2, 4, 1, SampleDataType::U8)?];
        let (_lock, context) = bootstrap_for_device(0)?;
        let mut buffer = DeviceMemory::from_slice(&context, &[1_u8; 8])?;
        let mut view = buffer.view_mut();

        let image_info =
            ImageInfo::from_device_buffer_mut(&mut view, SampleFormat::PlanarY, &planes)?;

        assert_eq!(image_info.buffer_kind(), ImageBufferKind::StridedDevice);
        assert_eq!(image_info.sample_format(), SampleFormat::PlanarY);
        assert_eq!(image_info.num_planes(), 1);
        Ok(())
    }

    #[test]
    fn test_image_info_jpeg_extension_uses_owned_stable_pointer() -> Result<()> {
        let planes = [ImagePlaneInfo::create(4, 2, 4, 1, SampleDataType::U8)?];
        let mut buffer = [0_u8; 8];
        let mut jpeg = JpegImageInfo::new();
        jpeg.set_encoding(JpegEncoding::BaselineDct);

        let mut image_info =
            ImageInfo::from_host_buffer(&mut buffer, SampleFormat::PlanarY, &planes)?;
        image_info.set_jpeg(jpeg);

        assert!(!image_info.raw.struct_next.is_null());
        assert_eq!(
            image_info.raw.struct_next,
            image_info
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
