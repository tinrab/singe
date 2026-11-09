//! Images in device memory and their native descriptions.

use std::{
    ffi::c_void,
    marker::PhantomData,
    mem::{MaybeUninit, align_of, size_of},
    sync::Arc,
};

use singe_cuda::{
    context::Context as CudaContext,
    memory::DeviceMemory,
    view::{DeviceOutputView, DeviceRegion, DeviceRepr, DeviceView},
};
use singe_nvjpeg_sys as sys;

use crate::{
    error::{Error, Result},
    types::{ChromaSubsampling, JpegComponentInfo, JpegInfo, OutputFormat},
    utility::{ceil_div_u32, checked_mul, checked_row_bytes, to_u64, to_usize},
};

/// Largest number of components, and so planes, that an image can have.
pub const MAX_COMPONENTS: usize = sys::NVJPEG_MAX_COMPONENT as usize;

/// Image size in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Size {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

/// Read-only image plane in device memory.
#[derive(Debug, Clone, Copy)]
pub struct Plane<'a, T> {
    region: DeviceRegion<'a>,
    size: Size,
    pitch: usize,
    _marker: PhantomData<&'a T>,
}

/// Writable image plane in device memory.
#[derive(Debug)]
pub struct PlaneMut<'a, T> {
    region: DeviceRegion<'a>,
    size: Size,
    pitch: usize,
    _marker: PhantomData<&'a mut MaybeUninit<T>>,
}

/// Read-only image in device memory with `C` planes.
#[derive(Debug, Clone, Copy)]
pub struct ImageView<'a, T, const C: usize> {
    planes: [Plane<'a, T>; C],
}

/// Writable image in device memory with `C` planes.
#[derive(Debug)]
pub struct ImageViewMut<'a, T, const C: usize> {
    planes: [PlaneMut<'a, T>; C],
}

/// Size and smallest row pitch of one decoded output plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OutputPlaneSpec {
    size: Size,
    row_bytes: usize,
}

/// Output planes needed to decode an image into an output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OutputSpec {
    format: OutputFormat,
    planes: [Option<OutputPlaneSpec>; MAX_COMPONENTS],
}

/// Native image description: a pointer and row pitch for each plane.
///
/// It does not own or borrow the memory it points to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Image {
    raw: sys::nvjpegImage_t,
}

impl Image {
    /// Creates a description with no planes.
    #[must_use]
    pub fn new() -> Self {
        Self {
            raw: sys::nvjpegImage_t::default(),
        }
    }

    /// Creates a description from plane pointers and their pitches.
    ///
    /// # Safety
    ///
    /// Each non-null pointer must point to device-accessible memory large enough for the operation that uses the description.
    /// The memory must stay valid, and used by nothing else, until the work on that operation's stream finishes.
    ///
    /// # Errors
    ///
    /// Returns an error if a null plane has a nonzero pitch, a non-null plane has a zero pitch, or a pitch does not fit nvJPEG's integer types.
    pub unsafe fn from_raw_parts(
        channels: [*mut u8; MAX_COMPONENTS],
        pitches: [usize; MAX_COMPONENTS],
    ) -> Result<Self> {
        let mut raw = sys::nvjpegImage_t::default();
        for index in 0..MAX_COMPONENTS {
            validate_image_channel_pitch(channels[index], pitches[index])?;
            raw.channel[index] = channels[index];
            raw.pitch[index] = to_u64(pitches[index], "pitch")?;
        }

        Ok(Self { raw })
    }

    pub(crate) fn from_view<T, const C: usize>(view: &ImageView<'_, T, C>) -> Result<Self> {
        validate_component_count(C)?;
        let mut image = Self::new();
        for (index, plane) in view.planes.iter().enumerate() {
            image.set_channel_raw(index, plane.as_ptr().cast_mut().cast(), plane.pitch)?;
        }
        Ok(image)
    }

    /// Creates a description from the planes of `view`.
    ///
    /// The description copies the pointers and pitches; it does not borrow `view` or initialize anything.
    ///
    /// # Errors
    ///
    /// Returns an error if `view` has no planes or more than [`MAX_COMPONENTS`], a plane's pointer and pitch disagree about whether it is present, or a pitch does not fit nvJPEG's integer types.
    pub fn from_view_mut<T, const C: usize>(view: &mut ImageViewMut<'_, T, C>) -> Result<Self> {
        validate_component_count(C)?;
        let mut image = Self::new();
        for (index, plane) in view.planes.iter_mut().enumerate() {
            image.set_channel_raw(index, plane.as_mut_ptr().cast(), plane.pitch)?;
        }
        Ok(image)
    }

    /// Sets the pointer and pitch of plane `index`.
    ///
    /// # Safety
    ///
    /// `ptr` must meet the requirements of [`from_raw_parts`](Self::from_raw_parts).
    ///
    /// # Errors
    ///
    /// Returns an error if `index` is not below [`MAX_COMPONENTS`], `ptr` and `pitch` disagree about whether the plane is present, or `pitch` does not fit nvJPEG's integer types.
    pub unsafe fn set_channel(&mut self, index: usize, ptr: *mut u8, pitch: usize) -> Result<()> {
        self.set_channel_raw(index, ptr, pitch)
    }

    fn set_channel_raw(&mut self, index: usize, ptr: *mut u8, pitch: usize) -> Result<()> {
        if index >= MAX_COMPONENTS {
            return Err(Error::OutOfRange {
                name: "index".into(),
            });
        }

        validate_image_channel_pitch(ptr, pitch)?;
        self.raw.channel[index] = ptr;
        self.raw.pitch[index] = to_u64(pitch, "pitch")?;
        Ok(())
    }

    /// Returns a pointer to the native description.
    #[must_use]
    pub const fn as_raw(&self) -> *const sys::nvjpegImage_t {
        &raw const self.raw
    }

    /// Returns a mutable pointer to the native description.
    pub const fn as_raw_mut(&mut self) -> *mut sys::nvjpegImage_t {
        &raw mut self.raw
    }

    /// Returns the device pointer of plane `index`.
    ///
    /// # Errors
    ///
    /// Returns an error if `index` is not below [`MAX_COMPONENTS`].
    pub fn channel_ptr(&self, index: usize) -> Result<*mut u8> {
        if index >= MAX_COMPONENTS {
            return Err(Error::OutOfRange {
                name: "index".into(),
            });
        }
        Ok(self.raw.channel[index])
    }

    /// Returns the row pitch of plane `index` in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if `index` is not below [`MAX_COMPONENTS`] or the pitch does not fit `usize`.
    pub fn channel_pitch(&self, index: usize) -> Result<usize> {
        if index >= MAX_COMPONENTS {
            return Err(Error::OutOfRange {
                name: "index".into(),
            });
        }
        to_usize(self.raw.pitch[index], "pitch")
    }

    /// Returns whether plane `index` has a non-null pointer.
    ///
    /// # Errors
    ///
    /// Returns an error if `index` is not below [`MAX_COMPONENTS`].
    pub fn has_channel(&self, index: usize) -> Result<bool> {
        Ok(!self.channel_ptr(index)?.is_null())
    }
}

impl Default for Image {
    fn default() -> Self {
        Self::new()
    }
}

impl Size {
    /// Creates a size of `width` by `height` pixels.
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Returns the width in pixels.
    #[must_use]
    pub const fn width(self) -> u32 {
        self.width
    }

    /// Returns the height in pixels.
    #[must_use]
    pub const fn height(self) -> u32 {
        self.height
    }

    /// Returns whether either dimension is zero.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.width == 0 || self.height == 0
    }
}

impl OutputPlaneSpec {
    /// Creates a plane of `size` with rows of at least `row_bytes` bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if a nonempty plane has `row_bytes` smaller than its width, or the plane's size in bytes overflows.
    pub fn create(size: Size, row_bytes: usize) -> Result<Self> {
        required_plane_bytes(size, row_bytes, 1)?;
        Ok(Self::from_parts(size, row_bytes))
    }

    const fn from_parts(size: Size, row_bytes: usize) -> Self {
        Self { size, row_bytes }
    }

    /// Returns the size in pixels.
    #[must_use]
    pub const fn size(self) -> Size {
        self.size
    }

    /// Returns the smallest row pitch in bytes.
    #[must_use]
    pub const fn row_bytes(self) -> usize {
        self.row_bytes
    }

    /// Returns the smallest size of the plane in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the size overflows `usize`.
    pub fn byte_len(self) -> Result<usize> {
        if self.size.is_empty() {
            return Ok(0);
        }
        checked_mul(
            self.row_bytes,
            self.size.height as usize,
            "output plane byte length",
        )
    }

    /// Allocates uninitialized device memory for the plane, with rows of exactly [`row_bytes`](Self::row_bytes) bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the plane's size overflows or the allocation fails.
    pub fn create_uninit_buffer(
        self,
        context: &Arc<CudaContext>,
    ) -> Result<DeviceMemory<MaybeUninit<u8>>> {
        Ok(DeviceMemory::<u8>::create_uninit(
            context,
            self.byte_len()?,
        )?)
    }
}

impl OutputSpec {
    /// Returns the smallest output planes for decoding the image described by `info` into `format`.
    ///
    /// - [`Unchanged`](OutputFormat::Unchanged) and [`Yuv`](OutputFormat::Yuv) use each component's own size.
    /// - Planar RGB and BGR use three full-size planes, and the interleaved formats one plane.
    /// - [`Nv12`](OutputFormat::Nv12) uses a full-size luma plane and an interleaved chroma plane, half the height rounded up, with rows rounded up to whole chroma pairs.
    /// - [`Yuy2`](OutputFormat::Yuy2) rows are rounded up to whole two-pixel groups.
    ///
    /// This computes sizes only; whether nvJPEG can decode the image this way depends on the image and backend.
    ///
    /// # Errors
    ///
    /// Returns an error if `info` has no components, a size overflows, or [`Nv12`](OutputFormat::Nv12) is requested for an image without 4:2:0 subsampling.
    pub fn for_jpeg(info: &JpegInfo, format: OutputFormat) -> Result<Self> {
        let mut planes = [None; MAX_COMPONENTS];
        let image_size = jpeg_image_size(info)?;

        match format {
            OutputFormat::Unchanged | OutputFormat::Yuv => {
                for (index, component) in jpeg_components(info).enumerate() {
                    let size = component_size(component);
                    planes[index] = Some(OutputPlaneSpec::from_parts(
                        size,
                        checked_row_bytes(size, 1)?,
                    ));
                }
            }
            OutputFormat::Y => {
                planes[0] = Some(OutputPlaneSpec::from_parts(
                    image_size,
                    checked_row_bytes(image_size, 1)?,
                ));
            }
            OutputFormat::Rgb | OutputFormat::Bgr => {
                for plane in planes.iter_mut().take(3) {
                    *plane = Some(OutputPlaneSpec::from_parts(
                        image_size,
                        checked_row_bytes(image_size, 1)?,
                    ));
                }
            }
            OutputFormat::RgbInterleaved | OutputFormat::BgrInterleaved => {
                planes[0] = Some(OutputPlaneSpec::from_parts(
                    image_size,
                    checked_row_bytes(image_size, 3)?,
                ));
            }
            OutputFormat::UnchangedInterleaved => {
                let component_count = jpeg_components(info).count();
                planes[0] = Some(OutputPlaneSpec::from_parts(
                    image_size,
                    checked_row_bytes(image_size, component_count)?,
                ));
            }
            OutputFormat::UnchangedInterleavedU16 => {
                let component_count = jpeg_components(info).count();
                planes[0] = Some(OutputPlaneSpec::from_parts(
                    image_size,
                    checked_row_bytes(
                        image_size,
                        component_count.checked_mul(2).ok_or(Error::OutOfRange {
                            name: "component count".into(),
                        })?,
                    )?,
                ));
            }
            OutputFormat::Nv12 => {
                if info.subsampling() != ChromaSubsampling::Css420 {
                    return Err(Error::UnsupportedDecodeConfiguration {
                        reason: "nv12 output requires 4:2:0 input".into(),
                    });
                }

                let uv_size = Size::new(image_size.width, ceil_div_u32(image_size.height, 2));
                planes[0] = Some(OutputPlaneSpec::from_parts(
                    image_size,
                    checked_row_bytes(image_size, 1)?,
                ));
                planes[1] = Some(OutputPlaneSpec::from_parts(
                    uv_size,
                    checked_mul(ceil_div_u32(image_size.width, 2) as usize, 2, "row bytes")?,
                ));
            }
            OutputFormat::Yuy2 => {
                planes[0] = Some(OutputPlaneSpec::from_parts(
                    image_size,
                    checked_mul(ceil_div_u32(image_size.width, 2) as usize, 4, "row bytes")?,
                ));
            }
        }

        Ok(Self { format, planes })
    }

    /// Returns the output format.
    #[must_use]
    pub const fn format(self) -> OutputFormat {
        self.format
    }

    /// Returns every plane slot, with `None` for unused slots.
    #[must_use]
    pub const fn planes(&self) -> &[Option<OutputPlaneSpec>; MAX_COMPONENTS] {
        &self.planes
    }

    /// Returns an iterator over the used planes.
    pub fn present_planes(&self) -> impl Iterator<Item = OutputPlaneSpec> + '_ {
        self.planes.iter().flatten().copied()
    }

    /// Returns the only plane, for single-plane formats such as [`OutputFormat::RgbInterleaved`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] if there is not exactly one plane.
    pub fn single_plane(&self) -> Result<OutputPlaneSpec> {
        let mut planes = self.present_planes();
        let Some(plane) = planes.next() else {
            return Err(Error::LengthMismatch {
                name: "planes".into(),
                expected: 1,
                actual: 0,
            });
        };
        if planes.next().is_some() {
            return Err(Error::LengthMismatch {
                name: "planes".into(),
                expected: 1,
                actual: self.plane_count(),
            });
        }
        Ok(plane)
    }

    /// Returns the number of used planes.
    #[must_use]
    pub fn plane_count(&self) -> usize {
        self.present_planes().count()
    }
}

impl<'a, T> Plane<'a, T> {
    /// Returns the CUDA context of the plane's memory.
    #[must_use]
    pub fn context(&self) -> &'a Arc<CudaContext> {
        self.region.context()
    }

    /// Returns the size of the plane's memory in bytes, including row padding.
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        self.region.byte_len()
    }

    /// Returns the plane's memory region.
    #[must_use]
    pub const fn region(&self) -> DeviceRegion<'a> {
        self.region
    }

    /// Creates a plane of `size` from the initialized values in `view`, with a row pitch of `pitch`.
    ///
    /// The pitch is in bytes.
    /// The memory must hold every row but the last at full pitch, plus the last row's pixels; padding after the last row is not needed.
    /// An empty plane needs no memory.
    ///
    /// # Errors
    ///
    /// Returns an error if the address or pitch is not aligned for `T`, a nonempty plane has a null pointer, the pitch is shorter than a row, a size overflows, or the memory is too small.
    pub fn create(view: DeviceView<'a, T>, size: Size, pitch: usize) -> Result<Self>
    where
        T: DeviceRepr,
    {
        // SAFETY: the initialized view retains readable storage for this lifetime.
        unsafe { Self::from_raw_parts(view.region(), size, pitch) }
    }

    /// Creates a plane of `size` from a memory region, with a row pitch of `pitch`.
    ///
    /// The pitch is in bytes.
    /// The memory must hold every row but the last at full pitch, plus the last row's pixels; padding after the last row is not needed.
    /// An empty plane needs no memory.
    ///
    /// # Safety
    ///
    /// Every pixel must be aligned, initialized, a valid `T`, and readable for `'a`; row padding may be uninitialized.
    /// Work that writes the pixels must be ordered before every use of the plane.
    ///
    /// # Errors
    ///
    /// Returns an error if the address or pitch is not aligned for `T`, a nonempty plane has a null pointer, the pitch is shorter than a row, a size overflows, or the memory is too small.
    pub unsafe fn from_raw_parts(region: DeviceRegion<'a>, size: Size, pitch: usize) -> Result<Self>
    where
        T: DeviceRepr,
    {
        let () = T::LAYOUT;
        validate_plane_alignment(region, pitch, align_of::<T>())?;
        validate_plane_ptr("plane", region.as_ptr().cast::<T>(), size)?;
        validate_plane_capacity("plane", region.byte_len(), size, pitch, size_of::<T>())?;
        Ok(Self {
            region,
            size,
            pitch,
            _marker: PhantomData,
        })
    }

    /// Returns the device pointer to the first pixel.
    ///
    /// It does not wait for work that writes the pixels.
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.region.as_ptr().cast()
    }

    /// Returns the size in pixels.
    #[must_use]
    pub const fn size(&self) -> Size {
        self.size
    }

    /// Returns the row pitch in bytes.
    #[must_use]
    pub const fn pitch(&self) -> usize {
        self.pitch
    }
}

impl<'a, T> PlaneMut<'a, T> {
    /// Returns the CUDA context of the plane's memory.
    #[must_use]
    pub fn context(&self) -> &Arc<CudaContext> {
        self.region.context()
    }

    /// Returns the size of the plane's memory in bytes, including row padding.
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        self.region.byte_len()
    }

    /// Returns the plane's memory region.
    ///
    /// The region does not claim that the pixels are initialized.
    #[must_use]
    pub fn region(&self) -> DeviceRegion<'_> {
        self.region
    }

    /// Creates a writable plane of `size` from `view`, with a row pitch of `pitch`.
    ///
    /// The memory may be uninitialized.
    /// The pitch is in bytes.
    /// The memory must hold every row but the last at full pitch, plus the last row's pixels; padding after the last row is not needed.
    /// An empty plane needs no memory.
    ///
    /// # Errors
    ///
    /// Returns an error if the address or pitch is not aligned for `T`, a nonempty plane has a null pointer, the pitch is shorter than a row, a size overflows, or the memory is too small.
    pub fn create(view: DeviceOutputView<'a, T>, size: Size, pitch: usize) -> Result<Self>
    where
        T: DeviceRepr,
    {
        // SAFETY: consuming the output view preserves its exclusive borrow.
        unsafe { Self::from_raw_parts(view.into_region(), size, pitch) }
    }

    /// Creates a writable plane of `size` from a memory region, with a row pitch of `pitch`.
    ///
    /// The pitch is in bytes.
    /// The memory must hold every row but the last at full pitch, plus the last row's pixels; padding after the last row is not needed.
    /// An empty plane needs no memory.
    ///
    /// # Safety
    ///
    /// The region must be aligned for `T` and writable by nothing else for `'a`, including work already queued.
    /// It need not be initialized.
    ///
    /// # Errors
    ///
    /// Returns an error if the address or pitch is not aligned for `T`, a nonempty plane has a null pointer, the pitch is shorter than a row, a size overflows, or the memory is too small.
    pub unsafe fn from_raw_parts(region: DeviceRegion<'a>, size: Size, pitch: usize) -> Result<Self>
    where
        T: DeviceRepr,
    {
        let () = T::LAYOUT;
        validate_plane_alignment(region, pitch, align_of::<T>())?;
        validate_plane_ptr("plane", region.as_ptr().cast::<T>(), size)?;
        validate_plane_capacity("plane", region.byte_len(), size, pitch, size_of::<T>())?;
        Ok(Self {
            region,
            size,
            pitch,
            _marker: PhantomData,
        })
    }

    /// Returns the writable device pointer to the first pixel.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.region.as_ptr().cast()
    }

    /// Returns the size in pixels.
    #[must_use]
    pub const fn size(&self) -> Size {
        self.size
    }

    /// Returns the row pitch in bytes.
    #[must_use]
    pub const fn pitch(&self) -> usize {
        self.pitch
    }
}

impl<'a, T, const C: usize> ImageView<'a, T, C> {
    /// Creates an image from `planes`.
    ///
    /// The operation that uses the image checks the planes' sizes and CUDA contexts.
    ///
    /// # Errors
    ///
    /// Returns an error if `C` is zero or larger than [`MAX_COMPONENTS`].
    pub fn create(planes: [Plane<'a, T>; C]) -> Result<Self> {
        validate_component_count(C)?;
        Ok(Self { planes })
    }

    /// Returns the planes.
    #[must_use]
    pub const fn planes(&self) -> &[Plane<'a, T>; C] {
        &self.planes
    }

    pub(crate) fn descriptor(&self) -> Result<Image> {
        Image::from_view(self)
    }
}

impl<'a, T, const C: usize> ImageViewMut<'a, T, C> {
    pub(crate) fn validate_output(
        &self,
        spec: &OutputSpec,
        context: &Arc<CudaContext>,
    ) -> Result<()> {
        if C != spec.plane_count() {
            return Err(Error::LengthMismatch {
                name: "output planes".into(),
                expected: spec.plane_count(),
                actual: C,
            });
        }

        for (plane, required) in self.planes.iter().zip(spec.present_planes()) {
            if spec.format() == OutputFormat::UnchangedInterleavedU16 {
                validate_plane_alignment(plane.region(), plane.pitch(), align_of::<u16>())?;
            }

            if plane.context().as_ref() != context.as_ref() {
                return Err(Error::StreamContextMismatch);
            }

            let minimum_row_bytes = required.row_bytes;
            if plane.size.width < required.size.width
                || plane.size.height < required.size.height
                || plane.pitch < minimum_row_bytes
            {
                return Err(Error::OutOfRange {
                    name: "output plane layout".into(),
                });
            }

            let prefix = checked_mul(
                plane.pitch,
                required.size.height.saturating_sub(1) as usize,
                "output plane pitch",
            )?;
            let bytes = prefix
                .checked_add(required.row_bytes)
                .ok_or(Error::OutOfRange {
                    name: "output plane byte length".into(),
                })?;
            if plane.byte_len() < bytes {
                return Err(Error::BufferTooSmall {
                    name: "output plane".into(),
                    required: bytes,
                    actual: plane.byte_len(),
                });
            }
        }
        Ok(())
    }

    /// Creates a writable image from `planes`.
    ///
    /// The operation that uses the image checks the planes' sizes and CUDA contexts.
    ///
    /// # Errors
    ///
    /// Returns an error if `C` is zero or larger than [`MAX_COMPONENTS`].
    pub fn create(planes: [PlaneMut<'a, T>; C]) -> Result<Self> {
        validate_component_count(C)?;
        Ok(Self { planes })
    }

    /// Returns the planes.
    #[must_use]
    pub const fn planes(&self) -> &[PlaneMut<'a, T>; C] {
        &self.planes
    }

    /// Returns the planes mutably.
    pub fn planes_mut(&mut self) -> &mut [PlaneMut<'a, T>; C] {
        &mut self.planes
    }

    /// Returns a native description of the image.
    ///
    /// The description copies the pointers and pitches; it does not borrow the image or initialize anything.
    ///
    /// # Errors
    ///
    /// Returns an error if a plane's pointer and pitch disagree about whether it is present, or a pitch does not fit nvJPEG's integer types.
    pub fn descriptor(&mut self) -> Result<Image> {
        Image::from_view_mut(self)
    }
}

/// Converts a typed pointer to the untyped pointer that nvJPEG 2000 image descriptions use.
///
/// # Safety
///
/// If the pointer goes into a device image description, it must point to device-accessible memory.
pub unsafe fn erase_device_ptr<T>(ptr: *mut T) -> *mut c_void {
    ptr.cast()
}

fn validate_component_count(count: usize) -> Result<()> {
    if count == 0 {
        return Err(Error::Empty {
            name: "planes".into(),
        });
    }

    if count > MAX_COMPONENTS {
        return Err(Error::OutOfRange {
            name: "planes".into(),
        });
    }
    Ok(())
}

fn validate_image_channel_pitch(ptr: *mut u8, pitch: usize) -> Result<()> {
    if ptr.is_null() && pitch != 0 {
        return Err(Error::NullPointer {
            name: "channel".into(),
        });
    }

    if !ptr.is_null() && pitch == 0 {
        return Err(Error::OutOfRange {
            name: "pitch".into(),
        });
    }
    Ok(())
}

fn validate_plane_alignment(
    region: DeviceRegion<'_>,
    pitch: usize,
    alignment: usize,
) -> Result<()> {
    if !region.as_ptr().addr().is_multiple_of(alignment) || !pitch.is_multiple_of(alignment) {
        return Err(Error::OutOfRange {
            name: "plane alignment".into(),
        });
    }
    Ok(())
}

fn validate_plane_ptr<T>(name: &'static str, ptr: *const T, size: Size) -> Result<()> {
    if !size.is_empty() && ptr.is_null() {
        return Err(Error::NullPointer { name: name.into() });
    }
    Ok(())
}

fn validate_plane_capacity(
    name: &'static str,
    actual: usize,
    size: Size,
    pitch: usize,
    element_size: usize,
) -> Result<()> {
    let required = required_plane_bytes(size, pitch, element_size)?;
    if actual < required {
        return Err(Error::BufferTooSmall {
            name: name.into(),
            required,
            actual,
        });
    }
    Ok(())
}

fn validate_plane_pitch(size: Size, pitch: usize, element_size: usize) -> Result<()> {
    let row_bytes = row_bytes(size, element_size)?;
    if pitch < row_bytes {
        return Err(Error::OutOfRange {
            name: "pitch".into(),
        });
    }
    Ok(())
}

fn required_plane_bytes(size: Size, pitch: usize, element_size: usize) -> Result<usize> {
    if size.is_empty() {
        return Ok(0);
    }

    validate_plane_pitch(size, pitch, element_size)?;
    let rows_before_last = (size.height - 1) as usize;
    let prefix_bytes = pitch
        .checked_mul(rows_before_last)
        .ok_or(Error::OutOfRange {
            name: "pitch".into(),
        })?;
    prefix_bytes
        .checked_add(row_bytes(size, element_size)?)
        .ok_or(Error::OutOfRange {
            name: "plane byte length".into(),
        })
}

fn row_bytes(size: Size, element_size: usize) -> Result<usize> {
    (size.width as usize)
        .checked_mul(element_size)
        .ok_or(Error::OutOfRange {
            name: "row byte length".into(),
        })
}

fn jpeg_components(info: &JpegInfo) -> impl Iterator<Item = JpegComponentInfo> + '_ {
    info.components().iter().copied()
}

fn jpeg_image_size(info: &JpegInfo) -> Result<Size> {
    let first = jpeg_components(info).next().ok_or(Error::Empty {
        name: "components".into(),
    })?;
    Ok(component_size(first))
}

fn component_size(component: JpegComponentInfo) -> Size {
    Size::new(component.width(), component.height())
}
