//! Native JPEG 2000 image descriptions and images to encode.

use std::{ffi::c_void, ptr};

use singe_cuda::{
    error::Error as CudaError,
    memory::DeviceMemory,
    view::{DeviceRegion, DeviceView},
};
use singe_nvjpeg_sys::k2 as sys;

use crate::{
    error::{Error, Result},
    image::{Plane, PlaneMut, Size},
    k2::{
        MAX_IMAGE_COMPONENTS,
        types::{ImageFormat, ImagePixel, ImageType},
    },
    utility::{checked_mul, to_u32, to_u64},
};

/// Native JPEG 2000 image description: a pointer and row pitch for each plane.
///
/// It owns no memory.
/// The number of components can differ from the number of planes: an interleaved image has one plane holding every component.
/// Safe encoding takes an [`EncodeInput`] instead, which borrows the pixels.
#[derive(Debug)]
pub struct Image {
    pub(crate) raw: sys::nvjpeg2kImage_t,
    pub(crate) pixel_data: Vec<*mut c_void>,
    pub(crate) pitch_in_bytes: Vec<usize>,
    pub(crate) _raw_pitch_in_bytes: Vec<sys::size_t>,
    plane_count: usize,
}

impl Image {
    /// Creates a description from one device allocation per component, with each plane's pitch in `pitch_in_bytes`.
    ///
    /// The description does not borrow the allocations, so keep them alive while it is used.
    ///
    /// # Errors
    ///
    /// Returns an error if `C` is not between 1 and 4, a pointer or pitch is invalid, a plane is misaligned, a size overflows, or an allocation is too small for its plane.
    pub fn from_device_planes<T: ImagePixel, const C: usize>(
        planes: [&DeviceMemory<T>; C],
        sizes: [Size; C],
        pitch_in_bytes: [usize; C],
    ) -> Result<Self> {
        validate_image_component_count(C)?;
        let mut pixel_data = Vec::with_capacity(C);
        let mut pitches = Vec::with_capacity(C);
        for index in 0..C {
            let plane = Plane::create(
                DeviceView::from_buffer(planes[index]),
                Size::new(sizes[index].width, sizes[index].height),
                pitch_in_bytes[index],
            )?;
            pixel_data.push(plane.as_ptr().cast_mut().cast());
            pitches.push(plane.pitch());
        }
        // SAFETY: the plane pointers and pitches describe buffers that stay borrowed for the image lifetime.
        unsafe { Self::from_raw_parts(C, pixel_data, pitches, T::IMAGE_TYPE) }
    }

    /// Creates a description from read-only planes.
    ///
    /// The description does not borrow the planes' memory, so keep it alive while the description is used.
    ///
    /// # Errors
    ///
    /// Returns an error if `C` is not between 1 and 4, a plane pointer is null, or a pitch is zero or does not fit nvJPEG 2000's integer types.
    pub fn from_planes<T: ImagePixel, const C: usize>(planes: &[Plane<'_, T>; C]) -> Result<Self> {
        validate_image_component_count(C)?;
        let pixel_data = planes
            .iter()
            .map(|plane| plane.as_ptr().cast_mut().cast())
            .collect::<Vec<_>>();
        let pitch_in_bytes = planes.iter().map(Plane::pitch).collect::<Vec<_>>();
        // SAFETY: the plane pointers and pitches describe buffers that stay borrowed for the image lifetime.
        unsafe { Self::from_raw_parts(C, pixel_data, pitch_in_bytes, T::IMAGE_TYPE) }
    }

    /// Creates a description from writable planes.
    ///
    /// The description does not borrow the planes' memory, so keep it alive while the description is used.
    ///
    /// # Errors
    ///
    /// Returns an error if `C` is not between 1 and 4, a plane pointer is null, or a pitch is zero or does not fit nvJPEG 2000's integer types.
    pub fn from_planes_mut<T: ImagePixel, const C: usize>(
        planes: &mut [PlaneMut<'_, T>; C],
    ) -> Result<Self> {
        validate_image_component_count(C)?;
        let pixel_data = planes
            .iter_mut()
            .map(|plane| plane.as_mut_ptr().cast())
            .collect::<Vec<_>>();
        let pitch_in_bytes = planes.iter().map(PlaneMut::pitch).collect::<Vec<_>>();
        // SAFETY: the plane pointers and pitches describe buffers that stay borrowed for the image lifetime.
        unsafe { Self::from_raw_parts(C, pixel_data, pitch_in_bytes, T::IMAGE_TYPE) }
    }

    /// Creates a description from plane pointers and pitches, for an image with `component_count` components.
    ///
    /// An interleaved image has one plane for all its components.
    ///
    /// # Safety
    ///
    /// Each pointer must point to device-accessible memory large enough for the operation that uses the description, valid and written by nothing else until the work on that operation's stream finishes.
    ///
    /// # Errors
    ///
    /// Returns an error if the component or plane count is not between 1 and 4, the pointer and pitch slices have different lengths, a pointer is null, a pitch is zero or does not fit nvJPEG 2000's integer types, or there are more planes than components.
    pub unsafe fn from_raw_parts(
        component_count: usize,
        pixel_data: Vec<*mut c_void>,
        pitch_in_bytes: Vec<usize>,
        pixel_type: ImageType,
    ) -> Result<Self> {
        validate_image_component_count(component_count)?;
        if pixel_data.is_empty() {
            return Err(Error::Empty {
                name: "pixel_data".into(),
            });
        }

        if pixel_data.len() != pitch_in_bytes.len() {
            return Err(Error::LengthMismatch {
                name: "pitch_in_bytes".into(),
                expected: pixel_data.len(),
                actual: pitch_in_bytes.len(),
            });
        }

        validate_image_component_count(pixel_data.len())?;
        for (index, (&ptr, &pitch)) in pixel_data.iter().zip(&pitch_in_bytes).enumerate() {
            validate_image_component_pointer(index, ptr)?;
            validate_image_component_pitch(index, pitch)?;
        }

        let mut raw_pitch_in_bytes = pitch_in_bytes
            .iter()
            .copied()
            .map(|pitch| to_u64(pitch, "pitch_in_bytes"))
            .collect::<Result<Vec<_>>>()?;
        let plane_count = pixel_data.len();
        if plane_count > component_count {
            return Err(Error::OutOfRange {
                name: "physical plane count".into(),
            });
        }

        let mut pixel_data = pixel_data;
        // Native arrays have logical-component capacity even for packed images.
        pixel_data.resize(component_count, ptr::null_mut());
        raw_pitch_in_bytes.resize(component_count, 0);
        let raw = sys::nvjpeg2kImage_t {
            pixel_data: pixel_data.as_mut_ptr(),
            pitch_in_bytes: raw_pitch_in_bytes.as_mut_ptr(),
            pixel_type: pixel_type.into(),
            num_components: to_u32(component_count, "component count")?,
        };
        Ok(Self {
            raw,
            pixel_data,
            pitch_in_bytes,
            _raw_pitch_in_bytes: raw_pitch_in_bytes,
            plane_count,
        })
    }

    /// Returns a pointer to the native description.
    #[must_use]
    pub const fn as_raw(&self) -> *const sys::nvjpeg2kImage_t {
        &raw const self.raw
    }

    /// Returns a mutable pointer to the native description.
    pub const fn as_raw_mut(&mut self) -> *mut sys::nvjpeg2kImage_t {
        &raw mut self.raw
    }

    /// Returns the number of components.
    #[must_use]
    pub fn component_count(&self) -> usize {
        self.raw.num_components as usize
    }

    /// Returns the number of planes: one for an interleaved image, and one per component otherwise.
    #[must_use]
    pub fn plane_count(&self) -> usize {
        self.plane_count
    }

    /// Returns the pointer in component slot `index`.
    ///
    /// An interleaved image uses slot 0, and its other slots are null.
    ///
    /// # Errors
    ///
    /// Returns an error if `index` is not below the number of components.
    pub fn component_ptr(&self, index: usize) -> Result<*mut c_void> {
        self.pixel_data
            .get(index)
            .copied()
            .ok_or(Error::OutOfRange {
                name: "component".into(),
            })
    }

    /// Returns each plane's row pitch in bytes.
    #[must_use]
    pub fn pitches(&self) -> &[usize] {
        &self.pitch_in_bytes
    }

    /// Returns the row pitch of plane `index` in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if `index` is not below the number of planes; an interleaved image has only plane 0.
    pub fn pitch_in_bytes(&self, index: usize) -> Result<usize> {
        self.pitch_in_bytes
            .get(index)
            .copied()
            .ok_or(Error::OutOfRange {
                name: "component".into(),
            })
    }

    /// Returns the type of each sample.
    #[must_use]
    pub fn pixel_type(&self) -> ImageType {
        ImageType::from(self.raw.pixel_type)
    }
}

pub(crate) fn validate_image_component_count(count: usize) -> Result<()> {
    if count == 0 {
        return Err(Error::Empty {
            name: "pixel_data".into(),
        });
    }

    if count > MAX_IMAGE_COMPONENTS {
        return Err(Error::OutOfRange {
            name: "pixel_data".into(),
        });
    }
    Ok(())
}

fn validate_image_component_pointer(index: usize, ptr: *mut c_void) -> Result<()> {
    if ptr.is_null() {
        return Err(Error::NullPointer {
            name: image_component_name(index).into(),
        });
    }
    Ok(())
}

fn validate_image_component_pitch(index: usize, pitch: usize) -> Result<()> {
    if pitch == 0 {
        return Err(Error::OutOfRange {
            name: image_component_name(index).into(),
        });
    }
    Ok(())
}

const fn image_component_name(index: usize) -> &'static str {
    match index {
        0 => "component 0",
        1 => "component 1",
        2 => "component 2",
        3 => "component 3",
        _ => "component",
    }
}

/// Image to encode, in device memory, borrowed for `'a`.
#[derive(Debug)]
pub struct EncodeInput<'a> {
    pub(crate) image: Image,
    pub(crate) regions: Vec<DeviceRegion<'a>>,
    pub(crate) sizes: Vec<Size>,
    pub(crate) format: ImageFormat,
}

impl<'a> EncodeInput<'a> {
    /// Creates a planar input from one initialized plane per component.
    ///
    /// # Errors
    ///
    /// Returns an error if `C` is not between 1 and 4; the planes do not match the layout or the description's pointers, or are in different CUDA contexts; a dimension is zero; a pitch is too small; a size overflows; or a plane's memory is too small.
    pub fn planar<T: ImagePixel, const C: usize>(planes: [Plane<'a, T>; C]) -> Result<Self> {
        let image = Image::from_planes(&planes)?;
        let regions = planes.iter().map(Plane::region).collect();
        let sizes = planes.iter().map(Plane::size).collect();
        // SAFETY: initialized planes preserve the readable source borrows.
        unsafe { Self::from_raw_parts(image, regions, sizes, ImageFormat::Planar) }
    }

    /// Creates an interleaved input from one initialized plane holding `component_count` components per pixel.
    ///
    /// `plane.size()` is the image size in pixels, and each row's pitch and memory must hold `component_count` samples per pixel.
    ///
    /// # Errors
    ///
    /// Returns an error if `component_count` is not between 1 and 4; the plane does not match the layout or the description's pointers; a dimension is zero; a pitch is too small; a size overflows; or the plane's memory is too small.
    pub fn interleaved<T: ImagePixel>(plane: Plane<'a, T>, component_count: usize) -> Result<Self> {
        // SAFETY: the plane pointers and pitches describe buffers that stay borrowed for the image lifetime.
        let image = unsafe {
            Image::from_raw_parts(
                component_count,
                vec![plane.as_ptr().cast_mut().cast()],
                vec![plane.pitch()],
                T::IMAGE_TYPE,
            )?
        };
        // SAFETY: the initialized plane preserves the readable source borrow.
        unsafe {
            Self::from_raw_parts(
                image,
                vec![plane.region()],
                vec![plane.size()],
                ImageFormat::Interleaved,
            )
        }
    }

    /// Creates an input from a native description and the memory regions it points into.
    ///
    /// # Safety
    ///
    /// The regions must hold initialized pixels in the description's sample type and layout, readable for `'a`.
    /// They must stay alive until every encode that uses them finishes, even on failure and for every replay of a captured graph.
    ///
    /// # Errors
    ///
    /// Returns an error if the planes do not match the layout or the description's pointers, or are in different CUDA contexts; a dimension is zero; a pitch is too small; a size overflows; or a plane's memory is too small.
    pub unsafe fn from_raw_parts(
        image: Image,
        regions: Vec<DeviceRegion<'a>>,
        sizes: Vec<Size>,
        format: ImageFormat,
    ) -> Result<Self> {
        let expected = match format {
            ImageFormat::Planar => image.component_count(),
            ImageFormat::Interleaved => 1,
        };
        if regions.len() != expected || sizes.len() != expected || image.plane_count() != expected {
            return Err(Error::LengthMismatch {
                name: "input planes".into(),
                expected,
                actual: regions.len(),
            });
        }

        let context = regions[0].context();
        for (index, region) in regions.iter().enumerate() {
            if region.context() != context {
                return Err(CudaError::MemoryContextMismatch.into());
            }

            if region.as_ptr().cast::<c_void>() != image.component_ptr(index)? {
                return Err(Error::OutOfRange {
                    name: "input plane pointer".into(),
                });
            }

            let samples = if format == ImageFormat::Interleaved {
                image.component_count()
            } else {
                1
            };
            let row_bytes = checked_mul(
                checked_mul(sizes[index].width() as usize, samples, "input samples")?,
                image.pixel_type().element_size(),
                "input row bytes",
            )?;
            validate_plane(
                *region,
                sizes[index],
                image.pitch_in_bytes(index)?,
                row_bytes,
            )?;
        }
        Ok(Self {
            image,
            regions,
            sizes,
            format,
        })
    }

    /// Returns the memory regions of the planes.
    #[must_use]
    pub fn regions(&self) -> &[DeviceRegion<'a>] {
        &self.regions
    }

    /// Returns whether the samples are planar or interleaved.
    #[must_use]
    pub fn format(&self) -> ImageFormat {
        self.format
    }

    /// Returns the native description of the pixels.
    #[must_use]
    pub fn descriptor(&self) -> &Image {
        &self.image
    }
}

pub(crate) fn validate_plane(
    region: DeviceRegion<'_>,
    size: Size,
    pitch: usize,
    row_bytes: usize,
) -> Result<()> {
    if size.is_empty() || pitch < row_bytes {
        return Err(Error::OutOfRange {
            name: "image plane geometry".into(),
        });
    }

    let required = (size.height() as usize - 1)
        .checked_mul(pitch)
        .and_then(|prefix| prefix.checked_add(row_bytes))
        .ok_or(Error::OutOfRange {
            name: "image plane extent".into(),
        })?;
    if region.byte_len() < required {
        return Err(Error::BufferTooSmall {
            name: "image plane".into(),
            required,
            actual: region.byte_len(),
        });
    }
    Ok(())
}
