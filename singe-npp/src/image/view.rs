use std::{marker::PhantomData, ptr::NonNull};

use singe_cuda::memory::DeviceMemory;
use singe_npp_sys as sys;

use crate::{
    error::{Error, Result},
    types::Size,
    utility::{checked_len, checked_step, ensure_positive, to_usize},
};

pub trait ChannelLayout: Copy + 'static {
    const CHANNELS: usize;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct C1;

impl ChannelLayout for C1 {
    const CHANNELS: usize = 1;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct C2;

impl ChannelLayout for C2 {
    const CHANNELS: usize = 2;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct C3;

impl ChannelLayout for C3 {
    const CHANNELS: usize = 3;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct C4;

impl ChannelLayout for C4 {
    const CHANNELS: usize = 4;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AC4;

impl ChannelLayout for AC4 {
    const CHANNELS: usize = 4;
}

pub type Gray = C1;
pub type Channels2 = C2;
pub type Rgb = C3;
pub type Rgba = C4;
pub type AlphaIgnoredRgba = AC4;

pub type MaskView<'a> = ImageView<'a, u8, C1>;
pub type MaskViewMut<'a> = ImageViewMut<'a, u8, C1>;

#[derive(Debug, Clone, Copy)]
pub struct ImageView<'a, T, L = C1> {
    ptr: NonNull<T>,
    size: Size,
    step: i32,
    _t: PhantomData<(&'a T, L)>,
}

impl<'a, T, L> ImageView<'a, T, L>
where
    L: ChannelLayout,
{
    pub fn from_memory(memory: &'a DeviceMemory<T>, size: Size) -> Result<Self> {
        let len = checked_len(size, L::CHANNELS)?;
        if len > memory.len() {
            return Err(Error::LengthMismatch {
                name: "image memory".into(),
                expected: len,
                actual: memory.len(),
            });
        }

        Self::from_memory_with_step(
            memory,
            size,
            checked_step(size.width, L::CHANNELS, size_of::<T>())?,
        )
    }

    pub fn from_memory_with_step(
        memory: &'a DeviceMemory<T>,
        size: Size,
        step: i32,
    ) -> Result<Self> {
        validate_image_shape::<T, L>(size, step)?;
        validate_image_memory::<T, L>(memory.len(), size, step)?;

        Ok(Self {
            ptr: NonNull::new(memory.as_ptr().cast_mut()).ok_or(Error::NullHandle)?,
            size,
            step,
            _t: PhantomData,
        })
    }

    /// # Safety
    ///
    /// `ptr` must be non-null CUDA device memory aligned for `T` that remains valid for `'a`.
    /// The allocation must cover `height - 1` full byte steps plus the final
    /// ROI row of `width * L::CHANNELS * size_of::<T>()` bytes. `step` is a
    /// byte pitch, not an element count, and must describe the actual row
    /// spacing visible to NPP.
    ///
    /// # Errors
    ///
    /// Returns an error if the image shape is invalid or `ptr` is null.
    pub unsafe fn from_raw_parts(ptr: *const T, size: Size, step: i32) -> Result<Self> {
        validate_image_shape::<T, L>(size, step)?;

        Ok(Self {
            ptr: NonNull::new(ptr.cast_mut()).ok_or(Error::NullHandle)?,
            size,
            step,
            _t: PhantomData,
        })
    }

    pub const fn size(&self) -> Size {
        self.size
    }

    pub const fn step(&self) -> i32 {
        self.step
    }

    pub fn byte_len(&self) -> Result<usize> {
        image_byte_len::<T, L>(self.size, self.step)
    }

    pub(crate) const fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    pub(crate) fn descriptor(&self) -> sys::NppiImageDescriptor {
        sys::NppiImageDescriptor {
            pData: self.as_ptr().cast_mut().cast(),
            nStep: self.step,
            oSize: self.size.into(),
        }
    }
}

#[derive(Debug)]
pub struct ImageViewMut<'a, T, L = C1> {
    ptr: NonNull<T>,
    size: Size,
    step: i32,
    _marker: PhantomData<(&'a mut T, L)>,
}

impl<'a, T, L> ImageViewMut<'a, T, L>
where
    L: ChannelLayout,
{
    pub fn from_memory(memory: &'a mut DeviceMemory<T>, size: Size) -> Result<Self> {
        let len = checked_len(size, L::CHANNELS)?;
        if len > memory.len() {
            return Err(Error::LengthMismatch {
                name: "image memory".into(),
                expected: len,
                actual: memory.len(),
            });
        }

        Self::from_memory_with_step(
            memory,
            size,
            checked_step(size.width, L::CHANNELS, size_of::<T>())?,
        )
    }

    pub fn from_memory_with_step(
        memory: &'a mut DeviceMemory<T>,
        size: Size,
        step: i32,
    ) -> Result<Self> {
        validate_image_shape::<T, L>(size, step)?;
        validate_image_memory::<T, L>(memory.len(), size, step)?;

        Ok(Self {
            ptr: NonNull::new(memory.as_mut_ptr()).ok_or(Error::NullHandle)?,
            size,
            step,
            _marker: PhantomData,
        })
    }

    /// # Safety
    ///
    /// `ptr` must be non-null CUDA device memory aligned for `T` that remains valid for `'a`.
    /// The allocation must cover `height - 1` full byte steps plus the final
    /// ROI row of `width * L::CHANNELS * size_of::<T>()` bytes. `step` is a
    /// byte pitch, not an element count, and must describe the actual row
    /// spacing visible to NPP. The pointed-to ROI must be uniquely writable for
    /// the returned view lifetime.
    ///
    /// # Errors
    ///
    /// Returns an error if the image shape is invalid or `ptr` is null.
    pub unsafe fn from_raw_parts(ptr: *mut T, size: Size, step: i32) -> Result<Self> {
        validate_image_shape::<T, L>(size, step)?;

        Ok(Self {
            ptr: NonNull::new(ptr).ok_or(Error::NullHandle)?,
            size,
            step,
            _marker: PhantomData,
        })
    }

    pub const fn size(&self) -> Size {
        self.size
    }

    pub const fn step(&self) -> i32 {
        self.step
    }

    pub fn byte_len(&self) -> Result<usize> {
        image_byte_len::<T, L>(self.size, self.step)
    }

    pub(crate) const fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    pub(crate) fn descriptor(&mut self) -> sys::NppiImageDescriptor {
        sys::NppiImageDescriptor {
            pData: self.as_mut_ptr().cast(),
            nStep: self.step,
            oSize: self.size.into(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PlanarImageView<'a, T, const C: usize> {
    planes: [ImageView<'a, T, C1>; C],
}

impl<'a, T, const C: usize> PlanarImageView<'a, T, C> {
    pub fn create(planes: [ImageView<'a, T, C1>; C]) -> Result<Self> {
        validate_plane_count(C)?;
        validate_same_planes(&planes)?;
        Ok(Self { planes })
    }

    pub const fn planes(&self) -> &[ImageView<'a, T, C1>; C] {
        &self.planes
    }
}

#[derive(Debug)]
pub struct PlanarImageViewMut<'a, T, const C: usize> {
    planes: [ImageViewMut<'a, T, C1>; C],
}

impl<'a, T, const C: usize> PlanarImageViewMut<'a, T, C> {
    pub fn create(planes: [ImageViewMut<'a, T, C1>; C]) -> Result<Self> {
        validate_plane_count(C)?;
        validate_same_planes(&planes)?;
        Ok(Self { planes })
    }

    pub const fn planes(&self) -> &[ImageViewMut<'a, T, C1>; C] {
        &self.planes
    }

    pub fn planes_mut(&mut self) -> &mut [ImageViewMut<'a, T, C1>; C] {
        &mut self.planes
    }
}

fn validate_image_shape<T, L>(size: Size, step: i32) -> Result<()>
where
    L: ChannelLayout,
{
    size.validate()?;

    let element_size = size_of::<T>();
    if element_size == 0 {
        return Err(Error::OutOfRange {
            name: "element size".into(),
        });
    }

    let minimum_step = checked_step(size.width, L::CHANNELS, element_size)?;
    if step < minimum_step {
        return Err(Error::OutOfRange {
            name: "step".into(),
        });
    }

    Ok(())
}

fn validate_image_memory<T, L>(memory_len: usize, size: Size, step: i32) -> Result<()>
where
    L: ChannelLayout,
{
    let element_size = size_of::<T>();
    let required_bytes = image_byte_len::<T, L>(size, step)?;
    let available_bytes = memory_len
        .checked_mul(element_size)
        .ok_or_else(|| Error::OutOfRange { name: "len".into() })?;

    if required_bytes > available_bytes {
        return Err(Error::LengthMismatch {
            name: "image memory".into(),
            expected: required_bytes,
            actual: available_bytes,
        });
    }

    Ok(())
}

fn image_byte_len<T, L>(size: Size, step: i32) -> Result<usize>
where
    L: ChannelLayout,
{
    size.validate()?;
    ensure_positive(step, "step")?;

    let row_bytes = (size.width as usize)
        .checked_mul(L::CHANNELS)
        .and_then(|value| value.checked_mul(size_of::<T>()))
        .ok_or_else(|| Error::OutOfRange {
            name: "byte_len".into(),
        })?;
    (size.height as usize - 1)
        .checked_mul(step as usize)
        .and_then(|value| value.checked_add(row_bytes))
        .ok_or_else(|| Error::OutOfRange {
            name: "byte_len".into(),
        })
}

fn validate_plane_count(count: usize) -> Result<()> {
    match count {
        2..=4 => Ok(()),
        _ => Err(Error::OutOfRange {
            name: "plane count".into(),
        }),
    }
}

fn validate_same_planes<T, P, const C: usize>(planes: &[P; C]) -> Result<()>
where
    P: PlaneInfo<T>,
{
    let Some(first) = planes.first() else {
        return Ok(());
    };

    for plane in planes.iter().skip(1) {
        if plane.size() != first.size() {
            let expected = (first.size().width as usize)
                .checked_mul(first.size().height as usize)
                .ok_or_else(|| Error::OutOfRange { name: "len".into() })?;
            let actual = (plane.size().width as usize)
                .checked_mul(plane.size().height as usize)
                .ok_or_else(|| Error::OutOfRange { name: "len".into() })?;
            return Err(Error::LengthMismatch {
                name: "plane size".into(),
                expected,
                actual,
            });
        }

        if plane.step() != first.step() {
            return Err(Error::LengthMismatch {
                name: "plane step".into(),
                expected: to_usize(first.step(), "plane step")?,
                actual: to_usize(plane.step(), "plane step")?,
            });
        }
    }

    Ok(())
}

trait PlaneInfo<T> {
    fn size(&self) -> Size;
    fn step(&self) -> i32;
}

impl<T> PlaneInfo<T> for ImageView<'_, T, C1> {
    fn size(&self) -> Size {
        self.size()
    }

    fn step(&self) -> i32 {
        self.step()
    }
}

impl<T> PlaneInfo<T> for ImageViewMut<'_, T, C1> {
    fn size(&self) -> Size {
        self.size()
    }

    fn step(&self) -> i32 {
        self.step()
    }
}

pub(crate) fn image_descriptors<T, L>(
    images: &[ImageView<'_, T, L>],
) -> Vec<sys::NppiImageDescriptor>
where
    L: ChannelLayout,
{
    images.iter().map(ImageView::descriptor).collect()
}

pub(crate) fn image_descriptors_mut<T, L>(
    images: &mut [ImageViewMut<'_, T, L>],
) -> Vec<sys::NppiImageDescriptor>
where
    L: ChannelLayout,
{
    images.iter_mut().map(ImageViewMut::descriptor).collect()
}
