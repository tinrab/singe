use std::{marker::PhantomData, ptr::NonNull};

use singe_cuda::{
    memory::{DeviceMemory, MemoryCopyKind},
    types::{Complex32, f16},
};
use singe_npp_sys as sys;

use crate::{
    error::{Error, Result},
    image::view::{AC4, C1, C2, C3, C4, ChannelLayout, ImageView, ImageViewMut},
    types::{ComplexI16, ComplexI32, DataTypeLike, Size},
    utility::checked_len,
};

/// RAII owner for pitched image memory allocated by NPP.
///
/// Values are created through [`Image::create`] and released with
/// [`sys::nppiFree`] from [`Drop`].
/// The raw pointer must not be freed by callers.
/// `step` is the NPP row pitch in bytes and can be larger than the logical ROI
/// row size.
#[derive(Debug)]
pub struct Image<T, L = C1> {
    ptr: *mut T,
    size: Size,
    step: i32,
    _t: PhantomData<L>,
}

/// Marker for image element/layout combinations that NPP can allocate.
pub trait SupportedImage<Layout>: DataTypeLike + private::ImageCreate<Layout>
where
    Layout: ChannelLayout,
{
}

impl<T, Layout> SupportedImage<Layout> for T
where
    T: DataTypeLike + private::ImageCreate<Layout>,
    Layout: ChannelLayout,
{
}

mod private {
    use super::*;

    pub trait ImageCreate<Layout>: DataTypeLike + Sized
    where
        Layout: ChannelLayout,
    {
        fn create(size: Size) -> Result<Image<Self, Layout>>;
    }
}

macro_rules! impl_image_create {
    ($layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        $(
            impl private::ImageCreate<$layout> for $ty {
                fn create(size: Size) -> Result<Image<Self, $layout>> {
                    Image::<Self, $layout>::$direct(size)
                }
            }
        )*
    };
}

impl<T, L> Image<T, L>
where
    L: ChannelLayout,
{
    pub fn create(size: Size) -> Result<Self>
    where
        T: SupportedImage<L>,
    {
        <T as private::ImageCreate<L>>::create(size)
    }

    fn create_u8_c1(size: Size) -> Result<Image<u8, C1>> {
        Image::create_with(size, sys::nppiMalloc_8u_C1)
    }

    fn create_u8_c2(size: Size) -> Result<Image<u8, C2>> {
        Image::create_with(size, sys::nppiMalloc_8u_C2)
    }

    fn create_u8_c3(size: Size) -> Result<Image<u8, C3>> {
        Image::create_with(size, sys::nppiMalloc_8u_C3)
    }

    fn create_u8_c4(size: Size) -> Result<Image<u8, C4>> {
        Image::create_with(size, sys::nppiMalloc_8u_C4)
    }

    fn create_u8_ac4(size: Size) -> Result<Image<u8, AC4>> {
        Image::create_with(size, sys::nppiMalloc_8u_C4)
    }

    fn create_i8_c1(size: Size) -> Result<Image<i8, C1>> {
        Image::<i8, C1>::create_signed_8(size, sys::nppiMalloc_8u_C1)
    }

    fn create_i8_c2(size: Size) -> Result<Image<i8, C2>> {
        Image::<i8, C2>::create_signed_8(size, sys::nppiMalloc_8u_C2)
    }

    fn create_i8_c3(size: Size) -> Result<Image<i8, C3>> {
        Image::<i8, C3>::create_signed_8(size, sys::nppiMalloc_8u_C3)
    }

    fn create_i8_c4(size: Size) -> Result<Image<i8, C4>> {
        Image::<i8, C4>::create_signed_8(size, sys::nppiMalloc_8u_C4)
    }

    fn create_i8_ac4(size: Size) -> Result<Image<i8, AC4>> {
        Image::<i8, AC4>::create_signed_8(size, sys::nppiMalloc_8u_C4)
    }

    fn create_u16_c1(size: Size) -> Result<Image<u16, C1>> {
        Image::create_with(size, sys::nppiMalloc_16u_C1)
    }

    fn create_u16_c2(size: Size) -> Result<Image<u16, C2>> {
        Image::create_with(size, sys::nppiMalloc_16u_C2)
    }

    fn create_u16_c3(size: Size) -> Result<Image<u16, C3>> {
        Image::create_with(size, sys::nppiMalloc_16u_C3)
    }

    fn create_u16_c4(size: Size) -> Result<Image<u16, C4>> {
        Image::create_with(size, sys::nppiMalloc_16u_C4)
    }

    fn create_u16_ac4(size: Size) -> Result<Image<u16, AC4>> {
        Image::create_with(size, sys::nppiMalloc_16u_C4)
    }

    fn create_f16_c1(size: Size) -> Result<Image<f16, C1>> {
        Image::<f16, C1>::create_f16(size, sys::nppiMalloc_16u_C1)
    }

    fn create_f16_c2(size: Size) -> Result<Image<f16, C2>> {
        Image::<f16, C2>::create_f16(size, sys::nppiMalloc_16u_C2)
    }

    fn create_f16_c3(size: Size) -> Result<Image<f16, C3>> {
        Image::<f16, C3>::create_f16(size, sys::nppiMalloc_16u_C3)
    }

    fn create_f16_c4(size: Size) -> Result<Image<f16, C4>> {
        Image::<f16, C4>::create_f16(size, sys::nppiMalloc_16u_C4)
    }

    fn create_f16_ac4(size: Size) -> Result<Image<f16, AC4>> {
        Image::<f16, AC4>::create_f16(size, sys::nppiMalloc_16u_C4)
    }

    fn create_i16_c1(size: Size) -> Result<Image<i16, C1>> {
        Image::create_with(size, sys::nppiMalloc_16s_C1)
    }

    fn create_i16_c2(size: Size) -> Result<Image<i16, C2>> {
        Image::create_with(size, sys::nppiMalloc_16s_C2)
    }

    fn create_i16_c3(size: Size) -> Result<Image<i16, C3>> {
        Image::<i16, C3>::create_signed_16(size, sys::nppiMalloc_16u_C3)
    }

    fn create_i16_c4(size: Size) -> Result<Image<i16, C4>> {
        Image::create_with(size, sys::nppiMalloc_16s_C4)
    }

    fn create_i16_ac4(size: Size) -> Result<Image<i16, AC4>> {
        Image::create_with(size, sys::nppiMalloc_16s_C4)
    }

    fn create_i16_complex_c1(size: Size) -> Result<Image<ComplexI16, C1>> {
        Image::create_with(size, sys::nppiMalloc_16sc_C1)
    }

    fn create_i16_complex_c2(size: Size) -> Result<Image<ComplexI16, C2>> {
        Image::create_with(size, sys::nppiMalloc_16sc_C2)
    }

    fn create_i16_complex_c3(size: Size) -> Result<Image<ComplexI16, C3>> {
        Image::create_with(size, sys::nppiMalloc_16sc_C3)
    }

    fn create_i16_complex_c4(size: Size) -> Result<Image<ComplexI16, C4>> {
        Image::create_with(size, sys::nppiMalloc_16sc_C4)
    }

    fn create_i16_complex_ac4(size: Size) -> Result<Image<ComplexI16, AC4>> {
        Image::create_with(size, sys::nppiMalloc_16sc_C4)
    }

    fn create_i32_c1(size: Size) -> Result<Image<i32, C1>> {
        Image::create_with(size, sys::nppiMalloc_32s_C1)
    }

    fn create_u32_c1(size: Size) -> Result<Image<u32, C1>> {
        size.validate()?;
        let mut step = 0;
        let ptr = unsafe { sys::nppiMalloc_32s_C1(size.width, size.height, &raw mut step) };
        let ptr = NonNull::new(ptr.cast()).ok_or(Error::NullHandle)?;

        Ok(Image {
            ptr: ptr.as_ptr(),
            size,
            step,
            _t: PhantomData,
        })
    }

    fn create_i32_c3(size: Size) -> Result<Image<i32, C3>> {
        Image::create_with(size, sys::nppiMalloc_32s_C3)
    }

    fn create_i32_c4(size: Size) -> Result<Image<i32, C4>> {
        Image::create_with(size, sys::nppiMalloc_32s_C4)
    }

    fn create_i32_ac4(size: Size) -> Result<Image<i32, AC4>> {
        Image::create_with(size, sys::nppiMalloc_32s_C4)
    }

    fn create_u32_ac4(size: Size) -> Result<Image<u32, AC4>> {
        size.validate()?;
        let mut step = 0;
        let ptr = unsafe { sys::nppiMalloc_32s_C4(size.width, size.height, &raw mut step) };
        let ptr = NonNull::new(ptr.cast()).ok_or(Error::NullHandle)?;

        Ok(Image {
            ptr: ptr.as_ptr(),
            size,
            step,
            _t: PhantomData,
        })
    }

    fn create_i32_complex_c1(size: Size) -> Result<Image<ComplexI32, C1>> {
        Image::create_with(size, sys::nppiMalloc_32sc_C1)
    }

    fn create_i32_complex_c2(size: Size) -> Result<Image<ComplexI32, C2>> {
        Image::create_with(size, sys::nppiMalloc_32sc_C2)
    }

    fn create_i32_complex_c3(size: Size) -> Result<Image<ComplexI32, C3>> {
        Image::create_with(size, sys::nppiMalloc_32sc_C3)
    }

    fn create_i32_complex_c4(size: Size) -> Result<Image<ComplexI32, C4>> {
        Image::create_with(size, sys::nppiMalloc_32sc_C4)
    }

    fn create_i32_complex_ac4(size: Size) -> Result<Image<ComplexI32, AC4>> {
        Image::create_with(size, sys::nppiMalloc_32sc_C4)
    }

    fn create_f32_c1(size: Size) -> Result<Image<f32, C1>> {
        Image::create_with(size, sys::nppiMalloc_32f_C1)
    }

    fn create_f32_c2(size: Size) -> Result<Image<f32, C2>> {
        Image::create_with(size, sys::nppiMalloc_32f_C2)
    }

    fn create_f32_c3(size: Size) -> Result<Image<f32, C3>> {
        Image::create_with(size, sys::nppiMalloc_32f_C3)
    }

    fn create_f32_c4(size: Size) -> Result<Image<f32, C4>> {
        Image::create_with(size, sys::nppiMalloc_32f_C4)
    }

    fn create_f32_ac4(size: Size) -> Result<Image<f32, AC4>> {
        Image::create_with(size, sys::nppiMalloc_32f_C4)
    }

    fn create_f32_complex_c1(size: Size) -> Result<Image<Complex32, C1>> {
        Image::create_with(size, sys::nppiMalloc_32fc_C1)
    }

    fn create_f32_complex_c2(size: Size) -> Result<Image<Complex32, C2>> {
        Image::create_with(size, sys::nppiMalloc_32fc_C2)
    }

    fn create_f32_complex_c3(size: Size) -> Result<Image<Complex32, C3>> {
        Image::create_with(size, sys::nppiMalloc_32fc_C3)
    }

    fn create_f32_complex_c4(size: Size) -> Result<Image<Complex32, C4>> {
        Image::create_with(size, sys::nppiMalloc_32fc_C4)
    }

    fn create_f32_complex_ac4(size: Size) -> Result<Image<Complex32, AC4>> {
        Image::create_with(size, sys::nppiMalloc_32fc_C4)
    }

    fn create_with<S>(
        size: Size,
        malloc: unsafe extern "C" fn(i32, i32, *mut i32) -> *mut S,
    ) -> Result<Self> {
        size.validate()?;
        let mut step = 0;
        let ptr = unsafe { malloc(size.width, size.height, &raw mut step) };
        let ptr = NonNull::new(ptr.cast()).ok_or(Error::NullHandle)?;

        Ok(Self {
            ptr: ptr.as_ptr(),
            size,
            step,
            _t: PhantomData,
        })
    }

    fn create_signed_8(
        size: Size,
        malloc: unsafe extern "C" fn(i32, i32, *mut i32) -> *mut u8,
    ) -> Result<Image<i8, L>> {
        size.validate()?;
        let mut step = 0;
        let ptr = unsafe { malloc(size.width, size.height, &raw mut step) };
        let ptr = NonNull::new(ptr.cast()).ok_or(Error::NullHandle)?;

        Ok(Image {
            ptr: ptr.as_ptr(),
            size,
            step,
            _t: PhantomData,
        })
    }

    fn create_signed_16(
        size: Size,
        malloc: unsafe extern "C" fn(i32, i32, *mut i32) -> *mut u16,
    ) -> Result<Image<i16, L>> {
        size.validate()?;
        let mut step = 0;
        let ptr = unsafe { malloc(size.width, size.height, &raw mut step) };
        let ptr = NonNull::new(ptr.cast()).ok_or(Error::NullHandle)?;

        Ok(Image {
            ptr: ptr.as_ptr(),
            size,
            step,
            _t: PhantomData,
        })
    }

    fn create_f16(
        size: Size,
        malloc: unsafe extern "C" fn(i32, i32, *mut i32) -> *mut u16,
    ) -> Result<Image<f16, L>> {
        size.validate()?;
        let mut step = 0;
        let ptr = unsafe { malloc(size.width, size.height, &raw mut step) };
        let ptr = NonNull::new(ptr.cast()).ok_or(Error::NullHandle)?;

        Ok(Image {
            ptr: ptr.as_ptr(),
            size,
            step,
            _t: PhantomData,
        })
    }

    pub fn copy_to_device_memory(&self) -> Result<DeviceMemory<T>> {
        let len = checked_len(self.size, L::CHANNELS)?;
        let mut destination = DeviceMemory::create(len)?;
        self.copy_into_device_memory(&mut destination)?;
        Ok(destination)
    }

    pub fn copy_to_host_vec(&self) -> Result<Vec<T>> {
        let len = checked_len(self.size, L::CHANNELS)?;
        if len == 0 {
            return Ok(Vec::new());
        }

        let mut host = Vec::<T>::with_capacity(len);
        unsafe {
            self.copy_rows_to(
                host.as_mut_ptr().cast(),
                row_bytes::<T, L>(self.size)?,
                MemoryCopyKind::DeviceToHost,
            )?;
            host.set_len(len);
        }
        Ok(host)
    }

    pub fn copy_into_device_memory(&self, destination: &mut DeviceMemory<T>) -> Result<()> {
        let len = checked_len(self.size, L::CHANNELS)?;
        if destination.len() != len {
            return Err(Error::LengthMismatch {
                name: "image memory".into(),
                expected: len,
                actual: destination.len(),
            });
        }
        if len == 0 {
            return Ok(());
        }

        unsafe {
            self.copy_rows_to(
                destination.as_mut_ptr().cast(),
                row_bytes::<T, L>(self.size)?,
                MemoryCopyKind::DeviceToDevice,
            )?;
        }
        Ok(())
    }

    /// Returns the image ROI size requested at allocation time.
    pub const fn size(&self) -> Size {
        self.size
    }

    /// Returns the NPP row pitch in bytes.
    pub const fn step(&self) -> i32 {
        self.step
    }

    /// Borrows this allocation as an immutable NPP image view.
    ///
    /// # Errors
    ///
    /// Returns an error if the stored size or pitch is not valid for an NPP image
    /// view.
    pub fn view(&self) -> Result<ImageView<'_, T, L>> {
        unsafe { ImageView::from_raw_parts(self.ptr.cast(), self.size, self.step) }
    }

    /// Borrows this allocation as a mutable NPP image view.
    ///
    /// # Errors
    ///
    /// Returns an error if the stored size or pitch is not valid for an NPP image
    /// view.
    pub fn view_mut(&mut self) -> Result<ImageViewMut<'_, T, L>> {
        unsafe { ImageViewMut::from_raw_parts(self.ptr.cast(), self.size, self.step) }
    }

    /// Returns the raw device pointer owned by this allocation.
    ///
    /// The pointer remains owned by `self` and is freed with `nppiFree` when the
    /// `Image` is dropped.
    pub const fn as_ptr(&self) -> *const T {
        self.ptr as _
    }

    /// Returns the mutable raw device pointer owned by this allocation.
    ///
    /// The pointer remains owned by `self` and is freed with `nppiFree` when the
    /// `Image` is dropped.
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    unsafe fn copy_rows_to(
        &self,
        destination: *mut u8,
        row_bytes: usize,
        kind: MemoryCopyKind,
    ) -> Result<()> {
        let height = self.size.height as usize;
        let source_step = self.step as usize;

        if source_step == row_bytes {
            unsafe {
                DeviceMemory::<u8>::copy(
                    destination,
                    self.ptr.cast(),
                    row_bytes
                        .checked_mul(height)
                        .ok_or_else(|| Error::OutOfRange { name: "len".into() })?,
                    kind,
                )?;
            }
            return Ok(());
        }

        for row in 0..height {
            unsafe {
                DeviceMemory::<u8>::copy(
                    destination.add(
                        row.checked_mul(row_bytes)
                            .ok_or_else(|| Error::OutOfRange { name: "len".into() })?,
                    ),
                    self.ptr.cast::<u8>().add(
                        row.checked_mul(source_step)
                            .ok_or_else(|| Error::OutOfRange { name: "len".into() })?,
                    ),
                    row_bytes,
                    kind,
                )?;
            }
        }

        Ok(())
    }
}

impl_image_create!(C1, [
    u8 => create_u8_c1,
    i8 => create_i8_c1,
    u16 => create_u16_c1,
    f16 => create_f16_c1,
    i16 => create_i16_c1,
    ComplexI16 => create_i16_complex_c1,
    i32 => create_i32_c1,
    u32 => create_u32_c1,
    ComplexI32 => create_i32_complex_c1,
    f32 => create_f32_c1,
    Complex32 => create_f32_complex_c1,
]);
impl_image_create!(C2, [
    u8 => create_u8_c2,
    i8 => create_i8_c2,
    u16 => create_u16_c2,
    f16 => create_f16_c2,
    i16 => create_i16_c2,
    ComplexI16 => create_i16_complex_c2,
    ComplexI32 => create_i32_complex_c2,
    f32 => create_f32_c2,
    Complex32 => create_f32_complex_c2,
]);
impl_image_create!(C3, [
    u8 => create_u8_c3,
    i8 => create_i8_c3,
    u16 => create_u16_c3,
    f16 => create_f16_c3,
    i16 => create_i16_c3,
    ComplexI16 => create_i16_complex_c3,
    i32 => create_i32_c3,
    ComplexI32 => create_i32_complex_c3,
    f32 => create_f32_c3,
    Complex32 => create_f32_complex_c3,
]);
impl_image_create!(C4, [
    u8 => create_u8_c4,
    i8 => create_i8_c4,
    u16 => create_u16_c4,
    f16 => create_f16_c4,
    i16 => create_i16_c4,
    ComplexI16 => create_i16_complex_c4,
    i32 => create_i32_c4,
    ComplexI32 => create_i32_complex_c4,
    f32 => create_f32_c4,
    Complex32 => create_f32_complex_c4,
]);
impl_image_create!(AC4, [
    u8 => create_u8_ac4,
    i8 => create_i8_ac4,
    u16 => create_u16_ac4,
    f16 => create_f16_ac4,
    i16 => create_i16_ac4,
    ComplexI16 => create_i16_complex_ac4,
    i32 => create_i32_ac4,
    u32 => create_u32_ac4,
    ComplexI32 => create_i32_complex_ac4,
    f32 => create_f32_ac4,
    Complex32 => create_f32_complex_ac4,
]);

pub fn create<T, L>(size: Size) -> Result<Image<T, L>>
where
    T: SupportedImage<L>,
    L: ChannelLayout,
{
    Image::<T, L>::create(size)
}

impl<T, L> Drop for Image<T, L> {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                sys::nppiFree(self.ptr.cast());
            }
        }
    }
}

fn row_bytes<T, L>(size: Size) -> Result<usize>
where
    L: ChannelLayout,
{
    let element_size = size_of::<T>();
    if element_size == 0 {
        return Err(Error::OutOfRange {
            name: "element size".into(),
        });
    }

    (size.width as usize)
        .checked_mul(L::CHANNELS)
        .and_then(|value| value.checked_mul(element_size))
        .ok_or_else(|| Error::OutOfRange { name: "len".into() })
}
