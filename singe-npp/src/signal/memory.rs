use std::{marker::PhantomData, ptr::NonNull};

use singe_cuda::types::{Complex32, Complex64};
use singe_npp_sys as sys;

use crate::{
    error::{Error, Result},
    signal::view::{SignalView, SignalViewMut},
    types::{ComplexI16, ComplexI32, ComplexI64, DataTypeLike},
    utility::to_u64,
};

/// RAII owner for signal memory allocated by NPP.
///
/// Values are created through [`Signal::create`] and released with
/// [`sys::nppsFree`] from [`Drop`].
/// The raw pointer must not be freed by callers.
#[derive(Debug)]
pub struct Signal<T> {
    ptr: NonNull<T>,
    len: usize,
    _t: PhantomData<T>,
}

impl<T> Signal<T> {
    fn create_u8(len: usize) -> Result<Signal<u8>> {
        Signal::create_with(len, sys::nppsMalloc_8u)
    }

    fn create_i8(len: usize) -> Result<Signal<i8>> {
        Signal::create_with(len, sys::nppsMalloc_8s)
    }

    fn create_u16(len: usize) -> Result<Signal<u16>> {
        Signal::create_with(len, sys::nppsMalloc_16u)
    }

    fn create_i16(len: usize) -> Result<Signal<i16>> {
        Signal::create_with(len, sys::nppsMalloc_16s)
    }

    fn create_i16_complex(len: usize) -> Result<Signal<ComplexI16>> {
        Signal::create_with(len, sys::nppsMalloc_16sc)
    }

    fn create_u32(len: usize) -> Result<Signal<u32>> {
        Signal::create_with(len, sys::nppsMalloc_32u)
    }

    fn create_i32(len: usize) -> Result<Signal<i32>> {
        Signal::create_with(len, sys::nppsMalloc_32s)
    }

    fn create_i32_complex(len: usize) -> Result<Signal<ComplexI32>> {
        Signal::create_with(len, sys::nppsMalloc_32sc)
    }

    fn create_f32(len: usize) -> Result<Signal<f32>> {
        Signal::create_with(len, sys::nppsMalloc_32f)
    }

    fn create_f32_complex(len: usize) -> Result<Signal<Complex32>> {
        let raw_len = to_u64(len, "len")?;
        let ptr = unsafe { sys::nppsMalloc_32fc(raw_len) };
        let ptr = NonNull::new(ptr as *mut Complex32).ok_or(Error::NullHandle)?;
        Ok(Signal {
            ptr,
            len,
            _t: PhantomData,
        })
    }

    fn create_i64(len: usize) -> Result<Signal<i64>> {
        Signal::create_with(len, sys::nppsMalloc_64s)
    }

    fn create_i64_complex(len: usize) -> Result<Signal<ComplexI64>> {
        Signal::create_with(len, sys::nppsMalloc_64sc)
    }

    fn create_f64(len: usize) -> Result<Signal<f64>> {
        Signal::create_with(len, sys::nppsMalloc_64f)
    }

    fn create_f64_complex(len: usize) -> Result<Signal<Complex64>> {
        let raw_len = to_u64(len, "len")?;
        let ptr = unsafe { sys::nppsMalloc_64fc(raw_len) };
        let ptr = NonNull::new(ptr as *mut Complex64).ok_or(Error::NullHandle)?;
        Ok(Signal {
            ptr,
            len,
            _t: PhantomData,
        })
    }

    fn create_with<S>(len: usize, malloc: unsafe extern "C" fn(u64) -> *mut S) -> Result<Self> {
        let raw_len = to_u64(len, "len")?;
        let ptr = unsafe { malloc(raw_len) };
        let ptr = NonNull::new(ptr as *mut T).ok_or(Error::NullHandle)?;

        Ok(Self {
            ptr,
            len,
            _t: PhantomData,
        })
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn view(&self) -> Result<SignalView<'_, T>> {
        unsafe { SignalView::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn view_mut(&mut self) -> Result<SignalViewMut<'_, T>> {
        unsafe { SignalViewMut::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub const fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }
}

impl<T> Drop for Signal<T> {
    fn drop(&mut self) {
        unsafe {
            sys::nppsFree(self.ptr.as_ptr().cast());
        }
    }
}

impl<T: SupportedSignal> Signal<T> {
    pub fn create(len: usize) -> Result<Self> {
        <T as private::SignalCreate>::create(len)
    }
}

/// Marker for signal element types that NPP can allocate.
pub trait SupportedSignal: DataTypeLike + private::SignalCreate {}

impl<T> SupportedSignal for T where T: DataTypeLike + private::SignalCreate {}

mod private {
    use super::*;

    pub trait SignalCreate: DataTypeLike {
        fn create(len: usize) -> Result<Signal<Self>>;
    }
}

macro_rules! impl_signal_create {
    ($ty:ty, $direct:ident) => {
        impl private::SignalCreate for $ty {
            fn create(len: usize) -> Result<Signal<Self>> {
                Signal::<Self>::$direct(len)
            }
        }
    };
}

impl_signal_create!(u8, create_u8);
impl_signal_create!(i8, create_i8);
impl_signal_create!(u16, create_u16);
impl_signal_create!(i16, create_i16);
impl_signal_create!(ComplexI16, create_i16_complex);
impl_signal_create!(u32, create_u32);
impl_signal_create!(i32, create_i32);
impl_signal_create!(ComplexI32, create_i32_complex);
impl_signal_create!(f32, create_f32);
impl_signal_create!(Complex32, create_f32_complex);
impl_signal_create!(i64, create_i64);
impl_signal_create!(ComplexI64, create_i64_complex);
impl_signal_create!(f64, create_f64);
impl_signal_create!(Complex64, create_f64_complex);

pub fn create<T: SupportedSignal>(len: usize) -> Result<Signal<T>> {
    Signal::<T>::create(len)
}
