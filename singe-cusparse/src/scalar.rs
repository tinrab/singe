//! Host values and borrowed device scalar regions for raw cuSPARSE calls.

use std::{mem::MaybeUninit, ptr, sync::Arc};

use singe_cuda::{
    context::Context as CudaContext,
    types::DevicePtr,
    view::{
        DeviceInput, DeviceInputView, DeviceOutput, DeviceOutputView, DeviceRegion,
        DeviceRegionProvider, DeviceRepr,
    },
};

use crate::{
    error::{Error, Result},
    types::PointerMode,
};

/// Host scalar value or borrowed device scalar input metadata.
///
/// Execution helpers select the handle's pointer mode from the scalar location.
/// An input scalar pair must use matching host or device locations.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum Scalar<'a, T: DeviceRepr> {
    /// Borrowed initialized host scalar.
    Host(&'a T),
    /// Borrowed device scalar metadata; producers must precede native reads.
    Device(DeviceScalar<'a, T>),
}

/// Host or device storage for a scalar result.
///
/// Execution helpers select the handle's output pointer mode from this location.
#[derive(Debug)]
#[non_exhaustive]
pub enum ScalarMut<'a, T: DeviceRepr> {
    /// Exclusively borrowed initialized host scalar to overwrite.
    Host(&'a mut T),
    /// Exclusively borrowed host storage that may be uninitialized.
    HostOutput(&'a mut MaybeUninit<T>),
    /// Exclusively borrowed device output storage.
    Device(DeviceScalarMut<'a, T>),
}

/// One borrowed device input element, whose producer must precede native reads.
#[derive(Debug, Clone, Copy)]
pub struct DeviceScalar<'a, T: DeviceRepr> {
    view: DeviceInputView<'a, T>,
}

/// One exclusive device output element, which may still be uninitialized.
#[derive(Debug)]
pub struct DeviceScalarMut<'a, T: DeviceRepr> {
    view: DeviceOutputView<'a, T>,
}

impl<'a, T: DeviceRepr> Scalar<'a, T> {
    /// Borrows an initialized host scalar without copying it.
    pub const fn host(value: &'a T) -> Self {
        let () = T::LAYOUT;
        Self::Host(value)
    }

    /// Borrows input metadata for the first device element.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer contains no elements or the first-element region cannot be borrowed.
    pub fn device(buffer: &'a (impl DeviceInput<T> + ?Sized)) -> Result<Self> {
        Ok(Self::Device(DeviceScalar::create(buffer)?))
    }

    pub(crate) const fn pointer_mode(self) -> PointerMode {
        match self {
            Self::Host(_) => PointerMode::Host,
            Self::Device(_) => PointerMode::Device,
        }
    }

    pub(crate) fn ptr(self) -> *const T {
        let () = T::LAYOUT;
        match self {
            Self::Host(v) => ptr::from_ref(v),
            Self::Device(v) => v.as_ptr(),
        }
    }

    pub(crate) fn validate_context(self, context: &Arc<CudaContext>) -> Result<()> {
        if let Self::Device(v) = self
            && v.context() != context
        {
            return Err(Error::PlanContextMismatch);
        }
        Ok(())
    }
}

impl<'a, T: DeviceRepr> ScalarMut<'a, T> {
    /// Exclusively borrows a host scalar for native output.
    pub const fn host(value: &'a mut T) -> Self {
        let () = T::LAYOUT;
        Self::Host(value)
    }

    /// Exclusively borrows possibly uninitialized host storage for native output.
    pub const fn host_output(value: &'a mut MaybeUninit<T>) -> Self {
        let () = T::LAYOUT;
        Self::HostOutput(value)
    }

    /// Exclusively borrows the first device element as scalar output storage.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer contains no elements or the first-element region cannot be borrowed.
    pub fn device(buffer: &'a mut (impl DeviceOutput<T> + ?Sized)) -> Result<Self> {
        Ok(Self::Device(DeviceScalarMut::create(buffer)?))
    }

    pub(crate) const fn pointer_mode(&self) -> PointerMode {
        match self {
            Self::Host(_) | Self::HostOutput(_) => PointerMode::Host,
            Self::Device(_) => PointerMode::Device,
        }
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut T {
        let () = T::LAYOUT;
        match self {
            Self::Host(v) => ptr::from_mut(*v),
            Self::HostOutput(v) => v.as_mut_ptr(),
            Self::Device(v) => v.as_mut_ptr(),
        }
    }

    pub(crate) fn validate_context(&self, context: &Arc<CudaContext>) -> Result<()> {
        if let Self::Device(v) = self
            && v.context() != context
        {
            return Err(Error::PlanContextMismatch);
        }
        Ok(())
    }
}

impl<'a, T: DeviceRepr> DeviceScalar<'a, T> {
    /// Borrows input metadata for the first device element without establishing initialization.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer contains no elements or the first-element region cannot be borrowed.
    pub fn create(buffer: &'a (impl DeviceInput<T> + ?Sized)) -> Result<Self> {
        if buffer.len() == 0 {
            return Err(Error::LengthMismatch {
                name: "device scalar".into(),
                expected: 1,
                actual: 0,
            });
        }
        Ok(Self {
            view: DeviceInputView::from_buffer(buffer).slice(..1)?,
        })
    }

    /// Describes a scalar in externally established live allocation metadata.
    /// This does not establish initialized input values or producer ordering.
    ///
    /// # Errors
    ///
    /// Returns an error if `T` is zero-sized, the region's byte count is not a multiple of its size, a nonempty region is misaligned, or the region contains fewer than one element.
    pub fn from_region(region: DeviceRegion<'a>) -> Result<Self> {
        Ok(Self {
            view: DeviceInputView::create(region)?.slice(..1)?,
        })
    }

    /// Returns the scalar's untyped CUDA-accessible address.
    #[must_use]
    pub fn as_device_ptr(&self) -> DevicePtr {
        // SAFETY: the address belongs to a live device allocation that stays borrowed for the operation that uses it.
        unsafe { DevicePtr::from_raw(self.as_ptr().cast_mut().cast()) }
    }

    /// Returns the typed CUDA-accessible input pointer without synchronizing producers.
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.view.as_ptr()
    }

    /// Returns the scalar region's actual CUDA context.
    #[must_use]
    pub fn context(&self) -> &Arc<CudaContext> {
        self.view.context()
    }

    /// Returns the exact byte range and allocation identity of the scalar input.
    #[must_use]
    pub fn region(&self) -> DeviceRegion<'a> {
        self.view.region()
    }
}

impl<'a, T: DeviceRepr> DeviceScalarMut<'a, T> {
    /// Exclusively borrows the first device element as scalar output storage.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer contains no elements or the first-element region cannot be borrowed.
    pub fn create(buffer: &'a mut (impl DeviceOutput<T> + ?Sized)) -> Result<Self> {
        if buffer.len() == 0 {
            return Err(Error::LengthMismatch {
                name: "device scalar output".into(),
                expected: 1,
                actual: 0,
            });
        }
        Ok(Self {
            view: DeviceOutputView::from_output(buffer).slice(..1)?,
        })
    }

    /// # Safety
    /// The region must be exclusively writable through every native scalar use.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte range does not contain exactly one element, its size calculation overflows, or a nonempty region is zero-sized or misaligned for `T`.
    pub unsafe fn from_raw_parts(region: DeviceRegion<'a>) -> Result<Self> {
        Ok(Self {
            // SAFETY: the caller upholds this function's safety contract.
            view: unsafe { DeviceOutputView::from_raw_parts(region, 1)? },
        })
    }

    /// Returns the scalar's untyped CUDA-accessible address.
    pub fn as_device_ptr(&mut self) -> DevicePtr {
        // SAFETY: the address belongs to a live device allocation that stays borrowed for the operation that uses it.
        unsafe { DevicePtr::from_raw(self.as_mut_ptr().cast()) }
    }

    /// Returns the typed CUDA-accessible output pointer without establishing completion.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.view.as_mut_ptr()
    }

    /// Returns the scalar region's actual CUDA context.
    #[must_use]
    pub fn context(&self) -> &Arc<CudaContext> {
        self.view.context()
    }

    /// Returns the exact byte range and allocation identity of the scalar output.
    #[must_use]
    pub fn region(&self) -> DeviceRegion<'_> {
        self.view.region()
    }
}

impl<'a, T: DeviceRepr> From<&'a T> for Scalar<'a, T> {
    fn from(v: &'a T) -> Self {
        Self::host(v)
    }
}

impl<'a, T: DeviceRepr> From<DeviceScalar<'a, T>> for Scalar<'a, T> {
    fn from(v: DeviceScalar<'a, T>) -> Self {
        Self::Device(v)
    }
}

impl<'a, T: DeviceRepr> From<&'a mut T> for ScalarMut<'a, T> {
    fn from(v: &'a mut T) -> Self {
        Self::host(v)
    }
}

impl<'a, T: DeviceRepr> From<DeviceScalarMut<'a, T>> for ScalarMut<'a, T> {
    fn from(v: DeviceScalarMut<'a, T>) -> Self {
        Self::Device(v)
    }
}
