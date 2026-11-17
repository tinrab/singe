//! Host values and borrowed device scalar regions for raw cuSPARSE calls.

use std::{ptr, sync::Arc};

use singe_cuda::{
    context::Context as CudaContext,
    error::Error as CudaError,
    types::DevicePtr,
    view::{DeviceInputView, DeviceRange, DeviceRegion, DeviceRepr},
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

/// One borrowed device input element, whose producer must precede native reads.
#[derive(Debug, Clone, Copy)]
pub struct DeviceScalar<'a, T: DeviceRepr> {
    view: DeviceInputView<'a, T>,
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
    pub fn device(buffer: &'a (impl DeviceRange<T> + ?Sized)) -> Result<Self> {
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
            return Err(Error::Cuda(CudaError::MemoryContextMismatch));
        }
        Ok(())
    }

    pub(crate) fn region(self) -> Option<DeviceRegion<'a>> {
        match self {
            Self::Host(_) => None,
            Self::Device(v) => Some(v.region()),
        }
    }
}

impl<'a, T: DeviceRepr> DeviceScalar<'a, T> {
    /// Borrows input metadata for the first device element without establishing initialization.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer contains no elements or the first-element region cannot be borrowed.
    pub fn create(buffer: &'a (impl DeviceRange<T> + ?Sized)) -> Result<Self> {
        if buffer.len() == 0 {
            return Err(Error::BufferTooSmall {
                name: "device scalar".into(),
                required: 1,
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
