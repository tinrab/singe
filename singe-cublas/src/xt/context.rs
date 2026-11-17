use std::{ffi::c_void, mem::ManuallyDrop, ptr};

use singe_core::report_cleanup_failure;
use singe_cublas_sys as sys;
use singe_cuda::device::Device;

use crate::{
    error::{Error, Result},
    try_ffi,
    utility::{to_i32, to_usize},
    xt::types::{BlasOperation, CpuRoutine, OperationType, PinningMemoryMode},
};

/// A stateful cuBLASXt handle with a fixed device selection.
///
/// cuBLASXt routines are blocking host APIs that distribute BLAS3 work across the devices selected for this context.
#[derive(Debug)]
pub struct Context {
    handle: Handle,
}

#[derive(Debug)]
struct Handle {
    raw: sys::cublasXtHandle_t,
}

// SAFETY: a cuBLASXt handle may be used from any host thread one call at a time, and `Handle` is not `Sync`, so moving it keeps calls serialized.
unsafe impl Send for Handle {}

impl Context {
    /// Creates a cuBLASXt context for the current CUDA device.
    ///
    /// This is the single-device convenience constructor.
    /// Use [`Context::create_for`] to select an explicit set of devices.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query the current device, cuBLASXt cannot initialize that device, cuBLASXt cannot allocate the handle, or cuBLASXt returns a null handle.
    pub fn create() -> Result<Self> {
        let device = Device::current()?;
        Self::create_for(&[device])
    }

    /// Creates a cuBLASXt context for `devices`.
    ///
    /// cuBLASXt treats device selection as static for a handle, so the selected device set is part of context construction.
    /// Create another context to use a different device set.
    ///
    /// # Errors
    ///
    /// Returns an error if `devices` is empty, too long for cuBLASXt, contains invalid device IDs, cuBLASXt cannot initialize one of the devices, cuBLASXt cannot allocate the handle, or cuBLASXt returns a null handle.
    pub fn create_for(devices: &[Device]) -> Result<Self> {
        if devices.is_empty() {
            return Err(Error::LengthMismatch {
                name: "devices".into(),
                expected: 1,
                actual: 0,
            });
        }

        let mut handle = ptr::null_mut();
        // SAFETY: `cublasXtCreate` writes the new handle to `handle`, a live local.
        unsafe {
            try_ffi!(sys::cublasXtCreate(&raw mut handle))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        let context = Self {
            handle: Handle { raw: handle },
        };
        context.select_devices(devices)?;
        Ok(context)
    }

    /// Wraps an existing cuBLASXt handle and takes ownership of it.
    ///
    /// # Safety
    ///
    /// `handle` must be a live cuBLASXt handle whose devices were selected exactly once.
    /// It must not be destroyed elsewhere or have its devices selected again.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NullHandle`] if `handle` is null.
    pub unsafe fn from_raw(handle: sys::cublasXtHandle_t) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: Handle { raw: handle },
        })
    }

    fn select_devices(&self, devices: &[Device]) -> Result<()> {
        let mut device_ids = devices
            .iter()
            .map(|device| to_i32(device.id(), "device id"))
            .collect::<Result<Vec<i32>>>()?;
        // SAFETY: the handle this context owns is live, and `device_ids` holds the count passed with it.
        unsafe {
            try_ffi!(sys::cublasXtDeviceSelect(
                self.as_raw(),
                to_i32(device_ids.len(), "device count")?,
                device_ids.as_mut_ptr(),
            ))?;
        }
        Ok(())
    }

    /// Sets the square tile dimension used by cuBLASXt.
    ///
    /// # Errors
    ///
    /// Returns an error if `block_dim` is zero, too large for cuBLASXt, or rejected by the library.
    pub fn set_block_dim(&mut self, block_dim: usize) -> Result<()> {
        if block_dim == 0 {
            return Err(Error::InvalidMatrixShape);
        }

        // SAFETY: the handle this context owns is live, and the block dimension is passed by value.
        unsafe {
            try_ffi!(sys::cublasXtSetBlockDim(
                self.as_raw(),
                to_i32(block_dim, "block dimension")?,
            ))?;
        }
        Ok(())
    }

    /// Returns the current cuBLASXt square tile dimension.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASXt cannot report the value.
    pub fn block_dim(&self) -> Result<usize> {
        let mut block_dim = 0;
        // SAFETY: the handle this context owns is live, and `block_dim` is a live local.
        unsafe {
            try_ffi!(sys::cublasXtGetBlockDim(self.as_raw(), &raw mut block_dim,))?;
        }
        to_usize(block_dim, "block dimension")
    }

    /// Sets whether cuBLASXt may pin pageable host memory passed to routines.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASXt rejects the mode.
    pub fn set_pinning_memory_mode(&mut self, mode: PinningMemoryMode) -> Result<()> {
        // SAFETY: the handle this context owns is live, and the mode is passed by value.
        unsafe {
            try_ffi!(sys::cublasXtSetPinningMemMode(self.as_raw(), mode.into(),))?;
        }
        Ok(())
    }

    /// Returns the current cuBLASXt host-memory pinning mode.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASXt cannot report the mode.
    pub fn pinning_memory_mode(&self) -> Result<PinningMemoryMode> {
        let mut mode = sys::cublasXtPinnedMemMode_t::CUBLASXT_PINNING_DISABLED;
        // SAFETY: the handle this context owns is live, and `mode` is a live local.
        unsafe {
            try_ffi!(sys::cublasXtGetPinningMemMode(self.as_raw(), &raw mut mode,))?;
        }
        Ok(mode.try_into()?)
    }

    /// Sets the fraction of work cuBLASXt should offload to a caller-provided CPU BLAS routine for the selected operation and type.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASXt rejects the operation/type combination or ratio.
    pub fn set_cpu_ratio(
        &mut self,
        operation: BlasOperation,
        operation_type: OperationType,
        ratio: f32,
    ) -> Result<()> {
        // SAFETY: the handle this context owns is live, and the selectors and ratio are passed by value.
        unsafe {
            try_ffi!(sys::cublasXtSetCpuRatio(
                self.as_raw(),
                operation.into(),
                operation_type.into(),
                ratio,
            ))?;
        }
        Ok(())
    }

    /// Sets the host GEMM routine that cuBLASXt runs for the share of GEMM work set by [`Context::set_cpu_ratio`].
    ///
    /// # Safety
    ///
    /// The routine must implement the Fortran BLAS GEMM of its element type, and cuBLASXt may call it from its own threads during later operations on this context.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASXt rejects the routine.
    pub unsafe fn set_cpu_routine(&mut self, routine: CpuRoutine) -> Result<()> {
        let (operation_type, routine): (OperationType, *mut c_void) = match routine {
            CpuRoutine::Sgemm(routine) => (OperationType::F32, routine as *mut c_void),
            CpuRoutine::Dgemm(routine) => (OperationType::F64, routine as *mut c_void),
            CpuRoutine::Cgemm(routine) => (OperationType::Complex32, routine as *mut c_void),
            CpuRoutine::Zgemm(routine) => (OperationType::Complex64, routine as *mut c_void),
        };
        // SAFETY: the handle is live, and the caller guarantees the routine follows the GEMM contract of its element type.
        unsafe {
            try_ffi!(sys::cublasXtSetCpuRoutine(
                self.as_raw(),
                BlasOperation::Gemm.into(),
                operation_type.into(),
                routine,
            ))
        }
    }

    /// Returns the raw cuBLASXt handle.
    ///
    /// The returned handle is borrowed and remains valid only while this context is alive.
    #[must_use]
    pub fn as_raw(&self) -> sys::cublasXtHandle_t {
        self.handle.raw
    }

    /// Consumes this context and returns the owned raw cuBLASXt handle.
    ///
    /// The caller becomes responsible for eventually destroying the handle.
    #[must_use]
    pub fn into_raw(self) -> sys::cublasXtHandle_t {
        let this = ManuallyDrop::new(self);
        this.handle.raw
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        // SAFETY: `Handle` owns `self.raw`, and `drop` destroys it exactly once.
        unsafe {
            if let Err(error) = try_ffi!(sys::cublasXtDestroy(self.raw)) {
                report_cleanup_failure("destroy", "cublasxt context", &error);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{error::Error, xt::context::Context};

    #[test]
    fn test_create_for_rejects_empty_devices() {
        let error = Context::create_for(&[]).unwrap_err();

        assert!(matches!(
            error,
            Error::LengthMismatch {
                ref name,
                expected: 1,
                actual: 0,
            } if name == "devices"
        ));
    }
}
