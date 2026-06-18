use std::{
    ffi::c_void,
    mem::{self, ManuallyDrop, size_of, size_of_val},
    ptr,
};

use singe_cuda::types::DevicePtr;
use singe_cudss_sys as sys;

use crate::{
    context::{Context, SharedHandle},
    error::{Error, Result},
    try_ffi,
    types::DataParameter,
    utility::{ensure_at_most_size, ensure_exact_size, to_size_t, to_usize},
};

/// Solver data object used to store cuDSS internal state and query results.
///
/// cuDSS stores analysis products, factors, memory allocations, and user-provided data associated with one solve in this object.
/// The object is tied to the [`Context`] that created it.
#[derive(Debug)]
pub struct Data {
    raw: sys::cudssData_t,
    handle: SharedHandle,
}

// cuDSS data objects are tied to an internally synchronized context handle.
// Ownership can move between threads; shared access still requires `&mut self` for mutating setters.
unsafe impl Send for Data {}

impl Data {
    /// Creates a solver data object for `context`.
    ///
    /// cuDSS stores analysis products, factors, memory allocations, query
    /// results, and copied user-provided data in this object.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot create the data object or returns a null
    /// handle.
    pub fn create(context: &Context) -> Result<Self> {
        unsafe {
            let mut raw = ptr::null_mut();
            try_ffi!(sys::cudssDataCreate(context.as_raw(), &raw mut raw))?;
            if raw.is_null() {
                return Err(Error::NullHandle);
            }
            Ok(Self {
                raw,
                handle: context.handle(),
            })
        }
    }

    /// Takes ownership of an existing cuDSS data handle associated with `context`.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid cuDSS data handle created for `context`. Ownership
    /// is transferred to the returned wrapper, which destroys the handle on
    /// drop with the same context.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw` is null.
    pub unsafe fn from_raw(context: &Context, raw: sys::cudssData_t) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            raw,
            handle: context.handle(),
        })
    }

    /// Sets a scalar solver data parameter.
    ///
    /// `T` must match the value type documented for `param`.
    ///
    /// # Errors
    ///
    /// Returns an error if the value size cannot be represented for cuDSS or if
    /// cuDSS rejects the parameter, value type, or value.
    pub fn set<T: Copy>(&mut self, param: DataParameter, value: &T) -> Result<()> {
        let handle = self.handle.lock();
        unsafe {
            try_ffi!(sys::cudssDataSet(
                handle.as_raw(),
                self.raw,
                param.into(),
                ptr::from_ref(value).cast(),
                to_size_t(size_of::<T>(), "data parameter size")?,
            ))
        }
    }

    /// Sets an array-valued solver data parameter.
    ///
    /// `T` and the slice length must match the value type documented for
    /// `param`.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice byte size cannot be represented for cuDSS
    /// or if cuDSS rejects the parameter, value type, or values.
    pub fn set_slice<T: Copy>(&mut self, param: DataParameter, values: &[T]) -> Result<()> {
        let handle = self.handle.lock();
        unsafe {
            try_ffi!(sys::cudssDataSet(
                handle.as_raw(),
                self.raw,
                param.into(),
                values.as_ptr().cast(),
                to_size_t(size_of_val(values), "data parameter size")?,
            ))
        }
    }

    /// Sets the opaque device communicator used by MG/MN mode.
    ///
    /// # Safety
    ///
    /// `communicator` must point to a live communicator whose concrete type
    /// matches the configured cuDSS communication layer's device backend. It
    /// must remain valid for every cuDSS phase that may use this `Data`.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the communicator pointer.
    pub unsafe fn set_device_communicator(&mut self, communicator: *mut c_void) -> Result<()> {
        self.set(DataParameter::CommunicationDevice, &communicator)
    }

    /// Sets the opaque host communicator used by MG/MN mode.
    ///
    /// # Safety
    ///
    /// `communicator` must point to a live communicator whose concrete type
    /// matches the configured cuDSS communication layer's host backend. It must
    /// remain valid for every cuDSS phase that may use this `Data`.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the communicator pointer.
    pub unsafe fn set_host_communicator(&mut self, communicator: *mut c_void) -> Result<()> {
        self.set(DataParameter::CommunicationHost, &communicator)
    }

    /// Retrieves a scalar solver data parameter.
    ///
    /// `T` must match the value type documented for `param`.
    ///
    /// # Errors
    ///
    /// Returns an error if the value size cannot be represented for cuDSS, if
    /// cuDSS rejects the parameter or value type, or if cuDSS reports an
    /// internal byte-count mismatch.
    pub fn get<T: Copy>(&self, param: DataParameter) -> Result<T> {
        let handle = self.handle.lock();
        unsafe {
            let mut value = mem::MaybeUninit::<T>::uninit();
            let mut size_written = 0;
            try_ffi!(sys::cudssDataGet(
                handle.as_raw(),
                self.raw,
                param.into(),
                value.as_mut_ptr().cast(),
                to_size_t(size_of::<T>(), "data parameter size")?,
                &raw mut size_written,
            ))?;
            ensure_exact_size(
                to_usize(size_written, "data parameter size_written")?,
                size_of::<T>(),
                "data parameter",
            )?;
            Ok(value.assume_init())
        }
    }

    /// Retrieves a scalar solver data parameter.
    ///
    /// This is the explicit low-level form of [`Self::get`] for parameters not
    /// covered by a named method.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the parameter or returns an internal error.
    pub fn get_raw<T: Copy>(&self, param: DataParameter) -> Result<T> {
        self.get(param)
    }

    /// Returns the number of bytes needed to retrieve `param`.
    ///
    /// This is useful for variable-length data parameters before allocating a
    /// destination buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot report the parameter size or if the
    /// reported byte count cannot be represented as `usize`.
    pub fn size_of(&self, param: DataParameter) -> Result<usize> {
        let handle = self.handle.lock();
        unsafe {
            let mut size_written = 0;
            try_ffi!(sys::cudssDataGet(
                handle.as_raw(),
                self.raw,
                param.into(),
                ptr::null_mut(),
                0,
                &raw mut size_written,
            ))?;
            to_usize(size_written, "size_written")
        }
    }

    /// Retrieves a scalar solver data parameter into an existing value.
    ///
    /// `T` must match the value type documented for `param`.
    ///
    /// # Errors
    ///
    /// Returns an error if the value size cannot be represented for cuDSS, if
    /// cuDSS rejects the parameter or value type, or if cuDSS reports an
    /// internal byte-count mismatch.
    pub fn get_into<T: Copy>(&self, param: DataParameter, value: &mut T) -> Result<usize> {
        let handle = self.handle.lock();
        unsafe {
            let mut size_written = 0;
            try_ffi!(sys::cudssDataGet(
                handle.as_raw(),
                self.raw,
                param.into(),
                ptr::from_mut(value).cast(),
                to_size_t(size_of::<T>(), "data parameter size")?,
                &raw mut size_written,
            ))?;
            let size_written = to_usize(size_written, "size_written")?;
            ensure_exact_size(size_written, size_of::<T>(), "data parameter")?;
            Ok(size_written)
        }
    }

    /// Retrieves an array-valued solver data parameter into a host slice.
    ///
    /// `T` and the slice length must match the value type documented for
    /// `param`.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice byte size cannot be represented for cuDSS,
    /// if cuDSS rejects the parameter or value type, or if cuDSS reports that
    /// more bytes are needed than the provided slice can hold.
    pub fn get_slice<T: Copy>(&self, param: DataParameter, values: &mut [T]) -> Result<usize> {
        let handle = self.handle.lock();
        unsafe {
            let mut size_written = 0;
            try_ffi!(sys::cudssDataGet(
                handle.as_raw(),
                self.raw,
                param.into(),
                values.as_mut_ptr().cast(),
                to_size_t(size_of_val(values), "data parameter size")?,
                &raw mut size_written,
            ))?;
            let size_written = to_usize(size_written, "size_written")?;
            ensure_at_most_size(size_written, size_of_val(values), "data parameter")?;
            Ok(size_written)
        }
    }

    /// Retrieves a solver data parameter into device memory.
    ///
    /// `values` must point to a device allocation with at least `byte_len`
    /// writable bytes, and the byte length must match the value type documented
    /// for `param`.
    ///
    /// # Errors
    ///
    /// Returns an error if `byte_len` cannot be represented for cuDSS, if cuDSS
    /// rejects the parameter or destination pointer, or if cuDSS reports that
    /// more bytes are needed than `byte_len`.
    pub fn get_device(
        &self,
        param: DataParameter,
        values: DevicePtr,
        byte_len: usize,
    ) -> Result<usize> {
        let handle = self.handle.lock();
        unsafe {
            let mut size_written = 0;
            try_ffi!(sys::cudssDataGet(
                handle.as_raw(),
                self.raw,
                param.into(),
                values.as_raw().cast(),
                to_size_t(byte_len, "data parameter byte_len")?,
                &raw mut size_written,
            ))?;
            let size_written = to_usize(size_written, "size_written")?;
            ensure_at_most_size(size_written, byte_len, "data parameter")?;
            Ok(size_written)
        }
    }

    /// Returns host-side device error information.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot report the value.
    pub fn info(&self) -> Result<i32> {
        self.get(DataParameter::Info)
    }

    /// Returns the number of nonzero entries in LU factors.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot report the value.
    pub fn lu_nnz(&self) -> Result<i64> {
        self.get(DataParameter::LuNnz)
    }

    /// Returns host and device memory estimates.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot report the values.
    pub fn memory_estimates(&self) -> Result<[i64; 16]> {
        self.get(DataParameter::MemoryEstimates)
    }

    /// Returns the minimum device memory required for hybrid memory mode.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot report the value.
    pub fn hybrid_device_memory_min(&self) -> Result<i64> {
        self.get(DataParameter::HybridDeviceMemoryMin)
    }

    /// Returns the Schur complement shape as `(rows, columns, sparse_nnz)`.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot report the value.
    pub fn schur_shape(&self) -> Result<[i64; 3]> {
        self.get(DataParameter::SchurShape)
    }

    /// Returns the actual iterative refinement step count.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot report the value.
    pub fn iterative_refinement_steps(&self) -> Result<i32> {
        self.get(DataParameter::IterativeRefinementSteps)
    }

    /// Returns the factorization floating-point operation count.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot report the value.
    pub fn flops(&self) -> Result<f64> {
        self.get(DataParameter::Flops)
    }

    /// Returns the raw cuDSS data handle.
    pub const fn as_raw(&self) -> sys::cudssData_t {
        self.raw
    }

    /// Consumes this wrapper and returns the owned raw cuDSS data handle.
    ///
    /// The caller becomes responsible for destroying the returned handle with the matching cuDSS context.
    pub fn into_raw(self) -> sys::cudssData_t {
        let data = ManuallyDrop::new(self);
        data.raw
    }
}

impl Drop for Data {
    fn drop(&mut self) {
        let handle = self.handle.lock();
        unsafe {
            let _ = sys::cudssDataDestroy(handle.as_raw(), self.raw);
        }
    }
}
