use std::{
    ffi::c_void,
    mem::{ManuallyDrop, MaybeUninit, size_of, size_of_val},
    ptr,
};

use singe_core::report_cleanup_failure;
use singe_cuda::{
    error::Error as CudaError,
    types::DevicePtr,
    view::{DeviceOutput, DeviceRepr},
};
use singe_cudss_sys as sys;

use crate::{
    context::Context,
    error::{Error, Result},
    try_ffi,
    types::DataParameter,
    utility::{ensure_exact_size, to_size_t, to_usize, validate_min_len},
};

/// Solver state that stores cuDSS internal data and query results for one solve.
///
/// cuDSS stores analysis products, factors, memory allocations, and user-provided data associated with one solve in this object.
/// The state borrows the [`Context`] that created it, since cuDSS destroys it through that handle.
#[derive(Debug)]
pub struct SolverState<'ctx> {
    raw: sys::cudssData_t,
    context: &'ctx Context,
}

impl<'ctx> SolverState<'ctx> {
    /// Creates solver state for `context`.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, cuDSS cannot create the state, or it returns a null handle.
    pub fn create(context: &'ctx Context) -> Result<Self> {
        context.bind()?;
        let mut raw = ptr::null_mut();
        // SAFETY: `context` owns a live cuDSS handle bound above, and `raw` is a local output checked for null below.
        unsafe {
            try_ffi!(sys::cudssDataCreate(context.as_raw(), &raw mut raw))?;
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self { raw, context })
    }

    /// Adopts a cuDSS data handle created by `context`.
    ///
    /// # Safety
    ///
    /// `raw` must be a uniquely owned cuDSS data handle created with `context`'s handle.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw` is null.
    pub unsafe fn from_raw(context: &'ctx Context, raw: sys::cudssData_t) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self { raw, context })
    }

    /// Transfers the native data handle to the caller, who destroys it with the handle of [`Self::context`].
    #[must_use]
    pub fn into_raw(self) -> sys::cudssData_t {
        ManuallyDrop::new(self).raw
    }

    /// Returns the native data handle without transferring ownership.
    #[must_use]
    pub const fn as_raw(&self) -> sys::cudssData_t {
        self.raw
    }

    /// Returns the context that created this state.
    #[must_use]
    pub const fn context(&self) -> &'ctx Context {
        self.context
    }

    /// Sets a scalar solver data parameter.
    ///
    /// # Safety
    ///
    /// `T` and any pointer indirection must match the parameter's native ABI, with initialized source bytes.
    /// Referenced buffers must stay valid for all native uses, including queued execution and captured replay.
    ///
    /// # Errors
    ///
    /// Returns an error if the value size cannot be represented for cuDSS or if cuDSS rejects the parameter, value type, or value.
    pub unsafe fn set_raw<T: Copy>(&mut self, param: DataParameter, value: &T) -> Result<()> {
        self.context.bind()?;
        // SAFETY: the handles are live in the bound context, `value` is readable for `size_of::<T>()` bytes, and the caller guarantees `T` matches the parameter's native ABI.
        unsafe {
            try_ffi!(sys::cudssDataSet(
                self.context.as_raw(),
                self.raw,
                param.into(),
                ptr::from_ref(value).cast(),
                to_size_t(size_of::<T>(), "data parameter size")?,
            ))
        }
    }

    /// Sets an array-valued solver data parameter.
    ///
    /// # Safety
    ///
    /// `T`, the slice length, initialized source bytes, and indirection must match the parameter's native array ABI.
    /// Referenced buffers must stay valid for all native uses, including queued execution and captured replay.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice byte size cannot be represented for cuDSS or if cuDSS rejects the parameter, value type, or values.
    pub unsafe fn set_raw_slice<T: Copy>(
        &mut self,
        param: DataParameter,
        values: &[T],
    ) -> Result<()> {
        self.context.bind()?;
        // SAFETY: the handles are live in the bound context, `values` is readable for its byte length, and the caller guarantees the element type and length match the parameter's native array ABI.
        unsafe {
            try_ffi!(sys::cudssDataSet(
                self.context.as_raw(),
                self.raw,
                param.into(),
                values.as_ptr().cast(),
                to_size_t(size_of_val(values), "data parameter size")?,
            ))
        }
    }

    /// Sets the opaque device communicator used by MGMN mode.
    ///
    /// # Safety
    ///
    /// `communicator` must point to a live communicator whose concrete type matches the configured communication layer's device backend.
    /// It must stay valid for every cuDSS phase that may use this state.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the communicator pointer.
    pub unsafe fn set_device_communicator(&mut self, communicator: *mut c_void) -> Result<()> {
        // SAFETY: the parameter's ABI is one pointer, and the caller retains the correctly typed communicator.
        unsafe { self.set_raw(DataParameter::CommunicationDevice, &communicator) }
    }

    /// Sets the opaque host communicator used by MGMN mode.
    ///
    /// # Safety
    ///
    /// `communicator` must point to a live communicator whose concrete type matches the configured communication layer's host backend.
    /// It must stay valid for every cuDSS phase that may use this state.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the communicator pointer.
    pub unsafe fn set_host_communicator(&mut self, communicator: *mut c_void) -> Result<()> {
        // SAFETY: the parameter's ABI is one pointer, and the caller retains the correctly typed communicator.
        unsafe { self.set_raw(DataParameter::CommunicationHost, &communicator) }
    }

    /// Returns the number of bytes cuDSS writes when retrieving `param`.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot report the parameter size or the byte count cannot be represented as `usize`.
    pub fn size_of(&self, param: DataParameter) -> Result<usize> {
        self.context.bind_data_query(param)?;
        let mut size_written = 0;
        // SAFETY: the handles are live in the bound context; a null destination with zero size only reports the needed byte count.
        unsafe {
            try_ffi!(sys::cudssDataGet(
                self.context.as_raw(),
                self.raw,
                param.into(),
                ptr::null_mut(),
                0,
                &raw mut size_written,
            ))?;
        }
        to_usize(size_written, "size_written")
    }

    /// Retrieves a scalar solver data parameter.
    ///
    /// The byte size cuDSS reports for `param` is checked against `T` before anything is written.
    ///
    /// # Safety
    ///
    /// The parameter must be a synchronous host output that does not read an existing destination value.
    /// The written bytes must form a valid `T`.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS reports a size other than `size_of::<T>()`, as [`Error::SizeMismatch`], or rejects the query.
    pub unsafe fn get_raw<T: Copy>(&self, param: DataParameter) -> Result<T> {
        let mut value = MaybeUninit::<T>::uninit();
        // SAFETY: `value` is writable for `size_of::<T>()` bytes, and the caller guarantees the parameter writes a valid `T`.
        unsafe {
            self.get_bytes(param, value.as_mut_ptr().cast(), size_of::<T>(), true)?;
            Ok(value.assume_init())
        }
    }

    /// Retrieves a scalar solver data parameter into an existing value.
    ///
    /// The byte size cuDSS reports for `param` is checked against `T` before anything is written.
    ///
    /// # Safety
    ///
    /// `T` and its initial contents must satisfy the parameter's native input and output ABI, and the written bytes must leave a valid `T`.
    /// For indirect device writes, the pointed-to output must stay exclusively writable through completion, with prior accesses ordered before the copy.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS reports a size other than `size_of::<T>()`, as [`Error::SizeMismatch`], or rejects the query.
    pub unsafe fn get_into_raw<T: Copy>(
        &self,
        param: DataParameter,
        value: &mut T,
    ) -> Result<usize> {
        // SAFETY: `value` is writable for `size_of::<T>()` bytes, and the caller guarantees `T` and its initial contents match the parameter's native ABI.
        unsafe { self.get_bytes(param, ptr::from_mut(value).cast(), size_of::<T>(), true) }
    }

    /// Retrieves an array-valued solver data parameter into a host slice.
    ///
    /// The byte size cuDSS reports for `param` is checked against the slice before anything is written.
    ///
    /// # Safety
    ///
    /// The parameter must be a synchronous host output, and the written bytes must form valid `T` values.
    /// Use [`Self::get_device`] for parameters that require device-visible destinations.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS needs more bytes than `values` holds, as [`Error::BufferTooSmall`], or rejects the query.
    pub unsafe fn get_raw_slice<T: Copy>(
        &self,
        param: DataParameter,
        values: &mut [T],
    ) -> Result<usize> {
        // SAFETY: `values` is writable for its byte length, and the caller guarantees the written bytes form valid `T` values.
        unsafe {
            self.get_bytes(
                param,
                values.as_mut_ptr().cast(),
                size_of_val(values),
                false,
            )
        }
    }

    /// Retrieves a solver data parameter into device memory.
    ///
    /// The byte size cuDSS reports for `param` is checked against `byte_len` before anything is written.
    ///
    /// # Safety
    ///
    /// `values` must identify `byte_len` live, exclusively writable device bytes accessible from this state's execution context.
    /// The destination must match the parameter's native output layout, with prior phases and accesses ordered before the copy, and stay valid through completion.
    /// Only the completed produced prefix may be read.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS needs more than `byte_len` bytes, as [`Error::BufferTooSmall`], or rejects the query.
    pub unsafe fn get_device_raw(
        &self,
        param: DataParameter,
        values: DevicePtr,
        byte_len: usize,
    ) -> Result<usize> {
        // SAFETY: the caller guarantees `values` addresses `byte_len` writable device bytes with the parameter's layout.
        unsafe { self.get_bytes(param, values.as_raw().cast(), byte_len, false) }
    }

    /// Queues a solver data copy into write-only device storage and returns the produced byte count.
    ///
    /// # Safety
    ///
    /// `T` and the destination layout must match the parameter's ABI.
    /// The output must stay exclusively writable through completion, with prior producer phases ordered before the copy.
    /// Only completed, whole produced elements may be read.
    ///
    /// # Errors
    ///
    /// Returns an error if the destination belongs to another CUDA context, cuDSS needs more bytes than it holds, or the query fails.
    pub unsafe fn get_device<T: DeviceRepr>(
        &self,
        param: DataParameter,
        values: &mut impl DeviceOutput<T>,
    ) -> Result<usize> {
        if values.context() != self.context.cuda_context() {
            return Err(CudaError::MemoryContextMismatch.into());
        }

        let byte_len = values.region().byte_len();
        // SAFETY: the output view supplies `byte_len` writable bytes in this context; the caller guarantees the layout matches the parameter and prior accesses are ordered before the copy.
        unsafe {
            self.get_device_raw(
                param,
                DevicePtr::from_raw(values.as_mut_ptr().cast()),
                byte_len,
            )
        }
    }

    /// Returns host-side device error information.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot report the value.
    pub fn info(&self) -> Result<i32> {
        // SAFETY: this parameter is a host `int`.
        unsafe { self.get_raw(DataParameter::Info) }
    }

    /// Returns the number of nonzero entries in the LU factors.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot report the value.
    pub fn lu_nnz(&self) -> Result<i64> {
        // SAFETY: this parameter is a host `int64_t`.
        unsafe { self.get_raw(DataParameter::LuNnz) }
    }

    /// Returns host and device memory estimates.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot report the values.
    pub fn memory_estimates(&self) -> Result<[i64; 16]> {
        // SAFETY: this parameter is a host `int64_t[16]`.
        unsafe { self.get_raw(DataParameter::MemoryEstimates) }
    }

    /// Returns the minimum device memory required for hybrid memory mode.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot report the value.
    pub fn hybrid_device_memory_min(&self) -> Result<i64> {
        // SAFETY: this parameter is a host `int64_t`.
        unsafe { self.get_raw(DataParameter::HybridDeviceMemoryMin) }
    }

    /// Returns the Schur complement shape as `(rows, columns, sparse_nnz)`.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot report the value.
    pub fn schur_shape(&self) -> Result<[i64; 3]> {
        // SAFETY: this parameter is a host `int64_t[3]`.
        unsafe { self.get_raw(DataParameter::SchurShape) }
    }

    /// Returns the number of iterative refinement steps performed.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot report the value.
    pub fn iterative_refinement_steps(&self) -> Result<i32> {
        // SAFETY: this parameter is a host `int`.
        unsafe { self.get_raw(DataParameter::IterativeRefinementSteps) }
    }

    /// Returns the factorization floating-point operation count.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot report the value.
    pub fn flops(&self) -> Result<f64> {
        // SAFETY: this parameter is a host `double`.
        unsafe { self.get_raw(DataParameter::Flops) }
    }

    unsafe fn get_bytes(
        &self,
        param: DataParameter,
        destination: *mut c_void,
        capacity: usize,
        exact: bool,
    ) -> Result<usize> {
        let required = self.size_of(param)?;
        if exact {
            ensure_exact_size(required, capacity, "data parameter")?;
        } else {
            validate_min_len(required, capacity, "data parameter")?;
        }

        self.context.bind_data_query(param)?;
        let mut size_written = 0;
        // SAFETY: the handles are live, the caller guarantees `destination` is writable for `capacity` bytes, and cuDSS needs at most that many.
        unsafe {
            try_ffi!(sys::cudssDataGet(
                self.context.as_raw(),
                self.raw,
                param.into(),
                destination,
                to_size_t(capacity, "data parameter size")?,
                &raw mut size_written,
            ))?;
        }

        let size_written = to_usize(size_written, "size_written")?;
        validate_min_len(size_written, capacity, "data parameter")?;
        Ok(size_written)
    }
}

impl Drop for SolverState<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.context.bind() {
            report_cleanup_failure(
                "bind the cuda context to destroy",
                "cudss solver state",
                &error,
            );
            return;
        }

        // SAFETY: `self.raw` is the data handle this wrapper owns, `into_raw` skips this drop when ownership leaves it, and the borrowed `self.context` that created it is live in the context bound above.
        let result = unsafe { try_ffi!(sys::cudssDataDestroy(self.context.as_raw(), self.raw)) };
        if let Err(error) = result {
            report_cleanup_failure("destroy", "cudss solver state", &error);
        }
    }
}
