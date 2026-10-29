use std::{
    ffi::c_void,
    mem::{self, size_of, size_of_val},
    ptr,
};

use singe_cuda::{
    types::DevicePtr,
    view::{DeviceOutput, DeviceRepr},
};
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
pub struct SolverState {
    raw: sys::cudssData_t,
    handle: SharedHandle,
}

// SAFETY: cuDSS data objects are tied to an internally synchronized context handle.
// Ownership can move between threads; shared access still requires `&mut self` for mutating setters.
unsafe impl Send for SolverState {}

impl SolverState {
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
        let handle = context.handle();
        let raw = {
            let owner = handle.lock();
            owner.bind()?;
            let mut raw = ptr::null_mut();
            // SAFETY: the actual creating context stays bound and retained.
            unsafe {
                try_ffi!(sys::cudssDataCreate(owner.as_raw(), &raw mut raw))?;
            }
            if raw.is_null() {
                return Err(Error::NullHandle);
            }
            raw
        };
        Ok(Self { raw, handle })
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
    ///
    /// # Safety
    ///
    /// T and any pointer indirection must match the parameter's native ABI, with
    /// initialized source bytes. Retain referenced buffers for all native uses,
    /// including queued execution and captured replay.
    pub unsafe fn set_raw<T: Copy>(&mut self, param: DataParameter, value: &T) -> Result<()> {
        let handle = self.handle.lock();
        handle.bind()?;
        // SAFETY: `cudssDataSet` receives live handles owned by the wrappers.
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
    ///
    /// # Safety
    ///
    /// T, slice length, initialized source bytes, and indirection must match the
    /// parameter's native array ABI. Retain referenced buffers for all native uses,
    /// including queued execution and captured replay.
    pub unsafe fn set_raw_slice<T: Copy>(
        &mut self,
        param: DataParameter,
        values: &[T],
    ) -> Result<()> {
        let handle = self.handle.lock();
        handle.bind()?;
        // SAFETY: `cudssDataSet` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
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
    /// must remain valid for every cuDSS phase that may use this `SolverState`.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the communicator pointer.
    pub unsafe fn set_device_communicator(&mut self, communicator: *mut c_void) -> Result<()> {
        // SAFETY: the caller retains the correctly typed native communicator.
        unsafe { self.set_raw(DataParameter::CommunicationDevice, &communicator) }
    }

    /// Sets the opaque host communicator used by MG/MN mode.
    ///
    /// # Safety
    ///
    /// `communicator` must point to a live communicator whose concrete type
    /// matches the configured cuDSS communication layer's host backend. It must
    /// remain valid for every cuDSS phase that may use this `SolverState`.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the communicator pointer.
    pub unsafe fn set_host_communicator(&mut self, communicator: *mut c_void) -> Result<()> {
        // SAFETY: the caller retains the correctly typed native communicator.
        unsafe { self.set_raw(DataParameter::CommunicationHost, &communicator) }
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
    ///
    /// # Safety
    ///
    /// The parameter must be a synchronous host output that does not read an
    /// existing destination value. Successful writes must initialize every byte as
    /// a valid T; matching byte counts alone does not establish value validity.
    pub unsafe fn get_raw<T: Copy>(&self, param: DataParameter) -> Result<T> {
        let handle = self.handle.lock();
        handle.bind_data_query(param)?;
        // SAFETY: the caller upholds this function's safety contract.
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
        handle.bind_data_query(param)?;
        // SAFETY: `cudssDataGet` receives live handles owned by the wrappers and valid output locations.
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
    ///
    /// # Safety
    ///
    /// T and its initial contents must satisfy the parameter's native input/output
    /// ABI. Writes must leave a valid T. For indirect device writes, retain the
    /// pointed-to output and resources with exclusive write access through completion
    /// on every path, and order prior accesses before the copy.
    pub unsafe fn get_into_raw<T: Copy>(
        &self,
        param: DataParameter,
        value: &mut T,
    ) -> Result<usize> {
        let handle = self.handle.lock();
        handle.bind_data_query(param)?;
        // SAFETY: the caller upholds this function's safety contract.
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
    ///
    /// # Safety
    ///
    /// T and the destination length must match this synchronous host parameter's
    /// ABI. Native writes must leave valid initialized T values. Use the device-output
    /// APIs for parameters that require device-visible destinations.
    pub unsafe fn get_raw_slice<T: Copy>(
        &self,
        param: DataParameter,
        values: &mut [T],
    ) -> Result<usize> {
        let handle = self.handle.lock();
        handle.bind_data_query(param)?;
        // SAFETY: the caller upholds this function's safety contract.
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
    ///
    /// # Safety
    ///
    /// The pointer must identify `byte_len` live, exclusively writable device bytes
    /// accessible from this data object's execution context. Match the parameter's
    /// native output layout, order prior phases and accesses, and retain all resources
    /// through completion on every path. Only the completed produced prefix may be
    /// consumed; a successful enqueue alone does not initialize output values.
    pub unsafe fn get_device_raw(
        &self,
        param: DataParameter,
        values: DevicePtr,
        byte_len: usize,
    ) -> Result<usize> {
        let handle = self.handle.lock();
        handle.bind_data_query(param)?;
        // SAFETY: the caller upholds this function's safety contract.
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

    /// Queues a solver-data copy into write-only device storage.
    ///
    /// Returns the produced byte count, which may cover only a prefix.
    ///
    /// # Errors
    ///
    /// Returns an error if the destination belongs to another context, the
    /// parameter cannot be queried in this execution mode, context binding or
    /// the native query fails, or a byte count cannot be represented. A native
    /// result claiming more bytes than the destination holds is also rejected.
    ///
    /// # Safety
    ///
    /// T and the destination layout must match the selected parameter's ABI.
    /// Retain the output, data object, and context with exclusive write access
    /// through completion on every path. Order prior producer phases before the
    /// copy. Only completed, whole produced elements may be promoted; padding,
    /// incomplete elements, and unused tails remain outside that guarantee.
    pub unsafe fn get_device<T: DeviceRepr>(
        &self,
        param: DataParameter,
        values: &mut impl DeviceOutput<T>,
    ) -> Result<usize> {
        if values.context() != self.handle.lock().cuda_context() {
            return Err(Error::ContextMismatch);
        }
        let byte_len = values.region().byte_len();
        // SAFETY: the output trait supplies valid writable storage; the caller
        // establishes parameter ABI, ordering, and queued lifetime requirements.
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
        // SAFETY: this parameter has the fixed numeric ABI of the return type.
        unsafe { self.get_raw(DataParameter::Info) }
    }

    /// Returns the number of nonzero entries in LU factors.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot report the value.
    pub fn lu_nnz(&self) -> Result<i64> {
        // SAFETY: this parameter has the fixed numeric ABI of the return type.
        unsafe { self.get_raw(DataParameter::LuNnz) }
    }

    /// Returns host and device memory estimates.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot report the values.
    pub fn memory_estimates(&self) -> Result<[i64; 16]> {
        // SAFETY: this parameter has the fixed numeric ABI of the return type.
        unsafe { self.get_raw(DataParameter::MemoryEstimates) }
    }

    /// Returns the minimum device memory required for hybrid memory mode.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot report the value.
    pub fn hybrid_device_memory_min(&self) -> Result<i64> {
        // SAFETY: this parameter has the fixed numeric ABI of the return type.
        unsafe { self.get_raw(DataParameter::HybridDeviceMemoryMin) }
    }

    /// Returns the Schur complement shape as `(rows, columns, sparse_nnz)`.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot report the value.
    pub fn schur_shape(&self) -> Result<[i64; 3]> {
        // SAFETY: this parameter has the fixed numeric ABI of the return type.
        unsafe { self.get_raw(DataParameter::SchurShape) }
    }

    /// Returns the actual iterative refinement step count.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot report the value.
    pub fn iterative_refinement_steps(&self) -> Result<i32> {
        // SAFETY: this parameter has the fixed numeric ABI of the return type.
        unsafe { self.get_raw(DataParameter::IterativeRefinementSteps) }
    }

    /// Returns the factorization floating-point operation count.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot report the value.
    pub fn flops(&self) -> Result<f64> {
        // SAFETY: this parameter has the fixed numeric ABI of the return type.
        unsafe { self.get_raw(DataParameter::Flops) }
    }

    /// Returns the raw cuDSS data handle.
    #[must_use]
    pub const fn as_raw(&self) -> sys::cudssData_t {
        self.raw
    }

    /// Consumes this wrapper and returns the owned raw cuDSS data handle.
    ///
    /// The caller becomes responsible for destroying the returned handle with the matching cuDSS context.
    #[must_use]
    pub fn into_raw(mut self) -> (sys::cudssData_t, Context) {
        let raw = mem::replace(&mut self.raw, ptr::null_mut());
        (raw, Context::from_handle(self.handle.clone()))
    }
}

impl Drop for SolverState {
    fn drop(&mut self) {
        if self.raw.is_null() {
            return;
        }
        let handle = self.handle.lock();
        let _ = handle.bind();
        // SAFETY: `cudssDataDestroy` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cudssDataDestroy(handle.as_raw(), self.raw)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cudss data: {error}");
            }
        }
    }
}
