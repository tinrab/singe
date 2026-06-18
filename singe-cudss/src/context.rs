use std::{
    mem::ManuallyDrop,
    path::Path,
    ptr,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
};

use singe_core::path_to_cstring;
use singe_cuda::{context::Context as CudaContext, stream::Stream};
use singe_cudss_sys as sys;

use crate::{
    config::Config,
    data::Data,
    error::{Error, Result},
    matrix::Matrix,
    memory::DeviceMemoryHandler,
    try_ffi,
    types::Phase,
};

/// cuDSS library handle bound to one CUDA context.
///
/// A context owns the cuDSS handle used by configuration, data, and matrix operations.
/// For single-GPU handles, cuDSS binds the handle to the current CUDA device at creation time.
#[derive(Debug)]
pub struct Context {
    handle: SharedHandle,
}

#[derive(Debug, Clone)]
pub(crate) struct SharedHandle(Arc<Mutex<Handle>>);

#[derive(Debug)]
pub(crate) struct Handle {
    raw: sys::cudssHandle_t,
    cuda_ctx: Arc<CudaContext>,
}

// cuDSS handles carry mutable solver state. This wrapper serializes raw handle
// access through `SharedHandle`, so the context can move between threads.
unsafe impl Send for Context {}
// `Handle` is protected by `SharedHandle` whenever it is shared. Moving the
// owned handle between threads is sound under the same context lifetime.
unsafe impl Send for Handle {}

impl Context {
    /// Creates a single-GPU cuDSS context bound to `cuda_ctx`.
    ///
    /// The CUDA context is made current before the cuDSS handle is created.
    /// cuDSS binds single-GPU handles to the current CUDA device at creation
    /// time.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be made current, if cuDSS
    /// cannot create the handle, or if cuDSS returns a null handle.
    pub fn create(cuda_ctx: &Arc<CudaContext>) -> Result<Self> {
        cuda_ctx.bind()?;
        unsafe {
            let mut handle = ptr::null_mut();
            try_ffi!(sys::cudssCreate(&raw mut handle))?;
            if handle.is_null() {
                return Err(Error::NullHandle);
            }
            Ok(Self {
                handle: SharedHandle::new(Handle {
                    raw: handle,
                    cuda_ctx: Arc::clone(cuda_ctx),
                }),
            })
        }
    }

    /// Takes ownership of an existing cuDSS handle associated with `cuda_ctx`.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid cuDSS handle associated with `cuda_ctx`. Ownership
    /// is transferred to the returned context, which destroys the handle when
    /// the last shared reference is dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw` is null.
    pub unsafe fn from_raw(raw: sys::cudssHandle_t, cuda_ctx: Arc<CudaContext>) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            handle: SharedHandle::new(Handle { raw, cuda_ctx }),
        })
    }

    /// Creates a multi-GPU cuDSS context for `devices`.
    ///
    /// The CUDA context is made current before the cuDSS handle is created. The
    /// current CUDA device must be the first listed device. Set matching device
    /// indices on the solver [`Config`] with
    /// [`Config::set_device_indices`](crate::config::Config::set_device_indices).
    ///
    /// Some solver features are not supported in cuDSS multi-GPU mode.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be made current, if cuDSS
    /// rejects the device list, if cuDSS cannot create the handle, or if cuDSS
    /// returns a null handle.
    pub fn create_multi_gpu(cuda_ctx: &Arc<CudaContext>, devices: &[i32]) -> Result<Self> {
        cuda_ctx.bind()?;
        unsafe {
            let mut handle = ptr::null_mut();
            try_ffi!(sys::cudssCreateMg(
                &raw mut handle,
                devices.len() as _,
                devices.as_ptr(),
            ))?;
            if handle.is_null() {
                return Err(Error::NullHandle);
            }
            Ok(Self {
                handle: SharedHandle::new(Handle {
                    raw: handle,
                    cuda_ctx: Arc::clone(cuda_ctx),
                }),
            })
        }
    }

    /// Sets the CUDA stream used by this cuDSS context.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to a different CUDA context or if
    /// cuDSS rejects the stream.
    pub fn set_stream(&self, stream: &Stream) -> Result<()> {
        let handle = self.handle.lock();
        if stream.context().as_raw() != handle.cuda_ctx.as_raw() {
            return Err(Error::StreamContextMismatch);
        }

        unsafe { try_ffi!(sys::cudssSetStream(handle.raw, stream.as_raw())) }
    }

    /// Sets per-device CUDA streams for a multi-GPU cuDSS context.
    ///
    /// `streams[i]` is used for operations on the corresponding device passed
    /// to [`Self::create_multi_gpu`]. If this is not called, cuDSS creates and
    /// manages internal per-device streams.
    ///
    /// The caller owns the streams; they must remain valid for every cuDSS
    /// operation that can use them.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the stream array, including a stream
    /// count that does not match the multi-GPU device count.
    pub fn set_multi_gpu_streams(&self, streams: &[&Stream]) -> Result<()> {
        let mut raw_streams = Vec::with_capacity(streams.len());
        for stream in streams {
            raw_streams.push(stream.as_raw());
        }

        let handle = self.handle.lock();
        unsafe {
            try_ffi!(sys::cudssSetMgStreams(
                handle.raw,
                raw_streams.as_ptr(),
                raw_streams.len() as _,
            ))
        }
    }

    /// Sets the communication layer library for MGMN mode.
    ///
    /// The communication layer must provide backends for both device and host
    /// memory buffers. After setting the layer, set the device and host
    /// communicators on [`Data`] with
    /// [`Data::set_device_communicator`](crate::data::Data::set_device_communicator)
    /// and [`Data::set_host_communicator`](crate::data::Data::set_host_communicator).
    ///
    /// # Errors
    ///
    /// Returns an error if `path` cannot be converted to a C string or if cuDSS
    /// rejects the communication layer.
    pub fn set_communication_layer(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path_to_cstring(path.as_ref())?;
        let handle = self.handle.lock();
        unsafe { try_ffi!(sys::cudssSetCommLayer(handle.raw, path.as_ptr())) }
    }

    /// Uses the default communication layer for MGMN mode.
    ///
    /// cuDSS resolves the library name from its environment configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot set the default communication layer.
    pub fn set_default_communication_layer(&self) -> Result<()> {
        let handle = self.handle.lock();
        unsafe { try_ffi!(sys::cudssSetCommLayer(handle.raw, ptr::null())) }
    }

    /// Sets the threading layer library for multi-threaded mode.
    ///
    /// # Errors
    ///
    /// Returns an error if `path` cannot be converted to a C string or if cuDSS
    /// rejects the threading layer.
    pub fn set_threading_layer(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path_to_cstring(path.as_ref())?;
        let handle = self.handle.lock();
        unsafe { try_ffi!(sys::cudssSetThreadingLayer(handle.raw, path.as_ptr())) }
    }

    /// Uses the default threading layer for multi-threaded mode.
    ///
    /// cuDSS resolves the library name from its environment configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot set the default threading layer.
    pub fn set_default_threading_layer(&self) -> Result<()> {
        let handle = self.handle.lock();
        unsafe { try_ffi!(sys::cudssSetThreadingLayer(handle.raw, ptr::null())) }
    }

    /// Creates a solver data object tied to this context.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot create the data object.
    pub fn create_data(&self) -> Result<Data> {
        Data::create(self)
    }

    /// Executes one or more solver phases.
    ///
    /// The usual solve sequence is analysis, factorization, then solve. Analysis
    /// performs reordering and symbolic factorization; factorization computes
    /// the numeric factors; solve uses those factors to compute the solution.
    ///
    /// Phase order must be:
    ///
    /// 1. [`Phase::REORDERING`] and [`Phase::SYMBOLIC_FACTORIZATION`], or
    ///    [`Phase::ANALYSIS`]
    /// 2. [`Phase::FACTORIZATION`]
    /// 3. optional [`Phase::REFACTORIZATION`]
    /// 4. [`Phase::SOLVE`]
    ///
    /// Phases may be combined when the combined value preserves this order, for
    /// example [`Phase::ANALYSIS`] | [`Phase::FACTORIZATION`]. Reusing analysis
    /// results is supported; after changing matrix values, rerun
    /// factorization/refactorization and solve.
    ///
    /// cuDSS reads settings from `config` and stores internal state in `data`.
    /// Additional data parameters, such as user permutations and communicators,
    /// should be configured on `data` before the phase that uses them.
    ///
    /// Matrix buffers must be device-visible unless hybrid execute mode is
    /// enabled. With hybrid execute mode, the input matrix, solution, or
    /// right-hand side may use host memory pointers.
    ///
    /// Matrix requirements:
    ///
    /// * The system matrix must be sparse (currently, only 3-array CSR format is supported).
    /// * Right-hand side and solution matrices must be dense, with column-major layout.
    /// * The input sparse matrix, or each matrix in a batch, must have data
    ///   consistent with its offsets, indices, indexing base, and matrix type.
    /// * Sparse column indices may be unsorted but must not contain repeated entries.
    /// * The input sparse matrix in CSR format must have its row offsets array start with the indexing base (0 or 1).
    /// * The system matrix, right-hand side, and solution must use matching
    ///   value and index types where applicable.
    /// * Non-uniform batches may vary by shape, but aggregate rows and nonzeros
    ///   must fit in `i32`.
    /// * In MGMN mode all processes must provide valid global shape metadata.
    ///   If no row distribution is set on a matrix, full matrix data must be
    ///   present on communicator rank `0`; other processes may use null data
    ///   pointers. If row distribution is set, the system matrix, solution, and
    ///   right-hand side must follow that distribution.
    ///
    /// Sparse matrix views may expose only a triangular portion of symmetric or
    /// Hermitian matrices without changing the underlying storage.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the phase combination, configuration,
    /// data object, matrix descriptors, or execution.
    pub fn execute(
        &self,
        phase: Phase,
        config: &Config,
        data: &mut Data,
        input_matrix: &Matrix<'_>,
        solution: &mut Matrix<'_>,
        rhs: &Matrix<'_>,
    ) -> Result<()> {
        let handle = self.handle.lock();
        unsafe {
            try_ffi!(sys::cudssExecute(
                handle.raw,
                phase.bits() as _,
                config.as_raw(),
                data.as_raw(),
                input_matrix.as_raw(),
                solution.as_raw(),
                rhs.as_raw(),
            ))
        }
    }

    /// Returns the current device memory handler.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot report the handler.
    pub fn device_memory_handler(&self) -> Result<DeviceMemoryHandler> {
        let handle = self.handle.lock();
        unsafe {
            let mut handler = sys::cudssDeviceMemHandler_t::default();
            try_ffi!(sys::cudssGetDeviceMemHandler(handle.raw, &raw mut handler,))?;
            Ok(DeviceMemoryHandler::from_raw(handler))
        }
    }

    /// Sets the device memory handler used by this context.
    ///
    /// cuDSS uses this handler for internal device allocations during
    /// [`Self::execute`]. Allocations remain owned by the [`Data`] object used
    /// for execution until that data object is destroyed.
    ///
    /// Detach the existing handler with [`Self::clear_device_memory_handler`]
    /// before setting a different handler.
    ///
    /// # Safety
    ///
    /// The handler callbacks and backing memory pool must satisfy
    /// [`DeviceMemoryHandler`] invariants. The memory pool must outlive this
    /// context and every [`Data`] object that may contain allocations from it.
    /// It is undefined behavior to replace a handler without first detaching it,
    /// to let the context outlive the attached memory pool, or to use a memory
    /// pool that is not stream-ordered.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the handler.
    pub unsafe fn set_device_memory_handler(&self, handler: &DeviceMemoryHandler) -> Result<()> {
        let handle = self.handle.lock();
        unsafe {
            try_ffi!(sys::cudssSetDeviceMemHandler(
                handle.raw,
                ptr::from_ref(handler.as_raw()),
            ))
        }
    }

    /// Clears the current device memory handler.
    ///
    /// After the handler is cleared, cuDSS uses its default internal allocation
    /// path for future device allocations.
    ///
    /// # Safety
    ///
    /// No live [`Data`] object may contain allocations that still require the
    /// current handler's backing memory pool unless that pool remains valid
    /// until those objects are destroyed.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot clear the handler.
    pub unsafe fn clear_device_memory_handler(&self) -> Result<()> {
        let handle = self.handle.lock();
        unsafe { try_ffi!(sys::cudssSetDeviceMemHandler(handle.raw, ptr::null())) }
    }

    /// Returns the raw cuDSS handle.
    pub fn as_raw(&self) -> sys::cudssHandle_t {
        self.handle.lock().raw
    }

    /// Consumes this context and returns the owned raw cuDSS handle.
    ///
    /// # Errors
    ///
    /// Returns [`Error::HandleShared`] if solver data objects still share the same underlying handle.
    pub fn into_raw(self) -> Result<sys::cudssHandle_t> {
        let handle = self.handle.into_inner()?;
        let handle = ManuallyDrop::new(handle);
        Ok(handle.raw)
    }

    /// Returns the CUDA context associated with this cuDSS handle.
    pub fn cuda_context(&self) -> Arc<CudaContext> {
        Arc::clone(&self.handle.lock().cuda_ctx)
    }

    pub(crate) fn handle(&self) -> SharedHandle {
        self.handle.clone()
    }
}

impl SharedHandle {
    fn new(handle: Handle) -> Self {
        Self(Arc::new(Mutex::new(handle)))
    }

    pub(crate) fn lock(&self) -> MutexGuard<'_, Handle> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn into_inner(self) -> Result<Handle> {
        let mutex = Arc::try_unwrap(self.0).map_err(|_| Error::HandleShared)?;
        Ok(mutex.into_inner().unwrap_or_else(PoisonError::into_inner))
    }
}

impl Handle {
    pub(crate) const fn as_raw(&self) -> sys::cudssHandle_t {
        self.raw
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if let Err(err) = self.cuda_ctx.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind cuda context before destroying cudss handle: {err}");
        }

        unsafe {
            if let Err(err) = try_ffi!(sys::cudssDestroy(self.raw)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cudss context: {err}");
            }
        }
    }
}
