use std::{
    mem,
    path::Path,
    ptr,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
};

use singe_core::path_to_cstring;
use singe_cuda::{context::Context as CudaContext, device::Device, stream::Stream};
use singe_cudss_sys as sys;

use crate::{
    config::Config,
    data::SolverState,
    error::{Error, Result},
    matrix::Matrix,
    memory::DeviceMemoryHandler,
    try_ffi,
    types::{DataParameter, Phase},
    utility::to_i32,
};

/// cuDSS library handle bound to one CUDA context.
///
/// A context owns the cuDSS handle used by configuration, data, and matrix operations.
/// For single-GPU handles, cuDSS binds the handle to the current CUDA device at creation time.
#[derive(Debug)]
pub struct Context {
    handle: SharedHandle,
    cuda_context: Arc<CudaContext>,
}

#[derive(Debug, Clone)]
pub(crate) struct SharedHandle(Arc<Mutex<Handle>>);

#[derive(Debug)]
pub(crate) struct Handle {
    raw: sys::cudssHandle_t,
    cuda_context: Arc<CudaContext>,
    streams: Vec<Stream>,
    mode: ContextMode,
}

/// Native handle creation mode and multi-GPU device order.
#[derive(Debug, Clone)]
pub enum ContextMode {
    /// Handle bound to the CUDA device current when it was created.
    Single,
    /// Multi-GPU handle with CUDA device ordinals in native device order.
    Multi(Vec<i32>),
}

/// An owned raw handle together with every context and stream owner it needs.
///
/// Destroy the native handle before releasing these owners, or transfer the
/// complete value back through [`Context::from_raw`].
#[derive(Debug)]
pub struct RawContext {
    /// Uniquely owned native cuDSS handle.
    pub handle: sys::cudssHandle_t,
    /// Retained CUDA context used to bind operations on the handle.
    pub cuda_context: Arc<CudaContext>,
    /// Selected streams retained for the handle's lifetime, in device order.
    pub streams: Vec<Stream>,
    /// Creation mode and, for a multi-GPU handle, the ordered device list.
    pub mode: ContextMode,
}

// SAFETY: cuDSS handles carry mutable solver state. This wrapper serializes raw handle
// access through `SharedHandle`, so the context can move between threads.
unsafe impl Send for Context {}

// `Handle` is protected by `SharedHandle` whenever it is shared. Moving the
// owned handle between threads is sound under the same context lifetime.
unsafe impl Send for Handle {}

impl Context {
    /// Creates a single-GPU cuDSS context bound to `cuda_context`.
    ///
    /// The CUDA context is made current before the cuDSS handle is created.
    /// cuDSS binds single-GPU handles to the current CUDA device at creation
    /// time.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be made current, if cuDSS
    /// cannot create the handle, or if cuDSS returns a null handle.
    pub fn create(cuda_context: &Arc<CudaContext>) -> Result<Self> {
        cuda_context.bind()?;
        // SAFETY: `cudssCreate` receives valid output locations.
        unsafe {
            let mut handle = ptr::null_mut();
            try_ffi!(sys::cudssCreate(&raw mut handle))?;
            if handle.is_null() {
                return Err(Error::NullHandle);
            }
            Ok(Self {
                handle: SharedHandle::new(Handle {
                    raw: handle,
                    cuda_context: Arc::clone(cuda_context),
                    streams: Vec::new(),
                    mode: ContextMode::Single,
                }),
                cuda_context: Arc::clone(cuda_context),
            })
        }
    }

    /// Takes ownership of an existing cuDSS handle associated with `cuda_context`.
    ///
    /// # Safety
    ///
    /// `parts.handle` must be a valid uniquely owned cuDSS handle associated
    /// with `parts.cuda_context` and the declared creation mode. `parts.streams`
    /// must retain every externally selected stream in native device order.
    /// The returned context destroys the handle when its last shared owner drops.
    ///
    /// # Errors
    ///
    /// Returns an error if `parts.handle` is null.
    pub unsafe fn from_raw(parts: RawContext) -> Result<Self> {
        if parts.handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            cuda_context: Arc::clone(&parts.cuda_context),
            handle: SharedHandle::new(Handle {
                raw: parts.handle,
                cuda_context: parts.cuda_context,
                streams: parts.streams,
                mode: parts.mode,
            }),
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
    pub fn create_multi_gpu(cuda_context: &Arc<CudaContext>, devices: &[i32]) -> Result<Self> {
        if devices.first().copied() != Some(cuda_context.device().id()) {
            return Err(Error::ContextMismatch);
        }
        cuda_context.bind()?;
        // SAFETY: `cudssCreateMg` receives valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            let mut handle = ptr::null_mut();
            try_ffi!(sys::cudssCreateMg(
                &raw mut handle,
                to_i32(devices.len(), "device count")?,
                devices.as_ptr(),
            ))?;
            if handle.is_null() {
                return Err(Error::NullHandle);
            }
            Ok(Self {
                handle: SharedHandle::new(Handle {
                    raw: handle,
                    cuda_context: Arc::clone(cuda_context),
                    streams: Vec::new(),
                    mode: ContextMode::Multi(devices.to_vec()),
                }),
                cuda_context: Arc::clone(cuda_context),
            })
        }
    }

    /// Sets and retains the CUDA stream used by this cuDSS context.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to a different CUDA context or if
    /// cuDSS rejects the stream.
    pub fn set_stream(&self, stream: &Stream) -> Result<()> {
        let mut handle = self.handle.lock();
        if stream.context() != &handle.cuda_context {
            return Err(Error::ContextMismatch);
        }
        handle.bind()?;
        // SAFETY: the selected stream belongs to the retained context.
        unsafe {
            try_ffi!(sys::cudssSetStream(handle.raw, stream.as_raw()))?;
        }
        handle.streams = vec![stream.clone()];
        drop(handle);
        Ok(())
    }

    /// Sets per-device CUDA streams for a multi-GPU cuDSS context.
    ///
    /// `streams[i]` is used for operations on the corresponding device passed
    /// to [`Self::create_multi_gpu`]. If this is not called, cuDSS creates and
    /// manages internal per-device streams.
    ///
    /// The handle retains every selected stream and its actual CUDA context.
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

        let mut handle = self.handle.lock();
        if let ContextMode::Multi(devices) = &handle.mode {
            if devices.len() != streams.len() {
                return Err(Error::LengthMismatch {
                    name: "streams".into(),
                    expected: devices.len(),
                    actual: streams.len(),
                });
            }
            if streams
                .iter()
                .zip(devices)
                .any(|(stream, device)| stream.context().device().id() != *device)
                || streams
                    .first()
                    .is_some_and(|stream| stream.context() != &handle.cuda_context)
            {
                return Err(Error::ContextMismatch);
            }
        }
        handle.bind()?;
        // SAFETY: `cudssSetMgStreams` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cudssSetMgStreams(
                handle.raw,
                raw_streams.as_ptr(),
                to_i32(raw_streams.len(), "stream count")?
            ))?;
        }
        handle.streams = streams.iter().map(|stream| (*stream).clone()).collect();
        drop(handle);
        Ok(())
    }

    /// Sets the communication layer library for MGMN mode.
    ///
    /// The communication layer must provide backends for both device and host
    /// memory buffers. After setting the layer, set the device and host
    /// communicators on [`SolverState`] with
    /// [`SolverState::set_device_communicator`](crate::data::SolverState::set_device_communicator)
    /// and [`SolverState::set_host_communicator`](crate::data::SolverState::set_host_communicator).
    ///
    /// # Errors
    ///
    /// Returns an error if `path` cannot be converted to a C string or if cuDSS
    /// rejects the communication layer.
    pub fn set_communication_layer(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path_to_cstring(path.as_ref())?;
        let handle = self.handle.lock();
        handle.bind()?;
        // SAFETY: `cudssSetCommLayer` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
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
        handle.bind()?;
        // SAFETY: `cudssSetCommLayer` receives live handles owned by the wrappers.
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
        handle.bind()?;
        // SAFETY: `cudssSetThreadingLayer` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
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
        handle.bind()?;
        // SAFETY: `cudssSetThreadingLayer` receives live handles owned by the wrappers.
        unsafe { try_ffi!(sys::cudssSetThreadingLayer(handle.raw, ptr::null())) }
    }

    /// Creates a solver data object tied to this context.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot create the data object.
    pub fn create_data(&self) -> Result<SolverState> {
        SolverState::create(self)
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
    /// # Safety
    ///
    /// Retain the context, configuration, data, every matrix descriptor, all
    /// backing arrays, pointer tables, and selected streams through completion
    /// on every path and throughout captured replay. Inputs must be initialized
    /// and prior accesses ordered before use. Solutions require exclusive access
    /// and must not overlap inputs. Pure solve destinations may be uninitialized;
    /// refinement-only or continuation phases require the preceding initialized
    /// solution/intermediate state. All phase and matrix requirements above apply.
    ///
    /// A successful enqueue does not establish completion or successful numerical
    /// factorization. Check completed solver status before using produced values.
    /// Full solve produces logical local solution elements only; pitch padding,
    /// unselected batches, and unused allocation tails are outside that guarantee.
    pub unsafe fn execute(
        &self,
        phase: Phase,
        config: &Config,
        data: &mut SolverState,
        input_matrix: &Matrix<'_>,
        solution: &mut Matrix<'_>,
        rhs: &Matrix<'_>,
    ) -> Result<()> {
        let handle = self.handle.lock();
        handle.bind()?;
        if [input_matrix, &*solution, rhs]
            .iter()
            .any(|matrix| matrix.cuda_context() != &handle.cuda_context)
        {
            return Err(Error::ContextMismatch);
        }
        // SAFETY: `cudssExecute` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cudssExecute(
                handle.raw,
                phase.bits().cast_signed(),
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
        handle.bind()?;
        // SAFETY: cuDSS writes only into the local output locations declared in this block.
        unsafe {
            let mut handler = sys::cudssDeviceMemHandler_t::default();
            try_ffi!(sys::cudssGetDeviceMemHandler(handle.raw, &raw mut handler,))?;
            Ok(DeviceMemoryHandler::from_raw(handler))
        }
    }

    /// Sets the device memory handler used by this context.
    ///
    /// cuDSS uses this handler for internal device allocations during
    /// [`Self::execute`]. Allocations remain owned by the [`SolverState`] object used
    /// for execution until that data object is destroyed.
    ///
    /// Detach the existing handler with [`Self::clear_device_memory_handler`]
    /// before setting a different handler.
    ///
    /// # Safety
    ///
    /// The handler callbacks and backing memory pool must satisfy
    /// [`DeviceMemoryHandler`] invariants. The memory pool must outlive this
    /// context and every [`SolverState`] object that may contain allocations from it.
    /// It is undefined behavior to replace a handler without first detaching it,
    /// to let the context outlive the attached memory pool, or to use a memory
    /// pool that is not stream-ordered.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the handler.
    pub unsafe fn set_device_memory_handler(&self, handler: &DeviceMemoryHandler) -> Result<()> {
        let handle = self.handle.lock();
        handle.bind()?;
        // SAFETY: `cudssSetDeviceMemHandler` receives live handles owned by the wrappers.
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
    /// No live [`SolverState`] object may contain allocations that still require the
    /// current handler's backing memory pool unless that pool remains valid
    /// until those objects are destroyed.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot clear the handler.
    pub unsafe fn clear_device_memory_handler(&self) -> Result<()> {
        let handle = self.handle.lock();
        handle.bind()?;
        // SAFETY: `cudssSetDeviceMemHandler` receives live handles owned by the wrappers.
        unsafe { try_ffi!(sys::cudssSetDeviceMemHandler(handle.raw, ptr::null())) }
    }

    /// Returns the raw cuDSS handle.
    #[must_use]
    pub fn as_raw(&self) -> sys::cudssHandle_t {
        self.handle.lock().raw
    }

    /// Transfers the raw handle together with its CUDA context and stream owners.
    /// Keep those owners alive until the raw handle is destroyed.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OwnershipShared`] if solver data objects still share the same underlying handle.
    pub fn into_raw(self) -> Result<RawContext> {
        let mut handle = self.handle.into_inner()?;
        Ok(RawContext {
            handle: mem::replace(&mut handle.raw, ptr::null_mut()),
            cuda_context: Arc::clone(&handle.cuda_context),
            streams: mem::take(&mut handle.streams),
            mode: mem::replace(&mut handle.mode, ContextMode::Single),
        })
    }

    /// Returns owned references to streams selected through this wrapper.
    /// Native default or internally managed streams are not represented here.
    #[must_use]
    pub fn streams(&self) -> Vec<Stream> {
        self.handle.lock().streams.clone()
    }

    pub(crate) fn from_handle(handle: SharedHandle) -> Self {
        let cuda_context = Arc::clone(handle.lock().cuda_context());
        Self {
            handle,
            cuda_context,
        }
    }

    /// Returns the CUDA context associated with this cuDSS handle.
    #[must_use]
    pub const fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.cuda_context
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
        let mutex = Arc::try_unwrap(self.0).map_err(|_| Error::OwnershipShared)?;
        Ok(mutex.into_inner().unwrap_or_else(PoisonError::into_inner))
    }
}

impl Handle {
    pub(crate) fn bind(&self) -> Result<()> {
        Ok(self.cuda_context.bind()?)
    }

    pub(crate) fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.cuda_context
    }

    pub(crate) fn bind_data_query(&self, param: DataParameter) -> Result<()> {
        if let ContextMode::Multi(devices) = &self.mode
            && matches!(
                param,
                DataParameter::MemoryEstimates | DataParameter::HybridDeviceMemoryMin
            )
        {
            let device = Device::current()?.id();
            if !devices.contains(&device) {
                return Err(Error::ContextMismatch);
            }
            // These native queries intentionally select data for the calling GPU.
            return Ok(());
        }
        self.bind()
    }

    pub(crate) const fn as_raw(&self) -> sys::cudssHandle_t {
        self.raw
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if self.raw.is_null() {
            return;
        }
        if let Err(error) = self.cuda_context.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind cuda context before destroying cudss handle: {error}");
        }

        // SAFETY: `cudssDestroy` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cudssDestroy(self.raw)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cudss context: {error}");
            }
        }
    }
}
