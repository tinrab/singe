use std::{convert::Infallible, path::Path, ptr, sync::Arc};

use singe_core::path_to_cstring;
use singe_cuda::{
    context::Context as CudaContext,
    device::Device,
    error::Error as CudaError,
    handle::{CreateHandle, Handle, NativeHandle, RawHandle, StreamHandle},
    stream::{Stream, StreamBinding},
    sys::runtime,
    view::DeviceRegion,
};
use singe_cudss_sys as sys;

use crate::{
    config::Config,
    error::{Error, Result},
    matrix::Matrix,
    memory::DeviceMemoryHandler,
    solver_state::SolverState,
    try_ffi,
    types::{DataParameter, Phase},
    utility::to_i32,
};

/// Marker type for the native cuDSS handle owned by [`Context`].
#[derive(Debug)]
pub struct CudssHandle(Infallible);

// SAFETY: cuDSS handles may be used and destroyed from any thread, one call at a time.
unsafe impl NativeHandle for CudssHandle {
    type Raw = sys::cudssHandle_t;

    type State = ();

    type Error = Error;

    const OBJECT: &'static str = "cudss context";

    fn null_handle() -> Error {
        Error::NullHandle
    }

    fn is_null(raw: Self::Raw) -> bool {
        raw.is_null()
    }

    unsafe fn destroy(raw: Self::Raw, (): ()) -> Result<()> {
        // SAFETY: the caller passes a live, uniquely owned handle with its context bound.
        unsafe { try_ffi!(sys::cudssDestroy(raw)) }
    }
}

// SAFETY: `cudssCreate` binds to the current context.
unsafe impl CreateHandle for CudssHandle {
    unsafe fn create(raw: *mut Self::Raw) -> Result<()> {
        // SAFETY: the caller provides a writable output location and binds the context.
        unsafe { try_ffi!(sys::cudssCreate(raw)) }
    }
}

// SAFETY: `cudssSetStream` only installs the stream; `Handle` retains it.
unsafe impl StreamHandle for CudssHandle {
    unsafe fn set_stream(raw: Self::Raw, stream: runtime::cudaStream_t) -> Result<()> {
        // SAFETY: the caller passes a live handle and a live stream of its bound context.
        unsafe { try_ffi!(sys::cudssSetStream(raw, stream)) }
    }
}

/// cuDSS library handle bound to one CUDA context.
///
/// A context owns the cuDSS handle used by solver state and execution.
/// For single-GPU handles, cuDSS binds the handle to the current CUDA device at creation time.
/// The handle is movable between threads, but it is not `Clone` or `Sync`.
#[derive(Debug)]
pub struct Context {
    handle: Handle<CudssHandle>,
    mode: ContextMode,
    multi_gpu_streams: Vec<Stream>,
}

/// Native handle creation mode and multi-GPU device order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextMode {
    /// Handle bound to the CUDA device current when it was created.
    Single,
    /// Multi-GPU handle over these CUDA devices, in native device order.
    Multi(Vec<Device>),
}

/// Raw ownership of a cuDSS handle and every owner it needs.
///
/// Destroy the native handle before releasing these owners, or transfer the complete value back through [`Context::from_raw`].
#[derive(Debug)]
pub struct RawContext {
    /// Native handle, its CUDA context, and the stream installed with [`Context::set_stream`].
    pub handle: RawHandle<CudssHandle>,
    /// Creation mode and, for a multi-GPU handle, the ordered device list.
    pub mode: ContextMode,
    /// Per-device streams installed with [`Context::set_multi_gpu_streams`], in device order.
    pub multi_gpu_streams: Vec<Stream>,
}

impl Context {
    /// Creates a single-GPU cuDSS context bound to `cuda_context`.
    ///
    /// The CUDA context is made current before the cuDSS handle is created.
    /// cuDSS binds single-GPU handles to the current CUDA device at creation time.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be made current, if cuDSS cannot create the handle, or if cuDSS returns a null handle.
    pub fn create(cuda_context: &Arc<CudaContext>) -> Result<Self> {
        Ok(Self {
            handle: Handle::create(cuda_context)?,
            mode: ContextMode::Single,
            multi_gpu_streams: Vec::new(),
        })
    }

    /// Creates a multi-GPU cuDSS context for `devices`.
    ///
    /// The first listed device must be the device of `cuda_context`.
    /// Set matching device indices on the solver [`Config`] with [`Config::set_device_indices`].
    ///
    /// Some solver features are not supported in cuDSS multi-GPU mode.
    ///
    /// # Errors
    ///
    /// Returns an error if the first device is not the device of `cuda_context`, if the CUDA context cannot be made current, if cuDSS rejects the device list or cannot create the handle, or if cuDSS returns a null handle.
    pub fn create_multi_gpu(cuda_context: &Arc<CudaContext>, devices: &[Device]) -> Result<Self> {
        if devices.first().copied() != Some(cuda_context.device()) {
            return Err(Error::DeviceMismatch {
                name: "devices".into(),
            });
        }

        let device_ids: Vec<i32> = devices.iter().map(|device| device.id()).collect();
        cuda_context.bind()?;
        let mut raw = ptr::null_mut();
        // SAFETY: `raw` is a valid output location and `device_ids` is borrowed for the call with the given length.
        unsafe {
            try_ffi!(sys::cudssCreateMg(
                &raw mut raw,
                to_i32(device_ids.len(), "device count")?,
                device_ids.as_ptr(),
            ))?;
        }

        // SAFETY: cuDSS created the uniquely owned handle in the bound `cuda_context`, with no stream installed.
        let handle = unsafe {
            Handle::from_raw(RawHandle {
                handle: raw,
                cuda_context: Arc::clone(cuda_context),
                stream: None,
                state: (),
            })?
        };
        Ok(Self {
            handle,
            mode: ContextMode::Multi(devices.to_vec()),
            multi_gpu_streams: Vec::new(),
        })
    }

    /// Adopts a cuDSS handle with its CUDA context and stream owners.
    ///
    /// # Safety
    ///
    /// `raw.handle.handle` must be a uniquely owned cuDSS handle created in `raw.handle.cuda_context` with the declared creation mode.
    /// `raw.handle.stream` and `raw.multi_gpu_streams` must be the streams installed on it.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle is null or the stream belongs to another CUDA context.
    pub unsafe fn from_raw(raw: RawContext) -> Result<Self> {
        let RawContext {
            handle,
            mode,
            multi_gpu_streams,
        } = raw;
        Ok(Self {
            // SAFETY: the caller guarantees the handle, context, and stream belong together and are uniquely owned.
            handle: unsafe { Handle::from_raw(handle)? },
            mode,
            multi_gpu_streams,
        })
    }

    /// Transfers the native handle together with its CUDA context and stream owners.
    #[must_use]
    pub fn into_raw(self) -> RawContext {
        let Self {
            handle,
            mode,
            multi_gpu_streams,
        } = self;
        RawContext {
            handle: handle.into_raw(),
            mode,
            multi_gpu_streams,
        }
    }

    /// Returns the raw cuDSS handle without transferring ownership.
    #[must_use]
    pub fn as_raw(&self) -> sys::cudssHandle_t {
        self.handle.as_raw()
    }

    /// Returns the CUDA context associated with this cuDSS handle.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        self.handle.cuda_context()
    }

    /// Returns the creation mode and, for a multi-GPU handle, its device list.
    #[must_use]
    pub const fn mode(&self) -> &ContextMode {
        &self.mode
    }

    /// Binds the CUDA context associated with this handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound.
    pub fn bind(&self) -> Result<()> {
        self.handle.bind()
    }

    /// Returns the stream installed with [`Self::set_stream`], or `None` for the legacy default stream.
    #[must_use]
    pub fn stream(&self) -> Option<&Stream> {
        self.handle.stream()
    }

    /// Returns the stream this handle's single-GPU operations run on.
    #[must_use]
    pub fn stream_binding(&self) -> StreamBinding {
        self.handle.stream_binding()
    }

    /// Runs `f`, then waits for the stream installed with [`Self::set_stream`], even when `f` fails.
    ///
    /// Per-device streams of a multi-GPU handle are not waited for.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream is capturing, `f` fails, or the wait fails.
    pub fn sync_scope<R>(&self, f: impl FnOnce() -> Result<R>) -> Result<R> {
        self.handle.sync_scope(f)
    }

    /// Orders later work on the installed stream after the producers of these regions.
    ///
    /// # Errors
    ///
    /// Returns an error if a region belongs to another CUDA context or the dependencies cannot be enqueued.
    pub fn prepare_regions(&self, regions: &[DeviceRegion<'_>]) -> Result<()> {
        self.handle.prepare_regions(regions)
    }

    /// Returns the per-device streams installed with [`Self::set_multi_gpu_streams`].
    #[must_use]
    pub fn multi_gpu_streams(&self) -> &[Stream] {
        &self.multi_gpu_streams
    }

    /// Installs `stream` for cuDSS operations on this handle and retains it, or the legacy default stream for `None`.
    ///
    /// On an error the handle keeps its previous stream.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context, if the CUDA context cannot be bound, or if cuDSS rejects the stream.
    pub fn set_stream(&mut self, stream: Option<&Stream>) -> Result<()> {
        self.handle.set_stream(stream)
    }

    /// Installs per-device CUDA streams on a multi-GPU cuDSS context and retains them.
    ///
    /// `streams[i]` is used for operations on the device at index `i` of the list passed to [`Self::create_multi_gpu`].
    /// If this is not called, cuDSS creates and manages internal per-device streams.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream count does not match the device count, a stream is on a different device than its slot, the first stream belongs to another CUDA context, or cuDSS rejects the streams.
    pub fn set_multi_gpu_streams(&mut self, streams: &[&Stream]) -> Result<()> {
        if let ContextMode::Multi(devices) = &self.mode {
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
                .any(|(stream, device)| stream.context().device() != *device)
                || streams
                    .first()
                    .is_some_and(|stream| stream.context() != self.cuda_context())
            {
                return Err(Error::StreamContextMismatch);
            }
        }

        let raw_streams: Vec<_> = streams.iter().map(|stream| stream.as_raw()).collect();
        self.bind()?;
        // SAFETY: the handle is live in the bound context, `raw_streams` is borrowed for the call with the given length, and the streams are retained below.
        unsafe {
            try_ffi!(sys::cudssSetMgStreams(
                self.as_raw(),
                raw_streams.as_ptr(),
                to_i32(raw_streams.len(), "stream count")?
            ))?;
        }

        self.multi_gpu_streams = streams.iter().map(|stream| (*stream).clone()).collect();
        Ok(())
    }

    /// Sets the communication layer library for MGMN mode.
    ///
    /// The communication layer must provide backends for both device and host memory buffers.
    /// After setting the layer, set the communicators with [`SolverState::set_device_communicator`] and [`SolverState::set_host_communicator`].
    ///
    /// # Safety
    ///
    /// `path` must name a cuDSS communication layer library built for this cuDSS version.
    /// Loading it runs its initialization code, and cuDSS calls it through the layer interface without checking it.
    ///
    /// # Errors
    ///
    /// Returns an error if `path` cannot be converted to a C string or if cuDSS rejects the communication layer.
    pub unsafe fn set_communication_layer(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path_to_cstring(path.as_ref())?;
        self.bind()?;
        // SAFETY: the handle is live in the bound context, `path` is a NUL-terminated string borrowed for the call, and the caller guarantees it names a communication layer library.
        unsafe { try_ffi!(sys::cudssSetCommLayer(self.as_raw(), path.as_ptr())) }
    }

    /// Uses the default communication layer for MGMN mode.
    ///
    /// cuDSS resolves the library name from its environment configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot set the default communication layer.
    pub fn set_default_communication_layer(&mut self) -> Result<()> {
        self.bind()?;
        // SAFETY: the handle is live in the bound context, and null selects the default layer.
        unsafe { try_ffi!(sys::cudssSetCommLayer(self.as_raw(), ptr::null())) }
    }

    /// Sets the threading layer library for multi-threaded mode.
    ///
    /// # Safety
    ///
    /// `path` must name a cuDSS threading layer library built for this cuDSS version.
    /// Loading it runs its initialization code, and cuDSS calls it through the layer interface without checking it.
    ///
    /// # Errors
    ///
    /// Returns an error if `path` cannot be converted to a C string or if cuDSS rejects the threading layer.
    pub unsafe fn set_threading_layer(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path_to_cstring(path.as_ref())?;
        self.bind()?;
        // SAFETY: the handle is live in the bound context, `path` is a NUL-terminated string borrowed for the call, and the caller guarantees it names a threading layer library.
        unsafe { try_ffi!(sys::cudssSetThreadingLayer(self.as_raw(), path.as_ptr())) }
    }

    /// Uses the default threading layer for multi-threaded mode.
    ///
    /// cuDSS resolves the library name from its environment configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot set the default threading layer.
    pub fn set_default_threading_layer(&mut self) -> Result<()> {
        self.bind()?;
        // SAFETY: the handle is live in the bound context, and null selects the default layer.
        unsafe { try_ffi!(sys::cudssSetThreadingLayer(self.as_raw(), ptr::null())) }
    }

    /// Enqueues one or more solver phases.
    ///
    /// The usual solve sequence is analysis, factorization, then solve.
    /// Analysis performs reordering and symbolic factorization; factorization computes the numeric factors; solve uses those factors to compute the solution.
    ///
    /// Phase order must be:
    ///
    /// 1. [`Phase::REORDERING`] and [`Phase::SYMBOLIC_FACTORIZATION`], or [`Phase::ANALYSIS`]
    /// 2. [`Phase::FACTORIZATION`]
    /// 3. optional [`Phase::REFACTORIZATION`]
    /// 4. [`Phase::SOLVE`]
    ///
    /// Phases may be combined when the combined value preserves this order, for example [`Phase::ANALYSIS`] | [`Phase::FACTORIZATION`].
    /// Reusing analysis results is supported; after changing matrix values, rerun factorization/refactorization and solve.
    ///
    /// cuDSS reads settings from `config` and stores internal state in `state`.
    /// Additional data parameters, such as user permutations and communicators, should be set on `state` before the phase that uses them.
    ///
    /// Matrix buffers must be device-visible unless hybrid execute mode is enabled.
    /// With hybrid execute mode, the input matrix, solution, or right-hand side may use host memory pointers.
    ///
    /// Matrix requirements:
    ///
    /// * The system matrix must be sparse (currently, only 3-array CSR format is supported).
    /// * Right-hand side and solution matrices must be dense, with column-major layout.
    /// * The input sparse matrix, or each matrix in a batch, must have data consistent with its offsets, indices, indexing base, and matrix type.
    /// * Sparse column indices may be unsorted but must not contain repeated entries.
    /// * The input sparse matrix in CSR format must have its row offsets array start with the indexing base (0 or 1).
    /// * The system matrix, right-hand side, and solution must use matching value and index types where applicable.
    /// * Non-uniform batches may vary by shape, but aggregate rows and nonzeros must fit in `i32`.
    /// * In MGMN mode all processes must provide valid global shape metadata.
    ///   If no row distribution is set on a matrix, full matrix data must be present on communicator rank `0`; other processes may use null data pointers.
    ///   If row distribution is set, the system matrix, solution, and right-hand side must follow that distribution.
    ///
    /// Sparse matrix views may expose only a triangular portion of symmetric or Hermitian matrices without changing the underlying storage.
    ///
    /// # Safety
    ///
    /// The crate's raw execution contract applies.
    /// The installed stream is ordered after the producers of buffers bound through typed device views and of pointer tables themselves.
    /// The caller orders the producers of every other buffer before this call: pointer-table entries, host buffers, and buffers of descriptors reclaimed with [`Matrix::from_raw`].
    /// Pure solve destinations may be uninitialized; refinement-only or continuation phases require the preceding initialized solution or intermediate state.
    /// All phase and matrix requirements above apply.
    ///
    /// A successful enqueue does not establish completion or successful numerical factorization.
    /// Check [`SolverState::info`] after completion before using produced values.
    /// Full solve produces logical local solution elements only; pitch padding, unselected batches, and unused allocation tails are outside that guarantee.
    ///
    /// # Errors
    ///
    /// Returns an error if `state` was created by another context, a matrix belongs to another CUDA context, dependency ordering fails, the context cannot be bound, or cuDSS rejects the phase combination, configuration, solver state, matrix descriptors, or execution.
    pub unsafe fn execute_async(
        &self,
        phase: Phase,
        config: &Config,
        state: &mut SolverState<'_>,
        input_matrix: &Matrix<'_>,
        solution: &mut Matrix<'_>,
        rhs: &Matrix<'_>,
    ) -> Result<()> {
        if !ptr::eq(state.context(), self) {
            return Err(Error::ContextMismatch);
        }

        if [input_matrix, &*solution, rhs]
            .iter()
            .any(|matrix| matrix.cuda_context() != self.cuda_context())
        {
            return Err(CudaError::MemoryContextMismatch.into());
        }

        let regions: Vec<_> = input_matrix
            .regions()
            .chain(solution.regions())
            .chain(rhs.regions())
            .collect();
        self.prepare_regions(&regions)?;
        self.bind()?;
        // SAFETY: the handle, config, state, and matrix descriptors are live wrappers of this context; the caller guarantees the crate's raw execution contract for their buffers.
        unsafe {
            try_ffi!(sys::cudssExecute(
                self.as_raw(),
                phase.bits().cast_signed(),
                config.as_raw(),
                state.as_raw(),
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
        self.bind()?;
        let mut handler = sys::cudssDeviceMemHandler_t::default();
        // SAFETY: the handle is live in the bound context, and `handler` is a valid output location.
        unsafe {
            try_ffi!(sys::cudssGetDeviceMemHandler(
                self.as_raw(),
                &raw mut handler
            ))?;
        }
        // SAFETY: cuDSS reports the handler installed on this context, which already satisfies the handler invariants.
        Ok(unsafe { DeviceMemoryHandler::from_raw(handler) })
    }

    /// Sets the device memory handler used by this context.
    ///
    /// cuDSS uses this handler for internal device allocations during [`Self::execute_async`].
    /// Allocations remain owned by the [`SolverState`] used for execution until it is destroyed.
    ///
    /// Detach the existing handler with [`Self::clear_device_memory_handler`] before setting a different handler.
    ///
    /// # Safety
    ///
    /// The handler callbacks and backing memory pool must satisfy [`DeviceMemoryHandler`] invariants.
    /// The memory pool must outlive this context and every [`SolverState`] that may contain allocations from it.
    /// It is undefined behavior to replace a handler without first detaching it, to let the context outlive the attached memory pool, or to use a memory pool that is not stream-ordered.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the handler.
    pub unsafe fn set_device_memory_handler(
        &mut self,
        handler: &DeviceMemoryHandler,
    ) -> Result<()> {
        self.bind()?;
        // SAFETY: the handle is live in the bound context; cuDSS copies the handler struct, and the caller guarantees the callbacks and memory pool meet `DeviceMemoryHandler`'s invariants.
        unsafe {
            try_ffi!(sys::cudssSetDeviceMemHandler(
                self.as_raw(),
                ptr::from_ref(handler.as_raw()),
            ))
        }
    }

    /// Clears the current device memory handler.
    ///
    /// After the handler is cleared, cuDSS uses its default internal allocation path for future device allocations.
    ///
    /// # Safety
    ///
    /// Allocations made through the current handler must stay valid until the [`SolverState`]s that own them are destroyed.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot clear the handler.
    pub unsafe fn clear_device_memory_handler(&mut self) -> Result<()> {
        self.bind()?;
        // SAFETY: the handle is live in the bound context, and null detaches the handler.
        unsafe { try_ffi!(sys::cudssSetDeviceMemHandler(self.as_raw(), ptr::null())) }
    }

    pub(crate) fn bind_data_query(&self, param: DataParameter) -> Result<()> {
        if let ContextMode::Multi(devices) = &self.mode
            && matches!(
                param,
                DataParameter::MemoryEstimates | DataParameter::HybridDeviceMemoryMin
            )
        {
            if !devices.contains(&Device::current()?) {
                return Err(Error::DeviceMismatch {
                    name: "context devices".into(),
                });
            }

            // These native queries intentionally select data for the calling GPU.
            return Ok(());
        }

        self.bind()
    }
}
