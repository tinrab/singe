use std::{
    any::Any,
    ffi::c_void,
    mem::take,
    ops::Deref,
    panic::{AssertUnwindSafe, catch_unwind},
    ptr,
    sync::{
        Arc, Mutex, OnceLock, PoisonError,
        atomic::{AtomicBool, Ordering},
    },
};

use singe_cuda_sys::{driver, runtime};

use crate::{
    error::{Error, Result, Status},
    graph::Graph,
    stream::{StreamBinding, StreamCaptureStatus},
    try_ffi,
};

type ErasedOwner = Arc<dyn Any + Send + Sync>;

/// A shared native resource retained by CUDA graph definitions and executable snapshots.
/// CUDA callbacks defer its destruction; [`collect_graph_resources`] releases it outside capture.
#[derive(Debug)]
pub struct GraphRetainedResource<T: Any + Send + Sync> {
    resource: Arc<T>,
    user_object: Mutex<Option<UserObject>>,
}

impl<T: Any + Send + Sync> GraphRetainedResource<T> {
    /// Stores `resource` in a shared owner, deferring native user-object creation until retention is needed.
    pub fn new(resource: T) -> Self {
        Self {
            resource: Arc::new(resource),
            user_object: Mutex::new(None),
        }
    }

    /// Retains this resource in the graph through a native CUDA user object.
    ///
    /// Native graph clones and executable snapshots keep independent references. Final resource destruction is deferred until [`collect_graph_resources`] runs outside capture.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph context cannot be bound, the user-object lock is poisoned, the graph handle is null, or CUDA cannot create or retain the user object.
    pub fn retain_on_graph(&self, graph: &Graph) -> Result<()> {
        graph.bind_context()?;
        self.retain_on_raw_graph(graph.as_raw())
    }

    /// Retains the resource in the stream's active capture, or collects completed resources outside capture.
    ///
    /// # Errors
    ///
    /// Propagates context binding, capture-query, resource-retention, and deferred collection errors. Implicit capture on the legacy stream and invalidated capture are left unchanged and return success without collecting resources.
    pub fn retain_on_capturing_stream(&self, stream: &StreamBinding) -> Result<()> {
        stream.context().bind()?;
        let mut status = runtime::cudaStreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        let mut graph = ptr::null_mut();
        // SAFETY: only status and the current capture's graph handle are requested.
        let result = unsafe {
            runtime::cudaStreamGetCaptureInfo(
                stream.as_raw(),
                &raw mut status,
                ptr::null_mut(),
                &raw mut graph,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        if result == runtime::cudaError_t::CUDA_ERROR_STREAM_CAPTURE_IMPLICIT {
            // The legacy stream cannot observe another stream's global capture; do not release resources there.
            // SAFETY: `cudaGetLastError` takes no arguments.
            unsafe {
                runtime::cudaGetLastError();
            }
            return Ok(());
        }
        try_ffi!(result)?;
        match status {
            runtime::cudaStreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE => {
                self.retain_on_raw_graph(graph)
            }
            runtime::cudaStreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE => {
                collect_graph_resources(stream)
            }
            runtime::cudaStreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_INVALIDATED => Ok(()),
        }
    }

    fn retain_on_raw_graph(&self, graph: runtime::cudaGraph_t) -> Result<()> {
        let mut object = self
            .user_object
            .lock()
            .map_err(|_| Error::ResourcePoisoned {
                name: "graph retained resource",
            })?;
        if object.is_none() {
            let owner: ErasedOwner = Arc::<T>::clone(&self.resource);
            let payload = Box::into_raw(Box::new(owner));
            // SAFETY: the callback receives exactly this Box and transfers its owner to the deferred queue.
            let created =
                unsafe { UserObject::create(payload.cast(), Some(defer_resource_release)) };
            match created {
                Ok(created) => *object = Some(created),
                Err(error) => {
                    // SAFETY: failed native creation did not adopt the Box.
                    unsafe {
                        drop(Box::from_raw(payload));
                    }
                    return Err(error);
                }
            }
        }
        object
            .as_ref()
            .ok_or(Error::ResourcePoisoned {
                name: "graph retained resource",
            })?
            .retain_on_graph(graph)
    }
}

impl<T: Any + Send + Sync> Deref for GraphRetainedResource<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.resource
    }
}

#[derive(Debug)]
pub struct UserObject(runtime::cudaUserObject_t);

// SAFETY: native retain/release supports cross-thread ownership; callers serialize lazy creation with a mutex.
unsafe impl Send for UserObject {}

impl UserObject {
    /// On success CUDA owns `payload` until invoking `destroy`; on failure ownership stays with the caller.
    pub(crate) unsafe fn create(
        payload: *mut c_void,
        destroy: runtime::cudaHostFn_t,
    ) -> Result<Self> {
        let mut handle = ptr::null_mut();
        // SAFETY: `cudaUserObjectCreate` receives valid output locations.
        unsafe {
            try_ffi!(runtime::cudaUserObjectCreate(
                &raw mut handle,
                payload,
                destroy,
                1,
                driver::CUuserObject_flags::CU_USER_OBJECT_NO_DESTRUCTOR_SYNC as u32,
            ))?;
        }
        Ok(Self(handle))
    }

    pub(crate) fn retain_on_graph(&self, graph: runtime::cudaGraph_t) -> Result<()> {
        if graph.is_null() {
            return Err(Error::NullHandle);
        }
        // SAFETY: this owner retains one reference while CUDA gives the graph an independent reference.
        unsafe { try_ffi!(runtime::cudaGraphRetainUserObject(graph, self.0, 1, 0)) }
    }
}

impl Drop for UserObject {
    fn drop(&mut self) {
        // SAFETY: release only this owner's reference; captured graphs and launches retain their own references.
        if let Err(error) = unsafe { try_ffi!(runtime::cudaUserObjectRelease(self.0, 1)) } {
            #[cfg(debug_assertions)]
            eprintln!("failed to release cuda graph user object: {error}");
        }
    }
}

#[derive(Default)]
struct DeferredResources {
    owners: Mutex<Vec<ErasedOwner>>,
    pending: AtomicBool,
}

fn deferred_resources() -> &'static DeferredResources {
    static RESOURCES: OnceLock<DeferredResources> = OnceLock::new();
    RESOURCES.get_or_init(DeferredResources::default)
}

unsafe extern "C" fn defer_resource_release(payload: *mut c_void) {
    // SAFETY: successful user-object creation passes exactly one Box<ErasedOwner> to this callback.
    let owner = unsafe { *Box::from_raw(payload.cast::<ErasedOwner>()) };
    let deferred = deferred_resources();
    let mut owners = deferred
        .owners
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    owners.push(owner);
    drop(owners);
    deferred.pending.store(true, Ordering::Release);
}

/// Releases graph resources whose native graph and launch references have ended.
/// Active or implicit capture leaves them queued. Destruction uses relaxed thread capture mode
/// so it cannot invalidate an unrelated global capture. Only resources already returned by CUDA's asynchronous
/// destruction callbacks are collected; callers may collect again after graph destruction when no later execution is planned.
///
/// # Errors
///
/// Returns an error if capture status cannot be queried, the thread capture mode cannot be exchanged, or the original stream context cannot be rebound after destruction. Returns `Error::ResourceDestructionPanicked` if any resource destructor panics and rebinding succeeds. Resources queued during active, invalidated, or implicit capture remain deferred without an error.
pub fn collect_graph_resources(stream: &StreamBinding) -> Result<()> {
    let deferred = deferred_resources();
    if !deferred.pending.load(Ordering::Acquire) {
        return Ok(());
    }
    match stream.capture_status() {
        Ok(StreamCaptureStatus::None) => {}
        Err(Error::Cuda {
            code: Status::StreamCaptureImplicit,
            ..
        }) => {
            // SAFETY: `cudaGetLastError` takes no arguments.
            unsafe {
                runtime::cudaGetLastError();
            }
            return Ok(());
        }
        Ok(_) => return Ok(()),
        Err(error) => return Err(error),
    }
    let mut mode = runtime::cudaStreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED;
    // SAFETY: this thread temporarily permits native destruction of resources no recorded graph can still use.
    unsafe {
        try_ffi!(runtime::cudaThreadExchangeStreamCaptureMode(&raw mut mode))?;
    }
    let _restore = CaptureModeRestore(mode);
    let owners = {
        let mut owners = deferred
            .owners
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        deferred.pending.store(false, Ordering::Release);
        take(&mut *owners)
    };
    let mut panicked = false;
    for owner in owners {
        panicked |= catch_unwind(AssertUnwindSafe(|| drop(owner))).is_err();
    }
    // A resource destructor may bind a different CUDA context.
    stream.context().bind()?;
    if panicked {
        return Err(Error::ResourceDestructionPanicked);
    }
    Ok(())
}

/// Whether CUDA destruction callbacks have queued native resources for collection.
#[must_use]
pub fn has_deferred_graph_resources() -> bool {
    deferred_resources().pending.load(Ordering::Acquire)
}

struct CaptureModeRestore(runtime::cudaStreamCaptureMode);

impl Drop for CaptureModeRestore {
    fn drop(&mut self) {
        let mut mode = self.0;
        // SAFETY: restore this thread's previous capture mode on every exit path.
        if let Err(error) =
            unsafe { try_ffi!(runtime::cudaThreadExchangeStreamCaptureMode(&raw mut mode)) }
        {
            #[cfg(debug_assertions)]
            eprintln!("failed to restore cuda thread capture mode: {error}");
        }
    }
}
