use std::{
    ffi::c_void,
    mem, ptr,
    sync::{Arc, Mutex},
};

use singe_cuda_sys::runtime;

use crate::{
    error::{Error, Result, Status},
    graph::{Graph, retained_resource::UserObject},
    memory::MemoryCopyKind,
    stream::{StreamBinding, StreamCaptureStatus},
    try_ffi,
    types::DevicePtr,
};

/// Immutable host bytes retained by queued copies and CUDA graphs that reference them.
/// CUDA user objects preserve the source through native graph clones, updates and executable launches.
#[derive(Debug)]
pub struct GraphHostBuffer {
    bytes: Arc<Vec<u8>>,
    user_object: Mutex<Option<UserObject>>,
}

impl PartialEq for GraphHostBuffer {
    fn eq(&self, other: &Self) -> bool {
        self.bytes == other.bytes
    }
}

impl Eq for GraphHostBuffer {}

impl GraphHostBuffer {
    /// Takes ownership of immutable host bytes for retained graph and stream copies.
    ///
    /// Native user-object allocation is deferred until the bytes are first retained by a graph.
    #[must_use]
    pub fn new(bytes: Vec<u8>) -> Self {
        Self {
            bytes: Arc::new(bytes),
            user_object: Mutex::new(None),
        }
    }

    /// Borrows the immutable host source bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Retains these bytes in a native graph without changing its nodes.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph context cannot be bound, its handle is null, the user-object lock is poisoned, or CUDA cannot create or retain a graph user object.
    pub fn retain_on_graph(&self, graph: &Graph) -> Result<()> {
        graph.bind_context()?;
        self.retain_on_raw_graph(graph.as_raw())
    }

    fn retain_on_raw_graph(&self, graph: runtime::cudaGraph_t) -> Result<()> {
        if graph.is_null() {
            return Err(Error::NullHandle);
        }
        let mut object = self
            .user_object
            .lock()
            .map_err(|_| Error::ResourcePoisoned {
                name: "graph host data",
            })?;
        if object.is_none() {
            let payload = Box::into_raw(Box::new(Arc::clone(&self.bytes)));
            // SAFETY: successful creation transfers this Box to a destructor that only releases ordinary host bytes.
            match unsafe { UserObject::create(payload.cast(), Some(release_host_data)) } {
                Ok(created) => *object = Some(created),
                Err(error) => {
                    // SAFETY: CUDA did not adopt the payload on failure.
                    unsafe {
                        drop(Box::from_raw(payload));
                    }
                    return Err(error);
                }
            }
        }
        let user_object = object.as_ref().ok_or(Error::ResourcePoisoned {
            name: "graph host data",
        })?;
        let result = user_object.retain_on_graph(graph);
        drop(object);
        result
    }

    /// Queues a copy on an ordinary stream or records it into the stream's current capture.
    /// Sources remain live through completion even if this object is dropped immediately afterward.
    ///
    /// # Safety
    /// `destination` must permit exclusive writes of `as_bytes().len()` bytes in the stream's context.
    /// Retain destination storage through completion and every replay of a captured copy.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream context cannot be bound, capture status or the active capture graph cannot be queried, capture is invalidated, source retention fails, or CUDA rejects the copy or completion callback. An empty source succeeds without querying CUDA. If callback registration fails after an ordinary-stream copy, cleanup attempts to drain the stream; source bytes remain retained if draining fails.
    pub unsafe fn copy_to_device(
        &self,
        destination: DevicePtr,
        stream: &StreamBinding,
    ) -> Result<()> {
        if self.bytes.is_empty() {
            return Ok(());
        }
        stream.context().bind()?;
        let capture = stream.capture_status()?;
        if capture == StreamCaptureStatus::Invalidated {
            return Err(Status::StreamCaptureInvalidated.into());
        }
        if capture == StreamCaptureStatus::Active {
            let mut graph = ptr::null_mut();
            let mut status = runtime::cudaStreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
            // SAFETY: only the capture graph and status outputs are requested; no borrowed graph owner is constructed.
            unsafe {
                try_ffi!(runtime::cudaStreamGetCaptureInfo(
                    stream.as_raw(),
                    &raw mut status,
                    ptr::null_mut(),
                    &raw mut graph,
                    ptr::null_mut(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                ))?;
            }
            self.retain_on_raw_graph(graph)?;
        }
        // SAFETY: the destination's access contract is the caller's, and capture already owns its source.
        let copy_result = unsafe {
            try_ffi!(runtime::cudaMemcpyAsync(
                destination.as_raw().cast(),
                self.bytes.as_ptr().cast(),
                self.bytes.len() as _,
                MemoryCopyKind::HostToDevice.into(),
                stream.as_raw(),
            ))
        };
        if capture == StreamCaptureStatus::Active {
            return copy_result;
        }
        let payload = Box::into_raw(Box::new(Arc::clone(&self.bytes)));
        // SAFETY: the callback only drops CPU bytes and runs after the copy, including stream failure notifications.
        let retain_result = unsafe {
            try_ffi!(runtime::cudaStreamAddCallback(
                stream.as_raw(),
                Some(release_after_copy),
                payload.cast(),
                0
            ))
        };
        if retain_result.is_err() {
            // SAFETY: failed registration left the payload with us; drain before releasing any queued source.
            let payload = unsafe { Box::from_raw(payload) };
            if stream.synchronize().is_err() {
                mem::forget(payload);
            }
        }
        copy_result.and(retain_result)
    }
}

unsafe extern "C" fn release_host_data(payload: *mut c_void) {
    // SAFETY: each successful native adoption invokes this exactly once; releasing Vec<u8> calls no CUDA API.
    unsafe {
        drop(Box::from_raw(payload.cast::<Arc<Vec<u8>>>()));
    }
}

unsafe extern "C" fn release_after_copy(
    _: runtime::cudaStream_t,
    _: runtime::cudaError_t,
    payload: *mut c_void,
) {
    // SAFETY: this payload came from the same Box allocation as CUDA user-object payloads.
    unsafe {
        release_host_data(payload);
    }
}
