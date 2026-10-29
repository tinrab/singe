//! Owned operation capture and resource-retaining replay.

use std::{
    mem,
    sync::{Arc, Mutex},
};

use crate::{
    error::{Error, Result},
    graph::{ExecutableGraph, Graph},
    stream::{Stream, StreamCaptureMode, StreamCaptureStatus},
    submission::{DeviceOperation, Execution, Operation, Resources, Submission},
};

/// Prepared work that can be recorded without allocation, compilation, waits or host readback.
///
/// # Safety
/// Register every owner and byte-range access with `Recording` before recording native work.
/// Outputs must expose pending storage only; recording does not execute producers.
/// Native operations and resources must remain valid for every replay.
pub unsafe trait Recordable: Sized {
    /// Value produced by recording, without implying producer execution or initialized results.
    type Output: 'static;

    /// Records this prepared operation and registers its replay resources with `recording`.
    ///
    /// # Errors
    ///
    /// Returns operation-specific capture or native recording errors. Implementations must preserve replay ownership for any nodes recorded before an error is returned.
    fn record(self, recording: &mut Recording) -> Result<Self::Output>;
}

/// Resource ledger for one active capture.
#[derive(Debug)]
pub struct Recording {
    stream: Stream,
    resources: Resources,
}

impl Recording {
    /// Returns the stream being captured by this recording.
    #[must_use]
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Registers replay resources before recording one native operation.
    ///
    /// # Safety
    /// Retain all referenced owners and declare all accessed ranges.
    /// The closure must be graph-safe and enqueue only on the supplied stream.
    /// Returned values must not claim that capture initialized their storage.
    ///
    /// # Errors
    ///
    /// Returns `Error::MemoryContextMismatch` if a nonportable access belongs to another context. Otherwise propagates the result of `enqueue`; registered resources remain in the recording ledger even if that callback fails.
    pub unsafe fn enqueue<T>(
        &mut self,
        resources: Resources,
        enqueue: impl FnOnce(&Stream) -> Result<T>,
    ) -> Result<T> {
        for access in &resources.accesses {
            access.validate_context(self.stream.context())?;
        }
        self.resources.extend(resources);
        enqueue(&self.stream)
    }
}

impl Drop for Recording {
    fn drop(&mut self) {
        if !matches!(self.stream.capture_status(), Ok(StreamCaptureStatus::None)) {
            mem::forget(mem::take(&mut self.resources));
            #[cfg(debug_assertions)]
            eprintln!("cuda capture did not end; retaining recorded resources");
        }
    }
}

/// A captured graph retaining storage independently of its extracted output.
#[derive(Debug)]
pub struct OwnedGraph<T> {
    graph: Graph,
    resources: Resources,
    output: Option<T>,
}

impl<T> OwnedGraph<T> {
    /// Takes the value produced during recording, returning `None` after the first successful take.
    ///
    /// The value does not establish that recorded producers have executed or completed; their outputs must preserve pending initialization.
    pub fn take_output(&mut self) -> Option<T> {
        self.output.take()
    }

    /// Borrows the recorded graph definition.
    ///
    /// The replay resource ledger remains owned by this [`OwnedGraph`].
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Instantiates replay with independent native execution state and the recorded storage owners.
    ///
    /// # Errors
    ///
    /// Propagates native graph-instantiation errors from [`Graph::instantiate`]. No executable or independent resource snapshot is returned if instantiation fails.
    pub fn instantiate(&self) -> Result<OwnedExecutable> {
        Ok(OwnedExecutable {
            executable: Arc::new(Mutex::new(self.graph.instantiate()?)),
            resources: self.resources.clone(),
        })
    }
}

/// Reusable replay with common allocation access tracking and independent submission retention.
#[derive(Debug, Clone)]
pub struct OwnedExecutable {
    executable: Arc<Mutex<ExecutableGraph>>,
    resources: Resources,
}

impl OwnedExecutable {
    /// Adopts an existing executable and its complete replay ownership and access declarations.
    ///
    /// # Safety
    /// Resources must retain everything referenced by every node, including private scratch.
    /// Declare writable graph-private storage so shared instances serialize its use.
    /// Raw mutations must preserve this ownership and access contract.
    #[must_use]
    pub unsafe fn from_executable(executable: ExecutableGraph, resources: Resources) -> Self {
        Self {
            executable: Arc::new(Mutex::new(executable)),
            resources,
        }
    }

    /// Submits one replay on `stream`, retaining the executable and its resources independently of the returned submission.
    ///
    /// Dropping the submission does not cancel queued graph work.
    ///
    /// # Errors
    ///
    /// Propagates execution setup, access-tracking, dependency-ordering, graph-launch, and completion registration errors. Returns `Error::ResourcePoisoned` if the executable lock is poisoned. A failure can occur after native work has been queued; the execution ledger retains resources through cleanup.
    pub fn launch(&self, stream: &Stream) -> Result<Submission<()>> {
        self.clone().submit_on(stream)
    }

    /// Accesses native graph mutation under the replay lock.
    ///
    /// # Safety
    /// Complete prior launches before mutation and preserve the retained ownership and access declarations.
    ///
    /// # Errors
    ///
    /// Returns `Error::ResourcePoisoned` if the executable lock is poisoned. Otherwise propagates errors returned by `access`.
    pub unsafe fn with_raw<T>(
        &self,
        access: impl FnOnce(&mut ExecutableGraph) -> Result<T>,
    ) -> Result<T> {
        let mut executable = self
            .executable
            .lock()
            .map_err(|_| Error::ResourcePoisoned {
                name: "executable graph",
            })?;
        access(&mut executable)
    }
}

unsafe impl Operation for OwnedExecutable {
    type Pending = Self::Output;

    type Output = ();

    fn enqueue(self, execution: &mut Execution) -> Result<()> {
        let mut resources = self.resources;
        resources.retain(Arc::clone(&self.executable));
        // SAFETY: the declared resources retain every buffer and the stream orders all access to them.
        unsafe {
            execution.enqueue(resources, |stream| {
                self.executable
                    .lock()
                    .map_err(|_| Error::ResourcePoisoned {
                        name: "executable graph",
                    })?
                    .launch(stream)
            })
        }
    }

    unsafe fn resolve(pending: Self::Pending) -> Result<Self::Output> {
        Ok(pending)
    }
}

// SAFETY: enqueue returns pending device metadata and resolution is the identity function.
unsafe impl DeviceOperation for OwnedExecutable {}

impl Stream {
    /// Captures already prepared owned work; no producer executes during recording.
    ///
    /// # Errors
    ///
    /// Propagates capture setup, operation recording, and capture finalization errors. Resources registered before failure remain retained until capture ends; if cleanup cannot establish that capture ended, the recording preserves those resources.
    pub fn capture_owned<O: Recordable>(&self, operation: O) -> Result<OwnedGraph<O::Output>> {
        let mut recording = Recording {
            stream: self.clone(),
            resources: Resources::new(),
        };
        let mut output = None;
        let graph = self.capture(StreamCaptureMode::Relaxed, |_| {
            output = Some(operation.record(&mut recording)?);
            Ok(())
        })?;
        Ok(OwnedGraph {
            graph,
            resources: mem::take(&mut recording.resources),
            output,
        })
    }
}
