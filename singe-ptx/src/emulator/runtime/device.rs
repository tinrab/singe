use std::{
    collections::BTreeMap,
    fmt::{self, Debug, Formatter},
    io,
    io::ErrorKind,
    marker::PhantomData,
    mem,
    sync::{Arc, Condvar, Mutex, atomic::AtomicBool},
    task::Poll,
    thread,
};

#[cfg(test)]
use crate::emulator::runtime::hooks::Hooks;
use crate::emulator::{
    ParsedModule,
    budget::{Budget, ChargedQueue},
    execution_error::{ExecutionError, SubmitError},
    memory::CHUNK_BYTES,
    runtime::{
        Admission, Client, CommandKind, Phase, Shared, admission,
        buffer::{Buffer, BufferView, KernelArg, byte_size, encode, foreign},
        completion::{Completion, Pending, Shutdown, Submission},
        debug, lock,
        module::{Kernel, Module},
        options::{EmulatorOptions, ExecutionReport, LaunchConfig, ShutdownMode},
        scalar::PtxScalar,
        submit::submit_poll,
        worker::{LoadPhase, ModuleLoad, Worker, run_worker},
    },
};

/// An emulated device with its own memory, served by one worker thread.
///
/// Clones share the device. Dropping the last clone and every handle created from it
/// requests a draining shutdown without waiting for it.
#[derive(Clone)]
pub struct Emulator {
    pub(crate) client: Arc<Client>,
}

impl Debug for Emulator {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Emulator").finish_non_exhaustive()
    }
}

impl Emulator {
    /// Creates an emulator and starts its worker thread, with the limits in `options`.
    ///
    /// # Errors
    ///
    /// Returns an error if `options.active_grids` is zero or the worker thread cannot be
    /// spawned.
    pub fn create(options: EmulatorOptions) -> Result<Self, io::Error> {
        if options.active_grids == 0 {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "active grid limit must be positive",
            ));
        }
        let shared = Arc::new(Shared {
            execution_budget: Arc::new(Budget::new(options.execution_storage_bytes)),
            #[cfg(test)]
            hooks: Hooks::default(),
            admission: Mutex::new(Admission {
                debug: debug::State::default(),
                phase: Phase::Running,
                queue: ChargedQueue::new(),
                reservations: 0,
                pending_dependencies: 0,
                next_stream: 1,
                next_launch: 0,
                staged: 0,
                reclaim: Vec::new(),
                shutdown_result: None,
                shutdown_wakers: BTreeMap::new(),
            }),
            changed: Condvar::new(),
            cancel: AtomicBool::new(false),
            options,
            #[cfg(test)]
            closed_notify: Mutex::new(None),
        });
        let worker_shared = Arc::clone(&shared);
        thread::Builder::new()
            .name("ptx-emulator".into())
            .spawn(move || run_worker(&worker_shared))?;
        Ok(Self {
            client: Arc::new(Client { shared }),
        })
    }

    /// Stops accepting commands and drains or cancels the accepted ones.
    ///
    /// Calling it again returns the same result. A later [`ShutdownMode::Cancel`] can
    /// interrupt an earlier drain.
    pub fn shutdown(&self, mode: ShutdownMode) -> Shutdown {
        self.client.shared.shutdown(mode);
        Shutdown::new(Arc::clone(&self.client.shared))
    }

    /// Loads `parsed`, allocating and initializing its global and constant variables.
    ///
    /// The module's storage layout is validated before the command is queued. Each load
    /// gets its own variables.
    ///
    /// # Errors
    ///
    /// Returns an error if the layout or an initializer is invalid, or if the command is
    /// not admitted. Allocation failures are reported through the submission.
    pub fn load(&self, parsed: ParsedModule) -> Result<Submission<Module>, SubmitError> {
        let layout = parsed.layout()?;
        submit_poll(
            &self.client,
            0,
            CommandKind::Transfer,
            false,
            0,
            move |_| {
                let mut loading = ModuleLoad {
                    layout: Some(layout),
                    addresses: Vec::new(),
                    phase: LoadPhase::Reserve,
                };
                let mut parsed = Some(parsed);
                move |worker: &mut Worker, terminal| {
                    let loaded = match loading.poll(worker, terminal) {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(result) => result?,
                    };
                    let parsed = parsed
                        .take()
                        .ok_or_else(|| ExecutionError::Invalid("module already loaded".into()))?;
                    Poll::Ready(Ok(Box::new(move |client| Module {
                        client,
                        parsed,
                        loaded,
                    }) as Pending<Module>))
                }
            },
        )
    }

    /// Returns stream zero, which the emulator's own transfer methods use.
    ///
    /// Commands on different streams are not ordered with each other.
    #[must_use]
    pub fn default_stream(&self) -> Stream {
        Stream {
            client: Arc::clone(&self.client),
            id: 0,
        }
    }

    /// Creates a new stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the emulator is closed or faulted, or if stream identifiers
    /// are exhausted.
    pub fn create_stream(&self) -> Result<Stream, SubmitError> {
        let mut state = lock(&self.client.shared.admission);
        admission(state.phase, false)?;
        let id = state.next_stream;
        state.next_stream = id
            .checked_add(1)
            .ok_or_else(|| SubmitError::InvalidArgument("stream identifiers exhausted".into()))?;
        drop(state);
        Ok(Stream {
            client: Arc::clone(&self.client),
            id,
        })
    }

    /// Allocates `len` zeroed elements on the default stream.
    ///
    /// See [`Stream::zeroed`].
    ///
    /// # Errors
    ///
    /// Returns an error if the byte size overflows or the command is not admitted.
    pub fn zeroed<T: PtxScalar>(&self, len: usize) -> Result<Submission<Buffer<T>>, SubmitError> {
        self.default_stream().zeroed(len)
    }

    /// Copies `values` into a new buffer on the default stream.
    ///
    /// See [`Stream::upload`].
    ///
    /// # Errors
    ///
    /// Returns an error if the byte size overflows, staging storage cannot hold the copy,
    /// or the command is not admitted.
    pub fn upload<T: PtxScalar>(&self, values: &[T]) -> Result<Submission<Buffer<T>>, SubmitError> {
        self.default_stream().upload(values)
    }

    /// Overwrites `view` with `values` on the default stream.
    ///
    /// See [`Stream::write`].
    ///
    /// # Errors
    ///
    /// Returns an error if `view` belongs to another emulator or is read-only, the lengths
    /// differ, staging storage cannot hold the copy, or the command is not admitted.
    pub fn write<T: PtxScalar>(
        &self,
        view: &BufferView<T>,
        values: &[T],
    ) -> Result<Submission<()>, SubmitError> {
        self.default_stream().write(view, values)
    }

    /// Copies `view` into a vector on the default stream.
    ///
    /// See [`Stream::download`].
    ///
    /// # Errors
    ///
    /// Returns an error if `view` belongs to another emulator, staging storage cannot hold
    /// the result, or the command is not admitted.
    pub fn download<T: PtxScalar>(
        &self,
        view: &BufferView<T>,
    ) -> Result<Submission<Vec<T>>, SubmitError> {
        self.default_stream().download(view)
    }

    /// Returns a submission that completes after every earlier command on the default
    /// stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the emulator is closed or faulted, or the command is not
    /// admitted.
    pub fn synchronize(&self) -> Result<Submission<()>, SubmitError> {
        self.default_stream().synchronize()
    }
}

/// An ordered queue of commands.
///
/// Commands on one stream run in submission order. The worker interleaves commands
/// from different streams; use [`Stream::record_event`] and [`Stream::wait_event`] to
/// order them.
#[derive(Clone)]
pub struct Stream {
    pub(crate) client: Arc<Client>,
    pub(crate) id: u64,
}

impl Debug for Stream {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Stream").finish_non_exhaustive()
    }
}

impl Stream {
    /// Returns the stream identifier. The default stream is zero.
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Returns a submission that completes after every earlier command on this stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the emulator is closed or faulted, or the command is not
    /// admitted.
    pub fn synchronize(&self) -> Result<Submission<()>, SubmitError> {
        submit_poll(
            &self.client,
            self.id,
            CommandKind::Immediate,
            false,
            0,
            |_| {
                |_: &mut Worker, terminal| {
                    Poll::Ready(terminal.map_or_else(|| Ok(Box::new(|_| ()) as Pending<()>), Err))
                }
            },
        )
    }

    /// Records an event that completes after every earlier command on this stream.
    ///
    /// The returned submission observes the event once. The [`Event`] itself can be
    /// passed to [`wait_event`](Self::wait_event) on any stream and queried with
    /// [`Event::status`] at any time.
    ///
    /// # Errors
    ///
    /// Returns an error if the emulator is closed or faulted, or the command is not
    /// admitted.
    pub fn record_event(&self) -> Result<(Event, Submission<()>), SubmitError> {
        let submission = submit_poll(&self.client, self.id, CommandKind::Event, false, 0, |_| {
            |_: &mut Worker, terminal| {
                Poll::Ready(terminal.map_or_else(|| Ok(Box::new(|_| ()) as Pending<()>), Err))
            }
        })?;
        let event = Event {
            client: Arc::clone(&self.client),
            completion: Arc::clone(&submission.completion),
        };
        Ok((event, submission))
    }

    /// Makes later commands on this stream wait for `event`.
    ///
    /// Neither the host nor the worker blocks while the stream waits. If the event's
    /// commands fail, the returned submission reports that failure.
    ///
    /// # Errors
    ///
    /// Returns an error if `event` belongs to another emulator or the command is not
    /// admitted.
    pub fn wait_event(&self, event: &Event) -> Result<Submission<()>, SubmitError> {
        foreign(&self.client, &event.client)?;
        let completion = Arc::clone(&event.completion);
        submit_poll(
            &self.client,
            self.id,
            CommandKind::Dependency,
            false,
            0,
            move |_| {
                move |_: &mut Worker, terminal| {
                    if let Some(error) = terminal {
                        return Poll::Ready(Err(error));
                    }
                    completion.status().map_or_else(
                        || Poll::Pending,
                        |result| Poll::Ready(result.map(|()| Box::new(|_| ()) as Pending<()>)),
                    )
                }
            },
        )
    }

    /// Launches `kernel` on this stream with `args` in parameter order.
    ///
    /// See [`Kernel::launch_on`].
    ///
    /// # Errors
    ///
    /// Returns an error if `kernel` belongs to another emulator, `config` does not meet
    /// the kernel's block requirements, an argument does not match its parameter, or the
    /// command is not admitted. Execution faults are reported through the submission.
    pub fn launch(
        &self,
        kernel: &Kernel,
        config: LaunchConfig,
        args: impl IntoIterator<Item = KernelArg>,
    ) -> Result<Submission<ExecutionReport>, SubmitError> {
        kernel.launch_on(self, config, args)
    }

    /// Allocates `len` zeroed elements. Zero-length buffers are allowed.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte size overflows or the command is not admitted.
    /// Allocation failures are reported through the submission.
    pub fn zeroed<T: PtxScalar>(&self, len: usize) -> Result<Submission<Buffer<T>>, SubmitError> {
        let size = byte_size::<T>(len)?;
        submit_poll(
            &self.client,
            self.id,
            CommandKind::Transfer,
            false,
            0,
            move |_| {
                let mut allocation = None;
                #[cfg(test)]
                let mut initialized = 0usize;
                move |worker: &mut Worker, terminal| {
                    if let Some(error) = terminal {
                        return Poll::Ready(Err(error));
                    }
                    if allocation.is_none() {
                        allocation = Some(worker.reserve_allocation(size)?);
                    }
                    let Some(current) = &allocation else {
                        return Poll::Ready(Err(ExecutionError::Memory(
                            "missing allocation reservation".into(),
                        )));
                    };
                    let ready = worker.memory.initialize_chunk(current.address, true)?;
                    #[cfg(test)]
                    {
                        initialized = initialized.saturating_add(CHUNK_BYTES).min(size);
                        worker.transfer_checkpoint(initialized, size);
                    }
                    if !ready {
                        return Poll::Pending;
                    }
                    let allocation = allocation.take().ok_or_else(|| {
                        ExecutionError::Memory("missing allocation reservation".into())
                    })?;
                    Poll::Ready(Ok(Box::new(move |client| Buffer {
                        view: BufferView {
                            client,
                            allocation,
                            offset: 0,
                            len,
                            marker: PhantomData,
                        },
                    }) as Pending<Buffer<T>>))
                }
            },
        )
    }

    /// Copies `values` into a new buffer.
    ///
    /// The values are copied into staging storage before this returns, so `values` can be
    /// reused immediately.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte size overflows, staging storage cannot hold the copy,
    /// or the command is not admitted. Allocation failures are reported through the
    /// submission.
    pub fn upload<T: PtxScalar>(&self, values: &[T]) -> Result<Submission<Buffer<T>>, SubmitError> {
        let len = values.len();
        let size = byte_size::<T>(len)?;
        submit_poll(
            &self.client,
            self.id,
            CommandKind::Transfer,
            false,
            size,
            move |staging| {
                let mut bytes = Some(encode(values));
                let mut staging = Some(staging);
                let mut allocation = None;
                let mut offset = 0usize;
                move |worker: &mut Worker, terminal| {
                    if let Some(error) = terminal {
                        return Poll::Ready(Err(error));
                    }
                    if allocation.is_none() {
                        allocation = Some(worker.reserve_allocation(size)?);
                    }
                    let Some(current) = &allocation else {
                        return Poll::Ready(Err(ExecutionError::Memory(
                            "missing allocation reservation".into(),
                        )));
                    };
                    let end = offset.saturating_add(CHUNK_BYTES).min(size);
                    let ready = worker.memory.initialize_chunk(current.address, false)?;
                    let snapshot = bytes.as_ref().ok_or_else(|| {
                        ExecutionError::Invalid("upload already completed".into())
                    })?;
                    worker
                        .memory
                        .initialize(current.address, offset, &snapshot[offset..end])?;
                    offset = end;
                    #[cfg(test)]
                    worker.transfer_checkpoint(offset, size);
                    if !ready {
                        return Poll::Pending;
                    }
                    drop(bytes.take());
                    drop(staging.take());
                    let allocation = allocation.take().ok_or_else(|| {
                        ExecutionError::Memory("missing allocation reservation".into())
                    })?;
                    Poll::Ready(Ok(Box::new(move |client| Buffer {
                        view: BufferView {
                            client,
                            allocation,
                            offset: 0,
                            len,
                            marker: PhantomData,
                        },
                    }) as Pending<Buffer<T>>))
                }
            },
        )
    }

    /// Overwrites all of `view` with `values`. Use a subview to write part of a buffer.
    ///
    /// The values are copied into staging storage before this returns.
    ///
    /// # Errors
    ///
    /// Returns an error if `view` belongs to another emulator or is read-only, the lengths
    /// differ, the byte size overflows, staging storage cannot hold the copy, or the
    /// command is not admitted.
    pub fn write<T: PtxScalar>(
        &self,
        view: &BufferView<T>,
        values: &[T],
    ) -> Result<Submission<()>, SubmitError> {
        foreign(&self.client, &view.client)?;
        if values.len() != view.len || !view.allocation.writable {
            return Err(SubmitError::InvalidArgument(
                "write length or read-only storage".into(),
            ));
        }
        let size = byte_size::<T>(values.len())?;
        let allocation = view.allocation.clone();
        let address = view.address();
        submit_poll(
            &self.client,
            self.id,
            CommandKind::Transfer,
            false,
            size,
            move |staging| {
                let mut bytes = Some(encode(values));
                let mut staging = Some(staging);
                let mut offset = 0usize;
                move |worker: &mut Worker, terminal| {
                    let _lease = &allocation;
                    if let Some(error) = terminal {
                        return Poll::Ready(Err(error));
                    }
                    let end = offset.saturating_add(CHUNK_BYTES).min(size);
                    let snapshot = bytes
                        .as_ref()
                        .ok_or_else(|| ExecutionError::Invalid("write already completed".into()))?;
                    worker.memory.write(
                        address + offset as u64,
                        &snapshot[offset..end],
                        1,
                        None,
                        None,
                    )?;
                    offset = end;
                    #[cfg(test)]
                    worker.transfer_checkpoint(offset, size);
                    if offset < size {
                        return Poll::Pending;
                    }
                    drop(bytes.take());
                    drop(staging.take());
                    Poll::Ready(Ok(Box::new(|_| ()) as Pending<()>))
                }
            },
        )
    }

    /// Copies `view` into a vector.
    ///
    /// Downloads are still accepted after a kernel fault, so completed writes can be read
    /// until shutdown starts.
    ///
    /// # Errors
    ///
    /// Returns an error if `view` belongs to another emulator, the byte size overflows,
    /// staging storage cannot hold the result, or the command is not admitted.
    pub fn download<T: PtxScalar>(
        &self,
        view: &BufferView<T>,
    ) -> Result<Submission<Vec<T>>, SubmitError> {
        foreign(&self.client, &view.client)?;
        let len = view.len;
        let size = byte_size::<T>(len)?;
        let allocation = view.allocation.clone();
        let address = view.address();
        submit_poll(
            &self.client,
            self.id,
            CommandKind::Transfer,
            true,
            size,
            move |staging| {
                let mut staging = Some(staging);
                let mut output = Vec::new();
                let mut started = false;
                let mut offset = 0usize;
                move |worker: &mut Worker, terminal| {
                    let _lease = &allocation;
                    if let Some(error) = terminal {
                        return Poll::Ready(Err(error));
                    }
                    if !started {
                        output
                            .try_reserve_exact(len)
                            .map_err(|_| ExecutionError::MemoryLimit)?;
                        started = true;
                    }
                    let end = offset.saturating_add(CHUNK_BYTES).min(size);
                    let chunk =
                        worker
                            .memory
                            .read(address + offset as u64, end - offset, 1, None, None)?;
                    output.extend(chunk.chunks_exact(T::SIZE).map(T::decode));
                    offset = end;
                    #[cfg(test)]
                    worker.transfer_checkpoint(offset, size);
                    if offset < size {
                        return Poll::Pending;
                    }
                    let output = mem::take(&mut output);
                    let staging = staging.take();
                    Poll::Ready(Ok(Box::new(move |_| {
                        drop(staging);
                        output
                    }) as Pending<Vec<T>>))
                }
            },
        )
    }
}

/// A marker that completes when the commands before it on its stream complete.
///
/// Its status stays available after its submission is consumed or dropped.
#[derive(Clone)]
pub struct Event {
    pub(crate) client: Arc<Client>,
    pub(crate) completion: Arc<Completion<Pending<()>>>,
}

impl Debug for Event {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Event").finish_non_exhaustive()
    }
}

impl Event {
    /// Returns `None` while the event is pending, then its result.
    #[must_use]
    pub fn status(&self) -> Option<Result<(), ExecutionError>> {
        self.completion.status()
    }
}
