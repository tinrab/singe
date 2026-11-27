use std::{
    collections::BTreeMap,
    fmt::{self, Debug, Formatter},
    io,
    io::ErrorKind,
    marker::PhantomData,
    mem,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicUsize},
    },
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
        debug,
        host::{HostMemory, submit_host_task},
        lock,
        module::{Kernel, Module},
        options::{EmulatorOptions, ExecutionReport, LaunchConfig, ShutdownMode, Topology},
        scalar::PtxScalar,
        submit::submit_poll,
        worker::{LoadPhase, ModuleLoad, Worker, run_worker},
    },
};

/// An emulated machine of one or more devices, laid out by [`EmulatorOptions::topology`] and served by one worker thread.
///
/// The worker runs the commands of every device in turn, one warp instruction at a time, so kernels on different devices make progress together and a run is deterministic.
/// A fault stops only the device that faulted; see [`Device::fault`].
///
/// Clones share the machine.
/// Dropping the last clone and every handle created from it requests a draining shutdown without waiting for it.
#[derive(Clone)]
pub struct Emulator {
    pub(crate) client: Arc<Client>,
}

impl Debug for Emulator {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Emulator").finish_non_exhaustive()
    }
}

/// One device of an [`Emulator`], with its own memory, modules, and streams.
///
/// Addresses are unique across the machine, so a pointer names its device as with CUDA's unified addressing.
/// A kernel may access another device's memory only after [`enable_peer_access`](Self::enable_peer_access).
#[derive(Clone)]
pub struct Device {
    pub(crate) client: Arc<Client>,
    pub(crate) index: usize,
}

impl Debug for Device {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Device")
            .field("index", &self.index)
            .finish_non_exhaustive()
    }
}

impl Emulator {
    /// Creates an emulator and starts its worker thread, with the limits in `options`.
    ///
    /// # Errors
    ///
    /// Returns an error if `options.active_grids` is zero or the worker thread cannot be spawned.
    pub fn create(options: EmulatorOptions) -> Result<Self, io::Error> {
        if options.active_grids == 0 {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "active grid limit must be positive",
            ));
        }

        let devices = options.topology.device_count();
        if devices == 0 {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "topology must have a device",
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
                next_stream: devices as u64,
                next_launch: 0,
                staged: 0,
                reclaim: Vec::new(),
                shutdown_result: None,
                shutdown_wakers: BTreeMap::new(),
                faults: Vec::new(),
            }),
            changed: Condvar::new(),
            cancel: AtomicBool::new(false),
            options,
            agents: Mutex::new(Vec::new()),
            agent_count: AtomicUsize::new(0),
            next_agent: AtomicUsize::new(0),
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
    /// Calling it again returns the same result.
    /// A later [`ShutdownMode::Cancel`] can interrupt an earlier drain.
    pub fn shutdown(&self, mode: ShutdownMode) -> Shutdown {
        self.client.shared.shutdown(mode);
        Shutdown::new(Arc::clone(&self.client.shared))
    }

    /// Returns the layout of the machine's devices.
    #[must_use]
    pub fn topology(&self) -> Topology {
        self.client.shared.options.topology
    }

    /// Returns the failure that faulted the first device to fault, if any, so its cause stays known after the command that hit it was dropped.
    #[must_use]
    pub fn fault(&self) -> Option<ExecutionError> {
        lock(&self.client.shared.admission)
            .faults
            .first()
            .map(|(_, error)| error.clone())
    }

    /// Returns the number of devices.
    #[must_use]
    pub fn device_count(&self) -> usize {
        self.topology().device_count()
    }

    /// Returns the device numbered `index`.
    ///
    /// # Errors
    ///
    /// Returns [`SubmitError::UnknownDevice`] if the machine has no device `index`.
    pub fn device(&self, index: usize) -> Result<Device, SubmitError> {
        if index < self.device_count() {
            Ok(Device {
                client: Arc::clone(&self.client),
                index,
            })
        } else {
            Err(SubmitError::UnknownDevice(index))
        }
    }

    /// Returns every device, in index order.
    #[must_use]
    pub fn devices(&self) -> Vec<Device> {
        (0..self.device_count())
            .map(|index| Device {
                client: Arc::clone(&self.client),
                index,
            })
            .collect()
    }

    /// Runs `task` on the worker, ordered with no stream, until it returns a result.
    ///
    /// The worker polls `task` between other commands, with access to the memory of every device.
    /// It suits management work, such as allocating or mapping memory, that must not wait behind running kernels.
    /// It belongs to no device, so it runs after devices fault.
    ///
    /// # Errors
    ///
    /// Returns an error if the emulator is closed or the command is not admitted.
    pub fn host_task<T, F>(&self, task: F) -> Result<Submission<T>, SubmitError>
    where
        T: Send + 'static,
        F: FnMut(&mut HostMemory<'_>) -> Poll<Result<T, ExecutionError>> + Send + 'static,
    {
        let stream = {
            let mut state = lock(&self.client.shared.admission);
            admission(&state, None, false)?;
            state.allocate_stream_id()?
        };
        submit_host_task(&self.client, stream, 0, false, task)
    }
}

impl Device {
    /// Returns the device's index in its machine.
    #[must_use]
    pub fn index(&self) -> usize {
        self.index
    }

    /// Returns the node the device belongs to.
    #[must_use]
    pub fn node(&self) -> usize {
        self.client.shared.options.topology.node_of(self.index)
    }

    /// Returns the failure that faulted this device, if any.
    ///
    /// A fault stops only its device, as a sticky error poisons only its CUDA context: commands on other devices keep running.
    #[must_use]
    pub fn fault(&self) -> Option<ExecutionError> {
        lock(&self.client.shared.admission)
            .faults
            .iter()
            .find(|(device, _)| *device == self.index)
            .map(|(_, error)| error.clone())
    }

    /// Returns the machine the device belongs to.
    #[must_use]
    pub fn emulator(&self) -> Emulator {
        Emulator {
            client: Arc::clone(&self.client),
        }
    }

    /// Returns whether kernels on this device can be given access to `peer`'s memory: both belong to one node of one machine.
    #[must_use]
    pub fn can_access_peer(&self, peer: &Self) -> bool {
        Arc::ptr_eq(&self.client, &peer.client)
            && self
                .client
                .shared
                .options
                .topology
                .are_peers(self.index, peer.index)
    }

    /// Lets kernels on this device access memory allocated on `peer`.
    ///
    /// Commands submitted after the returned submission completes see the change.
    ///
    /// # Errors
    ///
    /// Returns an error if `peer` belongs to another emulator or the command is not admitted.
    /// The submission fails if the devices are not peers in one node.
    pub fn enable_peer_access(&self, peer: &Self) -> Result<Submission<()>, SubmitError> {
        foreign(&self.client, &peer.client)?;
        let (accessor, owner) = (self.index, peer.index);
        self.emulator()
            .host_task(move |memory| Poll::Ready(memory.enable_peer_access(accessor, owner)))
    }

    /// Stops kernels on this device from accessing memory allocated on `peer`.
    ///
    /// # Errors
    ///
    /// Returns an error if `peer` belongs to another emulator or the command is not admitted.
    pub fn disable_peer_access(&self, peer: &Self) -> Result<Submission<()>, SubmitError> {
        foreign(&self.client, &peer.client)?;
        let (accessor, owner) = (self.index, peer.index);
        self.emulator().host_task(move |memory| {
            memory.disable_peer_access(accessor, owner);
            Poll::Ready(Ok(()))
        })
    }

    /// Copies all of `source`, on any device, into `destination`, on any device, on this device's default stream.
    ///
    /// # Errors
    ///
    /// Returns an error if a view belongs to another emulator, the lengths differ, `destination` is read-only, or the command is not admitted.
    pub fn copy<T: PtxScalar>(
        &self,
        source: &BufferView<T>,
        destination: &BufferView<T>,
    ) -> Result<Submission<()>, SubmitError> {
        self.default_stream().copy(source, destination)
    }

    /// Loads `parsed`, allocating and initializing its global and constant variables.
    ///
    /// The module's storage layout is validated before the command is queued.
    /// Each load gets its own variables.
    ///
    /// # Errors
    ///
    /// Returns an error if the layout or an initializer is invalid, or if the command is not admitted.
    /// Allocation failures are reported through the submission.
    pub fn load(&self, parsed: ParsedModule) -> Result<Submission<Module>, SubmitError> {
        let layout = parsed.layout()?;
        submit_poll(
            &self.client,
            self.index as u64,
            self.index,
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

    /// Returns the device's default stream, which the device's own transfer methods use.
    ///
    /// Commands on different streams are not ordered with each other.
    #[must_use]
    pub fn default_stream(&self) -> Stream {
        Stream {
            client: Arc::clone(&self.client),
            id: self.index as u64,
            device: self.index,
        }
    }

    /// Creates a new stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the emulator is closed, the device faulted, or stream identifiers are exhausted.
    pub fn create_stream(&self) -> Result<Stream, SubmitError> {
        let mut state = lock(&self.client.shared.admission);
        admission(&state, Some(self.index), false)?;
        let id = state.allocate_stream_id()?;
        drop(state);
        Ok(Stream {
            client: Arc::clone(&self.client),
            id,
            device: self.index,
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
    /// Returns an error if the byte size overflows, staging storage cannot hold the copy, or the command is not admitted.
    pub fn upload<T: PtxScalar>(&self, values: &[T]) -> Result<Submission<Buffer<T>>, SubmitError> {
        self.default_stream().upload(values)
    }

    /// Overwrites `view` with `values` on the default stream.
    ///
    /// See [`Stream::write`].
    ///
    /// # Errors
    ///
    /// Returns an error if `view` belongs to another emulator or is read-only, the lengths differ, staging storage cannot hold the copy, or the command is not admitted.
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
    /// Returns an error if `view` belongs to another emulator, staging storage cannot hold the result, or the command is not admitted.
    pub fn download<T: PtxScalar>(
        &self,
        view: &BufferView<T>,
    ) -> Result<Submission<Vec<T>>, SubmitError> {
        self.default_stream().download(view)
    }

    /// Returns a submission that completes after every earlier command on the default stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the emulator is closed, the device faulted, or the command is not admitted.
    pub fn synchronize(&self) -> Result<Submission<()>, SubmitError> {
        self.default_stream().synchronize()
    }
}

/// An ordered queue of commands.
///
/// Commands on one stream run in submission order.
/// The worker interleaves commands from different streams; use [`Stream::record_event`] and [`Stream::wait_event`] to order them.
#[derive(Clone)]
pub struct Stream {
    pub(crate) client: Arc<Client>,
    pub(crate) id: u64,
    pub(crate) device: usize,
}

impl Debug for Stream {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Stream").finish_non_exhaustive()
    }
}

impl Stream {
    /// Returns the stream identifier, unique in its machine.
    /// A device's default stream has the device's index.
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Returns the index of the device whose commands this stream orders.
    #[must_use]
    pub fn device(&self) -> usize {
        self.device
    }

    /// Runs `task` on the worker in this stream's order, polling it until it returns a result.
    ///
    /// The task sees the memory of every device through [`HostMemory`], as a host thread sees pinned and device memory through copies.
    /// Return [`Poll::Pending`] to wait for something, such as a flag another device sets, without blocking the worker.
    /// An error the task returns faults the stream's device, as a kernel fault does, so the device's later commands fail.
    ///
    /// # Errors
    ///
    /// Returns an error if the emulator is closed, the stream's device faulted, or the command is not admitted.
    pub fn host_task<T, F>(&self, task: F) -> Result<Submission<T>, SubmitError>
    where
        T: Send + 'static,
        F: FnMut(&mut HostMemory<'_>) -> Poll<Result<T, ExecutionError>> + Send + 'static,
    {
        submit_host_task(&self.client, self.id, self.device, true, task)
    }

    /// Copies all of `source` into `destination`; either may live on any device of the machine.
    ///
    /// # Errors
    ///
    /// Returns an error if a view belongs to another emulator, the lengths differ, `destination` is read-only, or the command is not admitted.
    pub fn copy<T: PtxScalar>(
        &self,
        source: &BufferView<T>,
        destination: &BufferView<T>,
    ) -> Result<Submission<()>, SubmitError> {
        foreign(&self.client, &source.client)?;
        foreign(&self.client, &destination.client)?;
        if source.len != destination.len || !destination.allocation.writable {
            return Err(SubmitError::InvalidArgument(
                "copy length or read-only storage".into(),
            ));
        }

        let size = byte_size::<T>(source.len)?;
        let leases = (source.allocation.clone(), destination.allocation.clone());
        let (from, to) = (source.address(), destination.address());
        let mut offset = 0usize;
        self.host_task(move |memory| {
            let _leases = &leases;
            let end = offset.saturating_add(CHUNK_BYTES).min(size);
            memory.copy(from + offset as u64, to + offset as u64, end - offset)?;
            offset = end;
            if offset < size {
                Poll::Pending
            } else {
                Poll::Ready(Ok(()))
            }
        })
    }

    /// Returns a submission that completes after every earlier command on this stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the emulator is closed, the stream's device faulted, or the command is not admitted.
    pub fn synchronize(&self) -> Result<Submission<()>, SubmitError> {
        submit_poll(
            &self.client,
            self.id,
            self.device,
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
    /// The returned submission observes the event once.
    /// The [`Event`] itself can be passed to [`wait_event`](Self::wait_event) on any stream and queried with [`Event::status`] at any time.
    ///
    /// # Errors
    ///
    /// Returns an error if the emulator is closed, the stream's device faulted, or the command is not admitted.
    pub fn record_event(&self) -> Result<(Event, Submission<()>), SubmitError> {
        let submission = submit_poll(
            &self.client,
            self.id,
            self.device,
            CommandKind::Event,
            false,
            0,
            |_| {
                |_: &mut Worker, terminal| {
                    Poll::Ready(terminal.map_or_else(|| Ok(Box::new(|_| ()) as Pending<()>), Err))
                }
            },
        )?;
        let event = Event {
            client: Arc::clone(&self.client),
            completion: Arc::clone(&submission.completion),
        };
        Ok((event, submission))
    }

    /// Makes later commands on this stream wait for `event`.
    ///
    /// Neither the host nor the worker blocks while the stream waits.
    /// If the event's commands fail, the returned submission reports that failure.
    ///
    /// # Errors
    ///
    /// Returns an error if `event` belongs to another emulator or the command is not admitted.
    pub fn wait_event(&self, event: &Event) -> Result<Submission<()>, SubmitError> {
        foreign(&self.client, &event.client)?;
        let completion = Arc::clone(&event.completion);
        submit_poll(
            &self.client,
            self.id,
            self.device,
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
    /// Returns an error if `kernel` belongs to another emulator, `config` does not meet the kernel's block requirements, an argument does not match its parameter, or the command is not admitted.
    /// Execution faults are reported through the submission.
    pub fn launch(
        &self,
        kernel: &Kernel,
        config: LaunchConfig,
        args: impl IntoIterator<Item = KernelArg>,
    ) -> Result<Submission<ExecutionReport>, SubmitError> {
        kernel.launch_on(self, config, args)
    }

    /// Allocates `len` zeroed elements.
    /// Zero-length buffers are allowed.
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
            self.device,
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
    /// The values are copied into staging storage before this returns, so `values` can be reused immediately.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte size overflows, staging storage cannot hold the copy, or the command is not admitted.
    /// Allocation failures are reported through the submission.
    pub fn upload<T: PtxScalar>(&self, values: &[T]) -> Result<Submission<Buffer<T>>, SubmitError> {
        let len = values.len();
        let size = byte_size::<T>(len)?;
        submit_poll(
            &self.client,
            self.id,
            self.device,
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

    /// Overwrites all of `view` with `values`.
    /// Use a subview to write part of a buffer.
    ///
    /// The values are copied into staging storage before this returns.
    ///
    /// # Errors
    ///
    /// Returns an error if `view` belongs to another emulator or is read-only, the lengths differ, the byte size overflows, staging storage cannot hold the copy, or the command is not admitted.
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
            self.device,
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
    /// Downloads are still accepted after a kernel fault, so completed writes can be read until shutdown starts.
    ///
    /// # Errors
    ///
    /// Returns an error if `view` belongs to another emulator, the byte size overflows, staging storage cannot hold the result, or the command is not admitted.
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
            self.device,
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

    /// Blocks until the event completes, then returns its result.
    ///
    /// # Errors
    ///
    /// Returns the failure of the commands before the event, or of the emulator.
    pub fn wait(&self) -> Result<(), ExecutionError> {
        self.completion.wait_status()
    }
}
