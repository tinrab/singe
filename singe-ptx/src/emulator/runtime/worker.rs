use std::{
    collections::BTreeSet,
    mem,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, PoisonError, atomic::Ordering},
    task::Poll,
};

#[cfg(test)]
use crate::emulator::engine::LaunchTraceEntry;

use crate::{
    ast::StateSpace,
    emulator::{
        CompiledKernel,
        budget::{ChargedQueue, Reservation},
        engine::{Engine, GridBuilder},
        execution_error::ExecutionError,
        layout::{InitialBits, InitialValue, ModuleLayout},
        memory::{CHUNK_BYTES, Memory, Owner, generic_address, specific, validate_address_size},
        runtime::{
            Admission, Command, CommandKind, Lease, Phase, Shared, Staging,
            buffer::AllocationRef,
            completion::Pending,
            debug, lock,
            module::Loaded,
            options::{ExecutionReport, LaunchConfig},
            submit::RetainedFailure,
        },
    },
};

/// Resources whose owners are gone, released by the worker outside the admission lock.
pub struct Reclaimable {
    pub debug: debug::Reclaimed,
    pub addresses: Vec<u64>,
}

impl Reclaimable {
    fn is_empty(&self) -> bool {
        self.debug.is_empty() && self.addresses.is_empty()
    }
}

pub struct Worker {
    pub memory: Memory,
    pub shared: Arc<Shared>,
    pub failure: Option<RetainedFailure>,
}

impl Worker {
    /// Removes released resources under the admission lock without dropping them.
    pub fn take_reclaimable(state: &mut Admission) -> Reclaimable {
        Reclaimable {
            debug: debug::Reclaimed::take(&mut state.debug),
            addresses: mem::take(&mut state.reclaim),
        }
    }

    /// Drops reclaimed resources after the admission lock is released.
    pub fn release(&mut self, reclaimable: Reclaimable) {
        drop(reclaimable.debug);
        for address in reclaimable.addresses {
            self.memory.release(address);
        }
    }

    pub fn reclaim(&mut self) {
        let reclaimable = Self::take_reclaimable(&mut lock(&self.shared.admission));
        self.release(reclaimable);
    }

    pub fn check_cancel(&self) -> Result<(), ExecutionError> {
        if self.shared.cancel.load(Ordering::Acquire) {
            Err(ExecutionError::Cancelled)
        } else {
            Ok(())
        }
    }

    pub fn reserve_allocation(&mut self, size: usize) -> Result<AllocationRef, ExecutionError> {
        self.check_cancel()?;
        let address = self
            .memory
            .reserve(size, 256, StateSpace::Global, Owner::Device)?;
        Ok(AllocationRef {
            _lease: Arc::new(Lease {
                shared: Arc::downgrade(&self.shared),
                addresses: vec![address],
            }),
            address,
            writable: true,
            space: StateSpace::Global,
        })
    }

    #[cfg(test)]
    pub fn transfer_checkpoint(&self, completed: usize, total: usize) {
        if let Some(hook) = lock(&self.shared.hooks.transfer).as_mut() {
            hook(completed, total);
        }
    }

    pub fn fault(&mut self, error: &ExecutionError, storage: Arc<Reservation>) {
        if !matches!(
            error.root_cause(),
            ExecutionError::Cancelled | ExecutionError::Faulted
        ) {
            if self.failure.is_none() {
                self.failure = Some(RetainedFailure::new(error.clone(), Some(storage)));
            }
            let mut state = lock(&self.shared.admission);
            state.debug.faulted = true;
            if state.phase == Phase::Running {
                state.phase = Phase::Faulted;
            }
            drop(state);
            debug::finish_observers(&self.shared, &ExecutionError::Faulted);
        }
    }
}

#[derive(Clone, Copy)]
pub enum LoadPhase {
    Reserve,
    Zero { variable: usize, initialized: usize },
    Initializers { variable: usize, value: usize },
}

pub struct ModuleLoad {
    pub layout: Option<ModuleLayout>,
    pub addresses: Vec<u64>,
    pub phase: LoadPhase,
}

impl ModuleLoad {
    pub fn poll(
        &mut self,
        worker: &mut Worker,
        terminal: Option<ExecutionError>,
    ) -> Poll<Result<Arc<Loaded>, ExecutionError>> {
        let result = terminal.map_or_else(|| self.advance(worker), |error| Poll::Ready(Err(error)));
        if matches!(result, Poll::Ready(Err(_))) {
            for address in self.addresses.drain(..) {
                worker.memory.release(address);
            }
        }
        result
    }

    pub fn advance(&mut self, worker: &mut Worker) -> Poll<Result<Arc<Loaded>, ExecutionError>> {
        let layout = self
            .layout
            .as_ref()
            .ok_or_else(|| ExecutionError::Invalid("module already loaded".into()))?;
        match self.phase {
            LoadPhase::Reserve => {
                if self.addresses.capacity() < layout.variables.len() {
                    self.addresses
                        .try_reserve_exact(layout.variables.len())
                        .map_err(|_| ExecutionError::MemoryLimit)?;
                }
                if let Some(variable) = layout.variables.get(self.addresses.len()) {
                    let address = if variable.space == StateSpace::Shared {
                        0
                    } else {
                        let size = usize::try_from(variable.layout.byte_len())
                            .map_err(|_| ExecutionError::MemoryLimit)?;
                        worker.memory.reserve(
                            size,
                            variable.layout.alignment(),
                            variable.space,
                            Owner::Device,
                        )?
                    };
                    self.addresses.push(address);
                    if variable.space != StateSpace::Shared {
                        validate_address_size(
                            address,
                            usize::try_from(variable.layout.byte_len())
                                .map_err(|_| ExecutionError::ExecutionStateLimit)?,
                            layout.address_size,
                        )?;
                    }
                } else {
                    self.phase = LoadPhase::Zero {
                        variable: 0,
                        initialized: 0,
                    };
                }
            }
            LoadPhase::Zero {
                variable,
                initialized,
            } => {
                if let Some(storage) = layout.variables.get(variable) {
                    let size = usize::try_from(storage.layout.byte_len())
                        .map_err(|_| ExecutionError::ExecutionStateLimit)?;
                    let ready = storage.space == StateSpace::Shared
                        || worker
                            .memory
                            .initialize_chunk(self.addresses[variable], true)?;
                    let initialized = initialized.saturating_add(CHUNK_BYTES).min(size);
                    #[cfg(test)]
                    if storage.space != StateSpace::Shared {
                        worker.transfer_checkpoint(initialized, size);
                    }
                    self.phase = if ready {
                        LoadPhase::Zero {
                            variable: variable + 1,
                            initialized: 0,
                        }
                    } else {
                        LoadPhase::Zero {
                            variable,
                            initialized,
                        }
                    };
                } else {
                    self.phase = LoadPhase::Initializers {
                        variable: 0,
                        value: 0,
                    };
                }
            }
            LoadPhase::Initializers { variable, value } => {
                if let Some(storage) = layout.variables.get(variable) {
                    if let Some(initial) = storage.initial_values.get(value) {
                        let bits = self.initial_bits(initial)?;
                        worker.memory.initialize(
                            self.addresses[variable],
                            usize::try_from(initial.offset)
                                .map_err(|_| ExecutionError::ExecutionStateLimit)?,
                            &bits.to_le_bytes()[..initial.width as usize],
                        )?;
                        self.phase = LoadPhase::Initializers {
                            variable,
                            value: value + 1,
                        };
                    } else {
                        self.phase = LoadPhase::Initializers {
                            variable: variable + 1,
                            value: 0,
                        };
                    }
                } else {
                    return Poll::Ready(self.finish(worker));
                }
            }
        }
        Poll::Pending
    }

    /// Resolves an initializer value, relocating symbol addresses against this load.
    fn initial_bits(&self, initial: &InitialValue) -> Result<u128, ExecutionError> {
        Ok(match &initial.value {
            InitialBits::Bits(bits) => *bits,
            InitialBits::Address(relocation) => {
                let address = self.addresses[relocation.target.0];
                let address = if relocation.generic {
                    generic_address(address)
                } else {
                    specific(address)
                };
                let address = address
                    .checked_add_signed(relocation.offset)
                    .ok_or_else(|| ExecutionError::Memory("relocation address overflow".into()))?;
                let address = relocation
                    .mask
                    .map_or(address, |mask| (address & mask) >> mask.trailing_zeros());
                if initial.width < 8 && address >= 1u64 << (initial.width * 8) {
                    return Err(ExecutionError::Memory(
                        "relocation exceeds destination width".into(),
                    ));
                }
                u128::from(address)
            }
        })
    }

    /// Hands the reserved and initialized storage over to a loaded module.
    fn finish(&mut self, worker: &Worker) -> Result<Arc<Loaded>, ExecutionError> {
        let addresses = mem::take(&mut self.addresses);
        let lease = Arc::new(Lease {
            shared: Arc::downgrade(&worker.shared),
            addresses: addresses
                .iter()
                .copied()
                .filter(|address| *address != 0)
                .collect(),
        });
        let layout = self
            .layout
            .take()
            .ok_or_else(|| ExecutionError::Invalid("module already loaded".into()))?;
        Ok(Arc::new(Loaded {
            layout,
            addresses,
            lease,
        }))
    }
}

pub struct LaunchWork {
    pub code: CompiledKernel,
    pub loaded: Arc<Loaded>,
    pub config: LaunchConfig,
    pub args: Vec<Vec<u8>>,
    pub _leases: Vec<AllocationRef>,
    pub staging: Staging,
    pub engine: Option<Engine>,
    pub builder: Option<GridBuilder>,
    pub pause: Option<debug::Registration>,
    pub stream_id: u64,
    pub result_storage: Option<Arc<Reservation>>,
}

impl LaunchWork {
    pub fn poll(
        &mut self,
        worker: &mut Worker,
        terminal: Option<ExecutionError>,
    ) -> Poll<Result<Pending<ExecutionReport>, ExecutionError>> {
        if let Some(error) = terminal {
            return Poll::Ready(Err(self.stop(worker, error)));
        }
        if self.engine.is_none() {
            match self.build_grid(worker) {
                Poll::Ready(Ok(())) => {}
                Poll::Ready(Err(error)) => return Poll::Ready(Err(error)),
                Poll::Pending => return Poll::Pending,
            }
        }
        let Some(engine) = self.engine.as_mut() else {
            return Poll::Ready(Err(ExecutionError::Invalid("missing launch state".into())));
        };
        if let Some(pause) = &self.pause {
            pause.inspect(engine);
            if self.staging.control.suspended.load(Ordering::Acquire) {
                return Poll::Pending;
            }
        }
        if self.pause.take().is_some() {
            engine.resume();
        }

        #[cfg(test)]
        if let Some(hook) = lock(&worker.shared.hooks.launch).as_mut() {
            hook(engine);
        }
        worker.reclaim();
        #[cfg(test)]
        let trace_start = engine.trace.len();
        #[cfg(test)]
        let launch_id = engine.launch_id();
        let mut step = engine.step(&mut worker.memory, &worker.shared.cancel);
        if matches!(step, Ok(None)) && engine.is_paused() {
            match debug::register(&worker.shared, engine, &self.staging.control) {
                Ok(pause) => self.pause = Some(pause),
                Err(error) => {
                    let error = if let Some(location) = engine.breakpoint_location() {
                        ExecutionError::Lane {
                            location: Arc::new(location),
                            cause: Arc::new(error),
                        }
                    } else {
                        error
                    };
                    step = Err(engine.failure(error));
                }
            }
        }

        #[cfg(test)]
        {
            let mut trace = lock(&worker.shared.hooks.scheduler_trace);
            let remaining = 4096usize.saturating_sub(trace.len());
            trace.extend(
                engine.trace[trace_start..]
                    .iter()
                    .take(remaining)
                    .map(|&entry| LaunchTraceEntry {
                        launch: launch_id,
                        entry,
                    }),
            );
        }
        let result = match step {
            Ok(None) => return Poll::Pending,
            Ok(Some(report)) => Ok(Box::new(move |_| report) as Pending<ExecutionReport>),
            Err(error) => Err(error),
        };
        let result_storage = engine.retain_result(&mut worker.memory);
        if let Err(error) = &result {
            worker.fault(error, Arc::clone(&result_storage));
        }
        self.result_storage = Some(result_storage);
        engine.release(&mut worker.memory);
        #[cfg(test)]
        {
            *lock(&worker.shared.hooks.trace) = engine.trace.clone();
        }
        self.engine = None;
        Poll::Ready(result)
    }

    /// Ends the launch early with `error`, releasing its grid and retaining its diagnostics.
    fn stop(&mut self, worker: &mut Worker, error: ExecutionError) -> ExecutionError {
        if let Some(pause) = self.pause.take() {
            pause.finish(&error);
        }
        if let Some(builder) = self.builder.take() {
            builder.release(&mut worker.memory);
        }
        if let Some(mut engine) = self.engine.take() {
            self.result_storage = Some(engine.retain_result(&mut worker.memory));
            let failure = engine.failure(error);
            engine.release(&mut worker.memory);
            #[cfg(test)]
            {
                *lock(&worker.shared.hooks.trace) = engine.trace.clone();
            }
            failure
        } else {
            error
        }
    }

    /// Constructs the grid in bounded turns, then identifies the finished engine.
    fn build_grid(&mut self, worker: &mut Worker) -> Poll<Result<(), ExecutionError>> {
        if self.builder.is_none() {
            self.builder = Some(GridBuilder::create(
                &self.code,
                self.loaded.addresses.len(),
                self.config,
                &worker.shared.options,
                &mut worker.memory,
            )?);
        }
        let builder = self
            .builder
            .as_mut()
            .ok_or_else(|| ExecutionError::Invalid("missing grid builder".into()))?;
        let advanced = builder.advance(
            &self.loaded.addresses,
            &self.args,
            &worker.shared.cancel,
            &mut worker.memory,
        );
        #[cfg(test)]
        if let Some(hook) = lock(&worker.shared.hooks.initialization).as_mut() {
            hook(self.staging.launch_id.unwrap_or_default());
        }
        match advanced {
            Ok(false) => return Poll::Pending,
            Ok(true) => {}
            Err(error) => {
                builder.release(&mut worker.memory);
                self.builder = None;
                return Poll::Ready(Err(error));
            }
        }
        self.engine = self.builder.take().map(GridBuilder::finish);
        if let Some(engine) = &mut self.engine {
            engine.identify(self.staging.launch_id.unwrap_or_default(), self.stream_id);
        }
        Poll::Ready(Ok(()))
    }
}

/// Commands the worker is running, polled round-robin.
type ActiveCommands = ChargedQueue<Command>;

pub fn run_worker(shared: &Arc<Shared>) {
    let mut worker = Worker {
        memory: Memory::new(
            shared.options.device_storage_bytes,
            Arc::clone(&shared.execution_budget),
        ),
        shared: Arc::clone(shared),
        failure: None,
    };
    let mut active = ActiveCommands::new();
    // An emulator panic becomes `WorkerFailed` for every pending result instead of leaving
    // blocked observers waiting forever.
    let result = catch_unwind(AssertUnwindSafe(|| {
        loop {
            let mut state = lock(&shared.admission);
            // Reclaim and command selection share one critical section, so anything released
            // before a command was admitted is reclaimed before that command runs.
            let reclaimable = Worker::take_reclaimable(&mut state);
            if !reclaimable.is_empty() {
                drop(state);
                worker.release(reclaimable);
                continue;
            }
            if let Some(command) = select_command(shared, &mut state, &active, &worker) {
                if command.kind == CommandKind::Dependency {
                    state.pending_dependencies += 1;
                }
                drop(state);
                active.push_back(command);
            } else if active.is_empty() {
                if matches!(state.phase, Phase::Draining | Phase::Cancelling) {
                    break;
                }
                drop(
                    shared
                        .changed
                        .wait(state)
                        .unwrap_or_else(PoisonError::into_inner),
                );
                continue;
            } else if active.iter().all(|command| command.control.is_suspended())
                && worker.failure.is_none()
                && !shared.cancel.load(Ordering::Acquire)
            {
                drop(
                    shared
                        .changed
                        .wait(state)
                        .unwrap_or_else(PoisonError::into_inner),
                );
                continue;
            } else {
                drop(state);
            }
            run_next(&mut worker, &mut active);
        }
    }));
    let failure = if result.is_err() {
        Some(RetainedFailure::new(ExecutionError::WorkerFailed, None))
    } else {
        worker.failure.take()
    };
    close(shared, worker, active, failure);
}

/// Picks the first queued command that may start: one per stream, behind no running
/// command of its stream, and within the active-grid limit unless the emulator is stopping.
fn select_command(
    shared: &Shared,
    state: &mut Admission,
    active: &ActiveCommands,
    worker: &Worker,
) -> Option<Command> {
    let launches = active
        .iter()
        .filter(|command| command.kind == CommandKind::Launch)
        .count();
    if active.len()
        >= shared
            .options
            .queued_commands
            .saturating_add(shared.options.active_grids)
    {
        return None;
    }
    let mut streams = BTreeSet::new();
    let index = state.queue.iter().position(|command| {
        streams.insert(command.stream)
            && !active
                .iter()
                .any(|running| running.stream == command.stream)
            && (command.kind != CommandKind::Launch
                || launches < shared.options.active_grids
                || shared.cancel.load(Ordering::Acquire)
                || worker.failure.is_some())
    })?;
    state.queue.remove(index)
}

/// Polls the oldest active command once, requeueing it unless it finished.
fn run_next(worker: &mut Worker, active: &mut ActiveCommands) {
    let Some(mut command) = active.pop_front() else {
        return;
    };
    if command.control.is_suspended()
        && worker.failure.is_none()
        && !worker.shared.cancel.load(Ordering::Acquire)
    {
        active.push_back(command);
    } else if !(command.poll)(worker, None) {
        if command.kind == CommandKind::Dependency {
            command.control.suspended.store(true, Ordering::Release);
        }
        active.push_back(command);
    } else {
        for dependency in active.iter() {
            if dependency.kind == CommandKind::Dependency {
                dependency.control.suspended.store(false, Ordering::Release);
            }
        }
    }
}

/// Closes admission, drops every queued and active command, and publishes the shutdown result.
fn close(
    shared: &Arc<Shared>,
    worker: Worker,
    active: ActiveCommands,
    failure: Option<RetainedFailure>,
) {
    // Drop queued producers outside admission: their leases and staging tokens also use it.
    let queue = {
        let mut state = lock(&shared.admission);
        state.phase = Phase::Closed;
        mem::take(&mut state.queue)
    };
    drop(queue);
    drop(active);
    drop(worker);
    debug::close(
        shared,
        failure
            .as_ref()
            .map_or(&ExecutionError::Cancelled, |failure| &failure.error),
    );
    let wakers = {
        let mut state = lock(&shared.admission);
        state.reclaim.clear();
        state.shutdown_result = Some(failure.map_or(Ok(()), Err));
        mem::take(&mut state.shutdown_wakers)
    };
    shared.changed.notify_all();
    for (_, waker) in wakers {
        let _ = catch_unwind(AssertUnwindSafe(|| waker.wake()));
    }
}
