//! Breakpoint control. These commands bypass the user command queue, and holding a
//! breakpoint identifier does not keep the emulator alive.

use std::{
    collections::BTreeMap,
    fmt::{self, Debug, Formatter},
    mem,
    sync::{Arc, Weak, atomic::Ordering},
};

use crate::emulator::{
    FaultLocation, LaneSnapshot,
    budget::{ChargedQueue, Reservation, allocation, node},
    engine::Engine,
    execution_error::{ExecutionError, SubmitError},
    runtime::{
        CommandControl, Completion, Emulator, Pending, Phase, Producer, Shared, Submission, lock,
    },
};

/// Identifies one suspension of a launch at `brkpt`.
///
/// Each suspension gets a new identifier, even at the same instruction.
#[derive(Clone, Debug)]
pub struct BreakpointId {
    sequence: u64,
    device: Weak<Shared>,
}

impl PartialEq for BreakpointId {
    fn eq(&self, other: &Self) -> bool {
        self.sequence == other.sequence && self.device.ptr_eq(&other.device)
    }
}

impl Eq for BreakpointId {}

/// A launch suspended at `brkpt`.
#[derive(Clone, Debug)]
pub struct Breakpoint {
    /// Identifier to pass to [`Emulator::resume`] and [`Emulator::inspect_lane`].
    pub id: BreakpointId,
    /// Lane and instruction that executed the `brkpt`.
    pub location: FaultLocation,
}

#[derive(Debug)]
struct Paused {
    control: Arc<CommandControl>,
    inspections: ChargedQueue<Inspection>,
    _storage: Reservation,
}

#[derive(Debug)]
struct Inspection {
    cta: [u32; 3],
    lane: [u32; 3],
    producer: Producer<LaneSnapshot>,
    storage: Reservation,
}

struct Notification {
    sequence: u64,
    result: Pending<Breakpoint>,
    _storage: Reservation,
}

impl Debug for Notification {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Notification").finish_non_exhaustive()
    }
}

#[derive(Debug)]
struct Observer {
    producer: Producer<Breakpoint>,
    storage: Reservation,
}

#[derive(Debug, Default)]
pub struct State {
    next: u64,
    pauses: BTreeMap<u64, Paused>,
    notifications: ChargedQueue<Notification>,
    observers: ChargedQueue<Observer>,
    pub(crate) orphaned: bool,
    pub(crate) faulted: bool,
}

impl State {
    pub(crate) fn has_paused_launches(&self) -> bool {
        self.pauses
            .values()
            .any(|pause| pause.control.suspended.load(Ordering::Acquire))
    }
}

pub struct Registration {
    sequence: u64,
    shared: Weak<Shared>,
}

impl Drop for Registration {
    fn drop(&mut self) {
        let error = self.shared.upgrade().map(|shared| {
            let phase = lock(&shared.admission).phase;
            match phase {
                Phase::Cancelling => ExecutionError::Cancelled,
                Phase::Faulted => ExecutionError::Faulted,
                Phase::Closed => ExecutionError::WorkerFailed,
                _ => ExecutionError::Invalid("breakpoint is no longer active".into()),
            }
        });
        if let Some(error) = error {
            self.finish(&error);
        }
    }
}

impl Registration {
    pub(crate) fn finish(&self, error: &ExecutionError) {
        if let Some(shared) = self.shared.upgrade() {
            let pause = lock(&shared.admission).debug.pauses.remove(&self.sequence);
            if let Some(mut pause) = pause {
                pause
                    .control
                    .inspection_pending
                    .store(false, Ordering::Release);
                for mut inspection in mem::take(&mut pause.inspections) {
                    drop(inspection.storage);
                    inspection.producer.finished = true;
                    inspection.producer.completion.publish(Err(error.clone()));
                }
            }
            shared.changed.notify_all();
        }
    }

    pub(crate) fn inspect(&self, engine: &Engine) {
        let Some(shared) = self.shared.upgrade() else {
            return;
        };
        let (requests, active) = {
            let mut state = lock(&shared.admission);
            let Some(pause) = state.debug.pauses.get_mut(&self.sequence) else {
                return;
            };
            pause
                .control
                .inspection_pending
                .store(false, Ordering::Release);
            let requests = mem::take(&mut pause.inspections);
            let active = pause.control.suspended.load(Ordering::Acquire);
            drop(state);
            (requests, active)
        };
        for mut request in requests {
            drop(request.storage);
            if request.producer.completion.is_abandoned() {
                request.producer.finished = true;
                continue;
            }
            let result = (|| {
                if !active {
                    return Err(ExecutionError::Invalid(
                        "breakpoint has been resumed".into(),
                    ));
                }
                let storage = shared.execution_budget.reserve(engine.snapshot_storage(
                    request.cta,
                    request.lane,
                    &shared.cancel,
                )?)?;
                let snapshot = engine.snapshot(request.cta, request.lane, &shared.cancel)?;
                Ok(Box::new(move |_| {
                    drop(storage);
                    snapshot
                }) as Pending<LaneSnapshot>)
            })();
            request.producer.finished = true;
            request.producer.completion.publish(result);
        }
    }
}

fn allowed(phase: Phase, faulted: bool) -> Result<(), SubmitError> {
    match phase {
        Phase::Running | Phase::Draining if faulted => Err(SubmitError::Faulted),
        Phase::Running | Phase::Draining => Ok(()),
        Phase::Faulted => Err(SubmitError::Faulted),
        Phase::Cancelling | Phase::Closed => Err(SubmitError::Closed),
    }
}

impl Emulator {
    /// Copies the registers and call stack of one lane in a suspended launch.
    ///
    /// `cta` selects a CTA in the grid and `lane` a thread in that CTA; any lane can be
    /// inspected, not only the one that hit the breakpoint. Inspection does not resume the
    /// launch, and it works while the command queue is full or the emulator is draining.
    /// The snapshot is owned and stays valid after the launch resumes.
    ///
    /// # Errors
    ///
    /// Returns an error if `id` belongs to another emulator or is no longer suspended, the
    /// emulator is closed or faulted, or the request cannot be stored. Invalid coordinates,
    /// or a resume that wins the race, are reported through the submission.
    pub fn inspect_lane(
        &self,
        id: &BreakpointId,
        cta: [u32; 3],
        lane: [u32; 3],
    ) -> Result<Submission<LaneSnapshot>, SubmitError> {
        let shared = &self.client.shared;
        if !id.device.ptr_eq(&Arc::downgrade(shared)) {
            return Err(SubmitError::ForeignResource);
        }
        let mut state = lock(&shared.admission);
        allowed(state.phase, state.debug.faulted)?;
        let pause = state
            .debug
            .pauses
            .get_mut(&id.sequence)
            .filter(|pause| pause.control.suspended.load(Ordering::Acquire))
            .ok_or_else(|| SubmitError::InvalidArgument("breakpoint is no longer active".into()))?;
        let mut storage = shared
            .execution_budget
            .reserve(allocation::<Inspection>() + allocation::<Completion<Pending<LaneSnapshot>>>())
            .map_err(|_| SubmitError::ExecutionStateLimit)?;
        let completion_storage = storage.split(allocation::<Completion<Pending<LaneSnapshot>>>());
        let completion = Arc::new(Completion::create(Some(completion_storage), false));
        pause.inspections.push_back(Inspection {
            cta,
            lane,
            producer: Producer {
                completion: Arc::clone(&completion),
                finished: false,
            },
            storage,
        });
        pause
            .control
            .inspection_pending
            .store(true, Ordering::Release);
        drop(state);
        shared.changed.notify_all();
        Ok(Submission {
            completion,
            client: Arc::clone(&self.client),
            launch_id: None,
        })
    }

    /// Returns a submission that completes when a launch next stops at `brkpt`.
    ///
    /// It does not use a stream or a command queue slot. Dropping the submission leaves the
    /// launch suspended; use [`paused_breakpoints`](Self::paused_breakpoints) to find it.
    ///
    /// # Errors
    ///
    /// Returns an error if the emulator is closed or faulted, or the request cannot be
    /// stored.
    pub fn next_breakpoint(&self) -> Result<Submission<Breakpoint>, SubmitError> {
        let shared = &self.client.shared;
        let mut state = lock(&shared.admission);
        allowed(state.phase, state.debug.faulted)?;
        let mut storage = shared
            .execution_budget
            .reserve(allocation::<Observer>() + allocation::<Completion<Pending<Breakpoint>>>())
            .map_err(|_| SubmitError::ExecutionStateLimit)?;
        let completion_storage = storage.split(allocation::<Completion<Pending<Breakpoint>>>());
        let completion = Arc::new(Completion::create(Some(completion_storage), false));
        let notification = state.debug.notifications.pop_front();
        if notification.is_none() {
            state.debug.observers.push_back(Observer {
                producer: Producer {
                    completion: Arc::clone(&completion),
                    finished: false,
                },
                storage,
            });
            drop(state);
        } else {
            drop(state);
            drop(storage);
            if let Some(notification) = notification {
                completion.publish(Ok(notification.result));
            }
        }
        Ok(Submission {
            completion,
            client: Arc::clone(&self.client),
            launch_id: None,
        })
    }

    /// Returns the identifiers of all suspended launches, including those nobody observed.
    #[must_use]
    pub fn paused_breakpoints(&self) -> Vec<BreakpointId> {
        let shared = &self.client.shared;
        lock(&shared.admission)
            .debug
            .pauses
            .iter()
            .filter(|(_, pause)| pause.control.suspended.load(Ordering::Acquire))
            .map(|(sequence, _)| BreakpointId {
                sequence: *sequence,
                device: Arc::downgrade(shared),
            })
            .collect()
    }

    /// Resumes the launch suspended at `id`.
    ///
    /// Resuming works while the emulator is draining. An identifier can be resumed once.
    ///
    /// # Errors
    ///
    /// Returns an error if `id` belongs to another emulator or is no longer suspended, the
    /// emulator is closed or faulted, or the request cannot be stored.
    pub fn resume(&self, id: &BreakpointId) -> Result<Submission<()>, SubmitError> {
        let shared = &self.client.shared;
        if !id.device.ptr_eq(&Arc::downgrade(shared)) {
            return Err(SubmitError::ForeignResource);
        }
        let state = lock(&shared.admission);
        allowed(state.phase, state.debug.faulted)?;
        let pause =
            state.debug.pauses.get(&id.sequence).ok_or_else(|| {
                SubmitError::InvalidArgument("breakpoint is no longer active".into())
            })?;
        let storage = shared
            .execution_budget
            .reserve(node::<Completion<Pending<()>>>())
            .map_err(|_| SubmitError::ExecutionStateLimit)?;
        if !pause.control.suspended.swap(false, Ordering::AcqRel) {
            return Err(SubmitError::InvalidArgument(
                "breakpoint has already been resumed".into(),
            ));
        }
        drop(state);
        let completion = Arc::new(Completion::create(Some(storage), false));
        completion.publish(Ok(Box::new(|_| ()) as Pending<()>));
        shared.changed.notify_all();
        Ok(Submission {
            completion,
            client: Arc::clone(&self.client),
            launch_id: None,
        })
    }
}

pub fn register(
    shared: &Arc<Shared>,
    engine: &Engine,
    control: &Arc<CommandControl>,
) -> Result<Registration, ExecutionError> {
    let mut state = lock(&shared.admission);
    if state.debug.orphaned {
        state.phase = Phase::Cancelling;
        shared.cancel.store(true, Ordering::Release);
        return Err(ExecutionError::Cancelled);
    }
    let record = shared.execution_budget.reserve(node::<Paused>())?;
    let notification_storage = shared
        .execution_budget
        .reserve(allocation::<Notification>())?;
    let result_storage = shared
        .execution_budget
        .reserve(engine.breakpoint_storage()? + node::<Breakpoint>())?;
    let sequence = state.debug.next;
    state.debug.next = sequence
        .checked_add(1)
        .ok_or(ExecutionError::ExecutionStateLimit)?;
    let breakpoint = Breakpoint {
        id: BreakpointId {
            sequence,
            device: Arc::downgrade(shared),
        },
        location: engine
            .breakpoint_location()
            .ok_or_else(|| ExecutionError::Invalid("missing breakpoint location".into()))?,
    };
    let result = Box::new(move |_| {
        drop(result_storage);
        breakpoint
    }) as Pending<Breakpoint>;
    control.suspended.store(true, Ordering::Release);
    state.debug.pauses.insert(
        sequence,
        Paused {
            control: Arc::clone(control),
            inspections: ChargedQueue::new(),
            _storage: record,
        },
    );
    let mut abandoned = Vec::new();
    let observer = loop {
        match state.debug.observers.pop_front() {
            Some(observer) if observer.producer.completion.is_abandoned() => {
                abandoned.push(observer);
            }
            observer => break observer,
        }
    };
    if let Some(mut observer) = observer {
        drop(state);
        drop(abandoned);
        drop(notification_storage);
        drop(observer.storage);
        observer.producer.finished = true;
        observer.producer.completion.publish(Ok(result));
    } else {
        state.debug.notifications.push_back(Notification {
            sequence,
            result,
            _storage: notification_storage,
        });
        drop(state);
        drop(abandoned);
    }
    Ok(Registration {
        sequence,
        shared: Arc::downgrade(shared),
    })
}

/// Debug queue entries whose observers are gone. They are removed under the admission
/// lock and dropped after it is released, since dropping them releases charged storage.
#[derive(Debug, Default)]
pub struct Reclaimed {
    observers: ChargedQueue<Observer>,
    notifications: ChargedQueue<Notification>,
}

impl Reclaimed {
    pub fn take(debug: &mut State) -> Self {
        let observers = debug
            .observers
            .extract(|observer| observer.producer.completion.is_abandoned());
        let pauses = &debug.pauses;
        let notifications = debug
            .notifications
            .extract(|notification| !pauses.contains_key(&notification.sequence));
        Self {
            observers,
            notifications,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.observers.is_empty() && self.notifications.is_empty()
    }
}

pub fn finish_observers(shared: &Shared, error: &ExecutionError) {
    let (observers, inspections) = {
        let mut state = lock(&shared.admission);
        let mut inspections = ChargedQueue::new();
        for pause in state.debug.pauses.values_mut() {
            inspections.append(&mut pause.inspections);
            pause
                .control
                .inspection_pending
                .store(false, Ordering::Release);
        }
        (mem::take(&mut state.debug.observers), inspections)
    };
    for mut inspection in inspections {
        drop(inspection.storage);
        inspection.producer.finished = true;
        inspection.producer.completion.publish(Err(error.clone()));
    }
    for mut observer in observers {
        drop(observer.storage);
        observer.producer.finished = true;
        observer.producer.completion.publish(Err(error.clone()));
    }
}

pub fn close(shared: &Shared, error: &ExecutionError) {
    finish_observers(shared, error);
    let state = mem::take(&mut lock(&shared.admission).debug);
    drop(state);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emulator::{EmulatorOptions, LaunchConfig, ParsedModule, ShutdownMode};
    use std::{
        thread,
        time::{Duration, Instant},
    };

    #[test]
    fn test_debug_observer_bursts_release_all_queue_storage() {
        let device = Emulator::create(EmulatorOptions::default()).unwrap();
        let budget = Arc::clone(&device.client.shared.execution_budget);
        let baseline = budget.used();
        for _ in 0..3 {
            let observers: Vec<_> = (0..64).map(|_| device.next_breakpoint().unwrap()).collect();
            assert!(budget.used() > baseline);
            drop(observers);
            device.synchronize().unwrap().wait().unwrap();
            assert!(
                lock(&device.client.shared.admission)
                    .debug
                    .observers
                    .is_empty()
            );
            assert_eq!(budget.used(), baseline);
        }
        device.shutdown(ShutdownMode::Drain).wait().unwrap();
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn test_debug_notification_bursts_release_nodes_and_keep_fifo_order() {
        let device = Emulator::create(EmulatorOptions::default()).unwrap();
        let budget = Arc::clone(&device.client.shared.execution_budget);
        let module = device
            .load(
                ParsedModule::parse(
                    ".version 9.4 .target sm_80 .address_size 64 .entry k() {brkpt; ret;}",
                )
                .unwrap(),
            )
            .unwrap()
            .wait()
            .unwrap();
        let kernel = module.kernel("k").unwrap();
        let streams: Vec<_> = (0..16).map(|_| device.create_stream().unwrap()).collect();
        let launches: Vec<_> = streams
            .iter()
            .map(|stream| {
                kernel
                    .launch_on(stream, LaunchConfig::create([1; 3], [1; 3], 0).unwrap(), [])
                    .unwrap()
            })
            .collect();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if lock(&device.client.shared.admission)
                .debug
                .notifications
                .len()
                == 16
            {
                break;
            }
            assert!(Instant::now() < deadline, "breakpoints did not arrive");
            thread::yield_now();
        }
        let queued = budget.used();
        let mut previous = None;
        let breakpoints: Vec<_> = (0..16)
            .map(|_| {
                let breakpoint = device.next_breakpoint().unwrap().wait().unwrap();
                if let Some(sequence) = previous {
                    assert!(sequence < breakpoint.id.sequence);
                }
                previous = Some(breakpoint.id.sequence);
                breakpoint
            })
            .collect();
        assert!(
            lock(&device.client.shared.admission)
                .debug
                .notifications
                .is_empty()
        );
        assert!(budget.used() < queued);
        for breakpoint in &breakpoints {
            device.resume(&breakpoint.id).unwrap().wait().unwrap();
        }
        for launch in launches {
            launch.wait().unwrap();
        }
        device.shutdown(ShutdownMode::Drain).wait().unwrap();
        assert_eq!(budget.used(), 0);
    }
    #[test]
    fn test_resumed_unobserved_notifications_release_queue_storage() {
        let device = Emulator::create(EmulatorOptions::default()).unwrap();
        let budget = Arc::clone(&device.client.shared.execution_budget);
        let module = device
            .load(
                ParsedModule::parse(
                    ".version 9.4 .target sm_80 .address_size 64 .entry k() {brkpt; ret;}",
                )
                .unwrap(),
            )
            .unwrap()
            .wait()
            .unwrap();
        let stream = device.create_stream().unwrap();
        let launch = module
            .kernel("k")
            .unwrap()
            .launch_on(
                &stream,
                LaunchConfig::create([1; 3], [1; 3], 0).unwrap(),
                [],
            )
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        let id = loop {
            if let Some(id) = device.paused_breakpoints().into_iter().next() {
                break id;
            }
            assert!(Instant::now() < deadline, "breakpoint did not arrive");
            thread::yield_now();
        };
        assert_eq!(
            lock(&device.client.shared.admission)
                .debug
                .notifications
                .len(),
            1
        );
        device.resume(&id).unwrap().wait().unwrap();
        launch.wait().unwrap();
        device.synchronize().unwrap().wait().unwrap();
        assert!(
            lock(&device.client.shared.admission)
                .debug
                .notifications
                .is_empty()
        );
        device.shutdown(ShutdownMode::Drain).wait().unwrap();
        assert_eq!(budget.used(), 0);
    }
}
