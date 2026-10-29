//! Deferred per-lane copies, group boundaries, and `sm_80` shared barriers.
//!
//! A copy completes only once a wait requests it, a barrier tracks it, or its issuing
//! lane exits, and each scheduler turn completes at most one copy or arrival. This
//! completes copies as late as PTX allows, which is deterministic and does not model
//! hardware latency.

use std::{
    collections::BTreeMap,
    sync::{Arc, atomic::AtomicBool},
};

use crate::{
    ast::{ScalarType, StateSpace},
    emulator::{
        budget::{ChargedQueue, node},
        engine::{Engine, Result, Step, Waiting, cancel},
        execution_error::{ExecutionError, FaultLocation},
        ir::{Address, AsyncCopy, AsyncCopyInput, AsyncOperation},
        memory::Memory,
        numeric,
    },
};

#[derive(Default)]
pub struct AsyncLane {
    pending: ChargedQueue<PendingEvent>,
    groups: ChargedQueue<Group>,
    pub(crate) issued: u64,
    completed: u64,
    pub(crate) requested: u64,
    committed: u64,
    last_copy: u64,
    last_commit: u64,
    scheduled: bool,
    next_ready: Option<usize>,
}

#[derive(Default)]
pub struct AsyncState {
    barriers: BTreeMap<u64, Barrier>,
    ready: Option<usize>,
    last_ready: Option<usize>,
    next_token: u64,
}

struct Group {
    id: u64,
    through: u64,
}

enum Event {
    Copy {
        destination: u64,
        bytes: [u8; 16],
        size: usize,
    },
    Arrive {
        address: u64,
    },
}

struct PendingEvent {
    event: Event,
    origin: FaultLocation,
    charge: usize,
}

struct Barrier {
    expected: u32,
    pending: u32,
    tracked: u32,
    current: u64,
    previous: Option<u64>,
    parity: bool,
    observed: bool,
}

// Each entry is a queue or map entry; no queue retains spare capacity.
const EVENT_CHARGE: usize = node::<PendingEvent>();
const GROUP_CHARGE: usize = node::<Group>();
const BARRIER_CHARGE: usize = node::<(u64, Barrier)>();
const MAX_ARRIVALS: u32 = (1 << 20) - 1;

fn invalid(message: &str) -> ExecutionError {
    ExecutionError::Invalid(message.into())
}

impl Engine {
    pub(crate) fn issue_copy(
        &mut self,
        memory: &mut Memory,
        lane: usize,
        copy: &AsyncCopy,
        cancellation: &AtomicBool,
    ) -> Result<()> {
        let AsyncCopy {
            destination,
            source,
            size,
            input,
        } = copy;
        let size = *size;
        let source_size = match input {
            None => size,
            Some(AsyncCopyInput::Bytes(count)) => {
                self.source(lane, count, ScalarType::U32)? as usize
            }
            Some(AsyncCopyInput::Ignore(predicate)) => {
                if self.predicate(lane, *predicate)? {
                    0
                } else {
                    size
                }
            }
        };
        if source_size > size {
            return Err(invalid("asynchronous copy source size exceeds copy size"));
        }
        let destination = self.address(lane, *destination, Some(StateSpace::Shared))?;
        memory.validate_write(
            destination,
            size,
            size,
            Some(StateSpace::Shared),
            Some(self.actor(lane)),
        )?;
        let state = &self.lanes[lane].asynchronous;
        // PTX leaves overlapping writes inside a group undefined. Diagnose
        // this case instead of accidentally defining an ordering for it.
        for (offset, event) in state.pending.iter().rev().enumerate() {
            cancel(cancellation)?;
            let sequence = state.issued - offset as u64;
            if sequence <= state.last_commit {
                break;
            }
            if let Event::Copy {
                destination: other,
                size: other_size,
                ..
            } = &event.event
                && destination < other + *other_size as u64
                && *other < destination + size as u64
            {
                return Err(invalid("overlapping asynchronous copies in one group"));
            }
        }
        let mut bytes = [0; 16];
        if source_size != 0 {
            let source = self.address(lane, *source, Some(StateSpace::Global))?;
            bytes[..source_size].copy_from_slice(memory.read(
                source,
                source_size,
                size,
                Some(StateSpace::Global),
                Some(self.actor(lane)),
            )?);
        }
        self.push_async(
            memory,
            lane,
            Event::Copy {
                destination,
                bytes,
                size,
            },
        )?;
        self.lanes[lane].asynchronous.last_copy = self.lanes[lane].asynchronous.issued;
        Ok(())
    }

    fn push_async(&mut self, memory: &mut Memory, lane: usize, event: Event) -> Result<()> {
        let mut charge = EVENT_CHARGE
            .checked_add(self.function(lane).name.len())
            .ok_or(ExecutionError::ExecutionStateLimit)?;
        for frame in &self.lanes[lane].frames {
            charge = charge
                .checked_add(size_of::<String>())
                .and_then(|charge| {
                    charge.checked_add(self.code.function(frame.function).name.len())
                })
                .ok_or(ExecutionError::ExecutionStateLimit)?;
        }
        let issued = self.lanes[lane]
            .asynchronous
            .issued
            .checked_add(1)
            .ok_or(ExecutionError::ExecutionStateLimit)?;
        memory.reserve_execution(self.grid, charge)?;
        let origin = self.fault_location(lane, self.current(lane).pc);
        let state = &mut self.lanes[lane].asynchronous;
        state.pending.push_back(PendingEvent {
            event,
            origin,
            charge,
        });
        state.issued = issued;
        Ok(())
    }

    pub(crate) fn commit_copies(&mut self, memory: &mut Memory, lane: usize) -> Result<()> {
        let state = &mut self.lanes[lane].asynchronous;
        let id = state
            .committed
            .checked_add(1)
            .ok_or(ExecutionError::ExecutionStateLimit)?;
        if state.last_copy > state.last_commit && state.last_copy > state.completed {
            memory.reserve_execution(self.grid, GROUP_CHARGE)?;
            state.groups.push_back(Group {
                id,
                through: state.last_copy,
            });
        }
        state.committed = id;
        state.last_commit = state.last_copy;
        Ok(())
    }

    pub(crate) fn wait_copies(
        &mut self,
        lane: usize,
        pending: u32,
        cancellation: &AtomicBool,
    ) -> Result<bool> {
        let state = &mut self.lanes[lane].asynchronous;
        let target = state.committed.saturating_sub(u64::from(pending));
        let mut through = state.completed;
        for group in state.groups.iter().rev() {
            cancel(cancellation)?;
            if group.id <= target {
                through = group.through;
                break;
            }
        }
        if through <= state.completed {
            return Ok(false);
        }
        self.request_async(lane, through);
        self.lanes[lane].waiting = Some(Waiting::Async { through });
        Ok(true)
    }

    pub(crate) fn request_async(&mut self, lane: usize, through: u64) {
        let state = &mut self.lanes[lane].asynchronous;
        state.requested = state.requested.max(through);
        if state.scheduled || state.requested <= state.completed {
            return;
        }
        state.scheduled = true;
        if let Some(last) = self.asynchronous.last_ready {
            self.lanes[last].asynchronous.next_ready = Some(lane);
        } else {
            self.asynchronous.ready = Some(lane);
        }
        self.asynchronous.last_ready = Some(lane);
    }

    pub(crate) fn advance_async(
        &mut self,
        memory: &mut Memory,
        cancellation: &AtomicBool,
    ) -> Result<bool> {
        cancel(cancellation)?;
        if let Some(lane) = self.asynchronous.ready {
            let state = &mut self.lanes[lane].asynchronous;
            self.asynchronous.ready = state.next_ready.take();
            if self.asynchronous.ready.is_none() {
                self.asynchronous.last_ready = None;
            }
            state.scheduled = false;
            let event = state
                .pending
                .pop_front()
                .ok_or_else(|| invalid("missing asynchronous operation"))?;
            // The destination was validated at issue and shared allocations
            // live until grid release, including after the issuing lane exits.
            let result = (|| match event.event {
                Event::Copy {
                    destination,
                    bytes,
                    size,
                } => memory.write(
                    destination,
                    &bytes[..size],
                    size,
                    Some(StateSpace::Shared),
                    Some(self.actor(lane)),
                ),
                Event::Arrive { address } => {
                    let barrier = self.barrier_mut(memory, address)?;
                    barrier.tracked -= 1;
                    barrier.pending -= 1;
                    self.complete_barrier(memory, address)
                }
            })();
            let result = result.map_err(|error| ExecutionError::Lane {
                location: Arc::new(event.origin),
                cause: Arc::new(error),
            });
            memory.release_execution(self.grid, event.charge);
            result?;
            let state = &mut self.lanes[lane].asynchronous;
            state.completed += 1;
            while state
                .groups
                .front()
                .is_some_and(|group| group.through <= state.completed)
            {
                state.groups.pop_front();
                memory.release_execution(self.grid, GROUP_CHARGE);
            }
            let completed = state.completed;
            let requested = state.requested;
            self.request_async(lane, requested);
            if matches!(self.lanes[lane].waiting, Some(Waiting::Async { through }) if through <= completed)
            {
                self.release_waiter(lane);
            }
            return Ok(true);
        }
        Ok(false)
    }

    pub(crate) fn barrier_address(
        &self,
        memory: &Memory,
        lane: usize,
        address: Address,
        space: Option<StateSpace>,
    ) -> Result<u64> {
        let address = self.address(lane, address, space)?;
        memory.validate_write(
            address,
            8,
            8,
            Some(StateSpace::Shared),
            Some(self.actor(lane)),
        )?;
        Ok(address)
    }

    fn barrier_mut(&mut self, memory: &Memory, address: u64) -> Result<&mut Barrier> {
        let barrier = self
            .asynchronous
            .barriers
            .get_mut(&address)
            .ok_or_else(|| invalid("mbarrier is not initialized"))?;
        if memory.read(address, 8, 8, Some(StateSpace::Shared), None)?
            != barrier.current.to_le_bytes()
        {
            return Err(invalid(
                "mbarrier storage was overwritten without invalidation",
            ));
        }
        Ok(barrier)
    }

    fn barrier_token(&mut self) -> Result<u64> {
        self.asynchronous.next_token = self
            .asynchronous
            .next_token
            .checked_add(1)
            .ok_or(ExecutionError::ExecutionStateLimit)?;
        Ok(self.asynchronous.next_token)
    }

    pub(crate) fn initialize_barrier(
        &mut self,
        memory: &mut Memory,
        address: u64,
        count: u32,
    ) -> Result<()> {
        if !(1..=MAX_ARRIVALS).contains(&count) {
            return Err(invalid("mbarrier arrival count must be in 1..=1048575"));
        }
        if self.asynchronous.barriers.contains_key(&address) {
            return Err(invalid(
                "mbarrier must be invalidated before reinitialization",
            ));
        }
        let current = self.barrier_token()?;
        memory.reserve_execution(self.grid, BARRIER_CHARGE)?;
        memory.write(
            address,
            &current.to_le_bytes(),
            8,
            Some(StateSpace::Shared),
            None,
        )?;
        self.asynchronous.barriers.insert(
            address,
            Barrier {
                expected: count,
                pending: count,
                tracked: 0,
                current,
                previous: None,
                parity: false,
                observed: true,
            },
        );
        Ok(())
    }

    pub(crate) fn invalidate_barrier(&mut self, memory: &mut Memory, address: u64) -> Result<()> {
        if self.barrier_mut(memory, address)?.tracked != 0 {
            return Err(invalid("mbarrier still tracks asynchronous operations"));
        }
        self.asynchronous.barriers.remove(&address);
        memory.release_execution(self.grid, BARRIER_CHARGE);
        Ok(())
    }

    pub(crate) fn arrive_barrier(&mut self, memory: &mut Memory, address: u64) -> Result<u64> {
        let barrier = self.barrier_mut(memory, address)?;
        if !barrier.observed {
            return Err(invalid("previous mbarrier phase was not observed complete"));
        }
        if barrier.pending <= barrier.tracked {
            return Err(invalid(
                "mbarrier arrival count is reserved for asynchronous operations",
            ));
        }
        let state = barrier.current;
        barrier.pending -= 1;
        self.complete_barrier(memory, address)?;
        Ok(state)
    }

    fn complete_barrier(&mut self, memory: &mut Memory, address: u64) -> Result<()> {
        if self.asynchronous.barriers[&address].pending != 0 {
            return Ok(());
        }
        let token = self.barrier_token()?;
        let barrier = self.barrier_mut(memory, address)?;
        barrier.previous = Some(barrier.current);
        barrier.current = token;
        barrier.parity = !barrier.parity;
        barrier.pending = barrier.expected;
        barrier.observed = false;
        memory.write(
            address,
            &token.to_le_bytes(),
            8,
            Some(StateSpace::Shared),
            None,
        )
    }

    pub(crate) fn track_copies(
        &mut self,
        memory: &mut Memory,
        lane: usize,
        address: u64,
        no_increment: bool,
    ) -> Result<()> {
        let barrier = self.barrier_mut(memory, address)?;
        if !barrier.observed {
            return Err(invalid("previous mbarrier phase was not observed complete"));
        }
        if no_increment && barrier.pending <= barrier.tracked {
            return Err(invalid(
                "mbarrier has no arrival available for asynchronous operations",
            ));
        }
        if !no_increment && barrier.pending == MAX_ARRIVALS {
            return Err(invalid("mbarrier pending arrival count exceeds 1048575"));
        }
        // Reserve the marker before changing the barrier's pending count.
        self.push_async(memory, lane, Event::Arrive { address })?;
        let barrier = self.barrier_mut(memory, address)?;
        barrier.tracked += 1;
        if !no_increment {
            barrier.pending += 1;
        }
        self.request_async(lane, self.lanes[lane].asynchronous.issued);
        Ok(())
    }

    pub(crate) fn test_barrier(
        &mut self,
        memory: &Memory,
        address: u64,
        state: u64,
        parity: bool,
    ) -> Result<bool> {
        let barrier = self.barrier_mut(memory, address)?;
        let complete = if parity {
            if barrier.parity != (state & 1 != 0) && barrier.previous.is_none() {
                return Err(invalid("mbarrier parity precedes its initial phase"));
            }
            barrier.parity != (state & 1 != 0)
        } else if state == barrier.current {
            false
        } else if Some(state) == barrier.previous {
            true
        } else {
            return Err(invalid(
                "mbarrier state is stale or belongs to another barrier",
            ));
        };
        if complete {
            barrier.observed = true;
        }
        Ok(complete)
    }
}

impl Engine {
    pub fn execute_async(
        &mut self,
        global: &mut Memory,
        lane: usize,
        operation: &AsyncOperation,
        cancellation: &AtomicBool,
    ) -> Result<Step> {
        match operation {
            AsyncOperation::Copy(copy) => {
                self.issue_copy(global, lane, copy, cancellation)?;
            }
            AsyncOperation::CommitCopies => self.commit_copies(global, lane)?,
            AsyncOperation::WaitCopies { pending, commit } => {
                if *commit {
                    self.commit_copies(global, lane)?;
                }
                if self.wait_copies(lane, *pending, cancellation)? {
                    return Ok(Step::Hold);
                }
            }
            AsyncOperation::TrackCopies {
                address,
                space,
                no_increment,
            } => {
                let address = self.barrier_address(global, lane, *address, *space)?;
                self.track_copies(global, lane, address, *no_increment)?;
            }
            AsyncOperation::BarrierInit {
                address,
                space,
                count,
            } => {
                let count = self.u32_source(lane, count)?;
                let address = self.barrier_address(global, lane, *address, *space)?;
                self.initialize_barrier(global, address, count)?;
            }
            AsyncOperation::BarrierInvalidate { address, space } => {
                let address = self.barrier_address(global, lane, *address, *space)?;
                self.invalidate_barrier(global, address)?;
            }
            AsyncOperation::BarrierArrive {
                destination,
                address,
                space,
            } => {
                let address = self.barrier_address(global, lane, *address, *space)?;
                let state = self.arrive_barrier(global, address)?;
                self.write(lane, *destination, u128::from(state), ScalarType::B64);
            }
            AsyncOperation::BarrierTest {
                destination,
                address,
                space,
                state,
                parity,
            } => {
                let address = self.barrier_address(global, lane, *address, *space)?;
                let ty = if *parity {
                    ScalarType::U32
                } else {
                    ScalarType::B64
                };
                let state = numeric::low_u64(self.source(lane, state, ty)?);
                let complete = self.test_barrier(global, address, state, *parity)?;
                self.write(lane, *destination, u128::from(complete), ScalarType::Pred);
            }
        }
        Ok(Step::Advance)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, atomic::AtomicBool};

    use crate::{
        ast::StateSpace,
        emulator::{
            ParsedModule,
            budget::Budget,
            engine::Engine,
            execution_error::ExecutionError,
            memory::{Memory, Owner},
            runtime::{EmulatorOptions, LaunchConfig},
        },
    };

    fn create(body: &str) -> (Engine, Memory, Arc<Budget>) {
        let parsed = ParsedModule::parse(&format!(
            ".version 9.4 .target sm_80 .address_size 64 .global .u32 data;
             .entry k() {{ .shared .align 16 .b8 tile[64];
                 .shared .align 8 .b64 ready;
                 .reg .b64 state; .reg .pred p; {body} ret; }}"
        ))
        .unwrap();
        let options = EmulatorOptions::default();
        let budget = Arc::new(Budget::new(options.execution_storage_bytes));
        let mut memory = Memory::new(4096, Arc::clone(&budget));
        let flag = AtomicBool::new(false);
        let data = memory
            .allocate(4, 4, StateSpace::Global, Owner::Device, true, &flag)
            .unwrap();
        memory
            .write(
                data,
                &42u32.to_le_bytes(),
                4,
                Some(StateSpace::Global),
                None,
            )
            .unwrap();
        let engine = Engine::create(
            &parsed.compile("k").unwrap(),
            &[data],
            LaunchConfig::create([1; 3], [1; 3], 0).unwrap(),
            &[],
            &options,
            &flag,
            &mut memory,
        )
        .unwrap();
        (engine, memory, budget)
    }

    fn suspend(engine: &mut Engine, memory: &mut Memory) {
        let flag = AtomicBool::new(false);
        for _ in 0..100 {
            assert!(engine.step(memory, &flag).unwrap().is_none());
            if engine.is_paused() {
                return;
            }
        }
        panic!("expected breakpoint");
    }

    #[test]
    fn test_issued_bytes_wait_for_a_later_scheduler_turn_and_release_their_storage() {
        let (mut engine, mut memory, budget) = create(
            "st.shared.u32 [tile],7; cp.async.ca.shared.global [tile],[data],4;
             cp.async.commit_group; brkpt; cp.async.wait_group 0; brkpt;",
        );
        let baseline = budget.used();
        suspend(&mut engine, &mut memory);
        let state = &engine.lanes[0].asynchronous;
        assert_eq!(
            (
                state.issued,
                state.completed,
                state.pending.len(),
                state.groups.len()
            ),
            (1, 0, 1, 1)
        );
        assert!(budget.used() > baseline);
        let destination = match state.pending.front().unwrap().event {
            super::Event::Copy { destination, .. } => destination,
            super::Event::Arrive { .. } => unreachable!(),
        };
        assert_eq!(
            memory
                .read(destination, 4, 4, Some(StateSpace::Shared), None)
                .unwrap(),
            7u32.to_le_bytes()
        );
        engine.resume();
        engine.step(&mut memory, &AtomicBool::new(false)).unwrap();
        assert_eq!(engine.lanes[0].asynchronous.completed, 0);
        assert!(engine.lanes[0].waiting.is_some());
        suspend(&mut engine, &mut memory);
        assert_eq!(
            memory
                .read(destination, 4, 4, Some(StateSpace::Shared), None)
                .unwrap(),
            42u32.to_le_bytes()
        );
        assert_eq!(budget.used(), baseline);
        engine.release(&mut memory);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn test_exhausted_pending_state_does_not_partially_change_a_group_or_barrier() {
        for body in [
            "brkpt; cp.async.ca.shared.global [tile],[data],4;",
            "cp.async.ca.shared.global [tile],[data],4; brkpt; cp.async.commit_group;",
            "brkpt; mbarrier.init.shared.b64 [ready],1;",
            "mbarrier.init.shared.b64 [ready],1; brkpt; cp.async.mbarrier.arrive.shared.b64 [ready];",
        ] {
            let (mut engine, mut memory, budget) = create(body);
            suspend(&mut engine, &mut memory);
            let pending = engine.lanes[0].asynchronous.pending.len();
            let groups = engine.lanes[0].asynchronous.groups.len();
            let barriers = engine.asynchronous.barriers.len();
            let block = budget
                .reserve(EmulatorOptions::default().execution_storage_bytes - budget.used())
                .unwrap();
            engine.resume();
            let error = engine
                .step(&mut memory, &AtomicBool::new(false))
                .unwrap_err();
            assert!(
                matches!(error.root_cause(), ExecutionError::ExecutionStateLimit),
                "{body}: {error}"
            );
            assert_eq!(engine.lanes[0].asynchronous.pending.len(), pending);
            assert_eq!(engine.lanes[0].asynchronous.groups.len(), groups);
            assert_eq!(engine.asynchronous.barriers.len(), barriers);
            for barrier in engine.asynchronous.barriers.values() {
                assert_eq!((barrier.pending, barrier.tracked), (1, 0));
            }
            engine.release(&mut memory);
            drop(block);
            assert_eq!(budget.used(), 0);
        }
    }

    #[test]
    fn test_cancellation_precedes_the_next_copy_completion() {
        let (mut engine, mut memory, budget) = create(
            "st.shared.u32 [tile],7; cp.async.ca.shared.global [tile],[data],4;
             brkpt; cp.async.wait_all;",
        );
        suspend(&mut engine, &mut memory);
        engine.resume();
        engine.step(&mut memory, &AtomicBool::new(false)).unwrap();
        let error = engine
            .step(&mut memory, &AtomicBool::new(true))
            .unwrap_err();
        assert!(matches!(error.root_cause(), ExecutionError::Cancelled));
        assert_eq!(engine.lanes[0].asynchronous.completed, 0);
        engine.release(&mut memory);
        assert_eq!(budget.used(), 0);
    }
}
