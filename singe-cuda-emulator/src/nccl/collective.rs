use std::{
    collections::HashMap,
    ffi::c_int,
    slice,
    sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError},
    task::Poll,
    time::{Duration, Instant},
};

use singe_nccl_sys::ncclResult_t;
use singe_ptx::emulator::{ExecutionError, HostMemory};

use crate::nccl::{
    reduce::{self, DataType, Reduction, Scalar},
    symmetric::Spaces,
};

/// How long a rank waits at a rendezvous for the others before the operation fails.
///
/// The emulator cannot see whether the missing ranks will still arrive, so a wait this long stands for a deadlock or a rank that never calls.
pub(crate) const RENDEZVOUS_TIMEOUT: Duration = Duration::from_secs(120);

/// What one rank asks a collective or transfer to do.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Kind {
    AllReduce(Reduction),
    Broadcast { root: usize },
    Reduce { reduction: Reduction, root: usize },
    AllGather,
    ReduceScatter(Reduction),
    AllToAll,
    Gather { root: usize },
    Scatter { root: usize },
    Send,
    Receive,
}

impl Kind {
    fn matches(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::AllReduce(reduction), Self::AllReduce(other_reduction))
            | (Self::ReduceScatter(reduction), Self::ReduceScatter(other_reduction)) => {
                reduction.matches(other_reduction)
            }
            (
                Self::Reduce { reduction, root },
                Self::Reduce {
                    reduction: other_reduction,
                    root: other_root,
                },
            ) => reduction.matches(other_reduction) && root == other_root,
            _ => self == other,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::AllReduce(_) => "all-reduce",
            Self::Broadcast { .. } => "broadcast",
            Self::Reduce { .. } => "reduce",
            Self::AllGather => "all-gather",
            Self::ReduceScatter(_) => "reduce-scatter",
            Self::AllToAll => "all-to-all",
            Self::Gather { .. } => "gather",
            Self::Scatter { .. } => "scatter",
            Self::Send => "send",
            Self::Receive => "receive",
        }
    }

    fn reduction(&self) -> Option<&Reduction> {
        match self {
            Self::AllReduce(reduction)
            | Self::ReduceScatter(reduction)
            | Self::Reduce { reduction, .. } => Some(reduction),
            Self::Broadcast { .. }
            | Self::AllGather
            | Self::AllToAll
            | Self::Gather { .. }
            | Self::Scatter { .. }
            | Self::Send
            | Self::Receive => None,
        }
    }

    pub(crate) fn root(&self) -> Option<usize> {
        match self {
            Self::Broadcast { root }
            | Self::Reduce { root, .. }
            | Self::Gather { root }
            | Self::Scatter { root } => Some(*root),
            Self::AllReduce(_)
            | Self::AllGather
            | Self::ReduceScatter(_)
            | Self::AllToAll
            | Self::Send
            | Self::Receive => None,
        }
    }
}

/// One rank's part of an operation: what it asks for and its buffers.
#[derive(Debug, Clone)]
pub(crate) struct Contribution {
    pub(crate) kind: Kind,
    pub(crate) data_type: DataType,
    pub(crate) count: usize,
    pub(crate) send: u64,
    pub(crate) receive: u64,
}

/// Identifies an operation within a clique: the collective's sequence number, or a transfer's ranks and sequence number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Key {
    Collective(u64),
    Transfer {
        from: usize,
        to: usize,
        sequence: u64,
    },
}

impl Key {
    /// Returns the rank of `party`: a collective's parties are its ranks, and a transfer's are its sender and receiver.
    fn rank(self, party: usize) -> usize {
        match self {
            Self::Collective(_) => party,
            Self::Transfer { from, to, .. } => {
                if party == 0 {
                    from
                } else {
                    to
                }
            }
        }
    }
}

#[derive(Default)]
struct Slot {
    arrivals: Vec<Option<Contribution>>,
    outcome: Option<Result<(), ExecutionError>>,
    departed: usize,
    since: Option<Instant>,
}

impl Slot {
    fn missing_ranks(&self, key: Key) -> impl Iterator<Item = usize> {
        self.arrivals
            .iter()
            .enumerate()
            .filter(|(_, arrival)| arrival.is_none())
            .map(move |(party, _)| key.rank(party))
    }
}

/// The ranks of communicators created from one unique identifier, and the operations they have in flight.
pub(crate) struct Clique {
    pub(crate) size: usize,
    slots: Mutex<HashMap<Key, Arc<Mutex<Slot>>>>,
    devices: Mutex<Vec<Option<usize>>>,
    known: Condvar,
    pub(crate) spaces: Mutex<Spaces>,
    /// Splits of this clique's communicators, by sequence number.
    pub(crate) splits: Mutex<HashMap<u64, Arc<Split>>>,
    /// Host signals that arrived and no wait has taken yet, by receiver, sender, signal, and context.
    pub(crate) signals: Mutex<HashMap<(usize, usize, c_int, c_int), u64>>,
}

pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Clique {
    pub(crate) fn new(size: usize) -> Self {
        Self {
            size,
            slots: Mutex::new(HashMap::new()),
            devices: Mutex::new(vec![None; size]),
            known: Condvar::new(),
            spaces: Mutex::new(Spaces::default()),
            splits: Mutex::new(HashMap::new()),
            signals: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) fn set_device(&self, rank: usize, device: usize) {
        lock(&self.devices)[rank] = Some(device);
        self.known.notify_all();
    }

    /// Returns each rank's device, waiting for ranks that have not joined yet.
    pub(crate) fn devices(&self) -> Result<Vec<usize>, ncclResult_t> {
        let devices = self
            .known
            .wait_timeout_while(lock(&self.devices), RENDEZVOUS_TIMEOUT, |devices| {
                devices.iter().any(Option::is_none)
            })
            .unwrap_or_else(PoisonError::into_inner)
            .0;
        devices
            .iter()
            .copied()
            .collect::<Option<Vec<usize>>>()
            .ok_or(ncclResult_t::ncclInvalidUsage)
    }

    /// Returns `rank`'s device if the rank has joined.
    pub(crate) fn device_of(&self, rank: usize) -> Option<usize> {
        lock(&self.devices).get(rank).copied().flatten()
    }

    fn slot(&self, key: Key) -> Arc<Mutex<Slot>> {
        let parties = match key {
            Key::Collective(_) => self.size,
            Key::Transfer { .. } => 2,
        };
        Arc::clone(lock(&self.slots).entry(key).or_insert_with(|| {
            Arc::new(Mutex::new(Slot {
                arrivals: vec![None; parties],
                ..Slot::default()
            }))
        }))
    }
}

/// One rank's part of an operation, polled until every party has arrived and the operation ran.
pub(crate) struct Part {
    clique: Arc<Clique>,
    key: Key,
    slot: Arc<Mutex<Slot>>,
    party: usize,
    contribution: Option<Contribution>,
    done: bool,
}

impl Part {
    pub(crate) fn new(
        clique: &Arc<Clique>,
        key: Key,
        party: usize,
        contribution: Contribution,
    ) -> Self {
        Self {
            slot: clique.slot(key),
            clique: Arc::clone(clique),
            key,
            party,
            contribution: Some(contribution),
            done: false,
        }
    }

    /// Arrives if it has not, runs the operation once every party has, and returns whether it ran.
    fn poll(&mut self, memory: &mut HostMemory<'_>) -> Result<bool, ExecutionError> {
        if self.done {
            return Ok(true);
        }

        let mut slot = lock(&self.slot);
        if let Some(contribution) = self.contribution.take() {
            slot.arrivals[self.party] = Some(contribution);
            slot.since.get_or_insert_with(Instant::now);
        }

        if slot.outcome.is_none() && slot.arrivals.iter().all(Option::is_some) {
            let arrivals: Vec<Contribution> = slot.arrivals.iter().flatten().cloned().collect();
            slot.outcome = Some(perform(memory, &arrivals));
        }

        let Some(outcome) = slot.outcome.clone() else {
            let faulted = slot.missing_ranks(self.key).find(|&rank| {
                self.clique
                    .device_of(rank)
                    .is_some_and(|device| memory.is_faulted(device))
            });
            if let Some(rank) = faulted {
                drop(slot);
                return Err(ExecutionError::Invalid(format!(
                    "nccl operation {:?} cannot finish: the device of rank {rank} faulted before the rank called it",
                    self.key
                )));
            }

            if slot
                .since
                .is_none_or(|since| since.elapsed() <= RENDEZVOUS_TIMEOUT)
            {
                return Ok(false);
            }

            let missing: Vec<usize> = slot.missing_ranks(self.key).collect();
            drop(slot);
            return Err(ExecutionError::Invalid(format!(
                "nccl operation {:?} waited {} s for ranks {missing:?}, which never called it",
                self.key,
                RENDEZVOUS_TIMEOUT.as_secs()
            )));
        };
        self.done = true;
        slot.departed += 1;
        let is_last = slot.departed == slot.arrivals.len();
        drop(slot);
        if is_last {
            lock(&self.clique.slots).remove(&self.key);
        }
        outcome.map(|()| true)
    }
}

/// Polls every part of a group on one stream; the group finishes when all of its parts have run.
pub(crate) fn poll_parts(
    parts: &mut [Part],
    memory: &mut HostMemory<'_>,
) -> Poll<Result<(), ExecutionError>> {
    let mut finished = true;
    for part in parts {
        finished &= part.poll(memory)?;
    }
    if finished {
        Poll::Ready(Ok(()))
    } else {
        Poll::Pending
    }
}

fn read(memory: &HostMemory<'_>, address: u64, len: usize) -> Result<Vec<u8>, ExecutionError> {
    Ok(memory.read(address, len)?.to_vec())
}

fn scalar(memory: &HostMemory<'_>, arrival: &Contribution) -> Result<Vec<u8>, ExecutionError> {
    match arrival.kind.reduction() {
        Some(Reduction::ScaledSum(Scalar::Host(bytes))) => Ok(bytes.clone()),
        Some(Reduction::ScaledSum(Scalar::Device(address))) => {
            read(memory, *address, arrival.data_type.size())
        }
        _ => Ok(Vec::new()),
    }
}

/// Runs an operation once every party has arrived, reading every input before writing any output, so in-place buffers work.
fn perform(memory: &mut HostMemory<'_>, arrivals: &[Contribution]) -> Result<(), ExecutionError> {
    let first = &arrivals[0];
    if matches!(first.kind, Kind::Send | Kind::Receive) {
        let (send, receive) = (&arrivals[0], &arrivals[1]);
        if send.data_type != receive.data_type || send.count != receive.count {
            return Err(ExecutionError::Invalid(format!(
                "nccl send of {} {:?} elements met a receive of {} {:?} elements",
                send.count, send.data_type, receive.count, receive.data_type
            )));
        }

        let bytes = send.count * send.data_type.size();
        return memory.copy(send.send, receive.receive, bytes);
    }

    for (rank, arrival) in arrivals.iter().enumerate() {
        if !arrival.kind.matches(&first.kind)
            || arrival.data_type != first.data_type
            || arrival.count != first.count
        {
            return Err(ExecutionError::Invalid(format!(
                "nccl ranks disagree on a collective: rank 0 called {} of {} {:?} elements, rank {rank} called {} of {} {:?} elements",
                first.kind.name(),
                first.count,
                first.data_type,
                arrival.kind.name(),
                arrival.count,
                arrival.data_type,
            )));
        }
    }

    let ranks = arrivals.len();
    let chunk = first.count * first.data_type.size();
    let scalars = arrivals
        .iter()
        .map(|arrival| scalar(memory, arrival))
        .collect::<Result<Vec<_>, _>>()?;
    let combined = |memory: &HostMemory<'_>,
                    reduction: &Reduction,
                    offset: usize|
     -> Result<Vec<u8>, ExecutionError> {
        let inputs = arrivals
            .iter()
            .map(|arrival| read(memory, arrival.send + offset as u64, chunk))
            .collect::<Result<Vec<_>, _>>()?;
        let pairs: Vec<(&[u8], &[u8])> = inputs
            .iter()
            .zip(&scalars)
            .map(|(input, scalar)| (input.as_slice(), scalar.as_slice()))
            .collect();
        let mut output = vec![0; chunk];
        reduce::reduce(first.data_type, reduction, &pairs, &mut output);
        Ok(output)
    };

    match &first.kind {
        Kind::AllReduce(reduction) => {
            let output = combined(memory, reduction, 0)?;
            for arrival in arrivals {
                memory.write(arrival.receive, &output)?;
            }
        }
        Kind::Reduce { reduction, root } => {
            let output = combined(memory, reduction, 0)?;
            memory.write(arrivals[*root].receive, &output)?;
        }
        Kind::ReduceScatter(reduction) => {
            let outputs = (0..ranks)
                .map(|rank| combined(memory, reduction, rank * chunk))
                .collect::<Result<Vec<_>, _>>()?;
            for (arrival, output) in arrivals.iter().zip(outputs) {
                memory.write(arrival.receive, &output)?;
            }
        }
        Kind::Broadcast { root } => {
            let input = read(memory, arrivals[*root].send, chunk)?;
            for arrival in arrivals {
                memory.write(arrival.receive, &input)?;
            }
        }
        Kind::AllGather | Kind::Gather { .. } => {
            let inputs = arrivals
                .iter()
                .map(|arrival| read(memory, arrival.send, chunk))
                .collect::<Result<Vec<_>, _>>()?;
            let targets = match &first.kind {
                Kind::Gather { root } => slice::from_ref(&arrivals[*root]),
                _ => arrivals,
            };
            for target in targets {
                for (rank, input) in inputs.iter().enumerate() {
                    memory.write(target.receive + (rank * chunk) as u64, input)?;
                }
            }
        }
        Kind::Scatter { root } => {
            let input = read(memory, arrivals[*root].send, chunk * ranks)?;
            for (rank, arrival) in arrivals.iter().enumerate() {
                memory.write(arrival.receive, &input[rank * chunk..(rank + 1) * chunk])?;
            }
        }
        Kind::AllToAll => {
            let inputs = arrivals
                .iter()
                .map(|arrival| read(memory, arrival.send, chunk * ranks))
                .collect::<Result<Vec<_>, _>>()?;
            for (to, arrival) in arrivals.iter().enumerate() {
                for (from, input) in inputs.iter().enumerate() {
                    memory.write(
                        arrival.receive + (from * chunk) as u64,
                        &input[to * chunk..(to + 1) * chunk],
                    )?;
                }
            }
        }
        Kind::Send | Kind::Receive => unreachable!("transfers return above"),
    }
    Ok(())
}

/// A rank's clique and rank after a split, or `None` for a rank the split leaves out.
type Assignment = Option<(Arc<Clique>, usize)>;

#[derive(Default)]
struct SplitState {
    arrivals: Vec<Option<(c_int, c_int, usize)>>,
    results: Option<Vec<Assignment>>,
}

/// One `ncclCommSplit` across the ranks of a clique, resolved once every rank has called it.
pub(crate) struct Split {
    state: Mutex<SplitState>,
    complete: Condvar,
}

impl Split {
    pub(crate) fn new(ranks: usize) -> Self {
        Self {
            state: Mutex::new(SplitState {
                arrivals: vec![None; ranks],
                ..SplitState::default()
            }),
            complete: Condvar::new(),
        }
    }

    /// Records a rank's color, key, and device; a negative color leaves the rank out of every new communicator.
    pub(crate) fn arrive(&self, rank: usize, color: c_int, key: c_int, device: usize) {
        let mut state = lock(&self.state);
        state.arrivals[rank] = Some((color, key, device));
        let Some(arrivals) = state.arrivals.iter().copied().collect::<Option<Vec<_>>>() else {
            return;
        };
        state.results = Some(assign(&arrivals));
        drop(state);
        self.complete.notify_all();
    }

    /// Returns the clique and rank of `rank`'s new communicator, waiting for the other ranks to call the split.
    pub(crate) fn resolve(&self, rank: usize) -> Result<(Arc<Clique>, usize), ncclResult_t> {
        let state = self
            .complete
            .wait_timeout_while(lock(&self.state), RENDEZVOUS_TIMEOUT, |state| {
                state.results.is_none()
            })
            .unwrap_or_else(PoisonError::into_inner)
            .0;
        state
            .results
            .as_ref()
            .and_then(|results| results[rank].clone())
            .ok_or(ncclResult_t::ncclInvalidUsage)
    }
}

fn assign(arrivals: &[(c_int, c_int, usize)]) -> Vec<Assignment> {
    let mut assignments = vec![None; arrivals.len()];
    let mut colors: Vec<c_int> = arrivals
        .iter()
        .map(|(color, _, _)| *color)
        .filter(|color| *color >= 0)
        .collect();
    colors.sort_unstable();
    colors.dedup();
    for color in colors {
        let mut members: Vec<usize> = (0..arrivals.len())
            .filter(|rank| arrivals[*rank].0 == color)
            .collect();
        members.sort_by_key(|rank| (arrivals[*rank].1, *rank));
        let clique = Arc::new(Clique::new(members.len()));
        for (new_rank, parent_rank) in members.into_iter().enumerate() {
            clique.set_device(new_rank, arrivals[parent_rank].2);
            assignments[parent_rank] = Some((Arc::clone(&clique), new_rank));
        }
    }
    assignments
}
