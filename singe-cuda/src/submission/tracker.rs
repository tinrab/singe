use std::{
    cell::Cell,
    collections::HashMap,
    marker::PhantomData,
    ops::Range,
    rc::Rc,
    sync::{Arc, Condvar, Mutex, OnceLock, Weak},
};

use crate::{
    error::{Error, Result},
    event::{Event, EventWaitFlags},
    stream::Stream,
    submission::{Access, AccessMode, Resources},
    utility::{lock_for_cleanup, ranges_overlap},
    view::{AllocationId, DeviceRegion},
};

thread_local! { static ACCESS_HELD: Cell<bool> = const { Cell::new(false) }; }

#[derive(Debug)]
struct AccessKey {
    id: AllocationId,
    range: Range<usize>,
    mode: AccessMode,
}

impl From<&Access> for AccessKey {
    fn from(access: &Access) -> Self {
        Self {
            id: access.id,
            range: access.range.clone(),
            mode: access.mode,
        }
    }
}

#[derive(Debug)]
struct PendingAccess {
    touched: Vec<Range<usize>>,
    written: Vec<Range<usize>>,
    event: Weak<Event>,
}

impl PendingAccess {
    fn create(accesses: &[AccessKey], event: Weak<Event>) -> Self {
        // Sorted and merged, so an access finds a conflict by binary search however many a submission made, as a graph's thousands of values do.
        Self {
            touched: merged(accesses.iter()),
            written: merged(
                accesses
                    .iter()
                    .filter(|access| access.mode != AccessMode::Read),
            ),
            event,
        }
    }

    fn conflicts(&self, access: &AccessKey) -> bool {
        let ranges = if access.mode == AccessMode::Read {
            &self.written
        } else {
            &self.touched
        };
        let first = ranges.partition_point(|range| range.end <= access.range.start);
        ranges
            .get(first)
            .is_some_and(|range| ranges_overlap(range, &access.range))
    }
}

fn merged<'a>(accesses: impl Iterator<Item = &'a AccessKey>) -> Vec<Range<usize>> {
    let mut ranges = accesses
        .map(|access| access.range.clone())
        .filter(|range| range.start < range.end)
        .collect::<Vec<_>>();
    ranges.sort_unstable_by_key(|range| range.start);
    let mut merged: Vec<Range<usize>> = Vec::with_capacity(ranges.len());
    for range in ranges {
        match merged.last_mut() {
            Some(last) if range.start <= last.end => last.end = last.end.max(range.end),
            _ => merged.push(range),
        }
    }
    merged
}

// Allocation identity is global: portable mapped host storage can have aliases in different contexts.
#[derive(Debug, Default)]
struct AccessState {
    occupied: bool,
    failed: bool,
    pending: Vec<PendingAccess>,
}

#[derive(Debug)]
struct AccessCell {
    id: AllocationId,
    state: Mutex<AccessState>,
    available: Condvar,
}

#[derive(Debug, Default)]
pub(crate) struct AccessTracker {
    failed: bool,
}

impl AccessTracker {
    // Multiple primary-context wrappers share failure state for the actual CUDA context.
    pub(crate) fn for_context(handle: usize) -> Result<Arc<Mutex<Self>>> {
        static TRACKERS: OnceLock<Mutex<HashMap<usize, Weak<Mutex<AccessTracker>>>>> =
            OnceLock::new();
        let mut trackers = TRACKERS
            .get_or_init(Mutex::default)
            .lock()
            .map_err(|_| Error::lock_poisoned("access trackers"))?;
        trackers.retain(|_, tracker| tracker.strong_count() != 0);
        if let Some(tracker) = trackers.get(&handle).and_then(Weak::upgrade) {
            return Ok(tracker);
        }

        let tracker = Arc::new(Mutex::new(Self::default()));
        trackers.insert(handle, Arc::downgrade(&tracker));
        drop(trackers);
        Ok(tracker)
    }

    pub(crate) fn fail(&mut self) {
        self.failed = true;
    }
}

// Gates are acquired in allocation order, then held through enqueue or host access.
// No mutex guard is held across user code, so unwind releases gates without poisoning them.
// The marker keeps this thread-local reentrancy guard on its originating thread.
pub(crate) struct AccessGuard<'a> {
    tracker: &'a Mutex<AccessTracker>,
    cells: Vec<Arc<AccessCell>>,
    acquired: usize,
    thread: PhantomData<Rc<()>>,
}

fn by_allocation(accesses: &[Access]) -> HashMap<AllocationId, Vec<AccessKey>> {
    let mut grouped = HashMap::<_, Vec<_>>::new();
    for access in accesses {
        grouped
            .entry(access.id)
            .or_default()
            .push(AccessKey::from(access));
    }
    grouped
}

pub(crate) fn lock_access<'a>(
    tracker: &'a Mutex<AccessTracker>,
    accesses: &[Access],
) -> Result<AccessGuard<'a>> {
    static ALLOCATIONS: OnceLock<Mutex<HashMap<AllocationId, Weak<AccessCell>>>> = OnceLock::new();
    if ACCESS_HELD.get() {
        return Err(Error::ReentrantAccess);
    }

    if tracker
        .lock()
        .map_err(|_| Error::lock_poisoned("access tracker"))?
        .failed
    {
        return Err(Error::lock_poisoned("access tracker"));
    }

    let cells = {
        let mut allocations = ALLOCATIONS
            .get_or_init(Mutex::default)
            .lock()
            .map_err(|_| Error::lock_poisoned("access allocations"))?;
        allocations.retain(|_, cell| cell.strong_count() != 0);
        let mut ids: Vec<_> = accesses.iter().map(|access| access.id).collect();
        ids.sort_unstable();
        ids.dedup();
        let cells = ids
            .into_iter()
            .map(|id| {
                if let Some(cell) = allocations.get(&id).and_then(Weak::upgrade) {
                    return cell;
                }

                let cell = Arc::new(AccessCell {
                    id,
                    state: Mutex::new(AccessState::default()),
                    available: Condvar::new(),
                });
                allocations.insert(id, Arc::downgrade(&cell));
                cell
            })
            .collect();
        drop(allocations);
        cells
    };
    ACCESS_HELD.set(true);
    let mut guard = AccessGuard {
        tracker,
        cells,
        acquired: 0,
        thread: PhantomData,
    };
    for cell in &guard.cells {
        let mut state = cell
            .state
            .lock()
            .map_err(|_| Error::lock_poisoned("allocation access state"))?;
        while state.occupied {
            state = cell
                .available
                .wait(state)
                .map_err(|_| Error::lock_poisoned("allocation access state"))?;
        }

        if state.failed {
            return Err(Error::lock_poisoned("allocation access state"));
        }

        state.occupied = true;
        drop(state);
        guard.acquired += 1;
    }

    if tracker
        .lock()
        .map_err(|_| Error::lock_poisoned("access tracker"))?
        .failed
    {
        return Err(Error::lock_poisoned("access tracker"));
    }
    Ok(guard)
}

impl AccessState {
    fn prune_completed(&mut self) -> Result<()> {
        let mut index = 0;
        while index < self.pending.len() {
            let complete = match self.pending[index].event.upgrade() {
                Some(event) => event.query()?,
                None => true,
            };
            if complete {
                self.pending.swap_remove(index);
            } else {
                index += 1;
            }
        }
        Ok(())
    }
}

impl AccessGuard<'_> {
    // Keep the global identity entry alive until every submitted use completes.
    pub(crate) fn retain(&self, resources: &mut Resources) {
        for cell in &self.cells {
            resources.retain(Arc::clone(cell));
        }
    }

    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "ordering and recording require exclusive access to the tracking transaction guard"
    )]
    pub(crate) fn order(&mut self, stream: &Stream, accesses: &[Access]) -> Result<()> {
        for access in accesses {
            if let Some(ready) = &access.readiness {
                stream.join(ready)?;
            }
        }

        let keys = by_allocation(accesses);
        for cell in &self.cells {
            let Some(keys) = keys.get(&cell.id) else {
                continue;
            };
            let mut state = cell
                .state
                .lock()
                .map_err(|_| Error::lock_poisoned("allocation access state"))?;
            state.prune_completed()?;
            for pending in &state.pending {
                if keys.iter().any(|key| pending.conflicts(key))
                    && let Some(event) = pending.event.upgrade()
                {
                    if event.context() == stream.context() {
                        stream.wait_event(&event, EventWaitFlags::DEFAULT)?;
                    } else {
                        // Avoid propagating a failed context's event wait into another context.
                        event.synchronize()?;
                    }
                }
            }
        }
        Ok(())
    }

    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "ordering and recording require exclusive access to the tracking transaction guard"
    )]
    pub(crate) fn record(&mut self, accesses: &[Access], event: &Arc<Event>) {
        let mut accesses = by_allocation(accesses);
        for cell in &self.cells {
            let accesses = accesses.remove(&cell.id).unwrap_or_default();
            lock_for_cleanup(&cell.state)
                .pending
                .push(PendingAccess::create(&accesses, Arc::downgrade(event)));
        }
    }

    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "ordering and recording require exclusive access to the tracking transaction guard"
    )]
    pub(crate) fn fail(&mut self) {
        lock_for_cleanup(self.tracker).fail();
        for cell in &self.cells {
            lock_for_cleanup(&cell.state).failed = true;
        }
    }

    pub(crate) fn synchronize(&mut self, region: DeviceRegion<'_>) -> Result<()> {
        self.synchronize_access(&Access::new(region, AccessMode::ReadWrite))
    }

    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "ordering and recording require exclusive access to the tracking transaction guard"
    )]
    pub(crate) fn synchronize_access(&mut self, access: &Access) -> Result<()> {
        let access = AccessKey::from(access);
        for cell in self.cells.iter().filter(|cell| cell.id == access.id) {
            let mut state = cell
                .state
                .lock()
                .map_err(|_| Error::lock_poisoned("allocation access state"))?;
            for pending in &state.pending {
                if pending.conflicts(&access)
                    && let Some(event) = pending.event.upgrade()
                {
                    event.synchronize()?;
                }
            }

            state.prune_completed()?;
        }
        Ok(())
    }
}

impl Drop for AccessGuard<'_> {
    fn drop(&mut self) {
        for cell in &self.cells[..self.acquired] {
            lock_for_cleanup(&cell.state).occupied = false;
            cell.available.notify_all();
        }

        ACCESS_HELD.set(false);
    }
}
