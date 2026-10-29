use std::{
    cell::Cell,
    collections::HashMap,
    marker::PhantomData,
    ops::Range,
    rc::Rc,
    sync::{Arc, Condvar, Mutex, OnceLock, PoisonError, Weak},
};

use crate::{
    error::{Error, Result},
    event::Event,
    stream::Stream,
    submission::{Access, AccessMode, Resources},
    view::{AllocationId, DeviceRegion, ranges_overlap},
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

impl AccessKey {
    fn conflicts(&self, other: &Self) -> bool {
        self.id == other.id
            && ranges_overlap(&self.range, &other.range)
            && (self.mode != AccessMode::Read || other.mode != AccessMode::Read)
    }
}

#[derive(Debug)]
struct PendingAccess {
    accesses: Vec<AccessKey>,
    event: Weak<Event>,
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
pub struct AccessTracker {
    failed: bool,
}

impl AccessTracker {
    // Multiple primary-context wrappers share failure state for the actual CUDA context.
    pub(crate) fn for_context(handle: usize) -> Arc<Mutex<Self>> {
        static TRACKERS: OnceLock<Mutex<HashMap<usize, Weak<Mutex<AccessTracker>>>>> =
            OnceLock::new();
        let mut trackers = TRACKERS
            .get_or_init(Mutex::default)
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        trackers.retain(|_, tracker| tracker.strong_count() != 0);
        if let Some(tracker) = trackers.get(&handle).and_then(Weak::upgrade) {
            return tracker;
        }
        let tracker = Arc::new(Mutex::new(Self::default()));
        trackers.insert(handle, Arc::downgrade(&tracker));
        tracker
    }

    pub(crate) fn fail(&mut self) {
        self.failed = true;
    }
}

// Gates are acquired in allocation order, then held through enqueue or host access.
// No mutex guard is held across user code, so unwind releases gates without poisoning them.
// The marker keeps this thread-local reentrancy guard on its originating thread.
pub struct AccessGuard<'a> {
    tracker: &'a Mutex<AccessTracker>,
    cells: Vec<Arc<AccessCell>>,
    acquired: usize,
    thread: PhantomData<Rc<()>>,
}

pub fn lock_access<'a>(
    tracker: &'a Mutex<AccessTracker>,
    accesses: &[Access],
) -> Result<AccessGuard<'a>> {
    static ALLOCATIONS: OnceLock<Mutex<HashMap<AllocationId, Weak<AccessCell>>>> = OnceLock::new();
    if ACCESS_HELD.get() {
        return Err(Error::ReentrantAccess);
    }
    if tracker
        .lock()
        .map_err(|_| Error::AccessTrackingPoisoned)?
        .failed
    {
        return Err(Error::AccessTrackingPoisoned);
    }
    let cells = {
        let mut allocations = ALLOCATIONS
            .get_or_init(Mutex::default)
            .lock()
            .map_err(|_| Error::AccessTrackingPoisoned)?;
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
            .map_err(|_| Error::AccessTrackingPoisoned)?;
        while state.occupied {
            state = cell
                .available
                .wait(state)
                .map_err(|_| Error::AccessTrackingPoisoned)?;
        }
        if state.failed {
            return Err(Error::AccessTrackingPoisoned);
        }
        state.occupied = true;
        drop(state);
        guard.acquired += 1;
    }
    if tracker
        .lock()
        .map_err(|_| Error::AccessTrackingPoisoned)?
        .failed
    {
        return Err(Error::AccessTrackingPoisoned);
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
        let keys: Vec<_> = accesses.iter().map(AccessKey::from).collect();
        for cell in &self.cells {
            let mut state = cell
                .state
                .lock()
                .map_err(|_| Error::AccessTrackingPoisoned)?;
            state.prune_completed()?;
            for pending in &state.pending {
                if keys.iter().any(|key| {
                    pending
                        .accesses
                        .iter()
                        .any(|previous| key.conflicts(previous))
                }) && let Some(event) = pending.event.upgrade()
                {
                    if event.context() == stream.context() {
                        stream.wait_event(&event)?;
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
        for cell in &self.cells {
            let accesses = accesses
                .iter()
                .filter(|access| access.id == cell.id)
                .map(AccessKey::from)
                .collect();
            cell.state
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .pending
                .push(PendingAccess {
                    accesses,
                    event: Arc::downgrade(event),
                });
        }
    }

    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "ordering and recording require exclusive access to the tracking transaction guard"
    )]
    pub(crate) fn fail(&mut self) {
        self.tracker
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .fail();
        for cell in &self.cells {
            cell.state
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .failed = true;
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
                .map_err(|_| Error::AccessTrackingPoisoned)?;
            for pending in &state.pending {
                if pending
                    .accesses
                    .iter()
                    .any(|previous| access.conflicts(previous))
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
            cell.state
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .occupied = false;
            cell.available.notify_all();
        }
        ACCESS_HELD.set(false);
    }
}
