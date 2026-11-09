use std::{
    cell::RefCell,
    collections::HashMap,
    fmt::{self, Debug, Formatter},
    marker::PhantomData,
    rc::Rc,
    sync::{Arc, Weak},
};

use crate::{
    dispatch::{DispatchKey, excluded},
    error::{CaptureError, Error, Result},
    storage::Storage,
};

thread_local! { static CAPTURE_READS: RefCell<Option<ReadPolicy>> = const { RefCell::new(None) }; }

struct ReadPolicy {
    tracked: HashMap<usize, Weak<Storage>>,
    allow_specialization: bool,
}

/// Owns observation policy on the recording thread, including error/unwind paths.
pub struct CaptureReadGuard {
    _thread: PhantomData<Rc<()>>,
}

impl CaptureReadGuard {
    pub(crate) fn enter(allow_specialization: bool) -> Result<Self> {
        CAPTURE_READS.with(|policy| {
            let mut policy = policy
                .try_borrow_mut()
                .map_err(|_| Error::from(CaptureError::SessionStillBorrowed))?;
            if policy.is_some() {
                return Err(CaptureError::AlreadyActive.into());
            }

            *policy = Some(ReadPolicy {
                tracked: HashMap::new(),
                allow_specialization,
            });
            Ok(Self {
                _thread: PhantomData,
            })
        })
    }
}

impl Debug for CaptureReadGuard {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CaptureReadGuard")
            .finish_non_exhaustive()
    }
}

impl Drop for CaptureReadGuard {
    fn drop(&mut self) {
        CAPTURE_READS.with(|policy| {
            if let Ok(mut policy) = policy.try_borrow_mut() {
                *policy = None;
            }
        });
    }
}

pub fn track_capture_storage(storage: &Arc<Storage>) -> Result<()> {
    CAPTURE_READS.with(|policy| {
        if let Some(policy) = policy
            .try_borrow_mut()
            .map_err(|_| Error::from(CaptureError::SessionStillBorrowed))?
            .as_mut()
        {
            // Weak ownership keeps this identity from being recycled without retaining every intermediate's underlying device allocation.
            policy
                .tracked
                .insert(storage.storage_identity(), Arc::downgrade(storage));
        }
        Ok(())
    })
}

pub fn check_storage_readback(identity: usize) -> Result<()> {
    if excluded().contains(DispatchKey::Capture) {
        return Ok(());
    }
    CAPTURE_READS.with(|policy| {
        let policy = policy
            .try_borrow()
            .map_err(|_| Error::from(CaptureError::SessionStillBorrowed))?;
        if let Some(policy) = policy.as_ref()
            && policy
                .tracked
                .get(&identity)
                .is_some_and(|owner| owner.upgrade().is_some())
            && !policy.allow_specialization
        {
            return Err(CaptureError::Readback.into());
        }
        Ok(())
    })
}
