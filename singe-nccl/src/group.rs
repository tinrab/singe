use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use crate::{
    error::{Error, Result},
    sys, try_ffi,
    types::SimulationInfo,
};

/// Start a group call.
///
/// Subsequent NCCL calls do not block on inter-CPU synchronization until
/// [`end`].
///
/// # Errors
///
/// Returns an error if NCCL rejects the group start.
pub fn start() -> Result<()> {
    // SAFETY: `ncclGroupStart` takes no arguments.
    unsafe {
        try_ffi!(sys::ncclGroupStart())?;
    }
    track_start();
    Ok(())
}

/// End a group call.
///
/// Returns when all operations since [`start`] have been processed.
/// This means the communication primitives have been enqueued to the provided streams, but are not necessarily complete.
///
/// When used with the [`Communicator::create_rank`](crate::communicator::Communicator::create_rank) call, the [`end`] call waits for all communicators to be initialized.
///
/// # Errors
///
/// Returns an error if any grouped NCCL operation fails or if NCCL rejects the
/// group end.
pub fn end() -> Result<()> {
    // SAFETY: `ncclGroupEnd` takes no arguments.
    let result = unsafe { try_ffi!(sys::ncclGroupEnd()) };
    track_end();
    result
}

/// Starts a group and returns NCCL's simulated execution metadata for it.
///
/// # Errors
///
/// Returns an error if NCCL cannot start or simulate the group.
pub fn simulate() -> Result<SimulationInfo> {
    Group::start()?.simulate_end()
}

/// A thread-bound NCCL operation group that is ended when consumed or dropped.
#[derive(Debug)]
pub struct Group {
    closed: bool,
    _thread: PhantomData<Rc<()>>,
}

impl Group {
    /// Starts a thread-local NCCL operation group.
    ///
    /// # Errors
    ///
    /// Returns an error if NCCL rejects the group start.
    pub fn start() -> Result<Self> {
        start()?;
        Ok(Self {
            closed: false,
            _thread: PhantomData,
        })
    }

    /// Ends the group, enqueueing its operations on their selected streams.
    ///
    /// This does not wait for GPU execution to complete.
    ///
    /// # Errors
    ///
    /// Returns an error if NCCL rejects group completion or a grouped operation fails.
    pub fn end(mut self) -> Result<()> {
        self.closed = true;
        end()
    }

    /// Simulates [`end`] and returns NCCL's estimated execution metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if NCCL cannot simulate the current group.
    pub fn simulate_end(mut self) -> Result<SimulationInfo> {
        let mut value = SimulationInfo::default().into_raw();
        self.closed = true;
        // SAFETY: `ncclGroupSimulateEnd` receives valid output locations.
        let result = unsafe { try_ffi!(sys::ncclGroupSimulateEnd(&raw mut value)) };
        track_end();
        result?;
        Ok(value.into())
    }
}

/// Runs `f` inside an NCCL group and ends the group before returning.
///
/// Group completion enqueues work; it does not synchronize the streams.
///
/// # Errors
///
/// Returns an error if group start, `f`, or group completion fails.
/// When both `f` and group completion fail, the error from `f` is returned.
pub fn with_group<R>(f: impl FnOnce() -> Result<R>) -> Result<R> {
    let group = Group::start()?;
    let result = f();
    let end_result = group.end();
    match (result, end_result) {
        (Ok(value), Ok(())) => Ok(value),
        (Ok(_), Err(error)) | (Err(error), Ok(())) => Err(error),
        (Err(error), Err(end_error)) => {
            #[cfg(debug_assertions)]
            eprintln!("failed to end nccl group after grouped operation error: {end_error}");
            Err(error)
        }
    }
}

impl Drop for Group {
    fn drop(&mut self) {
        if self.closed {
            return;
        }

        if let Err(error) = end() {
            #[cfg(debug_assertions)]
            eprintln!("failed to end nccl group: {error}");
        }
    }
}

#[derive(Default)]
struct Configuration {
    depth: usize,
    #[expect(
        clippy::option_option,
        reason = "unset group configuration differs from an explicit request for the native default cluster size"
    )]
    cluster_size: Option<Option<i32>>,
    event_communicators: Vec<usize>,
}

thread_local! {
    static CONFIGURATION: RefCell<Configuration> = RefCell::default();
}

pub(crate) fn track_start() {
    CONFIGURATION.with_borrow_mut(|config| config.depth += 1);
}

pub(crate) fn track_end() {
    CONFIGURATION.with_borrow_mut(|config| {
        config.depth = config.depth.saturating_sub(1);
        if config.depth == 0 {
            *config = Configuration::default();
        }
    });
}

pub(crate) fn validate_collective(
    comm: sys::ncclComm_t,
    cluster_size: Option<i32>,
    event: bool,
) -> Result<()> {
    CONFIGURATION.with_borrow_mut(|config| {
        if config.depth == 0 {
            return Ok(());
        }
        if config.cluster_size.is_some_and(|old| old != cluster_size)
            || event && config.event_communicators.contains(&(comm as usize))
        {
            return Err(Error::out_of_range("group collective configuration"));
        }
        config.cluster_size = Some(cluster_size);
        if event {
            config.event_communicators.push(comm as usize);
        }
        Ok(())
    })
}

/// Whether this thread has an unfinished NCCL group.
pub(crate) fn is_active() -> bool {
    CONFIGURATION.with_borrow(|config| config.depth != 0)
}
