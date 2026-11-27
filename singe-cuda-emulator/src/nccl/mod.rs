mod collective;
mod gin;
mod one_sided;
mod reduce;
mod symmetric;

use std::{
    cell::RefCell,
    collections::HashMap,
    ffi::{CStr, CString, c_char, c_int, c_void},
    mem, ptr, slice,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    task::Poll,
};

use singe_cuda_sys::{driver::size_t, runtime::cudaStream_t};
use singe_nccl_sys::{
    ncclCollConfig_t, ncclComm_t, ncclConfig_t, ncclDataType_t, ncclRedOp_t, ncclResult_t,
    ncclScalarResidence_t, ncclUniqueId,
};
use singe_ptx::emulator::{ExecutionError, HostMemory};

use crate::{
    last_fault,
    machine::{Machine, index, machine, provide_symbols, store},
    nccl::{
        collective::{Clique, Contribution, Key, Kind, Part, Split, lock, poll_parts},
        reduce::{DataType, Reduction, Scalar},
        symmetric::Symmetric,
    },
};

/// The NCCL version the emulator reports, 2.32.3.
const VERSION: c_int = 23203;

type Outcome = Result<(), ncclResult_t>;

struct Communicator {
    clique: Arc<Clique>,
    rank: usize,
    device: usize,
    /// A split this communicator came from, until the other ranks call it and its clique is known.
    pending: Option<(Arc<Split>, usize)>,
    collectives: u64,
    splits: u64,
    /// Window registrations, including the resource windows of device communicators, numbered alike on every rank.
    windows: u64,
    sends: HashMap<usize, u64>,
    receives: HashMap<usize, u64>,
    /// Reductions from `ncclRedOpCreatePreMulSum`, by operation number.
    reductions: HashMap<u32, Reduction>,
    next_reduction: u32,
}

#[derive(Default)]
struct Nccl {
    communicators: HashMap<usize, Communicator>,
    cliques: HashMap<ncclUniqueId, Arc<Clique>>,
    symmetric: Symmetric,
}

static NCCL: Mutex<Option<Nccl>> = Mutex::new(None);

fn with_nccl<T>(
    body: impl FnOnce(&mut Nccl) -> Result<T, ncclResult_t>,
) -> Result<T, ncclResult_t> {
    body(lock(&NCCL).get_or_insert_with(Nccl::default))
}

/// Operations issued inside a group on this thread, submitted together when the outermost group ends.
#[derive(Default)]
struct Group {
    depth: usize,
    pending: Vec<(cudaStream_t, usize, Part)>,
}

thread_local! {
    static GROUP: RefCell<Group> = RefCell::new(Group::default());
    /// The message `ncclGetLastError` last returned on this thread, kept alive until its next call.
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn outcome(body: impl FnOnce() -> Outcome) -> ncclResult_t {
    match body() {
        Ok(()) => ncclResult_t::ncclSuccess,
        Err(error) => error,
    }
}

fn emulated() -> Result<&'static Machine, ncclResult_t> {
    machine().map_err(|_| ncclResult_t::ncclSystemError)
}

fn invalid<T>(_: T) -> ncclResult_t {
    ncclResult_t::ncclInvalidArgument
}

fn cuda<T>(_: T) -> ncclResult_t {
    ncclResult_t::ncclUnhandledCudaError
}

/// Returns the key that identifies a clique to the GIN proxy and to window handles.
fn clique_key(clique: &Arc<Clique>) -> usize {
    Arc::as_ptr(clique).addr()
}

/// The rank a call runs on.
struct Caller {
    clique: Arc<Clique>,
    rank: usize,
    device: usize,
}

fn caller(comm: ncclComm_t) -> Result<Caller, ncclResult_t> {
    with_communicator(comm, |communicator| {
        Ok(Caller {
            clique: Arc::clone(&communicator.clique),
            rank: communicator.rank,
            device: communicator.device,
        })
    })
}

pub(crate) fn provide(name: &CStr) -> Option<*const ()> {
    provide_symbols!(
        name,
        ncclGetVersion => get_version,
        ncclGetUniqueId => get_unique_id,
        ncclGetErrorString => error_string,
        ncclGetLastError => last_error,
        ncclCommInitRank => init_rank,
        ncclCommInitRankConfig => init_rank_config,
        ncclCommInitAll => init_all,
        ncclCommDestroy => destroy,
        ncclCommAbort => destroy,
        ncclCommFinalize => finalize,
        ncclCommGetAsyncError => async_error,
        ncclCommCount => count,
        ncclCommCuDevice => device,
        ncclCommUserRank => user_rank,
        ncclGroupStart => group_start,
        ncclGroupEnd => group_end,
        ncclCommSplit => split,
        ncclCommRegister => register_buffer,
        ncclCommDeregister => deregister_buffer,
        ncclMemAlloc => symmetric::memory_allocate,
        ncclMemFree => symmetric::memory_free,
        ncclCommWindowRegister => symmetric::window_register,
        ncclCommWindowDeregister => symmetric::window_deregister,
        ncclDevCommCreate => symmetric::device_communicator_create,
        ncclDevCommDestroy => symmetric::device_communicator_destroy,
        ncclLsaBarrierCreateRequirement => symmetric::lsa_barrier_requirement,
        ncclGinBarrierCreateRequirement => symmetric::gin_barrier_requirement,
        ncclWinGetUserPtr => symmetric::window_user_pointer,
        ncclPutSignal => one_sided::put_signal,
        ncclSignal => one_sided::signal,
        ncclWaitSignal => one_sided::wait_signal,
        ncclCommQueryProperties => symmetric::query_properties,
        ncclTeamWorld => symmetric::team_world,
        ncclTeamLsa => symmetric::team_lsa,
        ncclTeamRail => symmetric::team_rail,
        ncclTeamRankToWorld => symmetric::team_rank_to_world,
        ncclTeamRankToLsa => symmetric::team_rank_to_lsa,
        ncclGetLsaDevicePointer => symmetric::lsa_pointer,
        ncclGetPeerDevicePointer => symmetric::peer_pointer,
        ncclGetLsaMultimemDevicePointer => symmetric::lsa_multimem_pointer,
        ncclGetMultimemDevicePointer => symmetric::multimem_pointer,
        ncclRedOpCreatePreMulSum => create_scaled_sum,
        ncclRedOpDestroy => destroy_reduction,
        ncclAllReduce => all_reduce,
        ncclAllReduceConfig => all_reduce_config,
        ncclBroadcast => broadcast,
        ncclBroadcastConfig => broadcast_config,
        ncclReduce => reduce,
        ncclReduceConfig => reduce_config,
        ncclAllGather => all_gather,
        ncclAllGatherConfig => all_gather_config,
        ncclReduceScatter => reduce_scatter,
        ncclReduceScatterConfig => reduce_scatter_config,
        ncclAlltoAll => all_to_all,
        ncclAlltoAllConfig => all_to_all_config,
        ncclGather => gather,
        ncclGatherConfig => gather_config,
        ncclScatter => scatter,
        ncclScatterConfig => scatter_config,
        ncclSend => send,
        ncclRecv => receive,
    )
}

unsafe extern "C" fn get_version(version: *mut c_int) -> ncclResult_t {
    // SAFETY: `version` is the caller's output slot.
    outcome(|| unsafe { store(version, VERSION) }.map_err(invalid))
}

unsafe extern "C" fn get_unique_id(id: *mut ncclUniqueId) -> ncclResult_t {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    outcome(|| {
        let mut unique = ncclUniqueId::default();
        let tag = b"singe-cuda-emulator";
        for (slot, byte) in unique.internal.iter_mut().zip(tag) {
            *slot = c_char::from_ne_bytes([*byte]);
        }

        let number = NEXT.fetch_add(1, Ordering::Relaxed).to_le_bytes();
        for (slot, byte) in unique.internal[tag.len()..].iter_mut().zip(number) {
            *slot = c_char::from_ne_bytes([byte]);
        }
        // SAFETY: `id` is the caller's output slot.
        unsafe { store(id, unique) }.map_err(invalid)
    })
}

unsafe extern "C" fn error_string(result: ncclResult_t) -> *const c_char {
    match result {
        ncclResult_t::ncclSuccess => c"no error",
        ncclResult_t::ncclUnhandledCudaError => c"unhandled cuda error",
        ncclResult_t::ncclSystemError => c"unhandled system error",
        ncclResult_t::ncclInternalError => c"internal error",
        ncclResult_t::ncclInvalidArgument => c"invalid argument",
        ncclResult_t::ncclInvalidUsage => c"invalid usage",
        ncclResult_t::ncclRemoteError => c"remote process exited or there was a network error",
        ncclResult_t::ncclInProgress => c"NCCL operation in progress",
        _ => c"unknown result code",
    }
    .as_ptr()
}

unsafe extern "C" fn last_error(comm: ncclComm_t) -> *const c_char {
    let fault = caller(comm).map_or_else(
        |_| last_fault(),
        |caller| emulated().ok()?.device(caller.device).ok()?.fault(),
    );
    let message = fault.map_or_else(String::new, |fault| fault.to_string());
    let message = CString::new(message.replace('\0', "")).unwrap_or_default();
    LAST_ERROR.with_borrow_mut(|last| {
        *last = message;
        last.as_ptr()
    })
}

fn join(clique: Arc<Clique>, rank: usize, device: usize) -> Result<usize, ncclResult_t> {
    clique.set_device(rank, device);
    add_communicator(clique, rank, device, None)
}

fn add_communicator(
    clique: Arc<Clique>,
    rank: usize,
    device: usize,
    pending: Option<(Arc<Split>, usize)>,
) -> Result<usize, ncclResult_t> {
    let handle = emulated()?.state().handle();
    with_nccl(|nccl| {
        nccl.communicators.insert(
            handle,
            Communicator {
                clique,
                rank,
                device,
                pending,
                collectives: 0,
                splits: 0,
                windows: 0,
                sends: HashMap::new(),
                receives: HashMap::new(),
                reductions: HashMap::new(),
                next_reduction: ncclRedOp_t::ncclNumOps.0,
            },
        );
        Ok(())
    })?;
    Ok(handle)
}

fn init(comm: *mut ncclComm_t, ranks: c_int, id: ncclUniqueId, rank: c_int) -> Outcome {
    let machine = emulated()?;
    let ranks = usize::try_from(ranks).map_err(invalid)?;
    let rank = usize::try_from(rank).map_err(invalid)?;
    if ranks == 0 || rank >= ranks {
        return Err(ncclResult_t::ncclInvalidArgument);
    }

    let device = machine.runtime_device().map_err(cuda)?;
    let clique = with_nccl(|nccl| {
        let clique = nccl
            .cliques
            .entry(id)
            .or_insert_with(|| Arc::new(Clique::new(ranks)));
        if clique.size == ranks {
            Ok(Arc::clone(clique))
        } else {
            Err(ncclResult_t::ncclInvalidUsage)
        }
    })?;
    let handle = join(clique, rank, device)?;
    // SAFETY: `comm` is the caller's output slot.
    unsafe { store(comm, ptr::without_provenance_mut(handle)) }.map_err(invalid)
}

unsafe extern "C" fn init_rank(
    comm: *mut ncclComm_t,
    ranks: c_int,
    id: ncclUniqueId,
    rank: c_int,
) -> ncclResult_t {
    outcome(|| init(comm, ranks, id, rank))
}

unsafe extern "C" fn init_rank_config(
    comm: *mut ncclComm_t,
    ranks: c_int,
    id: ncclUniqueId,
    rank: c_int,
    _config: *mut ncclConfig_t,
) -> ncclResult_t {
    outcome(|| init(comm, ranks, id, rank))
}

unsafe extern "C" fn init_all(
    comms: *mut ncclComm_t,
    count: c_int,
    devices: *const c_int,
) -> ncclResult_t {
    outcome(|| {
        let machine = emulated()?;
        let count = usize::try_from(count).map_err(invalid)?;
        if comms.is_null() || count == 0 {
            return Err(ncclResult_t::ncclInvalidArgument);
        }

        let clique = Arc::new(Clique::new(count));
        for rank in 0..count {
            let device = if devices.is_null() {
                rank
            } else {
                // SAFETY: the caller passes `count` device ordinals.
                index(unsafe { *devices.add(rank) }).map_err(invalid)?
            };
            machine.device(device).map_err(invalid)?;
            let handle = join(Arc::clone(&clique), rank, device)?;
            // SAFETY: the caller passes `count` output slots.
            unsafe { comms.add(rank).write(ptr::without_provenance_mut(handle)) };
        }
        Ok(())
    })
}

unsafe extern "C" fn destroy(comm: ncclComm_t) -> ncclResult_t {
    outcome(|| {
        with_nccl(|nccl| {
            nccl.communicators
                .remove(&comm.addr())
                .map(|_| ())
                .ok_or(ncclResult_t::ncclInvalidArgument)
        })
    })
}

unsafe extern "C" fn finalize(comm: ncclComm_t) -> ncclResult_t {
    outcome(|| with_communicator(comm, |_| Ok(())))
}

fn with_communicator<T>(
    comm: ncclComm_t,
    body: impl FnOnce(&mut Communicator) -> Result<T, ncclResult_t>,
) -> Result<T, ncclResult_t> {
    fn communicator(nccl: &mut Nccl, comm: ncclComm_t) -> Result<&mut Communicator, ncclResult_t> {
        nccl.communicators
            .get_mut(&comm.addr())
            .ok_or(ncclResult_t::ncclInvalidArgument)
    }
    let pending = with_nccl(|nccl| Ok(communicator(nccl, comm)?.pending.clone()))?;
    if let Some((split, rank)) = pending {
        let (clique, rank) = split.resolve(rank)?;
        with_nccl(|nccl| {
            let resolved = communicator(nccl, comm)?;
            resolved.clique = clique;
            resolved.rank = rank;
            resolved.pending = None;
            Ok(())
        })?;
    }
    with_nccl(|nccl| body(communicator(nccl, comm)?))
}

unsafe extern "C" fn split(
    comm: ncclComm_t,
    color: c_int,
    key: c_int,
    child: *mut ncclComm_t,
    _config: *mut ncclConfig_t,
) -> ncclResult_t {
    outcome(|| {
        let (split, rank, device) = with_communicator(comm, |communicator| {
            communicator.splits += 1;
            let split = Arc::clone(
                lock(&communicator.clique.splits)
                    .entry(communicator.splits)
                    .or_insert_with(|| Arc::new(Split::new(communicator.clique.size))),
            );
            Ok((split, communicator.rank, communicator.device))
        })?;
        split.arrive(rank, color, key, device);
        let handle = if color < 0 {
            0
        } else {
            add_communicator(Arc::new(Clique::new(1)), 0, device, Some((split, rank)))?
        };
        // SAFETY: `child` is the caller's output slot.
        unsafe { store(child, ptr::without_provenance_mut(handle)) }.map_err(invalid)
    })
}

unsafe extern "C" fn register_buffer(
    comm: ncclComm_t,
    _buffer: *mut c_void,
    _size: size_t,
    handle: *mut *mut c_void,
) -> ncclResult_t {
    outcome(|| {
        with_communicator(comm, |_| Ok(()))?;
        // SAFETY: `handle` is the caller's output slot; buffers need no registration on the emulator.
        unsafe { store(handle, ptr::without_provenance_mut(1)) }.map_err(invalid)
    })
}

unsafe extern "C" fn deregister_buffer(comm: ncclComm_t, _handle: *mut c_void) -> ncclResult_t {
    outcome(|| with_communicator(comm, |_| Ok(())))
}

unsafe extern "C" fn async_error(comm: ncclComm_t, error: *mut ncclResult_t) -> ncclResult_t {
    outcome(|| {
        let device = caller(comm)?.device;
        let is_faulted = emulated()?
            .device(device)
            .map_err(invalid)?
            .fault()
            .is_some();
        let value = if is_faulted {
            ncclResult_t::ncclInvalidUsage
        } else {
            ncclResult_t::ncclSuccess
        };
        // SAFETY: `error` is the caller's output slot.
        unsafe { store(error, value) }.map_err(invalid)
    })
}

unsafe extern "C" fn count(comm: ncclComm_t, value: *mut c_int) -> ncclResult_t {
    outcome(|| {
        let size = with_communicator(comm, |communicator| Ok(communicator.clique.size))?;
        // SAFETY: `value` is the caller's output slot.
        unsafe { store(value, c_int::try_from(size).map_err(invalid)?) }.map_err(invalid)
    })
}

unsafe extern "C" fn device(comm: ncclComm_t, value: *mut c_int) -> ncclResult_t {
    outcome(|| {
        let device = caller(comm)?.device;
        // SAFETY: `value` is the caller's output slot.
        unsafe { store(value, c_int::try_from(device).map_err(invalid)?) }.map_err(invalid)
    })
}

unsafe extern "C" fn user_rank(comm: ncclComm_t, value: *mut c_int) -> ncclResult_t {
    outcome(|| {
        let rank = caller(comm)?.rank;
        // SAFETY: `value` is the caller's output slot.
        unsafe { store(value, c_int::try_from(rank).map_err(invalid)?) }.map_err(invalid)
    })
}

unsafe extern "C" fn group_start() -> ncclResult_t {
    GROUP.with_borrow_mut(|group| group.depth += 1);
    ncclResult_t::ncclSuccess
}

unsafe extern "C" fn group_end() -> ncclResult_t {
    outcome(|| {
        let pending = GROUP.with_borrow_mut(|group| {
            group.depth = group
                .depth
                .checked_sub(1)
                .ok_or(ncclResult_t::ncclInvalidUsage)?;
            Ok(if group.depth == 0 {
                mem::take(&mut group.pending)
            } else {
                Vec::new()
            })
        })?;
        let mut streams: Vec<(cudaStream_t, usize, Vec<Part>)> = Vec::new();
        for (stream, device, part) in pending {
            match streams
                .iter_mut()
                .find(|(other, other_device, _)| *other == stream && *other_device == device)
            {
                Some((_, _, parts)) => parts.push(part),
                None => streams.push((stream, device, vec![part])),
            }
        }

        for (stream, device, parts) in streams {
            submit_parts(stream, device, parts)?;
        }
        Ok(())
    })
}

/// Queues `task` on `stream` of `device`.
fn submit(
    stream: cudaStream_t,
    device: usize,
    task: impl FnMut(&mut HostMemory<'_>) -> Poll<Result<(), ExecutionError>> + Send + 'static,
) -> Outcome {
    let stream = emulated()?.stream(stream, device).map_err(invalid)?;
    drop(stream.host_task(task).map_err(cuda)?);
    Ok(())
}

/// Queues `parts` on a stream as one host task that finishes when all of them ran.
fn submit_parts(stream: cudaStream_t, device: usize, mut parts: Vec<Part>) -> Outcome {
    submit(stream, device, move |memory| poll_parts(&mut parts, memory))
}

/// Issues one rank's part of an operation, now or when the enclosing group ends.
fn issue(
    comm: ncclComm_t,
    stream: cudaStream_t,
    peer: Option<usize>,
    contribution: Contribution,
) -> Outcome {
    let (part, device) = with_communicator(comm, |communicator| {
        let size = communicator.clique.size;
        if peer.is_some_and(|peer| peer >= size)
            || contribution.kind.root().is_some_and(|root| root >= size)
        {
            return Err(ncclResult_t::ncclInvalidArgument);
        }

        let (key, party) = match (&contribution.kind, peer) {
            (Kind::Send, Some(peer)) => {
                let sequence = communicator.sends.entry(peer).or_default();
                *sequence += 1;
                (
                    Key::Transfer {
                        from: communicator.rank,
                        to: peer,
                        sequence: *sequence,
                    },
                    0,
                )
            }
            (Kind::Receive, Some(peer)) => {
                let sequence = communicator.receives.entry(peer).or_default();
                *sequence += 1;
                (
                    Key::Transfer {
                        from: peer,
                        to: communicator.rank,
                        sequence: *sequence,
                    },
                    1,
                )
            }
            _ => {
                communicator.collectives += 1;
                (Key::Collective(communicator.collectives), communicator.rank)
            }
        };
        Ok((
            Part::new(&communicator.clique, key, party, contribution),
            communicator.device,
        ))
    })?;
    let ungrouped = GROUP.with_borrow_mut(|group| {
        if group.depth > 0 {
            group.pending.push((stream, device, part));
            None
        } else {
            Some(part)
        }
    });
    ungrouped.map_or(Ok(()), |part| submit_parts(stream, device, vec![part]))
}

/// Resolves a reduction, including the scaled sums a communicator created.
fn reduction(comm: ncclComm_t, op: ncclRedOp_t) -> Result<Reduction, ncclResult_t> {
    Reduction::try_from(op).or_else(|_| {
        with_communicator(comm, |communicator| {
            communicator
                .reductions
                .get(&op.0)
                .cloned()
                .ok_or(ncclResult_t::ncclInvalidArgument)
        })
    })
}

fn collective(
    send: *const c_void,
    receive: *mut c_void,
    count: size_t,
    data_type: ncclDataType_t,
    kind: Kind,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> Outcome {
    issue(
        comm,
        stream,
        None,
        Contribution {
            kind,
            data_type: DataType::try_from(data_type)?,
            count: usize::try_from(count).map_err(invalid)?,
            send: send.addr() as u64,
            receive: receive.addr() as u64,
        },
    )
}

unsafe extern "C" fn all_reduce(
    send: *const c_void,
    receive: *mut c_void,
    count: size_t,
    data_type: ncclDataType_t,
    op: ncclRedOp_t,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t {
    outcome(|| {
        let kind = Kind::AllReduce(reduction(comm, op)?);
        collective(send, receive, count, data_type, kind, comm, stream)
    })
}

unsafe extern "C" fn all_reduce_config(
    send: *const c_void,
    receive: *mut c_void,
    count: size_t,
    data_type: ncclDataType_t,
    op: ncclRedOp_t,
    comm: ncclComm_t,
    stream: cudaStream_t,
    _config: *const ncclCollConfig_t,
) -> ncclResult_t {
    // SAFETY: the arguments meet `ncclAllReduce`'s requirements, and the configuration only tunes performance.
    unsafe { all_reduce(send, receive, count, data_type, op, comm, stream) }
}

unsafe extern "C" fn broadcast(
    send: *const c_void,
    receive: *mut c_void,
    count: size_t,
    data_type: ncclDataType_t,
    root: c_int,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t {
    outcome(|| {
        let root = usize::try_from(root).map_err(invalid)?;
        let kind = Kind::Broadcast { root };
        collective(send, receive, count, data_type, kind, comm, stream)
    })
}

unsafe extern "C" fn broadcast_config(
    send: *const c_void,
    receive: *mut c_void,
    count: size_t,
    data_type: ncclDataType_t,
    root: c_int,
    comm: ncclComm_t,
    stream: cudaStream_t,
    _config: *const ncclCollConfig_t,
) -> ncclResult_t {
    // SAFETY: the arguments meet `ncclBroadcast`'s requirements, and the configuration only tunes performance.
    unsafe { broadcast(send, receive, count, data_type, root, comm, stream) }
}

unsafe extern "C" fn reduce(
    send: *const c_void,
    receive: *mut c_void,
    count: size_t,
    data_type: ncclDataType_t,
    op: ncclRedOp_t,
    root: c_int,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t {
    outcome(|| {
        let root = usize::try_from(root).map_err(invalid)?;
        let kind = Kind::Reduce {
            reduction: reduction(comm, op)?,
            root,
        };
        collective(send, receive, count, data_type, kind, comm, stream)
    })
}

unsafe extern "C" fn reduce_config(
    send: *const c_void,
    receive: *mut c_void,
    count: size_t,
    data_type: ncclDataType_t,
    op: ncclRedOp_t,
    root: c_int,
    comm: ncclComm_t,
    stream: cudaStream_t,
    _config: *const ncclCollConfig_t,
) -> ncclResult_t {
    // SAFETY: the arguments meet `ncclReduce`'s requirements, and the configuration only tunes performance.
    unsafe { reduce(send, receive, count, data_type, op, root, comm, stream) }
}

unsafe extern "C" fn all_gather(
    send: *const c_void,
    receive: *mut c_void,
    count: size_t,
    data_type: ncclDataType_t,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t {
    outcome(|| {
        collective(
            send,
            receive,
            count,
            data_type,
            Kind::AllGather,
            comm,
            stream,
        )
    })
}

unsafe extern "C" fn all_gather_config(
    send: *const c_void,
    receive: *mut c_void,
    count: size_t,
    data_type: ncclDataType_t,
    comm: ncclComm_t,
    stream: cudaStream_t,
    _config: *const ncclCollConfig_t,
) -> ncclResult_t {
    // SAFETY: the arguments meet `ncclAllGather`'s requirements, and the configuration only tunes performance.
    unsafe { all_gather(send, receive, count, data_type, comm, stream) }
}

unsafe extern "C" fn reduce_scatter(
    send: *const c_void,
    receive: *mut c_void,
    count: size_t,
    data_type: ncclDataType_t,
    op: ncclRedOp_t,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t {
    outcome(|| {
        let kind = Kind::ReduceScatter(reduction(comm, op)?);
        collective(send, receive, count, data_type, kind, comm, stream)
    })
}

unsafe extern "C" fn reduce_scatter_config(
    send: *const c_void,
    receive: *mut c_void,
    count: size_t,
    data_type: ncclDataType_t,
    op: ncclRedOp_t,
    comm: ncclComm_t,
    stream: cudaStream_t,
    _config: *const ncclCollConfig_t,
) -> ncclResult_t {
    // SAFETY: the arguments meet `ncclReduceScatter`'s requirements, and the configuration only tunes performance.
    unsafe { reduce_scatter(send, receive, count, data_type, op, comm, stream) }
}

unsafe extern "C" fn all_to_all(
    send: *const c_void,
    receive: *mut c_void,
    count: size_t,
    data_type: ncclDataType_t,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t {
    outcome(|| {
        collective(
            send,
            receive,
            count,
            data_type,
            Kind::AllToAll,
            comm,
            stream,
        )
    })
}

unsafe extern "C" fn all_to_all_config(
    send: *const c_void,
    receive: *mut c_void,
    count: size_t,
    data_type: ncclDataType_t,
    comm: ncclComm_t,
    stream: cudaStream_t,
    _config: *const ncclCollConfig_t,
) -> ncclResult_t {
    // SAFETY: the arguments meet `ncclAlltoAll`'s requirements, and the configuration only tunes performance.
    unsafe { all_to_all(send, receive, count, data_type, comm, stream) }
}

unsafe extern "C" fn gather(
    send: *const c_void,
    receive: *mut c_void,
    count: size_t,
    data_type: ncclDataType_t,
    root: c_int,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t {
    outcome(|| {
        let root = usize::try_from(root).map_err(invalid)?;
        let kind = Kind::Gather { root };
        collective(send, receive, count, data_type, kind, comm, stream)
    })
}

unsafe extern "C" fn gather_config(
    send: *const c_void,
    receive: *mut c_void,
    count: size_t,
    data_type: ncclDataType_t,
    root: c_int,
    comm: ncclComm_t,
    stream: cudaStream_t,
    _config: *const ncclCollConfig_t,
) -> ncclResult_t {
    // SAFETY: the arguments meet `ncclGather`'s requirements, and the configuration only tunes performance.
    unsafe { gather(send, receive, count, data_type, root, comm, stream) }
}

unsafe extern "C" fn scatter(
    send: *const c_void,
    receive: *mut c_void,
    count: size_t,
    data_type: ncclDataType_t,
    root: c_int,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t {
    outcome(|| {
        let root = usize::try_from(root).map_err(invalid)?;
        let kind = Kind::Scatter { root };
        collective(send, receive, count, data_type, kind, comm, stream)
    })
}

unsafe extern "C" fn scatter_config(
    send: *const c_void,
    receive: *mut c_void,
    count: size_t,
    data_type: ncclDataType_t,
    root: c_int,
    comm: ncclComm_t,
    stream: cudaStream_t,
    _config: *const ncclCollConfig_t,
) -> ncclResult_t {
    // SAFETY: the arguments meet `ncclScatter`'s requirements, and the configuration only tunes performance.
    unsafe { scatter(send, receive, count, data_type, root, comm, stream) }
}

unsafe extern "C" fn send(
    buffer: *const c_void,
    count: size_t,
    data_type: ncclDataType_t,
    peer: c_int,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t {
    outcome(|| {
        let peer = usize::try_from(peer).map_err(invalid)?;
        issue(
            comm,
            stream,
            Some(peer),
            Contribution {
                kind: Kind::Send,
                data_type: DataType::try_from(data_type)?,
                count: usize::try_from(count).map_err(invalid)?,
                send: buffer.addr() as u64,
                receive: 0,
            },
        )
    })
}

unsafe extern "C" fn receive(
    buffer: *mut c_void,
    count: size_t,
    data_type: ncclDataType_t,
    peer: c_int,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t {
    outcome(|| {
        let peer = usize::try_from(peer).map_err(invalid)?;
        issue(
            comm,
            stream,
            Some(peer),
            Contribution {
                kind: Kind::Receive,
                data_type: DataType::try_from(data_type)?,
                count: usize::try_from(count).map_err(invalid)?,
                send: 0,
                receive: buffer.addr() as u64,
            },
        )
    })
}

unsafe extern "C" fn create_scaled_sum(
    op: *mut ncclRedOp_t,
    scalar: *mut c_void,
    data_type: ncclDataType_t,
    residence: ncclScalarResidence_t,
    comm: ncclComm_t,
) -> ncclResult_t {
    outcome(|| {
        let width = DataType::try_from(data_type)?.size();
        if scalar.is_null() {
            return Err(ncclResult_t::ncclInvalidArgument);
        }

        let scalar = if residence == ncclScalarResidence_t::ncclScalarHostImmediate {
            // SAFETY: a host-immediate scalar is one element the caller passes by pointer.
            Scalar::Host(unsafe { slice::from_raw_parts(scalar.cast::<u8>(), width) }.to_vec())
        } else {
            Scalar::Device(scalar.addr() as u64)
        };
        let number = with_communicator(comm, |communicator| {
            let number = communicator.next_reduction;
            communicator.next_reduction += 1;
            communicator
                .reductions
                .insert(number, Reduction::ScaledSum(scalar));
            Ok(number)
        })?;
        // SAFETY: `op` is the caller's output slot.
        unsafe { store(op, ncclRedOp_t(number)) }.map_err(invalid)
    })
}

unsafe extern "C" fn destroy_reduction(op: ncclRedOp_t, comm: ncclComm_t) -> ncclResult_t {
    outcome(|| {
        with_communicator(comm, |communicator| {
            communicator
                .reductions
                .remove(&op.0)
                .map(|_| ())
                .ok_or(ncclResult_t::ncclInvalidArgument)
        })
    })
}
