use std::{
    ffi::{c_int, c_uint, c_void},
    slice,
    task::Poll,
    time::Instant,
};

use singe_cuda_sys::{driver::size_t, runtime::cudaStream_t};
use singe_nccl_sys::{
    ncclComm_t, ncclDataType_t, ncclResult_t, ncclWaitSignalDesc_t, ncclWindow_t,
};
use singe_ptx::emulator::ExecutionError;

use crate::nccl::{
    Caller, caller, clique_key,
    collective::{Clique, RENDEZVOUS_TIMEOUT},
    gin, invalid, lock, outcome,
    reduce::DataType,
    submit, symmetric,
};

/// Returns the calling rank and the peer it signals, after checking the peer, signal, and context.
fn sender(
    comm: ncclComm_t,
    peer: c_int,
    signal: c_int,
    context: c_int,
) -> Result<(Caller, usize), ncclResult_t> {
    let sender = caller(comm)?;
    let peer = usize::try_from(peer).map_err(invalid)?;
    if peer >= sender.clique.size || signal < 0 || context < 0 {
        return Err(ncclResult_t::ncclInvalidArgument);
    }
    Ok((sender, peer))
}

fn raise(clique: &Clique, to: usize, from: usize, signal: c_int, context: c_int) {
    *lock(&clique.signals)
        .entry((to, from, signal, context))
        .or_default() += 1;
}

pub(crate) unsafe extern "C" fn put_signal(
    source: *const c_void,
    count: size_t,
    data_type: ncclDataType_t,
    peer: c_int,
    window: ncclWindow_t,
    offset: size_t,
    signal: c_int,
    context: c_int,
    flags: c_uint,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t {
    outcome(|| {
        let (sender, peer) = sender(comm, peer, signal, context)?;
        let element_size = DataType::try_from(data_type)?.size();
        let byte_len = usize::try_from(count)
            .map_err(invalid)?
            .checked_mul(element_size)
            .ok_or(ncclResult_t::ncclInvalidArgument)?;
        let (key, sequence, _) = symmetric::window_identity(window)?;
        if flags != 0 || key != clique_key(&sender.clique) {
            return Err(ncclResult_t::ncclInvalidArgument);
        }

        let source = source.addr() as u64;
        let Caller {
            clique,
            rank,
            device,
        } = sender;
        submit(stream, device, move |memory| {
            let destination = gin::resolve(key, sequence, peer, offset, byte_len)?;
            memory.copy(source, destination, byte_len)?;
            raise(&clique, peer, rank, signal, context);
            Poll::Ready(Ok(()))
        })
    })
}

pub(crate) unsafe extern "C" fn signal(
    peer: c_int,
    signal: c_int,
    context: c_int,
    flags: c_uint,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t {
    outcome(|| {
        let (sender, peer) = sender(comm, peer, signal, context)?;
        if flags != 0 {
            return Err(ncclResult_t::ncclInvalidArgument);
        }

        let Caller {
            clique,
            rank,
            device,
        } = sender;
        submit(stream, device, move |_| {
            raise(&clique, peer, rank, signal, context);
            Poll::Ready(Ok(()))
        })
    })
}

pub(crate) unsafe extern "C" fn wait_signal(
    count: c_int,
    descriptors: *mut ncclWaitSignalDesc_t,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t {
    outcome(|| {
        let count = usize::try_from(count).map_err(invalid)?;
        if count > 0 && descriptors.is_null() {
            return Err(ncclResult_t::ncclInvalidArgument);
        }

        let descriptors = if count == 0 {
            Vec::new()
        } else {
            // SAFETY: the caller passes `count` descriptors.
            unsafe { slice::from_raw_parts(descriptors, count) }.to_vec()
        };
        let receiver = caller(comm)?;
        let mut waits = Vec::with_capacity(count);
        for descriptor in descriptors {
            let peer = usize::try_from(descriptor.peer).map_err(invalid)?;
            let expected = u64::try_from(descriptor.opCnt).map_err(invalid)?;
            if peer >= receiver.clique.size
                || expected == 0
                || descriptor.sigIdx < 0
                || descriptor.ctx < 0
            {
                return Err(ncclResult_t::ncclInvalidArgument);
            }

            waits.push((
                (receiver.rank, peer, descriptor.sigIdx, descriptor.ctx),
                expected,
            ));
        }

        let Caller {
            clique,
            rank,
            device,
        } = receiver;
        let mut since: Option<Instant> = None;
        submit(stream, device, move |memory| {
            let mut signals = lock(&clique.signals);
            let arrived =
                |key: &(usize, usize, c_int, c_int)| signals.get(key).copied().unwrap_or(0);
            if waits
                .iter()
                .all(|(key, expected)| arrived(key) >= *expected)
            {
                for (key, expected) in &waits {
                    if let Some(received) = signals.get_mut(key) {
                        *received -= expected;
                    }
                }
                return Poll::Ready(Ok(()));
            }

            let faulted = waits
                .iter()
                .filter(|(key, expected)| arrived(key) < *expected)
                .map(|((_, from, _, _), _)| *from)
                .find(|&from| {
                    clique
                        .device_of(from)
                        .is_some_and(|device| memory.is_faulted(device))
                });
            if let Some(from) = faulted {
                return Poll::Ready(Err(ExecutionError::Invalid(format!(
                    "rank {rank} waits for signals from rank {from}, whose device faulted"
                ))));
            }

            if since.get_or_insert_with(Instant::now).elapsed() > RENDEZVOUS_TIMEOUT {
                let missing: Vec<String> = waits
                    .iter()
                    .map(|(key, expected)| {
                        let (_, from, signal, context) = key;
                        format!(
                            "{} of {expected} from rank {from} on signal {signal}, context {context}",
                            arrived(key)
                        )
                    })
                    .collect();
                return Poll::Ready(Err(ExecutionError::Invalid(format!(
                    "rank {rank} waited {} s for signals and has {}",
                    RENDEZVOUS_TIMEOUT.as_secs(),
                    missing.join("; ")
                ))));
            }
            Poll::Pending
        })
    })
}
