use std::{
    collections::{BTreeMap, HashMap, btree_map::Entry},
    ffi::{c_int, c_void},
    mem::offset_of,
    ptr,
    sync::Arc,
};

use singe_cuda_sys::driver::size_t;
use singe_nccl_sys::{
    NCCL_API_MAGIC, NCCL_VERSION_CODE, NCCL_WIN_COLL_SYMMETRIC, ncclComm_t, ncclCommProperties_t,
    ncclDevComm_t, ncclDevCommRequirements_t, ncclDevResourceRequirements_t, ncclGinBarrierHandle,
    ncclGinConnectionType_t, ncclGinType_t, ncclLsaBarrierHandle, ncclMultimemHandle,
    ncclMultimemHandle_t, ncclNetDeviceType, ncclResourceWindow_vidmem, ncclResult_t, ncclTeam_t,
    ncclWindow_t, ncclWindow_vidmem,
};

use crate::{
    driver::GRANULARITY,
    machine::{Machine, store},
    nccl::{
        Caller, Outcome, caller, clique_key,
        collective::Clique,
        cuda, emulated,
        gin::{self, Layout, Placement},
        invalid, lock, outcome, with_communicator, with_nccl,
    },
};

/// The distance between LSA ranks' slots in a window's flat range, which NCCL's device code counts in units of 4 GiB.
const STRIDE: u64 = 1 << 32;

/// The size of each node's multicast object; window `n` takes the slot of `STRIDE` bytes at `n * STRIDE`.
const MULTICAST_BYTES: u64 = 1 << 44;

/// Bytes per resource-buffer handle unit in NCCL's device code.
const RESOURCE_UNIT: u64 = 128;

/// Memory from `ncclMemAlloc`: physical memory mapped at the address it is keyed by.
struct Allocation {
    device: usize,
    physical: u64,
    size: u64,
    windows: usize,
}

/// A registered window on one rank.
struct Window {
    device: usize,
    buffer: u64,
    clique: Arc<Clique>,
    sequence: u64,
    node: usize,
    base: u64,
    slot: u64,
    size: u64,
    lsa_rank: usize,
    lsa_size: usize,
    world_rank: usize,
    multicast: Multicast,
    allocation: u64,
}

#[derive(Clone, Copy)]
struct Multicast {
    handle: u64,
    base: u64,
}

struct Space {
    base: u64,
    users: usize,
}

/// The flat ranges and multicast objects of a clique's windows.
#[derive(Default)]
pub(crate) struct Spaces {
    windows: HashMap<(u64, usize), Space>,
    multicasts: HashMap<usize, Multicast>,
}

/// Symmetric memory, windows, and device communicators across every communicator.
#[derive(Default)]
pub(crate) struct Symmetric {
    allocations: BTreeMap<u64, Allocation>,
    windows: HashMap<u64, Window>,
    /// The memory behind each device communicator's resource window, by window.
    resources: HashMap<u64, u64>,
}

/// The ranks of a communicator that share a node with one rank.
struct Team {
    node: usize,
    devices: Vec<usize>,
    lsa_rank: usize,
}

fn lsa_team(machine: &Machine, clique: &Clique, rank: usize) -> Result<Team, ncclResult_t> {
    let devices = clique.devices()?;
    let node = machine.topology.node_of(devices[rank]);
    let members: Vec<usize> = (0..devices.len())
        .filter(|other| machine.topology.node_of(devices[*other]) == node)
        .collect();
    let lsa_rank = members
        .iter()
        .position(|member| *member == rank)
        .ok_or(ncclResult_t::ncclInternalError)?;
    Ok(Team {
        node,
        devices: members.iter().map(|member| devices[*member]).collect(),
        lsa_rank,
    })
}

fn reciprocal(value: usize) -> u32 {
    let value = u32::try_from(value).unwrap_or(u32::MAX).max(1);
    (u32::MAX / value).wrapping_add(u32::from(value.is_power_of_two()))
}

fn allocate(device: usize, bytes: u64) -> Result<u64, ncclResult_t> {
    let machine = emulated()?;
    let size = bytes.max(1).div_ceil(GRANULARITY) * GRANULARITY;
    let len = usize::try_from(size).map_err(invalid)?;
    let peers: Vec<usize> = (0..machine.devices.len())
        .filter(|other| *other == device || machine.topology.are_peers(device, *other))
        .collect();
    let (physical, address) = machine
        .host(move |memory| {
            let physical = memory.allocate_physical(device, len)?;
            let address = memory.reserve(size, GRANULARITY)?;
            memory.map(address, size, physical, 0)?;
            for peer in peers {
                memory.set_access(address, size, peer, true)?;
            }
            Ok((physical, address))
        })
        .map_err(cuda)?;
    machine.state().reservations.insert(address, size);
    with_nccl(|nccl| {
        nccl.symmetric.allocations.insert(
            address,
            Allocation {
                device,
                physical,
                size,
                windows: 0,
            },
        );
        Ok(())
    })?;
    Ok(address)
}

fn free(address: u64) -> Outcome {
    let machine = emulated()?;
    let allocation = with_nccl(|nccl| {
        let Entry::Occupied(entry) = nccl.symmetric.allocations.entry(address) else {
            return Err(ncclResult_t::ncclInvalidArgument);
        };
        if entry.get().windows > 0 {
            return Err(ncclResult_t::ncclInvalidUsage);
        }
        Ok(entry.remove())
    })?;
    machine
        .synchronize_device(allocation.device)
        .map_err(cuda)?;
    let (physical, size) = (allocation.physical, allocation.size);
    machine
        .host(move |memory| {
            memory.unmap(address, size)?;
            memory.free_reservation(address)?;
            memory.free(physical)
        })
        .map_err(cuda)?;
    machine.state().reservations.remove(&address);
    Ok(())
}

pub(crate) unsafe extern "C" fn memory_allocate(
    pointer: *mut *mut c_void,
    bytes: size_t,
) -> ncclResult_t {
    outcome(|| {
        let machine = emulated()?;
        let device = machine.runtime_device().map_err(cuda)?;
        let address = allocate(device, bytes)?;
        let address = usize::try_from(address).map_err(invalid)?;
        // SAFETY: `pointer` is the caller's output slot.
        unsafe { store(pointer, ptr::without_provenance_mut(address)) }.map_err(invalid)
    })
}

pub(crate) unsafe extern "C" fn memory_free(pointer: *mut c_void) -> ncclResult_t {
    outcome(|| free(pointer.addr() as u64))
}

/// Returns the multicast object of `team`'s node, creating it on first use.
fn node_multicast(
    machine: &Machine,
    spaces: &mut Spaces,
    team: &Team,
) -> Result<Multicast, ncclResult_t> {
    if let Some(multicast) = spaces.multicasts.get(&team.node) {
        return Ok(*multicast);
    }

    let devices = team.devices.clone();
    let multicast = machine
        .host(move |memory| {
            let handle = memory.create_multicast(&devices, MULTICAST_BYTES)?;
            let base = memory.map_multicast(handle)?;
            Ok(Multicast { handle, base })
        })
        .map_err(cuda)?;
    spaces.multicasts.insert(team.node, multicast);
    Ok(multicast)
}

/// Returns the flat range of window `sequence` on `team`'s node and the node's multicast object, creating either on first use.
fn space(
    machine: &Machine,
    clique: &Clique,
    sequence: u64,
    team: &Team,
) -> Result<(u64, Multicast), ncclResult_t> {
    let mut spaces = lock(&clique.spaces);
    let multicast = node_multicast(machine, &mut spaces, team)?;
    if let Some(space) = spaces.windows.get_mut(&(sequence, team.node)) {
        space.users += 1;
        return Ok((space.base, multicast));
    }

    let size = team.devices.len() as u64 * STRIDE;
    let base = machine
        .host(move |memory| memory.reserve(size, STRIDE))
        .map_err(cuda)?;
    machine.state().reservations.insert(base, size);
    spaces
        .windows
        .insert((sequence, team.node), Space { base, users: 1 });
    drop(spaces);
    Ok((base, multicast))
}

/// Registers `size` bytes at `buffer`, which must lie in memory from `ncclMemAlloc`, and returns the window.
///
/// Each rank maps its memory into its slot of a flat range shared by its node's ranks, and binds it into the node's multicast object; no rank waits for the others to register.
fn register(comm: ncclComm_t, buffer: u64, size: u64, flags: c_int) -> Result<u64, ncclResult_t> {
    let machine = emulated()?;
    if size == 0 {
        return Err(ncclResult_t::ncclInvalidArgument);
    }

    let (clique, rank, device, sequence) = with_communicator(comm, |communicator| {
        communicator.windows += 1;
        Ok((
            Arc::clone(&communicator.clique),
            communicator.rank,
            communicator.device,
            communicator.windows,
        ))
    })?;
    if (sequence + 1) * STRIDE > MULTICAST_BYTES {
        return Err(ncclResult_t::ncclSystemError);
    }

    let (allocation, physical, offset) = with_nccl(|nccl| {
        let (&start, allocation) = nccl
            .symmetric
            .allocations
            .range_mut(..=buffer)
            .next_back()
            .filter(|(start, allocation)| buffer + size <= **start + allocation.size)
            .ok_or(ncclResult_t::ncclInvalidArgument)?;
        if allocation.device != device {
            return Err(ncclResult_t::ncclInvalidArgument);
        }

        allocation.windows += 1;
        Ok((start, allocation.physical, buffer - start))
    })?;
    let team = lsa_team(machine, &clique, rank)?;
    let (base, multicast) = space(machine, &clique, sequence, &team)?;
    let slot = base + team.lsa_rank as u64 * STRIDE;
    let devices = team.devices.clone();
    let multicast_start = sequence * STRIDE;
    machine
        .host(move |memory| {
            memory.map(slot, size, physical, offset)?;
            for peer in devices {
                memory.set_access(slot, size, peer, true)?;
            }
            memory.bind_multicast(
                multicast.handle,
                multicast_start,
                device,
                physical,
                offset,
                size,
            )
        })
        .map_err(cuda)?;

    let lsa_rank = c_int::try_from(team.lsa_rank).map_err(invalid)?;
    let world_rank = c_int::try_from(rank).map_err(invalid)?;
    let multicast_pages = u32::try_from(multicast_start / 4096).map_err(invalid)?;
    let mut bytes = vec![0; size_of::<ncclWindow_vidmem>()];
    for (offset, value) in [
        (
            offset_of!(ncclWindow_vidmem, lsaFlatBase),
            base.to_le_bytes().as_slice(),
        ),
        (
            offset_of!(ncclWindow_vidmem, lsaRank),
            lsa_rank.to_le_bytes().as_slice(),
        ),
        (
            offset_of!(ncclWindow_vidmem, worldRank),
            world_rank.to_le_bytes().as_slice(),
        ),
        (
            offset_of!(ncclWindow_vidmem, stride4G),
            1u32.to_le_bytes().as_slice(),
        ),
        (
            offset_of!(ncclWindow_vidmem, mcOffset4K),
            multicast_pages.to_le_bytes().as_slice(),
        ),
        (
            offset_of!(ncclWindow_vidmem, winFlags),
            flags.to_le_bytes().as_slice(),
        ),
        (
            offset_of!(ncclWindow_vidmem, ginWinsDefaultBackend),
            sequence.to_le_bytes().as_slice(),
        ),
    ] {
        gin::write_at(&mut bytes, offset, value);
    }

    let window = machine.allocate(device, bytes.len() as u64).map_err(cuda)?;
    machine
        .host(move |memory| memory.write(window, &bytes))
        .map_err(cuda)?;

    gin::add_window(clique_key(&clique), sequence, rank, buffer, size);
    with_nccl(|nccl| {
        nccl.symmetric.windows.insert(
            window,
            Window {
                device,
                buffer,
                clique,
                sequence,
                node: team.node,
                base,
                slot,
                size,
                lsa_rank: team.lsa_rank,
                lsa_size: team.devices.len(),
                world_rank: rank,
                multicast,
                allocation,
            },
        );
        Ok(())
    })?;
    Ok(window)
}

/// Runs `body` on the record of a registered window, or fails for an unknown window.
fn with_window<T>(
    window: ncclWindow_t,
    body: impl FnOnce(&Window) -> Result<T, ncclResult_t>,
) -> Result<T, ncclResult_t> {
    with_nccl(|nccl| {
        let record = nccl
            .symmetric
            .windows
            .get(&(window.addr() as u64))
            .ok_or(ncclResult_t::ncclInvalidArgument)?;
        body(record)
    })
}

/// Returns a window's clique key, sequence number, and user buffer.
pub(crate) fn window_identity(window: ncclWindow_t) -> Result<(usize, u64, u64), ncclResult_t> {
    with_window(window, |record| {
        Ok((clique_key(&record.clique), record.sequence, record.buffer))
    })
}

pub(crate) unsafe extern "C" fn window_user_pointer(
    comm: ncclComm_t,
    window: ncclWindow_t,
    pointer: *mut *mut c_void,
) -> ncclResult_t {
    outcome(|| {
        with_communicator(comm, |_| Ok(()))?;
        let (_, _, buffer) = window_identity(window)?;
        let buffer = usize::try_from(buffer).map_err(invalid)?;
        // SAFETY: `pointer` is the caller's output slot.
        unsafe { store(pointer, ptr::without_provenance_mut(buffer)) }.map_err(invalid)
    })
}

fn deregister(window: u64) -> Outcome {
    let machine = emulated()?;
    let record = with_nccl(|nccl| {
        nccl.symmetric
            .windows
            .remove(&window)
            .ok_or(ncclResult_t::ncclInvalidArgument)
    })?;
    machine.synchronize_device(record.device).map_err(cuda)?;
    gin::remove_window(
        clique_key(&record.clique),
        record.sequence,
        record.world_rank,
    );
    let (slot, size, device) = (record.slot, record.size, record.device);
    let (multicast, start) = (record.multicast.handle, record.sequence * STRIDE);
    machine
        .host(move |memory| {
            memory.unmap(slot, size)?;
            memory.unbind_multicast(multicast, start, device)
        })
        .map_err(cuda)?;
    machine.free(window).map_err(cuda)?;
    with_nccl(|nccl| {
        if let Some(allocation) = nccl.symmetric.allocations.get_mut(&record.allocation) {
            allocation.windows -= 1;
        }
        Ok(())
    })?;
    let mut spaces = lock(&record.clique.spaces);
    let key = (record.sequence, record.node);
    let unused = spaces.windows.get_mut(&key).is_some_and(|space| {
        space.users -= 1;
        space.users == 0
    });
    let space = unused.then(|| spaces.windows.remove(&key)).flatten();
    drop(spaces);
    if let Some(space) = space {
        let base = space.base;
        machine
            .host(move |memory| memory.free_reservation(base))
            .map_err(cuda)?;
        machine.state().reservations.remove(&base);
    }
    Ok(())
}

pub(crate) unsafe extern "C" fn window_register(
    comm: ncclComm_t,
    buffer: *mut c_void,
    size: size_t,
    window: *mut ncclWindow_t,
    flags: c_int,
) -> ncclResult_t {
    outcome(|| {
        let handle = register(comm, buffer.addr() as u64, size, flags)?;
        let handle = usize::try_from(handle).map_err(invalid)?;
        // SAFETY: `window` is the caller's output slot.
        unsafe { store(window, ptr::without_provenance_mut(handle)) }.map_err(invalid)
    })
}

pub(crate) unsafe extern "C" fn window_deregister(
    comm: ncclComm_t,
    window: ncclWindow_t,
) -> ncclResult_t {
    outcome(|| {
        with_communicator(comm, |_| Ok(()))?;
        deregister(window.addr() as u64)
    })
}

/// Returns the bytes of `barriers` LSA barriers on a team of `ranks`: two epochs and a multicast inbox per barrier, then an inbox per barrier and rank.
fn barrier_bytes(barriers: u64, ranks: u64) -> u64 {
    (3 + ranks) * barriers * 4
}

pub(crate) unsafe extern "C" fn lsa_barrier_requirement(
    team: ncclTeam_t,
    barriers: c_int,
    handle: *mut ncclLsaBarrierHandle,
    requirement: *mut ncclDevResourceRequirements_t,
) -> ncclResult_t {
    outcome(|| {
        let count = u64::try_from(barriers).map_err(invalid)?;
        let ranks = u64::try_from(team.nRanks).map_err(invalid)?;
        // SAFETY: the caller passes writable output structures.
        let (handle, requirement) = unsafe { (handle.as_mut(), requirement.as_mut()) };
        let (Some(handle), Some(requirement)) = (handle, requirement) else {
            return Err(ncclResult_t::ncclInvalidArgument);
        };
        handle.nBarriers = barriers;
        *requirement = ncclDevResourceRequirements_t {
            bufferSize: barrier_bytes(count, ranks),
            bufferAlign: RESOURCE_UNIT,
            outBufferHandle: &raw mut handle.bufHandle,
            ..ncclDevResourceRequirements_t::default()
        };
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn gin_barrier_requirement(
    comm: ncclComm_t,
    team: ncclTeam_t,
    barriers: c_int,
    handle: *mut ncclGinBarrierHandle,
    requirement: *mut ncclDevResourceRequirements_t,
) -> ncclResult_t {
    outcome(|| {
        with_communicator(comm, |_| Ok(()))?;
        let signals = barriers
            .checked_mul(team.nRanks)
            .filter(|signals| *signals >= 0)
            .ok_or(ncclResult_t::ncclInvalidArgument)?;
        // SAFETY: the caller passes writable output structures.
        let (handle, requirement) = unsafe { (handle.as_mut(), requirement.as_mut()) };
        let (Some(handle), Some(requirement)) = (handle, requirement) else {
            return Err(ncclResult_t::ncclInvalidArgument);
        };
        *handle = ncclGinBarrierHandle::default();
        *requirement = ncclDevResourceRequirements_t {
            ginSignalCount: signals,
            outGinSignalStart: &raw mut handle.signal0,
            ..ncclDevResourceRequirements_t::default()
        };
        Ok(())
    })
}

/// What a device communicator's requirements ask for on one rank.
struct Resources {
    bytes: u64,
    lsa_barriers: c_int,
    gin: Option<GinResources>,
}

/// The GIN state and barriers a device communicator's requirements ask for.
struct GinResources {
    layout: Layout,
    world_barrier: ncclGinBarrierHandle,
    rail_barrier: ncclGinBarrierHandle,
}

/// Hands out consecutive indices from a counter.
fn take(cursor: &mut u32, count: c_int) -> Result<u32, ncclResult_t> {
    let start = *cursor;
    *cursor = cursor
        .checked_add(u32::try_from(count).map_err(invalid)?)
        .ok_or(ncclResult_t::ncclInvalidArgument)?;
    Ok(start)
}

/// Lays out the resource buffers, GIN signals, and GIN counters `requirements` asks for.
///
/// # Safety
///
/// The requirement lists must be valid linked lists whose output pointers are null or writable.
unsafe fn resources(
    requirements: &ncclDevCommRequirements_t,
    ranks: usize,
    lsa_size: usize,
    multicast: &ncclMultimemHandle_t,
) -> Result<Resources, ncclResult_t> {
    let connection = requirements.ginConnectionType;
    let is_gin = connection != ncclGinConnectionType_t::NCCL_GIN_CONNECTION_NONE;
    let is_proxy = requirements.ginType == ncclGinType_t::NCCL_GIN_TYPE_NONE
        || requirements.ginType == ncclGinType_t::NCCL_GIN_TYPE_PROXY;
    if (is_gin && (connection != ncclGinConnectionType_t::NCCL_GIN_CONNECTION_FULL || !is_proxy))
        || requirements.barrierCount > 0
        || requirements.lsaLLA2ABlockCount > 0
        || requirements.cftCaps != 0
    {
        return Err(ncclResult_t::ncclInvalidUsage);
    }

    let mut signals = u32::try_from(requirements.ginSignalCount.max(0)).map_err(invalid)?;
    let mut counters = u32::try_from(requirements.ginCounterCount.max(0)).map_err(invalid)?;
    let barriers = requirements.lsaBarrierCount.max(0);
    let bytes = barrier_bytes(u64::try_from(barriers).map_err(invalid)?, lsa_size as u64);
    let mut cursor = bytes.div_ceil(RESOURCE_UNIT) * RESOURCE_UNIT;

    let mut resource = requirements.resourceRequirementsList;
    // SAFETY: the caller guarantees a valid list.
    while let Some(request) = unsafe { resource.as_ref() } {
        if request.ginSignalCount > 0 || request.ginCounterCount > 0 {
            let signal_start = take(&mut signals, request.ginSignalCount.max(0))?;
            let counter_start = take(&mut counters, request.ginCounterCount.max(0))?;
            // SAFETY: the caller guarantees writable output pointers.
            unsafe {
                if !request.outGinSignalStart.is_null() {
                    request.outGinSignalStart.write(signal_start);
                }
                if !request.outGinCounterStart.is_null() {
                    request.outGinCounterStart.write(counter_start);
                }
            }
        }

        if request.bufferSize > 0 {
            let alignment = request.bufferAlign.max(RESOURCE_UNIT);
            cursor = cursor.div_ceil(alignment) * alignment;
            let handle = u32::try_from(cursor / RESOURCE_UNIT).map_err(invalid)?;
            if !request.outBufferHandle.is_null() {
                // SAFETY: the caller guarantees writable output pointers.
                unsafe { request.outBufferHandle.write(handle) };
            }

            cursor += request.bufferSize.div_ceil(RESOURCE_UNIT) * RESOURCE_UNIT;
        }

        resource = request.next;
    }

    let lsa_ranks = c_int::try_from(lsa_size).map_err(invalid)?;
    let mut team = requirements.teamRequirementsList;
    // SAFETY: the caller guarantees a valid list.
    while let Some(request) = unsafe { team.as_ref() } {
        let is_lsa = request.team.nRanks == lsa_ranks && request.team.stride == 1;
        if request.multimem {
            if !is_lsa {
                return Err(ncclResult_t::ncclInvalidUsage);
            }
            if !request.outMultimemHandle.is_null() {
                // SAFETY: the caller guarantees writable output pointers.
                unsafe { request.outMultimemHandle.write(*multicast) };
            }
        }

        team = request.next;
    }

    let world_ranks = c_int::try_from(ranks).map_err(invalid)?;
    let rail_ranks = c_int::try_from(ranks / lsa_size.max(1)).map_err(invalid)?;
    let world_count = requirements.worldGinBarrierCount.max(0);
    let rail_count = requirements.railGinBarrierCount.max(0);
    let world_barrier = ncclGinBarrierHandle {
        signal0: take(
            &mut signals,
            world_count
                .checked_mul(world_ranks)
                .ok_or(ncclResult_t::ncclInvalidArgument)?,
        )?,
        ..ncclGinBarrierHandle::default()
    };
    let rail_barrier = ncclGinBarrierHandle {
        signal0: take(
            &mut signals,
            rail_count
                .checked_mul(rail_ranks)
                .ok_or(ncclResult_t::ncclInvalidArgument)?,
        )?,
        ..ncclGinBarrierHandle::default()
    };
    if !is_gin && (signals > 0 || counters > 0) {
        return Err(ncclResult_t::ncclInvalidUsage);
    }

    let gin = is_gin.then(|| GinResources {
        layout: Layout {
            contexts: usize::try_from(requirements.ginContextCount)
                .unwrap_or(0)
                .max(1),
            ranks,
            queue_size: gin::queue_size(requirements.ginQueueDepth),
            signals,
            counters,
        },
        world_barrier,
        rail_barrier,
    });
    Ok(Resources {
        bytes: cursor.max(RESOURCE_UNIT),
        lsa_barriers: barriers,
        gin,
    })
}

pub(crate) unsafe extern "C" fn device_communicator_create(
    comm: ncclComm_t,
    requirements: *const ncclDevCommRequirements_t,
    device_communicator: *mut ncclDevComm_t,
) -> ncclResult_t {
    outcome(|| {
        let machine = emulated()?;
        // SAFETY: the caller passes valid requirements.
        let requirements =
            unsafe { requirements.as_ref() }.ok_or(ncclResult_t::ncclInvalidArgument)?;
        if device_communicator.is_null() {
            return Err(ncclResult_t::ncclInvalidArgument);
        }

        let Caller {
            clique,
            rank,
            device,
        } = caller(comm)?;
        let ranks = clique.size;
        let team = lsa_team(machine, &clique, rank)?;
        let lsa_size = team.devices.len();
        let multicast = node_multicast(machine, &mut lock(&clique.spaces), &team)?;
        let multimem = ncclMultimemHandle {
            mcBasePtr: ptr::without_provenance_mut(
                usize::try_from(multicast.base).map_err(invalid)?,
            ),
        };
        // SAFETY: the caller passes valid requirement lists.
        let resources = unsafe { resources(requirements, ranks, lsa_size, &multimem) }?;
        let memory = allocate(device, resources.bytes)?;
        let flags = c_int::try_from(NCCL_WIN_COLL_SYMMETRIC).map_err(invalid)?;
        let window = register(comm, memory, resources.bytes, flags)?;
        let (base, multicast_pages) = with_nccl(|nccl| {
            nccl.symmetric.resources.insert(window, memory);
            let record = &nccl.symmetric.windows[&window];
            Ok((
                record.base,
                u32::try_from(record.sequence * STRIDE / 4096).map_err(invalid)?,
            ))
        })?;

        let placement = match &resources.gin {
            Some(gin) => gin::install(
                machine,
                &gin.layout,
                device,
                clique_key(&clique),
                rank,
                window,
            )?,
            None => Placement {
                handle: 0,
                shadows: 0,
            },
        };
        let gin = resources.gin.as_ref();
        let value = ncclDevComm_t {
            magic: NCCL_API_MAGIC,
            version: NCCL_VERSION_CODE,
            rank: c_int::try_from(rank).map_err(invalid)?,
            nRanks: c_int::try_from(ranks).map_err(invalid)?,
            nRanks_rcp32: reciprocal(ranks),
            lsaRank: c_int::try_from(team.lsa_rank).map_err(invalid)?,
            lsaSize: c_int::try_from(lsa_size).map_err(invalid)?,
            lsaSize_rcp32: reciprocal(lsa_size),
            resourceWindow: ptr::without_provenance_mut(usize::try_from(window).map_err(invalid)?),
            resourceWindow_inlined: ncclResourceWindow_vidmem {
                lsaFlatBase: ptr::without_provenance_mut(usize::try_from(base).map_err(invalid)?),
                stride4G: 1,
                mcOffset4K: multicast_pages,
            },
            lsaMultimem: if requirements.lsaMultimem {
                multimem
            } else {
                ncclMultimemHandle::default()
            },
            lsaBarrier: ncclLsaBarrierHandle {
                bufHandle: 0,
                nBarriers: resources.lsa_barriers,
            },
            ginConnectionCount: u8::from(gin.is_some()),
            ginNetDeviceTypes: [
                u8::try_from(ncclNetDeviceType::NCCL_NET_DEVICE_GIN_PROXY.0).map_err(invalid)?,
                0,
                0,
                0,
            ],
            ginHandles: [
                ptr::without_provenance_mut(usize::try_from(placement.handle).map_err(invalid)?),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
            ],
            ginSignalCount: c_int::try_from(gin.map_or(0, |gin| gin.layout.signals))
                .map_err(invalid)?,
            ginCounterCount: c_int::try_from(gin.map_or(0, |gin| gin.layout.counters))
                .map_err(invalid)?,
            ginSignalShadows: ptr::without_provenance_mut(
                usize::try_from(placement.shadows).map_err(invalid)?,
            ),
            ginContextCount: u32::try_from(gin.map_or(0, |gin| gin.layout.contexts))
                .map_err(invalid)?,
            ginConnectionStride: 1,
            ginContextStride: 1,
            worldGinBarrier: gin
                .map_or_else(ncclGinBarrierHandle::default, |gin| gin.world_barrier),
            railGinBarrier: gin.map_or_else(ncclGinBarrierHandle::default, |gin| gin.rail_barrier),
            ..ncclDevComm_t::default()
        };
        // SAFETY: `device_communicator` is the caller's output slot, checked above.
        unsafe { store(device_communicator, value) }.map_err(invalid)
    })
}

pub(crate) unsafe extern "C" fn device_communicator_destroy(
    comm: ncclComm_t,
    device_communicator: *const ncclDevComm_t,
) -> ncclResult_t {
    outcome(|| {
        let device = caller(comm)?.device;
        // SAFETY: the caller passes the device communicator `ncclDevCommCreate` filled.
        let value =
            unsafe { device_communicator.as_ref() }.ok_or(ncclResult_t::ncclInvalidArgument)?;
        let window = value.resourceWindow.addr() as u64;
        // Kernels still using the device communicator need the GIN proxy until they finish.
        emulated()?.synchronize_device(device).map_err(cuda)?;
        let memory = with_nccl(|nccl| {
            nccl.symmetric
                .resources
                .remove(&window)
                .ok_or(ncclResult_t::ncclInvalidArgument)
        })?;
        if let Some(block) = gin::uninstall(window) {
            emulated()?.free(block).map_err(cuda)?;
        }

        deregister(window)?;
        free(memory)
    })
}

pub(crate) unsafe extern "C" fn query_properties(
    comm: ncclComm_t,
    properties: *mut ncclCommProperties_t,
) -> ncclResult_t {
    outcome(|| {
        let machine = emulated()?;
        // SAFETY: the caller passes a writable properties structure.
        let properties = unsafe { properties.as_mut() }.ok_or(ncclResult_t::ncclInvalidArgument)?;
        let Caller {
            clique,
            rank,
            device,
        } = caller(comm)?;
        let devices = clique.devices()?;
        let mut nodes: Vec<usize> = devices
            .iter()
            .map(|device| machine.topology.node_of(*device))
            .collect();
        nodes.sort_unstable();
        nodes.dedup();
        properties.rank = c_int::try_from(rank).map_err(invalid)?;
        properties.nRanks = c_int::try_from(clique.size).map_err(invalid)?;
        properties.cudaDev = c_int::try_from(device).map_err(invalid)?;
        properties.nvmlDev = properties.cudaDev;
        properties.deviceApiSupport = true;
        properties.multimemSupport = true;
        properties.nLsaTeams = c_int::try_from(nodes.len()).map_err(invalid)?;
        properties.hostRmaSupport = true;
        properties.commHash = clique_key(&clique) as u64;
        properties.ginConnectionType = ncclGinConnectionType_t::NCCL_GIN_CONNECTION_FULL;
        properties.ginSupport
            [usize::try_from(ncclGinType_t::NCCL_GIN_TYPE_PROXY.0).map_err(invalid)?] = true;
        properties.ginMinStride = 1;
        properties.devCommRuntimeVersionSize = size_of::<ncclDevComm_t>() as size_t;
        properties.cftSupport = false;
        Ok(())
    })
}

struct Position {
    rank: c_int,
    ranks: c_int,
    lsa_rank: c_int,
    lsa_size: c_int,
}

/// Returns a communicator's rank and rank count, and its rank and size in its LSA team, or `None` for an unknown communicator.
fn position(comm: ncclComm_t) -> Option<Position> {
    let machine = emulated().ok()?;
    let Caller { clique, rank, .. } = caller(comm).ok()?;
    let team = lsa_team(machine, &clique, rank).ok()?;
    Some(Position {
        rank: c_int::try_from(rank).ok()?,
        ranks: c_int::try_from(clique.size).ok()?,
        lsa_rank: c_int::try_from(team.lsa_rank).ok()?,
        lsa_size: c_int::try_from(team.devices.len()).ok()?,
    })
}

pub(crate) unsafe extern "C" fn team_world(comm: ncclComm_t) -> ncclTeam_t {
    position(comm).map_or_else(ncclTeam_t::default, |position| ncclTeam_t {
        nRanks: position.ranks,
        rank: position.rank,
        stride: 1,
    })
}

pub(crate) unsafe extern "C" fn team_lsa(comm: ncclComm_t) -> ncclTeam_t {
    position(comm).map_or_else(ncclTeam_t::default, |position| ncclTeam_t {
        nRanks: position.lsa_size,
        rank: position.lsa_rank,
        stride: 1,
    })
}

pub(crate) unsafe extern "C" fn team_rail(comm: ncclComm_t) -> ncclTeam_t {
    position(comm).map_or_else(ncclTeam_t::default, |position| ncclTeam_t {
        nRanks: position.ranks / position.lsa_size,
        rank: position.rank / position.lsa_size,
        stride: position.lsa_size,
    })
}

pub(crate) unsafe extern "C" fn team_rank_to_world(
    comm: ncclComm_t,
    team: ncclTeam_t,
    rank: c_int,
) -> c_int {
    position(comm).map_or(-1, |position| {
        position.rank + (rank - team.rank) * team.stride
    })
}

pub(crate) unsafe extern "C" fn team_rank_to_lsa(
    comm: ncclComm_t,
    team: ncclTeam_t,
    rank: c_int,
) -> c_int {
    position(comm).map_or(-1, |position| {
        position.lsa_rank + (rank - team.rank) * team.stride
    })
}

/// Writes the address `locate` computes from a window's record, or fails for an unknown window.
fn window_pointer(
    window: ncclWindow_t,
    pointer: *mut *mut c_void,
    locate: impl FnOnce(&Window) -> Result<u64, ncclResult_t>,
) -> Outcome {
    let address = with_window(window, locate)?;
    let address = usize::try_from(address).map_err(invalid)?;
    // SAFETY: `pointer` is the caller's output slot.
    unsafe { store(pointer, ptr::without_provenance_mut(address)) }.map_err(invalid)
}

pub(crate) unsafe extern "C" fn lsa_pointer(
    window: ncclWindow_t,
    offset: size_t,
    lsa_rank: c_int,
    pointer: *mut *mut c_void,
) -> ncclResult_t {
    outcome(|| {
        window_pointer(window, pointer, |record| {
            let lsa_rank = usize::try_from(lsa_rank).map_err(invalid)?;
            if lsa_rank >= record.lsa_size || offset >= record.size {
                return Err(ncclResult_t::ncclInvalidArgument);
            }
            Ok(record.base + lsa_rank as u64 * STRIDE + offset)
        })
    })
}

pub(crate) unsafe extern "C" fn peer_pointer(
    window: ncclWindow_t,
    offset: size_t,
    peer: c_int,
    pointer: *mut *mut c_void,
) -> ncclResult_t {
    outcome(|| {
        window_pointer(window, pointer, |record| {
            let index = i64::try_from(record.lsa_rank).map_err(invalid)? + i64::from(peer)
                - i64::try_from(record.world_rank).map_err(invalid)?;
            Ok(match usize::try_from(index) {
                Ok(index) if index < record.lsa_size => {
                    record.base + index as u64 * STRIDE + offset
                }
                _ => 0,
            })
        })
    })
}

pub(crate) unsafe extern "C" fn lsa_multimem_pointer(
    window: ncclWindow_t,
    offset: size_t,
    pointer: *mut *mut c_void,
) -> ncclResult_t {
    outcome(|| {
        window_pointer(window, pointer, |record| {
            Ok(record.multicast.base + record.sequence * STRIDE + offset)
        })
    })
}

pub(crate) unsafe extern "C" fn multimem_pointer(
    window: ncclWindow_t,
    offset: size_t,
    multimem: ncclMultimemHandle_t,
    pointer: *mut *mut c_void,
) -> ncclResult_t {
    outcome(|| {
        window_pointer(window, pointer, |record| {
            Ok(multimem.mcBasePtr.addr() as u64 + record.sequence * STRIDE + offset)
        })
    })
}
