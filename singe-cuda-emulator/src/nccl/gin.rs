use std::{
    collections::HashMap,
    ffi::c_int,
    sync::{Mutex, OnceLock},
};

use singe_nccl_sys::ncclResult_t;
use singe_ptx::emulator::{Agent, ExecutionError, HostMemory};

use crate::{
    machine::Machine,
    nccl::{cuda, invalid, lock},
};

/// Bytes of `ncclGinProxyGpuCtx_t`, which NCCL defines only in its device headers.
const CONTEXT_BYTES: u64 = 80;

/// Offset of `flushesAllPutsOnAnySignal` in `ncclGinProxyGpuCtx_t`.
const FLUSHES_OFFSET: usize = 72;

/// Bytes of one GIN descriptor, `ncclGinProxyGfd_t`.
const DESCRIPTOR_BYTES: u64 = 128;

/// Descriptors per peer queue when the requirements leave the depth to NCCL.
const QUEUE_DEPTH: u32 = 256;

const OP_PUT: u64 = 1 << 0;
const OP_INLINE: u64 = 1 << 1;
const OP_COUNTER: u64 = 1 << 2;
const OP_SIGNAL_INC: u64 = 1 << 3;
const OP_SIGNAL_ADD: u64 = 1 << 4;
const OP_VA_SIGNAL: u64 = 1 << 5;
const OP_GET: u64 = 1 << 6;
const OP_FLUSH: u64 = 1 << 7;

/// One proxy context of one rank: the device addresses of its queues, indices, counters, and signals.
struct Context {
    clique: usize,
    rank: usize,
    index: usize,
    ranks: usize,
    queue_size: u32,
    queues: u64,
    producers: u64,
    consumers: u64,
    counters: u64,
    signals: u64,
    last_visible_get: u64,
    /// Descriptors consumed per peer, the value last written to the consumer index.
    consumed: Vec<u32>,
    owner: u64,
}

struct Registry {
    contexts: Vec<Context>,
    /// Buffers and sizes of registered windows, by clique, window sequence number, and rank.
    windows: HashMap<(usize, u64, usize), (u64, u64)>,
    /// The memory holding each device communicator's GIN state, by the communicator's resource window.
    blocks: HashMap<u64, u64>,
}

impl Registry {
    fn window_address(
        &self,
        clique: usize,
        handle: u64,
        rank: usize,
        offset: u64,
        len: usize,
    ) -> Result<u64, ExecutionError> {
        let (buffer, size) = self
            .windows
            .get(&(clique, handle, rank))
            .copied()
            .ok_or_else(|| {
                ExecutionError::Invalid(format!("window {handle} is not registered on rank {rank}"))
            })?;
        if offset.checked_add(len as u64).is_none_or(|end| end > size) {
            return Err(ExecutionError::Invalid(format!(
                "{len} bytes at offset {offset} overrun rank {rank}'s part of window {handle}, which holds {size} bytes"
            )));
        }
        Ok(buffer + offset)
    }
}

/// The proxy's state, shared between the host API and the agent on the emulator's worker.
///
/// No thread holds it while waiting for the worker, since the agent locks it between scheduler steps.
static REGISTRY: Mutex<Option<Registry>> = Mutex::new(None);

static AGENT: OnceLock<Agent> = OnceLock::new();

fn with_registry<T>(body: impl FnOnce(&mut Registry) -> T) -> T {
    let mut registry = lock(&REGISTRY);
    body(registry.get_or_insert_with(|| Registry {
        contexts: Vec::new(),
        windows: HashMap::new(),
        blocks: HashMap::new(),
    }))
}

/// Returns the queue size NCCL's device code accepts for a requested depth: a power of two.
pub(crate) fn queue_size(depth: c_int) -> u32 {
    u32::try_from(depth)
        .ok()
        .filter(|depth| *depth > 0)
        .map_or(QUEUE_DEPTH, u32::next_power_of_two)
}

/// The size of a device communicator's GIN state on one rank.
pub(crate) struct Layout {
    pub(crate) contexts: usize,
    pub(crate) ranks: usize,
    pub(crate) queue_size: u32,
    pub(crate) signals: u32,
    pub(crate) counters: u32,
}

impl Layout {
    fn context_bytes(&self) -> u64 {
        let ranks = self.ranks as u64;
        let bytes = ranks * u64::from(self.queue_size) * DESCRIPTOR_BYTES
            + 4 * ranks * 4
            + 8 * u64::from(self.counters)
            + 16 * u64::from(self.signals);
        bytes.next_multiple_of(DESCRIPTOR_BYTES)
    }

    fn array_bytes(&self) -> u64 {
        (self.contexts as u64 * CONTEXT_BYTES).next_multiple_of(DESCRIPTOR_BYTES)
    }

    /// Returns the bytes of the whole block: the context array, each context's state, and the signal shadows.
    fn bytes(&self) -> u64 {
        let contexts = self.contexts as u64;
        self.array_bytes()
            + contexts * self.context_bytes()
            + contexts * u64::from(self.signals) * 8
    }
}

/// The device addresses of a rank's GIN state, for the device communicator.
pub(crate) struct Placement {
    pub(crate) handle: u64,
    pub(crate) shadows: u64,
}

pub(crate) fn write_at(bytes: &mut [u8], offset: usize, value: &[u8]) {
    bytes[offset..offset + value.len()].copy_from_slice(value);
}

/// Allocates a rank's GIN state on `device`, writes its context array, and proxies its queues from now on.
///
/// `owner` is the device communicator's resource window, which [`uninstall`] takes.
pub(crate) fn install(
    machine: &Machine,
    layout: &Layout,
    device: usize,
    clique: usize,
    rank: usize,
    owner: u64,
) -> Result<Placement, ncclResult_t> {
    let base = machine.allocate(device, layout.bytes()).map_err(cuda)?;
    let ranks = layout.ranks as u64;
    let rank_count = c_int::try_from(layout.ranks).map_err(invalid)?;
    let mut array = vec![0; layout.contexts * CONTEXT_BYTES as usize];
    let mut contexts = Vec::with_capacity(layout.contexts);
    let (entries, _) = array.as_chunks_mut::<{ CONTEXT_BYTES as usize }>();
    for (index, entry) in entries.iter_mut().enumerate() {
        let queues = base + layout.array_bytes() + index as u64 * layout.context_bytes();
        let producers = queues + ranks * u64::from(layout.queue_size) * DESCRIPTOR_BYTES;
        let consumers = producers + 4 * ranks;
        let last_issued_get = consumers + 4 * ranks;
        let last_visible_get = last_issued_get + 4 * ranks;
        let counters = last_visible_get + 4 * ranks;
        let signals = counters + 8 * u64::from(layout.counters);
        let signal_offsets = signals + 8 * u64::from(layout.signals);

        write_at(entry, 0, &rank_count.to_le_bytes());
        write_at(entry, 4, &layout.queue_size.to_le_bytes());
        for (offset, address) in [
            (8, queues),
            (16, producers),
            (24, consumers),
            (32, counters),
            (40, signals),
            (48, signal_offsets),
            (56, last_issued_get),
            (64, last_visible_get),
        ] {
            write_at(entry, offset, &address.to_le_bytes());
        }

        entry[FLUSHES_OFFSET] = 1;
        contexts.push(Context {
            clique,
            rank,
            index,
            ranks: layout.ranks,
            queue_size: layout.queue_size,
            queues,
            producers,
            consumers,
            counters,
            signals,
            last_visible_get,
            consumed: vec![0; layout.ranks],
            owner,
        });
    }

    machine
        .host(move |memory| memory.write(base, &array))
        .map_err(cuda)?;
    with_registry(|registry| {
        registry.contexts.extend(contexts);
        registry.blocks.insert(owner, base);
    });
    AGENT.get_or_init(|| machine.emulator.add_agent(poll));
    Ok(Placement {
        handle: base,
        shadows: base + layout.array_bytes() + layout.contexts as u64 * layout.context_bytes(),
    })
}

/// Stops proxying the contexts of the device communicator `owner` and returns the memory of its GIN state.
pub(crate) fn uninstall(owner: u64) -> Option<u64> {
    with_registry(|registry| {
        registry.contexts.retain(|context| context.owner != owner);
        registry.blocks.remove(&owner)
    })
}

pub(crate) fn add_window(clique: usize, sequence: u64, rank: usize, buffer: u64, size: u64) {
    with_registry(|registry| {
        registry
            .windows
            .insert((clique, sequence, rank), (buffer, size))
    });
}

pub(crate) fn remove_window(clique: usize, sequence: u64, rank: usize) {
    with_registry(|registry| registry.windows.remove(&(clique, sequence, rank)));
}

/// A descriptor's fields, decoded from its sixteen flagged words.
struct Descriptor {
    op: u64,
    size: u64,
    source_offset: u64,
    source_window: u64,
    inline_value: u64,
    destination_offset: u64,
    destination_window: u64,
    counter: u64,
    signal: u64,
    signal_value: u64,
}

impl Descriptor {
    /// Decodes a descriptor, or returns `None` while the device has not written every word yet.
    fn decode(bytes: &[u8]) -> Option<Self> {
        let words: Vec<u64> = bytes
            .as_chunks::<8>()
            .0
            .iter()
            .copied()
            .map(u64::from_le_bytes)
            .collect();
        if words.len() != 16 || words.iter().any(|word| word & 1 == 0) {
            return None;
        }
        Some(Self {
            op: (words[7] >> 8) & 0xffff,
            size: words[0] >> 7,
            source_offset: words[1] >> 1,
            source_window: words[2] >> 1,
            inline_value: ((words[1] >> 8) & 0xffff_ffff)
                | (((words[1] >> 40) & 0xffff) << 32)
                | (((words[2] >> 8) & 0xffff) << 48),
            destination_offset: words[3] >> 1,
            destination_window: words[4] >> 1,
            counter: (words[5] >> 1) & 0x7f_ffff,
            signal: (words[5] >> 24) & 0xff_ffff,
            signal_value: (words[5] >> 48)
                | (((words[6] >> 8) & 0xffff) << 16)
                | (((words[6] >> 24) & 0xffff_ffff) << 32),
        })
    }
}

/// Returns the address of `len` bytes at `offset` in rank `rank`'s part of window `handle`.
pub(crate) fn resolve(
    clique: usize,
    handle: u64,
    rank: usize,
    offset: u64,
    len: usize,
) -> Result<u64, ExecutionError> {
    with_registry(|registry| registry.window_address(clique, handle, rank, offset, len))
}

fn add(memory: &mut HostMemory<'_>, address: u64, value: u64) -> Result<(), ExecutionError> {
    let current: u64 = memory.read_value(address)?;
    memory.write_value(address, current.wrapping_add(value))
}

/// Performs one descriptor that rank `context.rank` posted for `peer`.
fn perform(
    memory: &mut HostMemory<'_>,
    registry: &Registry,
    context: &Context,
    peer: usize,
    descriptor: &Descriptor,
) -> Result<(), ExecutionError> {
    let op = descriptor.op;
    if op & OP_FLUSH != 0 {
        return Ok(());
    }

    let clique = context.clique;
    let size = usize::try_from(descriptor.size)
        .map_err(|_| ExecutionError::Invalid("gin size out of range".into()))?;

    if op & OP_GET != 0 {
        let source = registry.window_address(
            clique,
            descriptor.source_window,
            peer,
            descriptor.source_offset,
            size,
        )?;
        let destination = registry.window_address(
            clique,
            descriptor.destination_window,
            context.rank,
            descriptor.destination_offset,
            size,
        )?;
        return memory.copy(source, destination, size);
    }

    let increment = if op & OP_SIGNAL_ADD != 0 {
        descriptor.signal_value
    } else {
        1
    };
    if op & OP_VA_SIGNAL != 0 {
        let target = registry.window_address(
            clique,
            descriptor.source_window,
            peer,
            descriptor.source_offset,
            8,
        )?;
        return add(memory, target, increment);
    }

    if op & OP_PUT != 0 {
        let destination = |len| {
            registry.window_address(
                clique,
                descriptor.destination_window,
                peer,
                descriptor.destination_offset,
                len,
            )
        };
        if op & OP_INLINE != 0 {
            let len = size.min(8);
            memory.write(
                destination(len)?,
                &descriptor.inline_value.to_le_bytes()[..len],
            )?;
        } else if size > 0 {
            let source = registry.window_address(
                clique,
                descriptor.source_window,
                context.rank,
                descriptor.source_offset,
                size,
            )?;
            memory.copy(source, destination(size)?, size)?;
        }
    }

    if op & (OP_SIGNAL_INC | OP_SIGNAL_ADD) != 0 {
        let target = registry
            .contexts
            .iter()
            .find(|other| {
                other.clique == clique && other.rank == peer && other.index == context.index
            })
            .ok_or_else(|| {
                ExecutionError::Invalid(format!(
                    "gin signal for rank {peer}, which has no device communicator"
                ))
            })?;
        add(memory, target.signals + 8 * descriptor.signal, increment)?;
    }

    if op & OP_COUNTER != 0 {
        add(memory, context.counters + 8 * descriptor.counter, 1)?;
    }
    Ok(())
}

/// Performs every complete descriptor the ranks have posted, in order per peer queue, and advances the consumer indices.
fn poll(memory: &mut HostMemory<'_>) -> Result<(), ExecutionError> {
    let mut guard = lock(&REGISTRY);
    let Some(registry) = guard.as_mut() else {
        return Ok(());
    };
    for index in 0..registry.contexts.len() {
        for peer in 0..registry.contexts[index].ranks {
            let context = &registry.contexts[index];
            let peer_offset = 4 * peer as u64;
            let produced: u32 = memory.read_value(context.producers + peer_offset)?;
            let mut consumed = context.consumed[peer];
            while consumed != produced {
                let slot = context.queues
                    + (peer as u64 * u64::from(context.queue_size)
                        + u64::from(consumed & (context.queue_size - 1)))
                        * DESCRIPTOR_BYTES;
                let Some(descriptor) =
                    Descriptor::decode(memory.read(slot, DESCRIPTOR_BYTES as usize)?)
                else {
                    break;
                };
                perform(memory, registry, context, peer, &descriptor)?;
                memory.write(slot, &[0; DESCRIPTOR_BYTES as usize])?;
                consumed = consumed.wrapping_add(1);
                memory.write_value(context.last_visible_get + peer_offset, consumed)?;
                memory.write_value(context.consumers + peer_offset, consumed)?;
            }

            registry.contexts[index].consumed[peer] = consumed;
        }
    }

    drop(guard);
    Ok(())
}
