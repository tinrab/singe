use std::{
    fmt::{self, Debug, Formatter},
    result,
    sync::{Arc, Weak, atomic::Ordering},
    task::Poll,
};

use crate::emulator::{
    execution_error::{ExecutionError, SubmitError},
    memory::Memory,
    runtime::{
        Client, CommandKind, Shared,
        completion::{Pending, Submission},
        device::Emulator,
        lock,
        options::Topology,
        scalar::PtxScalar,
        submit::{RetainedFailure, submit_poll},
        worker::Worker,
    },
};

type Result<T> = result::Result<T, ExecutionError>;

/// The memory of every device in a machine, as host work running on the worker sees it.
///
/// Addresses are the generic addresses kernels use.
/// Host work bypasses peer access and mapping access checks, as copies through the CUDA API do, but still faults on unmapped or out-of-bounds addresses.
pub struct HostMemory<'a> {
    pub(crate) memory: &'a mut Memory,
    pub(crate) topology: Topology,
    pub(crate) failures: &'a [(usize, RetainedFailure)],
}

impl Debug for HostMemory<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostMemory").finish_non_exhaustive()
    }
}

impl HostMemory<'_> {
    /// Returns the layout of the machine's devices.
    #[must_use]
    pub fn topology(&self) -> Topology {
        self.topology
    }

    /// Returns whether a fault stopped `device`, so work that waits on it can give up.
    #[must_use]
    pub fn is_faulted(&self, device: usize) -> bool {
        self.failures.iter().any(|(faulted, _)| *faulted == device)
    }

    /// Returns the `len` bytes at `address`.
    ///
    /// # Errors
    ///
    /// Returns an error if the range is not mapped or holds undefined bytes.
    pub fn read(&self, address: u64, len: usize) -> Result<&[u8]> {
        self.memory.read(address, len, 1, None, None)
    }

    /// Writes `bytes` at `address`.
    ///
    /// # Errors
    ///
    /// Returns an error if the range is not mapped or is read-only.
    pub fn write(&mut self, address: u64, bytes: &[u8]) -> Result<()> {
        self.memory.write(address, bytes, 1, None, None)
    }

    /// Reads one value of `T` at `address`.
    ///
    /// # Errors
    ///
    /// Returns an error if the range is not mapped or holds undefined bytes.
    pub fn read_value<T: PtxScalar>(&self, address: u64) -> Result<T> {
        Ok(T::decode(self.read(address, T::SIZE)?))
    }

    /// Writes one value of `T` at `address`.
    ///
    /// # Errors
    ///
    /// Returns an error if the range is not mapped or is read-only.
    pub fn write_value<T: PtxScalar>(&mut self, address: u64, value: T) -> Result<()> {
        let mut bytes = Vec::with_capacity(T::SIZE);
        value.encode(&mut bytes);
        self.write(address, &bytes)
    }

    /// Copies `len` bytes from `source` to `destination`; the ranges may be on different devices.
    ///
    /// # Errors
    ///
    /// Returns an error if either range is not mapped, the source holds undefined bytes, or the destination is read-only.
    pub fn copy(&mut self, source: u64, destination: u64, len: usize) -> Result<()> {
        let bytes = self.read(source, len)?.to_vec();
        self.write(destination, &bytes)
    }

    /// Returns the device that owns the allocation containing `address`, if it is device memory.
    #[must_use]
    pub fn device_of(&self, address: u64) -> Option<usize> {
        self.memory.device_of(address)
    }

    /// Allocates `size` zeroed bytes on `device`, aligned to `alignment`, and returns their address.
    ///
    /// # Errors
    ///
    /// Returns an error if `device` does not exist, `alignment` is not a power of two, or the device's storage limit is reached.
    pub fn allocate(&mut self, device: usize, size: usize, alignment: u32) -> Result<u64> {
        self.check_device(device)?;
        if !alignment.is_power_of_two() {
            return Err(ExecutionError::Memory("allocation alignment".into()));
        }
        self.memory.allocate_device(device, size, alignment, false)
    }

    /// Frees an allocation from [`allocate`](Self::allocate) or [`allocate_physical`](Self::allocate_physical) by its address.
    ///
    /// # Errors
    ///
    /// Returns an error if no device allocation starts at `address`.
    pub fn free(&mut self, address: u64) -> Result<()> {
        if self.memory.device_of(address).is_none() || !self.memory.is_allocation_start(address) {
            return Err(ExecutionError::Memory(
                "no device allocation starts at this address".into(),
            ));
        }

        if self.memory.is_mapped(address) {
            return Err(ExecutionError::Memory(
                "physical memory is still mapped or bound".into(),
            ));
        }

        self.memory.release(address);
        Ok(())
    }

    /// Allocates `size` bytes of physical memory on `device`, reachable only once mapped, and returns its handle.
    ///
    /// # Errors
    ///
    /// Returns an error if `device` does not exist or the device's storage limit is reached.
    pub fn allocate_physical(&mut self, device: usize, size: usize) -> Result<u64> {
        self.check_device(device)?;
        self.memory.allocate_device(device, size, 1 << 12, true)
    }

    /// Reserves `size` bytes of virtual addresses with nothing mapped, aligned to `alignment`, and returns the first address.
    ///
    /// # Errors
    ///
    /// Returns an error if `size` is zero, `alignment` is not a power of two, or the address space is exhausted.
    pub fn reserve(&mut self, size: u64, alignment: u64) -> Result<u64> {
        self.memory.reserve_range(size, alignment)
    }

    /// Frees a reserved range, or a multicast mapping, by its first address.
    ///
    /// # Errors
    ///
    /// Returns an error if no range starts at `address`.
    pub fn free_reservation(&mut self, address: u64) -> Result<()> {
        self.memory.free_range(address)
    }

    /// Maps `size` bytes of the physical memory `physical`, from `offset`, at `address` in a reserved range.
    ///
    /// No device can access the mapping until [`set_access`](Self::set_access) allows it.
    ///
    /// # Errors
    ///
    /// Returns an error if `address` is not in a reserved range, the mapping overlaps another or runs past its range or its physical memory.
    pub fn map(&mut self, address: u64, size: u64, physical: u64, offset: u64) -> Result<()> {
        self.memory.map(address, size, physical, offset)
    }

    /// Unmaps the mappings that start in `address..address + size`.
    ///
    /// # Errors
    ///
    /// Returns an error if the range is not inside a reserved range.
    pub fn unmap(&mut self, address: u64, size: u64) -> Result<()> {
        self.memory.unmap(address, size)
    }

    /// Sets whether kernels on `device` may access the mappings that start in `address..address + size`.
    ///
    /// # Errors
    ///
    /// Returns an error if `device` does not exist or the range is not inside a reserved range.
    pub fn set_access(
        &mut self,
        address: u64,
        size: u64,
        device: usize,
        allowed: bool,
    ) -> Result<()> {
        self.memory.set_access(address, size, device, allowed)
    }

    /// Returns whether kernels on `device` may access the mapping that contains `address`.
    ///
    /// # Errors
    ///
    /// Returns an error if nothing is mapped at `address`.
    pub fn has_access(&self, address: u64, device: usize) -> Result<bool> {
        self.memory.has_access(address, device)
    }

    /// Creates a multicast object of `size` bytes for `devices`, which must share a node, and returns its handle.
    ///
    /// # Errors
    ///
    /// Returns an error if `devices` is empty or spans nodes.
    pub fn create_multicast(&mut self, devices: &[usize], size: u64) -> Result<u64> {
        for &device in devices {
            self.check_device(device)?;
        }
        self.memory.create_multicast(devices, size)
    }

    /// Binds `size` bytes of `device`'s physical memory `physical`, from `offset`, to the multicast object `multicast` from byte `start`.
    ///
    /// A `multimem` access reaches every device of the object, so each must have memory bound where the access lands.
    ///
    /// # Errors
    ///
    /// Returns an error if the object or the physical memory does not exist, the memory belongs to another device, the device does not take part, the binding runs past the memory or the object, or the device is already bound there.
    pub fn bind_multicast(
        &mut self,
        multicast: u64,
        start: u64,
        device: usize,
        physical: u64,
        offset: u64,
        size: u64,
    ) -> Result<()> {
        self.memory
            .bind_multicast(multicast, start, device, physical, offset, size)
    }

    /// Removes `device`'s binding that starts at byte `start` of the multicast object `multicast`.
    ///
    /// # Errors
    ///
    /// Returns an error if the object does not exist or the device has no binding there.
    pub fn unbind_multicast(&mut self, multicast: u64, start: u64, device: usize) -> Result<()> {
        self.memory.unbind_multicast(multicast, start, device)
    }

    /// Maps the multicast object `multicast` at a new address and returns it.
    ///
    /// Only `multimem` instructions access multicast addresses.
    ///
    /// # Errors
    ///
    /// Returns an error if the object does not exist or the address space is exhausted.
    pub fn map_multicast(&mut self, multicast: u64) -> Result<u64> {
        self.memory.map_multicast(multicast)
    }

    /// Destroys a multicast object; accesses through its mappings fault afterwards.
    pub fn destroy_multicast(&mut self, multicast: u64) {
        self.memory.destroy_multicast(multicast);
    }

    /// Lets kernels on `accessor` access memory allocated on `owner`.
    ///
    /// # Errors
    ///
    /// Returns an error if the devices are not peers in one node.
    pub fn enable_peer_access(&mut self, accessor: usize, owner: usize) -> Result<()> {
        self.memory.enable_peer_access(accessor, owner)
    }

    /// Stops kernels on `accessor` from accessing memory allocated on `owner`.
    pub fn disable_peer_access(&mut self, accessor: usize, owner: usize) {
        self.memory.disable_peer_access(accessor, owner);
    }

    fn check_device(&self, device: usize) -> Result<()> {
        if device < self.topology.device_count() {
            Ok(())
        } else {
            Err(ExecutionError::Memory(format!("no device {device}")))
        }
    }
}

/// Admits `task` as a command on `stream`, polled with the machine's memory until it returns a result.
///
/// When `faults` is set, an error the task returns faults `device`, as a kernel fault does.
/// Otherwise the task is not tied to `device`: it runs after `device` faulted.
pub(crate) fn submit_host_task<T, F>(
    client: &Arc<Client>,
    stream: u64,
    device: usize,
    faults: bool,
    mut task: F,
) -> result::Result<Submission<T>, SubmitError>
where
    T: Send + 'static,
    F: FnMut(&mut HostMemory<'_>) -> Poll<Result<T>> + Send + 'static,
{
    submit_poll(
        client,
        stream,
        device,
        CommandKind::Transfer,
        !faults,
        0,
        move |_| {
            move |worker: &mut Worker, terminal| {
                if let Some(error) = terminal {
                    return Poll::Ready(Err(error));
                }

                let topology = worker.shared.options.topology;
                let mut memory = HostMemory {
                    memory: &mut worker.memory,
                    topology,
                    failures: &worker.failures,
                };
                match task(&mut memory) {
                    Poll::Pending => Poll::Pending,
                    Poll::Ready(Err(error)) => {
                        if faults {
                            worker.fault(device, &error, None);
                        }
                        Poll::Ready(Err(error))
                    }
                    Poll::Ready(Ok(value)) => {
                        Poll::Ready(Ok(Box::new(move |_| value) as Pending<T>))
                    }
                }
            }
        },
    )
}

type AgentPoll = Box<dyn FnMut(&mut HostMemory<'_>) -> Result<()> + Send>;

pub(crate) struct AgentEntry {
    id: usize,
    poll: AgentPoll,
}

/// Host work that runs beside kernels for as long as this handle lives, such as a proxy thread serving queues that kernels fill.
///
/// The worker polls every agent once before each scheduler step while commands are running.
/// An agent that returns an error faults every device.
/// Agents are not polled once every device has faulted.
#[must_use = "dropping the handle removes the agent"]
pub struct Agent {
    shared: Weak<Shared>,
    id: usize,
}

impl Debug for Agent {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Agent")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl Drop for Agent {
    fn drop(&mut self) {
        if let Some(shared) = self.shared.upgrade() {
            let mut agents = lock(&shared.agents);
            agents.retain(|agent| agent.id != self.id);
            shared.agent_count.store(agents.len(), Ordering::Release);
        }
    }
}

impl Emulator {
    /// Starts polling `agent` on the worker until the returned handle drops.
    pub fn add_agent<F>(&self, agent: F) -> Agent
    where
        F: FnMut(&mut HostMemory<'_>) -> Result<()> + Send + 'static,
    {
        let shared = &self.client.shared;
        let id = shared.next_agent.fetch_add(1, Ordering::Relaxed);
        let mut agents = lock(&shared.agents);
        agents.push(AgentEntry {
            id,
            poll: Box::new(agent),
        });
        shared.agent_count.store(agents.len(), Ordering::Release);
        drop(agents);
        shared.changed.notify_one();
        Agent {
            shared: Arc::downgrade(shared),
            id,
        }
    }
}

/// Polls every agent once; the first error faults every device.
pub(crate) fn poll_agents(worker: &mut Worker) {
    if worker.shared.agent_count.load(Ordering::Acquire) == 0 || worker.is_all_faulted() {
        return;
    }

    let shared = Arc::clone(&worker.shared);
    let mut agents = lock(&shared.agents);
    let topology = shared.options.topology;
    let mut failure = None;
    for agent in agents.iter_mut() {
        let mut memory = HostMemory {
            memory: &mut worker.memory,
            topology,
            failures: &worker.failures,
        };
        if let Err(error) = (agent.poll)(&mut memory) {
            failure = Some(error);
            break;
        }
    }

    drop(agents);
    if let Some(error) = failure {
        worker.fault_all(&error);
    }
}
