use std::{
    collections::{BTreeMap, HashSet},
    ops::Range,
    result::Result as StdResult,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use crate::{
    ast::{AddressSize, StateSpace},
    emulator::{
        budget::{Budget, Reservation, allocation},
        execution_error::ExecutionError,
    },
};

type Result<T> = StdResult<T, ExecutionError>;

/// Bits below the window tag in a canonical address.
const WINDOW_BITS: u32 = 60;
/// Size of one canonical state-space window.
pub const WINDOW_BYTES: u64 = 1 << WINDOW_BITS;
/// Bits below the window tag in a compact generic address.
const COMPACT_BITS: u32 = 28;
const COMPACT_WINDOW: u64 = 1 << COMPACT_BITS;
/// Global allocations start after the compact generic windows of the other spaces.
const GLOBAL_WIDE_START: u64 = 5 * COMPACT_WINDOW;
/// First offset handed out in every window, so that null is never a valid address.
const FIRST_OFFSET: u64 = 65536;
pub const CHUNK_BYTES: usize = 65536;

/// One canonical address window per addressable state space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Window {
    Global,
    Const,
    Shared,
    Local,
    Param,
}

impl Window {
    pub(crate) fn of(space: StateSpace) -> Self {
        match space {
            StateSpace::Const => Self::Const,
            StateSpace::Shared => Self::Shared,
            StateSpace::Local => Self::Local,
            StateSpace::Param => Self::Param,
            StateSpace::Global | StateSpace::Reg | StateSpace::SReg | StateSpace::Tex => {
                Self::Global
            }
        }
    }

    fn index(self) -> usize {
        self as usize
    }

    fn space_name(self) -> &'static str {
        match self {
            Self::Global => "global",
            Self::Const => "const",
            Self::Shared => "shared",
            Self::Local => "local",
            Self::Param => "param",
        }
    }

    fn tag(self) -> u64 {
        self as u64
    }

    /// Returns the first canonical address of this window.
    pub(crate) fn base(self) -> u64 {
        self.tag() << WINDOW_BITS
    }

    /// Returns the exclusive upper bound of offsets inside this window.
    /// Per-CTA and per-lane spaces stay addressable through 32-bit state-space pointers.
    fn limit(self) -> u64 {
        match self {
            Self::Global | Self::Const => WINDOW_BYTES,
            Self::Shared | Self::Local | Self::Param => 1 << 32,
        }
    }

    /// Returns whether released addresses in this window may be reused after it wraps.
    /// Device-lifetime spaces never reuse addresses.
    fn is_transient(self) -> bool {
        matches!(self, Self::Shared | Self::Local | Self::Param)
    }
}

// Small state-space addresses have one generic representation in both u32 and u64.
// Global allocations skip these windows; large state-space offsets use the wide windows.
pub fn generic_address(address: u64) -> u64 {
    let tag = address >> WINDOW_BITS;
    let offset = specific(address);
    if (1..=4).contains(&tag) && offset < COMPACT_WINDOW {
        (tag << COMPACT_BITS) | offset
    } else {
        address
    }
}

pub fn canonical_address(address: u64) -> u64 {
    let tag = address >> COMPACT_BITS;
    if (1..=4).contains(&tag) {
        (tag << WINDOW_BITS) | (address & (COMPACT_WINDOW - 1))
    } else {
        address
    }
}

pub fn specific(address: u64) -> u64 {
    address & (WINDOW_BYTES - 1)
}

pub fn validate_address_size(address: u64, size: usize, address_size: AddressSize) -> Result<()> {
    if address_size == AddressSize::Bits32 {
        let end = address
            .checked_add(size.saturating_sub(1) as u64)
            .ok_or_else(|| ExecutionError::Memory("address range overflow".into()))?;
        if u32::try_from(generic_address(address)).is_err()
            || u32::try_from(generic_address(end)).is_err()
        {
            return Err(ExecutionError::Memory(
                "allocation exceeds 32-bit module address range".into(),
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Actor {
    pub grid: GridId,
    pub cta: usize,
    pub lane: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct GridId(u64);

impl GridId {
    pub fn sequence(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Owner {
    Device,
    Cta(GridId, usize),
    Lane(GridId, usize),
}

impl Owner {
    fn grid(self) -> Option<GridId> {
        match self {
            Self::Device => None,
            Self::Cta(grid, _) | Self::Lane(grid, _) => Some(grid),
        }
    }
}

/// One bit per allocation byte, set once the byte holds a defined value.
#[derive(Debug, Default)]
struct DefinedBytes {
    words: Vec<u64>,
    len: usize,
}

impl DefinedBytes {
    const BITS: usize = u64::BITS as usize;

    fn try_reserve_exact(&mut self, len: usize) -> Result<()> {
        self.words
            .try_reserve_exact(len.div_ceil(Self::BITS))
            .map_err(|_| ExecutionError::MemoryLimit)
    }

    fn get(&self, index: usize) -> bool {
        self.words[index / Self::BITS] & (1 << (index % Self::BITS)) != 0
    }

    fn set(&mut self, index: usize, value: bool) {
        let bit = 1 << (index % Self::BITS);
        let word = &mut self.words[index / Self::BITS];
        if value {
            *word |= bit;
        } else {
            *word &= !bit;
        }
    }

    /// Grows to `len` bytes, marking the new bytes as `value`.
    fn resize(&mut self, len: usize, value: bool) {
        let start = self.len;
        self.words.resize(len.div_ceil(Self::BITS), 0);
        self.len = len;
        if value {
            self.fill(start..len, true);
        }
    }

    fn fill(&mut self, range: Range<usize>, value: bool) {
        for index in range {
            self.set(index, value);
        }
    }

    fn all(&self, range: Range<usize>) -> bool {
        range.into_iter().all(|index| self.get(index))
    }

    fn extract(&self, range: Range<usize>) -> Vec<bool> {
        range.into_iter().map(|index| self.get(index)).collect()
    }

    fn assign(&mut self, offset: usize, values: &[bool]) {
        for (index, value) in values.iter().enumerate() {
            self.set(offset + index, *value);
        }
    }
}

pub struct Allocation {
    bytes: Vec<u8>,
    initialized: DefinedBytes,
    writable: bool,
    space: StateSpace,
    owner: Owner,
    charge: usize,
    size: usize,
}

pub struct Memory {
    allocations: BTreeMap<u64, Allocation>,
    /// Next free offset of each window.
    windows: [u64; 5],
    used: usize,
    limit: usize,
    next_grid: u64,
    grids: BTreeMap<GridId, GridStorage>,
    execution_budget: Arc<Budget>,
}

#[derive(Default)]
struct GridStorage {
    charge: usize,
    allocation_credit: usize,
    allocations: HashSet<u64>,
}

impl Memory {
    pub(crate) fn new(limit: usize, execution_budget: Arc<Budget>) -> Self {
        Self {
            allocations: BTreeMap::new(),
            windows: [FIRST_OFFSET; 5],
            used: 0,
            limit,
            next_grid: 0,
            grids: BTreeMap::new(),
            execution_budget,
        }
    }

    pub(crate) fn begin_grid(&mut self) -> Result<GridId> {
        let id = GridId(self.next_grid);
        self.next_grid = self
            .next_grid
            .checked_add(1)
            .ok_or(ExecutionError::ExecutionStateLimit)?;
        self.grids.insert(id, GridStorage::default());
        if let Err(error) = self.reserve_execution(id, allocation::<(GridId, GridStorage)>()) {
            self.grids.remove(&id);
            return Err(error);
        }
        Ok(id)
    }

    pub(crate) fn reserve_execution(&mut self, grid: GridId, charge: usize) -> Result<()> {
        let current = self
            .grids
            .get_mut(&grid)
            .ok_or_else(|| ExecutionError::Memory("retired grid".into()))?;
        current.charge += self.execution_budget.reserve(charge)?.commit();
        Ok(())
    }

    /// Precharges all initial allocations before construction can yield to another grid.
    pub(crate) fn reserve_grid_allocations(&mut self, grid: GridId, charge: usize) -> Result<()> {
        self.reserve_execution(grid, charge)?;
        if let Some(storage) = self.grids.get_mut(&grid) {
            storage.allocation_credit += charge;
        }
        Ok(())
    }

    pub(crate) fn release_execution(&mut self, grid: GridId, charge: usize) {
        if let Some(current) = self.grids.get_mut(&grid) {
            current.charge -= charge;
            self.execution_budget.release(charge);
        }
    }

    /// Moves a grid charge to a retained result without releasing or reserving bytes.
    pub(crate) fn retain_execution(&mut self, grid: GridId, charge: usize) -> Reservation {
        let storage = self.grids.get_mut(&grid).expect("live execution grid");
        assert!(charge <= storage.charge);
        storage.charge -= charge;
        Reservation::from_charged(Arc::clone(&self.execution_budget), charge)
    }

    pub(crate) fn release_grid(&mut self, grid: GridId) {
        if let Some(storage) = self.grids.remove(&grid) {
            for address in storage.allocations {
                let allocation = self
                    .allocations
                    .remove(&address)
                    .expect("indexed grid allocation");
                debug_assert_eq!(allocation.owner.grid(), Some(grid));
            }
            self.execution_budget.release(storage.charge);
        }
    }

    pub(crate) fn charge(size: usize) -> Option<usize> {
        // Bytes plus one definedness bit per byte, rounded up to whole words.
        size.checked_add(size.div_ceil(64) * 8)?
            .checked_add(allocation::<Allocation>())
    }

    pub(crate) fn allocate(
        &mut self,
        size: usize,
        alignment: u32,
        space: StateSpace,
        owner: Owner,
        initialized: bool,
        cancellation: &AtomicBool,
    ) -> Result<u64> {
        if cancellation.load(Ordering::Acquire) {
            return Err(ExecutionError::Cancelled);
        }
        let address = self.reserve(size, alignment, space, owner)?;
        loop {
            if cancellation.load(Ordering::Acquire) {
                self.release(address);
                return Err(ExecutionError::Cancelled);
            }
            if self.initialize_chunk(address, initialized)? {
                return Ok(address);
            }
        }
    }

    /// Chooses a window offset at or after `start` for a new allocation.
    fn place(window: Window, start: u64, size: usize, alignment: u32) -> Option<(u64, u64)> {
        let mask = u64::from(alignment) - 1;
        let align = |offset: u64| offset.checked_add(mask).map(|value| value & !mask);
        let size = (size as u64).max(1);
        let mut offset = align(start)?;
        // Keep pointer arithmetic, including the end pointer, in one generic representation.
        if offset < COMPACT_WINDOW
            && offset
                .checked_add(size)
                .is_none_or(|end| end >= COMPACT_WINDOW)
        {
            offset = align(if window == Window::Global {
                GLOBAL_WIDE_START
            } else {
                COMPACT_WINDOW
            })?;
        } else if window == Window::Global && offset < GLOBAL_WIDE_START {
            offset = align(GLOBAL_WIDE_START)?;
        }
        let end = offset
            .checked_add(size)
            .filter(|end| *end < window.limit())?;
        Some((offset, end))
    }

    /// Chooses the next free range in a window, scanning upward from the cursor.
    ///
    /// Addresses are not reused until the cursor reaches the end of a transient window, so
    /// stale pointers from earlier launches fault. The scan then wraps once, skipping live
    /// allocations. Device-lifetime windows never wrap.
    fn next_range(&self, window: Window, size: usize, alignment: u32) -> Result<(u64, u64)> {
        let starts = [self.windows[window.index()], FIRST_OFFSET];
        let passes = if window.is_transient() { 2 } else { 1 };
        for mut start in starts.into_iter().take(passes) {
            while let Some((offset, end)) = Self::place(window, start, size, alignment) {
                let base = window.base();
                let conflict = self
                    .allocations
                    .range(base..base + end)
                    .next_back()
                    .map(|(&address, allocation)| address - base + (allocation.size as u64).max(1))
                    .filter(|&conflict_end| conflict_end > offset);
                match conflict {
                    Some(conflict_end) => start = conflict_end,
                    None => return Ok((offset, end)),
                }
            }
        }
        Err(ExecutionError::AddressSpaceExhausted(window.space_name()))
    }
    /// Reserves an allocation of `size` bytes in `space` and returns its address.
    ///
    /// The allocation stays pending, and every access to it faults, until
    /// [`initialize_chunk`](Self::initialize_chunk) reports it ready.
    pub(crate) fn reserve(
        &mut self,
        size: usize,
        alignment: u32,
        space: StateSpace,
        owner: Owner,
    ) -> Result<u64> {
        let charge = Self::charge(size).ok_or(ExecutionError::MemoryLimit)?;
        let reservation = if let Some(grid) = owner.grid() {
            let storage = self
                .grids
                .get(&grid)
                .ok_or_else(|| ExecutionError::Memory("retired grid".into()))?;
            if storage.allocation_credit >= charge {
                None
            } else {
                Some(self.execution_budget.reserve(charge)?)
            }
        } else {
            self.used
                .checked_add(charge)
                .filter(|value| *value <= self.limit)
                .ok_or(ExecutionError::MemoryLimit)?;
            None
        };
        let window = Window::of(space);
        let (offset, end) = self.next_range(window, size, alignment)?;
        let address = window.base() + offset;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(size)
            .map_err(|_| ExecutionError::MemoryLimit)?;
        let mut initialized = DefinedBytes::default();
        initialized.try_reserve_exact(size)?;
        if let Some(grid) = owner.grid() {
            let current = self
                .grids
                .get_mut(&grid)
                .ok_or_else(|| ExecutionError::Memory("retired grid".into()))?;
            current
                .allocations
                .try_reserve(1)
                .map_err(|_| ExecutionError::ExecutionStateLimit)?;
            if let Some(reservation) = reservation {
                current.charge += reservation.commit();
            } else {
                current.allocation_credit -= charge;
            }
            current.allocations.insert(address);
        } else {
            self.used += charge;
        }
        self.allocations.insert(
            address,
            Allocation {
                bytes,
                initialized,
                writable: space != StateSpace::Const,
                space,
                owner,
                charge,
                size,
            },
        );
        self.windows[window.index()] = end;
        Ok(address)
    }

    /// Initializes at most one scheduler chunk and reports whether storage is ready.
    pub(crate) fn initialize_chunk(&mut self, address: u64, initialized: bool) -> Result<bool> {
        let allocation = self
            .allocations
            .get_mut(&address)
            .ok_or_else(|| ExecutionError::Memory("missing allocation reservation".into()))?;
        let end = allocation
            .bytes
            .len()
            .saturating_add(CHUNK_BYTES)
            .min(allocation.size);
        allocation.bytes.resize(end, 0);
        allocation.initialized.resize(end, initialized);
        Ok(end == allocation.size)
    }

    pub(crate) fn protect(&mut self, address: u64) {
        if let Some(allocation) = self.allocations.get_mut(&address) {
            allocation.writable = false;
        }
    }

    pub(crate) fn release(&mut self, address: u64) {
        let Some(allocation) = self.allocations.remove(&address) else {
            return;
        };
        if let Some(grid) = allocation.owner.grid() {
            self.grids
                .get_mut(&grid)
                .expect("live allocation grid")
                .allocations
                .remove(&address);
            self.release_execution(grid, allocation.charge);
        } else {
            self.used -= allocation.charge;
        }
    }

    fn locate(
        &self,
        address: u64,
        size: usize,
        alignment: usize,
        space: Option<StateSpace>,
        actor: Option<Actor>,
    ) -> Result<(u64, usize)> {
        if !address.is_multiple_of(alignment as u64) {
            return Err(ExecutionError::Memory("misaligned address".into()));
        }
        let (&base, allocation) = self
            .allocations
            .range(..=address)
            .next_back()
            .ok_or_else(|| ExecutionError::Memory("unmapped address".into()))?;
        let offset = usize::try_from(address - base)
            .map_err(|_| ExecutionError::Memory("address overflow".into()))?;
        if allocation.bytes.len() != allocation.size {
            return Err(ExecutionError::Memory("allocation is initializing".into()));
        }
        if offset
            .checked_add(size)
            .is_none_or(|end| end > allocation.bytes.len())
        {
            return Err(ExecutionError::Memory(
                "allocation bounds exceeded or stale address".into(),
            ));
        }
        if space.is_some_and(|s| s != allocation.space) {
            return Err(ExecutionError::Memory("wrong address space".into()));
        }
        if let Some(actor) = actor {
            let visible = match allocation.owner {
                Owner::Device => true,
                Owner::Cta(grid, id) => grid == actor.grid && id == actor.cta,
                Owner::Lane(grid, id) => grid == actor.grid && id == actor.lane,
            };
            if !visible {
                return Err(ExecutionError::Memory(
                    "storage belongs to another lane or CTA".into(),
                ));
            }
        }
        Ok((base, offset))
    }

    pub fn read(
        &self,
        address: u64,
        size: usize,
        alignment: usize,
        space: Option<StateSpace>,
        actor: Option<Actor>,
    ) -> Result<&[u8]> {
        let (base, offset) = self.locate(address, size, alignment, space, actor)?;
        let allocation = &self.allocations[&base];
        if !allocation.initialized.all(offset..offset + size) {
            return Err(ExecutionError::Indeterminate("memory bytes".into()));
        }
        Ok(&allocation.bytes[offset..offset + size])
    }

    pub fn validate_write(
        &self,
        address: u64,
        size: usize,
        alignment: usize,
        space: Option<StateSpace>,
        actor: Option<Actor>,
    ) -> Result<()> {
        let (base, _) = self.locate(address, size, alignment, space, actor)?;
        if !self.allocations[&base].writable {
            return Err(ExecutionError::Memory("read-only storage".into()));
        }
        Ok(())
    }

    pub fn write(
        &mut self,
        address: u64,
        bytes: &[u8],
        alignment: usize,
        space: Option<StateSpace>,
        actor: Option<Actor>,
    ) -> Result<()> {
        let (base, offset) = self.locate(address, bytes.len(), alignment, space, actor)?;
        let allocation = self
            .allocations
            .get_mut(&base)
            .ok_or_else(|| ExecutionError::Memory("unmapped allocation".into()))?;
        if !allocation.writable {
            return Err(ExecutionError::Memory("read-only storage".into()));
        }
        allocation.bytes[offset..offset + bytes.len()].copy_from_slice(bytes);
        allocation
            .initialized
            .fill(offset..offset + bytes.len(), true);
        Ok(())
    }

    /// Copies `size` bytes between two `.param` objects, as a call or return does.
    ///
    /// Undefined bytes, such as padding, stay undefined in the destination.
    pub fn copy_parameter(
        &mut self,
        source: u64,
        destination: u64,
        size: usize,
        actor: Actor,
        cancellation: &AtomicBool,
    ) -> Result<()> {
        let (source_base, source_offset) =
            self.locate(source, size, 1, Some(StateSpace::Param), Some(actor))?;
        let (destination_base, destination_offset) =
            self.locate(destination, size, 1, Some(StateSpace::Param), Some(actor))?;
        if !self.allocations[&destination_base].writable {
            return Err(ExecutionError::Memory("read-only parameter storage".into()));
        }
        for offset in (0..size).step_by(CHUNK_BYTES) {
            if cancellation.load(Ordering::Acquire) {
                return Err(ExecutionError::Cancelled);
            }
            let length = (size - offset).min(CHUNK_BYTES);
            let source = &self.allocations[&source_base];
            let range = source_offset + offset..source_offset + offset + length;
            let bytes = source.bytes[range.clone()].to_vec();
            let initialized = source.initialized.extract(range);
            let destination = self
                .allocations
                .get_mut(&destination_base)
                .ok_or_else(|| ExecutionError::Memory("missing parameter allocation".into()))?;
            let range = destination_offset + offset..destination_offset + offset + length;
            destination.bytes[range.clone()].copy_from_slice(&bytes);
            destination.initialized.assign(range.start, &initialized);
        }
        Ok(())
    }

    pub fn initialize(&mut self, base: u64, offset: usize, bytes: &[u8]) -> Result<()> {
        let allocation = self
            .allocations
            .get_mut(&base)
            .ok_or_else(|| ExecutionError::Memory("missing initializer allocation".into()))?;
        let end = offset
            .checked_add(bytes.len())
            .filter(|e| *e <= allocation.bytes.len())
            .ok_or_else(|| ExecutionError::Memory("initializer bounds".into()))?;
        allocation.bytes[offset..end].copy_from_slice(bytes);
        allocation.initialized.fill(offset..end, true);
        Ok(())
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        for storage in self.grids.values() {
            self.execution_budget.release(storage.charge);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_retirement_preserves_other_grids_and_persistent_buffers_after_early_releases() {
        for order in [[1, 2, 0], [0, 1, 2], [2, 0, 1]] {
            let budget = Arc::new(Budget::new(1 << 20));
            let mut memory = Memory::new(1 << 20, Arc::clone(&budget));
            let cancellation = AtomicBool::new(false);
            let persistent: Vec<_> = (0..128)
                .map(|_| {
                    memory
                        .allocate(4, 4, StateSpace::Global, Owner::Device, true, &cancellation)
                        .unwrap()
                })
                .collect();
            let device_charge = memory.used;
            let first = memory.begin_grid().unwrap();
            let second = memory.begin_grid().unwrap();
            let other = memory
                .allocate(
                    4,
                    4,
                    StateSpace::Shared,
                    Owner::Cta(second, 0),
                    true,
                    &cancellation,
                )
                .unwrap();
            memory.write(other, &[7; 4], 4, None, None).unwrap();
            let owned: Vec<_> = (0..3)
                .map(|lane| {
                    memory
                        .allocate(
                            4,
                            4,
                            StateSpace::Local,
                            Owner::Lane(first, lane),
                            true,
                            &cancellation,
                        )
                        .unwrap()
                })
                .collect();
            for index in order.into_iter().take(2) {
                let before = budget.used();
                memory.release(owned[index]);
                assert_eq!(budget.used(), before - Memory::charge(4).unwrap());
                assert!(memory.read(owned[index], 4, 4, None, None).is_err());
            }
            let remaining = memory
                .allocate(
                    8,
                    4,
                    StateSpace::Shared,
                    Owner::Cta(first, 0),
                    true,
                    &cancellation,
                )
                .unwrap();
            let other_charge = memory.grids[&second].charge;
            memory.release_grid(first);
            assert_eq!(budget.used(), other_charge);
            for address in owned {
                assert!(memory.read(address, 4, 4, None, None).is_err());
            }
            assert!(memory.read(remaining, 8, 4, None, None).is_err());
            assert_eq!(memory.read(other, 4, 4, None, None).unwrap(), [7; 4]);
            for address in persistent {
                assert_eq!(memory.read(address, 4, 4, None, None).unwrap(), [0; 4]);
            }
            assert_eq!(memory.used, device_charge);
            memory.release_grid(first);
            memory.release(remaining);
            assert_eq!(budget.used(), other_charge);
            memory.release_grid(second);
            assert_eq!(budget.used(), 0);
            assert_eq!(memory.allocations.len(), 128);
        }
    }

    #[test]
    fn test_failed_grid_reservations_leave_ownership_and_precharged_storage_unchanged() {
        let budget = Arc::new(Budget::new(4096));
        let mut memory = Memory::new(4096, Arc::clone(&budget));
        let grid = memory.begin_grid().unwrap();
        let first = memory
            .reserve(4, 4, StateSpace::Local, Owner::Lane(grid, 0))
            .unwrap();
        let baseline = budget.used();
        let full = budget.reserve(4096 - baseline).unwrap();
        assert!(matches!(
            memory.reserve(4, 4, StateSpace::Local, Owner::Lane(grid, 1)),
            Err(ExecutionError::ExecutionStateLimit)
        ));
        drop(full);
        assert_eq!(budget.used(), baseline);
        assert_eq!(memory.allocations.len(), 1);
        assert_eq!(memory.grids[&grid].allocations, HashSet::from([first]));

        let charge = Memory::charge(4).unwrap();
        memory.reserve_grid_allocations(grid, charge).unwrap();
        let precharged = budget.used();
        // Occupy the rest of the local window so the reservation cannot be placed.
        let blocker = first + 4;
        memory.allocations.insert(
            blocker,
            Allocation {
                bytes: Vec::new(),
                initialized: DefinedBytes::default(),
                writable: true,
                space: StateSpace::Local,
                owner: Owner::Device,
                charge: 0,
                size: usize::try_from((1 << 32) - specific(blocker)).unwrap(),
            },
        );
        let next = memory.windows[3];
        memory.windows[3] = specific(blocker);
        assert!(matches!(
            memory.reserve(4, 4, StateSpace::Local, Owner::Lane(grid, 1)),
            Err(ExecutionError::AddressSpaceExhausted("local"))
        ));
        assert_eq!(budget.used(), precharged);
        assert_eq!(memory.grids[&grid].allocation_credit, charge);
        assert_eq!(memory.grids[&grid].allocations, HashSet::from([first]));
        memory.allocations.remove(&blocker);
        memory.windows[3] = next;
        let second = memory
            .reserve(4, 4, StateSpace::Local, Owner::Lane(grid, 1))
            .unwrap();
        assert!(second > first);
        assert_eq!(budget.used(), precharged);
        assert_eq!(memory.grids[&grid].allocation_credit, 0);
        memory.release_grid(grid);
        assert_eq!(budget.used(), 0);
        assert!(memory.allocations.is_empty());
    }

    #[test]
    fn test_failed_initialization_and_cancellation_release_partial_grid_ownership() {
        let budget = Arc::new(Budget::new(1 << 20));
        let mut memory = Memory::new(4096, Arc::clone(&budget));
        let grid = memory.begin_grid().unwrap();
        let size = CHUNK_BYTES + 1;
        let address = memory
            .reserve(size, 4, StateSpace::Local, Owner::Lane(grid, 0))
            .unwrap();
        assert!(!memory.initialize_chunk(address, false).unwrap());
        assert!(memory.initialize(address, size, &[1]).is_err());
        memory.release(address);
        let baseline = budget.used();
        let cancelled = AtomicBool::new(true);
        assert!(matches!(
            memory.allocate(
                4,
                4,
                StateSpace::Local,
                Owner::Lane(grid, 1),
                false,
                &cancelled
            ),
            Err(ExecutionError::Cancelled)
        ));
        assert_eq!(budget.used(), baseline);
        let pending = memory
            .reserve(size, 4, StateSpace::Shared, Owner::Cta(grid, 0))
            .unwrap();
        assert!(!memory.initialize_chunk(pending, false).unwrap());
        memory.release_grid(grid);
        assert_eq!(budget.used(), 0);
        assert!(memory.allocations.is_empty());
    }

    #[test]
    fn test_allocations_and_end_pointers_stay_in_one_generic_address_window() {
        for space in [
            StateSpace::Global,
            StateSpace::Const,
            StateSpace::Shared,
            StateSpace::Local,
            StateSpace::Param,
        ] {
            for size in [0, 1, 2, 4] {
                for alignment in [1, 16] {
                    let mut memory = Memory::new(4096, Arc::new(Budget::new(4096)));
                    let slot = Window::of(space).index();
                    memory.windows[slot] = (1 << 28) - 2;
                    let address = memory
                        .allocate(
                            size,
                            alignment,
                            space,
                            Owner::Device,
                            true,
                            &AtomicBool::new(false),
                        )
                        .unwrap();
                    assert!(address.is_multiple_of(u64::from(alignment)));
                    for offset in 0..=size as u64 {
                        assert_eq!(
                            canonical_address(generic_address(address) + offset),
                            address + offset,
                            "{space:?}, size {size}, alignment {alignment}, offset {offset}"
                        );
                    }
                    assert_eq!(
                        memory.read(address, size, 1, Some(space), None).unwrap(),
                        vec![0; size]
                    );
                    validate_address_size(address, size, AddressSize::Bits64).unwrap();
                    let compact = space == StateSpace::Global || specific(address) < 1 << 28;
                    assert_eq!(
                        validate_address_size(address, size, AddressSize::Bits32).is_ok(),
                        compact
                    );
                    memory.release(address);
                    let next = memory
                        .allocate(
                            0,
                            alignment,
                            space,
                            Owner::Device,
                            true,
                            &AtomicBool::new(false),
                        )
                        .unwrap();
                    assert!(next > address);
                }
            }
        }
    }

    #[test]
    fn test_local_pointer_arithmetic_crosses_the_compact_boundary_only_in_wide_modules() {
        use crate::emulator::{EmulatorOptions, LaunchConfig, ParsedModule, engine::Engine};
        for bits in [32, 64] {
            let parsed = ParsedModule::parse(&format!(
                ".version 9.3 .target sm_80 .address_size {bits}
                .global .u32 output; .entry k() {{
                    .local .b8 scratch[4]; .reg .u{bits} a; .reg .u32 r;
                    mov.u{bits} a,scratch; cvta.local.u{bits} a,a;
                    add.u{bits} a,a,3; st.u8 [a],7;
                    ld.local.u8 r,[scratch+3]; st.global.u32 [output],r; ret;
                }}"
            ))
            .unwrap();
            let code = parsed.compile("k").unwrap();
            let options = EmulatorOptions::default();
            let budget = Arc::new(Budget::new(options.execution_storage_bytes));
            let mut memory = Memory::new(4096, Arc::clone(&budget));
            memory.windows[3] = (1 << 28) - 2;
            let flag = AtomicBool::new(false);
            let output = memory
                .allocate(4, 4, StateSpace::Global, Owner::Device, true, &flag)
                .unwrap();
            let config = LaunchConfig::create([1; 3], [1; 3], 0).unwrap();
            match Engine::create(&code, &[output], config, &[], &options, &flag, &mut memory) {
                Ok(mut engine) => {
                    assert_eq!(bits, 64);
                    while engine.step(&mut memory, &flag).unwrap().is_none() {}
                    assert_eq!(
                        memory.read(output, 4, 4, None, None).unwrap(),
                        7u32.to_le_bytes()
                    );
                    engine.release(&mut memory);
                }
                Err(error) => {
                    assert_eq!(bits, 32);
                    assert!(matches!(error, ExecutionError::Memory(_)));
                    assert_eq!(memory.read(output, 4, 4, None, None).unwrap(), [0; 4]);
                }
            }
            assert_eq!(budget.used(), 0);
            assert_eq!(memory.allocations.len(), 1);
        }
    }

    #[test]
    fn test_global_ranges_skip_compact_generic_windows() {
        let mut memory = Memory::new(4096, Arc::new(Budget::new(4096)));
        memory.windows[0] = (1 << 28) - 2;
        let cancellation = AtomicBool::new(false);
        let address = memory
            .allocate(4, 1, StateSpace::Global, Owner::Device, true, &cancellation)
            .unwrap();
        assert_eq!(address, 5 << 28);
        assert_eq!(canonical_address(address), address);
        assert_eq!(memory.read(address, 4, 1, None, None).unwrap(), [0; 4]);
        for space in [
            StateSpace::Const,
            StateSpace::Shared,
            StateSpace::Local,
            StateSpace::Param,
        ] {
            let address = Window::of(space).base() + FIRST_OFFSET;
            assert_eq!(canonical_address(generic_address(address)), address);
        }
    }

    #[test]
    fn test_live_grid_addresses_and_ownership_are_distinct() {
        let mut memory = Memory::new(4096, Arc::new(Budget::new(4096)));
        let flag = AtomicBool::new(false);
        let first = memory.begin_grid().unwrap();
        let second = memory.begin_grid().unwrap();
        let a = memory
            .allocate(4, 4, StateSpace::Shared, Owner::Cta(first, 0), true, &flag)
            .unwrap();
        let b = memory
            .allocate(4, 4, StateSpace::Shared, Owner::Cta(second, 0), true, &flag)
            .unwrap();
        assert_ne!(a, b);
        let other = Actor {
            grid: second,
            cta: 0,
            lane: 0,
        };
        assert!(matches!(
            memory.read(a, 4, 4, None, Some(other)),
            Err(ExecutionError::Memory(_))
        ));
        memory.release_grid(first);
        assert!(memory.read(a, 4, 4, None, None).is_err());
        let third = memory.begin_grid().unwrap();
        let c = memory
            .allocate(4, 4, StateSpace::Shared, Owner::Cta(third, 0), true, &flag)
            .unwrap();
        assert!(c > b);
        assert_eq!(memory.read(b, 4, 4, None, Some(other)).unwrap(), [0; 4]);
    }

    #[test]
    fn test_execution_storage_is_aggregate_and_released_with_the_grid() {
        let mut memory = Memory::new(4096, Arc::new(Budget::new(1024)));
        let flag = AtomicBool::new(false);
        let a = memory.begin_grid().unwrap();
        let b = memory.begin_grid().unwrap();
        memory.reserve_execution(a, 600).unwrap();
        assert!(matches!(
            memory.reserve_execution(b, 600),
            Err(ExecutionError::ExecutionStateLimit)
        ));
        memory.release_grid(a);
        memory.reserve_execution(b, 600).unwrap();
        memory
            .allocate(16, 8, StateSpace::Local, Owner::Lane(b, 0), false, &flag)
            .unwrap();
        memory.release_grid(b);
        assert_eq!(memory.execution_budget.used(), 0);
        assert!(memory.allocations.is_empty());
        assert!(memory.reserve_execution(a, 1).is_err());
    }

    #[test]
    fn test_pending_storage_reserves_capacity_and_rejects_access_until_initialized() {
        let mut memory = Memory::new(
            Memory::charge(CHUNK_BYTES + 1).unwrap(),
            Arc::new(Budget::new(4096)),
        );
        let address = memory
            .reserve(CHUNK_BYTES + 1, 4, StateSpace::Global, Owner::Device)
            .unwrap();
        assert!(
            memory
                .reserve(1, 1, StateSpace::Global, Owner::Device)
                .is_err()
        );
        assert!(!memory.initialize_chunk(address, true).unwrap());
        assert!(memory.read(address, 1, 1, None, None).is_err());
        assert!(memory.initialize_chunk(address, true).unwrap());
        assert_eq!(
            memory
                .read(address + CHUNK_BYTES as u64, 1, 1, None, None)
                .unwrap(),
            [0]
        );
        memory.release(address);
        assert_eq!(memory.used, 0);
    }

    #[test]
    fn test_transient_windows_wrap_around_live_ranges_and_device_windows_never_do() {
        let mut memory = Memory::new(1 << 20, Arc::new(Budget::new(1 << 30)));
        let flag = AtomicBool::new(false);
        let grid = memory.begin_grid().unwrap();
        let first = memory
            .reserve(4, 4, StateSpace::Local, Owner::Lane(grid, 0))
            .unwrap();
        memory.release_grid(grid);
        // Each round would exhaust a 4 GiB window within 64 rounds without restarting.
        let mut previous = first;
        let mut restarted = false;
        for _ in 0..256 {
            let grid = memory.begin_grid().unwrap();
            let addresses: Vec<_> = (0..4)
                .map(|lane| {
                    memory
                        .reserve(16 << 20, 16, StateSpace::Local, Owner::Lane(grid, lane))
                        .unwrap()
                })
                .collect();
            restarted |= addresses[0] < previous;
            previous = addresses[3];
            memory.release_grid(grid);
        }
        assert!(restarted);

        // Wrapping skips live ranges and fails only when no gap remains.
        let grid = memory.begin_grid().unwrap();
        memory.windows[Window::Local.index()] = FIRST_OFFSET;
        let low = memory
            .reserve(4, 4, StateSpace::Local, Owner::Lane(grid, 0))
            .unwrap();
        memory.windows[Window::Local.index()] = (1 << 32) - 8;
        let high = memory
            .reserve(4, 4, StateSpace::Local, Owner::Lane(grid, 1))
            .unwrap();
        let wrapped = memory
            .reserve(4, 4, StateSpace::Local, Owner::Lane(grid, 2))
            .unwrap();
        assert_eq!(specific(low), FIRST_OFFSET);
        assert_eq!(specific(high), (1 << 32) - 8);
        assert_eq!(wrapped, low + 4);
        memory.windows[Window::Local.index()] = (1 << 32) - 8;
        assert!(matches!(
            memory.next_range(
                Window::Local,
                usize::try_from((1 << 32) - FIRST_OFFSET).unwrap(),
                4
            ),
            Err(ExecutionError::AddressSpaceExhausted("local"))
        ));
        memory.release_grid(grid);

        memory.windows[Window::Global.index()] = WINDOW_BYTES - 8;
        let global = memory
            .allocate(4, 4, StateSpace::Global, Owner::Device, true, &flag)
            .unwrap();
        memory.release(global);
        assert!(matches!(
            memory.allocate(4, 4, StateSpace::Global, Owner::Device, true, &flag),
            Err(ExecutionError::AddressSpaceExhausted("global"))
        ));
    }
}
