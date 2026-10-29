//! Grid construction. The storage of the whole grid is reserved up front, then the grid
//! is built in bounded scheduler turns so other streams keep running.

use std::sync::atomic::AtomicBool;

use crate::{
    ast::StateSpace,
    emulator::{
        CompiledKernel,
        engine::{
            AsyncLane, Engine, Frame, Lane, Register, Result, Warp, cancel, frame_charge, product,
        },
        execution_error::ExecutionError,
        ir::FrameRole,
        layout::StorageLayout,
        memory::{CHUNK_BYTES, Memory, Owner, validate_address_size},
        runtime::{EmulatorOptions, LaunchConfig},
        target::{NAMED_BARRIERS, WARP_SIZE},
    },
};

const TABLE_CHUNK: usize = CHUNK_BYTES / size_of::<u64>();

#[derive(Clone, Copy)]
enum Phase {
    Cta,
    DynamicShared,
    Globals,
    Shared,
    Lane,
    Registers,
    Objects,
    FrameAllocations,
    Warps,
    Ready,
}

struct PendingAllocation {
    address: u64,
    argument: Option<usize>,
    initialized: bool,
    offset: usize,
}

pub struct GridBuilder {
    engine: Engine,
    storage: StoragePlan,
    dynamic_address: u64,
    phase: Phase,
    cta: usize,
    function: usize,
    object: usize,
    argument: usize,
    frame: Option<Frame>,
    pending: Option<PendingAllocation>,
    warp: usize,
    work_bytes: usize,
}

pub struct StoragePlan {
    pub(crate) charge: usize,
    dynamic_alignment: Option<u32>,
}

fn storage_size(layout: &StorageLayout) -> Result<usize> {
    usize::try_from(layout.byte_len()).map_err(|_| ExecutionError::ExecutionStateLimit)
}

fn storage_plan(code: &CompiledKernel, config: LaunchConfig) -> Result<StoragePlan> {
    let mut dynamic_alignment: Option<u32> = None;
    let mut shared = code
        .artifact
        .executable
        .module
        .variables
        .iter()
        .filter(|variable| variable.space == StateSpace::Shared)
        .map(|variable| (variable.layout, variable.dynamic))
        .chain(
            code.artifact
                .executable
                .functions
                .iter()
                .flat_map(|function| function.objects.iter())
                .filter(|object| object.space == StateSpace::Shared)
                .map(|object| (object.layout, object.dynamic)),
        )
        .try_fold(0usize, |sum, (layout, dynamic)| {
            if dynamic {
                dynamic_alignment = Some(dynamic_alignment.unwrap_or(1).max(layout.alignment()));
                Ok(sum)
            } else {
                sum.checked_add(
                    Memory::charge(storage_size(&layout)?)
                        .ok_or(ExecutionError::ExecutionStateLimit)?,
                )
                .ok_or(ExecutionError::ExecutionStateLimit)
            }
        })?;
    if dynamic_alignment.is_some() {
        shared = shared
            .checked_add(
                Memory::charge(config.shared_bytes).ok_or(ExecutionError::ExecutionStateLimit)?,
            )
            .ok_or(ExecutionError::ExecutionStateLimit)?;
    }
    let local = code
        .artifact
        .executable
        .entry_function()
        .objects
        .iter()
        .filter(|object| object.space != StateSpace::Shared)
        .try_fold(0usize, |sum, object| {
            sum.checked_add(
                Memory::charge(storage_size(&object.layout)?)
                    .ok_or(ExecutionError::ExecutionStateLimit)?,
            )
            .ok_or(ExecutionError::ExecutionStateLimit)
        })?;
    let charge = product(config.grid)?
        .checked_mul(
            shared
                .checked_add(
                    product(config.block)?
                        .checked_mul(local)
                        .ok_or(ExecutionError::ExecutionStateLimit)?,
                )
                .ok_or(ExecutionError::ExecutionStateLimit)?,
        )
        .ok_or(ExecutionError::ExecutionStateLimit)?;
    Ok(StoragePlan {
        charge,
        dynamic_alignment,
    })
}

impl GridBuilder {
    pub fn create(
        code: &CompiledKernel,
        global_count: usize,
        config: LaunchConfig,
        options: &EmulatorOptions,
        memory: &mut Memory,
    ) -> Result<Self> {
        let storage = storage_plan(code, config)?;
        let grid = memory.begin_grid()?;
        let engine =
            match Engine::reserve_grid(code, global_count, config, options, memory, grid, &storage)
            {
                Ok(engine) => engine,
                Err(error) => {
                    memory.release_grid(grid);
                    return Err(error);
                }
            };
        Ok(Self {
            engine,
            storage,
            dynamic_address: 0,
            phase: Phase::Cta,
            cta: 0,
            function: 0,
            object: 0,
            argument: 0,
            frame: None,
            pending: None,
            warp: 0,
            work_bytes: 0,
        })
    }

    pub fn release(&self, memory: &mut Memory) {
        self.engine.release(memory);
    }

    pub fn finish(self) -> Engine {
        self.engine
    }

    pub fn advance(
        &mut self,
        globals: &[u64],
        args: &[Vec<u8>],
        cancellation: &AtomicBool,
        memory: &mut Memory,
    ) -> Result<bool> {
        self.work_bytes = 0;
        for _ in 0..64 {
            if self.advance_one(globals, args, cancellation, memory)? {
                return Ok(true);
            }
            if self.work_bytes >= CHUNK_BYTES {
                break;
            }
        }
        Ok(false)
    }

    /// Performs one bounded unit of construction, returning whether the grid is ready.
    fn advance_one(
        &mut self,
        globals: &[u64],
        args: &[Vec<u8>],
        cancellation: &AtomicBool,
        memory: &mut Memory,
    ) -> Result<bool> {
        cancel(cancellation)?;
        if self.pending.is_some() {
            self.initialize_pending(args, memory)?;
            return Ok(false);
        }
        match self.phase {
            Phase::Cta => self.begin_cta(globals)?,
            Phase::DynamicShared => self.reserve_dynamic_shared(memory)?,
            Phase::Globals => self.bind_globals(globals, memory)?,
            Phase::Shared => self.bind_shared(memory)?,
            Phase::Lane => self.begin_lane()?,
            Phase::Registers => self.fill_registers()?,
            Phase::Objects => self.bind_frame_objects()?,
            Phase::FrameAllocations => self.allocate_frame_object(memory)?,
            Phase::Warps => self.build_warp()?,
            Phase::Ready => {}
        }
        Ok(matches!(self.phase, Phase::Ready))
    }

    /// Zero-initializes the pending allocation, then copies its kernel argument, one chunk
    /// per turn.
    fn initialize_pending(&mut self, args: &[Vec<u8>], memory: &mut Memory) -> Result<()> {
        let Some(pending) = &mut self.pending else {
            return Ok(());
        };
        if !pending.initialized {
            pending.initialized = memory.initialize_chunk(pending.address, false)?;
            self.work_bytes += CHUNK_BYTES;
            return Ok(());
        }
        if let Some(argument) = pending.argument {
            let value = &args[argument];
            let end = pending.offset.saturating_add(CHUNK_BYTES).min(value.len());
            memory.initialize(pending.address, pending.offset, &value[pending.offset..end])?;
            self.work_bytes += end - pending.offset;
            pending.offset = end;
            if end != value.len() {
                return Ok(());
            }
            memory.protect(pending.address);
        }
        self.pending = None;
        Ok(())
    }

    /// Reserves storage for one object and schedules its initialization.
    fn reserve(
        &mut self,
        layout: &StorageLayout,
        space: StateSpace,
        owner: Owner,
        argument: Option<usize>,
        memory: &mut Memory,
    ) -> Result<u64> {
        let size = storage_size(layout)?;
        self.reserve_bytes(size, layout.alignment(), space, owner, argument, memory)
    }

    fn reserve_bytes(
        &mut self,
        size: usize,
        alignment: u32,
        space: StateSpace,
        owner: Owner,
        argument: Option<usize>,
        memory: &mut Memory,
    ) -> Result<u64> {
        let address = memory.reserve(size, alignment, space, owner)?;
        validate_address_size(address, size, self.engine.code.module.address_size)?;
        self.pending = Some(PendingAllocation {
            address,
            argument,
            initialized: false,
            offset: 0,
        });
        Ok(address)
    }

    /// Starts the next CTA, or moves on to warp construction after the last one.
    fn begin_cta(&mut self, globals: &[u64]) -> Result<()> {
        let engine = &mut self.engine;
        if self.cta == product(engine.config.grid)? {
            self.cta = 0;
            self.phase = Phase::Warps;
            return Ok(());
        }
        let mut addresses = Vec::new();
        addresses
            .try_reserve_exact(globals.len())
            .map_err(|_| ExecutionError::ExecutionStateLimit)?;
        let mut functions = Vec::new();
        functions
            .try_reserve_exact(engine.code.functions.len())
            .map_err(|_| ExecutionError::ExecutionStateLimit)?;
        engine.globals.push(addresses);
        engine.shared.push(functions);
        engine.generations.push([0; NAMED_BARRIERS]);
        engine.remaining_lanes.push(product(engine.config.block)?);
        self.object = 0;
        self.dynamic_address = 0;
        self.phase = Phase::DynamicShared;
        Ok(())
    }

    /// Reserves the CTA's dynamic shared region, which every dynamic declaration aliases.
    fn reserve_dynamic_shared(&mut self, memory: &mut Memory) -> Result<()> {
        if let Some(alignment) = self.storage.dynamic_alignment {
            let owner = Owner::Cta(self.engine.grid, self.cta);
            let size = self.engine.config.shared_bytes;
            self.dynamic_address =
                self.reserve_bytes(size, alignment, StateSpace::Shared, owner, None, memory)?;
        }
        self.phase = Phase::Globals;
        Ok(())
    }

    /// Copies the module global addresses for this CTA, then binds its shared module
    /// variables one per turn.
    fn bind_globals(&mut self, globals: &[u64], memory: &mut Memory) -> Result<()> {
        let addresses = &mut self.engine.globals[self.cta];
        if addresses.len() < globals.len() {
            let end = addresses
                .len()
                .saturating_add(TABLE_CHUNK)
                .min(globals.len());
            self.work_bytes += (end - addresses.len()) * size_of::<u64>();
            addresses.extend_from_slice(&globals[addresses.len()..end]);
            return Ok(());
        }
        let variables = &self.engine.code.module.variables;
        if self.object == variables.len() {
            self.object = 0;
            self.function = 0;
            self.phase = Phase::Shared;
            return Ok(());
        }
        let id = self.object;
        self.object += 1;
        let variable = &variables[id];
        let address = if variable.dynamic {
            self.dynamic_address
        } else if variable.space == StateSpace::Shared {
            let layout = variable.layout;
            let owner = Owner::Cta(self.engine.grid, self.cta);
            self.reserve(&layout, StateSpace::Shared, owner, None, memory)?
        } else {
            return Ok(());
        };
        self.engine.globals[self.cta][id] = address;
        Ok(())
    }

    /// Builds each function's table of CTA-shared objects, one object per turn.
    fn bind_shared(&mut self, memory: &mut Memory) -> Result<()> {
        let engine = &mut self.engine;
        if self.function == engine.code.functions.len() {
            self.phase = Phase::Lane;
            return Ok(());
        }
        let function = &engine.code.functions[self.function];
        if engine.shared[self.cta].len() == self.function {
            let mut objects = Vec::new();
            objects
                .try_reserve_exact(function.objects.len())
                .map_err(|_| ExecutionError::ExecutionStateLimit)?;
            engine.shared[self.cta].push(objects);
        }
        let objects = &mut engine.shared[self.cta][self.function];
        if objects.len() < function.objects.len() {
            let end = objects
                .len()
                .saturating_add(TABLE_CHUNK)
                .min(function.objects.len());
            self.work_bytes += (end - objects.len()) * size_of::<u64>();
            objects.resize(end, 0);
            return Ok(());
        }
        if self.object == function.objects.len() {
            self.object = 0;
            self.function += 1;
            return Ok(());
        }
        let id = self.object;
        self.object += 1;
        let object = &function.objects[id];
        let address = if object.dynamic {
            self.dynamic_address
        } else if object.space == StateSpace::Shared {
            let layout = object.layout;
            let owner = Owner::Cta(engine.grid, self.cta);
            self.reserve(&layout, StateSpace::Shared, owner, None, memory)?
        } else {
            return Ok(());
        };
        self.engine.shared[self.cta][self.function][id] = address;
        Ok(())
    }

    /// Starts the next lane's entry frame, or the next CTA after its last lane.
    fn begin_lane(&mut self) -> Result<()> {
        let engine = &mut self.engine;
        let threads = product(engine.config.block)?;
        if engine.lanes.len() == (self.cta + 1) * threads {
            self.cta += 1;
            self.phase = Phase::Cta;
            return Ok(());
        }
        let function = engine.code.entry_function();
        let mut registers = Vec::new();
        registers
            .try_reserve_exact(function.registers.len())
            .map_err(|_| ExecutionError::ExecutionStateLimit)?;
        let mut objects = Vec::new();
        objects
            .try_reserve_exact(function.objects.len())
            .map_err(|_| ExecutionError::ExecutionStateLimit)?;
        let mut owned = Vec::new();
        owned
            .try_reserve_exact(function.objects.len())
            .map_err(|_| ExecutionError::ExecutionStateLimit)?;
        self.frame = Some(Frame {
            function: engine.code.entry,
            pc: 0,
            registers,
            objects,
            owned,
            returns: Vec::new(),
            charge: frame_charge(function)?,
        });
        self.object = 0;
        self.argument = 0;
        self.phase = Phase::Registers;
        Ok(())
    }

    fn frame(&mut self) -> Result<&mut Frame> {
        self.frame
            .as_mut()
            .ok_or_else(|| ExecutionError::Invalid("missing initial frame".into()))
    }

    fn fill_registers(&mut self) -> Result<()> {
        let function = self.frame()?.function;
        let count = self.engine.code.function(function).registers.len();
        let frame = self.frame()?;
        let end = frame
            .registers
            .len()
            .saturating_add(CHUNK_BYTES / size_of::<Register>())
            .min(count);
        let added = end - frame.registers.len();
        frame.registers.resize(end, Register::default());
        self.work_bytes += added * size_of::<Register>();
        if end == count {
            self.phase = Phase::Objects;
        }
        Ok(())
    }

    /// Copies the CTA-shared object addresses into the entry frame.
    fn bind_frame_objects(&mut self) -> Result<()> {
        let frame = self
            .frame
            .as_mut()
            .ok_or_else(|| ExecutionError::Invalid("missing initial frame".into()))?;
        let objects = &self.engine.shared[self.cta][frame.function.0];
        let end = frame
            .objects
            .len()
            .saturating_add(TABLE_CHUNK)
            .min(objects.len());
        self.work_bytes += (end - frame.objects.len()) * size_of::<u64>();
        frame
            .objects
            .extend_from_slice(&objects[frame.objects.len()..end]);
        if end == objects.len() {
            self.phase = Phase::FrameAllocations;
        }
        Ok(())
    }

    /// Allocates one lane-owned frame object per turn, then adds the finished lane.
    fn allocate_frame_object(&mut self, memory: &mut Memory) -> Result<()> {
        let function = self.frame()?.function;
        if self.object == self.engine.code.function(function).objects.len() {
            let frame = self
                .frame
                .take()
                .ok_or_else(|| ExecutionError::Invalid("missing initial frame".into()))?;
            let engine = &mut self.engine;
            let threads = product(engine.config.block)?;
            engine.lanes.push(Lane {
                cta: self.cta,
                thread: engine.lanes.len() % threads,
                frames: vec![frame],
                waiting: None,
                carry: None,
                asynchronous: AsyncLane::default(),
            });
            self.phase = Phase::Lane;
            return Ok(());
        }
        let id = self.object;
        self.object += 1;
        let object = self.engine.code.function(function).objects[id].clone();
        if object.space == StateSpace::Shared {
            return Ok(());
        }
        let argument = if matches!(object.role, FrameRole::Input | FrameRole::KernelInput) {
            self.argument += 1;
            Some(self.argument - 1)
        } else {
            None
        };
        let owner = Owner::Lane(self.engine.grid, self.engine.lanes.len());
        let address = self.reserve(&object.layout, object.space, owner, argument, memory)?;
        let frame = self.frame()?;
        frame.objects[id] = address;
        frame.owned.push(address);
        Ok(())
    }

    /// Adds one warp per turn, warp-major across CTAs, then marks the grid ready.
    fn build_warp(&mut self) -> Result<()> {
        let engine = &mut self.engine;
        let threads = product(engine.config.block)?;
        if self.warp == threads.div_ceil(WARP_SIZE) {
            self.phase = Phase::Ready;
            return Ok(());
        }
        let start = self.cta * threads + self.warp * WARP_SIZE;
        engine.warps.push(Warp {
            lanes: start..(start + WARP_SIZE).min((self.cta + 1) * threads),
            cursor: 0,
        });
        self.cta += 1;
        if self.cta == product(engine.config.grid)? {
            self.cta = 0;
            self.warp += 1;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, atomic::Ordering};

    use super::*;
    use crate::emulator::{ParsedModule, budget::Budget};

    #[test]
    fn test_dynamic_shared_aliases_charge_one_aligned_region_per_cta() {
        let code = ParsedModule::parse(
            ".version 9.3 .target sm_80 .address_size 64
            .extern .shared .align 4 .b8 a[];
            .extern .shared .align 16 .b8 b[];
            .entry k() { .extern .shared .align 64 .b8 c[]; ret; }",
        )
        .unwrap()
        .compile("k")
        .unwrap();
        for size in [0, CHUNK_BYTES + 1] {
            let config = LaunchConfig::create([2, 1, 1], [1; 3], size).unwrap();
            let storage = storage_plan(&code, config).unwrap();
            assert_eq!(storage.charge, 2 * Memory::charge(size).unwrap());
            assert_eq!(storage.dynamic_alignment, Some(64));
            let options = EmulatorOptions::default();
            let budget = Arc::new(Budget::new(options.execution_storage_bytes));
            let mut memory = Memory::new(0, Arc::clone(&budget));
            let flag = AtomicBool::new(false);
            let builder = GridBuilder::create(&code, 2, config, &options, &mut memory).unwrap();
            let reserved = budget.used();
            builder.release(&mut memory);
            assert_eq!(budget.used(), 0);
            let mut engine =
                Engine::create(&code, &[0, 0], config, &[], &options, &flag, &mut memory).unwrap();
            assert_eq!(budget.used(), reserved);
            for cta in 0..2 {
                let address = engine.globals[cta][0];
                assert!(address.is_multiple_of(64));
                assert_eq!(engine.globals[cta][1], address);
                assert_eq!(engine.shared[cta][0][0], address);
                if size == 0 {
                    assert!(memory.read(address, 0, 1, None, None).unwrap().is_empty());
                } else {
                    assert!(matches!(
                        memory.read(address, 1, 1, None, None),
                        Err(ExecutionError::Indeterminate(_))
                    ));
                }
            }
            assert_ne!(engine.globals[0][0], engine.globals[1][0]);
            while engine.step(&mut memory, &flag).unwrap().is_none() {}
            engine.release(&mut memory);
            assert_eq!(budget.used(), 0);

            let limited = Arc::new(Budget::new(reserved - 1));
            let mut memory = Memory::new(0, Arc::clone(&limited));
            assert!(matches!(
                GridBuilder::create(&code, 2, config, &options, &mut memory),
                Err(ExecutionError::ExecutionStateLimit)
            ));
            assert_eq!(limited.used(), 0);
        }
        let no_shared =
            ParsedModule::parse(".version 9.3 .target sm_80 .address_size 64 .entry k() {ret;}")
                .unwrap()
                .compile("k")
                .unwrap();
        assert_eq!(
            storage_plan(
                &no_shared,
                LaunchConfig::create([1; 3], [1; 3], usize::MAX).unwrap()
            )
            .unwrap()
            .charge,
            0
        );
    }

    #[test]
    fn test_grid_id_reads_keep_the_native_width_and_truncate_only_defined_legacy_reads() {
        let parsed = ParsedModule::parse(
            ".version 9.3 .target sm_80 .address_size 64
            .global .align 8 .u64 output[3]; .entry k() {
                .reg .u64 d; .reg .u32 r;
                mov.u64 d,%gridid; st.global.u64 [output],d;
                mov.u32 r,%gridid; cvt.u64.u32 d,r; st.global.u64 [output+8],d;
                cvt.u64.u16 d,%gridid; st.global.u64 [output+16],d; ret;
            }",
        )
        .unwrap();
        let code = parsed.compile("k").unwrap();
        let options = EmulatorOptions::default();
        let cancellation = AtomicBool::new(false);
        let mut memory = Memory::new(4096, Arc::new(Budget::new(options.execution_storage_bytes)));
        let address = memory
            .allocate(
                24,
                8,
                StateSpace::Global,
                Owner::Device,
                true,
                &cancellation,
            )
            .unwrap();
        let mut engine = Engine::create(
            &code,
            &[address],
            LaunchConfig::create([1; 3], [1; 3], 0).unwrap(),
            &[],
            &options,
            &cancellation,
            &mut memory,
        )
        .unwrap();
        let id = 0x1234_5678_abcd_ef05;
        engine.identify(id, 9);
        while engine.step(&mut memory, &cancellation).unwrap().is_none() {}
        let expected: Vec<_> = [id, id & 0xffff_ffff, id & 0xffff]
            .into_iter()
            .flat_map(u64::to_le_bytes)
            .collect();
        assert_eq!(memory.read(address, 24, 8, None, None).unwrap(), expected);
        engine.release(&mut memory);
    }

    #[test]
    fn test_initial_storage_is_precharged_and_every_construction_checkpoint_can_cancel() {
        let parsed = ParsedModule::parse(
            ".version 9.3 .target sm_80 .address_size 64
            .extern .shared .align 16 .b8 dynamic_a[];
            .entry k(.param .align 16 .b8 input[131073]) {
                .extern .shared .align 64 .b8 dynamic_b[];
                .shared .b8 shared[131073]; .local .b8 local[131073];
                .reg .b32 r<4097>; ret;
            }",
        )
        .unwrap();
        let code = parsed.compile("k").unwrap();
        let config = LaunchConfig::create([2, 1, 1], [2, 1, 1], 131_073).unwrap();
        let options = EmulatorOptions::default();
        let arguments = [vec![19; 131_073]];
        let mut checkpoints = 0;
        loop {
            let budget = Arc::new(Budget::new(options.execution_storage_bytes));
            let mut memory = Memory::new(0, Arc::clone(&budget));
            let cancellation = AtomicBool::new(false);
            let mut builder = GridBuilder::create(&code, 1, config, &options, &mut memory).unwrap();
            let reserved = budget.used();
            let mut complete = false;
            for _ in 0..checkpoints {
                complete = builder
                    .advance(&[0], &arguments, &cancellation, &mut memory)
                    .unwrap();
                assert_eq!(budget.used(), reserved);
                if complete {
                    break;
                }
            }
            if complete {
                let mut engine = builder.finish();
                assert!(engine.trace.is_empty());
                while engine.step(&mut memory, &cancellation).unwrap().is_none() {}
                engine.release(&mut memory);
                assert_eq!(budget.used(), 0);
                break;
            }
            cancellation.store(true, Ordering::Release);
            assert!(matches!(
                builder.advance(&[0], &arguments, &cancellation, &mut memory),
                Err(ExecutionError::Cancelled)
            ));
            builder.release(&mut memory);
            assert_eq!(budget.used(), 0);
            checkpoints += 1;
        }
        assert!(checkpoints > 16);
    }
}
