mod asynchronous;
mod collective;
mod control;
mod data;
mod float_unit;
mod initialization;
mod integer_unit;
mod matrix;
mod memory_unit;

use std::{
    mem,
    ops::Range,
    result::Result as StdResult,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

pub use initialization::GridBuilder;

use crate::{
    ast::{ScalarType, StateSpace},
    emulator::{
        CompiledKernel,
        budget::{Reservation, node},
        engine::{
            asynchronous::{AsyncLane, AsyncState},
            matrix::MatrixWaiting,
        },
        execution_error::{ExecutionError, FaultLocation, StalledLane},
        ir::{
            self, Address, AddressBase, BooleanOp, Combine, Destination, FrameRole, FunctionId,
            MemoryRef, Operation, Predicate, Shuffle, Source, SpecialRegister, Vote,
        },
        memory::{
            Actor, GridId, Memory, Owner, Window, canonical_address, specific,
            validate_address_size,
        },
        numeric::{self, extend, mask},
        runtime::{EmulatorOptions, ExecutionReport, LaunchConfig},
        snapshot::{FrameSnapshot, LaneSnapshot, RegisterSnapshot},
        target::{self, NAMED_BARRIERS, WARP_SIZE, WARPS_PER_CTA},
    },
};

type Result<T> = StdResult<T, ExecutionError>;

/// Whether an executed instruction moves its lane to the next instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    Advance,
    /// The operation set the program counter itself, or parked the lane until a
    /// rendezvous releases it.
    Hold,
}

/// One register's value. Every PTX write replaces the whole register, so a register is
/// either fully initialized or not at all.
#[derive(Clone, Copy, Default)]
struct Register {
    // Two words instead of a `u128` keep a register at 24 bytes rather than 32.
    words: [u64; 2],
    initialized: bool,
}

impl Register {
    fn new(value: u128) -> Self {
        Self {
            words: [numeric::low_u64(value), numeric::low_u64(value >> 64)],
            initialized: true,
        }
    }

    fn value(self) -> Option<u128> {
        self.initialized
            .then(|| u128::from(self.words[0]) | (u128::from(self.words[1]) << 64))
    }
}

struct Frame {
    function: FunctionId,
    pc: usize,
    registers: Vec<Register>,
    objects: Vec<u64>,
    owned: Vec<u64>,
    returns: Vec<u64>,
    charge: usize,
}

struct Lane {
    cta: usize,
    thread: usize,
    frames: Vec<Frame>,
    waiting: Option<Waiting>,
    carry: Option<bool>,
    asynchronous: AsyncLane,
}

impl Lane {
    /// Returns this lane's index within its warp.
    fn warp_lane(&self) -> u32 {
        u32::try_from(self.thread % WARP_SIZE).expect("warp lane is below the warp size")
    }

    /// Returns this lane's bit in a warp membership mask.
    fn lane_bit(&self) -> u32 {
        1 << self.warp_lane()
    }
}

struct Warp {
    lanes: Range<usize>,
    cursor: usize,
}

#[derive(Clone)]
enum Waiting {
    Matrix(MatrixWaiting),
    Async {
        through: u64,
    },
    Cta {
        barrier: usize,
        generation: u64,
        count: Option<usize>,
    },
    Warp {
        mask: u32,
        operation: WarpOperation,
    },
}

#[derive(Clone)]
enum WarpOperation {
    Barrier,
    ReduceAdd {
        destination: Destination,
        value: u32,
    },
    Shuffle {
        mode: Shuffle,
        destination: Destination,
        valid: Option<Destination>,
        value: u128,
        lane: u32,
        control: u32,
    },
    Vote {
        mode: Vote,
        destination: Destination,
        predicate: bool,
    },
}

impl WarpOperation {
    fn matches(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Barrier, Self::Barrier) | (Self::ReduceAdd { .. }, Self::ReduceAdd { .. }) => {
                true
            }
            (Self::Shuffle { mode: a, .. }, Self::Shuffle { mode: b, .. }) => a == b,
            (Self::Vote { mode: a, .. }, Self::Vote { mode: b, .. }) => a == b,
            _ => false,
        }
    }
}

/// The lane and instruction that suspended a launch at `brkpt`.
#[derive(Clone, Copy)]
struct PausedAt {
    lane: usize,
    pc: usize,
}

/// Indices whose waiters must be examined again, each listed at most once.
#[derive(Default)]
struct DirtySet {
    flags: Vec<bool>,
    pending: Vec<usize>,
}

impl DirtySet {
    /// Reserves every flag and list slot up front, so marking never allocates.
    fn create(count: usize) -> Result<Self> {
        let mut set = Self::default();
        set.flags
            .try_reserve_exact(count)
            .and_then(|()| set.pending.try_reserve_exact(count))
            .map_err(|_| ExecutionError::ExecutionStateLimit)?;
        set.flags.resize(count, false);
        Ok(set)
    }

    fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    fn mark(&mut self, index: usize) {
        if !mem::replace(&mut self.flags[index], true) {
            self.pending.push(index);
        }
    }

    /// Takes the marked indices and clears the set; hand the buffer back with `restore`.
    fn take(&mut self) -> Vec<usize> {
        let pending = mem::take(&mut self.pending);
        for &index in &pending {
            self.flags[index] = false;
        }
        pending
    }

    fn restore(&mut self, mut buffer: Vec<usize>) {
        buffer.clear();
        self.pending = buffer;
    }
}

pub struct Engine {
    asynchronous: AsyncState,
    code: Arc<ir::Executable>,
    config: LaunchConfig,
    options: EmulatorOptions,
    grid: GridId,
    globals: Vec<Vec<u64>>,
    generations: Vec<[u64; NAMED_BARRIERS]>,
    remaining_lanes: Vec<usize>,
    shared: Vec<Vec<Vec<u64>>>,
    lanes: Vec<Lane>,
    warps: Vec<Warp>,
    next_warp: usize,
    report: ExecutionReport,
    result_charge: usize,
    steps: u64,
    /// CTAs whose named barriers may have become ready.
    dirty_ctas: DirtySet,
    /// Warps whose collective or matrix operations may have become ready.
    dirty_warps: DirtySet,
    breakpoint: Option<PausedAt>,
    #[cfg(test)]
    pub(crate) trace: Vec<TraceEntry>,
}

/// One instruction issued to one lane, recorded by test builds.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceEntry {
    pub lane: usize,
    pub function: FunctionId,
    pub pc: usize,
}

/// A [`TraceEntry`] tagged with the launch that issued it.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchTraceEntry {
    pub launch: u64,
    pub entry: TraceEntry,
}

fn product(dimensions: [u32; 3]) -> Result<usize> {
    dimensions
        .into_iter()
        .try_fold(1usize, |n, d| n.checked_mul(d as usize))
        .ok_or(ExecutionError::ExecutionStateLimit)
}

fn coordinates(index: usize, dimensions: [u32; 3]) -> [u32; 3] {
    let x = index % dimensions[0] as usize;
    let y = index / dimensions[0] as usize % dimensions[1] as usize;
    let z = index / dimensions[0] as usize / dimensions[1] as usize;
    {
        let coordinate =
            |value| u32::try_from(value).expect("coordinates are bounded by u32 launch dimensions");
        [coordinate(x), coordinate(y), coordinate(z)]
    }
}

fn frame_charge(function: &ir::Function) -> Result<usize> {
    let transfer_peak = function
        .parameters
        .iter()
        .chain(&function.returns)
        .try_fold(0usize, |sum, parameter| {
            let bytes = usize::try_from(parameter.layout.byte_len())
                .map_err(|_| ExecutionError::ExecutionStateLimit)?;
            sum.checked_add(bytes)
                .ok_or(ExecutionError::ExecutionStateLimit)
        })?;
    function
        .registers
        .len()
        .checked_mul(size_of::<Register>())
        .and_then(|v| v.checked_add(function.objects.len().checked_mul(24)?))
        .and_then(|v| v.checked_add(function.returns.len().checked_mul(8)?))
        .and_then(|v| v.checked_add(size_of::<Frame>()))
        .and_then(|v| v.checked_add(transfer_peak.checked_mul(2)?))
        .ok_or(ExecutionError::ExecutionStateLimit)
}

/// Returns the storage reserved up front for a launch's result: the report, the deepest
/// possible fault location, and the largest possible stall list.
fn result_charge(
    code: &CompiledKernel,
    options: &EmulatorOptions,
    lane_count: usize,
) -> Result<usize> {
    // Reserve diagnostics before any instruction, including the largest possible stall list.
    // Error chains and reports are immutable and shared after publication.
    let name_bytes = code
        .artifact
        .executable
        .functions
        .iter()
        .map(|f| f.name.len())
        .max()
        .unwrap_or(0);
    let diagnostic_depth = if code.artifact.executable.functions.iter().any(|function| {
        function.instructions.iter().any(|instruction| {
            matches!(
                instruction.operation,
                Operation::Control(ir::ControlOperation::Call { .. })
            )
        })
    }) {
        options.call_depth
    } else {
        1
    };
    diagnostic_depth
        .checked_add(1)
        .and_then(|depth| depth.checked_mul(name_bytes.checked_add(size_of::<String>())?))
        .and_then(|bytes| bytes.checked_add(lane_count.checked_mul(2 * size_of::<StalledLane>())?))
        .and_then(|bytes| bytes.checked_add(1024 + size_of::<ExecutionReport>()))
        .ok_or(ExecutionError::ExecutionStateLimit)
}

/// Returns the per-CTA storage: global and shared address tables, barrier generations,
/// and scheduling state.
fn cta_charge(code: &CompiledKernel, global_count: usize) -> Result<usize> {
    let shared_charge = code
        .artifact
        .executable
        .functions
        .iter()
        .try_fold(0usize, |total, function| {
            function
                .objects
                .len()
                .checked_mul(8)?
                .checked_add(24)?
                .checked_add(total)
        })
        .ok_or(ExecutionError::ExecutionStateLimit)?;
    global_count
        .checked_mul(8)
        .and_then(|v| v.checked_add(shared_charge))
        .and_then(|v| {
            // Barrier generations, remaining lanes, and a dirty flag and list slot.
            v.checked_add(
                size_of::<[u64; NAMED_BARRIERS]>()
                    + 2 * size_of::<usize>()
                    + size_of::<bool>()
                    + 48,
            )
        })
        .ok_or(ExecutionError::ExecutionStateLimit)
}

fn cancel(flag: &AtomicBool) -> Result<()> {
    if flag.load(Ordering::Acquire) {
        Err(ExecutionError::Cancelled)
    } else {
        Ok(())
    }
}

impl Engine {
    fn inspection_lane(&self, cta: [u32; 3], lane: [u32; 3]) -> Result<usize> {
        let index = |coordinates: [u32; 3], dimensions: [u32; 3]| {
            if coordinates
                .into_iter()
                .zip(dimensions)
                .any(|(coordinate, bound)| coordinate >= bound)
            {
                return None;
            }
            coordinates.into_iter().zip(dimensions).rev().try_fold(
                0usize,
                |index, (coordinate, bound)| {
                    index
                        .checked_mul(bound as usize)?
                        .checked_add(coordinate as usize)
                },
            )
        };
        let block_size = product(self.config.block())?;
        let selected = index(cta, self.config.grid()).and_then(|cta| {
            index(lane, self.config.block())
                .and_then(|lane| cta.checked_mul(block_size)?.checked_add(lane))
        });
        selected
            .filter(|index| *index < self.lanes.len())
            .ok_or_else(|| {
                ExecutionError::Invalid("inspection coordinates are outside the launch".into())
            })
    }

    pub(crate) fn snapshot_storage(
        &self,
        cta: [u32; 3],
        lane: [u32; 3],
        cancellation: &AtomicBool,
    ) -> Result<usize> {
        cancel(cancellation)?;
        let selected = &self.lanes[self.inspection_lane(cta, lane)?];
        let frames = selected
            .frames
            .len()
            .checked_mul(size_of::<FrameSnapshot>());
        let mut size = frames
            .and_then(|frames| frames.checked_add(node::<LaneSnapshot>()))
            .ok_or(ExecutionError::ExecutionStateLimit)?;
        for frame in &selected.frames {
            cancel(cancellation)?;
            let function = &self.code.function(frame.function);
            size = size
                .checked_add(function.name.len())
                .and_then(|size| {
                    size.checked_add(
                        function
                            .registers
                            .len()
                            .checked_mul(size_of::<RegisterSnapshot>())?,
                    )
                })
                .ok_or(ExecutionError::ExecutionStateLimit)?;
            for register in &function.registers {
                cancel(cancellation)?;
                size = size
                    .checked_add(register.name.len())
                    .ok_or(ExecutionError::ExecutionStateLimit)?;
            }
        }
        Ok(size)
    }

    pub(crate) fn snapshot(
        &self,
        cta: [u32; 3],
        lane: [u32; 3],
        cancellation: &AtomicBool,
    ) -> Result<LaneSnapshot> {
        cancel(cancellation)?;
        let selected = &self.lanes[self.inspection_lane(cta, lane)?];
        let mut frames = Vec::with_capacity(selected.frames.len());
        for frame in &selected.frames {
            cancel(cancellation)?;
            let function = &self.code.function(frame.function);
            let mut registers = Vec::with_capacity(function.registers.len());
            for (layout, value) in function.registers.iter().zip(&frame.registers) {
                cancel(cancellation)?;
                registers.push(RegisterSnapshot {
                    name: layout.name.clone(),
                    ty: layout.ty,
                    declaration: layout.span,
                    value: value.value(),
                });
            }
            frames.push(FrameSnapshot {
                function: function.name.clone(),
                instruction: frame.pc,
                span: function
                    .instructions
                    .get(frame.pc)
                    .map(|instruction| instruction.span),
                registers,
            });
        }
        Ok(LaneSnapshot {
            cta,
            lane,
            carry: selected.carry,
            frames,
        })
    }

    #[cfg(test)]
    pub fn create(
        code: &CompiledKernel,
        globals: &[u64],
        config: LaunchConfig,
        args: &[Vec<u8>],
        options: &EmulatorOptions,
        cancellation: &AtomicBool,
        memory: &mut Memory,
    ) -> Result<Self> {
        let mut builder = GridBuilder::create(code, globals.len(), config, options, memory)?;
        loop {
            match builder.advance(globals, args, cancellation, memory) {
                Ok(true) => return Ok(builder.finish()),
                Ok(false) => {}
                Err(error) => {
                    builder.release(memory);
                    return Err(error);
                }
            }
        }
    }

    fn reserve_grid(
        code: &CompiledKernel,
        global_count: usize,
        config: LaunchConfig,
        options: &EmulatorOptions,
        memory: &mut Memory,
        grid: GridId,
        storage: &initialization::StoragePlan,
    ) -> Result<Self> {
        let ctas = product(config.grid)?;
        let threads = product(config.block)?;
        let lane_count = ctas
            .checked_mul(threads)
            .ok_or(ExecutionError::ExecutionStateLimit)?;
        let result_charge = result_charge(code, options, lane_count)?;
        let entry_charge = frame_charge(code.artifact.executable.entry_function())?;
        let cta_charge = cta_charge(code, global_count)?;
        let lane_charge = entry_charge
            .checked_add(
                size_of::<Lane>()
                    + size_of::<Warp>()
                    + size_of::<usize>()
                    + size_of::<bool>()
                    + 128usize.max(matrix::SCRATCH_BYTES_PER_LANE),
            )
            .ok_or(ExecutionError::ExecutionStateLimit)?;
        let core_charge = lane_count
            .checked_mul(lane_charge)
            .and_then(|v| v.checked_add(ctas.checked_mul(cta_charge)?))
            .and_then(|v| v.checked_add(result_charge))
            .and_then(|v| v.checked_add(size_of::<Self>() + size_of::<GridBuilder>()))
            .filter(|v| *v <= options.execution_storage_bytes)
            .ok_or(ExecutionError::ExecutionStateLimit)?;
        if options.call_depth == 0 {
            return Err(ExecutionError::CallDepth);
        }
        memory.reserve_execution(grid, core_charge)?;
        memory.reserve_grid_allocations(grid, storage.charge)?;
        let mut engine = Self {
            asynchronous: AsyncState::default(),
            code: Arc::clone(&code.artifact.executable),
            config,
            options: options.clone(),
            grid,
            globals: Vec::new(),
            generations: Vec::new(),
            remaining_lanes: Vec::new(),
            shared: Vec::new(),
            lanes: Vec::new(),
            warps: Vec::new(),
            next_warp: 0,
            report: ExecutionReport {
                launch_id: grid.sequence(),
                ..Default::default()
            },
            result_charge,
            steps: 0,
            dirty_ctas: DirtySet::create(ctas)?,
            dirty_warps: DirtySet::create(ctas * threads.div_ceil(WARP_SIZE))?,
            breakpoint: None,
            #[cfg(test)]
            trace: Vec::new(),
        };
        engine
            .lanes
            .try_reserve_exact(lane_count)
            .map_err(|_| ExecutionError::ExecutionStateLimit)?;
        engine
            .globals
            .try_reserve_exact(ctas)
            .map_err(|_| ExecutionError::ExecutionStateLimit)?;
        engine
            .shared
            .try_reserve_exact(ctas)
            .map_err(|_| ExecutionError::ExecutionStateLimit)?;
        engine
            .generations
            .try_reserve_exact(ctas)
            .map_err(|_| ExecutionError::ExecutionStateLimit)?;
        engine
            .remaining_lanes
            .try_reserve_exact(ctas)
            .map_err(|_| ExecutionError::ExecutionStateLimit)?;
        engine
            .warps
            .try_reserve_exact(ctas * threads.div_ceil(WARP_SIZE))
            .map_err(|_| ExecutionError::ExecutionStateLimit)?;
        Ok(engine)
    }

    fn frame(
        &self,
        function_id: FunctionId,
        actor: Actor,
        args: &[u64],
        returns: Vec<u64>,
        cancellation: &AtomicBool,
        memory: &mut Memory,
    ) -> Result<Frame> {
        let Actor { cta, lane, .. } = actor;
        let function = self.code.function(function_id);
        let charge = frame_charge(function)?;
        let mut frame = Frame {
            function: function_id,
            pc: 0,
            registers: vec![Register::default(); function.registers.len()],
            objects: self.shared[cta][function_id.0].clone(),
            owned: Vec::new(),
            returns,
            charge,
        };
        let mut argument = 0;
        for (id, object) in function.objects.iter().enumerate() {
            if object.space == StateSpace::Shared {
                continue;
            }
            let size = usize::try_from(object.layout.byte_len())
                .map_err(|_| ExecutionError::ExecutionStateLimit)?;
            let address = memory.allocate(
                size,
                object.layout.alignment(),
                object.space,
                Owner::Lane(self.grid, lane),
                false,
                cancellation,
            )?;
            frame.owned.push(address);
            validate_address_size(address, size, self.code.module.address_size)?;
            frame.objects[id] = address;
            if matches!(object.role, FrameRole::Input | FrameRole::KernelInput) {
                memory.copy_parameter(
                    args[argument],
                    address,
                    size,
                    Actor {
                        grid: self.grid,
                        cta,
                        lane,
                    },
                    cancellation,
                )?;
                memory.protect(address);
                argument += 1;
            }
        }
        Ok(frame)
    }

    fn current(&self, lane: usize) -> &Frame {
        &self.lanes[lane].frames[self.lanes[lane].frames.len() - 1]
    }

    /// Returns the function executing in the lane's innermost frame.
    fn function(&self, lane: usize) -> &ir::Function {
        self.code.function(self.current(lane).function)
    }

    fn actor(&self, lane: usize) -> Actor {
        Actor {
            grid: self.grid,
            cta: self.lanes[lane].cta,
            lane,
        }
    }

    pub fn release(&self, memory: &mut Memory) {
        memory.release_grid(self.grid);
    }

    pub fn identify(&mut self, launch_id: u64, stream_id: u64) {
        self.report.launch_id = launch_id;
        self.report.stream_id = stream_id;
    }

    #[cfg(test)]
    pub fn launch_id(&self) -> u64 {
        self.report.launch_id
    }

    fn current_mut(&mut self, lane: usize) -> &mut Frame {
        let index = self.lanes[lane].frames.len() - 1;
        &mut self.lanes[lane].frames[index]
    }

    fn object(&self, lane: usize, object: MemoryRef) -> u64 {
        match object {
            MemoryRef::Global(id) => self.globals[self.lanes[lane].cta][id.0],
            MemoryRef::Frame(id) => self.current(lane).objects[id.0],
        }
    }

    fn register(&self, lane: usize, id: ir::RegisterId, bits: u32) -> Result<u128> {
        let Some(value) = self.current(lane).registers[id.0].value() else {
            let name = &self.function(lane).registers[id.0].name;
            return Err(ExecutionError::Indeterminate(format!("register `{name}`")));
        };
        Ok(value & mask(bits))
    }

    fn source(&self, lane: usize, source: &Source, ty: ScalarType) -> Result<u128> {
        Ok(match source {
            Source::Register(id) => self.register(lane, *id, ty.bit_width())?,
            Source::Immediate(value) => *value,
            Source::Address { object, offset } => {
                let address = specific(self.object(lane, *object))
                    .checked_add_signed(*offset)
                    .ok_or_else(|| ExecutionError::Memory("address overflow".into()))?;
                if u128::from(address) > mask(ty.bit_width()) {
                    return Err(ExecutionError::Memory(
                        "address exceeds destination width".into(),
                    ));
                }
                u128::from(address)
            }
            Source::Special(special) => {
                let item = &self.lanes[lane];
                u128::from(match special {
                    SpecialRegister::ThreadId(axis) => {
                        coordinates(item.thread, self.config.block)[axis.index()]
                    }
                    SpecialRegister::BlockId(axis) => {
                        coordinates(item.cta, self.config.grid)[axis.index()]
                    }
                    SpecialRegister::BlockDimensions(axis) => self.config.block[axis.index()],
                    SpecialRegister::GridDimensions(axis) => self.config.grid[axis.index()],
                    SpecialRegister::LaneId => item.warp_lane(),
                    SpecialRegister::WarpId => u32::try_from(item.thread / WARP_SIZE)
                        .expect("validated block size bounds warp index"),
                    SpecialRegister::WarpCount => target::WARP_IDENTIFIER_COUNT,
                    SpecialRegister::SmId => 0,
                    SpecialRegister::SmCount => 1,
                    SpecialRegister::GridId => {
                        return Ok(u128::from(self.report.launch_id) & mask(ty.bit_width()));
                    }
                    SpecialRegister::LaneMask(selection) => selection.bits(item.warp_lane()),
                })
            }
        } & mask(ty.bit_width()))
    }

    /// Reads a source operand as an unsigned 32-bit value.
    fn u32_source(&self, lane: usize, source: &Source) -> Result<u32> {
        Ok(numeric::low_u32(self.source(
            lane,
            source,
            ScalarType::U32,
        )?))
    }

    /// Reads a source operand as an untyped 32-bit value.
    fn b32_source(&self, lane: usize, source: &Source) -> Result<u32> {
        Ok(numeric::low_u32(self.source(
            lane,
            source,
            ScalarType::B32,
        )?))
    }

    /// Evaluates the predicate a combining comparison folds into its result.
    fn combine(&self, lane: usize, combine: Option<Combine>) -> Result<Option<(BooleanOp, bool)>> {
        combine
            .map(|combine| {
                self.predicate(lane, combine.predicate)
                    .map(|value| (combine.op, value))
            })
            .transpose()
    }

    fn predicate(&self, lane: usize, predicate: Predicate) -> Result<bool> {
        match predicate {
            Predicate::Constant(value) => Ok(value),
            Predicate::Register { register, negated } => {
                Ok((self.register(lane, register, 1)? != 0) != negated)
            }
        }
    }

    fn write(&mut self, lane: usize, destination: Destination, value: u128, ty: ScalarType) {
        if let Destination::Register(id) = destination {
            let declared = self.function(lane).registers[id.0].ty;
            self.current_mut(lane).registers[id.0] =
                Register::new(extend(value, ty) & mask(declared.bit_width()));
        }
    }

    /// Leaves a destination register without a defined value, so reading it faults.
    fn write_unspecified(&mut self, lane: usize, destination: Destination) {
        if let Destination::Register(id) = destination {
            self.current_mut(lane).registers[id.0] = Register::default();
        }
    }

    fn address(&self, lane: usize, address: Address, space: Option<StateSpace>) -> Result<u64> {
        let base = match address.base {
            AddressBase::Register(id) => {
                let ty = self.function(lane).registers[id.0].ty;
                numeric::low_u64(self.register(lane, id, ty.bit_width())?)
            }
            AddressBase::Object(object) => {
                let address = self.object(lane, object);
                if space.is_some() {
                    specific(address)
                } else {
                    address
                }
            }
        };
        let value = base
            .checked_add_signed(address.offset)
            .ok_or_else(|| ExecutionError::Memory("address overflow".into()))?;
        let value = if space.is_none() {
            canonical_address(value)
        } else {
            value
        };
        Window::of(space.unwrap_or(StateSpace::Global))
            .base()
            .checked_add(value)
            .ok_or_else(|| ExecutionError::Memory("address window overflow".into()))
    }

    fn read_memory<'a>(
        &'a self,
        global: &'a Memory,
        lane: usize,
        address: u64,
        size: usize,
        align: usize,
        space: Option<StateSpace>,
    ) -> Result<&'a [u8]> {
        global.read(address, size, align, space, Some(self.actor(lane)))
    }

    fn store_memory(
        &self,
        global: &mut Memory,
        lane: usize,
        address: u64,
        bytes: &[u8],
        align: usize,
        space: Option<StateSpace>,
    ) -> Result<()> {
        global.write(address, bytes, align, space, Some(self.actor(lane)))
    }

    fn location(&self, lane: usize, error: ExecutionError) -> ExecutionError {
        ExecutionError::Lane {
            location: Arc::new(self.fault_location(lane, self.current(lane).pc)),
            cause: Arc::new(error),
        }
    }

    fn fault_location(&self, lane: usize, pc: usize) -> FaultLocation {
        let frame = self.current(lane);
        let function = &self.code.function(frame.function);
        FaultLocation {
            launch_id: self.report.launch_id,
            stream_id: self.report.stream_id,
            span: function.instructions[pc].span,
            function: function.name.clone(),
            cta: coordinates(self.lanes[lane].cta, self.config.grid),
            lane: coordinates(self.lanes[lane].thread, self.config.block),
            instruction: pc,
            call_stack: self.lanes[lane]
                .frames
                .iter()
                .map(|f| self.code.function(f.function).name.clone())
                .collect(),
        }
    }

    pub fn is_paused(&self) -> bool {
        self.breakpoint.is_some()
    }

    pub fn resume(&mut self) {
        self.breakpoint = None;
    }

    pub fn breakpoint_location(&self) -> Option<FaultLocation> {
        self.breakpoint
            .map(|paused| self.fault_location(paused.lane, paused.pc))
    }

    pub fn breakpoint_storage(&self) -> Result<usize> {
        let PausedAt { lane, .. } = self
            .breakpoint
            .ok_or_else(|| ExecutionError::Invalid("launch is not paused".into()))?;
        self.lanes[lane]
            .frames
            .iter()
            .try_fold(size_of::<FaultLocation>() + 256, |bytes, frame| {
                self.code
                    .function(frame.function)
                    .name
                    .len()
                    .checked_mul(2)
                    .and_then(|names| names.checked_add(size_of::<String>()))
                    .and_then(|frame| bytes.checked_add(frame))
                    .ok_or(ExecutionError::ExecutionStateLimit)
            })
    }

    pub fn step(
        &mut self,
        global: &mut Memory,
        cancellation: &AtomicBool,
    ) -> Result<Option<ExecutionReport>> {
        self.step_inner(global, cancellation)
            .map_err(|cause| self.failure(cause))
    }

    pub fn failure(&mut self, cause: ExecutionError) -> ExecutionError {
        ExecutionError::Launch {
            report: Arc::new(self.take_report()),
            cause: Arc::new(cause),
        }
    }

    pub fn retain_result(&self, memory: &mut Memory) -> Arc<Reservation> {
        Arc::new(memory.retain_execution(self.grid, self.result_charge))
    }

    fn step_inner(
        &mut self,
        global: &mut Memory,
        cancellation: &AtomicBool,
    ) -> Result<Option<ExecutionReport>> {
        cancel(cancellation)?;
        if self.is_paused() {
            return Ok(None);
        }
        let asynchronous_progress = self.advance_async(global, cancellation)?;
        if !self.dirty_ctas.is_empty() || !self.dirty_warps.is_empty() {
            self.resolve_waiters(cancellation)?;
        }
        let mut selected = None;
        for _ in 0..self.warps.len() {
            cancel(cancellation)?;
            let id = self.next_warp;
            self.next_warp = (id + 1) % self.warps.len();
            let warp = &mut self.warps[id];
            let size = warp.lanes.len();
            for _ in 0..size {
                let lane = warp.lanes.start + warp.cursor;
                warp.cursor = (warp.cursor + 1) % size;
                if !self.lanes[lane].frames.is_empty() && self.lanes[lane].waiting.is_none() {
                    selected = Some((id, lane));
                    break;
                }
            }
            if selected.is_some() {
                break;
            }
        }
        let Some((warp_id, leader)) = selected else {
            if asynchronous_progress {
                return Ok(None);
            }
            if self.lanes.iter().all(|lane| lane.frames.is_empty()) {
                return Ok(Some(self.take_report()));
            }
            return Err(ExecutionError::Stall(
                self.lanes
                    .iter()
                    .filter(|l| l.waiting.is_some())
                    .map(|l| StalledLane {
                        cta: coordinates(l.cta, self.config.grid),
                        lane: coordinates(l.thread, self.config.block),
                    })
                    .collect(),
            ));
        };
        let frame = self.current(leader);
        let (function, pc, depth) = (frame.function, frame.pc, self.lanes[leader].frames.len());
        let code = Arc::clone(&self.code);
        let instruction = &code.function(function).instructions[pc];
        let mut active = [0usize; WARP_SIZE];
        let mut active_count = 0;
        let mut active_mask = 0u32;
        for lane in self.warps[warp_id].lanes.clone() {
            if self.lanes[lane].waiting.is_some() || self.lanes[lane].frames.len() != depth {
                continue;
            }
            let frame = self.current(lane);
            if (frame.function, frame.pc) != (function, pc) {
                continue;
            }
            if self.steps >= self.options.launch_instructions {
                return Err(self.location(lane, ExecutionError::InstructionLimit));
            }
            self.steps += 1;
            let enabled = instruction
                .guard
                .map(|p| self.predicate(lane, p))
                .transpose()
                .map_err(|error| self.location(lane, error))?
                .unwrap_or(true);
            if enabled {
                active[active_count] = lane;
                active_count += 1;
                active_mask |= self.lanes[lane].lane_bit();
            } else {
                self.current_mut(lane).pc += 1;
            }
        }
        for &lane in &active[..active_count] {
            self.report.instructions += 1;
            #[cfg(test)]
            if self.trace.len() < 4096 {
                self.trace.push(TraceEntry { lane, function, pc });
            }
            self.execute(
                global,
                lane,
                &instruction.operation,
                active_mask,
                cancellation,
            )
            .map_err(|error| self.location(lane, error))?;
        }
        Ok(None)
    }

    fn take_report(&mut self) -> ExecutionReport {
        mem::take(&mut self.report)
    }

    fn execute(
        &mut self,
        global: &mut Memory,
        lane: usize,
        operation: &Operation,
        active_mask: u32,
        cancellation: &AtomicBool,
    ) -> Result<()> {
        let step = match operation {
            Operation::Integer(operation) => {
                self.execute_integer(lane, operation)?;
                Step::Advance
            }
            Operation::Float(operation) => {
                self.execute_float(lane, operation)?;
                Step::Advance
            }
            Operation::Data(operation) => {
                self.execute_data(lane, operation)?;
                Step::Advance
            }
            Operation::Memory(operation) => {
                self.execute_memory(global, lane, operation)?;
                Step::Advance
            }
            Operation::Async(operation) => {
                self.execute_async(global, lane, operation, cancellation)?
            }
            Operation::Control(operation) => {
                self.execute_control(global, lane, operation, cancellation)?
            }
            Operation::Collective(operation) => {
                self.execute_collective(global, lane, operation, active_mask)?
            }
        };
        if step == Step::Advance {
            self.current_mut(lane).pc += 1;
        }
        Ok(())
    }

    fn pop(&mut self, lane: usize, memory: &mut Memory) {
        self.mark_waiting(lane);
        if let Some(frame) = self.lanes[lane].frames.pop() {
            for address in frame.owned {
                memory.release(address);
            }
            memory.release_execution(self.grid, frame.charge);
        }
        if self.lanes[lane].frames.is_empty() {
            self.request_async(lane, self.lanes[lane].asynchronous.issued);
            self.report.completed_lanes += 1;
            let cta = self.lanes[lane].cta;
            self.remaining_lanes[cta] -= 1;
            if self.remaining_lanes[cta] == 0 {
                self.report.completed_ctas += 1;
            }
        }
    }

    fn wait_warp(&mut self, lane: usize, mask: u32, operation: WarpOperation) -> Result<()> {
        if mask & self.lanes[lane].lane_bit() == 0 {
            return Err(ExecutionError::Invalid(
                "executing lane is absent from warp membership mask".into(),
            ));
        }
        self.lanes[lane].waiting = Some(Waiting::Warp { mask, operation });
        self.mark_waiting(lane);
        Ok(())
    }

    /// Schedules the lane's CTA and warp for waiter resolution after its state changes.
    fn mark_waiting(&mut self, lane: usize) {
        let (cta, thread) = (self.lanes[lane].cta, self.lanes[lane].thread);
        // Warps are laid out warp-major: every CTA's first warp, then every second warp.
        let warp = thread / WARP_SIZE * self.globals.len() + cta;
        self.dirty_ctas.mark(cta);
        self.dirty_warps.mark(warp);
    }

    fn release_waiter(&mut self, lane: usize) {
        self.lanes[lane].waiting = None;
        self.current_mut(lane).pc += 1;
    }

    /// Resolves waiters in the CTAs and warps whose state changed since the last turn.
    fn resolve_waiters(&mut self, cancellation: &AtomicBool) -> Result<()> {
        let ctas = self.dirty_ctas.take();
        let warps = self.dirty_warps.take();
        let result = self.resolve_marked(&ctas, &warps, cancellation);
        self.dirty_ctas.restore(ctas);
        self.dirty_warps.restore(warps);
        result
    }

    fn resolve_marked(
        &mut self,
        ctas: &[usize],
        warps: &[usize],
        cancellation: &AtomicBool,
    ) -> Result<()> {
        self.resolve_matrices(warps, cancellation)?;
        for &cta in ctas {
            cancel(cancellation)?;
            self.resolve_cta_barriers(cta)?;
        }
        self.resolve_warp_collectives(warps, cancellation)
    }

    /// Releases every named barrier of one CTA whose arrival condition holds.
    fn resolve_cta_barriers(&mut self, cta: usize) -> Result<()> {
        let threads = product(self.config.block)?;
        let range = cta * threads..(cta + 1) * threads;
        // One membership bit per lane, grouped into warp masks.
        let mut live = [0u32; WARPS_PER_CTA];
        let mut live_count = 0;
        let mut participants = [[0u32; WARPS_PER_CTA]; NAMED_BARRIERS];
        let mut first: [Option<(usize, Option<usize>)>; NAMED_BARRIERS] = [None; NAMED_BARRIERS];
        for lane in range.clone() {
            let local = lane - range.start;
            let (warp, bit) = (local / WARP_SIZE, 1 << (local % WARP_SIZE));
            if !self.lanes[lane].frames.is_empty() {
                live[warp] |= bit;
                live_count += 1;
            }
            let Some(Waiting::Cta {
                barrier,
                generation,
                count,
            }) = self.lanes[lane].waiting
            else {
                continue;
            };
            if generation != self.generations[cta][barrier] {
                continue;
            }
            match first[barrier] {
                Some((leader, expected)) if expected != count => {
                    return Err(self.location(
                        leader,
                        ExecutionError::Invalid("inconsistent barrier counts".into()),
                    ));
                }
                Some(_) => {}
                None => first[barrier] = Some((lane, count)),
            }
            participants[barrier][warp] |= bit;
        }
        for (barrier, mut participants) in participants.into_iter().enumerate() {
            let Some((_, count)) = first[barrier] else {
                continue;
            };
            // Only warps whose live lanes have all arrived count toward the barrier.
            let mut arrived_count = 0;
            let mut arrived_warps = 0;
            for (waiting, &live) in participants.iter_mut().zip(&live) {
                if live != 0 && *waiting & live == live {
                    arrived_warps += 1;
                    arrived_count += live.count_ones() as usize;
                    *waiting = live;
                } else {
                    *waiting = 0;
                }
            }
            let ready = count.map_or(arrived_count == live_count, |count| {
                arrived_warps * WARP_SIZE >= count
            });
            if !ready {
                continue;
            }
            self.generations[cta][barrier] += 1;
            for (warp, mut arrived) in participants.into_iter().enumerate() {
                while arrived != 0 {
                    let bit = arrived.trailing_zeros() as usize;
                    self.release_waiter(range.start + warp * WARP_SIZE + bit);
                    arrived &= arrived - 1;
                }
            }
        }
        Ok(())
    }

    /// Completes every warp collective whose participants have all arrived.
    fn resolve_warp_collectives(
        &mut self,
        warps: &[usize],
        cancellation: &AtomicBool,
    ) -> Result<()> {
        for &warp in warps {
            cancel(cancellation)?;
            let range = self.warps[warp].lanes.clone();
            for leader in range.clone() {
                let Some(Waiting::Warp { mask, operation }) = &self.lanes[leader].waiting else {
                    continue;
                };
                let Some(arrivals) = self.collective_arrivals(range.clone(), *mask, operation)
                else {
                    continue;
                };
                self.complete_collective(&arrivals)?;
                for (lane, _) in arrivals {
                    self.release_waiter(lane);
                }
            }
        }
        Ok(())
    }

    /// Returns each live lane in `mask` with its operation once all of them wait at a
    /// matching collective with the same mask, or `None` while any is still missing.
    fn collective_arrivals(
        &self,
        range: Range<usize>,
        mask: u32,
        operation: &WarpOperation,
    ) -> Option<Vec<(usize, WarpOperation)>> {
        range
            .filter(|&lane| {
                mask & self.lanes[lane].lane_bit() != 0 && !self.lanes[lane].frames.is_empty()
            })
            .map(|lane| match &self.lanes[lane].waiting {
                Some(Waiting::Warp {
                    mask: waiting_mask,
                    operation: waiting,
                }) if *waiting_mask == mask && operation.matches(waiting) => {
                    Some((lane, waiting.clone()))
                }
                _ => None,
            })
            .collect()
    }

    /// Writes every participant's result for one collective.
    fn complete_collective(&mut self, arrivals: &[(usize, WarpOperation)]) -> Result<()> {
        let live_mask = arrivals
            .iter()
            .fold(0u32, |mask, (lane, _)| mask | self.lanes[*lane].lane_bit());
        let ballot = arrivals.iter().fold(0u32, |mask, (lane, operation)| {
            if matches!(
                operation,
                WarpOperation::Vote {
                    predicate: true,
                    ..
                }
            ) {
                mask | self.lanes[*lane].lane_bit()
            } else {
                mask
            }
        });
        let reduced = arrivals
            .iter()
            .filter_map(|(_, operation)| match operation {
                WarpOperation::ReduceAdd { value, .. } => Some(*value),
                _ => None,
            })
            .fold(0u32, u32::wrapping_add);
        for (lane, operation) in arrivals {
            match operation {
                WarpOperation::Barrier => {}
                WarpOperation::ReduceAdd { destination, .. } => {
                    self.write(*lane, *destination, u128::from(reduced), ScalarType::U32);
                }
                WarpOperation::Vote {
                    mode, destination, ..
                } => {
                    let (value, ty) = match mode {
                        Vote::Ballot => (ballot, ScalarType::B32),
                        Vote::All => (u32::from(ballot == live_mask), ScalarType::Pred),
                        Vote::Any => (u32::from(ballot != 0), ScalarType::Pred),
                        Vote::Uniform => (
                            u32::from(ballot == 0 || ballot == live_mask),
                            ScalarType::Pred,
                        ),
                    };
                    self.write(*lane, *destination, u128::from(value), ty);
                }
                WarpOperation::Shuffle {
                    destination, valid, ..
                } => {
                    let (result, valid_value) = self.shuffle(*lane, operation, arrivals)?;
                    self.write(*lane, *destination, result, ScalarType::B32);
                    if let Some(valid) = valid {
                        self.write(*lane, *valid, u128::from(valid_value), ScalarType::Pred);
                    }
                }
            }
        }
        Ok(())
    }

    /// Returns the value `lane` receives from a shuffle and whether its source lane was in
    /// range; an out-of-range shuffle returns the lane's own value.
    fn shuffle(
        &self,
        lane: usize,
        operation: &WarpOperation,
        arrivals: &[(usize, WarpOperation)],
    ) -> Result<(u128, bool)> {
        let WarpOperation::Shuffle {
            mode,
            value,
            lane: delta,
            control,
            ..
        } = operation
        else {
            return Err(ExecutionError::Invalid("expected a shuffle".into()));
        };
        let own = self.lanes[lane].warp_lane();
        let segment = (control >> 8) & 31;
        let min = own & segment;
        let max = min | (control & 31 & !segment);
        let delta = delta & 31;
        let (source, in_range) = match mode {
            Shuffle::Up => (own.wrapping_sub(delta), own >= delta && own - delta >= max),
            Shuffle::Down => (own + delta, own + delta <= max),
            Shuffle::Butterfly => (own ^ delta, own ^ delta <= max),
            Shuffle::Index => (min | (delta & !segment), (min | (delta & !segment)) <= max),
        };
        if !in_range {
            return Ok((*value, false));
        }
        arrivals
            .iter()
            .find_map(|(other, operation)| match operation {
                WarpOperation::Shuffle { value, .. }
                    if self.lanes[*other].warp_lane() == source =>
                {
                    Some((*value, true))
                }
                _ => None,
            })
            .ok_or_else(|| {
                self.location(
                    lane,
                    ExecutionError::Indeterminate("shuffle source did not participate".into()),
                )
            })
    }
}

fn combine_predicate(value: bool, combined: Option<(BooleanOp, bool)>) -> bool {
    combined.map_or(value, |(op, rhs)| op.apply(value, rhs))
}
