//! Executable form of a compiled kernel, produced by lowering and run by the engine.
//!
//! Operands are resolved to register indices, immediates, and memory references, so the engine never looks at names or qualifier strings.

use std::sync::Arc;

use crate::{
    ast::{CompareOp, FloatRoundingMode, IntRoundingMode, MulMode, ScalarType, Span, StateSpace},
    emulator::layout::{ModuleLayout, ParameterLayout, StorageLayout},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RegisterId(pub(crate) usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FunctionId(pub(crate) usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ProgramCounter(pub(crate) usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GlobalId(pub(crate) usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FrameId(pub(crate) usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MemoryRef {
    Global(GlobalId),
    Frame(FrameId),
}

/// Component of a three-dimensional special register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    pub(crate) fn index(self) -> usize {
        self as usize
    }
}

/// Lanes a `%lanemask_*` register selects relative to the executing lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LaneMask {
    Equal,
    LessEqual,
    Less,
    GreaterEqual,
    Greater,
}

impl LaneMask {
    /// Returns the warp mask for the lane at `lane` within its warp.
    pub(crate) fn bits(self, lane: u32) -> u32 {
        let equal = 1u32 << lane;
        match self {
            Self::Equal => equal,
            Self::LessEqual => u32::MAX >> (31 - lane),
            Self::Less => equal - 1,
            Self::GreaterEqual => u32::MAX << lane,
            Self::Greater => (u32::MAX << lane) & !equal,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpecialRegister {
    ThreadId(Axis),
    BlockId(Axis),
    BlockDimensions(Axis),
    GridDimensions(Axis),
    LaneId,
    WarpId,
    WarpCount,
    SmId,
    SmCount,
    GridId,
    LaneMask(LaneMask),
}

impl SpecialRegister {
    pub(crate) fn ty(self) -> ScalarType {
        if matches!(self, Self::GridId) {
            ScalarType::U64
        } else {
            ScalarType::U32
        }
    }

    pub(crate) fn legacy_type(self, ty: ScalarType) -> bool {
        let bits = match ty {
            ScalarType::U16 | ScalarType::S16 | ScalarType::B16 => 16,
            ScalarType::U32 | ScalarType::S32 | ScalarType::B32 => 32,
            _ => return false,
        };
        match self {
            Self::ThreadId(_)
            | Self::BlockId(_)
            | Self::BlockDimensions(_)
            | Self::GridDimensions(_) => bits == 16,
            Self::GridId => matches!(bits, 16 | 32),
            _ => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Source {
    Register(RegisterId),
    Immediate(u128),
    Special(SpecialRegister),
    Address { object: MemoryRef, offset: i64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Destination {
    Register(RegisterId),
    Discard,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Predicate {
    Register { register: RegisterId, negated: bool },
    Constant(bool),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AsyncCopyInput {
    Bytes(Source),
    Ignore(Predicate),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AsyncCopy {
    pub(crate) destination: Address,
    pub(crate) source: Address,
    pub(crate) size: usize,
    pub(crate) input: Option<AsyncCopyInput>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum MatrixLoad {
    Rows4([Destination; 4]),
    Columns2([Destination; 2]),
}

#[derive(Debug, Clone)]
pub(crate) struct MatrixMultiply {
    pub(crate) destinations: [Destination; 4],
    pub(crate) a: [Source; 4],
    pub(crate) b: [Source; 2],
    pub(crate) c: [Source; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AddressBase {
    Register(RegisterId),
    Object(MemoryRef),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Address {
    pub(crate) base: AddressBase,
    pub(crate) offset: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IntegerBinary {
    Add,
    Sub,
    Div,
    Rem,
    Min,
    Max,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CarryArithmetic {
    Add {
        a: Source,
        b: Source,
    },
    Subtract {
        a: Source,
        b: Source,
    },
    MultiplyAdd {
        a: Source,
        b: Source,
        c: Source,
        half: Half,
    },
}

/// Direction of `shl`/`shr` and `shf`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ShiftDirection {
    Left,
    Right,
}

/// How `shf`, `bmsk`, and `szext` treat a shift amount, position, or length beyond 31.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RangeMode {
    /// Limits the value to 32.
    Clamp,
    /// Uses the value modulo 32.
    Wrap,
}

impl RangeMode {
    pub(crate) fn limit(self, value: u32) -> u32 {
        match self {
            Self::Clamp => value.min(32),
            Self::Wrap => value & 31,
        }
    }
}

/// Which half of a double-width product an instruction keeps (`.lo` or `.hi`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Half {
    Low,
    High,
}

/// Direction of a `cvta` conversion between a state-space address and a generic address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AddressConversion {
    ToGeneric,
    ToSpace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BitwiseBinary {
    And,
    Or,
    Xor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FloatBinary {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BooleanOp {
    And,
    Or,
    Xor,
}

impl BooleanOp {
    pub(crate) fn apply(self, lhs: bool, rhs: bool) -> bool {
        match self {
            Self::And => lhs & rhs,
            Self::Or => lhs | rhs,
            Self::Xor => lhs ^ rhs,
        }
    }
}

/// A predicate folded into a comparison or `lop3` result with `.and`, `.or`, or `.xor`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Combine {
    pub(crate) op: BooleanOp,
    pub(crate) predicate: Predicate,
}

/// The optional predicate output of `lop3.and` and `lop3.or`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Lop3Predicate {
    pub(crate) destination: Destination,
    pub(crate) combine: Combine,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Vote {
    All,
    Any,
    Ballot,
    Uniform,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Shuffle {
    Up,
    Down,
    Butterfly,
    Index,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Atomic {
    Add,
    Min,
    Max,
    And,
    Or,
    Xor,
    Exchange,
    CompareSwap { compare: Source },
    Increment,
    Decrement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConversionRounding {
    None,
    Float(FloatRoundingMode),
    Integer(IntRoundingMode),
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ConversionControl {
    pub(crate) rounding: ConversionRounding,
    pub(crate) ftz: bool,
    pub(crate) clamp: Clamp,
    /// `.satfinite`: an infinite result becomes the largest finite value of its sign.
    pub(crate) finite: bool,
}

/// The clamp applied to a floating result; PTX does not allow `.sat` with `.relu`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Clamp {
    None,
    /// `.sat`: clamps to [0.0, 1.0], and NaN becomes 0.0.
    Saturate,
    /// `.relu`: negative results become 0.0.
    Relu,
}

impl Clamp {
    /// Combines the `.sat` and `.relu` qualifiers once lowering has rejected both together.
    pub(crate) fn new(saturate: bool, relu: bool) -> Self {
        if saturate {
            Self::Saturate
        } else if relu {
            Self::Relu
        } else {
            Self::None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FloatMath {
    Reciprocal,
    SquareRoot,
    ReciprocalSquareRoot,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MathControl {
    pub(crate) rounding: FloatRoundingMode,
    pub(crate) ftz: bool,
    /// `rcp.approx.ftz.f64`: reads only the upper 32 bits of the input encoding and rounds the result to the upper 32 bits.
    pub(crate) upper_word: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FloatFormat {
    F16,
    F16x2,
    Bf16,
    Bf16x2,
    F32,
    F64,
}

impl FloatFormat {
    pub(crate) fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "f16" => Self::F16,
            "f16x2" => Self::F16x2,
            "bf16" => Self::Bf16,
            "bf16x2" => Self::Bf16x2,
            "f32" => Self::F32,
            "f64" => Self::F64,
            _ => return None,
        })
    }

    pub(crate) fn storage(self) -> ScalarType {
        match self {
            Self::F16 => ScalarType::F16,
            Self::Bf16 => ScalarType::B16,
            Self::F16x2 | Self::Bf16x2 => ScalarType::B32,
            Self::F32 => ScalarType::F32,
            Self::F64 => ScalarType::F64,
        }
    }

    pub(crate) fn packed(self) -> bool {
        matches!(self, Self::F16x2 | Self::Bf16x2)
    }

    pub(crate) fn bf16(self) -> bool {
        matches!(self, Self::Bf16 | Self::Bf16x2)
    }

    pub(crate) fn half(self) -> bool {
        !matches!(self, Self::F32 | Self::F64)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct FloatControl {
    pub(crate) rounding: FloatRoundingMode,
    pub(crate) ftz: bool,
    pub(crate) clamp: Clamp,
    /// `.NaN` on `min`/`max`: a NaN input produces NaN instead of the other operand.
    pub(crate) nan: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FloatProperty {
    Finite,
    Infinite,
    Number,
    Nan,
    Normal,
    Subnormal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Unary {
    Neg,
    Abs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BitUnary {
    Popc,
    Clz,
    Bfind { shift: bool },
    Brev,
    Cnot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Permute {
    Generic,
    Forward,
    Backward,
    ReplicateByte,
    ClampLeft,
    ClampRight,
    ReplicateHalf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DotMode {
    FourBytes,
    TwoLow,
    TwoHigh,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IntegerPackWidth {
    Two,
    Four,
    Eight,
    Sixteen,
}

impl IntegerPackWidth {
    pub(crate) fn bits(self) -> u32 {
        match self {
            Self::Two => 2,
            Self::Four => 4,
            Self::Eight => 8,
            Self::Sixteen => 16,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct IntegerPackFormat {
    pub(crate) width: IntegerPackWidth,
    pub(crate) signed: bool,
}

#[derive(Debug, Clone)]
pub(crate) enum Operation {
    Integer(IntegerOperation),
    Float(FloatOperation),
    Data(DataOperation),
    Memory(MemoryOperation),
    Async(AsyncOperation),
    Control(ControlOperation),
    Collective(CollectiveOperation),
}

/// Loads, stores, atomics, and fences.
#[derive(Debug, Clone)]
pub(crate) enum MemoryOperation {
    Load {
        ty: ScalarType,
        space: Option<StateSpace>,
        destinations: Vec<Destination>,
        address: Address,
    },
    Store {
        ty: ScalarType,
        space: Option<StateSpace>,
        address: Address,
        sources: Vec<Source>,
    },
    ConvertAddress {
        ty: ScalarType,
        space: StateSpace,
        conversion: AddressConversion,
        destination: Destination,
        source: Source,
    },
    Atomic {
        op: Atomic,
        ty: ScalarType,
        space: Option<StateSpace>,
        destination: Destination,
        address: Address,
        value: Source,
    },
    /// Orders memory accesses; the interpreter already serializes every access.
    Fence,
}

/// Asynchronous copies and `mbarrier` objects.
#[derive(Debug, Clone)]
pub(crate) enum AsyncOperation {
    Copy(AsyncCopy),
    CommitCopies,
    WaitCopies {
        pending: u32,
        commit: bool,
    },
    TrackCopies {
        address: Address,
        space: Option<StateSpace>,
        no_increment: bool,
    },
    BarrierInit {
        address: Address,
        space: Option<StateSpace>,
        count: Source,
    },
    BarrierInvalidate {
        address: Address,
        space: Option<StateSpace>,
    },
    BarrierArrive {
        destination: Destination,
        address: Address,
        space: Option<StateSpace>,
    },
    BarrierTest {
        destination: Destination,
        address: Address,
        space: Option<StateSpace>,
        state: Source,
        parity: bool,
    },
}

/// Branches, calls, returns, and lane termination.
#[derive(Debug, Clone)]
pub(crate) enum ControlOperation {
    Branch {
        target: ProgramCounter,
    },
    Call {
        function: FunctionId,
        noreturn: bool,
        arguments: Vec<MemoryRef>,
        returns: Vec<MemoryRef>,
    },
    Return,
    Exit,
    Trap,
    Breakpoint,
}

/// Operations that rendezvous with other lanes of a warp or CTA.
#[derive(Debug, Clone)]
pub(crate) enum CollectiveOperation {
    MatrixLoad {
        layout: MatrixLoad,
        address: Address,
    },
    MatrixMultiply(Arc<MatrixMultiply>),
    CtaBarrier {
        barrier: Source,
        count: Option<Source>,
    },
    WarpBarrier {
        mask: Source,
    },
    Shuffle {
        mode: Shuffle,
        destination: Destination,
        valid: Option<Destination>,
        value: Source,
        lane: Source,
        control: Source,
        mask: Source,
    },
    Vote {
        mode: Vote,
        destination: Destination,
        predicate: Predicate,
        mask: Source,
    },
    ActiveMask {
        destination: Destination,
    },
    ReduceAdd {
        destination: Destination,
        value: Source,
        mask: Source,
    },
}

/// Integer arithmetic, logic, and bit-field operations on registers.
#[derive(Debug, Clone)]
pub(crate) enum IntegerOperation {
    DotProduct {
        mode: DotMode,
        a_ty: ScalarType,
        b_ty: ScalarType,
        destination: Destination,
        a: Source,
        b: Source,
        c: Source,
    },
    Pack {
        format: IntegerPackFormat,
        destination: Destination,
        a: Source,
        b: Source,
        c: Option<Source>,
    },
    Multiply24 {
        ty: ScalarType,
        half: Half,
        saturate: bool,
        destination: Destination,
        a: Source,
        b: Source,
        c: Option<Source>,
    },
    SumDifference {
        ty: ScalarType,
        destination: Destination,
        a: Source,
        b: Source,
        c: Source,
    },
    Carryless {
        half: Half,
        destination: Destination,
        a: Source,
        b: Source,
        c: Source,
    },
    FindSet {
        destination: Destination,
        mask: Source,
        base: Source,
        offset: Source,
    },
    MaskBits {
        range: RangeMode,
        destination: Destination,
        position: Source,
        length: Source,
    },
    ExtendBits {
        range: RangeMode,
        ty: ScalarType,
        destination: Destination,
        value: Source,
        length: Source,
    },
    Carry {
        ty: ScalarType,
        arithmetic: CarryArithmetic,
        input: bool,
        output: bool,
        destination: Destination,
    },
    Unary {
        op: Unary,
        ty: ScalarType,
        destination: Destination,
        source: Source,
    },
    BitUnary {
        op: BitUnary,
        ty: ScalarType,
        destination: Destination,
        source: Source,
    },
    Extract {
        ty: ScalarType,
        destination: Destination,
        value: Source,
        position: Source,
        length: Source,
    },
    Insert {
        ty: ScalarType,
        destination: Destination,
        value: Source,
        base: Source,
        position: Source,
        length: Source,
    },
    Lop3 {
        destination: Destination,
        a: Source,
        b: Source,
        c: Source,
        lut: u8,
        predicate: Option<Lop3Predicate>,
    },
    Funnel {
        direction: ShiftDirection,
        range: RangeMode,
        destination: Destination,
        a: Source,
        b: Source,
        amount: Source,
    },
    Permute {
        mode: Permute,
        destination: Destination,
        a: Source,
        b: Source,
        control: Source,
    },
    Binary {
        op: IntegerBinary,
        ty: ScalarType,
        saturate: bool,
        destination: Destination,
        a: Source,
        b: Source,
    },
    Multiply {
        saturate: bool,
        ty: ScalarType,
        mode: MulMode,
        destination: Destination,
        a: Source,
        b: Source,
        c: Option<Source>,
    },
    Bitwise {
        op: BitwiseBinary,
        ty: ScalarType,
        destination: Destination,
        a: Source,
        b: Source,
    },
    Not {
        ty: ScalarType,
        destination: Destination,
        source: Source,
    },
    Shift {
        direction: ShiftDirection,
        ty: ScalarType,
        destination: Destination,
        value: Source,
        amount: Source,
    },
}

/// Floating-point arithmetic, math, and classification.
#[derive(Debug, Clone)]
pub(crate) enum FloatOperation {
    Unary {
        op: Unary,
        format: FloatFormat,
        ftz: bool,
        destination: Destination,
        source: Source,
    },
    Test {
        property: FloatProperty,
        ty: ScalarType,
        destination: Destination,
        source: Source,
    },
    PackedConvert {
        format: FloatFormat,
        rounding: FloatRoundingMode,
        relu: bool,
        finite: bool,
        destination: Destination,
        high: Source,
        low: Source,
    },
    Binary {
        op: FloatBinary,
        format: FloatFormat,
        control: FloatControl,
        destination: Destination,
        a: Source,
        b: Source,
    },
    Exponential {
        ftz: bool,
        destination: Destination,
        source: Source,
    },
    Math {
        op: FloatMath,
        ty: ScalarType,
        control: MathControl,
        destination: Destination,
        source: Source,
    },
    Fma {
        format: FloatFormat,
        control: FloatControl,
        destination: Destination,
        a: Source,
        b: Source,
        c: Source,
    },
    CopySign {
        ty: ScalarType,
        destination: Destination,
        sign: Source,
        magnitude: Source,
    },
}

/// Moves, conversions, comparisons, and selections.
#[derive(Debug, Clone)]
pub(crate) enum DataOperation {
    Pack {
        destination: Destination,
        low: Source,
        high: Source,
    },
    Unpack {
        low: Destination,
        high: Destination,
        source: Source,
    },
    Move {
        ty: ScalarType,
        destination: Destination,
        source: Source,
    },
    Convert {
        from: ScalarType,
        to: ScalarType,
        control: ConversionControl,
        destination: Destination,
        source: Source,
    },
    CompareValue {
        comparison: CompareOp,
        from: ScalarType,
        to: ScalarType,
        ftz: bool,
        destination: Destination,
        a: Source,
        b: Source,
        combine: Option<Combine>,
    },
    SelectSign {
        ty: ScalarType,
        selector_ty: ScalarType,
        ftz: bool,
        destination: Destination,
        a: Source,
        b: Source,
        selector: Source,
    },
    Compare {
        comparison: CompareOp,
        ty: ScalarType,
        ftz: bool,
        destination: Destination,
        inverse: Option<Destination>,
        a: Source,
        b: Source,
        combine: Option<Combine>,
    },
    Select {
        ty: ScalarType,
        destination: Destination,
        a: Source,
        b: Source,
        predicate: Predicate,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct Instruction {
    pub(crate) span: Span,
    pub(crate) guard: Option<Predicate>,
    pub(crate) operation: Operation,
}

#[derive(Debug, Clone)]
pub(crate) struct FrameObject {
    pub(crate) space: StateSpace,
    pub(crate) layout: StorageLayout,
    pub(crate) role: FrameRole,
    pub(crate) dynamic: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FrameRole {
    KernelInput,
    Input,
    Return,
    Variable,
}

#[derive(Debug, Clone)]
pub(crate) struct RegisterLayout {
    pub(crate) name: String,
    pub(crate) ty: ScalarType,
    pub(crate) span: Span,
}

#[derive(Debug, Clone)]
pub(crate) struct Function {
    pub(crate) name: String,
    pub(crate) noreturn: bool,
    pub(crate) registers: Vec<RegisterLayout>,
    pub(crate) objects: Vec<FrameObject>,
    pub(crate) parameters: Vec<ParameterLayout>,
    pub(crate) returns: Vec<ParameterLayout>,
    pub(crate) instructions: Vec<Instruction>,
}

#[derive(Debug, Clone)]
pub(crate) struct Executable {
    pub(crate) entry: FunctionId,
    pub(crate) functions: Vec<Function>,
    pub(crate) module: ModuleLayout,
}

impl Executable {
    pub(crate) fn function(&self, id: FunctionId) -> &Function {
        &self.functions[id.0]
    }

    pub(crate) fn entry_function(&self) -> &Function {
        self.function(self.entry)
    }
}
