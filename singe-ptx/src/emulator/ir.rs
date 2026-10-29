//! Executable form of a compiled kernel, produced by lowering and run by the engine.
//!
//! Operands are resolved to register indices, immediates, and memory references, so
//! the engine never looks at names or qualifier strings.

use std::sync::Arc;

use crate::{
    ast::{CompareOp, FloatRoundingMode, IntRoundingMode, MulMode, ScalarType, Span, StateSpace},
    emulator::layout::{ModuleLayout, ParameterLayout, StorageLayout},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegisterId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FunctionId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProgramCounter(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlobalId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryRef {
    Global(GlobalId),
    Frame(FrameId),
}

/// Component of a three-dimensional special register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    pub fn index(self) -> usize {
        self as usize
    }
}

/// Lanes a `%lanemask_*` register selects relative to the executing lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaneMask {
    Equal,
    LessEqual,
    Less,
    GreaterEqual,
    Greater,
}

impl LaneMask {
    /// Returns the warp mask for the lane at `lane` within its warp.
    pub fn bits(self, lane: u32) -> u32 {
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
pub enum SpecialRegister {
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
    pub fn ty(self) -> ScalarType {
        if matches!(self, Self::GridId) {
            ScalarType::U64
        } else {
            ScalarType::U32
        }
    }

    pub fn legacy_type(self, ty: ScalarType) -> bool {
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
pub enum Source {
    Register(RegisterId),
    Immediate(u128),
    Special(SpecialRegister),
    Address { object: MemoryRef, offset: i64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Destination {
    Register(RegisterId),
    Discard,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Predicate {
    Register { register: RegisterId, negated: bool },
    Constant(bool),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AsyncCopyInput {
    Bytes(Source),
    Ignore(Predicate),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsyncCopy {
    pub destination: Address,
    pub source: Address,
    pub size: usize,
    pub input: Option<AsyncCopyInput>,
}

#[derive(Debug, Clone, Copy)]
pub enum MatrixLoad {
    Rows4([Destination; 4]),
    Columns2([Destination; 2]),
}

#[derive(Debug, Clone)]
pub struct MatrixMultiply {
    pub destinations: [Destination; 4],
    pub a: [Source; 4],
    pub b: [Source; 2],
    pub c: [Source; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressBase {
    Register(RegisterId),
    Object(MemoryRef),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Address {
    pub base: AddressBase,
    pub offset: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegerBinary {
    Add,
    Sub,
    Div,
    Rem,
    Min,
    Max,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CarryArithmetic {
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
pub enum ShiftDirection {
    Left,
    Right,
}

/// How `shf`, `bmsk`, and `szext` treat a shift amount, position, or length beyond 31.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeMode {
    /// Limits the value to 32.
    Clamp,
    /// Uses the value modulo 32.
    Wrap,
}

impl RangeMode {
    pub fn limit(self, value: u32) -> u32 {
        match self {
            Self::Clamp => value.min(32),
            Self::Wrap => value & 31,
        }
    }
}

/// Which half of a double-width product an instruction keeps (`.lo` or `.hi`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Half {
    Low,
    High,
}

/// Direction of a `cvta` conversion between a state-space address and a generic address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressConversion {
    ToGeneric,
    ToSpace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitwiseBinary {
    And,
    Or,
    Xor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatBinary {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BooleanOp {
    And,
    Or,
    Xor,
}

impl BooleanOp {
    pub fn apply(self, lhs: bool, rhs: bool) -> bool {
        match self {
            Self::And => lhs & rhs,
            Self::Or => lhs | rhs,
            Self::Xor => lhs ^ rhs,
        }
    }
}

/// A predicate folded into a comparison or `lop3` result with `.and`, `.or`, or `.xor`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Combine {
    pub op: BooleanOp,
    pub predicate: Predicate,
}

/// The optional predicate output of `lop3.and` and `lop3.or`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lop3Predicate {
    pub destination: Destination,
    pub combine: Combine,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vote {
    All,
    Any,
    Ballot,
    Uniform,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shuffle {
    Up,
    Down,
    Butterfly,
    Index,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Atomic {
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
pub enum ConversionRounding {
    None,
    Float(FloatRoundingMode),
    Integer(IntRoundingMode),
}

#[derive(Debug, Clone, Copy)]
pub struct ConversionControl {
    pub rounding: ConversionRounding,
    pub ftz: bool,
    pub clamp: Clamp,
    /// `.satfinite`: an infinite result becomes the largest finite value of its sign.
    pub finite: bool,
}

/// The clamp applied to a floating result; PTX does not allow `.sat` with `.relu`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Clamp {
    None,
    /// `.sat`: clamps to [0.0, 1.0], and NaN becomes 0.0.
    Saturate,
    /// `.relu`: negative results become 0.0.
    Relu,
}

impl Clamp {
    /// Combines the `.sat` and `.relu` qualifiers once lowering has rejected both together.
    pub fn new(saturate: bool, relu: bool) -> Self {
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
pub enum FloatMath {
    Reciprocal,
    SquareRoot,
    ReciprocalSquareRoot,
}

#[derive(Debug, Clone, Copy)]
pub struct MathControl {
    pub rounding: FloatRoundingMode,
    pub ftz: bool,
    /// `rcp.approx.ftz.f64`: reads only the upper 32 bits of the input encoding and
    /// rounds the result to the upper 32 bits.
    pub upper_word: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatFormat {
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
pub struct FloatControl {
    pub rounding: FloatRoundingMode,
    pub ftz: bool,
    pub clamp: Clamp,
    /// `.NaN` on `min`/`max`: a NaN input produces NaN instead of the other operand.
    pub nan: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatProperty {
    Finite,
    Infinite,
    Number,
    Nan,
    Normal,
    Subnormal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unary {
    Neg,
    Abs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitUnary {
    Popc,
    Clz,
    Bfind { shift: bool },
    Brev,
    Cnot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Permute {
    Generic,
    Forward,
    Backward,
    ReplicateByte,
    ClampLeft,
    ClampRight,
    ReplicateHalf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DotMode {
    FourBytes,
    TwoLow,
    TwoHigh,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegerPackWidth {
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
pub struct IntegerPackFormat {
    pub width: IntegerPackWidth,
    pub signed: bool,
}

#[derive(Debug, Clone)]
pub enum Operation {
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
pub enum MemoryOperation {
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
pub enum AsyncOperation {
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
pub enum ControlOperation {
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
pub enum CollectiveOperation {
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
pub enum IntegerOperation {
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
pub enum FloatOperation {
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
pub enum DataOperation {
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
pub struct Instruction {
    pub span: Span,
    pub guard: Option<Predicate>,
    pub operation: Operation,
}

#[derive(Debug, Clone)]
pub struct FrameObject {
    pub space: StateSpace,
    pub layout: StorageLayout,
    pub role: FrameRole,
    pub dynamic: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameRole {
    KernelInput,
    Input,
    Return,
    Variable,
}

#[derive(Debug, Clone)]
pub struct RegisterLayout {
    pub name: String,
    pub ty: ScalarType,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub noreturn: bool,
    pub registers: Vec<RegisterLayout>,
    pub objects: Vec<FrameObject>,
    pub parameters: Vec<ParameterLayout>,
    pub returns: Vec<ParameterLayout>,
    pub instructions: Vec<Instruction>,
}

#[derive(Debug, Clone)]
pub struct Executable {
    pub entry: FunctionId,
    pub functions: Vec<Function>,
    pub module: ModuleLayout,
}

impl Executable {
    pub fn function(&self, id: FunctionId) -> &Function {
        &self.functions[id.0]
    }

    pub fn entry_function(&self) -> &Function {
        self.function(self.entry)
    }
}
