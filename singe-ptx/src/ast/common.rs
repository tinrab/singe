/// Half-open byte range in the original PTX source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Span {
    /// Inclusive byte offset of the first source byte.
    pub start: usize,
    /// Exclusive byte offset after the last source byte.
    pub end: usize,
}

impl Span {
    /// Stores a half-open byte range without validating its bounds.
    #[must_use]
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// Maps the inclusive start offset to a one-based source position.
    #[must_use]
    pub fn start_position(self, source: &str) -> SourcePosition {
        SourcePosition::from_offset(source, self.start)
    }

    /// Maps the exclusive end offset to a one-based source position.
    #[must_use]
    pub fn end_position(self, source: &str) -> SourcePosition {
        SourcePosition::from_offset(source, self.end)
    }
}

/// One-based line and Unicode character column in the PTX source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourcePosition {
    /// One-based source line number.
    pub line: usize,
    /// One-based Unicode character column within the source line.
    pub column: usize,
}

impl SourcePosition {
    /// Maps a byte offset to a one-based line and Unicode character column.
    ///
    /// Offsets beyond the source are clamped to its end. An offset inside a UTF-8 character is rounded down to its boundary.
    #[must_use]
    pub fn from_offset(source: &str, offset: usize) -> Self {
        let clamped = source.floor_char_boundary(offset.min(source.len()));
        let prefix = &source[..clamped];
        let line = prefix.bytes().filter(|b| *b == b'\n').count() + 1;
        let column = prefix.rfind('\n').map_or_else(
            || prefix.chars().count() + 1,
            |index| prefix[index + 1..].chars().count() + 1,
        );
        Self { line, column }
    }
}

/// PTX storage space for a variable or pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateSpace {
    /// Thread-private registers, `.reg`.
    Reg,
    /// Read-only special registers, `.sreg`.
    SReg,
    /// Read-only constant memory, `.const`.
    Const,
    /// Device global memory, `.global`.
    Global,
    /// Thread-private addressable memory, `.local`.
    Local,
    /// Kernel or device-function parameter storage, `.param`.
    Param,
    /// Memory shared by the threads of a CTA, `.shared`.
    Shared,
    /// Legacy texture state space, `.tex`.
    Tex,
}

/// PTX scalar storage type, including opaque resource references.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    /// Uninterpreted bit pattern with 8 bits, `.b8`.
    B8,
    /// Uninterpreted bit pattern with 16 bits, `.b16`.
    B16,
    /// Uninterpreted bit pattern with 32 bits, `.b32`.
    B32,
    /// Uninterpreted bit pattern with 64 bits, `.b64`.
    B64,
    /// Uninterpreted bit pattern with 128 bits, `.b128`.
    B128,
    /// Signed integer with 8 bits, `.s8`.
    S8,
    /// Signed integer with 16 bits, `.s16`.
    S16,
    /// Signed integer with 32 bits, `.s32`.
    S32,
    /// Signed integer with 64 bits, `.s64`.
    S64,
    /// Unsigned integer with 8 bits, `.u8`.
    U8,
    /// Unsigned integer with 16 bits, `.u16`.
    U16,
    /// Unsigned integer with 32 bits, `.u32`.
    U32,
    /// Unsigned integer with 64 bits, `.u64`.
    U64,
    /// 16-bit floating-point value, `.f16`.
    F16,
    /// Two packed 16-bit floating-point values, `.f16x2`.
    F16x2,
    /// 32-bit floating-point value, `.f32`.
    F32,
    /// 64-bit floating-point value, `.f64`.
    F64,
    /// 16-bit floating-point value with eight exponent bits, `.bf16`.
    Bf16,
    /// TensorFloat-32 value stored in a 32-bit register, `.tf32`.
    Tf32,
    /// Boolean predicate value, `.pred`.
    Pred,
    /// Opaque texture reference, `.texref`.
    TexRef,
    /// Opaque sampler reference, `.samplerref`.
    SamplerRef,
    /// Opaque surface reference, `.surfref`.
    SurfRef,
}

impl ScalarType {
    const ALL: [(Self, &'static str); 23] = [
        (Self::B8, "b8"),
        (Self::B16, "b16"),
        (Self::B32, "b32"),
        (Self::B64, "b64"),
        (Self::B128, "b128"),
        (Self::S8, "s8"),
        (Self::S16, "s16"),
        (Self::S32, "s32"),
        (Self::S64, "s64"),
        (Self::U8, "u8"),
        (Self::U16, "u16"),
        (Self::U32, "u32"),
        (Self::U64, "u64"),
        (Self::F16, "f16"),
        (Self::F16x2, "f16x2"),
        (Self::F32, "f32"),
        (Self::F64, "f64"),
        (Self::Bf16, "bf16"),
        (Self::Tf32, "tf32"),
        (Self::Pred, "pred"),
        (Self::TexRef, "texref"),
        (Self::SamplerRef, "samplerref"),
        (Self::SurfRef, "surfref"),
    ];

    /// Parses a type qualifier written without its leading dot, such as `u32`.
    #[must_use]
    pub fn from_suffix(suffix: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .find(|(_, name)| *name == suffix)
            .map(|(ty, _)| *ty)
    }

    /// Returns the qualifier spelling without its leading dot, such as `u32`.
    #[must_use]
    pub fn suffix(self) -> &'static str {
        Self::ALL
            .iter()
            .find(|(ty, _)| *ty == self)
            .map(|(_, name)| *name)
            .expect("every scalar type has a spelling")
    }

    /// Returns the number of value bits. Predicates have one bit, and opaque
    /// references are 64-bit handles.
    #[must_use]
    pub fn bit_width(self) -> u32 {
        match self {
            Self::Pred => 1,
            Self::B8 | Self::S8 | Self::U8 => 8,
            Self::B16 | Self::S16 | Self::U16 | Self::F16 | Self::Bf16 => 16,
            Self::B32 | Self::S32 | Self::U32 | Self::F32 | Self::F16x2 | Self::Tf32 => 32,
            Self::B64
            | Self::S64
            | Self::U64
            | Self::F64
            | Self::TexRef
            | Self::SamplerRef
            | Self::SurfRef => 64,
            Self::B128 => 128,
        }
    }

    /// Returns the bytes one value occupies in memory, or `None` for predicates.
    #[must_use]
    pub fn byte_len(self) -> Option<u32> {
        (self != Self::Pred).then(|| self.bit_width() / 8)
    }

    /// Returns whether this is a signed integer type.
    #[must_use]
    pub fn is_signed(self) -> bool {
        matches!(self, Self::S8 | Self::S16 | Self::S32 | Self::S64)
    }

    /// Returns whether this is a signed or unsigned integer type.
    #[must_use]
    pub fn is_integer(self) -> bool {
        self.is_signed() || matches!(self, Self::U8 | Self::U16 | Self::U32 | Self::U64)
    }

    /// Returns whether this is an uninterpreted bit-pattern type.
    #[must_use]
    pub fn is_bits(self) -> bool {
        matches!(
            self,
            Self::B8 | Self::B16 | Self::B32 | Self::B64 | Self::B128
        )
    }

    /// Returns whether this is a floating-point type, including packed halves.
    #[must_use]
    pub fn is_float(self) -> bool {
        matches!(
            self,
            Self::F16 | Self::F16x2 | Self::Bf16 | Self::Tf32 | Self::F32 | Self::F64
        )
    }
}

/// Rounding modifier for a floating-point PTX result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatRoundingMode {
    /// Rounds to nearest, with ties to even, `.rn`.
    Rn,
    /// Rounds to nearest, with ties away from zero, `.rna`.
    Rna,
    /// Rounds toward zero, `.rz`.
    Rz,
    /// Rounds toward negative infinity, `.rm`.
    Rm,
    /// Rounds toward positive infinity, `.rp`.
    Rp,
}

/// Rounding modifier for conversion to an integer PTX result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntRoundingMode {
    /// Rounds to the nearest integer, with ties to even, `.rni`.
    Rni,
    /// Rounds to an integer toward zero, `.rzi`.
    Rzi,
    /// Rounds to an integer toward negative infinity, `.rmi`.
    Rmi,
    /// Rounds to an integer toward positive infinity, `.rpi`.
    Rpi,
}

/// Comparison modifier for PTX set and predicate instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    /// Ordered equality, `.eq`.
    Eq,
    /// Ordered inequality, `.ne`.
    Ne,
    /// Ordered less-than comparison, `.lt`.
    Lt,
    /// Ordered less-than-or-equal comparison, `.le`.
    Le,
    /// Ordered greater-than comparison, `.gt`.
    Gt,
    /// Ordered greater-than-or-equal comparison, `.ge`.
    Ge,
    /// Unsigned less-than comparison, `.lo`.
    Lo,
    /// Unsigned less-than-or-equal comparison, `.ls`.
    Ls,
    /// Unsigned greater-than comparison, `.hi`.
    Hi,
    /// Unsigned greater-than-or-equal comparison, `.hs`.
    Hs,
    /// Equality or unordered floating-point operands, `.equ`.
    Equ,
    /// Inequality or unordered floating-point operands, `.neu`.
    Neu,
    /// Less than or unordered floating-point operands, `.ltu`.
    Ltu,
    /// Less than or equal, or unordered floating-point operands, `.leu`.
    Leu,
    /// Greater than or unordered floating-point operands, `.gtu`.
    Gtu,
    /// Greater than or equal, or unordered floating-point operands, `.geu`.
    Geu,
    /// Both floating-point operands are numbers, `.num`.
    Num,
    /// At least one floating-point operand is NaN, `.nan`.
    Nan,
}

/// Part of the integer multiplication result selected by a PTX modifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MulMode {
    /// High half of the full-width integer product, `.hi`.
    Hi,
    /// Low half of the full-width integer product, `.lo`.
    Lo,
    /// Full-width product with twice the input width, `.wide`.
    Wide,
}

/// Symbol linkage qualifier on a PTX declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkingDirective {
    /// Declaration whose definition is supplied by another module, `.extern`.
    Extern,
    /// Definition visible to other modules, `.visible`.
    Visible,
    /// Weak definition that another linked definition can replace, `.weak`.
    Weak,
    /// Variable definition that the linker combines with matching common declarations, `.common`.
    Common,
}

/// Number of scalar components in a PTX vector declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorSize {
    /// Two scalar components, `.v2`.
    V2,
    /// Four scalar components, `.v4`.
    V4,
}
