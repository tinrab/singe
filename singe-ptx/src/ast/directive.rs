use crate::ast::{FloatLiteral, LinkingDirective, ScalarType, Span, StateSpace, VectorSize};

/// PTX variable declaration with its storage qualifiers and optional initializer.
#[derive(Debug, Clone, PartialEq)]
pub struct VariableDecl {
    /// Byte range of this syntax node in the original PTX source.
    pub span: Span,
    /// Declared variable symbol without array bounds.
    pub name: String,
    /// Scalar element type of the declared storage.
    pub ty: ScalarType,
    /// Explicit PTX storage space, or `None` when the declaration omits it.
    pub state_space: Option<StateSpace>,
    /// Explicit symbol linkage, or `None` for module-local linkage.
    pub linkage: Option<LinkingDirective>,
    /// Declaration attributes in source order.
    pub attributes: Vec<Attribute>,
    /// Explicit alignment in bytes, or `None` when no `.align` qualifier appears.
    pub alignment: Option<u32>,
    /// Explicit `.v2` or `.v4` scalar grouping, if present.
    pub vector: Option<VectorSize>,
    /// Number of registers requested by a numbered declaration such as `%r<4>`.
    pub count: Option<u32>,
    /// Array dimensions in source order. A `None` entry represents an unsized dimension.
    pub array_bounds: Vec<Option<u32>>,
    /// Optional constant initializer expression.
    pub initializer: Option<Initializer>,
}

/// Storage or address-identity attribute attached to a PTX declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Attribute {
    /// Managed-memory variable attribute, `.managed`.
    Managed {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
    },
    /// Unified-address identity attribute, `.unified`, with a two-word identifier.
    Unified {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// First 64-bit word of the unified-address identifier.
        uuid1: u64,
        /// Second 64-bit word of the unified-address identifier.
        uuid2: u64,
    },
}

/// Constant expression or aggregate used to initialize a PTX variable.
#[derive(Debug, Clone, PartialEq)]
pub enum Initializer {
    /// Signed integer constant.
    Integer {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Integer literal value before target-width conversion.
        value: i64,
    },
    /// Unsigned integer constant.
    UnsignedInteger {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Integer literal value before target-width conversion.
        value: u64,
    },
    /// Floating-point constant with its original literal representation.
    Float {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Floating-point literal representation.
        value: FloatLiteral,
    },
    /// Address of a named symbol.
    Symbol {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Symbol whose address the initializer takes.
        name: String,
    },
    /// Integer-width cast of another constant expression.
    Cast {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Integer type selected by the cast.
        ty: InitializerCastType,
        /// Expression to convert.
        expr: Box<Self>,
    },
    /// Generic-address conversion of a constant expression.
    Generic {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Symbol address expression to convert.
        expr: Box<Self>,
    },
    /// Constant expression restricted by an integer bit mask.
    Masked {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Bit mask applied to the constant value.
        mask: u64,
        /// Constant expression restricted by the bit mask.
        value: Box<Self>,
    },
    /// Aggregate initializer whose elements retain source order.
    List {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Aggregate elements in source order.
        values: Vec<Self>,
    },
    /// Unary constant expression.
    Unary {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Unary operator.
        op: InitializerUnaryOp,
        /// Operand of the operator.
        expr: Box<Self>,
    },
    /// Binary constant expression.
    Binary {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Binary operator.
        op: InitializerBinaryOp,
        /// Left operand of the binary constant expression.
        left: Box<Self>,
        /// Right operand of the binary constant expression.
        right: Box<Self>,
    },
    /// Constant expression selected by a condition.
    Conditional {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Expression whose nonzero value selects `then_expr`.
        condition: Box<Self>,
        /// Expression selected when `condition` is nonzero.
        then_expr: Box<Self>,
        /// Expression selected when `condition` is zero.
        else_expr: Box<Self>,
    },
}

/// Unary operator in a constant initializer expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitializerUnaryOp {
    /// Unary identity, `+`.
    Plus,
    /// Arithmetic negation, `-`.
    Minus,
    /// Logical negation, `!`.
    LogicalNot,
    /// Bitwise complement, `~`.
    BitwiseNot,
}

/// Binary operator in a constant initializer expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitializerBinaryOp {
    /// Addition, `+`.
    Add,
    /// Subtraction, `-`.
    Sub,
    /// Multiplication, `*`.
    Mul,
    /// Division, `/`.
    Div,
    /// Remainder, `%`.
    Rem,
    /// Left shift, `<<`.
    Shl,
    /// Right shift, `>>`.
    Shr,
    /// Less-than comparison, `<`.
    Lt,
    /// Less-than-or-equal comparison, `<=`.
    Le,
    /// Greater-than comparison, `>`.
    Gt,
    /// Greater-than-or-equal comparison, `>=`.
    Ge,
    /// Equality comparison, `==`.
    Eq,
    /// Inequality comparison, `!=`.
    Ne,
    /// Bitwise conjunction, `&`.
    BitAnd,
    /// Bitwise exclusive or, `^`.
    BitXor,
    /// Bitwise inclusive or, `|`.
    BitOr,
    /// Logical conjunction, `&&`.
    LogicalAnd,
    /// Logical disjunction, `||`.
    LogicalOr,
}

/// Integer interpretation selected by a constant initializer cast.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitializerCastType {
    /// Signed 64-bit interpretation, `.s64`.
    S64,
    /// Unsigned 64-bit interpretation, `.u64`.
    U64,
}

impl Initializer {
    /// Returns the byte span of this initializer expression.
    #[must_use]
    pub fn span(&self) -> Span {
        match self {
            Self::Integer { span, .. }
            | Self::UnsignedInteger { span, .. }
            | Self::Float { span, .. }
            | Self::Symbol { span, .. }
            | Self::Cast { span, .. }
            | Self::Generic { span, .. }
            | Self::Masked { span, .. }
            | Self::List { span, .. }
            | Self::Unary { span, .. }
            | Self::Binary { span, .. }
            | Self::Conditional { span, .. } => *span,
        }
    }
}
