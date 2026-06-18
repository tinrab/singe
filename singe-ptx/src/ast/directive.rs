use crate::ast::{LinkingDirective, ScalarType, Span, StateSpace, VectorSize};

#[derive(Debug, Clone, PartialEq)]
pub struct VariableDecl {
    pub span: Span,
    pub name: String,
    pub ty: ScalarType,
    pub state_space: Option<StateSpace>,
    pub linkage: Option<LinkingDirective>,
    pub attributes: Vec<Attribute>,
    pub alignment: Option<u32>,
    pub vector: Option<VectorSize>,
    pub count: Option<u32>,
    pub array_bounds: Vec<Option<u32>>,
    pub initializer: Option<Initializer>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Attribute {
    Managed { span: Span },
    Unified { span: Span, uuid1: u64, uuid2: u64 },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Initializer {
    Integer {
        span: Span,
        value: i64,
    },
    UnsignedInteger {
        span: Span,
        value: u64,
    },
    Float {
        span: Span,
        value: f64,
    },
    Symbol {
        span: Span,
        name: String,
    },
    Cast {
        span: Span,
        ty: InitializerCastType,
        expr: Box<Initializer>,
    },
    Generic {
        span: Span,
        expr: Box<Initializer>,
    },
    Masked {
        span: Span,
        mask: u64,
        value: Box<Initializer>,
    },
    List {
        span: Span,
        values: Vec<Initializer>,
    },
    Unary {
        span: Span,
        op: InitializerUnaryOp,
        expr: Box<Initializer>,
    },
    Binary {
        span: Span,
        op: InitializerBinaryOp,
        left: Box<Initializer>,
        right: Box<Initializer>,
    },
    Conditional {
        span: Span,
        condition: Box<Initializer>,
        then_expr: Box<Initializer>,
        else_expr: Box<Initializer>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitializerUnaryOp {
    Plus,
    Minus,
    LogicalNot,
    BitwiseNot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitializerBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Shl,
    Shr,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    BitAnd,
    BitXor,
    BitOr,
    LogicalAnd,
    LogicalOr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitializerCastType {
    S64,
    U64,
}
