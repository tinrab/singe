use crate::ast::Span;

#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    Register { span: Span, name: String },
    Immediate { span: Span, immediate: Immediate },
    Address { span: Span, address: Address },
    Vector { span: Span, elements: Vec<Operand> },
    Label { span: Span, name: String },
    BitBucket { span: Span },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Immediate {
    Integer { span: Span, value: i64 },
    UnsignedInteger { span: Span, value: u64 },
    Float { span: Span, value: f64 },
    HexFloat32 { span: Span, value: f32 },
    HexFloat64 { span: Span, value: f64 },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Address {
    Symbol {
        span: Span,
        name: String,
    },
    SymbolOffset {
        span: Span,
        symbol: String,
        offset: i64,
    },
    RegisterOffset {
        span: Span,
        register: String,
        offset: i64,
    },
}
