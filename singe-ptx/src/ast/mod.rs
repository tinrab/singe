mod common;
mod directive;
mod function;
mod instruction;
mod operand;

pub use self::{common::*, directive::*, function::*, instruction::*, operand::*};

#[derive(Debug, Clone, PartialEq)]
pub struct Module {
    pub span: Span,
    pub version: VersionDirective,
    pub target: TargetDirective,
    pub address_size: AddressSize,
    pub items: Vec<TopLevelItem>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VersionDirective {
    pub span: Span,
    pub major: u32,
    pub minor: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TargetDirective {
    pub span: Span,
    pub specifiers: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressSize {
    Bits32,
    Bits64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TopLevelItem {
    Function(Function),
    Variable(VariableDecl),
    Pragma {
        span: Span,
        value: String,
    },
    File {
        span: Span,
        index: u32,
        name: String,
        timestamp: Option<u64>,
        file_size: Option<u64>,
    },
    Alias {
        span: Span,
        alias: String,
        target: String,
    },
    Section(SectionDirective),
}

#[derive(Debug, Clone, PartialEq)]
pub struct SectionDirective {
    pub span: Span,
    pub name: String,
    pub lines: Vec<SectionLine>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SectionLine {
    Label {
        span: Span,
        name: String,
    },
    Data {
        span: Span,
        width: SectionDataWidth,
        values: Vec<SectionValue>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionDataWidth {
    B8,
    B16,
    B32,
    B64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SectionValue {
    Integer {
        span: Span,
        value: i64,
    },
    UnsignedInteger {
        span: Span,
        value: u64,
    },
    Label {
        span: Span,
        name: String,
    },
    LabelOffset {
        span: Span,
        label: String,
        offset: i64,
    },
    LabelDifference {
        span: Span,
        left: String,
        right: String,
    },
}
