mod common;
mod directive;
mod function;
mod instruction;
mod operand;

pub use crate::ast::{common::*, directive::*, function::*, instruction::*, operand::*};

/// Parsed PTX module with its required target directives and top-level declarations.
#[derive(Debug, Clone, PartialEq)]
pub struct Module {
    /// Byte range of this syntax node in the original PTX source.
    pub span: Span,
    /// PTX language version declared by the module.
    pub version: VersionDirective,
    /// Target architecture and feature specifiers.
    pub target: TargetDirective,
    /// Pointer width used by address-valued declarations.
    pub address_size: AddressSize,
    /// Module declarations and directives in source order.
    pub items: Vec<TopLevelItem>,
}

/// PTX language version declared by `.version`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VersionDirective {
    /// Byte range of this syntax node in the original PTX source.
    pub span: Span,
    /// Major PTX language version.
    pub major: u32,
    /// Minor PTX language version.
    pub minor: u32,
}

/// Target architecture and target features declared by `.target`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetDirective {
    /// Byte range of this syntax node in the original PTX source.
    pub span: Span,
    /// Architecture and target feature names in declaration order.
    pub specifiers: Vec<String>,
}

/// Pointer width declared by the module-level `.address_size` directive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressSize {
    /// 32-bit pointers, `.address_size 32`.
    Bits32,
    /// 64-bit pointers, `.address_size 64`.
    Bits64,
}

/// Declaration or directive that appears outside a PTX function body.
#[derive(Debug, Clone, PartialEq)]
pub enum TopLevelItem {
    /// Kernel entry point or device-function declaration.
    Function(Function),
    /// Module-level variable declaration.
    Variable(VariableDecl),
    /// Module-level compiler hint, `.pragma`.
    Pragma {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Compiler hint text from the quoted pragma string.
        value: String,
    },
    /// Source file metadata from `.file`.
    File {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Source file identifier used by `.loc` directives.
        index: u32,
        /// Quoted source filename from `.file`.
        name: String,
        /// Optional source file modification timestamp recorded by `.file`.
        timestamp: Option<u64>,
        /// Optional source file size in bytes.
        file_size: Option<u64>,
    },
    /// Symbol alias declared by `.alias`.
    Alias {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// New symbol spelling introduced by `.alias`.
        alias: String,
        /// Existing symbol named as the alias target.
        target: String,
    },
    /// Named section declared by `.section`.
    Section(SectionDirective),
}

/// Named PTX section containing labels and constant data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SectionDirective {
    /// Byte range of this syntax node in the original PTX source.
    pub span: Span,
    /// Section name exactly as declared.
    pub name: String,
    /// Section labels and data in source order.
    pub lines: Vec<SectionLine>,
}

/// Label or data declaration inside a named PTX section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SectionLine {
    /// Named position within the section.
    Label {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Label name.
        name: String,
    },
    /// One or more values with a declared storage width.
    Data {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Storage width of each datum in this declaration.
        width: SectionDataWidth,
        /// Section data values in source order.
        values: Vec<SectionValue>,
    },
}

/// Bit width of one datum in a PTX section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionDataWidth {
    /// Uninterpreted bit pattern with 8 bits, `.b8`.
    B8,
    /// Uninterpreted bit pattern with 16 bits, `.b16`.
    B16,
    /// Uninterpreted bit pattern with 32 bits, `.b32`.
    B32,
    /// Uninterpreted bit pattern with 64 bits, `.b64`.
    B64,
}

/// Integer or symbolic relocation expression in a PTX section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SectionValue {
    /// Signed integer datum.
    Integer {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Integer literal value before target-width conversion.
        value: i64,
    },
    /// Unsigned integer datum.
    UnsignedInteger {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Integer literal value before target-width conversion.
        value: u64,
    },
    /// Address of a label.
    Label {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Label whose address the datum stores.
        name: String,
    },
    /// Label address plus a signed byte offset.
    LabelOffset {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Referenced label symbol.
        label: String,
        /// Signed byte displacement from the label.
        offset: i64,
    },
    /// Difference between two label addresses.
    LabelDifference {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Label whose address is the minuend.
        left: String,
        /// Label whose address is the subtrahend.
        right: String,
    },
}
