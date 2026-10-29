use crate::ast::{
    Attribute, Instruction, LinkingDirective, ScalarType, Span, StateSpace, VariableDecl,
};

/// PTX kernel entry point, device function, or external function declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    /// Byte range of this syntax node in the original PTX source.
    pub span: Span,
    /// Explicit symbol linkage, or `None` for module-local linkage.
    pub linkage: Option<LinkingDirective>,
    /// Whether `.entry` marks this function as a kernel entry point.
    pub entry: bool,
    /// Function or kernel entry-point symbol.
    pub name: String,
    /// Declaration attributes in source order.
    pub attributes: Vec<Attribute>,
    /// Function-level directives in source order.
    pub directives: Vec<FunctionDirective>,
    /// Input parameters in declaration order.
    pub params: Vec<Parameter>,
    /// Return parameters in declaration order.
    pub return_params: Vec<Parameter>,
    /// Function statements, or `None` for a declaration without a body.
    pub body: Option<Vec<Statement>>,
}

/// PTX input or return parameter with its storage and pointer qualifiers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Parameter {
    /// Byte range of this syntax node in the original PTX source.
    pub span: Span,
    /// Formal parameter symbol.
    pub name: String,
    /// Scalar element type of the parameter storage.
    pub ty: ScalarType,
    /// Explicit PTX storage space, or `None` when the declaration omits it.
    pub state_space: Option<StateSpace>,
    /// Explicit alignment in bytes, or `None` when no `.align` qualifier appears.
    pub alignment: Option<u32>,
    /// Array dimensions in source order. A `None` entry represents an unsized dimension.
    pub array_bounds: Vec<Option<u32>>,
    /// Optional pointee state space and alignment from the `.ptr` qualifier.
    pub pointer: Option<PointerAnnotation>,
}

/// Pointee properties, separate from the parameter's own storage layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PointerAnnotation {
    /// Explicit PTX storage space, or `None` when the declaration omits it.
    pub state_space: Option<StateSpace>,
    /// Explicit alignment in bytes, or `None` when no `.align` qualifier appears.
    pub alignment: Option<u32>,
}

/// Label, instruction, declaration, directive, or nested block in a function body.
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    /// Named location in the function instruction stream.
    Label {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Label name, used as a branch target.
        name: String,
    },
    /// Executable PTX instruction.
    Instruction(Instruction),
    /// Function-local variable declaration.
    Variable(VariableDecl),
    /// Function-body metadata or control-flow directive.
    Directive(DirectiveStatement),
    /// Nested lexical block of PTX statements.
    Block {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Nested statements in source order.
        statements: Vec<Self>,
    },
}

/// Function-level PTX directive for launch limits, ABI registers, or compiler hints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FunctionDirective {
    /// Marks a function that does not return, `.noreturn`.
    NoReturn {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
    },
    /// Maximum registers allocated per thread, `.maxnreg`.
    MaxNReg {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Maximum register count.
        value: u32,
    },
    /// Maximum CTA dimensions, `.maxntid`.
    MaxNTid {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Maximum CTA dimensions; one to three values, in x, y, z order.
        values: Vec<u32>,
    },
    /// Required CTA dimensions, `.reqntid`.
    ReqNTid {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Required CTA dimensions; one to three values, in x, y, z order.
        values: Vec<u32>,
    },
    /// Minimum resident CTAs per multiprocessor, `.minnctapersm`.
    MinNCtaPerSm {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Minimum number of resident CTAs.
        value: u32,
    },
    /// Number of general-purpose registers preserved across calls, `.abi_preserve`.
    AbiPreserve {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Number of general-purpose registers.
        value: u32,
    },
    /// Number of control registers preserved across calls, `.abi_preserve_control`.
    AbiPreserveControl {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Number of control registers.
        value: u32,
    },
    /// Compiler hint string from a `.pragma` directive.
    Pragma {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Compiler hint text from the quoted pragma string.
        value: String,
    },
}

/// PTX directive that appears within a function body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DirectiveStatement {
    /// Compiler hint string from `.pragma`.
    Pragma {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Compiler hint text from the quoted pragma string.
        value: String,
    },
    /// Source mapping from `.loc`.
    Loc(LocDirective),
    /// Possible destinations of an indirect branch, `.branchtargets`.
    BranchTargets {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Allowed branch destinations in declaration order.
        labels: Vec<LabelPattern>,
    },
    /// Possible destinations of an indirect call, `.calltargets`.
    CallTargets {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Allowed device-function targets in declaration order.
        targets: Vec<String>,
    },
    /// Signature of an indirect call, `.callprototype`.
    CallPrototype {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Parameter and return signature for the indirect call.
        prototype: CallPrototype,
    },
}

/// Source location metadata from a PTX `.loc` directive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocDirective {
    /// Byte range of this syntax node in the original PTX source.
    pub span: Span,
    /// File index declared by a matching `.file` directive.
    pub file: u32,
    /// Source line recorded by `.loc`.
    pub line: u32,
    /// Source column recorded by `.loc`.
    pub column: u32,
    /// Optional function symbol associated with this source location.
    pub function_name: Option<LocFunctionName>,
    /// Optional call-site location for an inlined instruction.
    pub inlined_at: Option<LocInlineSite>,
}

/// Function symbol reference attached to a PTX source location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocFunctionName {
    /// Function label without a byte displacement.
    Label {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Referenced label symbol.
        label: String,
    },
    /// Function label plus a signed byte displacement.
    LabelOffset {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Referenced label symbol.
        label: String,
        /// Signed byte displacement from the label.
        offset: i64,
    },
}

/// Original call-site location for an inlined PTX instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocInlineSite {
    /// Byte range of this syntax node in the original PTX source.
    pub span: Span,
    /// File index declared by a matching `.file` directive.
    pub file: u32,
    /// Source line of the inlined call site.
    pub line: u32,
    /// Source column of the inlined call site.
    pub column: u32,
}

/// Single label or numbered label range in a branch-target directive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LabelPattern {
    /// Single explicitly named label.
    Name {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Label name.
        name: String,
    },
    /// Numbered labels sharing a prefix and a declared count.
    Range {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Common spelling before the numbered label suffix.
        prefix: String,
        /// Number of labels represented by the numbered range.
        count: u32,
    },
}

/// Indirect-call signature declared by a PTX `.callprototype` directive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallPrototype {
    /// Byte range of this syntax node in the original PTX source.
    pub span: Span,
    /// Return parameters in declaration order.
    pub return_params: Vec<Parameter>,
    /// Prototype name referenced by an indirect call instruction.
    pub placeholder: String,
    /// Input parameters in declaration order.
    pub params: Vec<Parameter>,
    /// Function-level directives in source order.
    pub directives: Vec<FunctionDirective>,
}
