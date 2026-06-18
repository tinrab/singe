use crate::ast::{Attribute, LinkingDirective, ScalarType, Span, StateSpace};

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub span: Span,
    pub linkage: Option<LinkingDirective>,
    pub entry: bool,
    pub name: String,
    pub attributes: Vec<Attribute>,
    pub directives: Vec<FunctionDirective>,
    pub params: Vec<Parameter>,
    pub return_params: Vec<Parameter>,
    pub body: Option<Vec<Statement>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    pub span: Span,
    pub name: String,
    pub ty: ScalarType,
    pub state_space: Option<StateSpace>,
    pub alignment: Option<u32>,
    pub array_bounds: Vec<Option<u32>>,
    pub ptr: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Label {
        span: Span,
        name: String,
    },
    Instruction(super::Instruction),
    Variable(super::VariableDecl),
    Directive(DirectiveStatement),
    Block {
        span: Span,
        statements: Vec<Statement>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum FunctionDirective {
    NoReturn { span: Span },
    MaxNReg { span: Span, value: u32 },
    MaxNTid { span: Span, values: Vec<u32> },
    ReqNTid { span: Span, values: Vec<u32> },
    MinNCtaPerSm { span: Span, value: u32 },
    AbiPreserve { span: Span, value: u32 },
    AbiPreserveControl { span: Span, value: u32 },
    Pragma { span: Span, value: String },
}

#[derive(Debug, Clone, PartialEq)]
pub enum DirectiveStatement {
    Pragma {
        span: Span,
        value: String,
    },
    Loc(LocDirective),
    BranchTargets {
        span: Span,
        labels: Vec<LabelPattern>,
    },
    CallTargets {
        span: Span,
        targets: Vec<String>,
    },
    CallPrototype {
        span: Span,
        prototype: CallPrototype,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct LocDirective {
    pub span: Span,
    pub file: u32,
    pub line: u32,
    pub column: u32,
    pub function_name: Option<LocFunctionName>,
    pub inlined_at: Option<LocInlineSite>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LocFunctionName {
    Label {
        span: Span,
        label: String,
    },
    LabelOffset {
        span: Span,
        label: String,
        offset: i64,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct LocInlineSite {
    pub span: Span,
    pub file: u32,
    pub line: u32,
    pub column: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LabelPattern {
    Name {
        span: Span,
        name: String,
    },
    Range {
        span: Span,
        prefix: String,
        count: u32,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct CallPrototype {
    pub span: Span,
    pub return_params: Vec<Parameter>,
    pub placeholder: String,
    pub params: Vec<Parameter>,
    pub directives: Vec<FunctionDirective>,
}
