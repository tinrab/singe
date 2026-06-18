#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub fn start_position(self, source: &str) -> SourcePosition {
        SourcePosition::from_offset(source, self.start)
    }

    pub fn end_position(self, source: &str) -> SourcePosition {
        SourcePosition::from_offset(source, self.end)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourcePosition {
    pub line: usize,
    pub column: usize,
}

impl SourcePosition {
    pub fn from_offset(source: &str, offset: usize) -> Self {
        let clamped = offset.min(source.len());
        let prefix = &source[..clamped];
        let line = prefix.bytes().filter(|b| *b == b'\n').count() + 1;
        let column = match prefix.rfind('\n') {
            Some(index) => prefix[index + 1..].chars().count() + 1,
            None => prefix.chars().count() + 1,
        };
        Self { line, column }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateSpace {
    Reg,
    SReg,
    Const,
    Global,
    Local,
    Param,
    Shared,
    Tex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    B8,
    B16,
    B32,
    B64,
    B128,
    S8,
    S16,
    S32,
    S64,
    U8,
    U16,
    U32,
    U64,
    F16,
    F16x2,
    F32,
    F64,
    Bf16,
    Tf32,
    Pred,
    TexRef,
    SamplerRef,
    SurfRef,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatRoundingMode {
    Rn,
    Rna,
    Rz,
    Rm,
    Rp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntRoundingMode {
    Rni,
    Rzi,
    Rmi,
    Rpi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Lo,
    Ls,
    Hi,
    Hs,
    Equ,
    Neu,
    Ltu,
    Leu,
    Gtu,
    Geu,
    Num,
    Nan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MulMode {
    Hi,
    Lo,
    Wide,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkingDirective {
    Extern,
    Visible,
    Weak,
    Common,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorSize {
    V2,
    V4,
}
