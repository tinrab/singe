//! Lowering turns each parsed instruction into an [`ir::Operation`]. It is the one place
//! that defines which instruction forms, operand types, and PTX versions the emulator
//! accepts.
//!
//! [`lower`] parses the opcode once and dispatches to one module per instruction family.
//! Each family consumes its qualifiers through [`Head`] in grammar order, so a qualifier
//! that no family accepts is reported by [`Head::finish`].

mod asynchronous;
mod bits;
mod compare;
mod control;
mod convert;
mod float;
mod integer;
mod matrix;
mod memory;
mod warp;

use crate::{
    ast::{FloatRoundingMode, Instruction, Operand, ScalarType, Span, StateSpace},
    emulator::{
        compiler::FunctionCompiler,
        error::{CompileError, Result},
        ir::{self, Operation},
        target::PtxVersion,
    },
};

macro_rules! opcodes {
    ($($variant:ident = $name:literal,)*) => {
        /// PTX instruction mnemonic, the first component of an instruction head.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum Opcode {
            $($variant,)*
        }

        impl Opcode {
            fn parse(name: &str) -> Option<Self> {
                match name {
                    $($name => Some(Self::$variant),)*
                    _ => None,
                }
            }
        }
    };
}

opcodes! {
    Abs = "abs",
    ActiveMask = "activemask",
    Add = "add",
    AddCarry = "addc",
    And = "and",
    Atom = "atom",
    Bar = "bar",
    Barrier = "barrier",
    Bfe = "bfe",
    Bfi = "bfi",
    Bfind = "bfind",
    Bmsk = "bmsk",
    Bra = "bra",
    Brev = "brev",
    Brkpt = "brkpt",
    Call = "call",
    Clmad = "clmad",
    Clz = "clz",
    Cnot = "cnot",
    Copysign = "copysign",
    Cp = "cp",
    Cvt = "cvt",
    Cvta = "cvta",
    Div = "div",
    Dp2a = "dp2a",
    Dp4a = "dp4a",
    Ex2 = "ex2",
    Exit = "exit",
    Fence = "fence",
    Fma = "fma",
    Fns = "fns",
    Ld = "ld",
    Ldmatrix = "ldmatrix",
    Lop3 = "lop3",
    Mad = "mad",
    Mad24 = "mad24",
    MadCarry = "madc",
    Max = "max",
    Mbarrier = "mbarrier",
    Membar = "membar",
    Min = "min",
    Mma = "mma",
    Mov = "mov",
    Mul = "mul",
    Mul24 = "mul24",
    Neg = "neg",
    Not = "not",
    Or = "or",
    Popc = "popc",
    Prmt = "prmt",
    Rcp = "rcp",
    Red = "red",
    Redux = "redux",
    Rem = "rem",
    Ret = "ret",
    Rsqrt = "rsqrt",
    Sad = "sad",
    Selp = "selp",
    Set = "set",
    Setp = "setp",
    Shf = "shf",
    Shfl = "shfl",
    Shl = "shl",
    Shr = "shr",
    Slct = "slct",
    Sqrt = "sqrt",
    St = "st",
    Sub = "sub",
    SubCarry = "subc",
    Szext = "szext",
    Testp = "testp",
    Trap = "trap",
    Vote = "vote",
    Xor = "xor",
}

/// Cursor over an instruction's qualifiers, such as `rn` and `f32` in `add.rn.f32`.
///
/// Families consume qualifiers in grammar order, so a qualifier out of place is left
/// over and [`finish`](Self::finish) rejects it.
struct Head<'a> {
    parts: &'a [&'a str],
    span: Span,
}

impl<'a> Head<'a> {
    fn take(&mut self, value: &str) -> bool {
        if self.parts.first() == Some(&value) {
            self.parts = &self.parts[1..];
            true
        } else {
            false
        }
    }

    fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    fn next(&mut self) -> Result<&'a str> {
        let Some((first, tail)) = self.parts.split_first() else {
            return Err(CompileError::instruction(
                self.span,
                "missing instruction qualifier",
            ));
        };
        self.parts = tail;
        Ok(first)
    }

    fn finish(self) -> Result<()> {
        if self.parts.is_empty() {
            Ok(())
        } else {
            Err(CompileError::unsupported(
                self.span,
                format!("instruction qualifiers {}", self.parts.join(".")),
            ))
        }
    }

    fn ty(&mut self) -> Result<ScalarType> {
        let name = self.next()?;
        scalar_type(name)
            .ok_or_else(|| CompileError::unsupported(self.span, format!("instruction type {name}")))
    }

    /// Consumes a `lo` or `hi` half selector.
    fn half(&mut self, message: &str) -> Result<ir::Half> {
        match self.next()? {
            "lo" => Ok(ir::Half::Low),
            "hi" => Ok(ir::Half::High),
            _ => Err(CompileError::instruction(self.span, message)),
        }
    }

    /// Consumes a `wrap` or `clamp` mode.
    fn range(&mut self, message: &str) -> Result<ir::RangeMode> {
        match self.next()? {
            "wrap" => Ok(ir::RangeMode::Wrap),
            "clamp" => Ok(ir::RangeMode::Clamp),
            _ => Err(CompileError::instruction(self.span, message)),
        }
    }

    fn rounding(&mut self) -> Option<FloatRoundingMode> {
        let value = match *self.parts.first()? {
            "rn" => FloatRoundingMode::Rn,
            "rz" => FloatRoundingMode::Rz,
            "rm" => FloatRoundingMode::Rm,
            "rp" => FloatRoundingMode::Rp,
            _ => return None,
        };
        self.parts = &self.parts[1..];
        Some(value)
    }

    /// Consumes a `cta`, `gpu`, or `sys` scope qualifier.
    fn take_scope(&mut self) -> bool {
        ["cta", "gpu", "sys"]
            .into_iter()
            .any(|scope| self.take(scope))
    }

    fn space(&mut self) -> Option<StateSpace> {
        let value = match *self.parts.first()? {
            "global" => StateSpace::Global,
            "const" => StateSpace::Const,
            "shared" | "shared::cta" => StateSpace::Shared,
            "local" => StateSpace::Local,
            "param" => StateSpace::Param,
            _ => return None,
        };
        self.parts = &self.parts[1..];
        Some(value)
    }
}

/// Returns the instruction type named by a qualifier; storage-only and opaque types are excluded.
fn scalar_type(name: &str) -> Option<ScalarType> {
    ScalarType::from_suffix(name).filter(|ty| {
        !matches!(
            ty,
            ScalarType::B128
                | ScalarType::Tf32
                | ScalarType::TexRef
                | ScalarType::SamplerRef
                | ScalarType::SurfRef
        )
    })
}

fn integer(ty: ScalarType) -> bool {
    matches!(
        ty,
        ScalarType::U16
            | ScalarType::U32
            | ScalarType::U64
            | ScalarType::S16
            | ScalarType::S32
            | ScalarType::S64
    )
}

fn float(ty: ScalarType) -> bool {
    matches!(ty, ScalarType::F32 | ScalarType::F64)
}

fn scalar(ty: ScalarType) -> bool {
    integer(ty) || float(ty) || matches!(ty, ScalarType::B16 | ScalarType::B32 | ScalarType::B64)
}

fn require(condition: bool, span: Span, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(CompileError::instruction(span, message))
    }
}

fn operands<const N: usize>(instruction: &Instruction) -> Result<&[Operand; N]> {
    instruction
        .operands
        .as_slice()
        .try_into()
        .map_err(|_| CompileError::instruction(instruction.span, format!("expected {N} operands")))
}

pub fn lower(
    compiler: &mut FunctionCompiler<'_, '_>,
    instruction: &Instruction,
) -> Result<Operation> {
    let span = instruction.span;
    let parts: Vec<_> = instruction.head.split('.').collect();
    let opcode = Opcode::parse(parts[0]).ok_or_else(|| {
        CompileError::unsupported(span, format!("instruction {}", instruction.head))
    })?;
    let mut qualifiers = Head {
        parts: &parts[1..],
        span,
    };
    validate(compiler, instruction, opcode, &qualifiers)?;
    let float_form = parts
        .last()
        .and_then(|name| ir::FloatFormat::parse(name))
        .is_some();
    let carry_form = parts.contains(&"cc");
    let head = &mut qualifiers;
    let result = match opcode {
        Opcode::Add
        | Opcode::Sub
        | Opcode::Mul
        | Opcode::Div
        | Opcode::Rem
        | Opcode::Min
        | Opcode::Max
            if float_form =>
        {
            float::binary(compiler, instruction, head, opcode)?
        }
        Opcode::Neg | Opcode::Abs if float_form => {
            float::unary(compiler, instruction, head, opcode)?
        }
        Opcode::Fma => float::fma(compiler, instruction, head)?,
        Opcode::Ex2 => float::exponential(compiler, instruction, head)?,
        Opcode::Rcp | Opcode::Sqrt | Opcode::Rsqrt => {
            float::math(compiler, instruction, head, opcode)?
        }
        Opcode::Testp => float::test(compiler, instruction, head)?,
        Opcode::Copysign => float::copy_sign(compiler, instruction, head)?,
        Opcode::AddCarry | Opcode::SubCarry | Opcode::MadCarry => {
            integer::carry(compiler, instruction, head, opcode)?
        }
        Opcode::Add | Opcode::Sub | Opcode::Mad if carry_form => {
            integer::carry(compiler, instruction, head, opcode)?
        }
        Opcode::Add | Opcode::Sub | Opcode::Div | Opcode::Rem | Opcode::Min | Opcode::Max => {
            integer::binary(compiler, instruction, head, opcode)?
        }
        Opcode::Mul | Opcode::Mad => integer::multiply(compiler, instruction, head, opcode)?,
        Opcode::Mul24 | Opcode::Mad24 => integer::multiply24(compiler, instruction, head, opcode)?,
        Opcode::Sad => integer::sum_difference(compiler, instruction, head)?,
        Opcode::Clmad => integer::carryless(compiler, instruction, head)?,
        Opcode::Dp4a | Opcode::Dp2a => integer::dot(compiler, instruction, head, opcode)?,
        Opcode::Neg | Opcode::Abs => integer::unary(compiler, instruction, head, opcode)?,
        Opcode::And | Opcode::Or | Opcode::Xor | Opcode::Not => {
            integer::bitwise(compiler, instruction, head, opcode)?
        }
        Opcode::Shl | Opcode::Shr => integer::shift(compiler, instruction, head, opcode)?,
        Opcode::Popc | Opcode::Clz | Opcode::Brev | Opcode::Bfind | Opcode::Cnot => {
            bits::unary(compiler, instruction, head, opcode)?
        }
        Opcode::Bfe => bits::extract(compiler, instruction, head)?,
        Opcode::Bfi => bits::insert(compiler, instruction, head)?,
        Opcode::Lop3 => bits::lop3(compiler, instruction, head)?,
        Opcode::Prmt => bits::permute(compiler, instruction, head)?,
        Opcode::Shf => bits::funnel(compiler, instruction, head)?,
        Opcode::Fns => bits::find_set(compiler, instruction, head)?,
        Opcode::Bmsk => bits::mask(compiler, instruction, head)?,
        Opcode::Szext => bits::extend(compiler, instruction, head)?,
        Opcode::Setp => compare::compare(compiler, instruction, head)?,
        Opcode::Set => compare::set(compiler, instruction, head)?,
        Opcode::Selp => compare::select(compiler, instruction, head)?,
        Opcode::Slct => compare::select_sign(compiler, instruction, head)?,
        Opcode::Mov => convert::mov(compiler, instruction, head)?,
        Opcode::Cvt => convert::convert(compiler, instruction, head)?,
        Opcode::Cvta => convert::address(compiler, instruction, head)?,
        Opcode::Ld | Opcode::St => {
            memory::access(compiler, instruction, head, opcode == Opcode::St)?
        }
        Opcode::Atom | Opcode::Red => {
            memory::atomic(compiler, instruction, head, opcode == Opcode::Atom)?
        }
        Opcode::Membar | Opcode::Fence => memory::fence(instruction, head, opcode)?,
        Opcode::Cp => asynchronous::copy(compiler, instruction, head)?,
        Opcode::Mbarrier => asynchronous::barrier(compiler, instruction, head)?,
        Opcode::Ldmatrix => matrix::load(compiler, instruction, head)?,
        Opcode::Mma => matrix::multiply(compiler, instruction, head)?,
        Opcode::Bra => control::branch(compiler, instruction, head)?,
        Opcode::Call => control::call(compiler, instruction, head)?,
        Opcode::Ret => control::ret(instruction, head)?,
        Opcode::Exit | Opcode::Trap | Opcode::Brkpt => control::terminate(instruction, opcode)?,
        Opcode::Bar | Opcode::Barrier => control::barrier(compiler, instruction, head, opcode)?,
        Opcode::Shfl => warp::shuffle(compiler, instruction, head)?,
        Opcode::Vote => warp::vote(compiler, instruction, head)?,
        Opcode::ActiveMask => warp::active_mask(compiler, instruction, head)?,
        Opcode::Redux => warp::reduce(compiler, instruction, head)?,
    };
    qualifiers.finish()?;
    Ok(result)
}

/// Applies the rules shared by every opcode: `shared::cta` and `bar.cta` need PTX 7.8,
/// special registers are only read by `mov` and `cvt`, and `_` is only accepted where an
/// instruction allows discards.
fn validate(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    opcode: Opcode,
    head: &Head<'_>,
) -> Result<()> {
    if head.parts.contains(&"shared::cta")
        || (matches!(opcode, Opcode::Bar | Opcode::Barrier) && head.parts.first() == Some(&"cta"))
    {
        compiler.require_version(PtxVersion::new(7, 8), instruction.span)?;
    }
    if !matches!(opcode, Opcode::Mov | Opcode::Cvt) {
        for operand in &instruction.operands {
            compiler.reject_special_registers(operand)?;
        }
    }
    let unpack = opcode == Opcode::Mov
        && head.parts == ["b32"]
        && matches!(instruction.operands.first(), Some(Operand::Vector { .. }));
    let lop3_predicate =
        opcode == Opcode::Lop3 && matches!(head.parts.first(), Some(&"and" | &"or"));
    let discards = unpack
        || lop3_predicate
        || matches!(
            opcode,
            Opcode::Ld
                | Opcode::Setp
                | Opcode::Set
                | Opcode::Shfl
                | Opcode::Atom
                | Opcode::Mbarrier
                | Opcode::Mma
                | Opcode::Ldmatrix
        );
    require(
        discards || !instruction.operands.iter().any(contains_discard),
        instruction.span,
        "this instruction does not support discarded operands",
    )
}

fn contains_discard(operand: &Operand) -> bool {
    match operand {
        Operand::BitBucket { .. } => true,
        Operand::Vector { elements, .. } | Operand::Tuple { elements, .. } => {
            elements.iter().any(contains_discard)
        }
        Operand::Pair { first, second, .. } => contains_discard(first) || contains_discard(second),
        Operand::Negated { operand, .. } => contains_discard(operand),
        _ => false,
    }
}

fn paired_destination(
    compiler: &FunctionCompiler<'_, '_>,
    operand: &Operand,
    ty: ScalarType,
) -> Result<(ir::Destination, Option<ir::Destination>)> {
    if let Operand::Pair { first, second, .. } = operand {
        let first = compiler.destination(first, ty)?;
        let second = compiler.destination(second, ScalarType::Pred)?;
        require(
            first != ir::Destination::Discard || second != ir::Destination::Discard,
            operand.span(),
            "both paired destinations cannot be discarded",
        )?;
        Ok((first, Some(second)))
    } else {
        Ok((compiler.destination(operand, ty)?, None))
    }
}
