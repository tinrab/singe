//! Moves, conversions, and address conversions.

use crate::{
    ast::{FloatRoundingMode, Instruction, IntRoundingMode, Operand, ScalarType, Span, StateSpace},
    emulator::{
        compiler::FunctionCompiler,
        error::{CompileError, Result},
        ir::{self, DataOperation, FloatOperation, IntegerOperation, MemoryOperation, Operation},
        layout::scalar_bytes,
        lowering::{Head, integer, operands, require, scalar, scalar_type},
        target::PtxVersion,
    },
};

/// Lowers `mov`, including the forms that pack or unpack a vector of equal parts.
pub(crate) fn mov(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
) -> Result<Operation> {
    let span = instruction.span;
    let ty = head.ty()?;
    require(
        scalar(ty) || ty == ScalarType::Pred,
        span,
        "unsupported mov type",
    )?;
    let [d, a] = operands(instruction)?;
    if matches!(ty, ScalarType::B32 | ScalarType::B64) && head.is_empty() {
        if let Operand::Vector { elements, .. } = a {
            let part = pack_part(ty, elements.len(), span)?;
            let parts = elements
                .iter()
                .map(|element| compiler.source(element, part))
                .collect::<Result<_>>()?;
            return Ok(Operation::Data(DataOperation::Pack {
                ty,
                part,
                destination: compiler.destination(d, ty)?,
                parts,
            }));
        }
        if let Operand::Vector { elements, .. } = d {
            let part = pack_part(ty, elements.len(), span)?;
            let destinations: Vec<_> = elements
                .iter()
                .map(|element| compiler.destination(element, part))
                .collect::<Result<_>>()?;
            require(
                destinations
                    .iter()
                    .any(|destination| *destination != ir::Destination::Discard),
                span,
                "unpack cannot discard every destination",
            )?;
            let source = compiler.source(a, ty)?;
            return Ok(Operation::Data(DataOperation::Unpack {
                ty,
                part,
                destinations,
                source,
            }));
        }
    }
    Ok(Operation::Data(DataOperation::Move {
        ty,
        destination: compiler.destination(d, ty)?,
        source: compiler.move_source(a, ty)?,
    }))
}

/// Returns the type of each of `count` parts that make up a value of `ty`.
fn pack_part(ty: ScalarType, count: usize, span: Span) -> Result<ScalarType> {
    match (ty, count) {
        (ScalarType::B32, 2) | (ScalarType::B64, 4) => Ok(ScalarType::B16),
        (ScalarType::B64, 2) => Ok(ScalarType::B32),
        _ => Err(CompileError::instruction(
            span,
            "mov packs two or four parts of equal width",
        )),
    }
}

pub(crate) fn convert(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
) -> Result<Operation> {
    let span = instruction.span;
    if head.take("pack") {
        return pack(compiler, instruction, head);
    }

    let control = conversion_control(head)?;
    let name = head.next()?;
    let packed = ir::FloatFormat::parse(name).filter(|f| f.packed());
    let to = match packed {
        Some(format) if format.bf16() => ScalarType::Bf16,
        Some(_) => ScalarType::F16,
        None => scalar_type(name)
            .ok_or_else(|| CompileError::unsupported(span, "conversion destination format"))?,
    };
    let from = head.ty()?;
    validate_half(compiler, span, from, to, packed.is_some(), control)?;
    if let Some(format) = packed {
        require(
            from == ScalarType::F32 && !control.ftz && control.clamp != ir::Clamp::Saturate,
            span,
            "packed conversion requires f32 inputs",
        )?;
        let ir::ConversionRounding::Float(rounding) = control.rounding else {
            return Err(CompileError::instruction(
                span,
                "packed conversion requires rn or rz",
            ));
        };
        let [d, a, b] = operands(instruction)?;
        return Ok(Operation::Float(FloatOperation::PackedConvert {
            format,
            rounding,
            relu: control.clamp == ir::Clamp::Relu,
            finite: control.finite,
            destination: compiler.float_destination(d, format)?,
            high: compiler.source(a, from)?,
            low: compiler.source(b, from)?,
        }));
    }

    validate_scalar(span, from, to, control)?;
    let [d, a] = operands(instruction)?;
    Ok(Operation::Data(DataOperation::Convert {
        from,
        to,
        control,
        destination: if to == ScalarType::Bf16 {
            compiler.float_destination(d, ir::FloatFormat::Bf16)?
        } else {
            compiler.wide_destination(d, to)?
        },
        source: if from == ScalarType::Bf16 {
            compiler.float_source(a, ir::FloatFormat::Bf16)?
        } else {
            compiler.conversion_source(a, from)?
        },
    }))
}

/// Consumes the rounding and modifier qualifiers that precede the `cvt` types.
fn conversion_control(head: &mut Head<'_>) -> Result<ir::ConversionControl> {
    let rounding = head.rounding().map_or_else(
        || {
            [
                ("rni", IntRoundingMode::Rni),
                ("rzi", IntRoundingMode::Rzi),
                ("rmi", IntRoundingMode::Rmi),
                ("rpi", IntRoundingMode::Rpi),
            ]
            .into_iter()
            .find(|(name, _)| head.take(name))
            .map_or(ir::ConversionRounding::None, |(_, mode)| {
                ir::ConversionRounding::Integer(mode)
            })
        },
        ir::ConversionRounding::Float,
    );
    let ftz = head.take("ftz");
    let saturate = head.take("sat");
    let relu = head.take("relu");
    let finite = head.take("satfinite");
    require(
        !(saturate && relu),
        head.span,
        "special half conversions do not support ftz or sat",
    )?;
    Ok(ir::ConversionControl {
        rounding,
        ftz,
        clamp: ir::Clamp::new(saturate, relu),
        finite,
    })
}

/// Checks the bf16, `relu`, `satfinite`, and packed half conversion rules.
fn validate_half(
    compiler: &FunctionCompiler<'_, '_>,
    span: Span,
    from: ScalarType,
    to: ScalarType,
    packed: bool,
    control: ir::ConversionControl,
) -> Result<()> {
    let saturate = control.clamp == ir::Clamp::Saturate;
    let special = control.clamp == ir::Clamp::Relu || control.finite;
    if to == ScalarType::Bf16 || from == ScalarType::Bf16 {
        require(
            matches!(
                (from, to),
                (ScalarType::F32, ScalarType::Bf16) | (ScalarType::Bf16, ScalarType::F32)
            ) && (!control.ftz || from == ScalarType::F32)
                && !saturate,
            span,
            "bf16 conversion form requires sm_90",
        )?;
        if from == ScalarType::Bf16 {
            compiler.require_version(PtxVersion::new(7, 1), span)?;
        }
    }

    require(
        !special || (from == ScalarType::F32 && matches!(to, ScalarType::F16 | ScalarType::Bf16)),
        span,
        "conversion modifier requires f32 to half",
    )?;
    if control.finite {
        compiler.require_version(PtxVersion::new(8, 1), span)?;
    }

    if packed || special {
        require(
            matches!(
                control.rounding,
                ir::ConversionRounding::Float(FloatRoundingMode::Rn | FloatRoundingMode::Rz)
            ),
            span,
            "this half conversion supports rn or rz",
        )?;
    }
    require(
        !special || (!control.ftz && !saturate),
        span,
        "special half conversions do not support ftz or sat",
    )
}

/// Checks that the rounding, `ftz`, and `sat` modifiers apply to a scalar conversion.
fn validate_scalar(
    span: Span,
    from: ScalarType,
    to: ScalarType,
    control: ir::ConversionControl,
) -> Result<()> {
    let float = |ty| {
        matches!(
            ty,
            ScalarType::F16 | ScalarType::Bf16 | ScalarType::F32 | ScalarType::F64
        )
    };
    let numeric = |ty| integer(ty) || float(ty) || matches!(ty, ScalarType::U8 | ScalarType::S8);
    require(
        numeric(to) && numeric(from),
        span,
        "cvt requires integer or floating-point types",
    )?;
    let valid_rounding = match control.rounding {
        ir::ConversionRounding::None => {
            !float(from) && !float(to)
                || from == to
                || (float(from) && float(to) && from.bit_width() < to.bit_width())
        }
        ir::ConversionRounding::Integer(_) => float(from) && (!float(to) || from == to),
        ir::ConversionRounding::Float(_) => {
            float(to) && (!float(from) || from.bit_width() > to.bit_width())
        }
    };
    require(
        valid_rounding,
        span,
        "rounding modifier does not match the conversion",
    )?;
    require(
        !control.ftz || from == ScalarType::F32 || to == ScalarType::F32,
        span,
        "ftz requires an f32 source or destination",
    )?;
    if control.clamp == ir::Clamp::Saturate && !float(from) && !float(to) {
        let source_bits = scalar_bytes(from, span)? * 8;
        let destination_bits = scalar_bytes(to, span)? * 8;
        let destination_contains_source = if from.is_signed() == to.is_signed() {
            destination_bits >= source_bits
        } else {
            to.is_signed() && destination_bits > source_bits
        };
        require(
            !destination_contains_source,
            span,
            "saturation is invalid when the destination contains the entire source range",
        )?;
    }
    Ok(())
}

/// Lowers `cvta` and `cvta.to`.
pub(crate) fn address(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
) -> Result<Operation> {
    let span = instruction.span;
    let conversion = if head.take("to") {
        ir::AddressConversion::ToSpace
    } else {
        ir::AddressConversion::ToGeneric
    };
    let space = head
        .space()
        .filter(|space| *space != StateSpace::Param)
        .ok_or_else(|| {
            CompileError::instruction(span, "cvta requires global, const, local, or shared space")
        })?;
    let ty = head.ty()?;
    require(
        matches!(ty, ScalarType::U32 | ScalarType::U64),
        span,
        "cvta requires u32 or u64",
    )?;
    let [d, a] = operands(instruction)?;
    let source = if conversion == ir::AddressConversion::ToGeneric {
        compiler.move_source(a, ty)?
    } else {
        compiler.source(a, ty)?
    };
    if let ir::Source::Address { object, .. } = source {
        require(
            compiler.object_properties(object).space == space,
            a.span(),
            "cvta symbol has the wrong state space",
        )?;
    }
    Ok(Operation::Memory(MemoryOperation::ConvertAddress {
        ty,
        space,
        conversion,
        destination: compiler.destination(d, ty)?,
        source,
    }))
}

/// Lowers `isspacep`, which tests whether a generic address points into a state space.
pub(crate) fn space_test(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
) -> Result<Operation> {
    let span = instruction.span;
    let space = if head.take("param::entry") {
        Some(StateSpace::Param)
    } else {
        head.space()
    }
    .ok_or_else(|| {
        CompileError::instruction(
            span,
            "isspacep requires const, global, local, shared, or param space",
        )
    })?;
    let [d, a] = operands(instruction)?;
    let (ty, address) = match compiler.source(a, ScalarType::U64) {
        Ok(address) => (ScalarType::U64, address),
        Err(_) => (ScalarType::U32, compiler.source(a, ScalarType::U32)?),
    };
    Ok(Operation::Data(DataOperation::SpaceTest {
        ty,
        space,
        destination: compiler.destination(d, ScalarType::Pred)?,
        address,
    }))
}

fn pack(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
) -> Result<Operation> {
    let span = instruction.span;
    require(head.take("sat"), span, "integer pack requires saturation")?;
    let format = match head.next()? {
        "u2" => ir::IntegerPackFormat {
            width: ir::IntegerPackWidth::Two,
            signed: false,
        },
        "s2" => ir::IntegerPackFormat {
            width: ir::IntegerPackWidth::Two,
            signed: true,
        },
        "u4" => ir::IntegerPackFormat {
            width: ir::IntegerPackWidth::Four,
            signed: false,
        },
        "s4" => ir::IntegerPackFormat {
            width: ir::IntegerPackWidth::Four,
            signed: true,
        },
        "u8" => ir::IntegerPackFormat {
            width: ir::IntegerPackWidth::Eight,
            signed: false,
        },
        "s8" => ir::IntegerPackFormat {
            width: ir::IntegerPackWidth::Eight,
            signed: true,
        },
        "u16" => ir::IntegerPackFormat {
            width: ir::IntegerPackWidth::Sixteen,
            signed: false,
        },
        "s16" => ir::IntegerPackFormat {
            width: ir::IntegerPackWidth::Sixteen,
            signed: true,
        },
        _ => {
            return Err(CompileError::instruction(
                span,
                "invalid packed integer format",
            ));
        }
    };
    require(
        head.ty()? == ScalarType::S32,
        span,
        "integer pack sources require s32",
    )?;
    let (d, a, b, c) = if format.width == ir::IntegerPackWidth::Sixteen {
        let [d, a, b] = operands(instruction)?;
        (d, a, b, None)
    } else {
        require(
            head.ty()? == ScalarType::B32,
            span,
            "integer pack third source requires b32",
        )?;
        let [d, a, b, c] = operands(instruction)?;
        (d, a, b, Some(compiler.source(c, ScalarType::B32)?))
    };
    Ok(Operation::Integer(IntegerOperation::Pack {
        format,
        destination: compiler.destination(d, ScalarType::U32)?,
        a: compiler.source(a, ScalarType::S32)?,
        b: compiler.source(b, ScalarType::S32)?,
        c,
    }))
}
