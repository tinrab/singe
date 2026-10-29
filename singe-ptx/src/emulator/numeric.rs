use std::cmp::Ordering;

use rustc_apfloat::{
    Float, FloatConvert, Round,
    ieee::{BFloat, Double, Half, Single},
};

use crate::{
    ast::{CompareOp, FloatRoundingMode, IntRoundingMode, ScalarType},
    emulator::ir::{
        Clamp, ConversionControl, ConversionRounding, FloatBinary, FloatControl, FloatFormat,
        FloatProperty, Unary,
    },
};

/// Selects the low 16 bits of a PTX register or packed word.
#[must_use]
pub fn low_u16(value: impl Into<u128>) -> u16 {
    u16::try_from(value.into() & u128::from(u16::MAX)).expect("masked to 16 bits")
}

/// Selects the low 32 bits of a PTX register or packed word.
#[must_use]
pub fn low_u32(value: impl Into<u128>) -> u32 {
    u32::try_from(value.into() & u128::from(u32::MAX)).expect("masked to 32 bits")
}

/// Selects the low 64 bits of a PTX register or packed word.
#[must_use]
pub fn low_u64(value: impl Into<u128>) -> u64 {
    u64::try_from(value.into() & u128::from(u64::MAX)).expect("masked to 64 bits")
}

pub fn mask(bits: u32) -> u128 {
    u128::MAX >> (128 - bits)
}

pub fn sign(value: u128, bits: u32) -> i128 {
    (value << (128 - bits)).cast_signed() >> (128 - bits)
}

pub fn extend(value: u128, ty: ScalarType) -> u128 {
    if ty.is_signed() {
        sign(value, ty.bit_width()) as u128
    } else {
        value & mask(ty.bit_width())
    }
}

pub fn rounding(mode: FloatRoundingMode) -> Round {
    match mode {
        FloatRoundingMode::Rn => Round::NearestTiesToEven,
        FloatRoundingMode::Rna => Round::NearestTiesToAway,
        FloatRoundingMode::Rz => Round::TowardZero,
        FloatRoundingMode::Rm => Round::TowardNegative,
        FloatRoundingMode::Rp => Round::TowardPositive,
    }
}

fn conversion_rounding(mode: ConversionRounding) -> Round {
    match mode {
        ConversionRounding::None => Round::NearestTiesToEven,
        ConversionRounding::Float(mode) => rounding(mode),
        ConversionRounding::Integer(mode) => match mode {
            IntRoundingMode::Rni => Round::NearestTiesToEven,
            IntRoundingMode::Rzi => Round::TowardZero,
            IntRoundingMode::Rmi => Round::TowardNegative,
            IntRoundingMode::Rpi => Round::TowardPositive,
        },
    }
}

pub fn flush<F: Float>(value: F, ftz: bool) -> F {
    if ftz && value.is_denormal() {
        F::ZERO.copy_sign(value)
    } else {
        value
    }
}

fn saturate_float<F: Float>(value: F, saturate: bool) -> F {
    if !saturate {
        return value;
    }
    if value.is_nan() || value < F::ZERO {
        F::ZERO
    } else {
        value.min(F::from_u128(1).value)
    }
}

fn result<F: Float>(value: F, ftz: bool, saturate: bool) -> u128 {
    let value = saturate_float(flush(value, ftz), saturate);
    if value.is_nan() && F::BITS == 64 {
        // PTX preserves binary64 payloads while quieting signaling NaNs.
        value.to_bits() | (1 << 51)
    } else if value.is_nan() {
        F::NAN.to_bits()
    } else {
        value.to_bits()
    }
}

fn lanes(format: FloatFormat, mut operation: impl FnMut(u32) -> u128) -> u128 {
    if format.packed() {
        operation(0) | (operation(16) << 16)
    } else {
        operation(0)
    }
}

fn binary<F: Float>(op: FloatBinary, a: u128, b: u128, control: FloatControl) -> u128 {
    let a = flush(F::from_bits(a), control.ftz);
    let b = flush(F::from_bits(b), control.ftz);
    let round = rounding(control.rounding);
    let value = match op {
        FloatBinary::Add => a.add_r(b, round).value,
        FloatBinary::Sub => a.sub_r(b, round).value,
        FloatBinary::Mul => a.mul_r(b, round).value,
        FloatBinary::Div => a.div_r(b, round).value,
        FloatBinary::Min | FloatBinary::Max if control.nan && (a.is_nan() || b.is_nan()) => F::NAN,
        FloatBinary::Min if a.is_zero() && b.is_zero() => {
            F::ZERO.copy_sign(if a.is_negative() { a } else { b })
        }
        FloatBinary::Max if a.is_zero() && b.is_zero() => {
            F::ZERO.copy_sign(if a.is_negative() { b } else { a })
        }
        FloatBinary::Min => a.min(b),
        FloatBinary::Max => a.max(b),
    };
    result(value, control.ftz, control.clamp == Clamp::Saturate)
}

pub fn float_binary(
    op: FloatBinary,
    format: FloatFormat,
    a: u128,
    b: u128,
    control: FloatControl,
) -> u128 {
    lanes(format, |shift| match format {
        FloatFormat::F16 | FloatFormat::F16x2 => {
            binary::<Half>(op, (a >> shift) & 65535, (b >> shift) & 65535, control)
        }
        FloatFormat::Bf16 | FloatFormat::Bf16x2 => {
            binary::<BFloat>(op, (a >> shift) & 65535, (b >> shift) & 65535, control)
        }
        FloatFormat::F32 => binary::<Single>(op, a, b, control),
        FloatFormat::F64 => binary::<Double>(op, a, b, control),
    })
}

fn fused<F: Float>(a: u128, b: u128, c: u128, control: FloatControl) -> u128 {
    let value = flush(F::from_bits(a), control.ftz)
        .mul_add_r(
            flush(F::from_bits(b), control.ftz),
            flush(F::from_bits(c), control.ftz),
            rounding(control.rounding),
        )
        .value;
    let value = if control.clamp == Clamp::Relu && value < F::ZERO {
        F::ZERO
    } else {
        value
    };
    result(value, control.ftz, control.clamp == Clamp::Saturate)
}

pub fn fma(format: FloatFormat, a: u128, b: u128, c: u128, control: FloatControl) -> u128 {
    lanes(format, |shift| match format {
        FloatFormat::F16 | FloatFormat::F16x2 => fused::<Half>(
            (a >> shift) & 65535,
            (b >> shift) & 65535,
            (c >> shift) & 65535,
            control,
        ),
        FloatFormat::Bf16 | FloatFormat::Bf16x2 => fused::<BFloat>(
            (a >> shift) & 65535,
            (b >> shift) & 65535,
            (c >> shift) & 65535,
            control,
        ),
        FloatFormat::F32 => fused::<Single>(a, b, c, control),
        FloatFormat::F64 => fused::<Double>(a, b, c, control),
    })
}

pub fn float_unary(op: Unary, format: FloatFormat, value: u128, ftz: bool) -> u128 {
    // PTX neg/abs operate on the sign bit and preserve NaN payloads.
    let bits = if format.half() {
        16
    } else if format == FloatFormat::F32 {
        32
    } else {
        64
    };
    lanes(format, |shift| {
        let mut value = (value >> shift) & mask(bits);
        let denormal = if bits == 16 {
            value & 0x7c00 == 0 && value & 0x3ff != 0
        } else {
            value & 0x7f80_0000 == 0 && value & 0x007f_ffff != 0
        };
        if ftz && denormal {
            value &= 1 << (bits - 1);
        }
        match op {
            Unary::Neg => value ^ (1 << (bits - 1)),
            Unary::Abs => value & !(1 << (bits - 1)),
        }
    })
}

fn property<F: Float>(property: FloatProperty, value: u128) -> bool {
    let value = F::from_bits(value);
    match property {
        FloatProperty::Finite => value.is_finite(),
        FloatProperty::Infinite => value.is_infinite(),
        FloatProperty::Number => !value.is_nan(),
        FloatProperty::Nan => value.is_nan(),
        // PTX classifies both signed zeros as normal numbers.
        FloatProperty::Normal => value.is_finite() && !value.is_denormal(),
        FloatProperty::Subnormal => value.is_denormal(),
    }
}

pub fn test_float(p: FloatProperty, ty: ScalarType, value: u128) -> bool {
    if ty == ScalarType::F64 {
        property::<Double>(p, value)
    } else {
        property::<Single>(p, value)
    }
}

fn float_integer<F: Float>(value: F, to: ScalarType, round: Round, source_double: bool) -> u128 {
    if value.is_nan() {
        // PTX specifies the high bit for a NaN converted to a 64-bit integer.
        return if to.bit_width() == 64 || source_double {
            1 << (to.bit_width() - 1)
        } else {
            0
        };
    }
    if to.is_signed() {
        value
            .to_i128_r(to.bit_width() as usize, round, &mut false)
            .value as u128
    } else {
        value
            .to_u128_r(to.bit_width() as usize, round, &mut false)
            .value
    }
}

fn to_float<F: Float>(value: u128, from: ScalarType, control: ConversionControl) -> F
where
    Double: FloatConvert<F>,
{
    let ConversionControl {
        rounding: mode,
        ftz,
        clamp,
        finite,
    } = control;
    let round = conversion_rounding(mode);
    let value = if matches!(
        from,
        ScalarType::F16 | ScalarType::Bf16 | ScalarType::F32 | ScalarType::F64
    ) {
        let source: Double = match from {
            ScalarType::F16 => Half::from_bits(value).convert(&mut false).value,
            ScalarType::Bf16 => BFloat::from_bits(value).convert(&mut false).value,
            ScalarType::F32 => {
                flush(Single::from_bits(value), ftz)
                    .convert(&mut false)
                    .value
            }
            _ => Double::from_bits(value),
        };
        let value: F = source.convert_r(round, &mut false).value;
        if matches!(mode, ConversionRounding::Integer(_)) {
            value.round_to_integral(round).value
        } else {
            value
        }
    } else if from.is_signed() {
        F::from_i128_r(sign(value, from.bit_width()), round).value
    } else {
        F::from_u128_r(value & mask(from.bit_width()), round).value
    };
    let value = if finite && value.is_infinite() {
        F::largest().copy_sign(value)
    } else {
        value
    };
    let value = if clamp == Clamp::Relu && value < F::ZERO {
        F::ZERO
    } else {
        value
    };
    saturate_float(value, clamp == Clamp::Saturate)
}

pub fn convert(value: u128, from: ScalarType, to: ScalarType, control: ConversionControl) -> u128 {
    let ConversionControl {
        rounding: mode,
        ftz,
        clamp,
        ..
    } = control;
    let round = conversion_rounding(mode);
    match to {
        ScalarType::F16 => result(to_float::<Half>(value, from, control), false, false),
        ScalarType::Bf16 => result(to_float::<BFloat>(value, from, control), false, false),
        ScalarType::F32 => result(to_float::<Single>(value, from, control), ftz, false),
        ScalarType::F64 => result(to_float::<Double>(value, from, control), false, false),
        _ => match from {
            ScalarType::F16 => float_integer(Half::from_bits(value), to, round, false),
            ScalarType::Bf16 => float_integer(BFloat::from_bits(value), to, round, false),
            ScalarType::F32 => {
                float_integer(flush(Single::from_bits(value), ftz), to, round, false)
            }
            ScalarType::F64 => float_integer(Double::from_bits(value), to, round, true),
            _ => {
                let value = if from.is_signed() {
                    sign(value, from.bit_width())
                } else {
                    (value & mask(from.bit_width())).cast_signed()
                };
                if clamp == Clamp::Saturate {
                    let (low, high) = if to.is_signed() {
                        (
                            -(1i128 << (to.bit_width() - 1)),
                            (1i128 << (to.bit_width() - 1)) - 1,
                        )
                    } else {
                        (0, (mask(to.bit_width())).cast_signed())
                    };
                    value.clamp(low, high) as u128
                } else {
                    value as u128
                }
            }
        },
    }
}

pub fn compare(op: CompareOp, ty: ScalarType, a: u128, b: u128, ftz: bool) -> bool {
    use CompareOp::{
        Eq, Equ, Ge, Geu, Gt, Gtu, Hi, Hs, Le, Leu, Lo, Ls, Lt, Ltu, Nan, Ne, Neu, Num,
    };

    let cmp = match ty {
        ScalarType::F16 => {
            flush(Half::from_bits(a), ftz).partial_cmp(&flush(Half::from_bits(b), ftz))
        }
        ScalarType::F32 => {
            flush(Single::from_bits(a), ftz).partial_cmp(&flush(Single::from_bits(b), ftz))
        }
        ScalarType::F64 => {
            flush(Double::from_bits(a), ftz).partial_cmp(&flush(Double::from_bits(b), ftz))
        }
        _ if ty.is_signed() => Some(sign(a, ty.bit_width()).cmp(&sign(b, ty.bit_width()))),
        _ => Some((a & mask(ty.bit_width())).cmp(&(b & mask(ty.bit_width())))),
    };
    match op {
        Eq => cmp == Some(Ordering::Equal),
        Ne => cmp.is_some() && cmp != Some(Ordering::Equal),
        Lt | Lo => cmp == Some(Ordering::Less),
        Le | Ls => matches!(cmp, Some(Ordering::Less | Ordering::Equal)),
        Gt | Hi => cmp == Some(Ordering::Greater),
        Ge | Hs => matches!(cmp, Some(Ordering::Greater | Ordering::Equal)),
        Equ => cmp.is_none() || cmp == Some(Ordering::Equal),
        Neu => cmp != Some(Ordering::Equal),
        Ltu => cmp.is_none() || cmp == Some(Ordering::Less),
        Leu => cmp != Some(Ordering::Greater),
        Gtu => cmp.is_none() || cmp == Some(Ordering::Greater),
        Geu => cmp != Some(Ordering::Less),
        Num => cmp.is_some(),
        Nan => cmp.is_none(),
    }
}

#[cfg(test)]
mod tests {
    use half::f16;

    use super::*;

    #[test]
    fn test_low_words_preserve_requested_register_bits() {
        let register = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210_u128;
        assert_eq!(low_u16(register), 0x3210);
        assert_eq!(low_u32(register), 0x7654_3210);
        assert_eq!(low_u64(register), 0xfedc_ba98_7654_3210);
        assert_eq!(low_u32(u128::MAX).cast_signed(), -1);
    }

    fn f32_bits(value: f32) -> u128 {
        u128::from(value.to_bits())
    }

    fn arithmetic(rounding: FloatRoundingMode, ftz: bool) -> FloatControl {
        FloatControl {
            rounding,
            ftz,
            clamp: Clamp::None,
            nan: false,
        }
    }

    fn integer_rounding(mode: IntRoundingMode) -> ConversionControl {
        ConversionControl {
            rounding: ConversionRounding::Integer(mode),
            ftz: false,
            clamp: Clamp::None,
            finite: false,
        }
    }

    #[test]
    fn test_float_binary_applies_the_requested_rounding() {
        let one = f32_bits(1.0);
        let half_ulp = f32_bits(2f32.powi(-24));
        let add = |rounding| {
            float_binary(
                FloatBinary::Add,
                FloatFormat::F32,
                one,
                half_ulp,
                arithmetic(rounding, false),
            )
        };
        assert_eq!(add(FloatRoundingMode::Rn), one);
        assert_eq!(add(FloatRoundingMode::Rz), one);
        assert_eq!(add(FloatRoundingMode::Rp), 0x3f80_0001);
        let third = float_binary(
            FloatBinary::Div,
            FloatFormat::F32,
            one,
            f32_bits(3.0),
            arithmetic(FloatRoundingMode::Rn, false),
        );
        assert_eq!(third, 0x3eaa_aaab);
    }

    #[test]
    fn test_float_binary_flushes_subnormals_only_with_ftz() {
        let add = |ftz| {
            float_binary(
                FloatBinary::Add,
                FloatFormat::F32,
                1,
                0,
                arithmetic(FloatRoundingMode::Rn, ftz),
            )
        };
        assert_eq!(add(false), 1);
        assert_eq!(add(true), 0);
    }

    #[test]
    fn test_packed_half_arithmetic_operates_per_lane() {
        let a = 0x4000_3c00; // (2.0, 1.0)
        let b = 0x3c00_3c00; // (1.0, 1.0)
        let control = arithmetic(FloatRoundingMode::Rn, false);
        assert_eq!(
            float_binary(FloatBinary::Add, FloatFormat::F16x2, a, b, control),
            0x4200_4000
        );
        assert_eq!(
            fma(
                FloatFormat::F32,
                f32_bits(2.0),
                f32_bits(3.0),
                f32_bits(1.0),
                control
            ),
            f32_bits(7.0)
        );
    }

    #[test]
    fn test_float_unary_and_classification_use_the_encoding() {
        assert_eq!(
            float_unary(Unary::Neg, FloatFormat::F32, f32_bits(1.0), false),
            0xbf80_0000
        );
        assert_eq!(
            float_unary(Unary::Abs, FloatFormat::F32, 0x8000_0000, false),
            0
        );
        assert_eq!(
            float_unary(Unary::Abs, FloatFormat::F32, 0x8000_0001, true),
            0
        );
        assert!(test_float(FloatProperty::Subnormal, ScalarType::F32, 1));
        assert!(test_float(FloatProperty::Normal, ScalarType::F32, 0));
        assert!(test_float(FloatProperty::Nan, ScalarType::F32, 0x7fc0_0000));
        assert!(test_float(
            FloatProperty::Infinite,
            ScalarType::F32,
            0x7f80_0000
        ));
        assert!(!test_float(
            FloatProperty::Finite,
            ScalarType::F32,
            0x7f80_0000
        ));
    }

    #[test]
    fn test_float_to_integer_conversion_rounds_and_saturates() {
        let to_s32 = |value: f32, mode| {
            low_u32(convert(
                f32_bits(value),
                ScalarType::F32,
                ScalarType::S32,
                integer_rounding(mode),
            ))
        };
        assert_eq!(to_s32(-2.5, IntRoundingMode::Rzi), (-2i32).cast_unsigned());
        assert_eq!(to_s32(2.5, IntRoundingMode::Rni), 2);
        assert_eq!(to_s32(-2.5, IntRoundingMode::Rmi), (-3i32).cast_unsigned());
        assert_eq!(to_s32(2.1, IntRoundingMode::Rpi), 3);
        assert_eq!(to_s32(3e9, IntRoundingMode::Rzi), 0x7fff_ffff);
        assert_eq!(to_s32(f32::NAN, IntRoundingMode::Rzi), 0);
    }

    #[test]
    fn test_every_fp16_pattern_matches_independent_host_comparisons() {
        // Binary32 represents every finite binary16 value exactly, so this oracle
        // does not share APFloat's comparison implementation or rounding code.
        let operations = [
            CompareOp::Eq,
            CompareOp::Ne,
            CompareOp::Lt,
            CompareOp::Le,
            CompareOp::Gt,
            CompareOp::Ge,
            CompareOp::Equ,
            CompareOp::Neu,
            CompareOp::Ltu,
            CompareOp::Leu,
            CompareOp::Gtu,
            CompareOp::Geu,
            CompareOp::Num,
            CompareOp::Nan,
        ];
        for bits in 0..=u16::MAX {
            for rhs in [
                0, 0x8000, 1, 0x8001, 0x3c00, 0xbc00, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7c01,
                0x7e01, 0xfe01,
            ] {
                for ftz in [false, true] {
                    let value = |bits: u16| {
                        let bits = if ftz && bits & 0x7c00 == 0 {
                            bits & 0x8000
                        } else {
                            bits
                        };
                        f16::from_bits(bits).to_f32()
                    };
                    let a = value(bits);
                    let b = value(rhs);
                    let unordered = a.is_nan() || b.is_nan();
                    let expected = [
                        a == b,
                        !unordered && a != b,
                        a < b,
                        a <= b,
                        a > b,
                        a >= b,
                        unordered || a == b,
                        unordered || a != b,
                        unordered || a < b,
                        unordered || a <= b,
                        unordered || a > b,
                        unordered || a >= b,
                        !unordered,
                        unordered,
                    ];
                    for (op, expected) in operations.into_iter().zip(expected) {
                        assert_eq!(
                            compare(op, ScalarType::F16, bits.into(), rhs.into(), ftz),
                            expected,
                            "{op:?} {bits:04x} {rhs:04x} ftz={ftz}"
                        );
                    }
                }
            }
        }
    }
}
