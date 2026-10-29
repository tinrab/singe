//! Scalar math without host floating-point arithmetic.

use rustc_apfloat::{
    Float, FloatConvert, Round,
    ieee::{Double, Single},
};

use crate::{
    ast::{FloatRoundingMode, ScalarType},
    emulator::{
        ir::{FloatMath, MathControl},
        numeric::{self, flush, rounding},
    },
};

// ln(2) in unsigned Q63 fixed-point form.
const LN_2_Q63: u128 = 0x58b9_0bfb_e8e7_bcd5;

// Take an integer square root of the normalized, scaled significand.
// The largest radicand has 107 bits, including for binary64 subnormal inputs.
fn square_root(bits: u64, fraction_bits: u32, exponent_bits: u32, mode: FloatRoundingMode) -> u64 {
    let exponent_mask = (1u64 << exponent_bits) - 1;
    let fraction_mask = (1u64 << fraction_bits) - 1;
    let exponent = (bits >> fraction_bits) & exponent_mask;
    let fraction = bits & fraction_mask;
    let sign = bits >> (fraction_bits + exponent_bits);
    let nan = (exponent_mask << fraction_bits) | (1 << (fraction_bits - 1));
    if exponent == exponent_mask && fraction != 0 {
        return if fraction_bits == 52 {
            bits | (1 << 51)
        } else {
            nan
        };
    }
    if exponent == 0 && fraction == 0 {
        return bits;
    }
    if sign != 0 {
        return nan;
    }
    if exponent == exponent_mask {
        return bits;
    }
    let encoding_offset = (1i32 << (exponent_bits - 1)) - 1;
    let (mantissa, power) = if exponent == 0 {
        let shift = fraction_bits - fraction.ilog2();
        (
            fraction << shift,
            1 - encoding_offset - (shift).cast_signed(),
        )
    } else {
        (
            (1 << fraction_bits) | fraction,
            i32::try_from(exponent).expect("floating exponent field fits i32") - encoding_offset,
        )
    };
    let radicand = u128::from(mantissa) << (fraction_bits + power.rem_euclid(2) as u32);
    let mut significand = radicand.isqrt();
    let remainder = radicand - significand * significand;
    // A midpoint would have a fractional square, so the integer radicand cannot tie.
    let increment = match mode {
        FloatRoundingMode::Rn | FloatRoundingMode::Rna => remainder > significand,
        FloatRoundingMode::Rp => remainder != 0,
        FloatRoundingMode::Rz | FloatRoundingMode::Rm => false,
    };
    significand += u128::from(increment);
    let mut power = power.div_euclid(2);
    if significand == 1u128 << (fraction_bits + 1) {
        significand >>= 1;
        power += 1;
    }
    (((power + encoding_offset) as u64) << fraction_bits)
        | (numeric::low_u64(significand) & fraction_mask)
}

fn reciprocal<F: Float>(value: F, round: Round) -> F {
    F::from_u128(1).value.div_r(value, round).value
}

/// Computes the binary32 base-2 exponential within the two-ulp bound PTX allows for
/// `ex2.approx.f32`.
pub fn exponential(bits: u128, ftz: bool) -> u128 {
    let value = flush(Single::from_bits(bits), ftz);
    if value.is_nan() {
        return 0x7fc0_0000;
    }
    if value >= Single::from_u128(128).value {
        return 0x7f80_0000;
    }
    if value <= Single::from_i128(-150).value {
        return 0;
    }

    // Range reduction gives x = integral + fraction, with fraction in [0, 1].
    // Software binary64 also handles tiny negative inputs rounding the fraction to 1.
    let wide: Double = value.convert(&mut false).value;
    let integral = wide.to_i128_r(32, Round::TowardNegative, &mut false).value;
    let fraction = (wide - Double::from_i128(integral).value).value;
    let fraction = fraction.scalbn(63).to_u128(64).value;

    // Q63 Taylor expansion of exp(fraction * ln(2)). Products fit in u128:
    // fraction <= 2^63, argument < 2^63, and every term <= 2^63.
    // Twenty terms leave truncation and fixed-point errors below 2^-55,
    // well below binary32 rounding error, including at subnormal boundaries.
    let argument = (fraction * LN_2_Q63) >> 63;
    let mut term = 1u128 << 63;
    let mut sum = term;
    for divisor in 1..=20 {
        term = ((term * argument) >> 63) / divisor;
        sum += term;
    }
    let result: Single = Double::from_u128(sum)
        .value
        .scalbn(
            i32::try_from(integral).expect("exponential range reduction bounds the integral part")
                - 63,
        )
        .convert(&mut false)
        .value;
    flush(result, ftz).to_bits()
}

fn double(op: FloatMath, value: Double, mode: FloatRoundingMode) -> Double {
    match op {
        FloatMath::Reciprocal => reciprocal(value, rounding(mode)),
        FloatMath::SquareRoot => Double::from_bits(u128::from(square_root(
            numeric::low_u64(value.to_bits()),
            52,
            11,
            mode,
        ))),
        FloatMath::ReciprocalSquareRoot => {
            let root = Double::from_bits(u128::from(square_root(
                numeric::low_u64(value.to_bits()),
                52,
                11,
                FloatRoundingMode::Rn,
            )));
            reciprocal(root, Round::NearestTiesToEven)
        }
    }
}

pub fn evaluate(op: FloatMath, ty: ScalarType, bits: u128, control: MathControl) -> u128 {
    if ty == ScalarType::F64 {
        let bits = if control.upper_word {
            bits & !0xffff_ffff
        } else {
            bits
        };
        let value = flush(Double::from_bits(bits), control.ftz);
        let result = flush(double(op, value, control.rounding), control.ftz);
        if control.upper_word {
            if result.is_nan() {
                return 0x7fff_ffff_0000_0000;
            }
            // Round to the upper-word format, then zero the low word.
            let bits = numeric::low_u64(result.to_bits());
            let low = bits & 0xffff_ffff;
            let high = (bits >> 32)
                + u64::from(low > 0x8000_0000 || (low == 0x8000_0000 && bits & (1 << 32) != 0));
            return flush(Double::from_bits(u128::from(high << 32)), true).to_bits();
        }
        result.to_bits()
    } else {
        let value = flush(Single::from_bits(bits), control.ftz);
        let result = match op {
            FloatMath::Reciprocal => reciprocal(value, rounding(control.rounding)),
            FloatMath::SquareRoot => Single::from_bits(u128::from(square_root(
                numeric::low_u64(value.to_bits()),
                23,
                8,
                control.rounding,
            ))),
            FloatMath::ReciprocalSquareRoot => {
                // Binary64 intermediates leave ample accuracy for the FP32 error bound.
                let wide: Double = value.convert(&mut false).value;
                double(op, wide, FloatRoundingMode::Rn)
                    .convert(&mut false)
                    .value
            }
        };
        flush(result, control.ftz).to_bits()
    }
}
