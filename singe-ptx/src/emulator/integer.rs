//! Integer instruction semantics, computed at the declared operand width so that Rust's
//! overflow and shift rules never apply.

use crate::{
    ast::{MulMode, ScalarType},
    emulator::{
        ExecutionError,
        ir::{
            BitUnary, DotMode, Half, IntegerBinary, IntegerPackFormat, Permute, RangeMode,
            ShiftDirection, Unary,
        },
        numeric::{self, extend, mask, sign},
    },
};

pub fn multiply24(a: u128, b: u128, c: u128, signed: bool, half: Half, saturate: bool) -> u128 {
    let a = if signed {
        sign(a & mask(24), 24)
    } else {
        (a & mask(24)).cast_signed()
    };
    let b = if signed {
        sign(b & mask(24), 24)
    } else {
        (b & mask(24)).cast_signed()
    };
    let product = (a * b) as u128;
    let part = match half {
        Half::Low => product,
        Half::High => product >> 16,
    } & mask(32);
    if saturate {
        (sign(part, 32) + sign(c, 32)).clamp(i128::from(i32::MIN), i128::from(i32::MAX)) as u128
    } else {
        part + c
    }
}

/// Adds the absolute difference of `a` and `b` to `c` (`sad`).
pub fn sum_difference(a: u128, b: u128, c: u128, ty: ScalarType) -> u128 {
    let [a, b, c] = [a, b, c].map(|value| extend(value, ty).cast_signed());
    (c + (a - b).abs()) as u128
}

pub fn carryless(a: u64, b: u64, c: u64, half: Half) -> u128 {
    let mut product = 0u128;
    for i in 0..64 {
        if a & (1 << i) != 0 {
            product ^= u128::from(b) << i;
        }
    }
    let part = match half {
        Half::Low => product,
        Half::High => product >> 64,
    };
    part ^ u128::from(c)
}

pub fn find_set(mask: u32, base: u32, offset: i32) -> Result<u32, ExecutionError> {
    if base >= 32 {
        return Err(ExecutionError::Indeterminate("fns base exceeds 31".into()));
    }
    if offset == 0 {
        return Ok(if mask & (1 << base) != 0 {
            base
        } else {
            u32::MAX
        });
    }
    let mut remaining = i64::from(offset).abs() - 1;
    let mut position = i64::from(base);
    while (0..32).contains(&position) {
        if mask & (1 << position) != 0 {
            if remaining == 0 {
                return Ok(u32::try_from(position).expect("selected bit position is in 0..32"));
            }
            remaining -= 1;
        }
        position += if offset > 0 { 1 } else { -1 };
    }
    Ok(u32::MAX)
}

pub fn bit_mask(position: u32, length: u32, range: RangeMode) -> u32 {
    let position = if range == RangeMode::Wrap {
        position & 31
    } else {
        position
    };
    let length = range.limit(length);
    if position >= 32 || length == 0 {
        0
    } else {
        numeric::low_u32(((1u64 << length) - 1) << position)
    }
}

pub fn extend_bits(value: u32, length: u32, range: RangeMode, signed: bool) -> u128 {
    let length = range.limit(length);
    if length == 0 {
        0
    } else if signed {
        sign(u128::from(value) & mask(length), length) as u128
    } else {
        u128::from(value) & mask(length)
    }
}

pub fn dot(mode: DotMode, a: u32, b: u32, c: u32, a_signed: bool, b_signed: bool) -> u128 {
    let (count, a_width, b_start) = match mode {
        DotMode::FourBytes => (4, 8, 0),
        DotMode::TwoLow => (2, 16, 0),
        DotMode::TwoHigh => (2, 16, 2),
    };
    let mut result = i128::from(c);
    for i in 0..count {
        let a = u128::from(a >> (a_width * i)) & mask(a_width);
        let b = u128::from(b >> (8 * (i + b_start))) & 255;
        let a = if a_signed {
            sign(a, a_width)
        } else {
            (a).cast_signed()
        };
        let b = if b_signed {
            sign(b, 8)
        } else {
            (b).cast_signed()
        };
        result += a * b;
    }
    result as u128
}

pub fn pack(a: i32, b: i32, c: u32, format: IntegerPackFormat) -> u128 {
    let bits = format.width.bits();
    let (low, high) = if format.signed {
        (-(1i64 << (bits - 1)), (1i64 << (bits - 1)) - 1)
    } else {
        (0, (1i64 << bits) - 1)
    };
    let a = i64::from(a).clamp(low, high) as u128 & mask(bits);
    let b = i64::from(b).clamp(low, high) as u128 & mask(bits);
    let preserved = if bits == 16 {
        0
    } else {
        u128::from(c) << (2 * bits)
    };
    preserved | (a << bits) | b
}

/// Evaluates a binary integer operation, or `None` when PTX leaves the result unspecified.
pub fn binary(op: IntegerBinary, ty: ScalarType, a: u128, b: u128) -> Option<u128> {
    use IntegerBinary::{Add, Div, Max, Min, Rem, Sub};
    if ty.is_signed() {
        let a = sign(a, ty.bit_width());
        let b = sign(b, ty.bit_width());
        let value = match op {
            Add => a.wrapping_add(b),
            Sub => a.wrapping_sub(b),
            Div => a.checked_div(b)?,
            Rem => a.checked_rem(b)?,
            Min => a.min(b),
            Max => a.max(b),
        };
        Some(value.cast_unsigned())
    } else {
        Some(match op {
            Add => a.wrapping_add(b),
            Sub => a.wrapping_sub(b),
            Div => a.checked_div(b)?,
            Rem => a.checked_rem(b)?,
            Min => a.min(b),
            Max => a.max(b),
        })
    }
}

/// Clamps a sign-extended value to the `s32` range, as `.sat` integer forms do.
pub fn saturate_s32(value: i128) -> u128 {
    value.clamp(i128::from(i32::MIN), i128::from(i32::MAX)) as u128
}

/// Negates or takes the absolute value of a signed integer.
pub fn unary(op: Unary, ty: ScalarType, value: u128) -> u128 {
    let value = sign(value, ty.bit_width());
    (match op {
        Unary::Neg => -value,
        Unary::Abs => value.abs(),
    }) as u128
}

/// Evaluates `popc`, `clz`, `brev`, `bfind`, and `cnot`.
pub fn bit_unary(op: BitUnary, ty: ScalarType, value: u128) -> u128 {
    let bits = ty.bit_width();
    let unused = 128 - bits;
    match op {
        BitUnary::Cnot => u128::from(value == 0),
        BitUnary::Popc => u128::from(value.count_ones()),
        BitUnary::Clz => u128::from(value.leading_zeros() - unused),
        BitUnary::Brev => value.reverse_bits() >> unused,
        BitUnary::Bfind { shift } => {
            // Signed inputs search for the most significant bit that differs from the sign.
            let value = if ty.is_signed() && value & (1 << (bits - 1)) != 0 {
                !value & mask(bits)
            } else {
                value
            };
            if value == 0 {
                u128::from(u32::MAX)
            } else if shift {
                u128::from(value.leading_zeros() - unused)
            } else {
                u128::from(value.ilog2())
            }
        }
    }
}

/// Extracts `length` bits at `position` (`bfe`), sign-extending signed types.
pub fn extract(value: u128, position: u32, length: u32, ty: ScalarType) -> u128 {
    let (position, length) = (position & 255, length & 255);
    let bits = ty.bit_width();
    let sign_bit =
        length != 0 && ty.is_signed() && value & (1 << (position + length - 1).min(bits - 1)) != 0;
    (0..bits)
        .filter(|&i| {
            if i < length && position + i < bits {
                value & (1 << (position + i)) != 0
            } else {
                sign_bit
            }
        })
        .fold(0, |result, i| result | (1 << i))
}

/// Inserts the low `length` bits of `value` into `base` at `position` (`bfi`).
pub fn insert(value: u128, base: u128, position: u32, length: u32, bits: u32) -> u128 {
    let (position, length) = (position & 255, length & 255);
    (0..length)
        .take_while(|i| position + i < bits)
        .fold(base, |base, i| {
            (base & !(1 << (position + i))) | (((value >> i) & 1) << (position + i))
        })
}

/// Applies a three-input lookup table to each bit (`lop3`).
pub fn lop3(a: u32, b: u32, c: u32, lut: u8) -> u32 {
    (0..8).filter(|i| lut & (1 << i) != 0).fold(0, |result, i| {
        result
            | (if i & 4 != 0 { a } else { !a })
                & (if i & 2 != 0 { b } else { !b })
                & (if i & 1 != 0 { c } else { !c })
    })
}

/// Shifts the 64-bit concatenation `b:a` and returns 32 bits (`shf`).
pub fn funnel(a: u32, b: u32, amount: u32, direction: ShiftDirection, range: RangeMode) -> u32 {
    let amount = range.limit(amount);
    let combined = (u64::from(b) << 32) | u64::from(a);
    numeric::low_u32(match direction {
        ShiftDirection::Left => (combined << amount) >> 32,
        ShiftDirection::Right => combined >> amount,
    })
}

/// Selects four bytes from the eight bytes of `b:a` (`prmt`).
pub fn permute(mode: Permute, a: u32, b: u32, control: u32) -> u32 {
    let source = (u64::from(b) << 32) | u64::from(a);
    let select = control & 3;
    (0..4u32).fold(0, |result, i| {
        let index = match mode {
            Permute::Generic => (control >> (4 * i)) & 7,
            Permute::Forward => select + i,
            Permute::Backward => (select + 8 - i) & 7,
            Permute::ReplicateByte => select,
            Permute::ClampLeft => i.max(select),
            Permute::ClampRight => i.min(select),
            Permute::ReplicateHalf => (select & 1) * 2 + (i & 1),
        };
        let byte = numeric::low_u32((source >> (index * 8)) & 255);
        // The generic mode can replicate the selected byte's sign instead.
        let byte = if mode == Permute::Generic && control & (8 << (4 * i)) != 0 {
            if byte & 128 != 0 { 255 } else { 0 }
        } else {
            byte
        };
        result | (byte << (8 * i))
    })
}

/// Returns the result type of `mul` or `mad` on `ty` in `mode`. `.wide` doubles the width.
pub fn product_type(ty: ScalarType, mode: MulMode) -> ScalarType {
    if mode != MulMode::Wide {
        return ty;
    }
    match ty {
        ScalarType::S16 => ScalarType::S32,
        ScalarType::U16 => ScalarType::U32,
        ty if ty.is_signed() => ScalarType::S64,
        _ => ScalarType::U64,
    }
}

/// Evaluates `mul` and `mad` in `.lo`, `.hi`, or `.wide` mode.
/// Saturation applies only to `mad.hi.sat.s32`.
pub fn multiply(
    a: u128,
    b: u128,
    addend: u128,
    ty: ScalarType,
    mode: MulMode,
    saturate: bool,
) -> u128 {
    let product = extend(a, ty).wrapping_mul(extend(b, ty));
    let product = if mode == MulMode::Hi {
        product >> ty.bit_width()
    } else {
        product
    };
    if saturate {
        saturate_s32(sign(product & mask(32), 32) + sign(addend, 32))
    } else {
        product.wrapping_add(addend)
    }
}

/// Shifts left, or right with sign extension for signed types.
pub fn shift(value: u128, amount: u128, ty: ScalarType, direction: ShiftDirection) -> u128 {
    let bits = ty.bit_width();
    let amount = numeric::low_u32(amount.min(u128::from(bits)));
    if direction == ShiftDirection::Left {
        value << amount
    } else if ty.is_signed() {
        (sign(value, bits) >> amount) as u128
    } else {
        value >> amount
    }
}

/// Adds or subtracts with a carry or borrow input, returning the result and carry-out.
pub fn carry(a: u128, b: u128, carry_in: bool, subtract: bool, bits: u32) -> (u128, bool) {
    let b = b + u128::from(carry_in);
    if subtract {
        (a.wrapping_sub(b), a < b)
    } else {
        let sum = a + b;
        (sum, sum > mask(bits))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S32: ScalarType = ScalarType::S32;
    const U32: ScalarType = ScalarType::U32;

    fn low32(value: u128) -> u128 {
        value & mask(32)
    }

    #[test]
    fn test_binary_leaves_division_by_zero_unspecified() {
        assert_eq!(binary(IntegerBinary::Div, U32, 7, 0), None);
        assert_eq!(binary(IntegerBinary::Rem, S32, 7, 0), None);
        assert_eq!(
            binary(
                IntegerBinary::Div,
                S32,
                u128::from(i32::MIN.cast_unsigned()),
                0xffff_ffff
            )
            .map(low32),
            Some(0x8000_0000)
        );
        assert_eq!(
            binary(IntegerBinary::Min, S32, 0xffff_ffff, 1).map(low32),
            Some(0xffff_ffff)
        );
        assert_eq!(binary(IntegerBinary::Min, U32, 0xffff_ffff, 1), Some(1));
    }

    #[test]
    fn test_bit_field_extract_and_insert_follow_ptx_clamping() {
        assert_eq!(extract(0xf0, 4, 4, U32), 0xf);
        assert_eq!(extract(0xf0, 4, 4, S32), 0xffff_ffff);
        assert_eq!(extract(0xf0, 4, 0, S32), 0);
        assert_eq!(extract(0x8000_0000, 31, 8, S32), 0xffff_ffff);
        assert_eq!(extract(0x8000_0000, 31, 8, U32), 1);
        assert_eq!(
            extract(0xff, 260, 4, U32),
            0xf,
            "position uses its low byte"
        );
        assert_eq!(insert(0b11, 0, 31, 2, 32), 0x8000_0000);
        assert_eq!(insert(0, 0xffff_ffff, 8, 8, 32), 0xffff_00ff);
        assert_eq!(insert(0xff, 0, 0, 0, 32), 0);
    }

    #[test]
    fn test_lop3_lookup_tables_select_inputs() {
        let (a, b, c) = (0xf0f0_f0f0, 0xcccc_cccc, 0xaaaa_aaaa);
        assert_eq!(lop3(a, b, c, 0xf0), a);
        assert_eq!(lop3(a, b, c, 0xcc), b);
        assert_eq!(lop3(a, b, c, 0xaa), c);
        assert_eq!(lop3(a, b, c, 0x80), a & b & c);
        assert_eq!(lop3(a, b, c, 0x96), a ^ b ^ c);
    }

    #[test]
    fn test_funnel_shifts_wrap_or_clamp_the_amount() {
        assert_eq!(
            funnel(0xf000_0000, 1, 4, ShiftDirection::Left, RangeMode::Clamp),
            0x1f
        );
        assert_eq!(
            funnel(0, 0x1234_5678, 40, ShiftDirection::Right, RangeMode::Clamp),
            0x1234_5678
        );
        assert_eq!(
            funnel(0, 0x1234_5678, 40, ShiftDirection::Right, RangeMode::Wrap),
            0x7800_0000
        );
        assert_eq!(
            funnel(0x1234_5678, 0, 0, ShiftDirection::Right, RangeMode::Wrap),
            0x1234_5678
        );
    }

    #[test]
    fn test_permute_modes_select_and_replicate_bytes() {
        let (a, b) = (0x3322_1100, 0x7766_5544);
        assert_eq!(permute(Permute::Generic, a, b, 0x3210), a);
        assert_eq!(permute(Permute::Generic, a, b, 0x7654), b);
        assert_eq!(permute(Permute::Generic, 0x80, 0, 0x8), 0x8080_80ff);
        assert_eq!(permute(Permute::Forward, a, b, 1), 0x4433_2211);
        assert_eq!(permute(Permute::Backward, a, b, 0), 0x5566_7700);
        assert_eq!(permute(Permute::ReplicateByte, a, b, 2), 0x2222_2222);
        assert_eq!(permute(Permute::ClampLeft, a, b, 2), 0x3322_2222);
        assert_eq!(permute(Permute::ClampRight, a, b, 1), 0x1111_1100);
        assert_eq!(permute(Permute::ReplicateHalf, a, b, 1), 0x3322_3322);
    }

    #[test]
    fn test_bit_unary_operations_count_from_the_declared_width() {
        assert_eq!(bit_unary(BitUnary::Clz, U32, 1), 31);
        assert_eq!(bit_unary(BitUnary::Clz, U32, 0), 32);
        assert_eq!(bit_unary(BitUnary::Popc, U32, 0xff), 8);
        assert_eq!(bit_unary(BitUnary::Brev, U32, 1), 0x8000_0000);
        assert_eq!(bit_unary(BitUnary::Cnot, U32, 5), 0);
        assert_eq!(bit_unary(BitUnary::Bfind { shift: false }, U32, 0x10), 4);
        assert_eq!(bit_unary(BitUnary::Bfind { shift: true }, U32, 1), 31);
        assert_eq!(
            bit_unary(BitUnary::Bfind { shift: false }, U32, 0),
            0xffff_ffff
        );
        assert_eq!(
            bit_unary(BitUnary::Bfind { shift: false }, S32, 0xffff_ffff),
            0xffff_ffff
        );
        assert_eq!(
            bit_unary(BitUnary::Bfind { shift: false }, S32, 0xffff_fff0),
            3
        );
    }

    #[test]
    fn test_multiply_modes_and_saturation() {
        assert_eq!(product_type(U32, MulMode::Wide), ScalarType::U64);
        assert_eq!(product_type(ScalarType::S16, MulMode::Wide), S32);
        assert_eq!(product_type(S32, MulMode::Hi), S32);
        assert_eq!(
            low32(multiply(0xffff_ffff, 1, 0, S32, MulMode::Hi, false)),
            0xffff_ffff
        );
        assert_eq!(multiply(0xffff_ffff, 1, 0, U32, MulMode::Hi, false), 0);
        assert_eq!(
            multiply(0xffff_ffff, 0xffff_ffff, 0, U32, MulMode::Wide, false) & mask(64),
            0xffff_fffe_0000_0001
        );
        let max = u128::from(i32::MAX.cast_unsigned());
        assert_eq!(
            low32(multiply(0x4000_0000, 4, max, S32, MulMode::Hi, true)),
            max,
            "mad.hi.sat.s32 clamps the sum"
        );
    }

    #[test]
    fn test_multiply24_uses_the_low_24_bits_of_each_input() {
        assert_eq!(multiply24(0xff00_0003, 5, 0, false, Half::Low, false), 15);
        assert_eq!(
            multiply24(0x00ff_ffff, 0x00ff_ffff, 0, false, Half::High, false),
            0xffff_fe00
        );
        assert_eq!(
            multiply24(0x00ff_ffff, 2, 0, true, Half::Low, false),
            0xffff_fffe
        );
        assert_eq!(
            multiply24(
                0x007f_ffff,
                0x007f_ffff,
                0x7fff_ffff,
                true,
                Half::High,
                true
            ),
            0x7fff_ffff
        );
    }

    #[test]
    fn test_carryless_multiply_selects_a_half_and_accumulates() {
        assert_eq!(carryless(3, 3, 0, Half::Low), 5);
        assert_eq!(carryless(3, 3, 1, Half::Low), 4);
        assert_eq!(carryless(1 << 63, 1 << 63, 0, Half::High), 1 << 62);
    }

    #[test]
    fn test_find_set_counts_set_bits_from_the_base() {
        let mask = 0b1010_0000;
        let cases = [
            (5, 0, 5),
            (4, 0, u32::MAX),
            (0, 1, 5),
            (0, 2, 7),
            (0, 3, u32::MAX),
            (31, -1, 7),
            (31, -2, 5),
        ];
        for (base, offset, expected) in cases {
            assert_eq!(
                find_set(mask, base, offset).unwrap(),
                expected,
                "{base} {offset}"
            );
        }
        assert!(find_set(mask, 32, 0).is_err());
    }

    #[test]
    fn test_bit_mask_and_extension_clamp_or_wrap_the_length() {
        assert_eq!(bit_mask(3, 4, RangeMode::Clamp), 0x78);
        assert_eq!(bit_mask(0, 40, RangeMode::Clamp), 0xffff_ffff);
        assert_eq!(bit_mask(33, 4, RangeMode::Clamp), 0);
        assert_eq!(bit_mask(33, 4, RangeMode::Wrap), 0x1e);
        assert_eq!(bit_mask(0, 32, RangeMode::Wrap), 0);
        assert_eq!(
            low32(extend_bits(0x80, 8, RangeMode::Clamp, true)),
            0xffff_ff80
        );
        assert_eq!(extend_bits(0x1ff, 8, RangeMode::Clamp, false), 0xff);
        assert_eq!(extend_bits(0x1ff, 40, RangeMode::Wrap, false), 0xff);
        assert_eq!(extend_bits(0x1ff, 40, RangeMode::Clamp, false), 0x1ff);
        assert_eq!(extend_bits(5, 0, RangeMode::Clamp, true), 0);
    }

    #[test]
    fn test_sum_difference_uses_the_signedness_of_the_type() {
        assert_eq!(sum_difference(3, 10, 1, U32), 8);
        assert_eq!(sum_difference(0xffff_ffff, 1, 0, S32), 2);
        assert_eq!(sum_difference(0xffff_ffff, 1, 0, U32), 0xffff_fffe);
    }

    #[test]
    fn test_dot_products_select_bytes_and_halves() {
        assert_eq!(
            dot(
                DotMode::FourBytes,
                0x0102_0304,
                0x0101_0101,
                5,
                false,
                false
            ),
            15
        );
        assert_eq!(
            low32(dot(DotMode::FourBytes, 0xff, 2, 0, true, false)),
            0xffff_fffe
        );
        assert_eq!(
            dot(DotMode::TwoLow, 0x0003_0002, 0x0000_0504, 0, false, false),
            23
        );
        assert_eq!(
            dot(DotMode::TwoHigh, 0x0003_0002, 0x0504_0000, 0, false, false),
            23
        );
    }

    #[test]
    fn test_integer_pack_saturates_each_field() {
        use crate::emulator::ir::IntegerPackWidth;

        let format = |width, signed| IntegerPackFormat { width, signed };
        assert_eq!(
            pack(300, -5, 0x1234, format(IntegerPackWidth::Eight, false)),
            0x1234_ff00
        );
        assert_eq!(pack(5, -9, 0, format(IntegerPackWidth::Two, true)), 0b0110);
        assert_eq!(
            pack(40000, -40000, 0, format(IntegerPackWidth::Sixteen, true)),
            0x7fff_8000
        );
    }

    #[test]
    fn test_shifts_and_carries_respect_the_operand_width() {
        assert_eq!(
            low32(shift(0x8000_0000, 40, S32, ShiftDirection::Right)),
            0xffff_ffff
        );
        assert_eq!(shift(0x8000_0000, 40, U32, ShiftDirection::Right), 0);
        assert_eq!(low32(shift(1, 32, U32, ShiftDirection::Left)), 0);
        assert_eq!(
            carry(0xffff_ffff, 0, true, false, 32),
            (0x1_0000_0000, true)
        );
        assert_eq!(carry(5, 2, false, false, 32), (7, false));
        assert!(carry(0, 0, true, true, 32).1);
        assert_eq!(carry(3, 1, true, true, 32), (1, false));
    }
}
