//! Conversions between `F64` values and the 8-bit floating formats, as `PyTorch`'s `float8_e4m3fn` and `float8_e5m2` define them.
//!
//! Encoding rounds to the nearest value, ties to even.
//! `E4M3` has no infinities, so values that round above its largest finite value become NaN; `E5M2` follows IEEE 754, so they become infinities.

/// An 8-bit floating format: one sign bit, `exponent` exponent bits, and `mantissa` mantissa bits.
#[derive(Debug, Clone, Copy)]
pub struct Minifloat {
    exponent: u32,
    mantissa: u32,
    /// Whether the format has no infinities, its all-ones exponent holding finite values but for NaN.
    finite: bool,
}

/// `float8_e4m3fn`: four exponent bits, three mantissa bits, and no infinities.
pub const E4M3: Minifloat = Minifloat {
    exponent: 4,
    mantissa: 3,
    finite: true,
};

/// `float8_e5m2`: five exponent bits, two mantissa bits, and IEEE 754 infinities.
pub const E5M2: Minifloat = Minifloat {
    exponent: 5,
    mantissa: 2,
    finite: false,
};

impl Minifloat {
    fn bias(self) -> i32 {
        (1 << (self.exponent - 1)) - 1
    }

    fn exponent_mask(self) -> u32 {
        (1 << self.exponent) - 1
    }

    fn mantissa_mask(self) -> u32 {
        (1 << self.mantissa) - 1
    }

    /// Returns the largest finite value.
    fn largest(self) -> f64 {
        let (exponent, mantissa) = if self.finite {
            (self.exponent_mask(), self.mantissa_mask() - 1)
        } else {
            (self.exponent_mask() - 1, self.mantissa_mask())
        };
        let units = f64::from((1 << self.mantissa) | mantissa);
        #[expect(
            clippy::cast_possible_wrap,
            reason = "8-bit formats have at most five exponent bits"
        )]
        let scale = exponent as i32 - self.bias() - self.mantissa as i32;
        units * 2_f64.powi(scale)
    }

    /// Returns the value `bits` encode.
    #[must_use]
    pub fn decode(self, bits: u8) -> f64 {
        let bits = u32::from(bits);
        let negative = bits & 0x80 != 0;
        let exponent = (bits >> self.mantissa) & self.exponent_mask();
        let mantissa = bits & self.mantissa_mask();
        let all_ones = exponent == self.exponent_mask();
        let magnitude = if self.finite && all_ones && mantissa == self.mantissa_mask() {
            f64::NAN
        } else if !self.finite && all_ones {
            if mantissa == 0 {
                f64::INFINITY
            } else {
                f64::NAN
            }
        } else {
            #[expect(
                clippy::cast_possible_wrap,
                reason = "8-bit formats have at most five exponent bits"
            )]
            let (units, scale) = if exponent == 0 {
                (mantissa, 1 - self.bias() - self.mantissa as i32)
            } else {
                (
                    mantissa | (1 << self.mantissa),
                    exponent as i32 - self.bias() - self.mantissa as i32,
                )
            };
            f64::from(units) * 2_f64.powi(scale)
        };
        if negative { -magnitude } else { magnitude }
    }

    /// Returns the bits of `value` rounded to the nearest value of the format, ties to even.
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap,
        reason = "the rounded mantissa is a whole number below 32, and exponents fit in 8 bits"
    )]
    pub fn encode(self, value: f64) -> u8 {
        let sign = if value.is_sign_negative() { 0x80 } else { 0 };
        // NaN keeps its sign, and in a format without infinities so does an overflow, which becomes NaN.
        let nan = sign | 0x7f;
        if value.is_nan() {
            return nan;
        }

        let magnitude = value.abs();
        let overflow = if self.finite {
            nan
        } else {
            sign | ((self.exponent_mask() << self.mantissa) as u8)
        };
        if magnitude.is_infinite() {
            return overflow;
        }

        if magnitude == 0.0 {
            return sign;
        }

        // The exponent of the binade the value lies in, clamped to the smallest normal one, below which values are subnormal.
        let mantissa = self.mantissa as i32;
        let mut exponent = (magnitude.log2().floor() as i32).max(1 - self.bias());
        // `log2` may land one binade off near powers of two.
        if magnitude >= 2_f64.powi(exponent + 1) {
            exponent += 1;
        }

        let units = (magnitude * 2_f64.powi(mantissa - exponent)).round_ties_even();
        if units * 2_f64.powi(exponent - mantissa) > self.largest() {
            return overflow;
        }

        let mut units = units as u32;
        if units >= 2 << self.mantissa {
            units >>= 1;
            exponent += 1;
        }

        let (exponent_field, mantissa_field) = if units < 1 << self.mantissa {
            (0, units)
        } else {
            (
                (exponent + self.bias()) as u32,
                units - (1 << self.mantissa),
            )
        };
        sign | ((exponent_field << self.mantissa) | mantissa_field) as u8
    }
}
