//! Floating-point unit: arithmetic, math, and classification.

use crate::{
    ast::ScalarType,
    emulator::{
        engine::{Engine, Result},
        ir::{self, FloatOperation},
        math,
        numeric::{self},
    },
};

impl Engine {
    pub fn execute_float(&mut self, lane: usize, operation: &FloatOperation) -> Result<()> {
        match operation {
            FloatOperation::Unary {
                op,
                format,
                ftz,
                destination,
                source,
            } => self.write(
                lane,
                *destination,
                numeric::float_unary(
                    *op,
                    *format,
                    self.source(lane, source, format.storage())?,
                    *ftz,
                ),
                format.storage(),
            ),
            FloatOperation::Exponential {
                ftz,
                destination,
                source,
            } => self.write(
                lane,
                *destination,
                math::exponential(self.source(lane, source, ScalarType::F32)?, *ftz),
                ScalarType::F32,
            ),
            FloatOperation::Math {
                op,
                ty,
                control,
                destination,
                source,
            } => self.write(
                lane,
                *destination,
                math::evaluate(*op, *ty, self.source(lane, source, *ty)?, *control),
                *ty,
            ),
            FloatOperation::Test {
                property,
                ty,
                destination,
                source,
            } => self.write(
                lane,
                *destination,
                u128::from(numeric::test_float(
                    *property,
                    *ty,
                    self.source(lane, source, *ty)?,
                )),
                ScalarType::Pred,
            ),
            FloatOperation::Binary {
                op,
                format,
                control,
                destination,
                a,
                b,
            } => self.write(
                lane,
                *destination,
                numeric::float_binary(
                    *op,
                    *format,
                    self.source(lane, a, format.storage())?,
                    self.source(lane, b, format.storage())?,
                    *control,
                ),
                format.storage(),
            ),
            FloatOperation::Fma {
                format,
                control,
                destination,
                a,
                b,
                c,
            } => self.write(
                lane,
                *destination,
                numeric::fma(
                    *format,
                    self.source(lane, a, format.storage())?,
                    self.source(lane, b, format.storage())?,
                    self.source(lane, c, format.storage())?,
                    *control,
                ),
                format.storage(),
            ),
            FloatOperation::PackedConvert {
                format,
                rounding,
                relu,
                finite,
                destination,
                high,
                low,
            } => {
                let to = if format.bf16() {
                    ScalarType::Bf16
                } else {
                    ScalarType::F16
                };
                let high = numeric::convert(
                    self.source(lane, high, ScalarType::F32)?,
                    ScalarType::F32,
                    to,
                    ir::ConversionControl {
                        rounding: ir::ConversionRounding::Float(*rounding),
                        ftz: false,
                        clamp: ir::Clamp::new(false, *relu),
                        finite: *finite,
                    },
                );
                let low = numeric::convert(
                    self.source(lane, low, ScalarType::F32)?,
                    ScalarType::F32,
                    to,
                    ir::ConversionControl {
                        rounding: ir::ConversionRounding::Float(*rounding),
                        ftz: false,
                        clamp: ir::Clamp::new(false, *relu),
                        finite: *finite,
                    },
                );
                self.write(lane, *destination, (high << 16) | low, ScalarType::B32);
            }
            FloatOperation::CopySign {
                ty,
                destination,
                sign,
                magnitude,
            } => {
                let sign_bit = 1 << (ty.bit_width() - 1);
                let a = self.source(lane, sign, *ty)?;
                let b = self.source(lane, magnitude, *ty)?;
                self.write(lane, *destination, (b & !sign_bit) | (a & sign_bit), *ty);
            }
        }
        Ok(())
    }
}
