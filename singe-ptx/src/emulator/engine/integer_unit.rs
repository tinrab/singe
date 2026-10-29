//! Integer unit: integer arithmetic, logic, and bit-field operations.

use crate::{
    ast::{MulMode, ScalarType},
    emulator::{
        engine::{Engine, Result},
        execution_error::ExecutionError,
        integer,
        ir::{BitUnary, BitwiseBinary, CarryArithmetic, Half, IntegerOperation},
        numeric::{self, mask},
    },
};

impl Engine {
    pub fn execute_integer(&mut self, lane: usize, operation: &IntegerOperation) -> Result<()> {
        match operation {
            IntegerOperation::DotProduct {
                mode,
                a_ty,
                b_ty,
                destination,
                a,
                b,
                c,
            } => {
                let ty = if a_ty.is_signed() || b_ty.is_signed() {
                    ScalarType::S32
                } else {
                    ScalarType::U32
                };
                let value = integer::dot(
                    *mode,
                    self.b32_source(lane, a)?,
                    self.b32_source(lane, b)?,
                    numeric::low_u32(self.source(lane, c, ty)?),
                    a_ty.is_signed(),
                    b_ty.is_signed(),
                );
                self.write(lane, *destination, value, ty);
            }
            IntegerOperation::Pack {
                format,
                destination,
                a,
                b,
                c,
            } => {
                let a = numeric::low_u32(self.source(lane, a, ScalarType::S32)?).cast_signed();
                let b = numeric::low_u32(self.source(lane, b, ScalarType::S32)?).cast_signed();
                let c = numeric::low_u32(
                    c.as_ref()
                        .map(|c| self.source(lane, c, ScalarType::B32))
                        .transpose()?
                        .unwrap_or(0),
                );
                self.write(
                    lane,
                    *destination,
                    integer::pack(a, b, c, *format),
                    ScalarType::U32,
                );
            }
            IntegerOperation::Multiply24 {
                ty,
                half,
                saturate,
                destination,
                a,
                b,
                c,
            } => {
                let a = self.source(lane, a, *ty)?;
                let b = self.source(lane, b, *ty)?;
                let c = c
                    .as_ref()
                    .map(|c| self.source(lane, c, *ty))
                    .transpose()?
                    .unwrap_or(0);
                self.write(
                    lane,
                    *destination,
                    integer::multiply24(a, b, c, ty.is_signed(), *half, *saturate),
                    *ty,
                );
            }
            IntegerOperation::SumDifference {
                ty,
                destination,
                a,
                b,
                c,
            } => {
                let value = integer::sum_difference(
                    self.source(lane, a, *ty)?,
                    self.source(lane, b, *ty)?,
                    self.source(lane, c, *ty)?,
                    *ty,
                );
                self.write(lane, *destination, value, *ty);
            }
            IntegerOperation::Carryless {
                half,
                destination,
                a,
                b,
                c,
            } => self.write(
                lane,
                *destination,
                integer::carryless(
                    numeric::low_u64(self.source(lane, a, ScalarType::U64)?),
                    numeric::low_u64(self.source(lane, b, ScalarType::U64)?),
                    numeric::low_u64(self.source(lane, c, ScalarType::U64)?),
                    *half,
                ),
                ScalarType::U64,
            ),
            IntegerOperation::FindSet {
                destination,
                mask,
                base,
                offset,
            } => {
                let value = integer::find_set(
                    self.b32_source(lane, mask)?,
                    self.u32_source(lane, base)?,
                    numeric::low_u32(self.source(lane, offset, ScalarType::S32)?).cast_signed(),
                )?;
                self.write(lane, *destination, u128::from(value), ScalarType::B32);
            }
            IntegerOperation::MaskBits {
                range,
                destination,
                position,
                length,
            } => self.write(
                lane,
                *destination,
                u128::from(integer::bit_mask(
                    self.u32_source(lane, position)?,
                    self.u32_source(lane, length)?,
                    *range,
                )),
                ScalarType::B32,
            ),
            IntegerOperation::ExtendBits {
                range,
                ty,
                destination,
                value,
                length,
            } => self.write(
                lane,
                *destination,
                integer::extend_bits(
                    numeric::low_u32(self.source(lane, value, *ty)?),
                    self.u32_source(lane, length)?,
                    *range,
                    ty.is_signed(),
                ),
                *ty,
            ),
            IntegerOperation::Carry {
                ty,
                arithmetic,
                input,
                output,
                destination,
            } => {
                let carry_in = if *input {
                    self.lanes[lane]
                        .carry
                        .ok_or_else(|| ExecutionError::Indeterminate("carry flag".into()))?
                } else {
                    false
                };
                let bits = ty.bit_width();
                let (a, b, subtract) = match arithmetic {
                    CarryArithmetic::Add { a, b } | CarryArithmetic::Subtract { a, b } => (
                        self.source(lane, a, *ty)?,
                        self.source(lane, b, *ty)?,
                        matches!(arithmetic, CarryArithmetic::Subtract { .. }),
                    ),
                    CarryArithmetic::MultiplyAdd { a, b, c, half } => {
                        let mode = match half {
                            Half::Low => MulMode::Lo,
                            Half::High => MulMode::Hi,
                        };
                        let a = self.source(lane, a, *ty)?;
                        let b = self.source(lane, b, *ty)?;
                        let product = integer::multiply(a, b, 0, *ty, mode, false);
                        (product & mask(bits), self.source(lane, c, *ty)?, false)
                    }
                };
                let (value, carry_out) = integer::carry(a, b, carry_in, subtract, bits);
                self.write(lane, *destination, value, *ty);
                if *output {
                    self.lanes[lane].carry = Some(carry_out);
                }
            }
            IntegerOperation::Unary {
                op,
                ty,
                destination,
                source,
            } => {
                let value = integer::unary(*op, *ty, self.source(lane, source, *ty)?);
                self.write(lane, *destination, value, *ty);
            }
            IntegerOperation::BitUnary {
                op,
                ty,
                destination,
                source,
            } => {
                let value = integer::bit_unary(*op, *ty, self.source(lane, source, *ty)?);
                let result_ty = if matches!(op, BitUnary::Brev | BitUnary::Cnot) {
                    *ty
                } else {
                    ScalarType::U32
                };
                self.write(lane, *destination, value, result_ty);
            }
            IntegerOperation::Extract {
                ty,
                destination,
                value,
                position,
                length,
            } => {
                let value = integer::extract(
                    self.source(lane, value, *ty)?,
                    self.u32_source(lane, position)?,
                    self.u32_source(lane, length)?,
                    *ty,
                );
                self.write(lane, *destination, value, *ty);
            }
            IntegerOperation::Insert {
                ty,
                destination,
                value,
                base,
                position,
                length,
            } => {
                let value = integer::insert(
                    self.source(lane, value, *ty)?,
                    self.source(lane, base, *ty)?,
                    self.u32_source(lane, position)?,
                    self.u32_source(lane, length)?,
                    ty.bit_width(),
                );
                self.write(lane, *destination, value, *ty);
            }
            IntegerOperation::Lop3 {
                destination,
                a,
                b,
                c,
                lut,
                predicate,
            } => {
                let result = integer::lop3(
                    self.b32_source(lane, a)?,
                    self.b32_source(lane, b)?,
                    self.b32_source(lane, c)?,
                    *lut,
                );
                let predicate = predicate
                    .map(|output| {
                        self.predicate(lane, output.combine.predicate)
                            .map(|rhs| (output.destination, output.combine.op, rhs))
                    })
                    .transpose()?;
                self.write(lane, *destination, u128::from(result), ScalarType::B32);
                if let Some((destination, op, rhs)) = predicate {
                    let value = op.apply(result != 0, rhs);
                    self.write(lane, destination, u128::from(value), ScalarType::Pred);
                }
            }
            IntegerOperation::Funnel {
                direction,
                range,
                destination,
                a,
                b,
                amount,
            } => {
                let value = integer::funnel(
                    self.b32_source(lane, a)?,
                    self.b32_source(lane, b)?,
                    self.u32_source(lane, amount)?,
                    *direction,
                    *range,
                );
                self.write(lane, *destination, u128::from(value), ScalarType::B32);
            }
            IntegerOperation::Permute {
                mode,
                destination,
                a,
                b,
                control,
            } => {
                let value = integer::permute(
                    *mode,
                    self.b32_source(lane, a)?,
                    self.b32_source(lane, b)?,
                    self.u32_source(lane, control)?,
                );
                self.write(lane, *destination, u128::from(value), ScalarType::B32);
            }
            IntegerOperation::Binary {
                op,
                ty,
                saturate,
                destination,
                a,
                b,
            } => {
                let a = self.source(lane, a, *ty)?;
                let b = self.source(lane, b, *ty)?;
                match integer::binary(*op, *ty, a, b) {
                    Some(value) if *saturate => {
                        let value = integer::saturate_s32(value.cast_signed());
                        self.write(lane, *destination, value, *ty);
                    }
                    Some(value) => self.write(lane, *destination, value, *ty),
                    None => self.write_unspecified(lane, *destination),
                }
            }
            IntegerOperation::Multiply {
                saturate,
                ty,
                mode,
                destination,
                a,
                b,
                c,
            } => {
                let result_ty = integer::product_type(*ty, *mode);
                let a = self.source(lane, a, *ty)?;
                let b = self.source(lane, b, *ty)?;
                let addend = c
                    .as_ref()
                    .map(|addend| self.source(lane, addend, result_ty))
                    .transpose()?
                    .unwrap_or(0);
                let value = integer::multiply(a, b, addend, *ty, *mode, *saturate);
                self.write(lane, *destination, value, result_ty);
            }
            IntegerOperation::Bitwise {
                op,
                ty,
                destination,
                a,
                b,
            } => {
                let a = self.source(lane, a, *ty)?;
                let b = self.source(lane, b, *ty)?;
                let value = match op {
                    BitwiseBinary::And => a & b,
                    BitwiseBinary::Or => a | b,
                    BitwiseBinary::Xor => a ^ b,
                };
                self.write(lane, *destination, value, *ty);
            }
            IntegerOperation::Not {
                ty,
                destination,
                source,
            } => self.write(lane, *destination, !self.source(lane, source, *ty)?, *ty),
            IntegerOperation::Shift {
                direction,
                ty,
                destination,
                value,
                amount,
            } => {
                let value = integer::shift(
                    self.source(lane, value, *ty)?,
                    self.source(lane, amount, ScalarType::U32)?,
                    *ty,
                    *direction,
                );
                self.write(lane, *destination, value, *ty);
            }
        }
        Ok(())
    }
}
