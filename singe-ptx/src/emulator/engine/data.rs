//! Moves, conversions, comparisons, and selections.

use crate::{
    ast::{CompareOp, ScalarType},
    emulator::{
        engine::{Engine, Result, combine_predicate},
        ir::DataOperation,
        memory::{WINDOW_BYTES, Window, canonical_address},
        numeric::{self, mask},
    },
};

impl Engine {
    pub(crate) fn execute_data(&mut self, lane: usize, operation: &DataOperation) -> Result<()> {
        match operation {
            DataOperation::Move {
                ty,
                destination,
                source,
            } => self.write(lane, *destination, self.source(lane, source, *ty)?, *ty),
            DataOperation::SpaceTest {
                ty,
                space,
                destination,
                address,
            } => {
                let address = canonical_address(numeric::low_u64(self.source(lane, address, *ty)?));
                let inside = address & !(WINDOW_BYTES - 1) == Window::of(*space).base();
                self.write(lane, *destination, u128::from(inside), ScalarType::Pred);
            }
            DataOperation::Pack {
                ty,
                part,
                destination,
                parts,
            } => {
                let width = part.bit_width();
                let mut value = 0;
                for (index, source) in parts.iter().enumerate() {
                    value |= self.source(lane, source, *part)? << (index as u32 * width);
                }

                self.write(lane, *destination, value, *ty);
            }
            DataOperation::Unpack {
                ty,
                part,
                destinations,
                source,
            } => {
                let width = part.bit_width();
                let value = self.source(lane, source, *ty)?;
                for (index, destination) in destinations.iter().enumerate() {
                    let bits = (value >> (index as u32 * width)) & mask(width);
                    self.write(lane, *destination, bits, *part);
                }
            }
            DataOperation::Convert {
                from,
                to,
                control,
                destination,
                source,
            } => self.write(
                lane,
                *destination,
                numeric::convert(self.source(lane, source, *from)?, *from, *to, *control),
                *to,
            ),
            DataOperation::SelectSign {
                ty,
                selector_ty,
                ftz,
                destination,
                a,
                b,
                selector,
            } => {
                let a = self.source(lane, a, *ty)?;
                let b = self.source(lane, b, *ty)?;
                let select = numeric::compare(
                    CompareOp::Ge,
                    *selector_ty,
                    self.source(lane, selector, *selector_ty)?,
                    0,
                    *ftz,
                );
                self.write(lane, *destination, if select { a } else { b }, *ty);
            }
            DataOperation::CompareValue {
                comparison,
                from,
                to,
                ftz,
                destination,
                a,
                b,
                combine,
            } => {
                let a = self.source(lane, a, *from)?;
                let b = self.source(lane, b, *from)?;
                let boolean = self.combine(lane, *combine)?;
                let packed = *from == ScalarType::F16x2;
                let mut value = 0;
                for index in 0..if packed { 2 } else { 1 } {
                    let ty = if packed { ScalarType::F16 } else { *from };
                    let a = if packed {
                        (a >> (16 * index)) & 65535
                    } else {
                        a
                    };
                    let b = if packed {
                        (b >> (16 * index)) & 65535
                    } else {
                        b
                    };
                    let result = numeric::compare(*comparison, ty, a, b, *ftz);
                    let result = combine_predicate(result, boolean);
                    let bits = if !result {
                        0
                    } else if matches!(to, ScalarType::F16 | ScalarType::F16x2) {
                        0x3c00
                    } else if *to == ScalarType::F32 {
                        0x3f80_0000
                    } else {
                        mask(if packed { 16 } else { to.bit_width() })
                    };
                    value |= bits << (16 * index);
                }

                self.write(lane, *destination, value, *to);
            }
            DataOperation::Compare {
                comparison,
                ty,
                ftz,
                destination,
                inverse,
                a,
                b,
                combine,
            } => {
                let a = self.source(lane, a, *ty)?;
                let b = self.source(lane, b, *ty)?;
                let packed = *ty == ScalarType::F16x2;
                let compare_ty = if packed { ScalarType::F16 } else { *ty };
                let result = numeric::compare(
                    *comparison,
                    compare_ty,
                    if packed { a & 65535 } else { a },
                    if packed { b & 65535 } else { b },
                    *ftz,
                );
                let second = if packed {
                    numeric::compare(*comparison, compare_ty, a >> 16, b >> 16, *ftz)
                } else {
                    !result
                };
                let boolean = self.combine(lane, *combine)?;
                self.write(
                    lane,
                    *destination,
                    u128::from(combine_predicate(result, boolean)),
                    ScalarType::Pred,
                );
                if let Some(inverse) = inverse {
                    self.write(
                        lane,
                        *inverse,
                        u128::from(combine_predicate(second, boolean)),
                        ScalarType::Pred,
                    );
                }
            }
            DataOperation::Select {
                ty,
                destination,
                a,
                b,
                predicate,
            } => {
                // Both data operands are inputs even when only one is selected.
                let a = self.source(lane, a, *ty)?;
                let b = self.source(lane, b, *ty)?;
                self.write(
                    lane,
                    *destination,
                    if self.predicate(lane, *predicate)? {
                        a
                    } else {
                        b
                    },
                    *ty,
                );
            }
        }
        Ok(())
    }
}
