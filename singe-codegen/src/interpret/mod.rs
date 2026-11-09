//! Runs a kernel on the CPU, as the reference every emitter is checked against.
//!
//! The interpreter follows the IR's semantics directly: each store runs once for each point of its access's domain, in store order, and each reduction walks its axes in order, combining one value at a time.
//! `F32` values are rounded to `F32` after every operation, so additions, multiplications, divisions, and square roots round as an exact GPU kernel does; transcendental functions are computed in `F64` and rounded, which may differ from a GPU's by a few units in the last place.

mod minifloat;

use std::{cmp::Ordering, collections::HashMap, ops::Range};

use half::{bf16, f16};
use thiserror::Error;

pub use self::minifloat::{E4M3, E5M2, Minifloat};
use crate::{
    abi::Bindings,
    ir::{
        Access, AxisId, BinaryOp, Combiner, ElementType, Expr, Kernel, Literal, NodeId, Size,
        UnaryOp, VerifyError,
    },
};

/// A reason the interpreter cannot run a kernel.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum InterpretError {
    #[error("kernel is invalid")]
    Verify(#[from] VerifyError),
    #[error("`{what}` count mismatch (expected {expected}, got {actual})")]
    LengthMismatch {
        what: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("operand {operand} holds {actual} bytes, but the kernel addresses {required}")]
    BufferTooSmall {
        operand: usize,
        required: usize,
        actual: usize,
    },
    #[error("scalar {scalar} is bound to a literal of another kind")]
    ScalarKind { scalar: usize },
    #[error("integer division by zero")]
    DivisionByZero,
    #[error("offset arithmetic overflows")]
    Overflow,
}

/// Runs `kernel` with `bindings` on `buffers`, one byte buffer per operand in operand order, writing its outputs in place, and returns the kernel's status word: the first status code it reported, or zero.
///
/// # Errors
///
/// Returns an error if the kernel is invalid, the bindings or buffers do not match it, an access falls outside its buffer, or an integer division divides by zero.
pub fn run(
    kernel: &Kernel,
    bindings: &Bindings,
    buffers: &mut [Vec<u8>],
) -> Result<u32, InterpretError> {
    kernel.verify()?;
    let arguments = kernel.argument_sizes().count();
    if bindings.sizes.len() != arguments {
        return Err(InterpretError::LengthMismatch {
            what: "argument size",
            expected: arguments,
            actual: bindings.sizes.len(),
        });
    }

    if bindings.scalars.len() != kernel.scalars.len() {
        return Err(InterpretError::LengthMismatch {
            what: "scalar",
            expected: kernel.scalars.len(),
            actual: bindings.scalars.len(),
        });
    }

    if buffers.len() != kernel.operands.len() {
        return Err(InterpretError::LengthMismatch {
            what: "buffer",
            expected: kernel.operands.len(),
            actual: buffers.len(),
        });
    }

    let mut arguments = bindings.sizes.iter();
    let sizes = kernel
        .sizes
        .iter()
        .map(|size| match size {
            Size::Static(value) => *value,
            Size::Argument => *arguments.next().expect("argument count was checked"),
        })
        .collect::<Vec<_>>();
    let mut interpreter = Interpreter {
        kernel,
        bindings,
        sizes,
        index: vec![0; kernel.axes.len()],
        status: 0,
    };
    for store in &kernel.stores {
        let axes = kernel.access_domain(&store.access);
        let mut result = Ok(());
        interpreter.for_each_point(&axes, &mut |interpreter| {
            if result.is_err() {
                return;
            }

            result = (|| {
                let mut memo = HashMap::new();
                let value = interpreter.evaluate(store.value, &mut memo, buffers)?;
                let operand = store.access.operand.0;
                let element = kernel.operands[operand].element;
                let offset = interpreter.offset(&store.access, &mut memo, buffers)?;
                let value = match store.combiner {
                    Some(combiner) => {
                        let current = read(&buffers[operand], operand, offset, element)?;
                        round(element, combine(combiner, current, value))
                    }
                    None => value,
                };
                write(&mut buffers[operand], operand, offset, element, value)
            })();
        });
        result?;
    }
    Ok(interpreter.status)
}

/// A computed value, in the representation its element type's arithmetic uses.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Value {
    Bool(bool),
    Int(i128),
    Float(f64),
}

impl Value {
    fn float(self) -> f64 {
        match self {
            Self::Float(value) => value,
            Self::Int(value) => value as f64,
            Self::Bool(value) => f64::from(u8::from(value)),
        }
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "float-to-integer conversion truncates toward zero and saturates by definition"
    )]
    fn int(self) -> i128 {
        match self {
            Self::Int(value) => value,
            Self::Bool(value) => i128::from(value),
            // Float-to-integer conversions truncate toward zero, saturate, and map NaN to zero, as CUDA's do.
            Self::Float(value) => value as i128,
        }
    }

    fn truth(self) -> bool {
        match self {
            Self::Bool(value) => value,
            Self::Int(value) => value != 0,
            Self::Float(value) => value != 0.0,
        }
    }
}

struct Interpreter<'a> {
    kernel: &'a Kernel,
    bindings: &'a Bindings,
    sizes: Vec<u64>,
    /// The current index along each axis.
    index: Vec<u64>,
    /// The first status code a check reported, or zero.
    status: u32,
}

impl Interpreter<'_> {
    /// Calls `visit` at every point of `axes`, the last axis varying fastest.
    fn for_each_point(&mut self, axes: &[AxisId], visit: &mut dyn FnMut(&mut Self)) {
        let Some((&first, rest)) = axes.split_first() else {
            visit(self);
            return;
        };
        let extent = self.sizes[self.kernel.axes[first.0].extent.0];
        for position in 0..extent {
            self.index[first.0] = position;
            self.for_each_point(rest, visit);
        }
    }

    fn offset(
        &mut self,
        access: &Access,
        memo: &mut HashMap<NodeId, Value>,
        buffers: &[Vec<u8>],
    ) -> Result<u64, InterpretError> {
        let mut offset = access.base.map_or(0, |base| self.sizes[base.0]);
        for &(axis, stride) in &access.strides {
            offset = self.index[axis.0]
                .checked_mul(self.sizes[stride.0])
                .and_then(|term| offset.checked_add(term))
                .ok_or(InterpretError::Overflow)?;
        }

        for &(node, stride) in &access.indirect {
            let index = u64::try_from(self.evaluate(node, memo, buffers)?.int())
                .map_err(|_| InterpretError::Overflow)?;
            offset = index
                .checked_mul(self.sizes[stride.0])
                .and_then(|term| offset.checked_add(term))
                .ok_or(InterpretError::Overflow)?;
        }
        Ok(offset)
    }

    /// Evaluates `node` at the current point, reusing values `memo` holds for this point.
    fn evaluate(
        &mut self,
        node: NodeId,
        memo: &mut HashMap<NodeId, Value>,
        buffers: &[Vec<u8>],
    ) -> Result<Value, InterpretError> {
        if let Some(value) = memo.get(&node) {
            return Ok(*value);
        }

        let element = self.kernel.nodes[node.0].element;
        let value = match &self.kernel.nodes[node.0].expr {
            Expr::Load(access) => {
                let operand = access.operand.0;
                let offset = self.offset(access, memo, buffers)?;
                read(
                    &buffers[operand],
                    operand,
                    offset,
                    self.kernel.operands[operand].element,
                )?
            }
            Expr::Literal(literal) => literal_value(*literal),
            Expr::Scalar(scalar) => {
                let literal = self.bindings.scalars[scalar.0];
                let kind_matches = match literal {
                    Literal::Bool(_) => element == ElementType::Bool,
                    Literal::Int(_) => element.is_integer(),
                    Literal::Float(_) => element.is_float(),
                };
                if !kind_matches {
                    return Err(InterpretError::ScalarKind { scalar: scalar.0 });
                }
                literal_value(literal)
            }
            Expr::AxisIndex(axis) => Value::Int(i128::from(self.index[axis.0])),
            Expr::Unary(op, value) => {
                let operand = self.evaluate(*value, memo, buffers)?;
                unary(*op, operand)
            }
            Expr::Binary(op, lhs, rhs) => {
                let lhs = self.evaluate(*lhs, memo, buffers)?;
                let rhs = self.evaluate(*rhs, memo, buffers)?;
                binary(*op, self.kernel.nodes[node.0].element, lhs, rhs)?
            }
            Expr::MulAdd(a, b, c) => {
                let a = self.evaluate(*a, memo, buffers)?.float();
                let b = self.evaluate(*b, memo, buffers)?.float();
                let c = self.evaluate(*c, memo, buffers)?.float();
                // Products of `F32` values are exact in `F64`, so this rounds once to `F64` and once more to the node's type.
                Value::Float(a.mul_add(b, c))
            }
            Expr::Select {
                condition,
                if_true,
                if_false,
            } => {
                // Both values are computed before one is chosen, as emitted kernels compute every node, so checks in the discarded one still report.
                let chosen = self.evaluate(*condition, memo, buffers)?.truth();
                let if_true = self.evaluate(*if_true, memo, buffers)?;
                let if_false = self.evaluate(*if_false, memo, buffers)?;
                if chosen { if_true } else { if_false }
            }
            Expr::Cast(value) => self.evaluate(*value, memo, buffers)?,
            Expr::Require {
                value,
                condition,
                status,
            } => {
                if !self.evaluate(*condition, memo, buffers)?.truth() && self.status == 0 {
                    self.status = *status;
                }
                self.evaluate(*value, memo, buffers)?
            }
            Expr::CheckIndex {
                value,
                extent,
                wrap,
                status,
            } => {
                let extent = i128::from(self.sizes[extent.0]);
                let mut index = self.evaluate(*value, memo, buffers)?.int();
                if *wrap && index < 0 {
                    index += extent;
                }
                if (0..extent).contains(&index) {
                    Value::Int(index)
                } else {
                    if self.status == 0 {
                        self.status = *status;
                    }
                    Value::Int(0)
                }
            }
            Expr::Reduce {
                combiner,
                axes,
                value,
            } => {
                let saved = self.index.clone();
                let mut accumulator = identity(*combiner, element);
                let mut result = Ok(());
                self.for_each_point(axes, &mut |interpreter| {
                    if result.is_err() {
                        return;
                    }
                    // Values that depend on the reduced axes change at every point, so each point starts a fresh memo.
                    match interpreter.evaluate(*value, &mut HashMap::new(), buffers) {
                        Ok(value) => {
                            accumulator = round(element, combine(*combiner, accumulator, value));
                        }
                        Err(error) => result = Err(error),
                    }
                });
                self.index = saved;
                result?;
                accumulator
            }
            Expr::Scan {
                combiner,
                axis,
                value,
            } => {
                let saved = self.index.clone();
                let mut accumulator = identity(*combiner, element);
                for position in 0..=saved[axis.0] {
                    self.index[axis.0] = position;
                    // Values that depend on the axis change at every point, so each point starts a fresh memo.
                    let point = self.evaluate(*value, &mut HashMap::new(), buffers);
                    match point {
                        Ok(point) => {
                            accumulator = round(element, combine(*combiner, accumulator, point));
                        }
                        Err(error) => {
                            self.index = saved;
                            return Err(error);
                        }
                    }
                }

                self.index = saved;
                accumulator
            }
            Expr::ArgReduce {
                maximum,
                axes,
                value,
            } => {
                let saved = self.index.clone();
                let (mut best, mut position, mut count) = (None::<Value>, 0_i128, 0_i128);
                let mut result = Ok(());
                self.for_each_point(axes, &mut |interpreter| {
                    if result.is_err() {
                        return;
                    }
                    match interpreter.evaluate(*value, &mut HashMap::new(), buffers) {
                        Ok(value) => {
                            if best.is_none_or(|best| ranks_above(value, best, *maximum)) {
                                best = Some(value);
                                position = count;
                            }

                            count += 1;
                        }
                        Err(error) => result = Err(error),
                    }
                });
                self.index = saved;
                result?;
                Value::Int(position)
            }
            Expr::SoftmaxSum { axes, logit, value } => {
                let saved = self.index.clone();
                let mut points = Vec::new();
                let mut result = Ok(());
                self.for_each_point(axes, &mut |interpreter| {
                    if result.is_err() {
                        return;
                    }

                    let mut memo = HashMap::new();
                    let point =
                        interpreter
                            .evaluate(*logit, &mut memo, buffers)
                            .and_then(|logit| {
                                let value = interpreter.evaluate(*value, &mut memo, buffers)?;
                                Ok((logit.float(), value.float()))
                            });
                    match point {
                        Ok(point) => points.push(point),
                        Err(error) => result = Err(error),
                    }
                });
                self.index = saved;
                result?;
                // The exact softmax, from the largest logit, which a one-pass target matches up to rounding.
                let largest = points
                    .iter()
                    .fold(f64::NEG_INFINITY, |largest, &(logit, _)| {
                        maximum(largest, logit)
                    });
                let (mut total, mut weighted) = (0.0, 0.0);
                for &(logit, value) in &points {
                    let weight = (logit - largest).exp();
                    total += weight;
                    weighted = weight.mul_add(value, weighted);
                }
                Value::Float(weighted / total)
            }
        };
        let value = round(element, value);
        memo.insert(node, value);
        Ok(value)
    }
}

fn literal_value(literal: Literal) -> Value {
    match literal {
        Literal::Bool(value) => Value::Bool(value),
        Literal::Int(value) => Value::Int(i128::from(value)),
        Literal::Float(value) => Value::Float(value),
    }
}

/// Converts `value` to `element`'s representation, rounding floats to the type's precision and wrapping integers to its width.
#[expect(
    clippy::cast_possible_truncation,
    reason = "rounding an `F64` to `F32` is the conversion this performs"
)]
fn round(element: ElementType, value: Value) -> Value {
    match element {
        ElementType::Bool => Value::Bool(value.truth()),
        ElementType::F16 => Value::Float(f64::from(f16::from_f64(value.float()))),
        ElementType::F8E4M3 => Value::Float(E4M3.decode(E4M3.encode(value.float()))),
        ElementType::F8E5M2 => Value::Float(E5M2.decode(E5M2.encode(value.float()))),
        ElementType::Bf16 => Value::Float(f64::from(bf16::from_f64(value.float()))),
        // An integer converts to `F32` with one rounding; through `F64` it could round twice.
        ElementType::F32 => Value::Float(f64::from(match value {
            Value::Int(integer) => integer as f32,
            other => other.float() as f32,
        })),
        ElementType::F64 => Value::Float(value.float()),
        integer => {
            let value = match value {
                // Conversions from floats saturate at the type's bounds.
                Value::Float(_) => {
                    let (low, high) = integer_bounds(integer);
                    value.int().clamp(low, high)
                }
                other => other.int(),
            };
            Value::Int(wrap(integer, value))
        }
    }
}

fn integer_bounds(element: ElementType) -> (i128, i128) {
    match element {
        ElementType::U8 => (0, i128::from(u8::MAX)),
        ElementType::I8 => (i128::from(i8::MIN), i128::from(i8::MAX)),
        ElementType::U16 => (0, i128::from(u16::MAX)),
        ElementType::I16 => (i128::from(i16::MIN), i128::from(i16::MAX)),
        ElementType::U32 => (0, i128::from(u32::MAX)),
        ElementType::I32 => (i128::from(i32::MIN), i128::from(i32::MAX)),
        ElementType::U64 => (0, i128::from(u64::MAX)),
        _ => (i128::from(i64::MIN), i128::from(i64::MAX)),
    }
}

/// Wraps `value` to `element`'s width, as integer arithmetic on a GPU does.
fn wrap(element: ElementType, value: i128) -> i128 {
    let bits = element.size_bytes() * 8;
    let modulus = 1_i128 << bits;
    let unsigned = value.rem_euclid(modulus);
    let signed = matches!(
        element,
        ElementType::I8 | ElementType::I16 | ElementType::I32 | ElementType::I64
    );
    if signed && unsigned >= modulus / 2 {
        unsigned - modulus
    } else {
        unsigned
    }
}

fn unary(op: UnaryOp, value: Value) -> Value {
    if let Value::Int(integer) = value {
        return match op {
            UnaryOp::Neg => Value::Int(-integer),
            UnaryOp::Abs => Value::Int(integer.abs()),
            UnaryOp::BitNot => Value::Int(!integer),
            _ => unreachable!(
                "verification allows only negation, absolute value, and complements on integers"
            ),
        };
    }

    if op == UnaryOp::Not {
        return Value::Bool(!value.truth());
    }

    let x = value.float();
    Value::Float(match op {
        UnaryOp::Neg => -x,
        UnaryOp::Abs => x.abs(),
        UnaryOp::Exp => x.exp(),
        UnaryOp::Log => x.ln(),
        UnaryOp::Sqrt => x.sqrt(),
        UnaryOp::Rsqrt => 1.0 / x.sqrt(),
        UnaryOp::Reciprocal => 1.0 / x,
        UnaryOp::Sin => x.sin(),
        UnaryOp::Cos => x.cos(),
        UnaryOp::Tan => x.tan(),
        UnaryOp::Tanh => x.tanh(),
        UnaryOp::Asin => x.asin(),
        UnaryOp::Acos => x.acos(),
        UnaryOp::Atan => x.atan(),
        UnaryOp::Sinh => x.sinh(),
        UnaryOp::Cosh => x.cosh(),
        UnaryOp::Asinh => x.asinh(),
        UnaryOp::Acosh => x.acosh(),
        UnaryOp::Atanh => x.atanh(),
        UnaryOp::Erf => libm::erf(x),
        UnaryOp::Log1p => x.ln_1p(),
        UnaryOp::Expm1 => x.exp_m1(),
        UnaryOp::Floor => x.floor(),
        UnaryOp::Ceil => x.ceil(),
        UnaryOp::Trunc => x.trunc(),
        UnaryOp::Round => x.round_ties_even(),
        UnaryOp::IsNan => return Value::Bool(x.is_nan()),
        UnaryOp::Not => unreachable!("handled above"),
        UnaryOp::BitNot => unreachable!("verification allows complements of integers only"),
    })
}

fn binary(
    op: BinaryOp,
    element: ElementType,
    lhs: Value,
    rhs: Value,
) -> Result<Value, InterpretError> {
    if op.is_comparison() {
        let ordering = match (lhs, rhs) {
            (Value::Int(lhs), Value::Int(rhs)) => Some(lhs.cmp(&rhs)),
            (Value::Bool(lhs), Value::Bool(rhs)) => Some(lhs.cmp(&rhs)),
            (lhs, rhs) => lhs.float().partial_cmp(&rhs.float()),
        };
        return Ok(Value::Bool(match op {
            BinaryOp::Equal => ordering.is_some_and(Ordering::is_eq),
            // NaN compares unequal to everything, so `!=` holds for it.
            BinaryOp::NotEqual => ordering.is_none_or(Ordering::is_ne),
            BinaryOp::Less => ordering.is_some_and(Ordering::is_lt),
            BinaryOp::LessEqual => ordering.is_some_and(Ordering::is_le),
            BinaryOp::Greater => ordering.is_some_and(Ordering::is_gt),
            _ => ordering.is_some_and(Ordering::is_ge),
        }));
    }

    if element == ElementType::Bool {
        return Ok(Value::Bool(match op {
            BinaryOp::And => lhs.truth() && rhs.truth(),
            _ => lhs.truth() || rhs.truth(),
        }));
    }

    if element.is_integer() {
        let (lhs, rhs) = (lhs.int(), rhs.int());
        return Ok(Value::Int(match op {
            // Operands hold at most 64 bits, so wrapping in 128 bits keeps the low bits every integer type wraps to.
            BinaryOp::Add => lhs.wrapping_add(rhs),
            BinaryOp::Sub => lhs.wrapping_sub(rhs),
            BinaryOp::Mul => lhs.wrapping_mul(rhs),
            BinaryOp::Div | BinaryOp::Rem if rhs == 0 => {
                return Err(InterpretError::DivisionByZero);
            }
            BinaryOp::Div => lhs / rhs,
            BinaryOp::Rem => lhs % rhs,
            BinaryOp::Max => lhs.max(rhs),
            BinaryOp::Min => lhs.min(rhs),
            BinaryOp::BitAnd => lhs & rhs,
            BinaryOp::BitOr => lhs | rhs,
            BinaryOp::BitXor => lhs ^ rhs,
            BinaryOp::ShiftLeft => shift_left(element, lhs, rhs),
            BinaryOp::ShiftRight => shift_right(element, lhs, rhs),
            _ => integer_power(lhs, rhs),
        }));
    }

    let (x, y) = (lhs.float(), rhs.float());
    Ok(Value::Float(match op {
        BinaryOp::Add => x + y,
        BinaryOp::Sub => x - y,
        BinaryOp::Mul => x * y,
        BinaryOp::Div => x / y,
        BinaryOp::Rem => x % y,
        BinaryOp::Pow => x.powf(y),
        BinaryOp::Max => maximum(x, y),
        BinaryOp::Min => minimum(x, y),
        _ => x.atan2(y),
    }))
}

/// Returns `value` shifted left by `shift` bits of an `element`, or zero for a shift outside `0..bits`.
fn shift_left(element: ElementType, value: i128, shift: i128) -> i128 {
    let bits = i128::try_from(element.size_bytes() * 8).unwrap_or(i128::MAX);
    match u32::try_from(shift) {
        Ok(shift) if i128::from(shift) < bits => value << shift,
        _ => 0,
    }
}

/// Returns `value` shifted right by `shift` bits of an `element`, filling with its sign, or the sign's fill alone for a shift outside `0..bits`.
fn shift_right(element: ElementType, value: i128, shift: i128) -> i128 {
    let bits = i128::try_from(element.size_bytes() * 8).unwrap_or(i128::MAX);
    match u32::try_from(shift) {
        Ok(shift) if i128::from(shift) < bits => value >> shift,
        _ => i128::from(value < 0).wrapping_neg(),
    }
}

/// Returns `base` to the power `exponent`, wrapping, and for a negative exponent the integer part of the reciprocal power, as `PyTorch` computes it: zero but for a base of one or minus one.
fn integer_power(base: i128, exponent: i128) -> i128 {
    if exponent < 0 {
        return match base {
            1 => 1,
            -1 if exponent % 2 == 0 => 1,
            -1 => -1,
            _ => 0,
        };
    }

    let (mut result, mut base, mut exponent) = (1_i128, base, exponent);
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = result.wrapping_mul(base);
        }

        base = base.wrapping_mul(base);
        exponent >>= 1;
    }
    result
}

/// Returns the larger value, NaN if either is NaN.
///
/// Of two equal values, such as zeros of either sign, it is `x`, as the emitted kernels choose.
fn maximum(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        f64::NAN
    } else if y > x {
        y
    } else {
        x
    }
}

// Whether a later `value` takes an argument reduction's place from `best`: strictly larger, or smaller without `maximum`, with NaN above every number and an earlier NaN kept.
fn ranks_above(value: Value, best: Value, maximum: bool) -> bool {
    if let (Value::Int(value), Value::Int(best)) = (value, best) {
        return if maximum { value > best } else { value < best };
    }

    let (value, best) = (value.float(), best.float());
    if value.is_nan() || best.is_nan() {
        return value.is_nan() && !best.is_nan();
    }
    if maximum { value > best } else { value < best }
}

/// Returns `log(exp(x) + exp(y))`, from the larger value so no exponential overflows.
fn log_sum_exp(x: f64, y: f64) -> f64 {
    let largest = maximum(x, y);
    if largest.is_infinite() {
        return largest;
    }
    largest + (minimum(x, y) - largest).exp().ln_1p()
}

/// Returns the smaller value, NaN if either is NaN.
///
/// Of two equal values, such as zeros of either sign, it is `x`, as the emitted kernels choose.
fn minimum(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        f64::NAN
    } else if y < x {
        y
    } else {
        x
    }
}

fn identity(combiner: Combiner, element: ElementType) -> Value {
    if element.is_integer() {
        let (low, high) = integer_bounds(element);
        return Value::Int(match combiner {
            Combiner::Sum => 0,
            Combiner::Product => 1,
            Combiner::Max | Combiner::LogSumExp => low,
            Combiner::Min => high,
        });
    }
    Value::Float(match combiner {
        Combiner::Sum => 0.0,
        Combiner::Product => 1.0,
        Combiner::Max | Combiner::LogSumExp => f64::NEG_INFINITY,
        Combiner::Min => f64::INFINITY,
    })
}

fn combine(combiner: Combiner, accumulator: Value, value: Value) -> Value {
    match (accumulator, value) {
        // A Boolean sum is a disjunction.
        (Value::Bool(accumulator), value) => Value::Bool(accumulator || value.truth()),
        (Value::Int(accumulator), Value::Int(value)) => Value::Int(match combiner {
            Combiner::Sum => accumulator + value,
            Combiner::Product => accumulator * value,
            Combiner::Max | Combiner::LogSumExp => accumulator.max(value),
            Combiner::Min => accumulator.min(value),
        }),
        (accumulator, value) => {
            let (x, y) = (accumulator.float(), value.float());
            Value::Float(match combiner {
                Combiner::Sum => x + y,
                Combiner::Product => x * y,
                Combiner::Max => maximum(x, y),
                Combiner::Min => minimum(x, y),
                Combiner::LogSumExp => log_sum_exp(x, y),
            })
        }
    }
}

/// Returns the byte range of element `offset` of a buffer of `element`s.
fn element_range(
    buffer_length: usize,
    operand: usize,
    offset: u64,
    element: ElementType,
) -> Result<Range<usize>, InterpretError> {
    let size = element.size_bytes();
    let start = usize::try_from(offset)
        .ok()
        .and_then(|offset| offset.checked_mul(size))
        .ok_or(InterpretError::Overflow)?;
    let end = start.checked_add(size).ok_or(InterpretError::Overflow)?;
    if end > buffer_length {
        return Err(InterpretError::BufferTooSmall {
            operand,
            required: end,
            actual: buffer_length,
        });
    }
    Ok(start..end)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the raw bits of an element narrower than 64 bits occupy the low bits"
)]
fn read(
    buffer: &[u8],
    operand: usize,
    offset: u64,
    element: ElementType,
) -> Result<Value, InterpretError> {
    let bytes = &buffer[element_range(buffer.len(), operand, offset, element)?];
    let array = |bytes: &[u8]| -> [u8; 8] {
        let mut array = [0; 8];
        array[..bytes.len()].copy_from_slice(bytes);
        array
    };
    let raw = u64::from_le_bytes(array(bytes));
    Ok(match element {
        ElementType::Bool => Value::Bool(raw != 0),
        ElementType::F16 => Value::Float(f64::from(f16::from_bits(raw as u16))),
        ElementType::F8E4M3 => Value::Float(E4M3.decode(raw as u8)),
        ElementType::F8E5M2 => Value::Float(E5M2.decode(raw as u8)),
        ElementType::Bf16 => Value::Float(f64::from(bf16::from_bits(raw as u16))),
        ElementType::F32 => Value::Float(f64::from(f32::from_bits(raw as u32))),
        ElementType::F64 => Value::Float(f64::from_bits(raw)),
        integer => {
            let bits = integer.size_bytes() * 8;
            Value::Int(wrap(integer, i128::from(raw) & ((1_i128 << bits) - 1)))
        }
    })
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "elements keep the low bits of their wrapped value, and `F32` stores round"
)]
fn write(
    buffer: &mut [u8],
    operand: usize,
    offset: u64,
    element: ElementType,
    value: Value,
) -> Result<(), InterpretError> {
    let range = element_range(buffer.len(), operand, offset, element)?;
    let raw: u64 = match round(element, value) {
        Value::Bool(value) => u64::from(value),
        Value::Float(value) => match element {
            ElementType::F16 => u64::from(f16::from_f64(value).to_bits()),
            ElementType::F8E4M3 => u64::from(E4M3.encode(value)),
            ElementType::F8E5M2 => u64::from(E5M2.encode(value)),
            ElementType::Bf16 => u64::from(bf16::from_f64(value).to_bits()),
            ElementType::F32 => u64::from((value as f32).to_bits()),
            _ => value.to_bits(),
        },
        // Two's complement bits of the wrapped value, truncated to the element's width below.
        Value::Int(value) => value as u64,
    };
    let size = element.size_bytes();
    buffer[range].copy_from_slice(&raw.to_le_bytes()[..size]);
    Ok(())
}

#[cfg(test)]
mod tests;
