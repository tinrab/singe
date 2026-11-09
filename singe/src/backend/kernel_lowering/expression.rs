//! Element-wise expressions, built into any kernel that a lowering is writing.

use std::f64::consts::FRAC_1_SQRT_2;

use singe_codegen::{
    abi::Bindings,
    ir::{BinaryOp, ElementType, Expr, Kernel, Literal, NodeId, UnaryOp},
};

use crate::{
    backend::{
        kernel_lowering::{
            ArgumentLabels, STATUS_DIVISION_BY_ZERO,
            domain::converted,
            element_type,
            group::{FusedOperand, FusedStep, is_fusible},
        },
        registry::{Operation, OperationAttributes, OperationRequest},
    },
    data_type::DataType,
    error::{Error, Result},
    scalar::Scalar,
    shape::Shape,
    value::ValueSpec,
};

/// A value of an element-wise expression: its node, computed in its type's compute type, and the type the graph gives it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ElementValue {
    pub node: NodeId,
    pub data_type: DataType,
}

/// Builds element-wise expressions over `shape` into `kernel`, adding the scalars they take to `bindings`.
///
/// Values stay in their compute type between operations, as one kernel holds them in registers, and round only when stored.
pub struct Expressions<'a> {
    pub kernel: &'a mut Kernel,
    pub bindings: &'a mut Bindings,
    pub labels: &'a mut ArgumentLabels,
    pub shape: &'a Shape,
}

impl Expressions<'_> {
    /// Applies element-wise `operation` with `attributes` to `inputs` and returns its value.
    ///
    /// Scalars the attributes carry, such as a scale or clamp bounds, become kernel scalars, so one compiled kernel serves every value.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation is not element-wise, or its inputs do not have the types it accepts.
    pub fn apply(
        &mut self,
        operation: Operation,
        attributes: &OperationAttributes,
        inputs: &[ElementValue],
    ) -> Result<ElementValue> {
        let specs = inputs
            .iter()
            .map(|input| ValueSpec::new(input.data_type, self.shape.clone()))
            .collect::<Vec<_>>();
        let inference = OperationRequest::new(operation, attributes.clone(), specs).infer()?;
        let output = inference.outputs()[0].data_type();
        let computation = inference.computation_data_type();
        let compute = element_type(computation)?.compute();
        // Operands convert to the computation's type, except a `where` condition and logical operands, which stay `bool`.
        let x = inputs
            .iter()
            .enumerate()
            .map(|(index, input)| {
                let keep = (operation == Operation::Where && index == 0)
                    || matches!(
                        operation,
                        Operation::LogicalAnd | Operation::LogicalOr | Operation::LogicalNot
                    );
                let target = if keep { ElementType::Bool } else { compute };
                self.convert(input.node, target)
            })
            .collect::<Vec<_>>();
        let node = self.expression(operation, attributes, &x, compute)?;
        let node = if operation == Operation::Cast {
            self.convert(node, element_type(output)?.compute())
        } else {
            node
        };
        Ok(ElementValue {
            node,
            data_type: output,
        })
    }

    /// Returns `node` converted to `element`, or `node` itself when it already has that type.
    pub fn convert(&mut self, node: NodeId, element: ElementType) -> NodeId {
        converted(self.kernel, node, element)
    }

    fn scalar(&mut self, element: ElementType, value: f64, label: &str) -> NodeId {
        self.bound(element, Literal::Float(value), label)
    }

    /// Returns a kernel scalar of `element` bound to `value`.
    fn bound(&mut self, element: ElementType, value: Literal, label: &str) -> NodeId {
        let scalar = self.kernel.add_scalar(element);
        self.bindings.scalars.push(value);
        self.labels.scalars.push(label.to_owned());
        self.kernel.add_node(element, Expr::Scalar(scalar))
    }

    /// Returns `value` as a literal of `element`, an integer or Boolean one for those types.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "integer constants of element-wise expressions are small whole numbers"
    )]
    fn constant(&mut self, element: ElementType, value: f64) -> NodeId {
        let literal = if element == ElementType::Bool {
            Literal::Bool(value != 0.0)
        } else if element.is_integer() {
            Literal::Int(value as i64)
        } else {
            Literal::Float(value)
        };
        self.kernel.literal(element, literal)
    }

    /// Returns the value of a scalar attribute in `compute`: its `F64` bits for `F64` kernels, its single-precision value for other floating kernels, as the CPU computes, and its integer or truth value otherwise.
    #[expect(
        clippy::cast_possible_wrap,
        reason = "unsigned scalars wrap to the kernel's two's complement integers, as PyTorch converts them"
    )]
    fn coefficient(&mut self, compute: ElementType, scalar: Scalar, label: &str) -> NodeId {
        let integer = match scalar {
            Scalar::Bool(value) => i64::from(value),
            Scalar::Signed(value) => value,
            Scalar::Unsigned(value) => value as i64,
            Scalar::Float(_) => 0,
        };
        match (compute, scalar) {
            (ElementType::Bool, scalar) => self.bound(
                ElementType::Bool,
                Literal::Bool(scalar.float() != 0.0),
                label,
            ),
            // Kernels take integer scalars as `I64`, which the conversion to a narrower type wraps.
            (_, Scalar::Bool(_) | Scalar::Signed(_) | Scalar::Unsigned(_))
                if compute.is_integer() =>
            {
                let wide = self.bound(ElementType::I64, Literal::Int(integer), label);
                self.convert(wide, compute)
            }
            (ElementType::F64, scalar) => {
                self.bound(compute, Literal::Float(scalar.float()), label)
            }
            (_, scalar) => self.bound(compute, Literal::Float(f64::from(scalar.single())), label),
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the element-wise operations define their expressions together"
    )]
    fn expression(
        &mut self,
        operation: Operation,
        attributes: &OperationAttributes,
        x: &[NodeId],
        compute: ElementType,
    ) -> Result<NodeId> {
        let input = |index: usize| {
            x.get(index)
                .copied()
                .ok_or_else(|| Error::missing_input("expression"))
        };
        let unary =
            |builder: &mut Self, op| -> Result<NodeId> { Ok(builder.kernel.unary(op, input(0)?)) };
        let binary = |builder: &mut Self, op| -> Result<NodeId> {
            Ok(builder.kernel.binary(op, input(0)?, input(1)?))
        };
        let integer = compute.is_integer();
        if compute == ElementType::Bool {
            return self.boolean(operation, attributes, x);
        }

        if integer && !has_integer_meaning(operation) {
            return Err(Error::unsupported_data_type(
                "integer element-wise operation",
            ));
        }
        Ok(match operation {
            Operation::Add => binary(self, BinaryOp::Add)?,
            Operation::Sub => binary(self, BinaryOp::Sub)?,
            Operation::Mul => binary(self, BinaryOp::Mul)?,
            Operation::Div => binary(self, BinaryOp::Div)?,
            Operation::Modulo | Operation::Remainder => {
                self.remainder(input(0)?, input(1)?, compute)
            }
            Operation::Atan2 => binary(self, BinaryOp::Atan2)?,
            Operation::Pow => binary(self, BinaryOp::Pow)?,
            Operation::BitwiseAnd => binary(self, BinaryOp::BitAnd)?,
            Operation::BitwiseOr => binary(self, BinaryOp::BitOr)?,
            Operation::BitwiseXor => binary(self, BinaryOp::BitXor)?,
            Operation::BitwiseLeftShift => binary(self, BinaryOp::ShiftLeft)?,
            Operation::BitwiseRightShift => binary(self, BinaryOp::ShiftRight)?,
            Operation::BitwiseNot => unary(self, UnaryOp::BitNot)?,
            Operation::Maximum => binary(self, BinaryOp::Max)?,
            Operation::Minimum => binary(self, BinaryOp::Min)?,
            Operation::Equal => binary(self, BinaryOp::Equal)?,
            Operation::NotEqual => binary(self, BinaryOp::NotEqual)?,
            Operation::Less => binary(self, BinaryOp::Less)?,
            Operation::LessEqual => binary(self, BinaryOp::LessEqual)?,
            Operation::Greater => binary(self, BinaryOp::Greater)?,
            Operation::GreaterEqual => binary(self, BinaryOp::GreaterEqual)?,
            Operation::Where => self.kernel.select(input(0)?, input(1)?, input(2)?),
            // x × (gate / (1 + exp(-gate))).
            Operation::SwiGlu => {
                let (value, gate) = (input(0)?, input(1)?);
                let sigmoid = self.sigmoid_ratio(gate, gate, compute);
                self.kernel.binary(BinaryOp::Mul, value, sigmoid)
            }
            Operation::Lerp => {
                let t = attributes
                    .lerp_t()
                    .ok_or_else(|| Error::invalid_argument("cuda lerp attributes"))?;
                let t = self.scalar(compute, f64::from(t), "lerp weight");
                let (start, end) = (input(0)?, input(1)?);
                let difference = self.kernel.binary(BinaryOp::Sub, end, start);
                self.kernel.mul_add(t, difference, start)
            }
            Operation::Clamp => {
                let clamp = attributes
                    .clamp()
                    .ok_or_else(|| Error::invalid_argument("cuda clamp attributes"))?;
                let low = self.scalar(compute, f64::from(clamp.min), "clamp minimum");
                let high = self.scalar(compute, f64::from(clamp.max), "clamp maximum");
                let value = input(0)?;
                // NaN in the value, then in either bound, propagates; otherwise the value is clamped.
                let above = self.kernel.binary(BinaryOp::Greater, value, high);
                let upper = self.kernel.select(above, high, value);
                let below = self.kernel.binary(BinaryOp::Less, value, low);
                let clamped = self.kernel.select(below, low, upper);
                let high_nan = self.kernel.unary(UnaryOp::IsNan, high);
                let clamped = self.kernel.select(high_nan, high, clamped);
                let low_nan = self.kernel.unary(UnaryOp::IsNan, low);
                let clamped = self.kernel.select(low_nan, low, clamped);
                let value_nan = self.kernel.unary(UnaryOp::IsNan, value);
                self.kernel.select(value_nan, value, clamped)
            }
            operation if operation.is_scalar_elementwise() => {
                let scalar = attributes
                    .scalar()
                    .ok_or_else(|| Error::invalid_argument("cuda scalar attributes"))?;
                let coefficient = self.coefficient(compute, scalar, operation.name());
                let value = input(0)?;
                match operation {
                    Operation::AddScalar => self.kernel.binary(BinaryOp::Add, value, coefficient),
                    Operation::SubScalar => self.kernel.binary(BinaryOp::Sub, value, coefficient),
                    Operation::RSubScalar => self.kernel.binary(BinaryOp::Sub, coefficient, value),
                    Operation::Scale => self.kernel.binary(BinaryOp::Mul, value, coefficient),
                    Operation::RDivScalar => self.kernel.binary(BinaryOp::Div, coefficient, value),
                    Operation::PowScalar => self.kernel.binary(BinaryOp::Pow, value, coefficient),
                    _ => {
                        return Err(Error::unsupported_operation(
                            "cuda generated scalar operation",
                        ));
                    }
                }
            }
            Operation::Cast => input(0)?,
            Operation::Sqrt => unary(self, UnaryOp::Sqrt)?,
            Operation::Exp => unary(self, UnaryOp::Exp)?,
            Operation::Log => unary(self, UnaryOp::Log)?,
            Operation::Sin => unary(self, UnaryOp::Sin)?,
            Operation::Cos => unary(self, UnaryOp::Cos)?,
            Operation::Tan => unary(self, UnaryOp::Tan)?,
            Operation::Sinh => unary(self, UnaryOp::Sinh)?,
            Operation::Cosh => unary(self, UnaryOp::Cosh)?,
            Operation::Tanh => unary(self, UnaryOp::Tanh)?,
            Operation::Asin => unary(self, UnaryOp::Asin)?,
            Operation::Acos => unary(self, UnaryOp::Acos)?,
            Operation::Atan => unary(self, UnaryOp::Atan)?,
            // Integers are their own ceiling, floor, and truncation.
            Operation::Ceil | Operation::Floor | Operation::Truncate | Operation::Round
                if integer =>
            {
                input(0)?
            }
            Operation::Round => unary(self, UnaryOp::Round)?,
            // Only floating values can be NaN.
            Operation::IsNan if compute.is_float() => unary(self, UnaryOp::IsNan)?,
            Operation::IsNan => self.constant(ElementType::Bool, 0.0),
            Operation::Ceil => unary(self, UnaryOp::Ceil)?,
            Operation::Floor => unary(self, UnaryOp::Floor)?,
            Operation::Truncate => unary(self, UnaryOp::Trunc)?,
            Operation::Neg | Operation::Abs | Operation::Square | Operation::Relu => {
                self.signed(operation, input(0)?, compute)
            }

            Operation::Asinh => unary(self, UnaryOp::Asinh)?,
            Operation::Acosh => unary(self, UnaryOp::Acosh)?,
            Operation::Atanh => unary(self, UnaryOp::Atanh)?,
            Operation::Erf => unary(self, UnaryOp::Erf)?,
            Operation::Log1p => unary(self, UnaryOp::Log1p)?,
            Operation::Expm1 => unary(self, UnaryOp::Expm1)?,
            Operation::Reciprocal => unary(self, UnaryOp::Reciprocal)?,
            // 1 / sqrt(x), with exact division and square root.
            Operation::Rsqrt => {
                let root = unary(self, UnaryOp::Sqrt)?;
                self.kernel.unary(UnaryOp::Reciprocal, root)
            }
            // (x × 0.5) × (1 + erf(x × √½)).
            Operation::GeluExact => {
                let value = input(0)?;
                let scale = self.constant(compute, FRAC_1_SQRT_2);
                let scaled = self.kernel.binary(BinaryOp::Mul, value, scale);
                let erf = self.kernel.unary(UnaryOp::Erf, scaled);
                self.gelu_product(value, erf, compute)
            }
            // (x × 0.5) × (1 + tanh(√(2/π) × (x + 0.044715 × x × x × x))).
            Operation::Gelu => {
                let value = input(0)?;
                let cubic = self.constant(compute, 0.044_715);
                let term = self.kernel.binary(BinaryOp::Mul, cubic, value);
                let term = self.kernel.binary(BinaryOp::Mul, term, value);
                let term = self.kernel.binary(BinaryOp::Mul, term, value);
                let inner = self.kernel.binary(BinaryOp::Add, value, term);
                let scale = self.constant(compute, 0.797_884_560_802_865_4);
                let scaled = self.kernel.binary(BinaryOp::Mul, scale, inner);
                let tanh = self.kernel.unary(UnaryOp::Tanh, scaled);
                self.gelu_product(value, tanh, compute)
            }
            // x / (1 + exp(-x)).
            Operation::Silu => {
                let value = input(0)?;
                self.sigmoid_ratio(value, value, compute)
            }
            // 1 / (1 + exp(-x)).
            Operation::Sigmoid => {
                let value = input(0)?;
                let one = self.constant(compute, 1.0);
                self.sigmoid_ratio(one, value, compute)
            }
            _ => {
                return Err(Error::unsupported_operation(
                    "cuda generated element-wise operation",
                ));
            }
        })
    }

    /// Returns the negation, absolute value, square, or rectification of `value`, which every numeric type defines.
    fn signed(&mut self, operation: Operation, value: NodeId, compute: ElementType) -> NodeId {
        match operation {
            Operation::Neg => self.kernel.unary(UnaryOp::Neg, value),
            Operation::Abs => self.kernel.unary(UnaryOp::Abs, value),
            Operation::Square => self.kernel.binary(BinaryOp::Mul, value, value),
            _ => {
                let zero = self.constant(compute, 0.0);
                let negative = self.kernel.binary(BinaryOp::Less, value, zero);
                self.kernel.select(negative, zero, value)
            }
        }
    }

    /// Returns the remainder of `a / b` with the sign of the divisor, as `PyTorch` computes it: the truncated remainder, plus `b` where the two signs differ.
    fn remainder(&mut self, a: NodeId, b: NodeId, compute: ElementType) -> NodeId {
        let zero = self.constant(compute, 0.0);
        let divisor = if compute.is_integer() {
            self.integer_divisor(b, compute)
        } else {
            b
        };
        let truncated = self.kernel.binary(BinaryOp::Rem, a, divisor);
        let nonzero = self.kernel.binary(BinaryOp::NotEqual, truncated, zero);
        let remainder_negative = self.kernel.binary(BinaryOp::Less, truncated, zero);
        let divisor_negative = self.kernel.binary(BinaryOp::Less, b, zero);
        let signs_differ =
            self.kernel
                .binary(BinaryOp::NotEqual, remainder_negative, divisor_negative);
        let adjust = self.kernel.binary(BinaryOp::And, nonzero, signs_differ);
        let adjusted = self.kernel.binary(BinaryOp::Add, truncated, b);
        self.kernel.select(adjust, adjusted, truncated)
    }

    /// Returns a divisor for integer remainders of `b` that C++ defines: one in place of zero, which the kernel reports, as `PyTorch` raises an error, and in place of minus one, whose remainders are all zero but which overflows for the smallest signed value.
    ///
    /// A compiler may assume a divisor is not zero once it divides by it, so checking the divisor it divides by would not survive optimization.
    fn integer_divisor(&mut self, b: NodeId, compute: ElementType) -> NodeId {
        let zero = self.constant(compute, 0.0);
        let one = self.constant(compute, 1.0);
        let nonzero = self.kernel.binary(BinaryOp::NotEqual, b, zero);
        let checked = self.kernel.require(b, nonzero, STATUS_DIVISION_BY_ZERO);
        let trivial = if matches!(
            compute,
            ElementType::I8 | ElementType::I16 | ElementType::I32 | ElementType::I64
        ) {
            let minus_one = self.constant(compute, -1.0);
            let negative_one = self.kernel.binary(BinaryOp::Equal, b, minus_one);
            let zero_divisor = self.kernel.unary(UnaryOp::Not, nonzero);
            self.kernel.binary(BinaryOp::Or, zero_divisor, negative_one)
        } else {
            self.kernel.unary(UnaryOp::Not, nonzero)
        };
        self.kernel.select(trivial, one, checked)
    }

    /// Returns `operation` on Boolean operands: sums and maxima are disjunctions, products and minima conjunctions.
    fn boolean(
        &mut self,
        operation: Operation,
        attributes: &OperationAttributes,
        x: &[NodeId],
    ) -> Result<NodeId> {
        let input = |index: usize| {
            x.get(index)
                .copied()
                .ok_or_else(|| Error::missing_input("boolean"))
        };
        if operation == Operation::IsNan {
            return Ok(self.constant(ElementType::Bool, 0.0));
        }

        let (op, rhs) = match operation {
            Operation::Add | Operation::Maximum | Operation::LogicalOr | Operation::BitwiseOr => {
                (BinaryOp::Or, input(1)?)
            }
            Operation::Mul | Operation::Minimum | Operation::LogicalAnd | Operation::BitwiseAnd => {
                (BinaryOp::And, input(1)?)
            }
            // `x ** true` is `x`, and `x ** false` is true.
            Operation::PowScalar => {
                let scalar = attributes
                    .scalar()
                    .ok_or_else(|| Error::invalid_argument("boolean scalar attributes"))?;
                let exponent = self.coefficient(ElementType::Bool, scalar, operation.name());
                let trivial = self.kernel.unary(UnaryOp::Not, exponent);
                (BinaryOp::Or, trivial)
            }
            Operation::AddScalar | Operation::Scale => {
                let scalar = attributes
                    .scalar()
                    .ok_or_else(|| Error::invalid_argument("boolean scalar attributes"))?;
                let coefficient = self.coefficient(ElementType::Bool, scalar, operation.name());
                let op = if operation == Operation::AddScalar {
                    BinaryOp::Or
                } else {
                    BinaryOp::And
                };
                (op, coefficient)
            }
            Operation::Equal => (BinaryOp::Equal, input(1)?),
            // The exclusive disjunction of truth values is their inequality.
            Operation::NotEqual | Operation::BitwiseXor => (BinaryOp::NotEqual, input(1)?),
            Operation::Less => (BinaryOp::Less, input(1)?),
            Operation::LessEqual => (BinaryOp::LessEqual, input(1)?),
            Operation::Greater => (BinaryOp::Greater, input(1)?),
            Operation::GreaterEqual => (BinaryOp::GreaterEqual, input(1)?),
            Operation::LogicalNot | Operation::BitwiseNot => {
                return Ok(self.kernel.unary(UnaryOp::Not, input(0)?));
            }
            Operation::Where => return Ok(self.kernel.select(input(0)?, input(1)?, input(2)?)),
            Operation::Cast | Operation::Copy | Operation::Contiguous => return input(0),
            _ => {
                return Err(Error::unsupported_data_type(
                    "boolean element-wise operation",
                ));
            }
        };
        Ok(self.kernel.binary(op, input(0)?, rhs))
    }

    /// Returns `numerator / (1 + exp(-x))`.
    fn sigmoid_ratio(&mut self, numerator: NodeId, x: NodeId, compute: ElementType) -> NodeId {
        let negated = self.kernel.unary(UnaryOp::Neg, x);
        let exponential = self.kernel.unary(UnaryOp::Exp, negated);
        let one = self.constant(compute, 1.0);
        let denominator = self.kernel.binary(BinaryOp::Add, one, exponential);
        self.kernel.binary(BinaryOp::Div, numerator, denominator)
    }

    /// Returns `(x × 0.5) × (1 + value)`.
    fn gelu_product(&mut self, x: NodeId, value: NodeId, compute: ElementType) -> NodeId {
        let half = self.constant(compute, 0.5);
        let one = self.constant(compute, 1.0);
        let scaled = self.kernel.binary(BinaryOp::Mul, x, half);
        let shifted = self.kernel.binary(BinaryOp::Add, one, value);
        self.kernel.binary(BinaryOp::Mul, scaled, shifted)
    }

    /// Computes `steps` from `loaded`, the loaded inputs they name, and returns every step's value.
    ///
    /// # Errors
    ///
    /// Returns an error if a step is not fusible, names an input or a later step that does not exist, or [`Self::apply`] fails for it.
    pub fn steps(
        &mut self,
        loaded: &[ElementValue],
        steps: &[FusedStep<'_>],
    ) -> Result<Vec<ElementValue>> {
        let mut values = Vec::with_capacity(steps.len());
        for (index, step) in steps.iter().enumerate() {
            if !is_fusible(step.operation) {
                return Err(Error::unsupported_operation("cuda fused kernel step"));
            }

            let operands = step
                .operands
                .iter()
                .map(|operand| match *operand {
                    FusedOperand::Input(slot) => loaded.get(slot).copied(),
                    FusedOperand::Step(earlier) if earlier < index => values.get(earlier).copied(),
                    FusedOperand::Step(_) => None,
                })
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| Error::invalid_argument("cuda fused kernel operand"))?;
            values.push(self.apply(step.operation, step.attributes, &operands)?);
        }
        Ok(values)
    }
}

/// Returns whether `operation` computes on integers as they are, rather than through a floating type that inference promotes them to.
const fn has_integer_meaning(operation: Operation) -> bool {
    matches!(
        operation,
        Operation::Add
            | Operation::Sub
            | Operation::Mul
            | Operation::Modulo
            | Operation::Remainder
            | Operation::Maximum
            | Operation::Minimum
            | Operation::Equal
            | Operation::NotEqual
            | Operation::Less
            | Operation::LessEqual
            | Operation::Greater
            | Operation::GreaterEqual
            | Operation::Where
            | Operation::Cast
            | Operation::Ceil
            | Operation::Floor
            | Operation::Truncate
            | Operation::Round
            | Operation::IsNan
            | Operation::Neg
            | Operation::Abs
            | Operation::Square
            | Operation::Relu
            | Operation::AddScalar
            | Operation::SubScalar
            | Operation::RSubScalar
            | Operation::Scale
            | Operation::PowScalar
            | Operation::Pow
            | Operation::BitwiseAnd
            | Operation::BitwiseOr
            | Operation::BitwiseXor
            | Operation::BitwiseNot
            | Operation::BitwiseLeftShift
            | Operation::BitwiseRightShift
    )
}
