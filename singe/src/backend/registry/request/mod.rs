mod operation;

pub use self::operation::{
    CheckedOperationRequest, OperationInference, OperationRequest, ResultAlias,
};

use crate::{
    backend::registry::{Operation, OperationSchema, OutputRole},
    data_type::DataType,
    error::{Error, Result},
    value::ValueSpec,
};

/// A catalog operation applied to two logical values without accessing storage.
#[derive(Debug, Clone)]
pub struct BinaryOperationRequest {
    operation: Operation,
    operands: [ValueSpec; 2],
    rule: BinaryTypeRule,
    output_role: OutputRole,
}

#[derive(Debug, Clone, Copy)]
enum BinaryTypeRule {
    Promote,
    TrueDivision,
    Boolean,
}

/// Logical results and the operand conversion required by a binary operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryOperationInference {
    outputs: [ValueSpec; 1],
    computation_data_type: DataType,
}

impl OperationSchema {
    /// Constructs a pure request for core arithmetic, comparisons, or Bool logic.
    ///
    /// Returns `None` for operations outside this binary inference family.
    #[must_use]
    pub fn binary_request(&self, lhs: ValueSpec, rhs: ValueSpec) -> Option<BinaryOperationRequest> {
        let rule = match self.operation() {
            Operation::Add
            | Operation::Sub
            | Operation::Mul
            | Operation::Maximum
            | Operation::Minimum
            | Operation::Modulo
            | Operation::Remainder
            | Operation::BitwiseAnd
            | Operation::BitwiseOr
            | Operation::BitwiseXor
            | Operation::BitwiseLeftShift
            | Operation::BitwiseRightShift
            | Operation::Pow
            | Operation::Equal
            | Operation::NotEqual
            | Operation::Less
            | Operation::LessEqual
            | Operation::Greater
            | Operation::GreaterEqual => BinaryTypeRule::Promote,
            Operation::Div | Operation::Atan2 => BinaryTypeRule::TrueDivision,
            Operation::LogicalAnd | Operation::LogicalOr => BinaryTypeRule::Boolean,
            _ => return None,
        };
        Some(BinaryOperationRequest {
            operation: self.operation(),
            operands: [lhs, rhs],
            rule,
            output_role: self.single_output_role()?,
        })
    }
}

impl BinaryOperationRequest {
    /// Infers ordered results and operand dtype without reading or allocating tensors.
    ///
    /// # Errors
    ///
    /// Returns an error for incompatible dimensions or unsupported dtype semantics.
    pub fn infer(&self) -> Result<BinaryOperationInference> {
        let [lhs, rhs] = &self.operands;
        let shape = lhs.shape().broadcast_with(rhs.shape())?;
        let computation_data_type = match self.rule {
            // Logical operations read any operand as its truth value.
            BinaryTypeRule::Boolean => DataType::Bool,
            BinaryTypeRule::Promote | BinaryTypeRule::TrueDivision => {
                let promoted = lhs.data_type().promote(
                    lhs.shape().rank(),
                    rhs.data_type(),
                    rhs.shape().rank(),
                )?;
                if matches!(self.rule, BinaryTypeRule::TrueDivision) && !promoted.is_float() {
                    DataType::F32
                } else {
                    let bitwise = matches!(
                        self.operation,
                        Operation::BitwiseAnd
                            | Operation::BitwiseOr
                            | Operation::BitwiseXor
                            | Operation::BitwiseLeftShift
                            | Operation::BitwiseRightShift
                    );
                    if bitwise && promoted.is_float() {
                        return Err(Error::unsupported_data_type(
                            "bitwise operation of floating values",
                        ));
                    }

                    // Boolean sums, products, maxima, minima, and bitwise operations are disjunctions and conjunctions; subtraction, remainders, shifts, and powers have no Boolean meaning.
                    if promoted == DataType::Bool
                        && self.output_role != OutputRole::Mask
                        && !matches!(
                            self.operation,
                            Operation::Add
                                | Operation::Mul
                                | Operation::Maximum
                                | Operation::Minimum
                                | Operation::BitwiseAnd
                                | Operation::BitwiseOr
                                | Operation::BitwiseXor
                        )
                    {
                        return Err(Error::invalid_argument(
                            "boolean subtraction, remainder, shift, or power",
                        ));
                    }
                    promoted
                }
            }
        };
        Ok(BinaryOperationInference {
            outputs: [ValueSpec::new(
                self.output_role.data_type(computation_data_type),
                shape,
            )],
            computation_data_type,
        })
    }

    /// Returns the canonical catalog opcode.
    #[must_use]
    pub fn operation(&self) -> Operation {
        self.operation
    }
}

impl BinaryOperationInference {
    /// Returns result specifications in schema order.
    #[must_use]
    pub fn outputs(&self) -> &[ValueSpec; 1] {
        &self.outputs
    }

    /// Returns the dtype required for both operands before execution.
    #[must_use]
    pub fn computation_data_type(&self) -> DataType {
        self.computation_data_type
    }
}
