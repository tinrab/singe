use std::slice;

use thiserror::Error;

use crate::ir::{
    Access, AxisId, AxisKind, BinaryOp, Combiner, ElementType, Expr, Kernel, Literal, NodeId,
    OperandRole, UnaryOp,
};

/// A way a kernel breaks the IR's rules.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum VerifyError {
    #[error("{what} {index} does not exist")]
    Missing { what: &'static str, index: usize },
    #[error("node {node} refers to node {operand}, which is not defined before it")]
    ForwardReference { node: usize, operand: usize },
    #[error("node {node} has type `{actual}`, expected `{expected}`")]
    Type {
        node: usize,
        expected: &'static str,
        actual: &'static str,
    },
    #[error("node {node} applies `{op}` to a `{element}` value")]
    Operation {
        node: usize,
        op: &'static str,
        element: &'static str,
    },
    #[error("node {node} is a `{element}` literal of another kind")]
    Literal { node: usize, element: &'static str },
    #[error("node {node} reduces no axes")]
    EmptyReduction { node: usize },
    #[error("node {node} reduces parallel axis {axis}")]
    ReducedParallelAxis { node: usize, axis: usize },
    #[error("node {node} loads output operand {operand}")]
    LoadFromOutput { node: usize, operand: usize },
    #[error("store {store} writes input operand {operand}")]
    StoreToInput { store: usize, operand: usize },
    #[error("store {store} writes `{actual}` into an operand of `{expected}`")]
    StoreType {
        store: usize,
        expected: &'static str,
        actual: &'static str,
    },
    #[error(
        "store {store} writes a value that depends on axis {axis}, which its access does not use"
    )]
    StoreDomain { store: usize, axis: usize },
    #[error(
        "store {store} combines with `{combiner}` into an operand of `{element}`, which no atomic operation supports"
    )]
    StoreCombiner {
        store: usize,
        combiner: &'static str,
        element: &'static str,
    },
    #[error("output operand {operand} is never stored")]
    UnwrittenOutput { operand: usize },
}

impl Kernel {
    /// Checks that the kernel follows the IR's rules: every reference exists, types agree, only reduction axes are reduced, and stores write outputs with values their accesses cover.
    ///
    /// # Errors
    ///
    /// Returns the first rule the kernel breaks.
    pub fn verify(&self) -> Result<(), VerifyError> {
        for axis in &self.axes {
            self.check_size(axis.extent.0)?;
        }

        for (index, node) in self.nodes.iter().enumerate() {
            self.verify_node(index, node.element, &node.expr)?;
        }

        for (index, store) in self.stores.iter().enumerate() {
            self.verify_access(&store.access, self.nodes.len())?;
            let operand = self.operands[store.access.operand.0];
            if operand.role != OperandRole::Output {
                return Err(VerifyError::StoreToInput {
                    store: index,
                    operand: store.access.operand.0,
                });
            }

            let value = self.node_element(store.value, self.nodes.len())?;
            if value != operand.element.compute() {
                return Err(VerifyError::StoreType {
                    store: index,
                    expected: operand.element.compute().name(),
                    actual: value.name(),
                });
            }

            if let Some(combiner) = store.combiner
                && combiner != Combiner::Sum
            {
                return Err(VerifyError::StoreCombiner {
                    store: index,
                    combiner: combiner.name(),
                    element: operand.element.name(),
                });
            }

            let covered = self.access_domain(&store.access);
            if let Some(axis) = self
                .domain(store.value)
                .into_iter()
                .find(|axis| !covered.contains(axis))
            {
                return Err(VerifyError::StoreDomain {
                    store: index,
                    axis: axis.0,
                });
            }
        }

        for (index, operand) in self.operands.iter().enumerate() {
            if operand.role == OperandRole::Output
                && !self
                    .stores
                    .iter()
                    .any(|store| store.access.operand.0 == index)
            {
                return Err(VerifyError::UnwrittenOutput { operand: index });
            }
        }
        Ok(())
    }

    fn verify_node(
        &self,
        index: usize,
        element: ElementType,
        expr: &Expr,
    ) -> Result<(), VerifyError> {
        let expect = |expected: ElementType| {
            if element == expected {
                Ok(())
            } else {
                Err(VerifyError::Type {
                    node: index,
                    expected: expected.name(),
                    actual: element.name(),
                })
            }
        };
        let operation = |op: &'static str, operand: ElementType| VerifyError::Operation {
            node: index,
            op,
            element: operand.name(),
        };
        match expr {
            Expr::Load(access) => {
                self.verify_access(access, index)?;
                let operand = self.operands[access.operand.0];
                if operand.role != OperandRole::Input {
                    return Err(VerifyError::LoadFromOutput {
                        node: index,
                        operand: access.operand.0,
                    });
                }
                expect(operand.element.compute())
            }
            Expr::Literal(literal) => {
                let matches = match literal {
                    Literal::Bool(_) => element == ElementType::Bool,
                    Literal::Int(_) => element.is_integer(),
                    Literal::Float(_) => element.is_float(),
                };
                if matches {
                    Ok(())
                } else {
                    Err(VerifyError::Literal {
                        node: index,
                        element: element.name(),
                    })
                }
            }
            Expr::Scalar(scalar) => {
                let declared = *self.scalars.get(scalar.0).ok_or(VerifyError::Missing {
                    what: "scalar",
                    index: scalar.0,
                })?;
                expect(declared)
            }
            Expr::AxisIndex(axis) => {
                self.check_axis(axis.0)?;
                if element.is_integer() {
                    Ok(())
                } else {
                    Err(operation("axis index", element))
                }
            }
            Expr::Unary(op, value) => {
                let operand = self.node_element(*value, index)?;
                let valid = match op {
                    UnaryOp::Not => operand == ElementType::Bool,
                    UnaryOp::BitNot => operand.is_integer(),
                    UnaryOp::Neg | UnaryOp::Abs => operand.is_float() || operand.is_integer(),
                    _ => operand.is_float(),
                };
                if !valid {
                    return Err(operation(op.name(), operand));
                }
                expect(if op.is_predicate() {
                    ElementType::Bool
                } else {
                    operand
                })
            }
            Expr::Binary(op, lhs, rhs) => {
                let left = self.node_element(*lhs, index)?;
                let right = self.node_element(*rhs, index)?;
                if left != right {
                    return Err(VerifyError::Type {
                        node: index,
                        expected: left.name(),
                        actual: right.name(),
                    });
                }

                let valid = match op {
                    BinaryOp::And | BinaryOp::Or => left == ElementType::Bool,
                    BinaryOp::Atan2 => left.is_float(),
                    BinaryOp::BitAnd
                    | BinaryOp::BitOr
                    | BinaryOp::BitXor
                    | BinaryOp::ShiftLeft
                    | BinaryOp::ShiftRight => left.is_integer(),
                    op if op.is_comparison() => true,
                    _ => left.is_float() || left.is_integer(),
                };
                if !valid {
                    return Err(operation(op.name(), left));
                }
                expect(if op.is_comparison() {
                    ElementType::Bool
                } else {
                    left
                })
            }
            Expr::MulAdd(a, b, c) => {
                let a = self.node_element(*a, index)?;
                if !a.is_float() {
                    return Err(operation("mul_add", a));
                }

                expect(a)?;
                expect(self.node_element(*b, index)?)?;
                expect(self.node_element(*c, index)?)
            }
            Expr::Select {
                condition,
                if_true,
                if_false,
            } => {
                let condition = self.node_element(*condition, index)?;
                if condition != ElementType::Bool {
                    return Err(operation("select", condition));
                }

                expect(self.node_element(*if_true, index)?)?;
                expect(self.node_element(*if_false, index)?)
            }
            Expr::Cast(value) => self.node_element(*value, index).map(|_| ()),
            Expr::Reduce {
                combiner,
                axes,
                value,
            } => {
                self.check_reduced(axes, index)?;
                let operand = self.node_element(*value, index)?;
                if operand == ElementType::Bool
                    || (*combiner == Combiner::LogSumExp && !operand.is_float())
                {
                    return Err(operation(combiner.name(), operand));
                }
                expect(operand)
            }
            Expr::Scan {
                combiner,
                axis,
                value,
            } => {
                self.check_reduced(slice::from_ref(axis), index)?;
                let operand = self.node_element(*value, index)?;
                if operand == ElementType::Bool || *combiner == Combiner::LogSumExp {
                    return Err(operation(combiner.name(), operand));
                }
                expect(operand)
            }
            Expr::ArgReduce { axes, value, .. } => {
                self.check_reduced(axes, index)?;
                let operand = self.node_element(*value, index)?;
                if !operand.is_float() && !operand.is_integer() {
                    return Err(operation("arg_reduce", operand));
                }
                expect(ElementType::I64)
            }
            Expr::SoftmaxSum { axes, logit, value } => {
                self.check_reduced(axes, index)?;
                for operand in [*logit, *value] {
                    let operand = self.node_element(operand, index)?;
                    if !operand.is_float() {
                        return Err(operation("softmax_sum", operand));
                    }
                }
                expect(self.node_element(*value, index)?)
            }
            Expr::Require {
                value, condition, ..
            } => {
                let condition = self.node_element(*condition, index)?;
                if condition != ElementType::Bool {
                    return Err(operation("require", condition));
                }
                expect(self.node_element(*value, index)?)
            }
            Expr::CheckIndex { value, extent, .. } => {
                self.check_size(extent.0)?;
                let operand = self.node_element(*value, index)?;
                if !operand.is_integer() {
                    return Err(operation("check_index", operand));
                }
                expect(ElementType::U64)
            }
        }
    }

    /// Checks `access`, which `user`, a node or the stores after every node, refers to.
    fn verify_access(&self, access: &Access, user: usize) -> Result<(), VerifyError> {
        if access.operand.0 >= self.operands.len() {
            return Err(VerifyError::Missing {
                what: "operand",
                index: access.operand.0,
            });
        }

        if let Some(base) = access.base {
            self.check_size(base.0)?;
        }

        for &(axis, stride) in &access.strides {
            self.check_axis(axis.0)?;
            self.check_size(stride.0)?;
        }

        for &(node, stride) in &access.indirect {
            self.check_size(stride.0)?;
            let element = self.node_element(node, user)?;
            if !element.is_integer() {
                return Err(VerifyError::Operation {
                    node: node.0,
                    op: "index",
                    element: element.name(),
                });
            }
        }
        Ok(())
    }

    /// Returns the type of `node`, which `user` refers to and must follow.
    fn node_element(&self, node: NodeId, user: usize) -> Result<ElementType, VerifyError> {
        if node.0 >= user {
            return Err(VerifyError::ForwardReference {
                node: user,
                operand: node.0,
            });
        }
        Ok(self.nodes[node.0].element)
    }

    fn check_size(&self, index: usize) -> Result<(), VerifyError> {
        if index < self.sizes.len() {
            Ok(())
        } else {
            Err(VerifyError::Missing {
                what: "size",
                index,
            })
        }
    }

    /// Checks that reduction node `node` reduces at least one axis and only reduction axes.
    fn check_reduced(&self, axes: &[AxisId], node: usize) -> Result<(), VerifyError> {
        if axes.is_empty() {
            return Err(VerifyError::EmptyReduction { node });
        }

        for axis in axes {
            self.check_axis(axis.0)?;
            if self.axes[axis.0].kind != AxisKind::Reduction {
                return Err(VerifyError::ReducedParallelAxis { node, axis: axis.0 });
            }
        }
        Ok(())
    }

    fn check_axis(&self, index: usize) -> Result<(), VerifyError> {
        if index < self.axes.len() {
            Ok(())
        } else {
            Err(VerifyError::Missing {
                what: "axis",
                index,
            })
        }
    }
}
