//! The canonical text form of a kernel, one declaration, node, or store per line.
//!
//! Two kernels print the same text exactly when they are equal, so the text keys compiled-kernel caches.

use std::fmt::{self, Display, Formatter};

use crate::ir::{Access, AxisKind, Expr, Kernel, Literal, OperandRole, Size};

impl Display for Kernel {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "kernel {}", self.name)?;
        for (index, size) in self.sizes.iter().enumerate() {
            match size {
                Size::Static(value) => writeln!(f, "size s{index} = {value}")?,
                Size::Argument => writeln!(f, "size s{index} = argument")?,
            }
        }

        for (index, operand) in self.operands.iter().enumerate() {
            let role = match operand.role {
                OperandRole::Input => "input",
                OperandRole::Output => "output",
            };
            writeln!(f, "operand o{index} = {role} {}", operand.element.name())?;
        }

        for (index, scalar) in self.scalars.iter().enumerate() {
            writeln!(f, "scalar c{index} = {}", scalar.name())?;
        }

        for (index, axis) in self.axes.iter().enumerate() {
            let kind = match axis.kind {
                AxisKind::Parallel => "parallel",
                AxisKind::Reduction => "reduction",
            };
            writeln!(f, "axis a{index} = {kind} s{}", axis.extent.0)?;
        }

        for (index, node) in self.nodes.iter().enumerate() {
            write!(f, "%{index} = {} ", node.element.name())?;
            match &node.expr {
                Expr::Load(access) => write!(f, "load {}", AccessText(access))?,
                Expr::Literal(Literal::Bool(value)) => write!(f, "literal {value}")?,
                Expr::Literal(Literal::Int(value)) => write!(f, "literal {value}")?,
                // Debug prints the shortest text that reads back as the same value.
                Expr::Literal(Literal::Float(value)) => write!(f, "literal {value:?}")?,
                Expr::Scalar(scalar) => write!(f, "scalar c{}", scalar.0)?,
                Expr::AxisIndex(axis) => write!(f, "index a{}", axis.0)?,
                Expr::Unary(op, value) => write!(f, "{} %{}", op.name(), value.0)?,
                Expr::Binary(op, lhs, rhs) => write!(f, "{} %{} %{}", op.name(), lhs.0, rhs.0)?,
                Expr::MulAdd(a, b, c) => write!(f, "mul_add %{} %{} %{}", a.0, b.0, c.0)?,
                Expr::Select {
                    condition,
                    if_true,
                    if_false,
                } => write!(f, "select %{} %{} %{}", condition.0, if_true.0, if_false.0)?,
                Expr::Cast(value) => write!(f, "cast %{}", value.0)?,
                Expr::Reduce {
                    combiner,
                    axes,
                    value,
                } => {
                    write!(f, "reduce {} [", combiner.name())?;
                    for (position, axis) in axes.iter().enumerate() {
                        let separator = if position == 0 { "" } else { " " };
                        write!(f, "{separator}a{}", axis.0)?;
                    }

                    write!(f, "] %{}", value.0)?;
                }
                Expr::Scan {
                    combiner,
                    axis,
                    value,
                } => write!(f, "scan {} [a{}] %{}", combiner.name(), axis.0, value.0)?,
                Expr::ArgReduce {
                    maximum,
                    axes,
                    value,
                } => {
                    let name = if *maximum { "argmax" } else { "argmin" };
                    write!(f, "{name} [")?;
                    for (position, axis) in axes.iter().enumerate() {
                        let separator = if position == 0 { "" } else { " " };
                        write!(f, "{separator}a{}", axis.0)?;
                    }

                    write!(f, "] %{}", value.0)?;
                }
                Expr::SoftmaxSum { axes, logit, value } => {
                    write!(f, "softmax_sum [")?;
                    for (position, axis) in axes.iter().enumerate() {
                        let separator = if position == 0 { "" } else { " " };
                        write!(f, "{separator}a{}", axis.0)?;
                    }

                    write!(f, "] %{} %{}", logit.0, value.0)?;
                }
                Expr::CheckIndex {
                    value,
                    extent,
                    wrap,
                    status,
                } => {
                    let wrap = if *wrap { " wrap" } else { "" };
                    write!(
                        f,
                        "check_index %{} s{}{wrap} status {status}",
                        value.0, extent.0
                    )?;
                }
                Expr::Require {
                    value,
                    condition,
                    status,
                } => write!(f, "require %{} %{} status {status}", value.0, condition.0)?,
            }

            writeln!(f)?;
        }

        for store in &self.stores {
            let operator = store.combiner.map_or_else(
                || "=".to_owned(),
                |combiner| format!("{}=", combiner.name()),
            );
            writeln!(
                f,
                "store {} {operator} %{}",
                AccessText(&store.access),
                store.value.0
            )?;
        }
        Ok(())
    }
}

/// Prints an access as `o0[s1 + a0*s2 + a1*s3 + %4*s5]`.
struct AccessText<'a>(&'a Access);

impl Display for AccessText<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let access = self.0;
        write!(f, "o{}[", access.operand.0)?;
        let mut first = access.base.is_none();
        if let Some(base) = access.base {
            write!(f, "s{}", base.0)?;
        }

        for &(axis, stride) in &access.strides {
            let separator = if first { "" } else { " + " };
            write!(f, "{separator}a{}*s{}", axis.0, stride.0)?;
            first = false;
        }

        for &(node, stride) in &access.indirect {
            let separator = if first { "" } else { " + " };
            write!(f, "{separator}%{}*s{}", node.0, stride.0)?;
            first = false;
        }

        if first {
            write!(f, "0")?;
        }
        write!(f, "]")
    }
}
