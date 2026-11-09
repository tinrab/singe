//! CUDA C++ source for NVRTC.
//!
//! The kernel is one `extern "C"` function named after the IR kernel, declaring the parameters [`abi::parameters`](crate::abi::parameters) lists: `oN` for operands, `cN` for scalars, and `sN` for argument sizes.
//! Each node becomes a `const` local `vN` in node order, which is the graph's topological order; a node that depends on reduction axes is emitted inside every loop that walks them.
//! Literals are written as exact bit patterns, so no constant is rounded twice.
//! A kernel whose every store writes F16 or BF16 computes F32 `exp` and division with `__expf` and `__fdividef`, whose errors are far below their rounding.
//!
//! A row schedule whose reduction axes have static extents, and whose threads each cover at most 16 of their points, unrolls its loops and keeps the loads and values that several loops read in registers, so each row is read from memory, and each value computed, once.

mod indexing;
mod mma;
mod prelude;
mod reduction;
mod row_cache;
pub mod spelling;
mod vector;
mod writer;

pub use self::{prelude::PRELUDE, writer::SourceWriter};
use self::{
    reduction::{LoopAction, reduced_axes},
    row_cache::RowCache,
    spelling::{
        binary_expression, compute_type, literal_expression, load_expression, storage_type,
        unary_expression,
    },
};

use crate::{
    emit::{EmitError, IndexWidth, KernelSource},
    ir::{AxisId, AxisKind, BinaryOp, ElementType, Expr, Kernel, NodeId, OperandRole, UnaryOp},
    schedule::{Schedule, ThreadSchedule},
};

/// Returns CUDA C++ source for `kernel` under `schedule`.
///
/// # Errors
///
/// Returns an error if the kernel or schedule is invalid.
pub fn emit(
    kernel: &Kernel,
    schedule: ThreadSchedule,
    index: IndexWidth,
) -> Result<KernelSource, EmitError> {
    kernel.verify()?;
    Schedule::from(schedule).validate()?;
    if Schedule::from(schedule).tiles() != [1, 1] {
        return Err(EmitError::Unsupported {
            reason: "a parallel tile, which runs as the kernel `parallel_tile::tiled` returns",
        });
    }

    let mut writer = Writer {
        kernel,
        schedule,
        domains: (0..kernel.nodes.len())
            .map(|node| kernel.domain(NodeId(node)))
            .collect(),
        cache: None,
        approximate: kernel.stores.iter().all(|store| {
            matches!(
                kernel.operands[store.access.operand.0].element,
                ElementType::F16 | ElementType::Bf16
            )
        }),
        width: index,
        index: match index {
            IndexWidth::U32 => "unsigned int",
            IndexWidth::U64 => "unsigned long long",
        },
        suffix: match index {
            IndexWidth::U32 => "U",
            IndexWidth::U64 => "ULL",
        },
        source: SourceWriter::new(),
    };
    writer.cache = writer.row_cache()?;
    writer.signature();
    writer.body()?;
    Ok(writer.source.finish(kernel.name.clone()))
}

struct Writer<'a> {
    kernel: &'a Kernel,
    schedule: ThreadSchedule,
    /// The domain of each node.
    domains: Vec<Vec<AxisId>>,
    /// How a row schedule keeps values in registers, when it can.
    cache: Option<RowCache>,
    width: IndexWidth,
    /// Whether every store writes F16 or BF16, whose rounding hides the error of approximate F32 functions.
    /// `__fdividef` returns zero for divisors above 2^126, where an exact quotient of any F32 dividend rounds to a BF16 subnormal or zero anyway.
    approximate: bool,
    /// The type indices and offsets are computed in, and the suffix of its literals.
    index: &'static str,
    suffix: &'static str,
    source: SourceWriter,
}

impl Writer<'_> {
    fn signature(&mut self) {
        for (index, operand) in self.kernel.operands.iter().enumerate() {
            let storage = storage_type(operand.element);
            let ty = match operand.role {
                OperandRole::Input => format!("const {storage}* __restrict__"),
                OperandRole::Output => format!("{storage}* __restrict__"),
            };
            self.source.parameter(ty, format!("o{index}"));
        }

        if self.kernel.reports_status() {
            self.source.parameter("unsigned int*", "status");
        }

        for (index, &scalar) in self.kernel.scalars.iter().enumerate() {
            self.source.parameter(
                format!("const {}", compute_type(scalar)),
                format!("c{index}"),
            );
        }

        for size in self.kernel.argument_sizes() {
            // Sizes arrive as 64-bit values; a 32-bit kernel narrows them first thing.
            let name = if self.width == IndexWidth::U64 {
                "s"
            } else {
                "wide_s"
            };
            self.source
                .parameter("const unsigned long long", format!("{name}{}", size.0));
        }
        if self.width == IndexWidth::U32 {
            for size in self.kernel.argument_sizes() {
                let n = size.0;
                self.line(&format!(
                    "const unsigned int s{n} = static_cast<unsigned int>(wide_s{n});"
                ));
            }
        }
    }

    fn body(&mut self) -> Result<(), EmitError> {
        if let ThreadSchedule::Mma { m, n, k } = self.schedule {
            return self.mma_body(m, n, k);
        }

        let ix = self.index;
        let parallel = self.axes_of_kind(AxisKind::Parallel);
        if let (ThreadSchedule::Elements { vector, .. }, Some(&inner)) =
            (self.schedule, parallel.last())
            && vector > 1
        {
            return self.vector_body(&parallel, inner, vector);
        }

        let total = self.product(&parallel);
        match self.schedule {
            ThreadSchedule::Mma { .. } => unreachable!("mma schedules emit their own body"),
            ThreadSchedule::Elements { .. } => {
                self.line(&format!(
                    "const {ix} linear = static_cast<{ix}>(blockIdx.x) * blockDim.x + threadIdx.x;"
                ));
                self.line(&format!("if (linear >= {total}) return;"));
            }
            ThreadSchedule::Rows { .. } => {
                self.line(&format!("const {ix} linear = blockIdx.x;"));
            }
            ThreadSchedule::Warps { lanes, .. } => {
                self.line(&format!(
                    "const unsigned int warp_lane = threadIdx.x % {lanes}U;"
                ));
                self.line(&format!("const {ix} row = static_cast<{ix}>(blockIdx.x) * (blockDim.x / {lanes}U) + threadIdx.x / {lanes}U;"));
                // Lanes past the last row repeat it, so every lane of a warp takes part in its shuffles; they return before storing.
                self.line(&format!(
                    "const {ix} linear = row < {total} ? row : {total} - 1;"
                ));
            }
            ThreadSchedule::Columns { threads, splits } => {
                let points = threads / splits;
                self.line(&format!(
                    "const unsigned int split = threadIdx.x / {points}U;"
                ));
                self.line(&format!("const {ix} row = static_cast<{ix}>(blockIdx.x) * {points}U + threadIdx.x % {points}U;"));
                // Threads past the last point repeat it, so every thread takes part in the block's synchronizations; they return before storing.
                self.line(&format!(
                    "const {ix} linear = row < {total} ? row : {total} - 1;"
                ));
            }
        }

        self.decompose("linear", &parallel);
        if let Some(cache) = &self.cache {
            let declarations = cache
                .values
                .iter()
                .map(|node| {
                    format!(
                        "{} cached{}[{}];",
                        compute_type(self.kernel.nodes[node.0].element),
                        node.0,
                        cache.count * cache.vector
                    )
                })
                .collect::<Vec<_>>();
            for declaration in declarations {
                self.line(&declaration);
            }
        }

        let mut reduced = vec![false; self.kernel.nodes.len()];
        for node in 0..self.kernel.nodes.len() {
            let node = NodeId(node);
            if !self.is_outer(node) || reduced[node.0] {
                continue;
            }
            if let Some(axes) = reduced_axes(&self.kernel.nodes[node.0].expr) {
                // Later reductions over the same axes that need nothing this one's loop cannot see share its loop.
                let mut group = vec![node];
                for later in node.0 + 1..self.kernel.nodes.len() {
                    let later = NodeId(later);
                    if self.is_outer(later)
                        && reduced_axes(&self.kernel.nodes[later.0].expr) == Some(axes)
                        && self.joins(later, node, &reduced)
                    {
                        group.push(later);
                    }
                }

                for &member in &group {
                    reduced[member.0] = true;
                }

                self.reduce_group(&group)?;
            } else {
                let expression = self.expression(node);
                self.define(node, &expression);
            }
        }

        if let ThreadSchedule::Warps { .. } | ThreadSchedule::Columns { .. } = self.schedule {
            self.line(&format!("if (row >= {total}) return;"));
        }

        for store in &self.kernel.stores {
            let loop_axes = self
                .kernel
                .access_domain(&store.access)
                .into_iter()
                .filter(|axis| self.kernel.axes[axis.0].kind == AxisKind::Reduction)
                .collect::<Vec<_>>();
            let write = self.write(store);
            if loop_axes.is_empty() {
                match self.schedule {
                    ThreadSchedule::Mma { .. } => unreachable!("mma schedules emit their own body"),
                    // Every thread of a row computes the row's values, and one writes them.
                    ThreadSchedule::Rows { .. } => {
                        self.line(&format!("if (threadIdx.x == 0) {write}"));
                    }
                    ThreadSchedule::Warps { .. } => {
                        self.line(&format!("if (warp_lane == 0) {write}"));
                    }
                    ThreadSchedule::Columns { .. } => {
                        self.line(&format!("if (split == 0) {write}"));
                    }
                    ThreadSchedule::Elements { .. } => self.line(&write),
                }
            } else {
                self.reduction_loop(&loop_axes, &LoopAction::Store(store))?;
            }
        }
        Ok(())
    }

    fn define(&mut self, node: NodeId, expression: &str) {
        let ty = compute_type(self.kernel.nodes[node.0].element);
        self.line(&format!("const {ty} v{} = {expression};", node.0));
    }

    fn expression(&self, node: NodeId) -> String {
        let element = self.kernel.nodes[node.0].element;
        let ty = compute_type(element);
        let suffix = if element == ElementType::F32 { "f" } else { "" };
        let v = |node: &NodeId| format!("v{}", node.0);
        match &self.kernel.nodes[node.0].expr {
            Expr::Load(access) => load_expression(
                self.kernel.operands[access.operand.0].element,
                &format!("o{}[{}]", access.operand.0, self.offset(access)),
            ),
            Expr::Literal(literal) => literal_expression(element, *literal),
            Expr::Scalar(scalar) => format!("c{}", scalar.0),
            Expr::AxisIndex(axis) => format!("static_cast<{ty}>(a{})", axis.0),
            Expr::Unary(op, value) => {
                let operand_element = self.kernel.nodes[value.0].element;
                if self.approximate && *op == UnaryOp::Exp && operand_element == ElementType::F32 {
                    format!("__expf({})", v(value))
                } else {
                    unary_expression(*op, operand_element, &v(value))
                }
            }
            Expr::Binary(op, lhs, rhs) => {
                let operand_element = self.kernel.nodes[lhs.0].element;
                if self.approximate && *op == BinaryOp::Div && operand_element == ElementType::F32 {
                    format!("__fdividef({}, {})", v(lhs), v(rhs))
                } else {
                    binary_expression(*op, operand_element, &v(lhs), &v(rhs))
                }
            }
            Expr::MulAdd(a, b, c) => format!("fma{suffix}({}, {}, {})", v(a), v(b), v(c)),
            Expr::Select {
                condition,
                if_true,
                if_false,
            } => format!("({} ? {} : {})", v(condition), v(if_true), v(if_false)),
            Expr::Cast(value) if element == ElementType::Bool => format!("({} != 0)", v(value)),
            Expr::Cast(value) => format!("static_cast<{ty}>({})", v(value)),
            Expr::Reduce { .. }
            | Expr::SoftmaxSum { .. }
            | Expr::ArgReduce { .. }
            | Expr::Scan { .. } => {
                unreachable!("reductions are emitted by `reduce_group` or `nested_reduction`")
            }
            Expr::Require {
                value,
                condition,
                status,
            } => format!(
                "(singe_require(status, {}, {status}U), {})",
                v(condition),
                v(value)
            ),
            Expr::CheckIndex {
                value,
                extent,
                wrap,
                status,
            } => format!(
                "singe_check_index(status, static_cast<long long>({}), {}, {wrap}, {status}U)",
                v(value),
                self.size(*extent)
            ),
        }
    }

    fn line(&mut self, line: &str) {
        self.source.line(line);
    }
}

// Whether `code` uses identifier `name` as a whole word, not as the prefix of a longer one such as `a0_base`.
pub(crate) fn mentions(code: &str, name: &str) -> bool {
    let word = |byte: u8| byte.is_ascii_alphanumeric() || byte == b'_';
    code.match_indices(name).any(|(start, _)| {
        let before = code.as_bytes().get(start.wrapping_sub(1)).copied();
        let after = code.as_bytes().get(start + name.len()).copied();
        !before.is_some_and(word) && !after.is_some_and(word)
    })
}

#[cfg(test)]
mod tests;
