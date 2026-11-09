//! The structure of a plain contraction, `out[batch, i, j] = Σ_k lhs[batch, i, k] · rhs[batch, k, j]`, which [`Schedule::TileMatmul`](crate::schedule::Schedule::TileMatmul) runs on tensor cores.

use crate::ir::{Access, AxisId, AxisKind, BinaryOp, Combiner, ElementType, Expr, Kernel, NodeId};

/// The loads and axes of a plain contraction kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Contraction {
    /// The load of the left operand, which walks the rows and the contraction.
    pub lhs: NodeId,
    /// The load of the right operand, which walks the contraction and the columns.
    pub rhs: NodeId,
    /// The reduction axis the product sums over.
    pub reduced: AxisId,
    /// The parallel axis the left operand walks and the right one does not.
    pub rows: AxisId,
    /// The parallel axis the right operand walks and the left one does not.
    pub columns: AxisId,
    /// The other parallel axes, in axis order.
    pub batch: Vec<AxisId>,
}

impl Contraction {
    /// Returns the contraction `kernel` computes, or `None` for any other kernel.
    ///
    /// A contraction stores, once and without combining, a sum over the kernel's one reduction axis of the product of two direct loads of one `F16`, `BF16`, or `F32` operand type, whose rows and columns each have their own parallel axis that the store walks.
    #[must_use]
    pub fn of(kernel: &Kernel) -> Option<Self> {
        let [store] = kernel.stores.as_slice() else {
            return None;
        };
        if store.combiner.is_some() {
            return None;
        }

        let Expr::Reduce {
            combiner: Combiner::Sum,
            axes,
            value,
        } = &kernel.nodes[store.value.0].expr
        else {
            return None;
        };
        let &[reduced] = axes.as_slice() else {
            return None;
        };
        let reductions = kernel
            .axes
            .iter()
            .filter(|axis| axis.kind == AxisKind::Reduction)
            .count();
        let Expr::Binary(BinaryOp::Mul, x, y) = kernel.nodes[value.0].expr else {
            return None;
        };
        let load = |node: NodeId| match &kernel.nodes[node.0].expr {
            Expr::Load(access) if access.indirect.is_empty() => Some(access),
            _ => None,
        };
        let (x_access, y_access) = (load(x)?, load(y)?);
        let element = kernel.operands[x_access.operand.0].element;
        if reductions != 1
            || element != kernel.operands[y_access.operand.0].element
            || !matches!(
                element,
                ElementType::F16 | ElementType::Bf16 | ElementType::F32
            )
        {
            return None;
        }

        let walks = |access: &Access, axis: AxisId| access.axes().any(|walked| walked == axis);
        if !walks(x_access, reduced) || !walks(y_access, reduced) {
            return None;
        }

        let parallel = (0..kernel.axes.len())
            .map(AxisId)
            .filter(|axis| kernel.axes[axis.0].kind == AxisKind::Parallel)
            .collect::<Vec<_>>();
        let own = |access: &Access, other: &Access| {
            parallel
                .iter()
                .copied()
                .filter(|&axis| walks(access, axis) && !walks(other, axis))
                .collect::<Vec<_>>()
        };
        let (&[x_own], &[y_own]) = (
            own(x_access, y_access).as_slice(),
            own(y_access, x_access).as_slice(),
        ) else {
            return None;
        };
        // The rows come before the columns, as the output's dimensions order them.
        let (lhs, rhs, rows, columns) = if x_own < y_own {
            (x, y, x_own, y_own)
        } else {
            (y, x, y_own, x_own)
        };
        if parallel.iter().any(|&axis| !walks(&store.access, axis)) || store.access.base.is_some() {
            return None;
        }
        Some(Self {
            lhs,
            rhs,
            reduced,
            rows,
            columns,
            batch: parallel
                .into_iter()
                .filter(|&axis| axis != rows && axis != columns)
                .collect(),
        })
    }
}

// A contraction whose factors and result may be computed: `out[batch, i, j] = epilogue(Σ_k lhs(batch, i, k) · rhs(batch, k, j))`, which [`Schedule::Mma`](crate::schedule::Schedule::Mma) runs on tensor cores.
// Each factor is any expression of its own axes whose value is exactly a `F16` or `BF16` value, both of one type, so a tensor-core product of the rounded factors is the product the kernel computes; the epilogue is any expression of the parallel axes and the sum.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusedContraction {
    pub sum: NodeId,
    pub lhs: NodeId,
    pub rhs: NodeId,
    pub element: ElementType,
    pub reduced: AxisId,
    pub rows: AxisId,
    pub columns: AxisId,
    pub batch: Vec<AxisId>,
}

impl FusedContraction {
    #[must_use]
    pub fn of(kernel: &Kernel) -> Option<Self> {
        let [store] = kernel.stores.as_slice() else {
            return None;
        };
        let reductions = (0..kernel.axes.len())
            .map(AxisId)
            .filter(|axis| kernel.axes[axis.0].kind == AxisKind::Reduction)
            .collect::<Vec<_>>();
        let &[reduced] = reductions.as_slice() else {
            return None;
        };
        if store.combiner.is_some() || store.access.base.is_some() {
            return None;
        }

        // The kernel's one reduction is a sum of a product over the reduction axis.
        let mut sums = kernel.nodes.iter().enumerate().filter(|(_, node)| {
            matches!(
                node.expr,
                Expr::Reduce { .. }
                    | Expr::SoftmaxSum { .. }
                    | Expr::ArgReduce { .. }
                    | Expr::Scan { .. }
            )
        });
        let (sum, node) = sums.next()?;
        if sums.next().is_some() {
            return None;
        }

        let Expr::Reduce {
            combiner: Combiner::Sum,
            axes,
            value,
        } = &node.expr
        else {
            return None;
        };
        if axes.as_slice() != [reduced] {
            return None;
        }

        let Expr::Binary(BinaryOp::Mul, x, y) = kernel.nodes[value.0].expr else {
            return None;
        };
        let element = half_value(kernel, x)?;
        if half_value(kernel, y)? != element {
            return None;
        }

        let parallel = (0..kernel.axes.len())
            .map(AxisId)
            .filter(|axis| kernel.axes[axis.0].kind == AxisKind::Parallel)
            .collect::<Vec<_>>();
        let (x_domain, y_domain) = (kernel.domain(x), kernel.domain(y));
        if !x_domain.contains(&reduced) || !y_domain.contains(&reduced) {
            return None;
        }

        let own = |domain: &[AxisId], other: &[AxisId]| {
            parallel
                .iter()
                .copied()
                .filter(|axis| domain.contains(axis) && !other.contains(axis))
                .collect::<Vec<_>>()
        };
        let (&[x_own], &[y_own]) = (
            own(&x_domain, &y_domain).as_slice(),
            own(&y_domain, &x_domain).as_slice(),
        ) else {
            return None;
        };
        let (lhs, rhs, rows, columns) = if x_own < y_own {
            (x, y, x_own, y_own)
        } else {
            (y, x, y_own, x_own)
        };
        let walks = |axis: AxisId| store.access.axes().any(|walked| walked == axis);
        if parallel.iter().any(|&axis| !walks(axis)) {
            return None;
        }
        Some(Self {
            sum: NodeId(sum),
            lhs,
            rhs,
            element,
            reduced,
            rows,
            columns,
            batch: parallel
                .into_iter()
                .filter(|&axis| axis != rows && axis != columns)
                .collect(),
        })
    }
}

// The half-precision type whose values `node` holds exactly: a load of a `F16` or `BF16` operand, or a value rounded to one, before or after widening back.
fn half_value(kernel: &Kernel, node: NodeId) -> Option<ElementType> {
    let half = |element: ElementType| {
        matches!(element, ElementType::F16 | ElementType::Bf16).then_some(element)
    };
    match &kernel.nodes[node.0].expr {
        Expr::Load(access) if access.indirect.is_empty() => {
            half(kernel.operands[access.operand.0].element)
        }
        Expr::Cast(inner) => half(kernel.nodes[node.0].element).or_else(|| {
            (kernel.nodes[node.0].element == ElementType::F32)
                .then(|| half(kernel.nodes[inner.0].element))
                .flatten()
        }),
        _ => None,
    }
}
