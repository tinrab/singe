//! Parallel tiles: one schedule point covering several points of up to two parallel axes, sharing the loads those axes do not move.
//!
//! Tiling axis `a` by `t` rewrites a kernel so `a` takes `extent / t` values, and every node and store that depends on `a` appears `t` times, the copy for `i` reading the point `a · t + i`.
//! Nodes that do not depend on `a`, such as a matmul's weight for its row axis, appear once, so a row schedule whose reduction loops keep loads in registers reads them from memory once for all `t` points.
//! A second tile rewrites the result along another axis the same way, so a skinny matmul shares its weight across rows and its inputs across columns.
//! Targets run the rewritten kernel under the schedule without its tile.

use crate::{
    ir::{Access, AxisId, AxisKind, BinaryOp, Expr, Kernel, Literal, NodeId, Size, SizeId, Store},
    schedule::{Schedule, ScheduleError, size_values, space::operand_lengths},
};

/// Returns the parallel axis a tile of `kernel` covers: the one along which the loads that walk reduction axes but not it span the most elements, when some load does.
#[must_use]
pub fn tiled_axis(kernel: &Kernel, arguments: &[u64]) -> Option<AxisId> {
    shared_axis(kernel, arguments, None)
}

/// Returns the two parallel axes a schedule's tiles cover: [`tiled_axis`], and the axis that shares the most among the others.
#[must_use]
pub fn tiled_axes(kernel: &Kernel, arguments: &[u64]) -> [Option<AxisId>; 2] {
    let first = tiled_axis(kernel, arguments);
    [
        first,
        first.and_then(|first| shared_axis(kernel, arguments, Some(first))),
    ]
}

/// Returns the parallel axis other than `except` along which the most elements of loads that walk reduction axes are shared.
fn shared_axis(kernel: &Kernel, arguments: &[u64], except: Option<AxisId>) -> Option<AxisId> {
    let lengths = operand_lengths(kernel, arguments)?;
    let reduces = |axis: AxisId| kernel.axes[axis.0].kind == AxisKind::Reduction;
    let loads = kernel
        .nodes
        .iter()
        .filter_map(|node| match &node.expr {
            Expr::Load(access) => Some(access),
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut best: Option<(u64, AxisId)> = None;
    for axis in (0..kernel.axes.len()).map(AxisId) {
        if reduces(axis) || Some(axis) == except {
            continue;
        }

        let shared = loads
            .iter()
            .filter(|access| !access.axes().any(|walked| walked == axis))
            .filter(|access| access.axes().any(reduces))
            .map(|access| lengths[access.operand.0])
            .sum::<u64>();
        if shared > 0 && best.is_none_or(|(most, _)| shared > most) {
            best = Some((shared, axis));
        }
    }
    best.map(|(_, axis)| axis)
}

/// Returns the kernel and argument sizes that run `kernel` with `arguments` under `schedule`'s parallel tiles, or `None` when the schedule has none.
///
/// The first tile covers [`tiled_axis`], and the second the axis that shares the most among the others.
///
/// # Errors
///
/// Returns an error if the kernel has no axis to tile, a tile does not divide its axis's extent, or an access moves along a tiled axis with a stride or base given at launch.
pub fn tiled(
    kernel: &Kernel,
    schedule: Schedule,
    arguments: &[u64],
) -> Result<Option<(Kernel, Vec<u64>)>, ScheduleError> {
    let [first, second] = schedule.tiles();
    if [first, second] == [1, 1] {
        return Ok(None);
    }

    let untileable = ScheduleError::Untileable {
        reason: "no load is shared across a parallel axis",
    };
    let [axis, other] = tiled_axes(kernel, arguments);
    let (tiled, arguments) = tile_parallel_axis(
        kernel,
        arguments,
        axis.ok_or_else(|| untileable.clone())?,
        first,
    )?;
    if second == 1 {
        return Ok(Some((tiled, arguments)));
    }
    tile_parallel_axis(&tiled, &arguments, other.ok_or(untileable)?, second).map(Some)
}

/// Returns `kernel` with parallel axis `axis` tiled by `factor`, and the argument sizes it takes in place of `arguments`.
///
/// # Errors
///
/// Returns an error if `factor` does not divide the axis's extent, or an access moves along the axis with a stride or base given at launch.
pub fn tile_parallel_axis(
    kernel: &Kernel,
    arguments: &[u64],
    axis: AxisId,
    factor: u32,
) -> Result<(Kernel, Vec<u64>), ScheduleError> {
    let sizes = size_values(kernel, arguments)?;
    let extent = sizes[kernel.axes[axis.0].extent.0];
    let step = u64::from(factor);
    if factor == 1 {
        return Ok((kernel.clone(), arguments.to_vec()));
    }

    if kernel.axes[axis.0].kind != AxisKind::Parallel || extent % step != 0 {
        return Err(ScheduleError::TileFactor { factor, extent });
    }

    let mut tiled = Kernel {
        name: kernel.name.clone(),
        operands: kernel.operands.clone(),
        sizes: kernel.sizes.clone(),
        scalars: kernel.scalars.clone(),
        axes: kernel.axes.clone(),
        nodes: Vec::new(),
        stores: Vec::new(),
    };
    let mut arguments = arguments.to_vec();
    let tiled_extent = match kernel.sizes[kernel.axes[axis.0].extent.0] {
        Size::Static(_) => tiled.add_size(Size::Static(extent / step)),
        Size::Argument => {
            arguments.push(extent / step);
            tiled.add_size(Size::Argument)
        }
    };
    tiled.axes[axis.0].extent = tiled_extent;
    let mut rewrite = Rewrite {
        kernel,
        sizes: &sizes,
        axis,
        factor,
        copies: Vec::with_capacity(kernel.nodes.len()),
    };
    for index in 0..kernel.nodes.len() {
        let node = NodeId(index);
        let count = if rewrite.depends(node) { factor } else { 1 };
        let mut copies = Vec::with_capacity(count as usize);
        for copy in 0..count {
            copies.push(rewrite.node(&mut tiled, node, copy)?);
        }

        rewrite.copies.push(copies);
    }

    for store in &kernel.stores {
        let count = if rewrite.walks(&store.access) || rewrite.depends(store.value) {
            factor
        } else {
            1
        };
        for copy in 0..count {
            let access = rewrite.access(&mut tiled, &store.access, copy)?;
            tiled.stores.push(Store {
                access,
                value: rewrite.copy(store.value, copy),
                combiner: store.combiner,
            });
        }
    }
    Ok((tiled, arguments))
}

/// The state of one rewrite: the source kernel, its size values, the tiled axis, and each source node's copies in the tiled kernel.
struct Rewrite<'a> {
    kernel: &'a Kernel,
    sizes: &'a [u64],
    axis: AxisId,
    factor: u32,
    copies: Vec<Vec<NodeId>>,
}

impl Rewrite<'_> {
    fn depends(&self, node: NodeId) -> bool {
        self.kernel.domain(node).contains(&self.axis)
    }

    fn walks(&self, access: &Access) -> bool {
        self.kernel.access_domain(access).contains(&self.axis)
    }

    /// Returns the copy of source node `node` for tile point `copy`, which is its only copy when it does not depend on the axis.
    fn copy(&self, node: NodeId, copy: u32) -> NodeId {
        let copies = &self.copies[node.0];
        copies[if copies.len() == 1 { 0 } else { copy as usize }]
    }

    /// Adds the copy of source node `node` for tile point `copy` to `tiled`.
    fn node(&self, tiled: &mut Kernel, node: NodeId, copy: u32) -> Result<NodeId, ScheduleError> {
        let source = &self.kernel.nodes[node.0];
        let element = source.element;
        let map = |id: NodeId| self.copy(id, copy);
        let expr = match &source.expr {
            Expr::Load(access) => Expr::Load(self.access(tiled, access, copy)?),
            Expr::AxisIndex(axis) if *axis == self.axis => {
                // The source index is `a · factor + copy`.
                let index = tiled.add_node(element, Expr::AxisIndex(*axis));
                let factor = tiled.literal(element, Literal::Int(i64::from(self.factor)));
                let scaled = tiled.binary(BinaryOp::Mul, index, factor);
                let offset = tiled.literal(element, Literal::Int(i64::from(copy)));
                return Ok(tiled.binary(BinaryOp::Add, scaled, offset));
            }
            Expr::Unary(op, value) => Expr::Unary(*op, map(*value)),
            Expr::Binary(op, lhs, rhs) => Expr::Binary(*op, map(*lhs), map(*rhs)),
            Expr::MulAdd(a, b, c) => Expr::MulAdd(map(*a), map(*b), map(*c)),
            Expr::Select {
                condition,
                if_true,
                if_false,
            } => Expr::Select {
                condition: map(*condition),
                if_true: map(*if_true),
                if_false: map(*if_false),
            },
            Expr::Cast(value) => Expr::Cast(map(*value)),
            Expr::Reduce {
                combiner,
                axes,
                value,
            } => Expr::Reduce {
                combiner: *combiner,
                axes: axes.clone(),
                value: map(*value),
            },
            Expr::Scan {
                combiner,
                axis,
                value,
            } => Expr::Scan {
                combiner: *combiner,
                axis: *axis,
                value: map(*value),
            },
            Expr::ArgReduce {
                maximum,
                axes,
                value,
            } => Expr::ArgReduce {
                maximum: *maximum,
                axes: axes.clone(),
                value: map(*value),
            },
            Expr::SoftmaxSum { axes, logit, value } => Expr::SoftmaxSum {
                axes: axes.clone(),
                logit: map(*logit),
                value: map(*value),
            },
            Expr::CheckIndex {
                value,
                extent,
                wrap,
                status,
            } => Expr::CheckIndex {
                value: map(*value),
                extent: *extent,
                wrap: *wrap,
                status: *status,
            },
            Expr::Require {
                value,
                condition,
                status,
            } => Expr::Require {
                value: map(*value),
                condition: map(*condition),
                status: *status,
            },
            expr @ (Expr::Literal(_) | Expr::Scalar(_) | Expr::AxisIndex(_)) => expr.clone(),
        };
        Ok(tiled.add_node(element, expr))
    }

    /// Returns `access` for tile point `copy`: its stride along the axis times the factor, and its base advanced by `copy` strides.
    fn access(
        &self,
        tiled: &mut Kernel,
        access: &Access,
        copy: u32,
    ) -> Result<Access, ScheduleError> {
        let indirect = access
            .indirect
            .iter()
            .map(|&(node, stride)| (self.copy(node, copy), stride))
            .collect();
        let Some(&(_, stride)) = access.strides.iter().find(|&&(axis, _)| axis == self.axis) else {
            return Ok(Access {
                operand: access.operand,
                base: access.base,
                strides: access.strides.clone(),
                indirect,
            });
        };
        let stride_value = self.static_value(stride)?;
        let base_value = access.base.map_or(Ok(0), |base| self.static_value(base))?;
        let offset = base_value + u64::from(copy) * stride_value;
        let wide = tiled.add_size(Size::Static(stride_value * u64::from(self.factor)));
        let strides = access
            .strides
            .iter()
            .map(|&(axis, size)| (axis, if axis == self.axis { wide } else { size }))
            .collect();
        Ok(Access {
            operand: access.operand,
            base: (offset != 0).then(|| tiled.add_size(Size::Static(offset))),
            strides,
            indirect,
        })
    }

    fn static_value(&self, size: SizeId) -> Result<u64, ScheduleError> {
        match self.kernel.sizes[size.0] {
            Size::Static(_) => Ok(self.sizes[size.0]),
            Size::Argument => Err(ScheduleError::Untileable {
                reason: "an access moves along the tiled axis by a launch argument",
            }),
        }
    }
}

#[cfg(test)]
mod tests;
