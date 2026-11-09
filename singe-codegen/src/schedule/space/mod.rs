//! The schedules worth measuring for one kernel, and the memory each operand of a launch spans.
//!
//! The candidates follow from the kernel's structure: element kernels vary threads and vector width, row kernels the strategy that covers a row (a thread, some lanes of a warp, or a block) with its threads, lanes, vector width, and parallel tile, and tile targets their tile size.
//! Every candidate launches for the given argument sizes; an emitter may still reject one, which the tuner skips.

use crate::{
    ir::{AxisId, AxisKind, BinaryOp, Expr, Kernel, Literal, NodeId},
    schedule::{
        MAX_TILE, Schedule, contraction::Contraction, parallel_tile::tiled_axes,
        reduces_contiguously, reduction_vector, size_values, vector_width,
    },
};

/// Returns the schedules to measure for `kernel` when its argument sizes take the values in `arguments`, with tile schedules when `tile` holds.
///
/// Returns nothing when `arguments` does not match the kernel.
#[must_use]
pub fn candidates(kernel: &Kernel, arguments: &[u64], tile: bool) -> Vec<Schedule> {
    let Ok(sizes) = size_values(kernel, arguments) else {
        return Vec::new();
    };
    let points = |kind| {
        kernel
            .axes
            .iter()
            .filter(|axis| axis.kind == kind)
            .map(|axis| sizes[axis.extent.0])
            .fold(1_u64, u64::saturating_mul)
    };
    let has_reduction = kernel
        .axes
        .iter()
        .any(|axis| axis.kind == AxisKind::Reduction);
    let mut schedules = Vec::new();
    if has_reduction {
        let width = points(AxisKind::Reduction);
        let span = width.max(1).next_power_of_two();
        let vectors = powers_of_two(reduction_vector(kernel, 16));
        if width <= 64 {
            for threads in [128, 256] {
                schedules.push(Schedule::Elements { threads, vector: 1 });
            }
        }

        for &vector in &vectors {
            for lanes in [1, 2, 4, 8, 16, 32] {
                let covered = u64::from(lanes * vector);
                // Every lane has a vector, and none walks more than 64 of them.
                if covered > span || width.div_ceil(covered) > 64 {
                    continue;
                }
                for threads in [128, 256] {
                    schedules.push(Schedule::Warps {
                        threads,
                        vector,
                        lanes,
                        tiles: [1, 1],
                    });
                }
            }
            for threads in [32, 64, 128, 256, 512, 1024] {
                let covered = u64::from(threads * vector);
                if (covered > span && threads > 32) || width.div_ceil(covered) > 64 {
                    continue;
                }

                schedules.push(Schedule::Rows {
                    threads,
                    vector,
                    tiles: [1, 1],
                });
            }
        }

        // Column schedules for reductions over outer axes, whose consecutive points are consecutive in memory; a block holds at least eight points, so its loads coalesce.
        if !reduces_contiguously(kernel) {
            for threads in [128, 256] {
                for splits in [2, 4, 8, 16, 32] {
                    if threads / splits >= 8 && width >= u64::from(splits) {
                        schedules.push(Schedule::Columns { threads, splits });
                    }
                }
            }
        }

        // Parallel tiles of the widest vector schedules, for kernels whose loads some parallel axis shares; a tile holds at most 32 points.
        let [first, second] = tiled_axes(kernel, arguments);
        if let Some(first) = first {
            let extent =
                |axis: Option<AxisId>| axis.map_or(1, |axis| sizes[kernel.axes[axis.0].extent.0]);
            let divides = |axis, factor: u32| extent(axis) % u64::from(factor) == 0;
            let widest = vectors.last().copied().unwrap_or(1);
            let bases = schedules
                .iter()
                .copied()
                .filter(|schedule| match *schedule {
                    Schedule::Rows {
                        threads, vector, ..
                    } => vector == widest && threads <= 256,
                    Schedule::Warps { vector, lanes, .. } => vector == widest && lanes >= 16,
                    _ => false,
                })
                .collect::<Vec<_>>();
            for outer in [1, 2, 4, 8, 16] {
                for inner in [1, 2, 4, 8] {
                    if [outer, inner] == [1, 1]
                        || outer * inner > 32
                        || !divides(Some(first), outer)
                        || (inner > 1 && (second.is_none() || !divides(second, inner)))
                    {
                        continue;
                    }

                    let tiles = [outer, inner];
                    schedules.extend(bases.iter().map(|schedule| match *schedule {
                        Schedule::Rows {
                            threads, vector, ..
                        } => Schedule::Rows {
                            threads,
                            vector,
                            tiles,
                        },
                        Schedule::Warps {
                            threads,
                            vector,
                            lanes,
                            ..
                        } => Schedule::Warps {
                            threads,
                            vector,
                            lanes,
                            tiles,
                        },
                        other => other,
                    }));
                }
            }
        }

        if tile && span <= u64::from(MAX_TILE) {
            schedules.push(Schedule::TileRows {
                width: u32::try_from(span).unwrap_or(MAX_TILE),
            });
        }
        // Tensor-core tiles for plain contractions, from the 16-point tiles a short contraction fits to the 128-point tiles of large ones.
        if tile && Contraction::of(kernel).is_some() {
            for [m, n, k] in [
                [16, 64, 64],
                [32, 64, 64],
                [64, 64, 32],
                [64, 64, 64],
                [64, 128, 32],
                [128, 64, 32],
                [128, 128, 32],
                [128, 128, 64],
                [128, 256, 32],
            ] {
                schedules.push(Schedule::TileMatmul { m, n, k });
            }
        }
    } else {
        for vector in powers_of_two(vector_width(kernel)) {
            for threads in [64, 128, 256, 512] {
                schedules.push(Schedule::Elements { threads, vector });
            }
        }
        if tile {
            for size in [256, 1024, 4096] {
                schedules.push(Schedule::Tiles { size });
            }
        }
    }

    schedules.retain(|schedule| schedule.launch(kernel, arguments).is_ok());
    schedules
}

/// Returns how many elements of each operand a launch with argument sizes `arguments` addresses: one past its largest offset, or zero for an operand no access reaches.
///
/// Returns `None` when an access adds an indexed term whose largest value [`index_bound`] cannot find, as for indices loaded from memory, or `arguments` does not match the kernel.
#[must_use]
pub fn operand_lengths(kernel: &Kernel, arguments: &[u64]) -> Option<Vec<u64>> {
    let sizes = size_values(kernel, arguments).ok()?;
    let mut lengths = vec![0_u64; kernel.operands.len()];
    let loads = kernel.nodes.iter().filter_map(|node| match &node.expr {
        Expr::Load(access) => Some(access),
        _ => None,
    });
    for access in loads.chain(kernel.stores.iter().map(|store| &store.access)) {
        let mut last = access.base.map_or(0, |base| sizes[base.0]);
        for &(node, stride) in &access.indirect {
            last =
                last.checked_add(index_bound(kernel, &sizes, node)?.checked_mul(sizes[stride.0])?)?;
        }

        let mut empty = false;
        for &(axis, stride) in &access.strides {
            let extent = sizes[kernel.axes[axis.0].extent.0];
            empty |= extent == 0;
            last = last.checked_add(extent.saturating_sub(1).checked_mul(sizes[stride.0])?)?;
        }
        if !empty {
            let length = &mut lengths[access.operand.0];
            *length = (*length).max(last + 1);
        }
    }
    Some(lengths)
}

/// Returns the largest value integer `node` of `kernel` takes, with sizes of the values `sizes`, when it follows from axis indices, non-negative literals, sums, products, minimums, selections, and checked indices alone.
fn index_bound(kernel: &Kernel, sizes: &[u64], node: NodeId) -> Option<u64> {
    let bound = |node: NodeId| index_bound(kernel, sizes, node);
    match &kernel.nodes[node.0].expr {
        Expr::Literal(Literal::Int(value)) => u64::try_from(*value).ok(),
        Expr::AxisIndex(axis) => Some(sizes[kernel.axes[axis.0].extent.0].saturating_sub(1)),
        Expr::CheckIndex { extent, .. } => Some(sizes[extent.0].saturating_sub(1)),
        Expr::Cast(value) if kernel.nodes[value.0].element.is_integer() => bound(*value),
        Expr::Binary(BinaryOp::Add, lhs, rhs) => bound(*lhs)?.checked_add(bound(*rhs)?),
        Expr::Binary(BinaryOp::Mul, lhs, rhs) => bound(*lhs)?.checked_mul(bound(*rhs)?),
        // A minimum is at most either operand, so one bounded operand bounds it, as a clamp to the last index does.
        Expr::Binary(BinaryOp::Min, lhs, rhs) => match (bound(*lhs), bound(*rhs)) {
            (Some(lhs), Some(rhs)) => Some(lhs.min(rhs)),
            (bounded, None) | (None, bounded) => bounded,
        },
        Expr::Select {
            if_true, if_false, ..
        } => Some(bound(*if_true)?.max(bound(*if_false)?)),
        _ => None,
    }
}

/// Returns the powers of two from one up to `limit`.
fn powers_of_two(limit: u32) -> Vec<u32> {
    let mut powers = vec![1];
    while powers[powers.len() - 1] * 2 <= limit {
        powers.push(powers[powers.len() - 1] * 2);
    }
    powers
}

#[cfg(test)]
mod tests;
