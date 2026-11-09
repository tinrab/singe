use crate::{
    emit::{
        EmitError,
        cuda_c::{
            Writer,
            reduction::{Accumulator, LoopAction},
            vector::VectorSpan,
        },
    },
    ir::{AxisId, AxisKind, Expr, NodeId, Size},
    schedule::ThreadSchedule,
};

/// The registers a row schedule keeps values in: `count` vectors of the reduction axes per thread, `step` vectors apart.
pub struct RowCache {
    pub count: u32,
    pub step: u32,
    /// The points of the innermost reduction axis each of the `count` iterations covers.
    pub vector: u32,
    /// The loads, and while registers allow the computed values, that several loops read, each kept in `cached{node}[count * vector]`.
    pub values: Vec<NodeId>,
    /// The cached values an earlier loop has filled.
    pub filled: Vec<NodeId>,
}

impl Writer<'_> {
    /// Returns how a row schedule can keep loads in registers: when every loop walks all the reduction axes, their extents are static, and each thread keeps at most 64 values of each.
    pub fn row_cache(&self) -> Result<Option<RowCache>, EmitError> {
        let (step, vector) = match self.schedule {
            ThreadSchedule::Rows {
                threads, vector, ..
            } => (threads, vector),
            ThreadSchedule::Warps { vector, lanes, .. } => (lanes, vector),
            ThreadSchedule::Elements { .. }
            | ThreadSchedule::Columns { .. }
            | ThreadSchedule::Mma { .. } => return Ok(None),
        };
        let reduction = self.axes_of_kind(AxisKind::Reduction);
        if reduction.is_empty() {
            return Ok(None);
        }

        // A scan's passes walk contiguous chunks of the row, not the cache's strided points.
        if self
            .kernel
            .nodes
            .iter()
            .any(|node| matches!(node.expr, Expr::Scan { .. }))
        {
            return Ok(None);
        }

        // Every loop must walk all the reduction axes, so a point's register is the same in each.
        let mut loops = Vec::new();
        for index in 0..self.kernel.nodes.len() {
            if self.is_outer(NodeId(index))
                && let Some((axes, accumulator)) = Accumulator::of(self.kernel, NodeId(index))
            {
                loops.push((axes, LoopAction::Combine(vec![accumulator])));
            }
        }

        for store in &self.kernel.stores {
            let axes = self
                .kernel
                .access_domain(&store.access)
                .into_iter()
                .filter(|axis| self.kernel.axes[axis.0].kind == AxisKind::Reduction)
                .collect::<Vec<_>>();
            if !axes.is_empty() {
                loops.push((axes, LoopAction::Store(store)));
            }
        }

        let mut reads = vec![0_usize; self.kernel.nodes.len()];
        let mut spans = Vec::new();
        for (axes, action) in &loops {
            let mut sorted = axes.clone();
            sorted.sort();
            if sorted != reduction {
                return Ok(None);
            }

            // Vector loops also need every loop to cover the same vectors of the same axes.
            if vector > 1 {
                let Some(span) = self.vector_span(axes, &action.roots(), action, vector)? else {
                    return Ok(None);
                };
                spans.push(span);
            }
            for node in self.needed(&action.roots(), axes)? {
                reads[node.0] += 1;
            }
        }

        let span = spans.first().copied();
        if spans.iter().any(|&other| Some(other) != span) {
            return Ok(None);
        }

        // A vector loop covers its grouped axis in groups of points, the last of which may be partial, and the innermost axis it spans whole.
        let mut groups = 1_u64;
        for axis in &reduction {
            let Size::Static(extent) = self.kernel.sizes[self.kernel.axes[axis.0].extent.0] else {
                return Ok(None);
            };
            groups = groups.saturating_mul(match span {
                Some(span) if span.grouped == *axis => {
                    extent.div_ceil(u64::from(span.grouped_points(vector)))
                }
                Some(VectorSpan {
                    within: Some((within, _)),
                    ..
                }) if within == *axis => 1,
                _ => extent,
            });
        }

        // Each thread covers `count` vectors of the row, and keeps at most 64 values of each cached load, as a thread that walks a strided row of 64 channels alone does.
        let count = groups.div_ceil(u64::from(step));
        if count == 0 || count * u64::from(vector) > 64 {
            return Ok(None);
        }

        let (mut values, computed): (Vec<_>, Vec<_>) = (0..self.kernel.nodes.len())
            .filter(|&node| reads[node] > 1)
            .map(NodeId)
            .partition(|node| matches!(self.kernel.nodes[node.0].expr, Expr::Load(_)));
        // Values several loops compute, such as a softmax's exponentials, are kept too while every cached value of a thread fits in 128 registers.
        let total = u64::try_from(values.len() + computed.len()).unwrap_or(u64::MAX);
        if total.saturating_mul(count * u64::from(vector)) <= 128 {
            values.extend(computed);
        }
        Ok(Some(RowCache {
            count: u32::try_from(count).expect("at most 64 points"),
            step,
            vector,
            values,
            filled: Vec::new(),
        }))
    }

    /// Emits an unrolled loop over the reduction axes `axes` in which each thread covers its points of the row, reading cached loads from registers once an earlier loop filled them.
    pub fn cached_loop(
        &mut self,
        axes: &[AxisId],
        roots: &[NodeId],
        action: &LoopAction<'_>,
    ) -> Result<(), EmitError> {
        let (ix, sx) = (self.index, self.suffix);
        let Some(cache) = &self.cache else {
            unreachable!("the caller checked for a cache");
        };
        let (count, step) = (cache.count, cache.step);
        let start = match self.schedule {
            ThreadSchedule::Rows { .. } => "threadIdx.x",
            _ => "warp_lane",
        };
        let total = self.product(axes);
        self.line("#pragma unroll");
        self.line(&format!(
            "for (unsigned int point = 0; point < {count}U; ++point) {{"
        ));
        self.line(&format!("const {ix} flat = {start} + point * {step}{sx};"));
        self.line(&format!("if (flat < {total}) {{"));
        self.decompose("flat", axes);
        let mut filled = Vec::new();
        for node in self.needed(roots, axes)? {
            let (cached, filled_before) = self.cache.as_ref().map_or((false, false), |cache| {
                (cache.values.contains(&node), cache.filled.contains(&node))
            });
            if filled_before {
                self.define(node, &format!("cached{}[point]", node.0));
            } else {
                self.emit_value(node)?;
                if cached {
                    self.line(&format!("cached{0}[point] = v{0};", node.0));
                    filled.push(node);
                }
            }
        }

        let statement = self.action_statement(action, None);
        self.line(&statement);
        self.line("}");
        self.line("}");
        if let Some(cache) = &mut self.cache {
            cache.filled.extend(filled);
        }
        Ok(())
    }
}
