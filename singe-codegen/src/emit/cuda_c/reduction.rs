use std::mem;

use crate::{
    emit::{
        EmitError,
        cuda_c::{
            Writer,
            spelling::{combiner_tag, compute_type, identity, store_expression},
        },
    },
    ir::{AxisId, AxisKind, Combiner, Expr, Kernel, NodeId, Store},
    schedule::ThreadSchedule,
};

/// What each point of a loop over reduction axes does.
pub enum LoopAction<'a> {
    /// Combines each accumulator's value into it.
    Combine(Vec<Accumulator>),
    /// Writes a store whose access walks the loop's axes.
    Store(&'a Store),
}

impl LoopAction<'_> {
    /// Returns the nodes the action reads.
    pub fn roots(&self) -> Vec<NodeId> {
        match self {
            Self::Combine(accumulators) => {
                accumulators.iter().flat_map(Accumulator::roots).collect()
            }
            Self::Store(store) => {
                let mut roots = vec![store.value];
                roots.extend(store.access.indirect.iter().map(|&(node, _)| node));
                roots
            }
        }
    }
}

/// The accumulator of reduction `node`, which folds `value` at each point of its loop.
#[derive(Debug, Clone, Copy)]
pub struct Accumulator {
    pub node: NodeId,
    pub fold: Fold,
    pub value: NodeId,
}

impl Accumulator {
    /// Returns the axes reduction `node` of `kernel` reduces and its accumulator, or `None` for any other node.
    pub fn of(kernel: &Kernel, node: NodeId) -> Option<(Vec<AxisId>, Self)> {
        let (axes, fold, value) = match &kernel.nodes[node.0].expr {
            Expr::Reduce {
                combiner,
                axes,
                value,
            } => (axes, Fold::Combine(*combiner), *value),
            Expr::SoftmaxSum { axes, logit, value } => {
                (axes, Fold::Softmax { logit: *logit }, *value)
            }
            Expr::ArgReduce {
                maximum,
                axes,
                value,
            } => (axes, Fold::Arg { maximum: *maximum }, *value),
            _ => return None,
        };
        Some((axes.clone(), Self { node, fold, value }))
    }

    /// Returns the nodes the accumulator reads at each point.
    pub fn roots(&self) -> Vec<NodeId> {
        match self.fold {
            Fold::Combine(_) | Fold::Arg { .. } => vec![self.value],
            Fold::Softmax { logit } => vec![logit, self.value],
        }
    }
}

/// How an accumulator folds the points of its loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fold {
    /// Combines each value into `r{node}`.
    Combine(Combiner),
    /// Weights each value by the softmax of `logit`, keeping the running maximum logit in `r{node}_max`, the rescaled sum of exponentials in `r{node}_sum`, and the rescaled weighted sum in `r{node}`.
    Softmax { logit: NodeId },
    /// Keeps the best value so far in `r{node}_value` and its position over the loop's axes in `r{node}`, the largest with `maximum` and the smallest without.
    Arg { maximum: bool },
}

/// Returns the value `accumulator` holds once a thread has folded every point itself.
pub fn thread_result(accumulator: &Accumulator) -> String {
    let n = accumulator.node.0;
    match accumulator.fold {
        Fold::Combine(_) | Fold::Arg { .. } => format!("r{n}"),
        Fold::Softmax { .. } => format!("(r{n} / r{n}_sum)"),
    }
}

/// Returns the axes a reduction reduces, or `None` for any other node.
pub fn reduced_axes(expr: &Expr) -> Option<&[AxisId]> {
    match expr {
        Expr::Reduce { axes, .. }
        | Expr::SoftmaxSum { axes, .. }
        | Expr::ArgReduce { axes, .. } => Some(axes),
        _ => None,
    }
}

/// Returns the nodes `expr` reads.
pub fn operands(expr: &Expr) -> Vec<NodeId> {
    match expr {
        Expr::Load(access) => access.indirect.iter().map(|&(node, _)| node).collect(),
        Expr::Literal(_) | Expr::Scalar(_) | Expr::AxisIndex(_) => Vec::new(),
        Expr::Unary(_, value)
        | Expr::Cast(value)
        | Expr::Reduce { value, .. }
        | Expr::ArgReduce { value, .. }
        | Expr::Scan { value, .. }
        | Expr::CheckIndex { value, .. } => vec![*value],
        Expr::Binary(_, lhs, rhs)
        | Expr::SoftmaxSum {
            logit: lhs,
            value: rhs,
            ..
        } => vec![*lhs, *rhs],
        Expr::Require {
            value, condition, ..
        } => vec![*value, *condition],
        Expr::MulAdd(a, b, c) => vec![*a, *b, *c],
        Expr::Select {
            condition,
            if_true,
            if_false,
        } => vec![*condition, *if_true, *if_false],
    }
}

impl Writer<'_> {
    /// Returns whether reduction `later` can share the loop of reduction `first`: nothing it reads is a reduction not yet emitted, or an outer value defined after `first`, which the loop could not see.
    pub fn joins(&self, later: NodeId, first: NodeId, reduced: &[bool]) -> bool {
        let expr = &self.kernel.nodes[later.0].expr;
        if reduced_axes(expr).is_none() {
            return false;
        }

        let mut pending = operands(expr);
        let mut seen = vec![false; self.kernel.nodes.len()];
        while let Some(node) = pending.pop() {
            if mem::replace(&mut seen[node.0], true) {
                continue;
            }

            // Reductions inside another's body run in its loop, so only outer ones must come first.
            let reduction =
                reduced_axes(&self.kernel.nodes[node.0].expr).is_some() && self.is_outer(node);
            if (reduction && !reduced[node.0]) || (self.is_outer(node) && node.0 > first.0) {
                return false;
            }
            if !self.is_outer(node) {
                pending.extend(operands(&self.kernel.nodes[node.0].expr));
            }
        }
        true
    }

    /// Emits the reductions `nodes`, all over the same axes, in one loop that combines each value into its own accumulator.
    pub fn reduce_group(&mut self, nodes: &[NodeId]) -> Result<(), EmitError> {
        let mut accumulators = Vec::with_capacity(nodes.len());
        let mut axes = Vec::new();
        for &node in nodes {
            let (reduced, accumulator) =
                Accumulator::of(self.kernel, node).expect("the caller groups reductions");
            self.declare(&accumulator);
            accumulators.push(accumulator);
            axes = reduced;
        }

        self.reduction_loop(&axes, &LoopAction::Combine(accumulators.clone()))?;
        for accumulator in accumulators {
            let node = accumulator.node;
            let ty = compute_type(self.kernel.nodes[node.0].element);
            let n = node.0;
            let value = match (self.schedule, accumulator.fold) {
                (ThreadSchedule::Elements { .. }, _) => thread_result(&accumulator),
                (ThreadSchedule::Mma { .. }, _) => {
                    unreachable!("mma schedules emit their own body")
                }
                (ThreadSchedule::Rows { .. }, Fold::Combine(combiner)) => {
                    let tag = combiner_tag(combiner);
                    self.line(&format!("__shared__ {ty} partials{n}[32];"));
                    format!("singe_block_reduce<{ty}, {tag}>(r{n}, partials{n})")
                }
                (ThreadSchedule::Rows { .. }, Fold::Softmax { .. }) => {
                    self.line(&format!("__shared__ {ty} partials{n}[96];"));
                    format!("singe_block_softmax<{ty}>(r{n}_max, r{n}_sum, r{n}, partials{n})")
                }
                (ThreadSchedule::Warps { lanes, .. }, Fold::Combine(combiner)) => {
                    let tag = combiner_tag(combiner);
                    format!("singe_warp_reduce<{ty}, {tag}, {lanes}>(r{n})")
                }
                (ThreadSchedule::Warps { lanes, .. }, Fold::Softmax { .. }) => {
                    format!("singe_warp_softmax<{ty}, {lanes}>(r{n}_max, r{n}_sum, r{n})")
                }
                // Every thread folds the splits of its point in split order, so later values see the same result in each split.
                (ThreadSchedule::Columns { threads, splits }, Fold::Combine(combiner)) => {
                    let tag = combiner_tag(combiner);
                    let points = threads / splits;
                    self.line(&format!("__shared__ {ty} partials{n}[{threads}];"));
                    self.line(&format!("partials{n}[threadIdx.x] = r{n};"));
                    self.line("__syncthreads();");
                    self.line(&format!(
                        "{ty} column{n} = partials{n}[threadIdx.x % {points}U];"
                    ));
                    self.line(&format!("for (unsigned int part = 1; part < {splits}U; ++part) column{n} = singe_combine<{ty}, {tag}>(column{n}, partials{n}[part * {points}U + threadIdx.x % {points}U]);"));
                    format!("column{n}")
                }
                (ThreadSchedule::Columns { .. }, Fold::Softmax { .. }) => {
                    return Err(EmitError::Unsupported {
                        reason: "softmax folds under column schedules",
                    });
                }
                (ThreadSchedule::Rows { .. }, Fold::Arg { maximum }) => {
                    let vt = self.arg_type(&accumulator);
                    self.line(&format!("__shared__ {vt} values{n}[32];"));
                    self.line(&format!("__shared__ long long indices{n}[32];"));
                    format!(
                        "singe_block_arg<{vt}, {maximum}>(r{n}_value, r{n}, values{n}, indices{n})"
                    )
                }
                (ThreadSchedule::Warps { lanes, .. }, Fold::Arg { maximum }) => {
                    let vt = self.arg_type(&accumulator);
                    format!("singe_warp_arg<{vt}, {maximum}, {lanes}>(r{n}_value, r{n})")
                }
                // Every thread steps through the splits of its point in split order; ties go to the earlier position whatever the order.
                (ThreadSchedule::Columns { threads, splits }, Fold::Arg { maximum }) => {
                    let vt = self.arg_type(&accumulator);
                    let points = threads / splits;
                    self.line(&format!("__shared__ {vt} values{n}[{threads}];"));
                    self.line(&format!("__shared__ long long indices{n}[{threads}];"));
                    self.line(&format!("values{n}[threadIdx.x] = r{n}_value;"));
                    self.line(&format!("indices{n}[threadIdx.x] = r{n};"));
                    self.line("__syncthreads();");
                    self.line(&format!("for (unsigned int part = 0; part < {splits}U; ++part) singe_arg_step<{vt}, {maximum}>(r{n}_value, r{n}, values{n}[part * {points}U + threadIdx.x % {points}U], indices{n}[part * {points}U + threadIdx.x % {points}U]);"));
                    format!("r{n}")
                }
            };
            self.define(node, &value);
        }
        Ok(())
    }

    /// Declares the registers `accumulator` folds its loop's points into, at the fold's identity.
    pub fn declare(&mut self, accumulator: &Accumulator) {
        let element = self.kernel.nodes[accumulator.node.0].element;
        let ty = compute_type(element);
        let n = accumulator.node.0;
        match accumulator.fold {
            Fold::Combine(combiner) => {
                self.line(&format!("{ty} r{n} = {};", identity(combiner, element)));
            }
            Fold::Softmax { .. } => {
                let largest = identity(Combiner::Max, element);
                self.line(&format!(
                    "{ty} r{n}_max = {largest}, r{n}_sum = static_cast<{ty}>(0), r{n} = static_cast<{ty}>(0);"
                ));
            }
            // A negative position marks an accumulator that has seen no point yet.
            Fold::Arg { .. } => {
                let vt = self.arg_type(accumulator);
                self.line(&format!(
                    "{vt} r{n}_value = static_cast<{vt}>(0); long long r{n} = -1;"
                ));
            }
        }
    }

    /// Returns the compute type of the values an argument reduction compares.
    fn arg_type(&self, accumulator: &Accumulator) -> &'static str {
        compute_type(self.kernel.nodes[accumulator.value.0].element)
    }

    /// Returns the current point's row-major position over the axes reduction `node` reduces.
    fn arg_position(&self, node: NodeId) -> String {
        let axes = reduced_axes(&self.kernel.nodes[node.0].expr).unwrap_or_default();
        let mut position = String::from("0LL");
        for axis in axes {
            let extent = self.size(self.kernel.axes[axis.0].extent);
            position = format!(
                "({position} * static_cast<long long>({extent}) + static_cast<long long>(a{}))",
                axis.0
            );
        }
        position
    }

    /// Emits reduction `node`, which a loop over other reduction axes reads, as a loop of each thread's own over the reduction's axes, and defines its value.
    pub fn nested_reduction(&mut self, node: NodeId) -> Result<(), EmitError> {
        let ix = self.index;
        let (axes, accumulator) =
            Accumulator::of(self.kernel, node).expect("the caller passes a reduction");
        self.declare(&accumulator);
        for axis in &axes {
            let extent = self.size(self.kernel.axes[axis.0].extent);
            self.line(&format!(
                "for ({ix} a{0} = 0; a{0} < {extent}; ++a{0}) {{",
                axis.0
            ));
        }

        let mut needed = vec![false; self.kernel.nodes.len()];
        for root in accumulator.roots() {
            self.mark_scope(
                root,
                &|writer: &Self, node: NodeId| {
                    writer.domains[node.0]
                        .iter()
                        .any(|axis| axes.contains(axis))
                },
                &[],
                &mut needed,
            );
        }

        for inner in (0..needed.len()).filter(|&index| needed[index]) {
            self.emit_value(NodeId(inner))?;
        }

        let statement = self.fold_statement(&accumulator);
        self.line(&statement);
        for _ in &axes {
            self.line("}");
        }

        self.define(node, &thread_result(&accumulator));
        Ok(())
    }

    /// Defines `node`, emitting its loop first when it is a reduction inside another's loop.
    pub fn emit_value(&mut self, node: NodeId) -> Result<(), EmitError> {
        if reduced_axes(&self.kernel.nodes[node.0].expr).is_some() {
            return self.nested_reduction(node);
        }

        // A scan folds the current point into its running value, which its loop declared.
        if let Expr::Scan {
            combiner, value, ..
        } = self.kernel.nodes[node.0].expr
        {
            let ty = compute_type(self.kernel.nodes[node.0].element);
            let (n, tag) = (node.0, combiner_tag(combiner));
            self.line(&format!(
                "r{n} = singe_combine<{ty}, {tag}>(r{n}, v{});",
                value.0
            ));
            self.define(node, &format!("r{n}"));
            return Ok(());
        }

        let expression = self.expression(node);
        self.define(node, &expression);
        Ok(())
    }

    /// Returns the statement that folds the current point's values into `accumulator`.
    pub fn fold_statement(&self, accumulator: &Accumulator) -> String {
        let ty = compute_type(self.kernel.nodes[accumulator.node.0].element);
        let (n, value) = (accumulator.node.0, accumulator.value.0);
        match accumulator.fold {
            Fold::Combine(combiner) => {
                let tag = combiner_tag(combiner);
                format!("r{n} = singe_combine<{ty}, {tag}>(r{n}, v{value});")
            }
            Fold::Softmax { logit } => format!(
                "singe_softmax_step<{ty}>(r{n}_max, r{n}_sum, r{n}, v{}, v{value});",
                logit.0
            ),
            Fold::Arg { maximum } => format!(
                "singe_arg_step<{}, {maximum}>(r{n}_value, r{n}, v{value}, {});",
                self.arg_type(accumulator),
                self.arg_position(accumulator.node)
            ),
        }
    }

    /// Emits a loop over the reduction axes `axes` that computes what `action` needs at each point and does it, covering the schedule's vector of points of the innermost axis at once where the loop's accesses allow.
    pub fn reduction_loop(
        &mut self,
        axes: &[AxisId],
        action: &LoopAction<'_>,
    ) -> Result<(), EmitError> {
        let roots = action.roots();
        let scans = self
            .needed(&roots, axes)?
            .into_iter()
            .filter(|node| matches!(self.kernel.nodes[node.0].expr, Expr::Scan { .. }))
            .collect::<Vec<_>>();
        if !scans.is_empty() {
            return self.scan_loop(axes, &scans, action);
        }

        let vector = match self.schedule {
            ThreadSchedule::Rows { vector, .. } | ThreadSchedule::Warps { vector, .. } => vector,
            ThreadSchedule::Elements { .. }
            | ThreadSchedule::Columns { .. }
            | ThreadSchedule::Mma { .. } => 1,
        };
        // Argument reductions walk points one at a time, so each fold knows its position.
        let arg = matches!(action, LoopAction::Combine(accumulators) if accumulators
            .iter()
            .any(|accumulator| matches!(accumulator.fold, Fold::Arg { .. })));
        if vector > 1 && !arg && self.vector_loop(axes, &roots, vector, action)? {
            return Ok(());
        }

        // A vector schedule caches only when every loop is a vector loop, so this one walks points.
        if self.cache.as_ref().is_some_and(|cache| cache.vector == 1) {
            return self.cached_loop(axes, &roots, action);
        }

        self.open_loop(axes);
        self.inner(&roots, axes)?;
        let statement = self.action_statement(action, None);
        self.line(&statement);
        self.line("}");
        Ok(())
    }

    // Emits a store loop over the kernel's one reduction axis that reads the inclusive scans `scans` along it.
    // An element schedule's thread walks the axis in order, carrying each running value; a row schedule's threads each take a contiguous chunk, total it, offset it by an exclusive scan of the block's totals, and walk it again.
    fn scan_loop(
        &mut self,
        axes: &[AxisId],
        scans: &[NodeId],
        action: &LoopAction<'_>,
    ) -> Result<(), EmitError> {
        let ix = self.index;
        if !matches!(action, LoopAction::Store(_))
            || self.axes_of_kind(AxisKind::Reduction).as_slice() != axes
        {
            return Err(EmitError::Unsupported {
                reason: "a scan read outside a store loop over the kernel's one reduction axis",
            });
        }

        let mut folds = Vec::with_capacity(scans.len());
        for &scan in scans {
            let Expr::Scan {
                combiner, value, ..
            } = self.kernel.nodes[scan.0].expr
            else {
                unreachable!("the caller passes scans");
            };
            let element = self.kernel.nodes[scan.0].element;
            folds.push((
                scan,
                combiner,
                value,
                compute_type(element),
                identity(combiner, element),
            ));
        }

        for &(scan, _, _, ty, ref start) in &folds {
            self.line(&format!("{ty} r{} = {start};", scan.0));
        }

        let roots = action.roots();
        match self.schedule {
            ThreadSchedule::Mma { .. } => unreachable!("mma schedules emit their own body"),
            ThreadSchedule::Elements { .. } => self.open_loop(axes),
            ThreadSchedule::Rows { .. } => {
                let total = self.product(axes);
                self.line(&format!(
                    "const {ix} chunk = ({total} + blockDim.x - 1) / blockDim.x;"
                ));
                self.line(&format!("const {ix} chunk_begin = threadIdx.x * chunk;"));
                self.line(&format!(
                    "const {ix} chunk_end = chunk_begin + chunk < {total} ? chunk_begin + chunk : {total};"
                ));
                // The first pass totals the thread's chunk.
                let values = folds.iter().map(|fold| fold.2).collect::<Vec<_>>();
                self.line(&format!(
                    "for ({ix} flat = chunk_begin; flat < chunk_end; ++flat) {{"
                ));
                self.decompose("flat", axes);
                for node in self.needed(&values, axes)? {
                    if matches!(self.kernel.nodes[node.0].expr, Expr::Scan { .. }) {
                        return Err(EmitError::Unsupported {
                            reason: "a scan of a scan",
                        });
                    }

                    self.emit_value(node)?;
                }

                for &(scan, combiner, value, ty, _) in &folds {
                    let tag = combiner_tag(combiner);
                    self.line(&format!(
                        "r{0} = singe_combine<{ty}, {tag}>(r{0}, v{1});",
                        scan.0, value.0
                    ));
                }

                self.line("}");
                // The block's exclusive scan of the totals starts each chunk where the chunks before it end.
                for &(scan, combiner, _, ty, ref start) in &folds {
                    let (n, tag) = (scan.0, combiner_tag(combiner));
                    self.line(&format!("__shared__ {ty} scan_partials{n}[32];"));
                    self.line(&format!(
                        "r{n} = singe_block_exclusive_scan<{ty}, {tag}>(r{n}, scan_partials{n}, {start});"
                    ));
                }

                self.line(&format!(
                    "for ({ix} flat = chunk_begin; flat < chunk_end; ++flat) {{"
                ));
                self.decompose("flat", axes);
            }
            ThreadSchedule::Warps { .. } | ThreadSchedule::Columns { .. } => {
                return Err(EmitError::Unsupported {
                    reason: "a scan under a warp or column schedule",
                });
            }
        }

        self.inner(&roots, axes)?;
        let statement = self.action_statement(action, None);
        self.line(&statement);
        self.line("}");
        Ok(())
    }

    /// Returns the statement that does `action` with the values defined at the current point, writing a store into the staging array `staging` at `lane` when given.
    pub fn action_statement(&self, action: &LoopAction<'_>, staging: Option<&str>) -> String {
        match (action, staging) {
            (LoopAction::Combine(accumulators), _) => accumulators
                .iter()
                .map(|accumulator| self.fold_statement(accumulator))
                .collect::<Vec<_>>()
                .join("\n"),
            (LoopAction::Store(store), Some(staging)) => {
                let element = self.kernel.operands[store.access.operand.0].element;
                format!(
                    "{staging}[lane] = {};",
                    store_expression(element, &format!("v{}", store.value.0))
                )
            }
            (LoopAction::Store(store), None) => self.write(store),
        }
    }

    /// Opens a loop over `axes` and declares their indices; the caller closes it.
    pub fn open_loop(&mut self, axes: &[AxisId]) {
        let ix = self.index;
        match self.schedule {
            ThreadSchedule::Mma { .. } => unreachable!("mma schedules emit their own body"),
            ThreadSchedule::Elements { .. } => {
                for axis in axes {
                    let extent = self.size(self.kernel.axes[axis.0].extent);
                    self.line(&format!(
                        "for ({ix} a{0} = 0; a{0} < {extent}; ++a{0})",
                        axis.0
                    ));
                }

                self.line("{");
            }
            ThreadSchedule::Rows { .. } => {
                let total = self.product(axes);
                self.line(&format!(
                    "for ({ix} flat = threadIdx.x; flat < {total}; flat += blockDim.x) {{"
                ));
                self.decompose("flat", axes);
            }
            ThreadSchedule::Warps { lanes, .. } => {
                let total = self.product(axes);
                self.line(&format!(
                    "for ({ix} flat = warp_lane; flat < {total}; flat += {lanes}U) {{"
                ));
                self.decompose("flat", axes);
            }
            ThreadSchedule::Columns { splits, .. } => {
                let total = self.product(axes);
                self.line(&format!(
                    "for ({ix} flat = split; flat < {total}; flat += {splits}U) {{"
                ));
                self.decompose("flat", axes);
            }
        }
    }

    /// Emits, inside a loop over `loop_axes`, every node `roots` need that depends on reduction axes.
    pub fn inner(&mut self, roots: &[NodeId], loop_axes: &[AxisId]) -> Result<(), EmitError> {
        for node in self.needed(roots, loop_axes)? {
            self.emit_value(node)?;
        }
        Ok(())
    }

    /// Returns, in node order, every node `roots` need that depends on reduction axes, checking that a loop over `loop_axes` can compute them.
    pub fn needed(&self, roots: &[NodeId], loop_axes: &[AxisId]) -> Result<Vec<NodeId>, EmitError> {
        let mut needed = vec![false; self.kernel.nodes.len()];
        for &root in roots {
            self.mark_scope(
                root,
                &|writer: &Self, node: NodeId| !writer.is_outer(node),
                &[],
                &mut needed,
            );
        }

        let mut nodes = Vec::new();
        for index in needed
            .iter()
            .enumerate()
            .filter(|(_, needed)| **needed)
            .map(|(index, _)| index)
        {
            if self.domains[index].iter().any(|axis| {
                self.kernel.axes[axis.0].kind == AxisKind::Reduction && !loop_axes.contains(axis)
            }) {
                return Err(EmitError::Unsupported {
                    reason: "a value that depends on reduction axes its loop does not walk",
                });
            }

            nodes.push(NodeId(index));
        }
        Ok(nodes)
    }

    /// Marks `node` and the nodes it depends on that a loop defines: those `inside` holds for, which the loop's scope cannot see from outside it.
    ///
    /// The body of a reduction the loop reads belongs to the reduction's own loop: nodes that depend on axes in `hidden`, those of the reductions passed on the way, are not marked, though their operands may be.
    pub fn mark_scope(
        &self,
        node: NodeId,
        inside: &dyn Fn(&Self, NodeId) -> bool,
        hidden: &[AxisId],
        needed: &mut [bool],
    ) {
        let deeper = self.domains[node.0]
            .iter()
            .any(|axis| hidden.contains(axis));
        if !deeper {
            if needed[node.0] || !inside(self, node) {
                return;
            }

            needed[node.0] = true;
        }

        let expr = &self.kernel.nodes[node.0].expr;
        let mut hidden = hidden.to_vec();
        hidden.extend(reduced_axes(expr).into_iter().flatten());
        for operand in operands(expr) {
            self.mark_scope(operand, inside, &hidden, needed);
        }
    }

    /// Returns whether `node` depends only on parallel axes, so it is defined once per thread outside any loop.
    pub fn is_outer(&self, node: NodeId) -> bool {
        self.domains[node.0]
            .iter()
            .all(|axis| self.kernel.axes[axis.0].kind == AxisKind::Parallel)
    }
}
