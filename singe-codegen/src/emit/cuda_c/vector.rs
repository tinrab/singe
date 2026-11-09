use crate::{
    emit::{
        EmitError,
        cuda_c::{
            Writer, mentions,
            reduction::{LoopAction, reduced_axes},
            spelling::{compute_type, load_expression, storage_type, store_expression},
        },
    },
    ir::{Access, AxisId, AxisKind, ElementType, Expr, NodeId, Size},
    schedule::ThreadSchedule,
};

/// How the `vector` lanes of a group in a vector loop cover the loop's innermost axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectorSpan {
    /// The axis whose points the groups split, consecutive points to a group.
    pub grouped: AxisId,
    /// The innermost axis and its static extent, when each group covers all of it at each of `vector / extent` points of `grouped`, which comes before it.
    pub within: Option<(AxisId, u32)>,
}

impl VectorSpan {
    /// Returns how many points of the grouped axis a group of `vector` lanes covers.
    pub fn grouped_points(self, vector: u32) -> u32 {
        self.within.map_or(vector, |(_, extent)| vector / extent)
    }
}

impl Writer<'_> {
    /// Emits the body of an element schedule whose threads each cover `vector` consecutive points of the innermost parallel axis `inner`.
    ///
    /// Values that depend on `inner` are computed once per lane in an unrolled loop.
    /// Accesses that walk `inner` with a stride of one go through a staging array, read or written with one wide access when every lane is in range and the address is aligned, and lane by lane otherwise; other accesses that walk `inner` are guarded lane by lane.
    pub fn vector_body(
        &mut self,
        parallel: &[AxisId],
        inner: AxisId,
        vector: u32,
    ) -> Result<(), EmitError> {
        let (ix, sx) = (self.index, self.suffix);
        if self
            .kernel
            .axes
            .iter()
            .any(|axis| axis.kind == AxisKind::Reduction)
        {
            return Err(EmitError::Unsupported {
                reason: "a vector element schedule for a kernel with reduction axes",
            });
        }

        let extent = self.size(self.kernel.axes[inner.0].extent);
        let groups = format!("(({extent} + {}{sx}) / {vector}{sx})", vector - 1);
        let decomposition = parallel
            .iter()
            .map(|&axis| {
                if axis == inner {
                    (format!("a{}_group", axis.0), groups.clone())
                } else {
                    (
                        format!("a{}", axis.0),
                        self.size(self.kernel.axes[axis.0].extent),
                    )
                }
            })
            .collect::<Vec<_>>();
        let total = decomposition
            .iter()
            .map(|(_, extent)| extent.as_str())
            .collect::<Vec<_>>()
            .join(" * ");
        self.line(&format!(
            "const {ix} linear = static_cast<{ix}>(blockIdx.x) * blockDim.x + threadIdx.x;"
        ));
        self.line(&format!("if (linear >= {total}) return;"));
        self.decompose_named("linear", &decomposition);
        let base = format!("a{}_base", inner.0);
        self.line(&format!(
            "const {ix} {base} = a{}_group * {vector}{sx};",
            inner.0
        ));
        let lane_nodes = (0..self.kernel.nodes.len())
            .filter(|&node| self.domains[node].contains(&inner))
            .map(NodeId)
            .collect::<Vec<_>>();
        let staged_loads = lane_nodes
            .iter()
            .filter(|node| match &self.kernel.nodes[node.0].expr {
                Expr::Load(access) => self.walks_contiguously(access, inner),
                _ => false,
            })
            .copied()
            .collect::<Vec<_>>();
        let staged_stores = (0..self.kernel.stores.len())
            .filter(|&store| {
                let store = &self.kernel.stores[store];
                store.combiner.is_none() && self.walks_contiguously(&store.access, inner)
            })
            .collect::<Vec<_>>();
        if !staged_loads.is_empty() || !staged_stores.is_empty() {
            self.line(&format!(
                "const bool full = {base} + {vector}{sx} <= {extent};"
            ));
        }

        for node in 0..self.kernel.nodes.len() {
            if !lane_nodes.contains(&NodeId(node)) {
                let expression = self.expression(NodeId(node));
                self.define(NodeId(node), &expression);
            }
        }

        for &node in &staged_loads {
            let Expr::Load(access) = &self.kernel.nodes[node.0].expr else {
                unreachable!("only loads are staged");
            };
            let element = self.kernel.operands[access.operand.0].element;
            let storage = storage_type(element);
            let pointer = format!(
                "(o{} + {})",
                access.operand.0,
                self.offset_at(access, &[(inner, &base)])
            );
            let staging = format!("raw{}", node.0);
            self.line(&format!("alignas(16) {storage} {staging}[{vector}];"));
            self.staged_copy(
                element,
                vector,
                &format!("{staging}[lane] = {base} + lane < {extent} ? {pointer}[lane] : static_cast<{storage}>(0);"),
                &format!("*reinterpret_cast<WIDE*>({staging}) = *reinterpret_cast<const WIDE*>({pointer});"),
                &pointer,
            );
        }

        for (index, store) in self.kernel.stores.iter().enumerate() {
            if staged_stores.contains(&index) {
                let storage = storage_type(self.kernel.operands[store.access.operand.0].element);
                self.line(&format!("alignas(16) {storage} staged{index}[{vector}];"));
            } else if !self.kernel.access_domain(&store.access).contains(&inner) {
                let write = self.write(store);
                self.line(&write);
            }
        }

        self.line("#pragma unroll");
        self.line(&format!(
            "for (unsigned int lane = 0; lane < {vector}U; ++lane) {{"
        ));
        // The lane's index is declared only when the loop body reads it, which it does not when every access is staged.
        let index = format!("a{}", inner.0);
        let body = self.source.body.len();
        for &node in &lane_nodes {
            let expression = match &self.kernel.nodes[node.0].expr {
                Expr::Load(access) if staged_loads.contains(&node) => load_expression(
                    self.kernel.operands[access.operand.0].element,
                    &format!("raw{}[lane]", node.0),
                ),
                Expr::Load(_) => format!(
                    "(a{} < {extent} ? {} : static_cast<{}>(0))",
                    inner.0,
                    self.expression(node),
                    compute_type(self.kernel.nodes[node.0].element)
                ),
                // Lanes past the end compute on zeros, which must not fail a check.
                Expr::Require {
                    value,
                    condition,
                    status,
                } => format!(
                    "(singe_require(status, v{} || a{} >= {extent}, {status}U), v{})",
                    condition.0, inner.0, value.0
                ),
                _ => self.expression(node),
            };
            self.define(node, &expression);
        }

        for (index, store) in self.kernel.stores.iter().enumerate() {
            if staged_stores.contains(&index) {
                let element = self.kernel.operands[store.access.operand.0].element;
                self.line(&format!(
                    "staged{index}[lane] = {};",
                    store_expression(element, &format!("v{}", store.value.0))
                ));
            } else if self.kernel.access_domain(&store.access).contains(&inner) {
                let write = self.write(store);
                self.line(&format!("if (a{} < {extent}) {write}", inner.0));
            }
        }

        if mentions(&self.source.body[body..], &index) {
            self.source
                .body
                .insert_str(body, &format!("const {ix} {index} = {base} + lane;\n"));
        }

        self.line("}");
        for &index in &staged_stores {
            let access = &self.kernel.stores[index].access;
            let element = self.kernel.operands[access.operand.0].element;
            let pointer = format!(
                "(o{} + {})",
                access.operand.0,
                self.offset_at(access, &[(inner, &base)])
            );
            self.staged_copy(
                element,
                vector,
                &format!("if ({base} + lane < {extent}) {pointer}[lane] = staged{index}[lane];"),
                &format!("*reinterpret_cast<WIDE*>({pointer}) = *reinterpret_cast<const WIDE*>(staged{index});"),
                &pointer,
            );
        }
        Ok(())
    }

    /// Emits a copy between a staging array and `vector` consecutive elements at `pointer`: `wide`, with `WIDE` naming the access type, when every lane is in range and `pointer` is aligned, and `per_lane` for each lane otherwise.
    pub fn staged_copy(
        &mut self,
        element: ElementType,
        vector: u32,
        per_lane: &str,
        wide: &str,
        pointer: &str,
    ) {
        let bytes = element.size_bytes() * vector as usize;
        let wide_type = match bytes {
            2 => Some("unsigned short"),
            4 => Some("unsigned int"),
            8 => Some("uint2"),
            16 => Some("uint4"),
            _ => None,
        };
        if let Some(wide_type) = wide_type {
            self.line(&format!(
                "if (full && reinterpret_cast<unsigned long long>({pointer}) % {bytes}ULL == 0) {{"
            ));
            self.line(&wide.replace("WIDE", wide_type));
            self.line("} else {");
        }

        self.line("#pragma unroll");
        self.line(&format!(
            "for (unsigned int lane = 0; lane < {vector}U; ++lane) {per_lane}"
        ));
        if wide_type.is_some() {
            self.line("}");
        }
    }

    /// Returns whether `access` walks `axis` exactly once, with a stride of one, and its indexed terms do not depend on it.
    pub fn walks_contiguously(&self, access: &Access, axis: AxisId) -> bool {
        let mut strides = access.strides.iter().filter(|(walked, _)| *walked == axis);
        matches!(
            (strides.next(), strides.next()),
            (Some(&(_, stride)), None) if self.kernel.sizes[stride.0] == Size::Static(1)
        ) && access
            .indirect
            .iter()
            .all(|&(node, _)| !self.domains[node.0].contains(&axis))
    }

    /// Emits a loop over `axes` whose iterations each cover `vector` points of the innermost axes, as [`VectorSpan`] lays them out, and returns `false`, emitting nothing, when an access the loop makes walks them other than contiguously.
    ///
    /// Contiguous loads and the store go through staging arrays, moved with one wide access when every lane is in range and the address is aligned; each lane in range then computes its values and does `action`.
    pub fn vector_loop(
        &mut self,
        axes: &[AxisId],
        roots: &[NodeId],
        vector: u32,
        action: &LoopAction<'_>,
    ) -> Result<bool, EmitError> {
        let (ix, sx) = (self.index, self.suffix);
        let Some(span) = self.vector_span(axes, roots, action, vector)? else {
            return Ok(false);
        };
        let grouped = span.grouped;
        let points = span.grouped_points(vector);
        let needed = self.needed(roots, axes)?;
        let extent = self.size(self.kernel.axes[grouped.0].extent);
        let groups = format!("(({extent} + {}{sx}) / {points}{sx})", points - 1);
        let decomposition = axes
            .iter()
            .filter(|&&axis| span.within.is_none_or(|(within, _)| within != axis))
            .map(|&axis| {
                if axis == grouped {
                    (format!("a{}_group", axis.0), groups.clone())
                } else {
                    (
                        format!("a{}", axis.0),
                        self.size(self.kernel.axes[axis.0].extent),
                    )
                }
            })
            .collect::<Vec<_>>();
        let total = decomposition
            .iter()
            .map(|(_, extent)| extent.as_str())
            .collect::<Vec<_>>()
            .join(" * ");
        let (start, step) = match self.schedule {
            ThreadSchedule::Mma { .. } => unreachable!("mma schedules emit their own body"),
            ThreadSchedule::Rows { .. } => ("threadIdx.x", "blockDim.x".to_owned()),
            ThreadSchedule::Warps { lanes, .. } => ("warp_lane", format!("{lanes}U")),
            ThreadSchedule::Columns { splits, .. } => ("split", format!("{splits}U")),
            ThreadSchedule::Elements { .. } => {
                unreachable!("element schedules walk reductions point by point")
            }
        };
        let cache = self.cache.as_ref().map(|cache| (cache.count, cache.step));
        if let Some((count, step)) = cache {
            self.line("#pragma unroll");
            self.line(&format!(
                "for (unsigned int point = 0; point < {count}U; ++point) {{"
            ));
            self.line(&format!("const {ix} flat = {start} + point * {step}{sx};"));
            self.line(&format!("if (flat < {total}) {{"));
        } else {
            self.line(&format!(
                "for ({ix} flat = {start}; flat < {total}; flat += {step}) {{"
            ));
        }

        self.decompose_named("flat", &decomposition);
        let base = format!("a{}_base", grouped.0);
        self.line(&format!(
            "const {ix} {base} = a{}_group * {points}{sx};",
            grouped.0
        ));
        let (lane_nodes, outer_nodes): (Vec<_>, Vec<_>) = needed.into_iter().partition(|node| {
            self.domains[node.0].contains(&grouped)
                || span
                    .within
                    .is_some_and(|(within, _)| self.domains[node.0].contains(&within))
        });
        let (cached, filled) = self
            .cache
            .as_ref()
            .map_or((Vec::new(), Vec::new()), |cache| {
                (cache.values.clone(), cache.filled.clone())
            });
        // Loads an earlier loop kept in registers need no staging.
        let staged_loads = lane_nodes
            .iter()
            .filter(|node| {
                matches!(self.kernel.nodes[node.0].expr, Expr::Load(_)) && !filled.contains(node)
            })
            .copied()
            .collect::<Vec<_>>();
        let staged_store = match action {
            LoopAction::Store(store) => Some(*store),
            LoopAction::Combine(_) => None,
        };
        if !staged_loads.is_empty() || staged_store.is_some() {
            self.line(&format!(
                "const bool full = {base} + {points}{sx} <= {extent};"
            ));
        }

        for node in outer_nodes {
            self.emit_value(node)?;
        }

        let mut shared = Vec::new();
        for &node in &staged_loads {
            let Expr::Load(access) = &self.kernel.nodes[node.0].expr else {
                unreachable!("only loads are staged");
            };
            if self.stage_load(node, access, span, &base, &extent, vector) {
                shared.push(node);
            }
        }

        if let Some(store) = staged_store {
            let storage = storage_type(self.kernel.operands[store.access.operand.0].element);
            self.line(&format!("alignas(16) {storage} staged[{vector}];"));
        }

        self.line("#pragma unroll");
        self.line(&format!(
            "for (unsigned int lane = 0; lane < {vector}U; ++lane) {{"
        ));
        let within = span
            .within
            .map(|(within, extent)| (within, format!("{extent}{sx}")));
        match &within {
            Some((within, inner)) => {
                self.line(&format!(
                    "const {ix} a{} = {base} + lane / {inner};",
                    grouped.0
                ));
                self.line(&format!("const {ix} a{} = lane % {inner};", within.0));
            }
            None => self.line(&format!("const {ix} a{} = {base} + lane;", grouped.0)),
        }

        self.line(&format!("if (a{} < {extent}) {{", grouped.0));
        let mut newly_filled = Vec::new();
        for &node in &lane_nodes {
            let expression = match &self.kernel.nodes[node.0].expr {
                _ if filled.contains(&node) => {
                    format!("cached{}[point * {vector}U + lane]", node.0)
                }
                expr if reduced_axes(expr).is_some() => {
                    self.nested_reduction(node)?;
                    if cached.contains(&node) {
                        self.line(&format!(
                            "cached{0}[point * {vector}U + lane] = v{0};",
                            node.0
                        ));
                        newly_filled.push(node);
                    }

                    continue;
                }
                Expr::Load(access) => {
                    let lane = match &within {
                        Some((_, inner)) if shared.contains(&node) => format!("lane / {inner}"),
                        _ => "lane".to_owned(),
                    };
                    load_expression(
                        self.kernel.operands[access.operand.0].element,
                        &format!("raw{}[{lane}]", node.0),
                    )
                }
                _ => self.expression(node),
            };
            self.define(node, &expression);
            if cached.contains(&node) && !filled.contains(&node) {
                self.line(&format!(
                    "cached{0}[point * {vector}U + lane] = v{0};",
                    node.0
                ));
                newly_filled.push(node);
            }
        }

        let statement = self.action_statement(action, staged_store.map(|_| "staged"));
        self.line(&statement);
        self.line("}");
        self.line("}");
        if let Some(store) = staged_store {
            let element = self.kernel.operands[store.access.operand.0].element;
            let pointer = format!(
                "(o{} + {})",
                store.access.operand.0,
                self.offset_at(&store.access, &[(grouped, &base)])
            );
            self.staged_copy(
                element,
                vector,
                &format!("if ({base} + lane < {extent}) {pointer}[lane] = staged[lane];"),
                &format!(
                    "*reinterpret_cast<WIDE*>({pointer}) = *reinterpret_cast<const WIDE*>(staged);"
                ),
                &pointer,
            );
        }

        if cache.is_some() {
            self.line("}");
        }

        self.line("}");
        if let Some(cache) = &mut self.cache {
            cache.filled.extend(newly_filled);
        }
        Ok(true)
    }

    /// Returns how a loop over `axes` that does `action` may cover `vector` points at once, or `None` when its accesses do not allow it.
    ///
    /// A static innermost axis shorter than the vector is covered whole at each of several points of the axis before it, when every load the loop makes either walks both contiguously, as one array, or reads one element per point of the outer axis, as a packed byte serves its nibbles.
    /// Otherwise each group covers `vector` consecutive points of the innermost axis, which every load walks contiguously or not at all, and the store, if any, walks contiguously and replaces elements.
    pub fn vector_span(
        &self,
        axes: &[AxisId],
        roots: &[NodeId],
        action: &LoopAction<'_>,
        vector: u32,
    ) -> Result<Option<VectorSpan>, EmitError> {
        let Some(&inner) = axes.last() else {
            return Ok(None);
        };
        if let [.., grouped, _] = axes
            && let LoopAction::Combine(_) = action
            && let Size::Static(extent) = self.kernel.sizes[self.kernel.axes[inner.0].extent.0]
            && let Ok(extent) = u32::try_from(extent)
            && extent > 1
            && extent < vector
            && vector.is_multiple_of(extent)
        {
            let span = VectorSpan {
                grouped: *grouped,
                within: Some((inner, extent)),
            };
            let mut legal = true;
            for node in self.needed(roots, axes)? {
                if let Expr::Load(access) = &self.kernel.nodes[node.0].expr {
                    legal &= self.span_load(access, span).is_some();
                }
            }
            if legal {
                return Ok(Some(span));
            }
        }

        let span = VectorSpan {
            grouped: inner,
            within: None,
        };
        for node in self.needed(roots, axes)? {
            if let Expr::Load(access) = &self.kernel.nodes[node.0].expr
                && self.span_load(access, span).is_none()
            {
                return Ok(None);
            }
        }
        Ok(match action {
            LoopAction::Store(store) => (store.combiner.is_none()
                && self.walks_contiguously(&store.access, inner))
            .then_some(span),
            LoopAction::Combine(_) => Some(span),
        })
    }

    /// Returns how `access` reads the points of `span`: `Some(false)` for none or one contiguous element per lane, `Some(true)` for one element per point of the grouped axis that every point of the innermost axis shares, and `None` otherwise.
    pub fn span_load(&self, access: &Access, span: VectorSpan) -> Option<bool> {
        let grouped = self.static_stride(access, span.grouped)?;
        let Some((within, extent)) = span.within else {
            return matches!(grouped, 0 | 1).then_some(false);
        };
        match (grouped, self.static_stride(access, within)?) {
            (0, 0) => Some(false),
            (grouped, 1) if grouped == u64::from(extent) => Some(false),
            (1, 0) => Some(true),
            _ => None,
        }
    }

    /// Returns the static stride of `access` along `axis`, zero when it does not walk it, or `None` when it walks it by an argument stride, more than once, or through an indirect index.
    pub fn static_stride(&self, access: &Access, axis: AxisId) -> Option<u64> {
        if access
            .indirect
            .iter()
            .any(|&(node, _)| self.domains[node.0].contains(&axis))
        {
            return None;
        }

        let mut strides = access.strides.iter().filter(|(walked, _)| *walked == axis);
        match (strides.next(), strides.next()) {
            (None, _) => Some(0),
            (Some(&(_, stride)), None) => match self.kernel.sizes[stride.0] {
                Size::Static(stride) => Some(stride),
                Size::Argument => None,
            },
            _ => None,
        }
    }

    /// Declares the staging array `raw{node}` and fills it with the elements `access` reads for a group of `vector` lanes of `span` from `base` along its grouped axis, zero past `extent`, and returns whether the lanes of each grouped point share one element.
    pub fn stage_load(
        &mut self,
        node: NodeId,
        access: &Access,
        span: VectorSpan,
        base: &str,
        extent: &str,
        vector: u32,
    ) -> bool {
        let sx = self.suffix;
        let element = self.kernel.operands[access.operand.0].element;
        let storage = storage_type(element);
        let mut renamed = vec![(span.grouped, base.to_owned())];
        if let Some((within, _)) = span.within {
            renamed.push((within, format!("0{sx}")));
        }

        let renamed = renamed
            .iter()
            .map(|(axis, name)| (*axis, name.as_str()))
            .collect::<Vec<_>>();
        let pointer = format!(
            "(o{} + {})",
            access.operand.0,
            self.offset_at(access, &renamed)
        );
        let shared = self.span_load(access, span) == Some(true);
        let (count, point) = match span.within {
            Some((_, inner)) if !shared => (vector, format!("lane / {inner}{sx}")),
            Some((_, inner)) => (vector / inner, "lane".to_owned()),
            None => (vector, "lane".to_owned()),
        };
        let staging = format!("raw{}", node.0);
        self.line(&format!("alignas(16) {storage} {staging}[{count}];"));
        self.staged_copy(
            element,
            count,
            &format!("{staging}[lane] = {base} + {point} < {extent} ? {pointer}[lane] : static_cast<{storage}>(0);"),
            &format!("*reinterpret_cast<WIDE*>({staging}) = *reinterpret_cast<const WIDE*>({pointer});"),
            &pointer,
        );
        shared
    }
}
