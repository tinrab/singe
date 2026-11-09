//! CUDA Tile IR for `tileiras`, built in memory with `cutile-ir`.
//!
//! The kernel is one entry function named after the IR kernel, whose parameters are [`abi::parameters`](crate::abi::parameters) in order: a pointer per operand, then each scalar and argument size as a zero-dimensional tile.
//! Every value is a one-dimensional tile whose elements are the points one tile block covers.
//! Under [`TileSchedule::Tiles`](crate::schedule::TileSchedule::Tiles) those are consecutive points of the flattened parallel axes; under [`TileSchedule::TileRows`](crate::schedule::TileSchedule::TileRows) they are one row: a point of the parallel axes and the flattened reduction axes, whose reductions fold the whole tile with `reduce`.
//! Accesses become tiles of pointers that loads and stores address under a mask of the points in range, so any strided or indexed access is one gather or scatter.
//!
//! Index checks and requirements report their status code with an atomic compare-and-swap on the status word, skipped when its address is null, and adding stores are atomic adds.
//! Functions Tile IR has no operation for are rejected; their targets keep the CUDA C++ emitter.

mod matmul;

use cutile_ir::{
    builder::{OpBuilder, append_op, build_single_block_region},
    bytecode::Opcode,
    ir::{
        Attribute, BlockId, DenseElements, FuncType, Location, Module, PointerType, RegionId,
        ScalarType, TileElementType, TileType, Type, Value,
    },
};

use crate::{
    abi::{Parameter, parameters},
    emit::EmitError,
    ir::{
        Access, AxisId, AxisKind, BinaryOp, Combiner, ElementType, Expr, Kernel, Literal, NodeId,
        Size, SizeId, Store, UnaryOp,
    },
    schedule::{Schedule, TileSchedule, contraction::Contraction},
};

/// Returns a Tile IR module holding `kernel` as one entry function under `schedule`.
///
/// # Errors
///
/// Returns an error if the kernel or schedule is invalid, or the kernel uses something this emitter does not support.
pub fn emit(kernel: &Kernel, schedule: TileSchedule) -> Result<Module, EmitError> {
    kernel.verify()?;
    Schedule::from(schedule).validate()?;
    // Tile IR converts to its 8-bit floating types with its own overflow rules, so 8-bit floats stay with the CUDA C emitter, whose conversions follow PyTorch's.
    if kernel
        .operands
        .iter()
        .any(|operand| matches!(operand.element, ElementType::F8E4M3 | ElementType::F8E5M2))
    {
        return Err(EmitError::Unsupported {
            reason: "8-bit floating operands",
        });
    }

    // Shifts and integer powers follow PyTorch's rules for out-of-range shifts and negative exponents, which the CUDA C prelude defines.
    if kernel.nodes.iter().any(|node| match node.expr {
        Expr::Binary(BinaryOp::ShiftLeft | BinaryOp::ShiftRight, ..)
        | Expr::Unary(UnaryOp::BitNot, _) => true,
        Expr::Binary(BinaryOp::Pow, lhs, _) => kernel.nodes[lhs.0].element.is_integer(),
        _ => false,
    }) {
        return Err(EmitError::Unsupported {
            reason: "shifts, complements, and integer powers",
        });
    }

    let lanes = match schedule {
        TileSchedule::Tiles { size } => size,
        TileSchedule::TileRows { width } => width,
        TileSchedule::TileMatmul { m, n, k } => {
            let contraction = Contraction::of(kernel).ok_or(EmitError::Unsupported {
                reason: "a tensor-core schedule of a kernel that is not a plain contraction",
            })?;
            let (mut writer, region, parameter_types) = start(kernel, 1);
            writer.matmul(&contraction, [m, n, k].map(i64::from))?;
            return Ok(writer.finish(region, parameter_types));
        }
    };
    let has_reduction = kernel
        .axes
        .iter()
        .any(|axis| axis.kind == AxisKind::Reduction);
    if has_reduction != matches!(schedule, TileSchedule::TileRows { .. }) {
        return Err(EmitError::Unsupported {
            reason: "a tile schedule that does not match the kernel's reduction axes",
        });
    }

    let (mut writer, region, parameter_types) = start(kernel, i64::from(lanes));
    match schedule {
        TileSchedule::Tiles { .. } => writer.element_indices(),
        _ => writer.row_indices(),
    }

    for node in 0..kernel.nodes.len() {
        let value = writer.node(NodeId(node))?;
        writer.values[node] = Some(value);
    }

    for store in &kernel.stores {
        writer.combined_store(store)?;
    }
    Ok(writer.finish(region, parameter_types))
}

/// Returns a writer of `kernel`'s entry function whose tiles hold `lanes` points, with the function's body region and parameter types.
fn start(kernel: &Kernel, lanes: i64) -> (Writer<'_>, RegionId, Vec<Type>) {
    let mut module = Module::new(kernel.name.clone());
    let parameter_types = parameters(kernel)
        .into_iter()
        .map(|parameter| match parameter {
            Parameter::Operand(operand) => {
                pointer_type(storage_scalar(kernel.operands[operand.0].element), &[])
            }
            Parameter::Scalar(_, element) => tile_type(compute_scalar(element), &[]),
            Parameter::Size(_) => tile_type(ScalarType::I64, &[]),
            Parameter::Status => pointer_type(ScalarType::I32, &[]),
        })
        .collect::<Vec<_>>();
    let (region, block, arguments) = build_single_block_region(&mut module, &parameter_types);
    let operands = kernel.operands.len();
    let status = usize::from(kernel.reports_status());
    let scalars = operands + status;
    let mut writer = Writer {
        kernel,
        module,
        block,
        lanes,
        operands: arguments[..operands].to_vec(),
        status: (status == 1).then(|| arguments[operands]),
        scalars: arguments[scalars..scalars + kernel.scalars.len()].to_vec(),
        sizes: vec![None; kernel.sizes.len()],
        axes: vec![None; kernel.axes.len()],
        values: vec![None; kernel.nodes.len()],
        in_range: None,
        first_lane: None,
    };
    for (size, &argument) in kernel
        .argument_sizes()
        .zip(&arguments[scalars + kernel.scalars.len()..])
    {
        writer.sizes[size.0] = Some(argument);
    }
    (writer, region, parameter_types)
}

struct Writer<'a> {
    kernel: &'a Kernel,
    module: Module,
    block: BlockId,
    /// The points one tile block covers, the length of every tile.
    lanes: i64,
    /// The pointer parameter of each operand.
    operands: Vec<Value>,
    /// The status word's pointer parameter, for a kernel that reports status codes.
    status: Option<Value>,
    /// The parameter of each scalar.
    scalars: Vec<Value>,
    /// The zero-dimensional `I64` value of each size, once made.
    sizes: Vec<Option<Value>>,
    /// The `I64` index tile of each axis.
    axes: Vec<Option<Value>>,
    /// The tile of each node, once computed.
    values: Vec<Option<Value>>,
    /// Which lanes cover points in range: the parallel points under tiles, the reduction points under tile rows.
    in_range: Option<Value>,
    /// Under tile rows, the lane that stores values of the row itself.
    first_lane: Option<Value>,
}

impl Writer<'_> {
    /// Ends the entry function with a return and returns the module that holds it.
    fn finish(mut self, region: RegionId, parameter_types: Vec<Type>) -> Module {
        self.emit(OpBuilder::new(Opcode::Return, Location::Unknown));
        let function_type = FuncType {
            inputs: parameter_types,
            results: Vec::new(),
        };
        let mut module = self.module;
        let (entry, _) = OpBuilder::new(Opcode::Entry, Location::Unknown)
            .attr("sym_name", Attribute::String(self.kernel.name.clone()))
            .attr("function_type", Attribute::Type(Type::Func(function_type)))
            .region(region)
            .build(&mut module);
        module.functions.push(entry);
        module
    }

    /// Defines the axis tiles of a [`TileSchedule::Tiles`](crate::schedule::TileSchedule::Tiles) block: lane `i` covers flat parallel point `block × lanes + i`.
    fn element_indices(&mut self) {
        let block = self.block_id();
        let lanes = self.constant_i64(self.lanes);
        let base = self.integer(Opcode::MulI, block, lanes, &[]);
        let base = self.splat(base, ScalarType::I64);
        let iota = self.iota();
        let flat = self.integer(Opcode::AddI, base, iota, &[]);
        let parallel = self.axes_of_kind(AxisKind::Parallel);
        let total = self.product(&parallel);
        let total = self.splat(total, ScalarType::I64);
        self.in_range = Some(self.compare_unsigned(2, flat, total));
        for (axis, index) in self.decompose(flat, &parallel, true) {
            self.axes[axis.0] = Some(index);
        }
    }

    /// Defines the axis tiles of a [`TileSchedule::TileRows`](crate::schedule::TileSchedule::TileRows) block: the block is one parallel point, and lane `i` covers flat reduction point `i`.
    fn row_indices(&mut self) {
        let block = self.block_id();
        let parallel = self.axes_of_kind(AxisKind::Parallel);
        for (axis, index) in self.decompose(block, &parallel, false) {
            let index = self.splat(index, ScalarType::I64);
            self.axes[axis.0] = Some(index);
        }

        let iota = self.iota();
        let reduction = self.axes_of_kind(AxisKind::Reduction);
        let total = self.product(&reduction);
        let total = self.splat(total, ScalarType::I64);
        self.in_range = Some(self.compare_unsigned(2, iota, total));
        let zero = self.constant_i64(0);
        let zero = self.splat(zero, ScalarType::I64);
        self.first_lane = Some(self.compare_unsigned(0, iota, zero));
        for (axis, index) in self.decompose(iota, &reduction, true) {
            self.axes[axis.0] = Some(index);
        }
    }

    /// Returns the index of each of `axes` within `flat`, the last axis varying fastest; `tiled` says whether `flat` is a tile or a zero-dimensional value.
    fn decompose(&mut self, flat: Value, axes: &[AxisId], tiled: bool) -> Vec<(AxisId, Value)> {
        let mut indices = Vec::new();
        let mut divisor: Option<Value> = None;
        for (position, &axis) in axes.iter().enumerate().rev() {
            let extent = self.size_value(self.kernel.axes[axis.0].extent);
            let extent = if tiled {
                self.splat(extent, ScalarType::I64)
            } else {
                extent
            };
            let quotient = divisor.map_or(flat, |divisor| {
                self.integer(Opcode::DivI, flat, divisor, &[UNSIGNED, ZERO])
            });
            // The outermost index needs no remainder, since the flat index is in range.
            let index = if position == 0 {
                quotient
            } else {
                self.integer(Opcode::RemI, quotient, extent, &[UNSIGNED])
            };
            indices.push((axis, index));
            divisor = Some(divisor.map_or(extent, |divisor| {
                self.integer(Opcode::MulI, divisor, extent, &[])
            }));
        }
        indices
    }

    /// Returns the tile `node` computes.
    fn node(&mut self, node: NodeId) -> Result<Value, EmitError> {
        let element = self.kernel.nodes[node.0].element;
        let scalar = compute_scalar(element);
        let value = |writer: &Self, node: &NodeId| {
            writer.values[node.0].expect("nodes follow their operands")
        };
        Ok(match &self.kernel.nodes[node.0].expr {
            Expr::Load(access) => self.load(access),
            Expr::Literal(literal) => {
                let constant = self.constant(element, *literal);
                self.splat(constant, scalar)
            }
            Expr::Scalar(scalar_id) => {
                let parameter = self.scalars[scalar_id.0];
                self.splat(parameter, scalar)
            }
            Expr::AxisIndex(axis) => {
                let index = self.axes[axis.0].expect("every axis has an index");
                self.convert(index, ElementType::I64, element)
            }
            Expr::Unary(op, operand) => {
                let operand_element = self.kernel.nodes[operand.0].element;
                let operand = value(self, operand);
                self.unary(*op, operand_element, operand)?
            }
            Expr::Binary(op, lhs, rhs) => {
                let operand_element = self.kernel.nodes[lhs.0].element;
                let (lhs, rhs) = (value(self, lhs), value(self, rhs));
                self.binary(*op, operand_element, lhs, rhs)
            }
            Expr::MulAdd(a, b, c) => {
                let (a, b, c) = (value(self, a), value(self, b), value(self, c));
                let ty = self.tile(scalar);
                self.emit_one(
                    OpBuilder::new(Opcode::Fma, Location::Unknown)
                        .operand(a)
                        .operand(b)
                        .operand(c)
                        .attr("rounding_mode", Attribute::i32(NEAREST_EVEN))
                        .result(ty),
                )
            }
            Expr::Select {
                condition,
                if_true,
                if_false,
            } => {
                let (condition, if_true, if_false) = (
                    value(self, condition),
                    value(self, if_true),
                    value(self, if_false),
                );
                self.select(condition, if_true, if_false, scalar)
            }
            Expr::Cast(operand) => {
                let from = self.kernel.nodes[operand.0].element;
                self.convert(value(self, operand), from, element)
            }
            Expr::Reduce {
                combiner,
                axes,
                value: operand,
            } => {
                if *combiner == Combiner::LogSumExp {
                    return Err(EmitError::Unsupported {
                        reason: "a log-sum-exp reduction",
                    });
                }

                if axes.len() != self.axes_of_kind(AxisKind::Reduction).len() {
                    return Err(EmitError::Unsupported {
                        reason: "a reduction over some of a tile row's reduction axes",
                    });
                }

                let operand = value(self, operand);
                self.reduce(*combiner, element, operand)
            }
            Expr::SoftmaxSum { .. } => {
                return Err(EmitError::Unsupported {
                    reason: "a softmax-weighted sum",
                });
            }
            Expr::ArgReduce { .. } => {
                return Err(EmitError::Unsupported {
                    reason: "an argument reduction",
                });
            }
            Expr::Scan { .. } => {
                return Err(EmitError::Unsupported { reason: "a scan" });
            }
            Expr::CheckIndex {
                value: operand,
                extent,
                wrap,
                status,
            } => {
                let from = self.kernel.nodes[operand.0].element;
                let index = self.convert(value(self, operand), from, ElementType::I64);
                self.check_index(index, *extent, *wrap, *status)
            }
            Expr::Require {
                value: operand,
                condition,
                status,
            } => {
                let condition = value(self, condition);
                let truth = self.constant(ElementType::Bool, Literal::Bool(true));
                let truth = self.splat(truth, ScalarType::I1);
                let failing = self.integer(Opcode::XOrI, condition, truth, &[]);
                self.report(failing, *status);
                value(self, operand)
            }
        })
    }

    /// Returns the tile of elements `access` addresses, in its operand's compute type; lanes out of range read nothing.
    fn load(&mut self, access: &Access) -> Value {
        let element = self.kernel.operands[access.operand.0].element;
        let storage = storage_scalar(element);
        let pointers = self.pointers(access, storage);
        // A row's own elements are in range in every lane, but its reduction points only in some.
        let mask = self.access_mask(access);
        let ty = self.tile(storage);
        let mut builder = OpBuilder::new(Opcode::LoadPtrTko, Location::Unknown)
            .result(ty)
            .result(Type::Token)
            .operand(pointers)
            .attr("memory_ordering_semantics", Attribute::i32(0));
        let mut segments = vec![1, 0, 0, 0];
        if let Some(mask) = mask {
            builder = builder.operand(mask);
            segments[1] = 1;
        }

        let builder = builder.attr("operandSegmentSizes", segment_sizes(&segments));
        let raw = self.emit(builder)[0];
        self.widened(raw, element)
    }

    /// Returns `index`, a signed `I64` tile, as an index into `0..extent`: negative values count from the end when `wrap` holds, and values still out of range report `status` and become zero.
    fn check_index(&mut self, index: Value, extent: SizeId, wrap: bool, status: u32) -> Value {
        let extent = self.size_value(extent);
        let extent = self.splat(extent, ScalarType::I64);
        let zero = self.constant_i64(0);
        let zero = self.splat(zero, ScalarType::I64);
        let signed = |writer: &mut Self, predicate, x, y| {
            let ty = writer.tile(ScalarType::I1);
            writer.emit_one(
                OpBuilder::new(Opcode::CmpI, Location::Unknown)
                    .operand(x)
                    .operand(y)
                    .attr("comparison_predicate", Attribute::i32(predicate))
                    .attr("signedness", Attribute::i32(SIGNED))
                    .result(ty),
            )
        };
        let index = if wrap {
            let negative = signed(self, 2, index, zero);
            let wrapped = self.integer(Opcode::AddI, index, extent, &[]);
            self.select(negative, wrapped, index, ScalarType::I64)
        } else {
            index
        };
        let below = signed(self, 2, index, zero);
        let above = signed(self, 5, index, extent);
        let outside = self.integer(Opcode::OrI, below, above, &[]);
        self.report(outside, status);
        self.select(outside, zero, index, ScalarType::I64)
    }

    /// Writes `status` to the status word, if it is still zero and its address is not null, from each lane where `failing` holds that covers a point.
    fn report(&mut self, failing: Value, status: u32) {
        let pointer = self.status.expect("status kernels take a status word");
        let address = self.emit_one(
            OpBuilder::new(Opcode::PtrToInt, Location::Unknown)
                .operand(pointer)
                .result(tile_type(ScalarType::I64, &[])),
        );
        let address = self.splat(address, ScalarType::I64);
        let null = self.constant_i64(0);
        let null = self.splat(null, ScalarType::I64);
        let present = self.compare_unsigned(1, address, null);
        // Under tile rows a value of the row itself is checked in every lane, and counts even when the row is empty.
        let lanes = match (self.in_range, self.first_lane) {
            (Some(in_range), Some(first)) => self.integer(Opcode::OrI, in_range, first, &[]),
            (Some(in_range), None) => in_range,
            _ => unreachable!("every schedule masks its lanes"),
        };
        let mask = self.integer(Opcode::AndI, failing, lanes, &[]);
        let mask = self.integer(Opcode::AndI, mask, present, &[]);
        let pointers = self.splat_pointer(pointer, ScalarType::I32);
        let zero = self.constant(ElementType::I32, Literal::Int(0));
        let zero = self.splat(zero, ScalarType::I32);
        let code = self.constant(ElementType::I32, Literal::Int(i64::from(status)));
        let code = self.splat(code, ScalarType::I32);
        let ty = self.tile(ScalarType::I32);
        self.emit(
            OpBuilder::new(Opcode::AtomicCAS, Location::Unknown)
                .result(ty)
                .result(Type::Token)
                .operand(pointers)
                .operand(zero)
                .operand(code)
                .operand(mask)
                .attr("memory_ordering_semantics", Attribute::i32(RELAXED))
                .attr("memory_scope", Attribute::i32(DEVICE))
                .attr("operandSegmentSizes", segment_sizes(&[1, 1, 1, 1, 0])),
        );
    }

    /// Writes `value` to the elements `access` addresses, at the lanes in range, or adds it atomically for a combining store.
    fn combined_store(&mut self, store: &Store) -> Result<(), EmitError> {
        if store.combiner.is_none() {
            return self.store(&store.access, store.value);
        }

        let element = self.kernel.operands[store.access.operand.0].element;
        let storage = storage_scalar(element);
        let pointers = self.pointers(&store.access, storage);
        let value = self.values[store.value.0].expect("stores follow every node");
        let mask =
            self.access_mask(&store.access)
                .or(self.first_lane)
                .ok_or(EmitError::Unsupported {
                    reason: "a store without a lane mask",
                })?;
        let mode = if element.is_float() { ADD_FLOAT } else { ADD };
        let ty = self.tile(storage);
        self.emit(
            OpBuilder::new(Opcode::AtomicRMW, Location::Unknown)
                .result(ty)
                .result(Type::Token)
                .operand(pointers)
                .operand(value)
                .operand(mask)
                .attr("memory_ordering_semantics", Attribute::i32(RELAXED))
                .attr("memory_scope", Attribute::i32(DEVICE))
                .attr("mode", Attribute::i32(mode))
                .attr("operandSegmentSizes", segment_sizes(&[1, 1, 1, 0])),
        );
        Ok(())
    }

    /// Writes `value` to the elements `access` addresses, at the lanes in range.
    fn store(&mut self, access: &Access, value: NodeId) -> Result<(), EmitError> {
        let element = self.kernel.operands[access.operand.0].element;
        let storage = storage_scalar(element);
        let pointers = self.pointers(access, storage);
        let value = self.values[value.0].expect("stores follow every node");
        let value = self.narrowed(value, element);
        // Values of the row itself are written once, by the first lane.
        let mask = self
            .access_mask(access)
            .or(self.first_lane)
            .ok_or(EmitError::Unsupported {
                reason: "a store without a lane mask",
            })?;
        self.emit(
            OpBuilder::new(Opcode::StorePtrTko, Location::Unknown)
                .result(Type::Token)
                .operand(pointers)
                .operand(value)
                .operand(mask)
                .attr("memory_ordering_semantics", Attribute::i32(0))
                .attr("operandSegmentSizes", segment_sizes(&[1, 1, 1, 0])),
        );
        Ok(())
    }

    /// Returns the lanes `access` may address: every lane's point under tiles, and under tile rows the reduction points in range, or `None` for an access of the row itself.
    fn access_mask(&self, access: &Access) -> Option<Value> {
        if self.first_lane.is_none() {
            return self.in_range;
        }

        let walks_row = self
            .kernel
            .access_domain(access)
            .iter()
            .any(|axis| self.kernel.axes[axis.0].kind == AxisKind::Reduction);
        walks_row.then_some(self.in_range).flatten()
    }

    /// Returns the tile of pointers `access` addresses, to elements stored as `storage`.
    fn pointers(&mut self, access: &Access, storage: ScalarType) -> Value {
        let mut offset = None;
        let mut add = |writer: &mut Self, term: Value| {
            offset = Some(offset.map_or(term, |sum| writer.integer(Opcode::AddI, sum, term, &[])));
        };
        if let Some(base) = access.base {
            let base = self.size_value(base);
            let base = self.splat(base, ScalarType::I64);
            add(self, base);
        }

        for &(axis, stride) in &access.strides {
            if self.kernel.sizes[stride.0] == Size::Static(0) {
                continue;
            }

            let index = self.axes[axis.0].expect("every axis has an index");
            let term = self.scaled(index, stride);
            add(self, term);
        }

        for &(node, stride) in &access.indirect {
            let element = self.kernel.nodes[node.0].element;
            let index = self.values[node.0].expect("indices precede their accesses");
            let index = self.convert(index, element, ElementType::I64);
            let term = self.scaled(index, stride);
            add(self, term);
        }

        let offset = offset.unwrap_or_else(|| {
            let zero = self.constant_i64(0);
            self.splat(zero, ScalarType::I64)
        });
        let operand = self.operands[access.operand.0];
        let base = self.splat_pointer(operand, storage);
        let ty = pointer_type(storage, &[self.lanes]);
        self.emit_one(
            OpBuilder::new(Opcode::Offset, Location::Unknown)
                .operand(base)
                .operand(offset)
                .result(ty),
        )
    }

    /// Returns `index × stride`, or `index` itself for a stride of one.
    fn scaled(&mut self, index: Value, stride: SizeId) -> Value {
        if self.kernel.sizes[stride.0] == Size::Static(1) {
            return index;
        }

        let stride = self.size_value(stride);
        let stride = self.splat(stride, ScalarType::I64);
        self.integer(Opcode::MulI, index, stride, &[])
    }

    /// Returns `operand` folded over the whole tile with `combiner`, its lanes out of range counting as the identity, splatted back across the tile.
    fn reduce(&mut self, combiner: Combiner, element: ElementType, operand: Value) -> Value {
        let scalar = compute_scalar(element);
        let identity = identity(combiner, element);
        let identity_value = self.constant(element, identity);
        let identity_tile = self.splat(identity_value, scalar);
        let in_range = self.in_range.expect("tile rows mask their lanes");
        let masked = self.select(in_range, operand, identity_tile, scalar);
        let point = tile_type(scalar, &[]);
        let (region, body, arguments) =
            build_single_block_region(&mut self.module, &[point.clone(), point.clone()]);
        let folded = self.emit_in(
            body,
            combine_op(combiner, element, arguments[0], arguments[1], point.clone()),
        )[0];
        self.emit_in(
            body,
            OpBuilder::new(Opcode::Yield, Location::Unknown).operand(folded),
        );
        let identity_attribute = match identity {
            Literal::Float(value) => Attribute::Float(value, Type::Scalar(scalar)),
            Literal::Int(value) => Attribute::Integer(value, Type::Scalar(scalar)),
            Literal::Bool(value) => Attribute::Integer(i64::from(value), Type::Scalar(scalar)),
        };
        let reduced = self.emit_one(
            OpBuilder::new(Opcode::Reduce, Location::Unknown)
                .operand(masked)
                .attr("dim", Attribute::i32(0))
                .attr("identities", Attribute::Array(vec![identity_attribute]))
                .result(point)
                .region(region),
        );
        self.splat(reduced, scalar)
    }

    fn unary(&mut self, op: UnaryOp, element: ElementType, x: Value) -> Result<Value, EmitError> {
        let scalar = compute_scalar(element);
        let ty = self.tile(scalar);
        if element.is_integer() {
            return Ok(match op {
                UnaryOp::Neg => self.emit_one(
                    OpBuilder::new(Opcode::NegI, Location::Unknown)
                        .operand(x)
                        .attr("overflow", Attribute::i32(0))
                        .result(ty),
                ),
                _ => self.emit_one(
                    OpBuilder::new(Opcode::AbsI, Location::Unknown)
                        .operand(x)
                        .result(ty),
                ),
            });
        }

        if op == UnaryOp::Not {
            let truth = self.constant(ElementType::Bool, Literal::Bool(true));
            let truth = self.splat(truth, ScalarType::I1);
            return Ok(self.integer(Opcode::XOrI, x, truth, &[]));
        }

        let simple = |opcode| OpBuilder::new(opcode, Location::Unknown).operand(x);
        Ok(match op {
            UnaryOp::Neg => self.emit_one(simple(Opcode::NegF).result(ty)),
            UnaryOp::Abs => self.emit_one(simple(Opcode::AbsF).result(ty)),
            UnaryOp::Exp => self.emit_one(simple(Opcode::Exp).result(ty)),
            UnaryOp::Log => self.emit_one(simple(Opcode::Log).result(ty)),
            UnaryOp::Sqrt => self.emit_one(
                simple(Opcode::Sqrt)
                    .attr("rounding_mode", Attribute::i32(NEAREST_EVEN))
                    .result(ty),
            ),
            UnaryOp::Rsqrt => self.emit_one(simple(Opcode::Rsqrt).result(ty)),
            UnaryOp::Reciprocal => {
                let one = self.constant(element, Literal::Float(1.0));
                let one = self.splat(one, scalar);
                self.float(Opcode::DivF, one, x, scalar)
            }
            UnaryOp::Sin => self.emit_one(simple(Opcode::Sin).result(ty)),
            UnaryOp::Cos => self.emit_one(simple(Opcode::Cos).result(ty)),
            UnaryOp::Tan => self.emit_one(simple(Opcode::Tan).result(ty)),
            UnaryOp::Tanh => self.emit_one(simple(Opcode::TanH).result(ty)),
            UnaryOp::Sinh => self.emit_one(simple(Opcode::SinH).result(ty)),
            UnaryOp::Cosh => self.emit_one(simple(Opcode::CosH).result(ty)),
            UnaryOp::Floor => self.emit_one(simple(Opcode::Floor).result(ty)),
            UnaryOp::Ceil => self.emit_one(simple(Opcode::Ceil).result(ty)),
            // Truncation rounds negative values up and others down.
            UnaryOp::Trunc => {
                let floor = self.emit_one(simple(Opcode::Floor).result(ty.clone()));
                let ceil = self.emit_one(simple(Opcode::Ceil).result(ty));
                let zero = self.constant(element, Literal::Float(0.0));
                let zero = self.splat(zero, scalar);
                let negative = self.compare_float(BinaryOp::Less, x, zero);
                self.select(negative, ceil, floor, scalar)
            }
            UnaryOp::IsNan => self.compare_float(BinaryOp::NotEqual, x, x),
            UnaryOp::Round => self.round_half_even(x, element),
            UnaryOp::Not => unreachable!("handled above"),
            UnaryOp::BitNot => unreachable!("`emit` rejects complements"),
            UnaryOp::Asin
            | UnaryOp::Acos
            | UnaryOp::Atan
            | UnaryOp::Asinh
            | UnaryOp::Acosh
            | UnaryOp::Atanh
            | UnaryOp::Erf
            | UnaryOp::Log1p
            | UnaryOp::Expm1 => {
                return Err(EmitError::Unsupported {
                    reason: "a function Tile IR has no operation for",
                });
            }
        })
    }

    /// Returns `x` rounded to the nearest integer, ties to even: the floor, or the next integer above it when the fraction is above a half, or exactly a half above an odd floor.
    ///
    /// Every step is exact, and NaN and infinities stay themselves.
    fn round_half_even(&mut self, x: Value, element: ElementType) -> Value {
        let scalar = compute_scalar(element);
        let ty = self.tile(scalar);
        let floor = self.emit_one(
            OpBuilder::new(Opcode::Floor, Location::Unknown)
                .operand(x)
                .result(ty),
        );
        let constant = |writer: &mut Self, value| {
            let constant = writer.constant(element, Literal::Float(value));
            writer.splat(constant, scalar)
        };
        let (half, one, two) = (
            constant(self, 0.5),
            constant(self, 1.0),
            constant(self, 2.0),
        );
        let fraction = self.float(Opcode::SubF, x, floor, scalar);
        let above = self.float(Opcode::AddF, floor, one, scalar);
        // The floor is odd when it is not twice the floor of its half.
        let halved = self.float(Opcode::DivF, floor, two, scalar);
        let halved = self.emit_one(
            OpBuilder::new(Opcode::Floor, Location::Unknown)
                .operand(halved)
                .result(self.tile(scalar)),
        );
        let doubled = self.float(Opcode::MulF, halved, two, scalar);
        let odd = self.compare_float(BinaryOp::NotEqual, floor, doubled);
        let more = self.compare_float(BinaryOp::Greater, fraction, half);
        let tie = self.compare_float(BinaryOp::Equal, fraction, half);
        let odd_tie = self.integer(Opcode::AndI, tie, odd, &[]);
        let up = self.integer(Opcode::OrI, more, odd_tie, &[]);
        let rounded = self.select(up, above, floor, scalar);
        // Infinities have no fraction to round, and NaN stays NaN.
        let finite = self.compare_float(BinaryOp::Equal, fraction, fraction);
        self.select(finite, rounded, x, scalar)
    }

    fn binary(&mut self, op: BinaryOp, element: ElementType, x: Value, y: Value) -> Value {
        let scalar = compute_scalar(element);
        if op.is_comparison() {
            return if element.is_float() {
                self.compare_float(op, x, y)
            } else {
                let predicate = predicate(op);
                let signedness = signedness(element);
                let ty = self.tile(ScalarType::I1);
                self.emit_one(
                    OpBuilder::new(Opcode::CmpI, Location::Unknown)
                        .operand(x)
                        .operand(y)
                        .attr("comparison_predicate", Attribute::i32(predicate))
                        .attr("signedness", Attribute::i32(signedness))
                        .result(ty),
                )
            };
        }

        if element.is_float() {
            let ty = self.tile(scalar);
            let pair = |opcode| {
                OpBuilder::new(opcode, Location::Unknown)
                    .operand(x)
                    .operand(y)
            };
            return match op {
                BinaryOp::Add => self.float(Opcode::AddF, x, y, scalar),
                BinaryOp::Sub => self.float(Opcode::SubF, x, y, scalar),
                BinaryOp::Mul => self.float(Opcode::MulF, x, y, scalar),
                BinaryOp::Div => self.float(Opcode::DivF, x, y, scalar),
                BinaryOp::Rem => self.emit_one(pair(Opcode::RemF).result(ty)),
                BinaryOp::Pow => self.emit_one(pair(Opcode::Pow).result(ty)),
                BinaryOp::Atan2 => self.emit_one(pair(Opcode::Atan2).result(ty)),
                // NaN in either operand propagates, as the IR's maximum and minimum do.
                BinaryOp::Max => self.emit_one(
                    pair(Opcode::MaxF)
                        .attr("propagate_nan", Attribute::Bool(true))
                        .result(ty),
                ),
                BinaryOp::Min => self.emit_one(
                    pair(Opcode::MinF)
                        .attr("propagate_nan", Attribute::Bool(true))
                        .result(ty),
                ),
                _ => unreachable!("verification allows no other float operation"),
            };
        }

        let signed = &[signedness(element)][..];
        match op {
            BinaryOp::Add => self.integer(Opcode::AddI, x, y, &[]),
            BinaryOp::Sub => self.integer(Opcode::SubI, x, y, &[]),
            BinaryOp::Mul => self.integer(Opcode::MulI, x, y, &[]),
            BinaryOp::Div => self.integer(Opcode::DivI, x, y, &[signed[0], ZERO]),
            BinaryOp::Rem => self.integer(Opcode::RemI, x, y, signed),
            BinaryOp::Max => self.integer(Opcode::MaxI, x, y, signed),
            BinaryOp::Min => self.integer(Opcode::MinI, x, y, signed),
            BinaryOp::And | BinaryOp::BitAnd => self.integer(Opcode::AndI, x, y, &[]),
            BinaryOp::Or | BinaryOp::BitOr => self.integer(Opcode::OrI, x, y, &[]),
            BinaryOp::BitXor => self.integer(Opcode::XOrI, x, y, &[]),
            _ => unreachable!(
                "`emit` rejects shifts and integer powers, and verification allows no other integer operation"
            ),
        }
    }

    /// Returns `x op y` for a floating-point arithmetic `opcode`, rounded to nearest even.
    fn float(&mut self, opcode: Opcode, x: Value, y: Value, scalar: ScalarType) -> Value {
        let ty = self.tile(scalar);
        self.emit_one(
            OpBuilder::new(opcode, Location::Unknown)
                .operand(x)
                .operand(y)
                .attr("rounding_mode", Attribute::i32(NEAREST_EVEN))
                .result(ty),
        )
    }

    /// Returns `x op y` for an integer `opcode`, with the operation's signedness and rounding attributes in order when it takes them.
    fn integer(&mut self, opcode: Opcode, x: Value, y: Value, attributes: &[i64]) -> Value {
        let ty = self.module.value_type(x).clone();
        let mut builder = OpBuilder::new(opcode, Location::Unknown)
            .operand(x)
            .operand(y)
            .result(ty);
        builder = match opcode {
            Opcode::AddI | Opcode::SubI | Opcode::MulI => {
                builder.attr("overflow", Attribute::i32(0))
            }
            Opcode::DivI => builder
                .attr("signedness", Attribute::i32(attributes[0]))
                .attr("rounding", Attribute::i32(attributes[1])),
            Opcode::RemI | Opcode::MaxI | Opcode::MinI => {
                builder.attr("signedness", Attribute::i32(attributes[0]))
            }
            _ => builder,
        };
        self.emit_one(builder)
    }

    /// Returns `x op y` for a floating-point comparison: `!=` holds for NaN, and the other comparisons do not.
    fn compare_float(&mut self, op: BinaryOp, x: Value, y: Value) -> Value {
        let ordering = i64::from(op != BinaryOp::NotEqual);
        let ty = self.tile(ScalarType::I1);
        self.emit_one(
            OpBuilder::new(Opcode::CmpF, Location::Unknown)
                .operand(x)
                .operand(y)
                .attr("comparison_predicate", Attribute::i32(predicate(op)))
                .attr("comparison_ordering", Attribute::i32(ordering))
                .result(ty),
        )
    }

    /// Returns the unsigned comparison of two `I64` tiles under `predicate`.
    fn compare_unsigned(&mut self, predicate: i64, x: Value, y: Value) -> Value {
        let ty = self.tile(ScalarType::I1);
        self.emit_one(
            OpBuilder::new(Opcode::CmpI, Location::Unknown)
                .operand(x)
                .operand(y)
                .attr("comparison_predicate", Attribute::i32(predicate))
                .attr("signedness", Attribute::i32(UNSIGNED))
                .result(ty),
        )
    }

    fn select(&mut self, condition: Value, x: Value, y: Value, scalar: ScalarType) -> Value {
        let ty = self.tile(scalar);
        self.emit_one(
            OpBuilder::new(Opcode::Select, Location::Unknown)
                .operand(condition)
                .operand(x)
                .operand(y)
                .result(ty),
        )
    }

    /// Returns `value`, a tile of `from`'s compute type, converted to `to`'s, as the IR's casts convert: floats round to nearest even, floats become integers toward zero and saturating, and values become `bool` by comparing with zero.
    fn convert(&mut self, value: Value, from: ElementType, to: ElementType) -> Value {
        let (from_scalar, to_scalar) = (compute_scalar(from), compute_scalar(to));
        if from_scalar == to_scalar {
            return value;
        }

        let ty = self.tile(to_scalar);
        if to == ElementType::Bool {
            let zero = if from.is_float() {
                Literal::Float(0.0)
            } else {
                Literal::Int(0)
            };
            let zero = self.constant(from, zero);
            let zero = self.splat(zero, from_scalar);
            return if from.is_float() {
                self.compare_float(BinaryOp::NotEqual, value, zero)
            } else {
                self.binary(BinaryOp::NotEqual, from, value, zero)
            };
        }

        let builder = OpBuilder::new(conversion(from, to), Location::Unknown)
            .operand(value)
            .result(ty);
        let builder = match conversion(from, to) {
            Opcode::FToF => builder.attr("rounding_mode", Attribute::i32(NEAREST_EVEN)),
            Opcode::IToF => builder
                .attr("signedness", Attribute::i32(signedness(from)))
                .attr("rounding_mode", Attribute::i32(NEAREST_EVEN)),
            Opcode::FToI => builder
                .attr("saturating", Attribute::Bool(true))
                .attr("signedness", Attribute::i32(signedness(to)))
                .attr("rounding_mode", Attribute::i32(NEAREST_INT_TO_ZERO)),
            Opcode::ExtI => builder.attr("signedness", Attribute::i32(signedness(from))),
            _ => builder.attr("overflow", Attribute::i32(0)),
        };
        self.emit_one(builder)
    }

    /// Returns a loaded tile of `element`'s storage type in its compute type.
    fn widened(&mut self, raw: Value, element: ElementType) -> Value {
        match element {
            ElementType::Bool => {
                let zero = self.constant(ElementType::U8, Literal::Int(0));
                let zero = self.splat(zero, ScalarType::I8);
                self.binary(BinaryOp::NotEqual, ElementType::U8, raw, zero)
            }
            ElementType::F16 | ElementType::Bf16 => {
                let ty = self.tile(ScalarType::F32);
                self.emit_one(
                    OpBuilder::new(Opcode::FToF, Location::Unknown)
                        .operand(raw)
                        .attr("rounding_mode", Attribute::i32(NEAREST_EVEN))
                        .result(ty),
                )
            }
            _ => raw,
        }
    }

    /// Returns a tile of `element`'s compute type in its storage type, rounding to nearest even.
    fn narrowed(&mut self, value: Value, element: ElementType) -> Value {
        let storage = storage_scalar(element);
        match element {
            ElementType::Bool => {
                let ty = self.tile(storage);
                self.emit_one(
                    OpBuilder::new(Opcode::ExtI, Location::Unknown)
                        .operand(value)
                        .attr("signedness", Attribute::i32(UNSIGNED))
                        .result(ty),
                )
            }
            ElementType::F16 | ElementType::Bf16 => {
                let ty = self.tile(storage);
                self.emit_one(
                    OpBuilder::new(Opcode::FToF, Location::Unknown)
                        .operand(value)
                        .attr("rounding_mode", Attribute::i32(NEAREST_EVEN))
                        .result(ty),
                )
            }
            _ => value,
        }
    }

    /// Returns the zero-dimensional `I64` value of `size`.
    fn size_value(&mut self, size: SizeId) -> Value {
        if let Some(value) = self.sizes[size.0] {
            return value;
        }

        let Size::Static(value) = self.kernel.sizes[size.0] else {
            unreachable!("argument sizes are parameters");
        };
        let value = self.constant_i64(value.cast_signed());
        self.sizes[size.0] = Some(value);
        value
    }

    /// Returns the product of the extents of `axes`, one for none, as a zero-dimensional `I64`.
    fn product(&mut self, axes: &[AxisId]) -> Value {
        let mut product = self.constant_i64(1);
        for &axis in axes {
            let extent = self.size_value(self.kernel.axes[axis.0].extent);
            product = self.integer(Opcode::MulI, product, extent, &[]);
        }
        product
    }

    /// Returns this tile block's index along the grid's first dimension, as a zero-dimensional `I64`.
    fn block_id(&mut self) -> Value {
        let index = tile_type(ScalarType::I32, &[]);
        let block = self.emit(
            OpBuilder::new(Opcode::GetTileBlockId, Location::Unknown)
                .result(index.clone())
                .result(index.clone())
                .result(index),
        )[0];
        let ty = tile_type(ScalarType::I64, &[]);
        self.emit_one(
            OpBuilder::new(Opcode::ExtI, Location::Unknown)
                .operand(block)
                .attr("signedness", Attribute::i32(UNSIGNED))
                .result(ty),
        )
    }

    /// Returns the `I64` tile `0, 1, …, lanes - 1`.
    fn iota(&mut self) -> Value {
        let ty = self.tile(ScalarType::I64);
        self.emit_one(OpBuilder::new(Opcode::Iota, Location::Unknown).result(ty))
    }

    fn constant_i64(&mut self, value: i64) -> Value {
        self.constant(ElementType::I64, Literal::Int(value))
    }

    /// Returns a zero-dimensional constant of `element`'s compute type.
    fn constant(&mut self, element: ElementType, literal: Literal) -> Value {
        let scalar = compute_scalar(element);
        let ty = tile_type(scalar, &[]);
        self.emit_one(
            OpBuilder::new(Opcode::Constant, Location::Unknown)
                .attr(
                    "value",
                    Attribute::DenseElements(DenseElements {
                        element_type: ty.clone(),
                        shape: Vec::new(),
                        data: literal_bytes(scalar, literal),
                    }),
                )
                .result(ty),
        )
    }

    /// Returns a zero-dimensional `value` of `scalar` repeated across a tile.
    fn splat(&mut self, value: Value, scalar: ScalarType) -> Value {
        let one = tile_type(scalar, &[1]);
        let reshaped = self.emit_one(
            OpBuilder::new(Opcode::Reshape, Location::Unknown)
                .operand(value)
                .result(one),
        );
        let ty = self.tile(scalar);
        self.emit_one(
            OpBuilder::new(Opcode::Broadcast, Location::Unknown)
                .operand(reshaped)
                .result(ty),
        )
    }

    /// Returns a zero-dimensional pointer to `storage` repeated across a tile.
    fn splat_pointer(&mut self, pointer: Value, storage: ScalarType) -> Value {
        let one = pointer_type(storage, &[1]);
        let reshaped = self.emit_one(
            OpBuilder::new(Opcode::Reshape, Location::Unknown)
                .operand(pointer)
                .result(one),
        );
        let ty = pointer_type(storage, &[self.lanes]);
        self.emit_one(
            OpBuilder::new(Opcode::Broadcast, Location::Unknown)
                .operand(reshaped)
                .result(ty),
        )
    }

    fn tile(&self, scalar: ScalarType) -> Type {
        tile_type(scalar, &[self.lanes])
    }

    fn axes_of_kind(&self, kind: AxisKind) -> Vec<AxisId> {
        self.kernel
            .axes
            .iter()
            .enumerate()
            .filter(|(_, axis)| axis.kind == kind)
            .map(|(index, _)| AxisId(index))
            .collect()
    }

    fn emit_one(&mut self, builder: OpBuilder) -> Value {
        self.emit(builder)[0]
    }

    fn emit(&mut self, builder: OpBuilder) -> Vec<Value> {
        let block = self.block;
        self.emit_in(block, builder)
    }

    fn emit_in(&mut self, block: BlockId, builder: OpBuilder) -> Vec<Value> {
        let (op, results) = builder.build(&mut self.module);
        append_op(&mut self.module, block, op);
        results
    }
}

/// The `memory_ordering_semantics` of atomics that need no ordering with other accesses.
const RELAXED: i64 = 1;

/// The `memory_scope` of atomics every thread of the device sees.
const DEVICE: i64 = 1;

/// The atomic `mode` that adds integers.
const ADD: i64 = 3;

/// The atomic `mode` that adds floating-point values.
const ADD_FLOAT: i64 = 4;

/// The `signedness` of unsigned integers.
const UNSIGNED: i64 = 0;

/// The `signedness` of signed integers.
const SIGNED: i64 = 1;

/// The `rounding_mode` that rounds to nearest, ties to even.
const NEAREST_EVEN: i64 = 0;

/// The `rounding` of integer division toward zero.
const ZERO: i64 = 1;

/// The `rounding_mode` of float-to-integer conversion, toward zero.
const NEAREST_INT_TO_ZERO: i64 = 6;

/// Returns the operation that combines two zero-dimensional values with `combiner`.
fn combine_op(
    combiner: Combiner,
    element: ElementType,
    lhs: Value,
    rhs: Value,
    ty: Type,
) -> OpBuilder {
    let builder = |opcode| {
        OpBuilder::new(opcode, Location::Unknown)
            .operand(lhs)
            .operand(rhs)
            .result(ty)
    };
    if element.is_float() {
        match combiner {
            Combiner::Sum => {
                builder(Opcode::AddF).attr("rounding_mode", Attribute::i32(NEAREST_EVEN))
            }
            Combiner::Product => {
                builder(Opcode::MulF).attr("rounding_mode", Attribute::i32(NEAREST_EVEN))
            }
            Combiner::Max => builder(Opcode::MaxF).attr("propagate_nan", Attribute::Bool(true)),
            Combiner::Min => builder(Opcode::MinF).attr("propagate_nan", Attribute::Bool(true)),
            Combiner::LogSumExp => unreachable!("`emit` rejects log-sum-exp reductions"),
        }
    } else {
        let signed = Attribute::i32(signedness(element));
        match combiner {
            Combiner::Sum => builder(Opcode::AddI).attr("overflow", Attribute::i32(0)),
            Combiner::Product => builder(Opcode::MulI).attr("overflow", Attribute::i32(0)),
            Combiner::Max => builder(Opcode::MaxI).attr("signedness", signed),
            Combiner::Min => builder(Opcode::MinI).attr("signedness", signed),
            Combiner::LogSumExp => unreachable!("`emit` rejects log-sum-exp reductions"),
        }
    }
}

/// Returns the value a reduction with `combiner` starts from.
fn identity(combiner: Combiner, element: ElementType) -> Literal {
    let (low, high) = match element {
        ElementType::U8 => (0, i64::from(u8::MAX)),
        ElementType::I8 => (i64::from(i8::MIN), i64::from(i8::MAX)),
        ElementType::U16 => (0, i64::from(u16::MAX)),
        ElementType::I16 => (i64::from(i16::MIN), i64::from(i16::MAX)),
        ElementType::U32 => (0, i64::from(u32::MAX)),
        ElementType::I32 => (i64::from(i32::MIN), i64::from(i32::MAX)),
        // The all-ones pattern is the largest `U64`.
        ElementType::U64 => (0, -1),
        _ => (i64::MIN, i64::MAX),
    };
    match (combiner, element.is_float()) {
        (Combiner::Sum, true) => Literal::Float(0.0),
        (Combiner::Product, true) => Literal::Float(1.0),
        (Combiner::Max, true) => Literal::Float(f64::NEG_INFINITY),
        (Combiner::Min, true) => Literal::Float(f64::INFINITY),
        (Combiner::LogSumExp, _) => unreachable!("`emit` rejects log-sum-exp reductions"),
        (Combiner::Sum, false) => Literal::Int(0),
        (Combiner::Product, false) => Literal::Int(1),
        (Combiner::Max, false) => Literal::Int(low),
        (Combiner::Min, false) => Literal::Int(high),
    }
}

/// Returns the conversion between two different compute types, neither `bool` as a target.
fn conversion(from: ElementType, to: ElementType) -> Opcode {
    match (from.is_float(), to.is_float()) {
        (true, true) => Opcode::FToF,
        (false, true) => Opcode::IToF,
        (true, false) => Opcode::FToI,
        (false, false) if bit_width(to) > bit_width(from) => Opcode::ExtI,
        (false, false) => Opcode::TruncI,
    }
}

fn bit_width(element: ElementType) -> usize {
    match element {
        ElementType::Bool => 1,
        other => other.size_bytes() * 8,
    }
}

/// Returns the `signedness` an integer of `element` takes; `bool` is unsigned.
fn signedness(element: ElementType) -> i64 {
    match element {
        ElementType::I8 | ElementType::I16 | ElementType::I32 | ElementType::I64 => SIGNED,
        _ => UNSIGNED,
    }
}

/// Returns the `comparison_predicate` of a comparison.
fn predicate(op: BinaryOp) -> i64 {
    match op {
        BinaryOp::Equal => 0,
        BinaryOp::NotEqual => 1,
        BinaryOp::Less => 2,
        BinaryOp::LessEqual => 3,
        BinaryOp::Greater => 4,
        _ => 5,
    }
}

/// Returns the Tile IR type values of `element` are computed in.
fn compute_scalar(element: ElementType) -> ScalarType {
    match element {
        ElementType::Bool => ScalarType::I1,
        ElementType::F16 | ElementType::Bf16 => ScalarType::F32,
        other => storage_scalar(other),
    }
}

/// Returns the Tile IR type elements of `element` are stored as; `bool` is a byte.
fn storage_scalar(element: ElementType) -> ScalarType {
    match element {
        ElementType::Bool | ElementType::U8 | ElementType::I8 => ScalarType::I8,
        ElementType::U16 | ElementType::I16 => ScalarType::I16,
        ElementType::U32 | ElementType::I32 => ScalarType::I32,
        ElementType::U64 | ElementType::I64 => ScalarType::I64,
        ElementType::F16 => ScalarType::F16,
        ElementType::Bf16 => ScalarType::BF16,
        ElementType::F32 => ScalarType::F32,
        ElementType::F64 => ScalarType::F64,
        ElementType::F8E4M3 => ScalarType::F8E4M3FN,
        ElementType::F8E5M2 => ScalarType::F8E5M2,
    }
}

/// Returns the little-endian bytes of `literal` as a `scalar`; `bool` true is all ones.
#[expect(
    clippy::cast_possible_truncation,
    reason = "a literal is its value in the constant's type"
)]
fn literal_bytes(scalar: ScalarType, literal: Literal) -> Vec<u8> {
    match (scalar, literal) {
        (ScalarType::I1, Literal::Bool(value)) => vec![if value { 0xff } else { 0 }],
        (ScalarType::F32, Literal::Float(value)) => (value as f32).to_le_bytes().to_vec(),
        (ScalarType::F64, Literal::Float(value)) => value.to_le_bytes().to_vec(),
        (scalar, Literal::Int(value)) => {
            let bytes = value.to_le_bytes();
            bytes[..integer_bytes(scalar)].to_vec()
        }
        (scalar, Literal::Float(value)) => {
            let bytes = (value as i64).to_le_bytes();
            bytes[..integer_bytes(scalar)].to_vec()
        }
        (scalar, Literal::Bool(value)) => {
            let bytes = i64::from(value).to_le_bytes();
            bytes[..integer_bytes(scalar)].to_vec()
        }
    }
}

fn integer_bytes(scalar: ScalarType) -> usize {
    match scalar {
        ScalarType::I1 | ScalarType::I8 => 1,
        ScalarType::I16 => 2,
        ScalarType::I32 => 4,
        _ => 8,
    }
}

fn tile_type(scalar: ScalarType, shape: &[i64]) -> Type {
    Type::Tile(TileType {
        shape: shape.to_vec(),
        element_type: TileElementType::Scalar(scalar),
    })
}

fn pointer_type(pointee: ScalarType, shape: &[i64]) -> Type {
    Type::Tile(TileType {
        shape: shape.to_vec(),
        element_type: TileElementType::Pointer(Box::new(PointerType { pointee })),
    })
}

/// Returns an `operandSegmentSizes` attribute.
fn segment_sizes(sizes: &[i64]) -> Attribute {
    Attribute::Array(sizes.iter().map(|&size| Attribute::i32(size)).collect())
}

#[cfg(test)]
mod tests;
