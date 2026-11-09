//! Tensor-core contractions for [`Schedule::TileMatmul`](crate::schedule::Schedule::TileMatmul).
//!
//! Each tile block computes an `m` by `n` tile of one batch point's output in an `F32` accumulator, which a `for` loop over the contraction carries through `mmaf` products of an `m` by `k` tile of the left operand and a `k` by `n` tile of the right.
//! Operand tiles are gathers through two-dimensional pointer tiles, masked past the rows, columns, and contraction and padded with zeros, so any strides and tails work.
//! Loading through tensor views and partition views instead, with the transposed weight mapped by the view, took three to six times as long under tileiras for the decoder benchmark's linears.

use std::mem;

use cutile_ir::{
    builder::{OpBuilder, build_single_block_region},
    bytecode::Opcode,
    ir::{Attribute, DenseElements, Location, ScalarType, Type, Value},
};

use crate::{
    emit::{
        EmitError,
        tile::{
            NEAREST_EVEN, UNSIGNED, Writer, ZERO, pointer_type, segment_sizes, storage_scalar,
            tile_type,
        },
    },
    ir::{Access, AxisId, Expr, Kernel, NodeId, Size, SizeId},
    schedule::contraction::Contraction,
};

/// The `comparison_predicate` of `<`.
const LESS: i64 = 2;

/// The `signedness` of the signed comparisons of indices.
const SIGNED: i64 = 1;

impl Writer<'_> {
    /// Emits `contraction` as tiles of `[m, n, k]` points.
    pub fn matmul(
        &mut self,
        contraction: &Contraction,
        [m, n, k]: [i64; 3],
    ) -> Result<(), EmitError> {
        let kernel = self.kernel;
        // Sizes are made once and cached, so they must be made before the loop body, whose values its outside cannot read.
        for size in 0..kernel.sizes.len() {
            self.size_value(SizeId(size));
        }

        let extent =
            |writer: &mut Self, axis: AxisId| writer.size_value(kernel.axes[axis.0].extent);
        let rows = extent(self, contraction.rows);
        let columns = extent(self, contraction.columns);
        let depth = extent(self, contraction.reduced);
        let row_tiles = self.ceil_div(rows, m);
        let column_tiles = self.ceil_div(columns, n);
        // Blocks walk the column tiles fastest, then the row tiles, then the batch points.
        let block = self.block_id();
        let column_tile = self.integer(Opcode::RemI, block, column_tiles, &[UNSIGNED]);
        let rest = self.integer(Opcode::DivI, block, column_tiles, &[UNSIGNED, ZERO]);
        let row_tile = self.integer(Opcode::RemI, rest, row_tiles, &[UNSIGNED]);
        let batch = self.integer(Opcode::DivI, rest, row_tiles, &[UNSIGNED, ZERO]);
        for (axis, index) in self.decompose(batch, &contraction.batch, false) {
            self.axes[axis.0] = Some(index);
        }

        let row_index = self.tile_indices(row_tile, m);
        let column_index = self.tile_indices(column_tile, n);
        let lhs = load_access(kernel, contraction.lhs)?;
        let rhs = load_access(kernel, contraction.rhs)?;
        let store = &kernel.stores[0];
        let element = kernel.operands[lhs.operand.0].element;
        let storage = storage_scalar(element);
        // The offsets and masks of the rows and columns, which every step shares.
        let lhs_rows = self.axis_offsets(lhs, contraction.rows, row_index, [m, k], 0);
        let rhs_columns = self.axis_offsets(rhs, contraction.columns, column_index, [k, n], 1);
        let rows_in = self.in_range(row_index, rows);
        let rows_in = self.spread(rows_in, ScalarType::I1, &[m, 1], &[m, k]);
        let columns_in = self.in_range(column_index, columns);
        let columns_in = self.spread(columns_in, ScalarType::I1, &[1, n], &[k, n]);
        let accumulator = tile_type(ScalarType::F32, &[m, n]);
        let zero = self.zero(ScalarType::F32);
        let initial = self.spread(zero, ScalarType::F32, &[1, 1], &[m, n]);
        let steps = self.ceil_div(depth, k);
        let first = self.constant_i64(0);
        let step = self.constant_i64(1);
        let (region, body, arguments) = build_single_block_region(
            &mut self.module,
            &[tile_type(ScalarType::I64, &[]), accumulator.clone()],
        );
        let outer = mem::replace(&mut self.block, body);
        let width = self.constant_i64(k);
        let start = self.integer(Opcode::MulI, arguments[0], width, &[]);
        let depth_index = self.tile_indices_from(start, k);
        let depth_in = self.in_range(depth_index, depth);
        let lhs_depth = self.axis_offsets(lhs, contraction.reduced, depth_index, [m, k], 1);
        let lhs_offsets = self.integer(Opcode::AddI, lhs_rows, lhs_depth, &[]);
        let lhs_depth_in = self.spread(depth_in, ScalarType::I1, &[1, k], &[m, k]);
        let lhs_mask = self.integer(Opcode::AndI, rows_in, lhs_depth_in, &[]);
        let lhs_tile = self.gather(lhs, lhs_offsets, lhs_mask, storage, [m, k]);
        let rhs_depth = self.axis_offsets(rhs, contraction.reduced, depth_index, [k, n], 0);
        let rhs_offsets = self.integer(Opcode::AddI, rhs_columns, rhs_depth, &[]);
        let rhs_depth_in = self.spread(depth_in, ScalarType::I1, &[k, 1], &[k, n]);
        let rhs_mask = self.integer(Opcode::AndI, columns_in, rhs_depth_in, &[]);
        let rhs_tile = self.gather(rhs, rhs_offsets, rhs_mask, storage, [k, n]);
        let product = self.emit_one(
            OpBuilder::new(Opcode::MmaF, Location::Unknown)
                .operand(lhs_tile)
                .operand(rhs_tile)
                .operand(arguments[1])
                .result(accumulator.clone()),
        );
        self.emit(OpBuilder::new(Opcode::Continue, Location::Unknown).operand(product));
        self.block = outer;
        let result = self.emit_one(
            OpBuilder::new(Opcode::For, Location::Unknown)
                .operand(first)
                .operand(steps)
                .operand(step)
                .operand(initial)
                .result(accumulator)
                .region(region),
        );
        // The output tile, rounded to its storage type and written where its rows and columns are in range.
        let output = kernel.operands[store.access.operand.0].element;
        let output_storage = storage_scalar(output);
        let value = if output_storage == ScalarType::F32 {
            result
        } else {
            self.emit_one(
                OpBuilder::new(Opcode::FToF, Location::Unknown)
                    .operand(result)
                    .attr("rounding_mode", Attribute::i32(NEAREST_EVEN))
                    .result(tile_type(output_storage, &[m, n])),
            )
        };
        let store_rows = self.axis_offsets(&store.access, contraction.rows, row_index, [m, n], 0);
        let store_columns =
            self.axis_offsets(&store.access, contraction.columns, column_index, [m, n], 1);
        let offsets = self.integer(Opcode::AddI, store_rows, store_columns, &[]);
        let rows_out = self.in_range(row_index, rows);
        let rows_out = self.spread(rows_out, ScalarType::I1, &[m, 1], &[m, n]);
        let columns_out = self.in_range(column_index, columns);
        let columns_out = self.spread(columns_out, ScalarType::I1, &[1, n], &[m, n]);
        let mask = self.integer(Opcode::AndI, rows_out, columns_out, &[]);
        let pointers = self.pointer_tile(&store.access, offsets, output_storage, [m, n]);
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

    /// Returns `⌈value / divisor⌉` of a zero-dimensional `I64`.
    fn ceil_div(&mut self, value: Value, divisor: i64) -> Value {
        let bias = self.constant_i64(divisor - 1);
        let biased = self.integer(Opcode::AddI, value, bias, &[]);
        let divisor = self.constant_i64(divisor);
        self.integer(Opcode::DivI, biased, divisor, &[UNSIGNED, ZERO])
    }

    /// Returns the `I64` tile `tile × length, …, tile × length + length - 1` of tile index `tile`.
    fn tile_indices(&mut self, tile: Value, length: i64) -> Value {
        let size = self.constant_i64(length);
        let start = self.integer(Opcode::MulI, tile, size, &[]);
        self.tile_indices_from(start, length)
    }

    /// Returns the `I64` tile `start, …, start + length - 1`.
    fn tile_indices_from(&mut self, start: Value, length: i64) -> Value {
        let iota = self.emit_one(
            OpBuilder::new(Opcode::Iota, Location::Unknown)
                .result(tile_type(ScalarType::I64, &[length])),
        );
        let start = self.spread(start, ScalarType::I64, &[1], &[length]);
        self.integer(Opcode::AddI, start, iota, &[])
    }

    /// Returns where the indices of a one-dimensional `I64` tile are below the zero-dimensional `extent`.
    fn in_range(&mut self, indices: Value, extent: Value) -> Value {
        let shape = self.shape(indices);
        let extent = self.spread(extent, ScalarType::I64, &[1], &shape);
        self.emit_one(
            OpBuilder::new(Opcode::CmpI, Location::Unknown)
                .operand(indices)
                .operand(extent)
                .attr("comparison_predicate", Attribute::i32(LESS))
                .attr("signedness", Attribute::i32(SIGNED))
                .result(tile_type(ScalarType::I1, &shape)),
        )
    }

    /// Returns the offsets `access` adds along `axis`, whose indices are the one-dimensional tile `indices`, laid along dimension `dimension` of a tile of `shape`, plus, along the first dimension, its base and its terms of the axes already indexed.
    fn axis_offsets(
        &mut self,
        access: &Access,
        axis: AxisId,
        indices: Value,
        shape: [i64; 2],
        dimension: usize,
    ) -> Value {
        let length = shape[dimension];
        let mut along = [1, 1];
        along[dimension] = length;
        let mut offsets: Option<Value> = None;
        for &(walked, stride) in &access.strides {
            if walked != axis {
                continue;
            }

            let stride = self.size_value(stride);
            let stride = self.spread(stride, ScalarType::I64, &[1], &[length]);
            let term = self.integer(Opcode::MulI, indices, stride, &[]);
            offsets = Some(offsets.map_or(term, |sum| self.integer(Opcode::AddI, sum, term, &[])));
        }

        let zero = self.constant_i64(0);
        let offsets =
            offsets.unwrap_or_else(|| self.spread(zero, ScalarType::I64, &[1], &[length]));
        let mut offsets = self.spread(offsets, ScalarType::I64, &along, &shape);
        // The base and the batch axes' terms are zero-dimensional, added once, with the first dimension's offsets.
        if dimension == 0 {
            let mut fixed = access.base.map_or(zero, |base| self.size_value(base));
            for &(walked, stride) in &access.strides {
                let Some(index) = self.axes[walked.0] else {
                    continue;
                };
                if self.kernel.sizes[stride.0] == Size::Static(0) {
                    continue;
                }

                let stride = self.size_value(stride);
                let term = self.integer(Opcode::MulI, index, stride, &[]);
                fixed = self.integer(Opcode::AddI, fixed, term, &[]);
            }

            let fixed = self.spread(fixed, ScalarType::I64, &[1, 1], &shape);
            offsets = self.integer(Opcode::AddI, offsets, fixed, &[]);
        }
        offsets
    }

    /// Returns the tile of `shape` that `access` reads at `offsets`, zero where `mask` is false.
    fn gather(
        &mut self,
        access: &Access,
        offsets: Value,
        mask: Value,
        storage: ScalarType,
        shape: [i64; 2],
    ) -> Value {
        let pointers = self.pointer_tile(access, offsets, storage, shape);
        let zero = self.zero(storage);
        let padding = self.spread(zero, storage, &[1, 1], &shape);
        self.emit(
            OpBuilder::new(Opcode::LoadPtrTko, Location::Unknown)
                .result(tile_type(storage, &shape))
                .result(Type::Token)
                .operand(pointers)
                .operand(mask)
                .operand(padding)
                .attr("memory_ordering_semantics", Attribute::i32(0))
                .attr("operandSegmentSizes", segment_sizes(&[1, 1, 1, 0])),
        )[0]
    }

    /// Returns the pointers to `access`'s operand, of elements stored as `storage`, at `offsets`, a tile of `shape`.
    fn pointer_tile(
        &mut self,
        access: &Access,
        offsets: Value,
        storage: ScalarType,
        shape: [i64; 2],
    ) -> Value {
        let operand = self.operands[access.operand.0];
        let reshaped = self.emit_one(
            OpBuilder::new(Opcode::Reshape, Location::Unknown)
                .operand(operand)
                .result(pointer_type(storage, &[1, 1])),
        );
        let base = self.emit_one(
            OpBuilder::new(Opcode::Broadcast, Location::Unknown)
                .operand(reshaped)
                .result(pointer_type(storage, &shape)),
        );
        self.emit_one(
            OpBuilder::new(Opcode::Offset, Location::Unknown)
                .operand(base)
                .operand(offsets)
                .result(pointer_type(storage, &shape)),
        )
    }

    /// Returns `value`, a tile of `scalar` of shape `from`, whose dimensions are each one or those of `to`, reshaped to `from` and broadcast to `to`.
    fn spread(&mut self, value: Value, scalar: ScalarType, from: &[i64], to: &[i64]) -> Value {
        let reshaped = self.emit_one(
            OpBuilder::new(Opcode::Reshape, Location::Unknown)
                .operand(value)
                .result(tile_type(scalar, from)),
        );
        if from == to {
            return reshaped;
        }
        self.emit_one(
            OpBuilder::new(Opcode::Broadcast, Location::Unknown)
                .operand(reshaped)
                .result(tile_type(scalar, to)),
        )
    }

    /// Returns a zero-dimensional zero of `scalar`.
    fn zero(&mut self, scalar: ScalarType) -> Value {
        let ty = tile_type(scalar, &[]);
        let bytes = match scalar {
            ScalarType::F16 | ScalarType::BF16 => 2,
            ScalarType::F64 => 8,
            _ => 4,
        };
        self.emit_one(
            OpBuilder::new(Opcode::Constant, Location::Unknown)
                .attr(
                    "value",
                    Attribute::DenseElements(DenseElements {
                        element_type: ty.clone(),
                        shape: Vec::new(),
                        data: vec![0; bytes],
                    }),
                )
                .result(ty),
        )
    }

    /// Returns the shape of tile `value`.
    fn shape(&self, value: Value) -> Vec<i64> {
        match self.module.value_type(value) {
            Type::Tile(tile) => tile.shape.clone(),
            _ => Vec::new(),
        }
    }
}

/// Returns the access of load `node`.
fn load_access(kernel: &Kernel, node: NodeId) -> Result<&Access, EmitError> {
    match &kernel.nodes[node.0].expr {
        Expr::Load(access) => Ok(access),
        _ => Err(EmitError::Unsupported {
            reason: "a contraction operand that is not a load",
        }),
    }
}
