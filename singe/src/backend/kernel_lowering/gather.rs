//! Gathers and scatters through index tensors, as accesses whose offsets add checked index values.

use std::iter::repeat_n;

use singe_codegen::{
    ir::{Access, AxisKind, ElementType, NodeId, SizeId},
    schedule::element_schedule,
};

use crate::{
    backend::kernel_lowering::{
        STATUS_INDEX_OUT_OF_BOUNDS,
        domain::{DomainKernel, LoweredKernel},
        element_type,
    },
    data_type::DataType,
    error::{Error, Result},
    layout::Layout,
    shape::Shape,
    utility::to_u64,
    value::ValueSpec,
};

/// The input dimensions that index tensors select along, and the tensors that select them.
///
/// The index tensors select consecutive input dimensions from `axis`, one each, and broadcast to `index_shape`, which replaces those dimensions in the selected value's shape.
pub struct Selection<'a> {
    pub axis: usize,
    /// The extent of each selected dimension, which bounds its index values.
    pub extents: &'a [i64],
    pub index_shape: &'a Shape,
    /// The index tensors, of `I32` or `I64`, in dimension order.
    pub indices: &'a [(DataType, &'a Layout)],
    /// Whether a negative index counts from the end of its dimension.
    pub wrap: bool,
}

impl Selection<'_> {
    /// Returns the selected value's dimensions when the selected tensor has `dimensions`: those before `axis`, the index shape, then those after the selected ones.
    fn dimensions(&self, dimensions: &[i64]) -> Result<Vec<i64>> {
        let end = self.axis + self.extents.len();
        if end > dimensions.len() {
            return Err(Error::DimensionOutOfBounds);
        }
        Ok(dimensions[..self.axis]
            .iter()
            .chain(self.index_shape.dimensions())
            .chain(&dimensions[end..])
            .copied()
            .collect())
    }

    /// Returns `strides` of the selected tensor placed on the selected value's dimensions: zero along the index shape, which the index tensors walk instead.
    fn direct_strides(&self, strides: &[i64]) -> Vec<i64> {
        let end = self.axis + self.extents.len();
        strides[..self.axis]
            .iter()
            .copied()
            .chain(repeat_n(0, self.index_shape.dimensions().len()))
            .chain(strides[end..].iter().copied())
            .collect()
    }

    /// Returns each index tensor's strides placed on the selected value's `rank` dimensions, zero outside the index shape.
    fn index_strides(&self, rank: usize) -> Result<Vec<Vec<i64>>> {
        let after = rank - self.axis - self.index_shape.dimensions().len();
        self.indices
            .iter()
            .map(|(_, layout)| {
                let expanded = layout.expand(self.index_shape)?;
                Ok(repeat_n(0, self.axis)
                    .chain(expanded.strides().iter().copied())
                    .chain(repeat_n(0, after))
                    .collect())
            })
            .collect()
    }

    /// Loads each index tensor from the columns after the first, checks its values against their dimensions, and returns them with the strides of the selected tensor that `strides` gives.
    fn indexed_terms(
        &self,
        builder: &mut DomainKernel,
        strides: &[i64],
    ) -> Result<Vec<(NodeId, SizeId)>> {
        let mut terms = Vec::with_capacity(self.indices.len());
        for (position, (&(data_type, _), &extent)) in
            self.indices.iter().zip(self.extents).enumerate()
        {
            if !matches!(data_type, DataType::I32 | DataType::I64) {
                return Err(Error::unsupported_data_type("cuda index tensor"));
            }

            let access = builder.input(element_type(data_type)?, 1 + position);
            let value = builder.kernel.load(access);
            let value = builder.convert(value, ElementType::I64);
            let extent = builder.size(to_u64(extent, "cuda indexed extent")?);
            let index =
                builder
                    .kernel
                    .check_index(value, extent, self.wrap, STATUS_INDEX_OUT_OF_BOUNDS);
            let stride = builder.size(to_u64(
                strides[self.axis + position],
                "cuda indexed stride",
            )?);
            terms.push((index, stride));
        }
        Ok(terms)
    }
}

/// Lowers reading `input`, of `data_type`, at the elements `selection` picks into a new contiguous output, as `gather`, `index_select`, `embedding`, and `index` compute.
///
/// The kernel copies each element's bits, and an index out of range reports [`STATUS_INDEX_OUT_OF_BOUNDS`].
///
/// # Errors
///
/// Returns an error if the selection does not fit the input, an index tensor is not `I32` or `I64`, or a size does not fit a kernel.
pub fn gather(
    name: &str,
    data_type: DataType,
    input: &Layout,
    selection: &Selection<'_>,
) -> Result<LoweredKernel> {
    let dimensions = selection.dimensions(input.dimensions())?;
    let mut columns = vec![selection.direct_strides(input.strides())];
    columns.extend(selection.index_strides(dimensions.len())?);
    columns.push(
        Layout::contiguous(dimensions.as_slice())?
            .strides()
            .to_vec(),
    );
    let kinds = vec![AxisKind::Parallel; dimensions.len()];
    let mut builder = DomainKernel::create(name, &dimensions, &kinds, &columns)?;
    let bits = bits_type(data_type);
    // The input is the first operand, before the index tensors.
    let direct = builder.input(bits, 0);
    let indirect = selection.indexed_terms(&mut builder, input.strides())?;
    let access = Access { indirect, ..direct };
    let value = builder.kernel.load(access);
    let output = builder.output(bits, columns.len() - 1);
    builder.kernel.store(output, value);
    let schedule = element_schedule(&builder.kernel, 128);
    Ok(builder.finish(
        ValueSpec::new(data_type, Shape::create(dimensions)?),
        schedule,
    ))
}

/// Lowers writing `values` at the elements `selection` picks in a contiguous buffer that `buffer` describes, adding them to the elements there when `accumulate` holds, as `index_put` computes.
///
/// Only the selected elements are written, so the caller fills the buffer first.
/// Plain writes copy bits, and an index out of range reports [`STATUS_INDEX_OUT_OF_BOUNDS`].
///
/// # Errors
///
/// Returns an error if `values` does not have the selected shape, an index tensor is not `I32` or `I64`, the data type cannot accumulate, or a size does not fit a kernel.
pub fn scatter(
    name: &str,
    buffer: ValueSpec,
    values: &Layout,
    selection: &Selection<'_>,
    accumulate: bool,
) -> Result<LoweredKernel> {
    let data_type = buffer.data_type();
    let target = Layout::from_shape(buffer.shape());
    let dimensions = selection.dimensions(target.dimensions())?;
    if values.dimensions() != dimensions.as_slice() {
        return Err(Error::shape_mismatch(
            "cuda index put values",
            dimensions,
            values.dimensions().to_vec(),
        ));
    }

    let mut columns = vec![values.strides().to_vec()];
    columns.extend(selection.index_strides(dimensions.len())?);
    columns.push(selection.direct_strides(target.strides()));
    let kinds = vec![AxisKind::Parallel; dimensions.len()];
    let mut builder = DomainKernel::create(name, &dimensions, &kinds, &columns)?;
    let element = if accumulate {
        if data_type == DataType::F4E2M1 {
            return Err(Error::unsupported_data_type("cuda index put accumulation"));
        }
        element_type(data_type)?
    } else {
        bits_type(data_type)
    };
    let value = builder.input(element, 0);
    let value = builder.kernel.load(value);
    let indirect = selection.indexed_terms(&mut builder, target.strides())?;
    let access = Access {
        indirect,
        ..builder.output(element, columns.len() - 1)
    };
    if accumulate {
        builder.kernel.store_sum(access, value);
    } else {
        builder.kernel.store(access, value);
    }

    let schedule = element_schedule(&builder.kernel, 128);
    Ok(builder.finish(buffer, schedule))
}

/// Returns the unsigned integer type of `data_type`'s size, which moves its elements' bits unchanged.
pub const fn bits_type(data_type: DataType) -> ElementType {
    match data_type.size_bytes() {
        1 => ElementType::U8,
        2 => ElementType::U16,
        4 => ElementType::U32,
        _ => ElementType::U64,
    }
}
