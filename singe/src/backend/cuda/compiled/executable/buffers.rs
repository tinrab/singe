use std::{
    collections::HashMap,
    sync::{Arc, Weak},
};

use crate::{
    backend::cuda::graph_value::{CudaGraphValue, CudaGraphValueInner},
    data_type::DataType,
    error::{Error, Result},
    graph::ir::ValueId,
    layout::Layout,
};

pub fn output_buffer_summaries(
    values: &HashMap<ValueId, CudaGraphValue>,
    outputs: &[ValueId],
) -> Result<Vec<GraphOutputBufferSummary>> {
    output_value_tensors(values, outputs)?
        .into_iter()
        .map(|(value, tensor)| {
            Ok(GraphOutputBufferSummary::new(
                value,
                tensor.data_type(),
                tensor.layout().clone(),
                tensor.byte_size()?,
            ))
        })
        .collect()
}

pub fn output_buffer_allocation_records(
    values: &HashMap<ValueId, CudaGraphValue>,
    outputs: &[ValueId],
) -> Result<Vec<GraphOutputBufferAllocation>> {
    output_value_tensors(values, outputs)?
        .into_iter()
        .map(|(value, tensor)| {
            Ok(GraphOutputBufferAllocation::new(
                value,
                tensor.data_type(),
                tensor.layout().clone(),
                tensor.byte_size()?,
                tensor,
            ))
        })
        .collect()
}

pub fn output_value_tensors<'a>(
    values: &'a HashMap<ValueId, CudaGraphValue>,
    outputs: &[ValueId],
) -> Result<Vec<(ValueId, &'a CudaGraphValue)>> {
    outputs
        .iter()
        .map(|&value| {
            Ok((
                value,
                values
                    .get(&value)
                    .ok_or_else(|| Error::missing_output("output value tensors"))?,
            ))
        })
        .collect()
}

/// Allocation-compatible graph output buffer summary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphOutputBufferSummary {
    /// Graph IR value produced as a graph output.
    value: ValueId,
    /// `CudaGraphValue` element data type required by the output buffer.
    data_type: DataType,
    /// `CudaGraphValue` layout required by the output buffer.
    layout: Layout,
    /// Output buffer byte size.
    bytes: usize,
}

impl GraphOutputBufferSummary {
    /// Stores output identity, required data type and layout, and the supplied logical byte count without validation.
    #[must_use]
    pub fn new(value: ValueId, data_type: DataType, layout: Layout, bytes: usize) -> Self {
        Self {
            value,
            data_type,
            layout,
            bytes,
        }
    }

    /// Returns the graph IR value produced as a graph output.
    #[must_use]
    pub fn value(&self) -> ValueId {
        self.value
    }

    /// Returns the tensor element data type required by this output buffer.
    #[must_use]
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    /// Returns the tensor layout required by this output buffer.
    #[must_use]
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Returns the output buffer byte size.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

/// Output buffer metadata with a weak reference to the graph value that owns its allocation.
/// This descriptor does not keep that value or its allocation alive.
#[derive(Debug, Clone)]
pub struct GraphOutputBufferAllocation {
    value: ValueId,
    data_type: DataType,
    layout: Layout,
    bytes: usize,
    tensor: Weak<CudaGraphValueInner>,
}

impl GraphOutputBufferAllocation {
    /// Stores output metadata and a weak reference to `tensor` without validating that they agree.
    #[must_use]
    pub fn new(
        value: ValueId,
        data_type: DataType,
        layout: Layout,
        bytes: usize,
        tensor: &CudaGraphValue,
    ) -> Self {
        Self {
            value,
            data_type,
            layout,
            bytes,
            tensor: Arc::downgrade(&tensor.inner),
        }
    }

    /// Returns the output IR value ID.
    #[must_use]
    pub fn value(&self) -> ValueId {
        self.value
    }

    /// Returns the recorded output element data type.
    #[must_use]
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    /// Returns the recorded output layout.
    #[must_use]
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Returns the recorded logical output byte count.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Upgrades the weak reference to a retained graph value.
    /// Returns `None` if no strong owner of that value remains.
    #[must_use]
    pub fn tensor(&self) -> Option<CudaGraphValue> {
        self.tensor.upgrade().map(|inner| CudaGraphValue { inner })
    }
}

/// Allocation-compatible graph-private reusable buffer summary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphReusableBufferSummary {
    /// `CudaGraphValue` element data type required by the reusable buffer.
    data_type: DataType,
    /// `CudaGraphValue` layout required by the reusable buffer.
    layout: Layout,
    /// Number of retained reusable buffers with this allocation key.
    entries: usize,
    /// Total retained bytes across buffers with this allocation key.
    bytes: usize,
}

impl GraphReusableBufferSummary {
    /// Stores a reusable buffer bucket's data type, layout, entry count, and total byte count without validation.
    #[must_use]
    pub fn new(data_type: DataType, layout: Layout, entries: usize, bytes: usize) -> Self {
        Self {
            data_type,
            layout,
            entries,
            bytes,
        }
    }

    /// Returns the tensor element data type required by this reusable buffer key.
    #[must_use]
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    /// Returns the tensor layout required by this reusable buffer key.
    #[must_use]
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Returns retained reusable buffers with this allocation key.
    #[must_use]
    pub fn entries(&self) -> usize {
        self.entries
    }

    /// Returns total retained bytes across buffers with this allocation key.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

/// Concrete graph-private reusable work-buffer allocation retained by a compiled graph.
#[derive(Debug, Clone)]
pub struct GraphReusableBufferAllocation {
    data_type: DataType,
    layout: Layout,
    bytes: usize,
    tensor: CudaGraphValue,
}

impl GraphReusableBufferAllocation {
    /// Retains `tensor` with the supplied reusable buffer metadata without validating that they agree.
    #[must_use]
    pub fn new(data_type: DataType, layout: Layout, bytes: usize, tensor: CudaGraphValue) -> Self {
        Self {
            data_type,
            layout,
            bytes,
            tensor,
        }
    }

    /// Returns the recorded buffer element data type.
    #[must_use]
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    /// Returns the recorded buffer layout.
    #[must_use]
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Returns the recorded buffer byte count.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Returns the retained graph value that owns the reusable buffer.
    #[must_use]
    pub fn tensor(&self) -> &CudaGraphValue {
        &self.tensor
    }
}
