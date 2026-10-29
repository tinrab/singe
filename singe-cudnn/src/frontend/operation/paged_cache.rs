use serde::{Deserialize, Serialize};

use crate::{
    backend::{
        execution::paged_cache::PagedCacheLoadOperation as BackendPagedCacheLoadOperation,
        tensor::TensorId,
    },
    data_type::DataType,
    error::Result,
    frontend::{
        graph::{Graph, infer_paged_cache_dimensions},
        lower::{LoweredOperation, LoweringContext, tensor_at},
    },
};

/// Logical axes of the reconstructed cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PagedCacheLayout {
    /// `[batch, heads, sequence, hidden]`, as used by the value matmul.
    SequenceHidden,
    /// `[batch, heads, hidden, sequence]`, as used by the key matmul.
    HiddenSequence,
}

/// Diagnostic scope, logical output layout and optional sequence length for paged-cache loads.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PagedCacheLoadConfig {
    #[serde(default)]
    name: Option<String>,
    /// Logical sequence/hidden axis order of the reconstructed cache.
    pub layout: PagedCacheLayout,
    /// Optional logical output sequence length; `None` derives it from explicit output dimensions or page-table capacity.
    pub max_sequence_length: Option<i64>,
}

impl PagedCacheLoadConfig {
    /// Selects the reconstructed cache layout without an explicit sequence length or diagnostic scope.
    #[must_use]
    pub fn new(layout: PagedCacheLayout) -> Self {
        Self {
            name: None,
            layout,
            max_sequence_length: None,
        }
    }

    /// Names inferred results without replacing explicit output tensor names.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Returns the nonempty diagnostic scope, or `None` when unnamed.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref().filter(|name| !name.is_empty())
    }

    /// Sets the logical output length without adding a slice after the load.
    /// Unpacked page tables must contain exactly `ceil(length / block_size)` entries per batch.
    /// Passing `None` derives the length from explicit output dimensions or infers the full page-table capacity.
    /// Retains the selected layout and diagnostic scope.
    #[must_use]
    pub fn with_max_sequence_length(mut self, length: impl Into<Option<i64>>) -> Self {
        self.max_sequence_length = length.into();
        self
    }
}

pub use crate::backend::execution::paged_cache::PagedCacheLoadTensors;

/// A cache load retaining logical geometry and scope independently of physical strides.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PagedCacheLoadOperation {
    /// Logical identities of the container, reconstructed output, sequence extents, and page table.
    pub tensors: PagedCacheLoadTensors<TensorId>,
    /// Retained reconstructed layout, optional sequence length, and diagnostic scope.
    pub config: PagedCacheLoadConfig,
}

impl PagedCacheLoadOperation {
    pub(crate) fn validate_graph(&self, graph: &Graph) -> Result<()> {
        let PagedCacheLoadTensors {
            container,
            output,
            sequence,
            page_table,
        } = self.tensors;
        let container_tensor = graph.tensor_spec(container)?;
        let output_tensor = graph.tensor_spec(output)?;
        graph.validate_tensor_rank(output, 4, "paged cache load output")?;
        let sequence_tensor = graph.tensor_spec(sequence)?;
        let page_table_tensor = graph.tensor_spec(page_table)?;
        graph.validate_tensor_data_type_supported(
            sequence,
            &[DataType::I32, DataType::I64],
            "paged cache load sequence",
        )?;
        graph.validate_tensor_data_type_supported(
            page_table,
            &[DataType::I32, DataType::I64],
            "paged cache load page table",
        )?;
        let sequence_length = match self.config.layout {
            PagedCacheLayout::HiddenSequence => output_tensor.shape.dimensions()[3],
            PagedCacheLayout::SequenceHidden => output_tensor.shape.dimensions()[2],
        };
        let expected = infer_paged_cache_dimensions(
            &container_tensor.shape,
            &sequence_tensor.shape,
            &page_table_tensor.shape,
            self.config.layout,
            Some(self.config.max_sequence_length.unwrap_or(sequence_length)),
        )?;
        graph.validate_tensor_dimensions(output, &expected, "paged cache load output shape")?;
        let block_size = container_tensor.shape.dimensions()[2];
        let required_blocks =
            sequence_length / block_size + i64::from(sequence_length % block_size != 0);
        if page_table_tensor.ragged_offset.is_none() {
            graph.validate_tensor_dimensions(
                page_table,
                &[sequence_tensor.shape.dimensions()[0], 1, required_blocks, 1],
                "paged cache load page table shape",
            )?;
        }
        Ok(())
    }

    pub(crate) fn lower(&self, context: &LoweringContext<'_>) -> Result<LoweredOperation> {
        let tensors = context.backend_tensors();
        Ok(LoweredOperation::PagedCacheLoad(
            BackendPagedCacheLoadOperation::create(
                self.tensors.try_map(|id| tensor_at(tensors, id))?,
            )?,
        ))
    }
}
