use crate::{
    backend::tensor::{Shape, TensorId},
    error::{Error, Result},
    frontend::{
        graph::{Graph, TensorOutputConfig},
        operation::{
            Operation, PagedCacheLayout, PagedCacheLoadConfig, PagedCacheLoadOperation,
            PagedCacheLoadTensors,
        },
    },
    utility::check_range,
};

const PAGED_CACHE_LOAD_RANK: &str = "paged cache load rank";
const PAGED_CACHE_LOAD_SHAPE: &str = "paged cache load shape";

impl Graph {
    /// Reconstructs a cache using caller-declared dimensions, strides, encoding and placement.
    /// Configuration describes logical axes independently of the declared physical strides.
    /// An explicit sequence length must match the output; otherwise the output supplies that length.
    /// Unpacked page tables must match the rounded-up logical output length.
    /// Packed page tables retain the C++ frontend's ragged-offset exception.
    /// Rank-one sequence lengths and their dynamic bounds are promoted to `[B, 1, 1, 1]` after validation succeeds.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensor identities, unsupported sequence or page-table integer encodings, incompatible container, sequence, page-table or output geometry, an invalid sequence length, overflowing inferred extents, or failure to promote sequence metadata and its runtime bounds.
    pub fn paged_cache_load_into(
        &mut self,
        tensors: PagedCacheLoadTensors<TensorId>,
        config: PagedCacheLoadConfig,
    ) -> Result<()> {
        let operation = PagedCacheLoadOperation { tensors, config };
        operation.validate_graph(self)?;
        self.promote_index_tensors([tensors.sequence])?;
        self.operations.push(Operation::PagedCacheLoad(operation));
        Ok(())
    }

    /// Creates a reconstructed cache with independently selected placement and encoding.
    /// Omitted encoding follows I/O or intermediate policy, falling back to the container's type.
    /// Explicit encoding overrides the policy; native support determines compatible combinations.
    /// An explicit logical sequence length can end inside the final page.
    /// Sequence lengths may be rank-one or use the native `[B, 1, 1, 1]` layout.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensor identities, unsupported sequence or page-table integer encodings, incompatible container, sequence, page-table or output geometry, an invalid sequence length, overflowing inferred extents, or failure to promote sequence metadata and its runtime bounds.
    pub fn paged_cache_load(
        &mut self,
        container: TensorId,
        sequence: TensorId,
        page_table: TensorId,
        config: PagedCacheLoadConfig,
        output: TensorOutputConfig,
    ) -> Result<TensorId> {
        let container_tensor = self.tensor_spec(container)?;
        let sequence_tensor = self.tensor_spec(sequence)?;
        let page_table_tensor = self.tensor_spec(page_table)?;
        let output_shape = infer_paged_cache_output(
            &container_tensor.shape,
            &sequence_tensor.shape,
            &page_table_tensor.shape,
            &config,
        )?;
        let output_spec = output.into_spec(
            self.config.data_type_policy,
            container_tensor.data_type,
            output_shape,
        );
        let checkpoint = self.mutation_checkpoint();
        let output = self.inferred_tensor(output_spec, config.name(), "yOut");
        if let Err(error) = self.paged_cache_load_into(
            PagedCacheLoadTensors::new(container, output, sequence, page_table),
            config,
        ) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(output)
    }
}

fn shape_rank_mismatch(name: &'static str, expected: usize, actual: usize) -> Error {
    Error::LengthMismatch {
        name: name.into(),
        expected,
        actual,
    }
}

pub fn infer_paged_cache_dimensions(
    container: &Shape,
    sequence: &Shape,
    page_table: &Shape,
    layout: PagedCacheLayout,
    max_sequence_length: Option<i64>,
) -> Result<[i64; 4]> {
    for shape in [container, page_table] {
        if shape.rank() != 4 {
            return Err(shape_rank_mismatch(PAGED_CACHE_LOAD_RANK, 4, shape.rank()));
        }
    }

    let container_dims = container.dimensions();
    let sequence_dims = sequence.dimensions();
    let page_table_dims = page_table.dimensions();
    if !matches!(sequence_dims.len(), 1 | 4) {
        return Err(shape_rank_mismatch(
            PAGED_CACHE_LOAD_RANK,
            4,
            sequence_dims.len(),
        ));
    }

    if sequence_dims[0] != page_table_dims[0] || page_table_dims[1] != 1 || page_table_dims[3] != 1
    {
        return Err(Error::ShapeMismatch {
            name: PAGED_CACHE_LOAD_SHAPE.into(),
            expected: sequence_dims.to_vec(),
            actual: page_table_dims.to_vec(),
        });
    }
    if sequence_dims
        .iter()
        .skip(1)
        .any(|&dimension| dimension != 1)
    {
        return Err(Error::ShapeMismatch {
            name: "paged cache load sequence shape".into(),
            expected: vec![sequence_dims[0], 1, 1, 1],
            actual: sequence_dims.to_vec(),
        });
    }

    let batch = sequence_dims[0];
    let heads = container_dims[1];
    let block_size = container_dims[2];
    check_range!("paged cache block size", block_size > 0)?;
    let hidden = container_dims[3];
    let blocks = page_table_dims[2];
    let sequence_length = match max_sequence_length {
        Some(length) => {
            check_range!("paged cache sequence length", length > 0)?;
            length
        }
        None => blocks
            .checked_mul(block_size)
            .ok_or_else(|| Error::out_of_range("paged cache sequence length"))?,
    };

    Ok(match layout {
        PagedCacheLayout::HiddenSequence => [batch, heads, hidden, sequence_length],
        PagedCacheLayout::SequenceHidden => [batch, heads, sequence_length, hidden],
    })
}

pub fn infer_paged_cache_output(
    container: &Shape,
    sequence: &Shape,
    page_table: &Shape,
    config: &PagedCacheLoadConfig,
) -> Result<Shape> {
    let dimensions = infer_paged_cache_dimensions(
        container,
        sequence,
        page_table,
        config.layout,
        config.max_sequence_length,
    )?;
    let order = match config.layout {
        PagedCacheLayout::SequenceHidden => [3, 2, 1, 0],
        PagedCacheLayout::HiddenSequence => [2, 3, 1, 0],
    };
    Shape::packed(dimensions, order)
}
