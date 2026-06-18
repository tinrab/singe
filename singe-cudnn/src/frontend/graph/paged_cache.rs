use crate::{
    data_type::DataType,
    error::{Error, Result},
    frontend::{graph::Graph, infer::infer_paged_cache_output, operation::Operation, support},
    tensor::{TensorId, TensorSpec},
    version,
};

impl Graph {
    pub fn paged_cache_load(
        &mut self,
        container: TensorId,
        output: TensorId,
        sequence: TensorId,
        page_table: TensorId,
    ) -> Result<()> {
        self.validate_paged_cache_load_support_surface_for_version(version()?.raw())?;

        let container_tensor = self.tensor_config(container)?.clone();
        let output_tensor = self.tensor_config(output)?.clone();
        let sequence_tensor = self.tensor_config(sequence)?.clone();
        let page_table_tensor = self.tensor_config(page_table)?.clone();

        self.validate_tensor_data_type_supported(
            sequence,
            &[DataType::I32, DataType::I64],
            "paged cache load sequence",
        )?;
        self.validate_tensor_data_type_supported(
            page_table,
            &[DataType::I32, DataType::I64],
            "paged cache load page table",
        )?;
        self.validate_tensor_data_type(
            output,
            self.effective_intermediate_data_type(container_tensor.data_type),
            "paged cache load output",
        )?;

        infer_paged_cache_output(
            &container_tensor.shape,
            &sequence_tensor.shape,
            &page_table_tensor.shape,
            false,
        )?;

        let container_dims = container_tensor.shape.dimensions();
        let sequence_dims = sequence_tensor.shape.dimensions();
        let output_dims = output_tensor.shape.dimensions();
        let output_strides = output_tensor.shape.strides();
        let block_size = container_dims[2];
        let hidden = container_dims[3];
        let block_table_size = page_table_tensor.shape.dimensions()[2];
        let batch = sequence_dims[0];
        let heads = container_dims[1];
        let output_is_transposed = output_strides[2] == 1;
        let sequence_length = if output_is_transposed {
            output_dims[3]
        } else {
            output_dims[2]
        };

        let expected_output_dimensions = if output_is_transposed {
            vec![batch, heads, hidden, sequence_length]
        } else {
            vec![batch, heads, sequence_length, hidden]
        };

        self.validate_tensor_dimensions(
            output,
            &expected_output_dimensions,
            "paged cache load output shape",
        )?;
        let required_block_table_size = {
            let required_block_table_size = sequence_length
                .checked_add(block_size - 1)
                .ok_or_else(|| Error::OutOfRange {
                    name: "paged cache load output shape".into(),
                })?;
            required_block_table_size / block_size
        };
        if page_table_tensor.ragged_offset.is_none()
            && required_block_table_size != block_table_size
        {
            return Err(Error::FrontendTensorDimensionsMismatch {
                tensor_id: page_table,
                operation: "paged cache load page table shape".into(),
                expected: vec![batch, 1, required_block_table_size, 1],
                actual: page_table_tensor.shape.dimensions().to_vec(),
            });
        }

        self.operations.push(Operation::PagedCacheLoad {
            container,
            output,
            sequence,
            page_table,
        });
        Ok(())
    }

    pub fn paged_cache_load_infer(
        &mut self,
        container: TensorId,
        sequence: TensorId,
        page_table: TensorId,
        transposed: bool,
    ) -> Result<TensorId> {
        let checkpoint = self.mutation_checkpoint();
        let container_tensor = self.tensor_config(container)?.clone();
        let sequence_tensor = self.tensor_config(sequence)?.clone();
        let page_table_tensor = self.tensor_config(page_table)?.clone();
        let output_shape = infer_paged_cache_output(
            &container_tensor.shape,
            &sequence_tensor.shape,
            &page_table_tensor.shape,
            transposed,
        )?;
        let output_data_type = self.effective_intermediate_data_type(container_tensor.data_type);
        let output = self.tensor(TensorSpec::new(output_data_type, output_shape).virtual_tensor());
        if let Err(error) = self.paged_cache_load(container, output, sequence, page_table) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(output)
    }

    pub(crate) fn validate_paged_cache_load_support_surface_for_version(
        &self,
        cudnn_version: u64,
    ) -> Result<()> {
        support::PAGED_CACHE_LOAD.require_descriptor_match(cudnn_version)
    }
}
