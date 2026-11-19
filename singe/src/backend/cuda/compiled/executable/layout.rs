use std::collections::HashMap;

use crate::{
    backend::{
        cuda::{
            compiled::{executable::CudaCompiledGraph, plan::lowering::CudaGraphProvider},
            diagnostics::{graph_layout_name, graph_provider_name},
            graph_binding::operation_dependencies,
            graph_value::CudaGraphValue,
        },
        registry::{LayoutSupport, Operation, OperationSupport, operation_name},
    },
    error::{Error, Result},
    graph::ir::{Graph, ValueId},
    utility::checked_mul_i64,
};

impl CudaCompiledGraph {
    /// Checks cached operand layouts against a provider's graph layout requirement.
    /// Providers requiring contiguous storage reject nonzero offsets, except supported cuBLASLt matrix transpose and batch layouts.
    ///
    /// # Errors
    ///
    /// Returns an error if IR metadata or cached dependencies are missing, or an operand violates the provider's layout requirement.
    pub fn validate_graph_layout_requirements(
        ir: &Graph,
        id: ValueId,
        operation_support: &OperationSupport,
        provider: Option<CudaGraphProvider>,
        cache: &HashMap<ValueId, CudaGraphValue>,
    ) -> Result<()> {
        let value = id;
        let output_data_type = ir.required_value_data_type(value)?;
        let output_shape = ir.required_value_shape(value)?;
        if !matches!(
            operation_support.layout(),
            LayoutSupport::MaterializesToContiguous | LayoutSupport::RequiresContiguous
        ) {
            return Ok(());
        }

        for dependency_id in operation_dependencies(ir, id)? {
            let tensor = cache
                .get(&dependency_id)
                .ok_or_else(|| Error::missing_input("validate cuda graph layout requirements"))?;
            if !tensor.is_contiguous() || tensor.storage_offset() != 0 {
                if provider == Some(CudaGraphProvider::CublasLt)
                    && operation_support.operation() == Operation::MatMul
                    && graph_cublaslt_matmul_layout_supported(tensor)
                {
                    continue;
                }

                let provider = provider.map_or("none", graph_provider_name);
                return Err(Error::unsupported_operation_with_details(
                    operation_name(operation_support.operation()),
                    format!(
                        "node {:?}, provider {provider}, layout {}, cuda graph requires contiguous input tensors, dtype {:?}, shape {:?}, dependency {:?} layout dims {:?}, strides {:?}, offset {}",
                        id,
                        graph_layout_name(operation_support.layout()),
                        output_data_type,
                        output_shape.dimensions(),
                        dependency_id,
                        tensor.shape().dimensions(),
                        tensor.layout().strides(),
                        tensor.storage_offset()
                    ),
                ));
            }
        }

        Ok(())
    }
}

fn graph_cublaslt_matmul_layout_supported(tensor: &CudaGraphValue) -> bool {
    if tensor.storage_offset() != 0 {
        return false;
    }

    let (Some((batch, &[rows, columns])), Some((batch_strides, matrix_strides))) = (
        tensor.shape().dimensions().split_last_chunk::<2>(),
        tensor.layout().strides().split_last_chunk::<2>(),
    ) else {
        return false;
    };
    if batch.len() != batch_strides.len() {
        return false;
    }

    let Ok(mut expected_stride) =
        checked_mul_i64(rows, columns, "cuda graph cuBLASLt matrix element count")
    else {
        return false;
    };
    for (&dimension, &stride) in batch.iter().zip(batch_strides).rev() {
        if stride != expected_stride {
            return false;
        }

        let Ok(next_stride) = checked_mul_i64(
            expected_stride,
            dimension,
            "cuda graph cuBLASLt batch stride",
        ) else {
            return false;
        };
        expected_stride = next_stride;
    }

    *matrix_strides == [columns, 1] || *matrix_strides == [1, rows]
}
