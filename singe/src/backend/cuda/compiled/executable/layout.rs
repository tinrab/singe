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
    utility::{checked_mul_i64, checked_product_i64},
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

    let dimensions = tensor.shape().dimensions();
    let strides = tensor.layout().strides();
    match dimensions.len() {
        2 => strides == [dimensions[1], 1] || strides == [1, dimensions[0]],
        rank if rank > 2 => {
            let matrix_offset = rank - 2;
            let Ok(matrix_element_count) = checked_product_i64(
                &dimensions[matrix_offset..],
                "cuda graph cuBLASLt matrix element count",
            ) else {
                return false;
            };
            let mut expected_stride = matrix_element_count;
            for index in (0..matrix_offset).rev() {
                if strides[index] != expected_stride {
                    return false;
                }

                let Ok(next_stride) = checked_mul_i64(
                    expected_stride,
                    dimensions[index],
                    "cuda graph cuBLASLt batch stride",
                ) else {
                    return false;
                };
                expected_stride = next_stride;
            }

            let matrix_strides = &strides[matrix_offset..];
            matrix_strides == [dimensions[matrix_offset + 1], 1]
                || matrix_strides == [1, dimensions[matrix_offset]]
        }
        _ => false,
    }
}
