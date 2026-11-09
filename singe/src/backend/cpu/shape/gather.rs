use std::sync::Arc;

use crate::{
    backend::{
        cpu::iteration::typed_values,
        index_problem::{indexing_problem, normalized_indices},
        registry::{Operation, OperationAttributes},
    },
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    tensor::{Tensor, tensor_data_type_cases},
};

pub fn select(operation: Operation, input: &Tensor, indices: &Tensor, dim: i64) -> Result<Tensor> {
    // Selection copies elements unchanged, so 8-bit floats move as their `U8` bits.
    if matches!(input.data_type(), DataType::F8E4M3 | DataType::F8E5M2) {
        let bits = Tensor::create_from_storage(
            Arc::clone(input.storage()),
            DataType::U8,
            input.layout().clone(),
        )?;
        let selected = select(operation, &bits, indices, dim)?;
        return Tensor::create_from_storage(
            Arc::clone(selected.storage()),
            input.data_type(),
            selected.layout().clone(),
        );
    }

    for tensor in [input, indices] {
        if tensor.device() != Device::Cpu {
            return Err(Error::device_mismatch(
                "cpu indexing operand",
                Device::Cpu,
                tensor.device(),
            ));
        }
    }

    let attributes = if operation == Operation::Embedding {
        OperationAttributes::None
    } else {
        OperationAttributes::IndexSelect(dim)
    };
    let (problem, bounds) =
        indexing_problem(operation, &attributes, input.shape(), indices.shape())?;
    let indices = normalized_indices(indices, problem.selected_dimension, bounds)?;
    if problem.output_shape.element_count() == 0 {
        return Tensor::from_bytes(Device::Cpu, input.data_type(), problem.output_shape, &[]);
    }

    macro_rules! gather_type {
        ($($ty:ty => $variant:ident = $one:expr),+ $(,)?) => {
            match input.data_type() {
                $(DataType::$variant => {
                    let values = typed_values::<$ty>(input)?;
                    let mut output = Vec::with_capacity(problem.output_shape.element_count());
                    for outer in 0..problem.outer_size {
                        for &selected in &indices {
                            for inner in 0..problem.inner_size {
                                // Checked input/output shapes bound this flattened logical address; the layout validates its allocation-relative span.
                                let linear = (outer * problem.selected_dimension + selected) * problem.inner_size + inner;
                                output.push(values[input.layout().linear_offset(linear)]);
                            }
                        }
                    }
                    drop(values);
                    Tensor::from_vec(Device::Cpu, problem.output_shape, output)
                },)+
                _ => Err(Error::unsupported_data_type("cpu indexing input")),
            }
        };
    }
    tensor_data_type_cases!(gather_type)
}
