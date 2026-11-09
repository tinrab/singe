use std::fmt::{self, Display, Formatter};

use crate::{data_type::DataType, error::Error, tensor::Tensor};

impl Display for Tensor {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let data_type = self.data_type();
        let shape = self.shape();
        let dims = shape.dimensions();
        let rank = dims.len();

        if shape.element_count() == 0 {
            return write!(f, "tensor([], shape={}, device={:?})", shape, self.device());
        }

        let is_small = self.device().is_cpu() && shape.element_count() <= 100 && rank <= 3;
        if is_small {
            let result = match data_type {
                DataType::Bool => self
                    .to_vec::<bool>()
                    .map(|data| format_tensor_recursive(f, &data, dims, 0, 0)),
                DataType::U8 => self
                    .to_vec::<u8>()
                    .map(|data| format_tensor_recursive(f, &data, dims, 0, 0)),
                DataType::I8 => self
                    .to_vec::<i8>()
                    .map(|data| format_tensor_recursive(f, &data, dims, 0, 0)),
                DataType::U16 => self
                    .to_vec::<u16>()
                    .map(|data| format_tensor_recursive(f, &data, dims, 0, 0)),
                DataType::I16 => self
                    .to_vec::<i16>()
                    .map(|data| format_tensor_recursive(f, &data, dims, 0, 0)),
                DataType::U32 => self
                    .to_vec::<u32>()
                    .map(|data| format_tensor_recursive(f, &data, dims, 0, 0)),
                DataType::I32 => self
                    .to_vec::<i32>()
                    .map(|data| format_tensor_recursive(f, &data, dims, 0, 0)),
                DataType::U64 => self
                    .to_vec::<u64>()
                    .map(|data| format_tensor_recursive(f, &data, dims, 0, 0)),
                DataType::I64 => self
                    .to_vec::<i64>()
                    .map(|data| format_tensor_recursive(f, &data, dims, 0, 0)),
                DataType::F32 => self
                    .to_vec::<f32>()
                    .map(|data| format_tensor_recursive(f, &data, dims, 0, 0)),
                DataType::F64 => self
                    .to_vec::<f64>()
                    .map(|data| format_tensor_recursive(f, &data, dims, 0, 0)),
                DataType::F16
                | DataType::Bf16
                | DataType::F8E4M3
                | DataType::F8E5M2
                | DataType::F4E2M1 => Err(Error::unsupported_data_type("tensor display")),
            };
            if let Ok(result) = result {
                return result;
            }
        }

        write!(
            f,
            "tensor(shape={}, device={:?}, dtype={})",
            shape,
            self.device(),
            data_type,
        )
    }
}

pub trait TensorDisplayValue {
    fn fmt_tensor_value(&self, f: &mut Formatter<'_>) -> fmt::Result;
}

impl TensorDisplayValue for f32 {
    fn fmt_tensor_value(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{self:.4}")
    }
}

impl TensorDisplayValue for f64 {
    fn fmt_tensor_value(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{self:.4}")
    }
}

macro_rules! impl_tensor_display_value {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl TensorDisplayValue for $ty {
                fn fmt_tensor_value(&self, f: &mut Formatter<'_>) -> fmt::Result {
                    write!(f, "{self}")
                }
            }
        )+
    };
}

impl_tensor_display_value!(bool, u8, i8, u16, i16, u32, i32, u64, i64);

pub fn format_tensor_recursive<T: TensorDisplayValue>(
    f: &mut Formatter<'_>,
    data: &[T],
    dimensions: &[i64],
    dimension_index: usize,
    mut offset: usize,
) -> fmt::Result {
    if dimensions.is_empty() {
        if let Some(val) = data.get(offset) {
            val.fmt_tensor_value(f)?;
        } else {
            write!(f, "???")?;
        }
        return Ok(());
    }

    let Ok(current_dim) = usize::try_from(dimensions[0]) else {
        return write!(f, "???");
    };
    let remaining_dims = &dimensions[1..];

    write!(f, "[")?;

    let Some(inner_dim_size) = dimensions_element_count(remaining_dims) else {
        return write!(f, "???");
    };

    let max_items = 10;
    let half = max_items / 2;

    for i in 0..current_dim {
        if i == half && current_dim > max_items {
            write!(f, "... ")?;
            let skipped = (current_dim - max_items).checked_mul(inner_dim_size);
            let Some(new_offset) = skipped.and_then(|skipped| offset.checked_add(skipped)) else {
                return write!(f, "???");
            };
            offset = new_offset;
            continue;
        }

        if dimension_index == 0 && dimensions.len() > 1 {
            write!(f, "\n  ")?;
            for _ in 0..=dimension_index {
                write!(f, " ")?;
            }
        }

        format_tensor_recursive(f, data, remaining_dims, dimension_index + 1, offset)?;
        let Some(new_offset) = offset.checked_add(inner_dim_size) else {
            return write!(f, "???");
        };
        offset = new_offset;

        if i < current_dim - 1 {
            if remaining_dims.is_empty() {
                write!(f, ", ")?;
            } else {
                write!(f, ",")?;
            }
        }
    }

    if dimension_index == 0 && dimensions.len() > 1 {
        writeln!(f)?;
    }

    write!(f, "]")?;
    Ok(())
}

fn dimensions_element_count(dimensions: &[i64]) -> Option<usize> {
    dimensions.iter().try_fold(1usize, |acc, &dimension| {
        let dimension = usize::try_from(dimension).ok()?;
        acc.checked_mul(dimension)
    })
}
