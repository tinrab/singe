use singe_onnx::{Node, TensorDataType};

use crate::{
    data_type::DataType,
    onnx::{
        attributes::Attributes,
        error::{Error, Result},
        importer::Importer,
        utility::node_label,
    },
    shape::{Shape, normalize_dim_usize, transform::normalize_dim},
    tensor::Tensor,
};

pub fn axes_from_attribute_or_input(
    importer: &Importer,
    node: &Node,
    attribute: &str,
    input_index: usize,
) -> Result<Option<Vec<i64>>> {
    let attrs = Attributes::create(node)?;
    if let Some(axes) = attrs.ints(attribute)? {
        return Ok(Some(axes));
    }

    let Some(name) = node.inputs.get(input_index) else {
        return Ok(None);
    };
    if name.is_empty() {
        return Ok(None);
    }
    Ok(Some(shape_input_values(
        importer,
        node,
        input_index,
        attribute,
    )?))
}

pub fn normalize_shape_bound(value: i64, rank: i64) -> i64 {
    if value < 0 {
        (rank + value).clamp(0, rank)
    } else {
        value.min(rank)
    }
}

pub fn normalize_slice_index(value: i64, dimension: i64) -> i64 {
    if value < 0 {
        (dimension + value).clamp(0, dimension)
    } else {
        value.min(dimension)
    }
}

pub fn normalized_axes(axes: &[i64], rank: i64) -> Result<Vec<i64>> {
    let mut normalized = axes
        .iter()
        .map(|axis| normalize_dim(*axis, rank).map_err(Into::into))
        .collect::<Result<Vec<_>>>()?;
    normalized.sort_unstable();
    normalized.dedup();
    if normalized.len() != axes.len() {
        return Err(Error::UnsupportedShape {
            name: "axes".into(),
            reason: "axes must not contain duplicates".into(),
        });
    }
    Ok(normalized)
}

pub fn normalized_unsqueeze_axes(
    axes: &[i64],
    input_rank: i64,
    axis_count: usize,
) -> Result<Vec<i64>> {
    let output_rank = input_rank
        .checked_add(
            i64::try_from(axis_count).map_err(|_| Error::UnsupportedShape {
                name: "Unsqueeze".into(),
                reason: "too many axes".into(),
            })?,
        )
        .ok_or_else(|| Error::UnsupportedShape {
            name: "Unsqueeze".into(),
            reason: "output rank overflow".into(),
        })?;
    normalized_axes(axes, output_rank)
}

pub fn expand_output_shape(name: &str, input: &Shape, mut target: Vec<i64>) -> Result<Shape> {
    let input_dimensions = input.dimensions();
    let input_rank = input_dimensions.len();
    if target.len() < input_rank {
        let mut dimensions = input_dimensions[..input_rank - target.len()].to_vec();
        dimensions.append(&mut target);
        target = dimensions;
    }

    let target_rank = target.len();
    let input_offset =
        target_rank
            .checked_sub(input_rank)
            .ok_or_else(|| Error::UnsupportedShape {
                name: name.into(),
                reason: "expand target rank is smaller than input rank".into(),
            })?;
    for (axis, dimension) in target.iter_mut().enumerate() {
        if *dimension == -1 {
            if axis < input_offset {
                return Err(Error::UnsupportedShape {
                    name: name.into(),
                    reason: "expand -1 target dimension cannot create a leading axis".into(),
                });
            }

            *dimension = input_dimensions[axis - input_offset];
        }

        if *dimension < 0 {
            return Err(Error::UnsupportedShape {
                name: name.into(),
                reason: "expand target dimensions must be non-negative".into(),
            });
        }
        if axis >= input_offset {
            let input_dimension = input_dimensions[axis - input_offset];
            if input_dimension != 1 && input_dimension != *dimension {
                return Err(Error::UnsupportedShape {
                    name: name.into(),
                    reason: "expand target is not broadcast-compatible with input".into(),
                });
            }
        }
    }
    Shape::create(target).map_err(Into::into)
}

pub fn shape_input_values(
    importer: &Importer,
    node: &Node,
    index: usize,
    label: &str,
) -> Result<Vec<i64>> {
    let (input_name, tensor) = importer.required_static_input_tensor(node, index, label)?;
    if tensor.shape().rank() != 1 {
        return Err(Error::invalid_initializer(
            &input_name,
            "shape and index operands must have rank one",
        ));
    }
    static_i64_values(&input_name, &tensor)
}

fn static_i64_values(name: &str, tensor: &Tensor) -> Result<Vec<i64>> {
    match tensor.data_type() {
        DataType::I64 => tensor.to_vec::<i64>().map_err(Into::into),
        data_type => Err(Error::UnsupportedType {
            name: name.into(),
            data_type: TensorDataType::Int64,
            reason: format!("shape tensor must be int64, got {data_type}"),
        }),
    }
}

pub fn shape_from_static_dimensions(dimensions: Vec<i64>) -> Result<Shape> {
    if dimensions.is_empty() {
        Ok(Shape::scalar())
    } else {
        Shape::create(dimensions).map_err(Into::into)
    }
}

pub fn onnx_padding(name: &str, rank: i64, pads: &[i64]) -> Result<Vec<(i64, i64)>> {
    let rank = usize::try_from(rank).map_err(|_| Error::UnsupportedShape {
        name: name.into(),
        reason: format!("invalid rank {rank}"),
    })?;
    let expected = rank.checked_mul(2).ok_or_else(|| Error::UnsupportedShape {
        name: name.into(),
        reason: "pad vector rank overflow".into(),
    })?;
    if pads.len() != expected {
        return Err(Error::UnsupportedShape {
            name: name.into(),
            reason: "pad vector must contain begin and end padding for every input axis".into(),
        });
    }

    let (begins, ends) = pads.split_at(rank);
    begins
        .iter()
        .copied()
        .zip(ends.iter().copied())
        .map(|(before, after)| {
            if before < 0 || after < 0 {
                return Err(Error::UnsupportedShape {
                    name: name.into(),
                    reason: "negative padding is not supported".into(),
                });
            }
            Ok((before, after))
        })
        .collect()
}

pub fn padded_shape(name: &str, input: &Shape, padding: &[(i64, i64)]) -> Result<Shape> {
    let dimensions = input
        .dimensions()
        .iter()
        .copied()
        .zip(padding.iter().copied())
        .map(|(dimension, (before, after))| {
            dimension
                .checked_add(before)
                .and_then(|value| value.checked_add(after))
                .filter(|value| *value >= 0)
                .ok_or_else(|| Error::UnsupportedShape {
                    name: name.into(),
                    reason: "padded output dimension is invalid".into(),
                })
        })
        .collect::<Result<Vec<_>>>()?;
    Shape::create(dimensions).map_err(Into::into)
}

pub fn validate_permutation(node: &Node, order: &[i64], rank: i64) -> Result<()> {
    let rank_usize = usize::try_from(rank).map_err(|_| Error::UnsupportedShape {
        name: node_label(node),
        reason: "permutation rank is out of range".into(),
    })?;
    if order.len() != rank_usize {
        return Err(Error::invalid_attribute(
            node_label(node),
            "perm",
            format!("expected {rank} axes, got {}", order.len()),
        ));
    }

    let mut seen = vec![false; rank_usize];
    for &axis in order {
        let axis = normalize_dim_usize(axis, rank).map_err(|_| {
            Error::invalid_attribute(
                node_label(node),
                "perm",
                format!("axis {axis} is outside input rank {rank}"),
            )
        })?;
        if seen[axis] {
            return Err(Error::invalid_attribute(
                node_label(node),
                "perm",
                "axes must not contain duplicates",
            ));
        }

        seen[axis] = true;
    }
    Ok(())
}
