//! Output shapes and layouts of view operations, from the operation and its attributes alone.

use crate::{
    backend::registry::{Operation, OperationAttributes},
    error::{Error, Result},
    graph::operation::view_shape::{view_expand_output_shape, view_flatten_output_shape},
    layout::Layout,
    shape::Shape,
};

/// Returns the layout a view `operation` with `attributes` gives an input with layout `input`.
///
/// Operations that are not views keep the input layout.
///
/// # Errors
///
/// Returns an error if the attributes do not match the operation or the view is invalid for `input`.
pub fn view_output_layout(
    operation: Operation,
    attributes: &OperationAttributes,
    input: &Layout,
) -> Result<Layout> {
    match operation {
        Operation::Permute => input.permute(
            attributes
                .permutation_order()
                .ok_or_else(|| Error::missing_node("view output layout"))?,
        ),
        Operation::Transpose => {
            let transpose = attributes
                .transpose()
                .ok_or_else(|| Error::missing_node("view output layout"))?;
            input.transpose(transpose.dim0, transpose.dim1)
        }
        Operation::Squeeze => input.squeeze(
            attributes
                .dimension()
                .ok_or_else(|| Error::missing_node("view output layout"))?,
        ),
        Operation::Unsqueeze => input.unsqueeze(
            attributes
                .dimension()
                .ok_or_else(|| Error::missing_node("view output layout"))?,
        ),
        Operation::Expand => input.expand(
            attributes
                .shape()
                .ok_or_else(|| Error::missing_node("view output layout"))?,
        ),
        Operation::Slice => {
            let slice = attributes
                .slice()
                .ok_or_else(|| Error::missing_node("view output layout"))?;
            input.slice(slice.dim, slice.start, slice.length, slice.step)
        }
        Operation::Reshape | Operation::Flatten => {
            let shape = view_output_shape(operation, attributes, input.shape())?;
            Ok(input
                .reshape_view(&shape)?
                .unwrap_or_else(|| Layout::from_shape(&shape)))
        }
        Operation::Contiguous => Ok(Layout::from_shape(input.shape())),
        Operation::AsStrided => {
            let strided = attributes
                .strided()
                .ok_or_else(|| Error::missing_node("view output layout"))?;
            if input.is_contiguous() {
                strided.layout(input.storage_offset())
            } else {
                Ok(Layout::from_shape(&Shape::create(&strided.dimensions)?))
            }
        }
        _ => Ok(input.clone()),
    }
}

/// Returns the shape a view `operation` with `attributes` gives an input of shape `input`.
///
/// Operations that are not views, and views whose attributes do not match, keep the input shape.
///
/// # Errors
///
/// Returns an error if a reshape or expand has no target shape, or the view is invalid for `input`.
pub fn view_output_shape(
    operation: Operation,
    attributes: &OperationAttributes,
    input: &Shape,
) -> Result<Shape> {
    match operation {
        Operation::Transpose => attributes.transpose().map_or_else(
            || Ok(input.clone()),
            |transpose| input.transpose(transpose.dim0, transpose.dim1),
        ),
        Operation::Permute => attributes
            .permutation_order()
            .map_or_else(|| Ok(input.clone()), |order| input.permute(order)),
        Operation::Reshape => attributes
            .shape()
            .cloned()
            .ok_or_else(|| Error::invalid_argument("graph reshape attributes")),
        Operation::Expand => attributes
            .shape()
            .map(|shape| view_expand_output_shape(input, shape))
            .transpose()?
            .ok_or_else(|| Error::invalid_argument("graph expand attributes")),
        Operation::Squeeze => attributes
            .dimension()
            .map_or_else(|| Ok(input.clone()), |dim| input.squeeze(dim)),
        Operation::Unsqueeze => attributes
            .dimension()
            .map_or_else(|| Ok(input.clone()), |dim| input.unsqueeze(dim)),
        Operation::Slice => {
            let slice = attributes
                .slice()
                .ok_or_else(|| Error::missing_node("view output shape"))?;
            Ok(Layout::from_shape(input)
                .slice(slice.dim, slice.start, slice.length, slice.step)?
                .shape()
                .clone())
        }
        Operation::Flatten => attributes.flatten().map_or_else(
            || Ok(input.clone()),
            |flatten| view_flatten_output_shape(input, flatten.start_dim, flatten.end_dim),
        ),
        _ => Ok(input.clone()),
    }
}
