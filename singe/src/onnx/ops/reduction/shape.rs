use crate::{
    onnx::{
        error::{Error, Result},
        ops::normalized_axes,
    },
    shape::Shape,
};

pub fn reduction_shape(input: &Shape, axes: &[i64], keepdim: bool) -> Result<Shape> {
    let axes = normalized_axes(axes, input.rank())?;
    let mut dimensions = input.dimensions().to_vec();
    if keepdim {
        for axis in axes {
            dimensions[reduction_axis_to_usize(axis)?] = 1;
        }
    } else {
        for axis in axes.into_iter().rev() {
            dimensions.remove(reduction_axis_to_usize(axis)?);
        }
    }
    Shape::create(dimensions).map_err(Into::into)
}

pub fn reduction_axis_to_usize(axis: i64) -> Result<usize> {
    usize::try_from(axis).map_err(|_| Error::UnsupportedShape {
        name: "reduction axes".into(),
        reason: "normalized axis is out of range".into(),
    })
}
