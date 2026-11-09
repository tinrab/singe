use crate::{
    backend::spec,
    error::{Error, Result},
    shape::Shape,
    utility::to_usize,
};

pub fn view_start_offset_delta(start: i64, stride: i64, base_offset: usize) -> Result<usize> {
    if start < 0 || stride < 0 {
        return Err(Error::invalid_argument("graph view offset"));
    }

    let delta = start
        .checked_mul(stride)
        .ok_or_else(|| Error::out_of_range("graph view offset"))
        .and_then(|value| to_usize(value, "graph view offset"))?;
    base_offset
        .checked_add(delta)
        .ok_or_else(|| Error::out_of_range("graph view offset"))
}

pub fn view_flatten_output_shape(
    input_shape: &Shape,
    start_dim: i64,
    end_dim: i64,
) -> Result<Shape> {
    Ok(spec::flatten_problem(input_shape, start_dim, end_dim, "graph view flatten")?.output_shape)
}

pub fn view_expand_output_shape(input_shape: &Shape, output_shape: &Shape) -> Result<Shape> {
    Ok(spec::expand_problem(input_shape, output_shape, "graph view expand")?.output_shape)
}
