use crate::{
    backend::{
        registry::{
            OperationAttributes, OperationInference, StridedAttributes,
            infer::{InferenceContext, require_indices},
        },
        spec::{self, MultiIndexProblem},
    },
    data_type::DataType,
    error::{Error, Result},
    layout::Layout,
    value::ValueSpec,
};

/// Writes a source of the destination's shape and data type, returning no results.
pub fn assign(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let destination = context.first();
    let source = context.input(1)?;
    if source.shape() != destination.shape() {
        return Err(Error::shape_mismatch(
            "assign source",
            destination.shape().dimensions(),
            source.shape().dimensions(),
        ));
    }

    if source.data_type() != destination.data_type() {
        return Err(Error::data_type_mismatch(
            "assign source",
            destination.data_type(),
            source.data_type(),
        ));
    }
    Ok(OperationInference {
        outputs: Vec::new(),
        aliases: Vec::new(),
        computation_data_type: destination.data_type(),
        kernel_input_data_types: vec![destination.data_type(); 2],
    })
}

/// Views a flat operand through strided addressing.
pub fn as_strided(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let strided = context.attributes(OperationAttributes::strided)?;
    let layout = strided_layout(context.first(), strided)?;
    Ok(context.same_type(layout.shape()))
}

/// Writes a source into a copy of a flat operand at strided positions.
pub fn strided_scatter(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let strided = context.attributes(OperationAttributes::strided)?;
    let base = context.first();
    let layout = strided_layout(base, strided)?;
    if layout.has_internal_overlap() {
        return Err(Error::invalid_argument(
            "strided scatter addresses some elements more than once",
        ));
    }

    let source = context.input(1)?;
    if source.shape() != layout.shape() {
        return Err(Error::shape_mismatch(
            "strided scatter source",
            layout.dimensions(),
            source.shape().dimensions(),
        ));
    }

    if source.data_type() != base.data_type() {
        return Err(Error::data_type_mismatch(
            "strided scatter source",
            base.data_type(),
            source.data_type(),
        ));
    }
    Ok(context.same_type(base.shape()))
}

/// Checks that strided addressing stays inside a rank-one operand.
fn strided_layout(base: &ValueSpec, strided: &StridedAttributes) -> Result<Layout> {
    if base.shape().rank() != 1 {
        return Err(Error::invalid_argument(
            "strided addressing requires a flat operand",
        ));
    }

    let layout = strided.layout(0)?;
    if layout.storage_span_element_count()? > base.shape().element_count() {
        return Err(Error::out_of_range("strided addressing"));
    }
    Ok(layout)
}

/// Pads each dimension.
pub fn pad(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let pad = context.attributes(OperationAttributes::pad)?;
    Ok(context.same_type(
        &spec::pad_problem(context.shape(), &pad.padding, "operation pad")?.output_shape,
    ))
}

/// Swaps two dimensions.
pub fn transpose(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let transpose = context.attributes(OperationAttributes::transpose)?;
    Ok(context.same_type(&context.shape().transpose(transpose.dim0, transpose.dim1)?))
}

/// Reorders dimensions.
pub fn permute(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let order = context.attributes(OperationAttributes::permutation_order)?;
    Ok(context.same_type(&context.shape().permute(order)?))
}

/// Changes dimensions while keeping the element count.
pub fn reshape(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let shape = context.attributes(OperationAttributes::shape)?;
    Ok(context.same_type(&context.shape().reshape(shape.dimensions())?))
}

/// Broadcasts unit dimensions to a larger shape.
pub fn expand(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let shape = context.attributes(OperationAttributes::shape)?;
    Ok(context
        .same_type(&spec::expand_problem(context.shape(), shape, "operation expand")?.output_shape))
}

/// Removes a unit dimension.
pub fn squeeze(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let dim = context.value(OperationAttributes::dimension)?;
    Ok(context.same_type(&context.shape().squeeze(dim)?))
}

/// Inserts a unit dimension.
pub fn unsqueeze(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let dim = context.value(OperationAttributes::dimension)?;
    Ok(context.same_type(&context.shape().unsqueeze(dim)?))
}

/// Combines a range of dimensions.
pub fn flatten(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let flatten = context.attributes(OperationAttributes::flatten)?;
    let problem = spec::flatten_problem(
        context.shape(),
        flatten.start_dim,
        flatten.end_dim,
        "operation flatten",
    )?;
    Ok(context.same_type(&problem.output_shape))
}

/// Selects a range along one dimension.
pub fn slice(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let slice = context.attributes(OperationAttributes::slice)?;
    let layout = Layout::from_shape(context.shape()).slice(
        slice.dim,
        slice.start,
        slice.length,
        slice.step,
    )?;
    Ok(context.same_type(layout.shape()))
}

/// Joins operands of one data type along one dimension.
pub fn concat(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let dim = context.value(OperationAttributes::concat_dimension)?;
    context.require_same_types()?;
    let shapes = context
        .inputs
        .iter()
        .map(ValueSpec::shape)
        .collect::<Vec<_>>();
    Ok(context.same_type(&spec::concat_problem(&shapes, dim, "operation concat")?.output_shape))
}

/// Selects slices along one dimension with `I64` indices.
pub fn index_select(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let dim = context.value(OperationAttributes::index_select_dimension)?;
    let indices = context.input(1)?;
    require_indices(indices)?;
    let problem = spec::index_select_problem(
        context.shape(),
        indices.shape(),
        dim,
        "operation index select",
    )?;
    Ok(context.same_type(&problem.output_shape))
}

/// Selects elements of consecutive dimensions with broadcast `I64` index tensors.
pub fn index(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let dim = context.value(OperationAttributes::dimension)?;
    let problem = index_operands(context, 1, dim)?;
    Ok(context.same_type(&problem.output_shape))
}

/// Writes values into the elements `index` selects, returning a copy of the input.
pub fn index_put(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let attributes = context.value(|attributes| attributes.index_put().copied())?;
    let problem = index_operands(context, 2, attributes.dim)?;
    let input = context.first();
    let values = context.input(1)?;
    if values.data_type() != input.data_type() {
        return Err(Error::data_type_mismatch(
            "index put values",
            input.data_type(),
            values.data_type(),
        ));
    }

    if values.shape() != &problem.output_shape {
        return Err(Error::shape_mismatch(
            "index put values",
            problem.output_shape.dimensions(),
            values.shape().dimensions(),
        ));
    }
    Ok(context.same_type(input.shape()))
}

// Checks an in-place index write as `index_put` checks its operands, and infers no result, as `assign` does.
pub fn index_write(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let mut inference = index_put(context)?;
    inference.outputs.clear();
    inference.aliases.clear();
    Ok(inference)
}

/// Checks the index tensors that start at operand `first_index` and infers their indexing problem.
fn index_operands(
    context: &InferenceContext<'_>,
    first_index: usize,
    dim: i64,
) -> Result<MultiIndexProblem> {
    let indices = context.inputs.get(first_index..).unwrap_or_default();
    for index in indices {
        require_indices(index)?;
    }

    let shapes = indices.iter().map(ValueSpec::shape).collect::<Vec<_>>();
    spec::multi_index_problem(context.shape(), &shapes, dim)
}

/// Gathers values along one dimension with `I32` or `I64` indices.
pub fn take(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let dim = context.value(OperationAttributes::index_select_dimension)?;
    let indices = context.input(1)?;
    if !matches!(indices.data_type(), DataType::I32 | DataType::I64) {
        return Err(Error::unsupported_data_type(
            "gather indices must be i32 or i64",
        ));
    }

    let problem = spec::gather_problem(context.shape(), indices.shape(), dim, "operation gather")?;
    Ok(context.same_type(&problem.output_shape))
}

/// Repeats key or value heads.
pub fn repeat_kv(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let repeats = context.value(OperationAttributes::repeat_kv_repeats)?;
    let problem = spec::repeat_kv_problem(context.shape(), repeats, "operation repeat kv")?;
    Ok(context.same_type(&problem.output_shape))
}
