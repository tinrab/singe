use crate::{
    backend::registry::{OperationAttributes, OperationInference, infer::InferenceContext},
    data_type::DataType,
    error::{Error, Result},
    shape::Shape,
    value::ValueSpec,
};

/// Creates floating values of the attribute's shape and data type.
pub fn uniform(context: &InferenceContext<'_>) -> Result<OperationInference> {
    generated(context)
}

/// Creates floating values of the attribute's shape and data type.
pub fn normal(context: &InferenceContext<'_>) -> Result<OperationInference> {
    generated(context)
}

/// Samples a Boolean per floating probability.
pub fn bernoulli(context: &InferenceContext<'_>) -> Result<OperationInference> {
    context.attributes(OperationAttributes::random)?;
    let probabilities = context.first();
    require_float(probabilities)?;
    require_state(context.input(1)?)?;
    Ok(context.finish(probabilities.shape(), probabilities.data_type()))
}

/// Samples category indices per row of a vector or matrix of floating weights.
pub fn multinomial(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let random = context.attributes(OperationAttributes::random)?;
    let weights = context.first();
    require_float(weights)?;
    require_state(context.input(1)?)?;
    if random.samples < 1 {
        return Err(Error::invalid_argument(
            "multinomial needs at least one sample",
        ));
    }

    let dimensions = match weights.shape().dimensions() {
        [categories] if *categories > 0 => vec![random.samples],
        [rows, categories] if *categories > 0 => vec![*rows, random.samples],
        _ => {
            return Err(Error::invalid_argument(
                "multinomial weights must be a nonempty vector or matrix of categories",
            ));
        }
    };
    Ok(context.finish(&Shape::create(dimensions)?, weights.data_type()))
}

fn generated(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let random = context.attributes(OperationAttributes::random)?;
    require_state(context.first())?;
    if !random.data_type.is_float() {
        return Err(Error::unsupported_data_type(
            "random values must be floating point",
        ));
    }

    let mut inference = context.finish(&Shape::create(&random.dimensions)?, random.data_type);
    inference.computation_data_type = random.data_type;
    Ok(inference)
}

fn require_float(input: &ValueSpec) -> Result<()> {
    if matches!(input.data_type(), DataType::F32 | DataType::F64) {
        Ok(())
    } else {
        Err(Error::unsupported_data_type(
            "random probabilities must be f32 or f64",
        ))
    }
}

fn require_state(state: &ValueSpec) -> Result<()> {
    if state.data_type() == DataType::I64 && state.shape().dimensions() == [2] {
        Ok(())
    } else {
        Err(Error::invalid_argument(
            "generator state must be an i64 vector [seed, offset]",
        ))
    }
}
