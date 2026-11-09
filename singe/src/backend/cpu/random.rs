use std::ops::{Add, Mul};

use crate::{
    backend::registry::RandomAttributes,
    data_type::DataType,
    error::{Error, Result},
    random::{GeneratorState, normal_values, unit_values},
    shape::Shape,
    tensor::Tensor,
    utility::to_usize,
};

/// Reads the `[seed, offset]` generator state operand.
fn generator_state(state: &Tensor) -> Result<GeneratorState> {
    match state.to_vec::<i64>()?.as_slice() {
        [seed, offset] => Ok(GeneratorState {
            seed: seed.cast_unsigned(),
            offset: offset.cast_unsigned(),
        }),
        _ => Err(Error::invalid_argument(
            "generator state must hold a seed and an offset",
        )),
    }
}

fn floating(values: Vec<f32>, shape: Shape, data_type: DataType, state: &Tensor) -> Result<Tensor> {
    match data_type {
        DataType::F32 => Tensor::from_vec(state.device(), shape, values),
        DataType::F64 => Tensor::from_vec(
            state.device(),
            shape,
            values.into_iter().map(f64::from).collect::<Vec<_>>(),
        ),
        _ => Err(Error::unsupported_data_type(
            "random values must be f32 or f64",
        )),
    }
}

/// Samples `first + (second - first) * u` for uniform `u` in `[0, 1)`.
#[expect(
    clippy::suboptimal_flops,
    reason = "CUDA kernels compile without fused multiply-add, and matching their rounding keeps samples identical"
)]
pub fn uniform(state: &Tensor, random: &RandomAttributes) -> Result<Tensor> {
    let shape = Shape::create(&random.dimensions)?;
    let values = unit_values(
        generator_state(state)?,
        random.offset,
        shape.element_count(),
    )
    .into_iter()
    .map(|unit| random.first + (random.second - random.first) * unit)
    .collect();
    floating(values, shape, random.data_type, state)
}

/// Samples `first + second * z` for standard normal `z`.
#[expect(
    clippy::suboptimal_flops,
    reason = "CUDA kernels compile without fused multiply-add, and matching their rounding keeps samples identical"
)]
pub fn normal(state: &Tensor, random: &RandomAttributes) -> Result<Tensor> {
    let shape = Shape::create(&random.dimensions)?;
    let values = normal_values(
        generator_state(state)?,
        random.offset,
        shape.element_count(),
    )
    .into_iter()
    .map(|standard| random.first + random.second * standard)
    .collect();
    floating(values, shape, random.data_type, state)
}

/// Samples `true` where a uniform value falls below the element's probability.
pub fn bernoulli(
    probabilities: &Tensor,
    state: &Tensor,
    random: &RandomAttributes,
) -> Result<Tensor> {
    let units = unit_values(
        generator_state(state)?,
        random.offset,
        probabilities.shape().element_count(),
    );
    let samples = match probabilities.data_type() {
        DataType::F32 => probabilities
            .to_vec::<f32>()?
            .into_iter()
            .zip(units)
            .map(|(probability, unit)| unit < probability)
            .collect::<Vec<_>>(),
        DataType::F64 => probabilities
            .to_vec::<f64>()?
            .into_iter()
            .zip(units)
            .map(|(probability, unit)| f64::from(unit) < probability)
            .collect(),
        _ => return Err(Error::unsupported_data_type("bernoulli probabilities")),
    };
    Tensor::from_vec(
        probabilities.device(),
        probabilities.shape().clone(),
        samples,
    )
}

/// Samples category indices per row by inverse cumulative distribution of the row's weights.
pub fn multinomial(weights: &Tensor, state: &Tensor, random: &RandomAttributes) -> Result<Tensor> {
    let dimensions = weights.shape().dimensions();
    let (rows, categories) = match dimensions {
        [categories] => (1, *categories),
        [rows, categories] => (*rows, *categories),
        _ => {
            return Err(Error::invalid_argument(
                "multinomial weights must be a vector or matrix",
            ));
        }
    };
    let rows = to_usize(rows, "multinomial rows")?;
    let categories = to_usize(categories, "multinomial categories")?;
    let samples = to_usize(random.samples, "multinomial samples")?;
    let units = unit_values(generator_state(state)?, random.offset, rows * samples);
    let indices = match weights.data_type() {
        DataType::F32 => sample_rows(
            &weights.to_vec::<f32>()?,
            &units,
            categories,
            samples,
            |unit| unit,
        ),
        DataType::F64 => sample_rows(
            &weights.to_vec::<f64>()?,
            &units,
            categories,
            samples,
            f64::from,
        ),
        _ => return Err(Error::unsupported_data_type("multinomial weights")),
    };
    let mut output = vec![random.samples];
    if dimensions.len() == 2 {
        output.insert(0, dimensions[0]);
    }
    Tensor::from_vec(weights.device(), Shape::create(output)?, indices)
}

fn sample_rows<T>(
    weights: &[T],
    units: &[f32],
    categories: usize,
    samples: usize,
    widen: impl Fn(f32) -> T,
) -> Vec<i64>
where
    T: Copy + Default + PartialOrd + Add<Output = T> + Mul<Output = T>,
{
    units
        .iter()
        .enumerate()
        .map(|(element, &unit)| {
            let row = &weights[element / samples * categories..][..categories];
            let total = row.iter().fold(T::default(), |sum, &weight| sum + weight);
            let target = widen(unit) * total;
            let mut cumulative = T::default();
            let mut chosen = categories - 1;
            for (category, &weight) in row.iter().enumerate() {
                cumulative = cumulative + weight;
                if target < cumulative {
                    chosen = category;
                    break;
                }
            }
            i64::try_from(chosen).unwrap_or(i64::MAX)
        })
        .collect()
}
