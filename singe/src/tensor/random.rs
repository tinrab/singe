use crate::{
    backend::registry::{Operation, OperationAttributes, RandomAttributes},
    data_type::DataType,
    device::Device,
    dispatch::call_one,
    error::Result,
    random::{self, Generator},
    shape::Shape,
    tensor::Tensor,
    utility::to_usize,
};

impl Tensor {
    /// Samples values uniformly from `[0, 1)`.
    ///
    /// Uses `generator`, or the default generator of `device`.
    ///
    /// # Errors
    ///
    /// Returns an error if `data_type` is not floating point or the backend fails.
    pub fn rand(
        device: Device,
        shape: impl Into<Shape>,
        data_type: DataType,
        generator: Option<&Generator>,
    ) -> Result<Self> {
        Self::uniform(device, shape, 0.0, 1.0, data_type, generator)
    }

    /// Samples values uniformly from `[low, high)`.
    ///
    /// # Errors
    ///
    /// Returns an error if `data_type` is not floating point or the backend fails.
    pub fn uniform(
        device: Device,
        shape: impl Into<Shape>,
        low: f32,
        high: f32,
        data_type: DataType,
        generator: Option<&Generator>,
    ) -> Result<Self> {
        sample_shaped(
            Operation::Uniform,
            4,
            device,
            &shape.into(),
            low,
            high,
            data_type,
            generator,
        )
    }

    /// Samples standard normal values.
    ///
    /// # Errors
    ///
    /// Returns an error if `data_type` is not floating point or the backend fails.
    pub fn randn(
        device: Device,
        shape: impl Into<Shape>,
        data_type: DataType,
        generator: Option<&Generator>,
    ) -> Result<Self> {
        Self::normal(device, shape, 0.0, 1.0, data_type, generator)
    }

    /// Samples normal values with `mean` and standard deviation `std`.
    ///
    /// # Errors
    ///
    /// Returns an error if `data_type` is not floating point or the backend fails.
    pub fn normal(
        device: Device,
        shape: impl Into<Shape>,
        mean: f32,
        std: f32,
        data_type: DataType,
        generator: Option<&Generator>,
    ) -> Result<Self> {
        sample_shaped(
            Operation::Normal,
            2,
            device,
            &shape.into(),
            mean,
            std,
            data_type,
            generator,
        )
    }

    /// Samples `true` with each element's probability.
    ///
    /// # Errors
    ///
    /// Returns an error if this tensor is not `F32` or `F64`, or the backend fails.
    pub fn bernoulli(&self, generator: Option<&Generator>) -> Result<Self> {
        let blocks = random::blocks(self.shape().element_count(), 4);
        sample(
            Operation::Bernoulli,
            self.device(),
            random_attributes(&Shape::scalar(), self.data_type(), 0.0, 0.0, 0),
            &[self],
            blocks,
            generator,
        )
    }

    /// Samples `samples` category indices per row of nonnegative weights, with replacement.
    ///
    /// A vector of weights gives a vector of indices; a matrix gives one row of indices per row.
    ///
    /// # Errors
    ///
    /// Returns an error if this tensor is not a nonempty `F32` or `F64` vector or matrix, `samples` is not positive, or the backend fails.
    pub fn multinomial(&self, samples: i64, generator: Option<&Generator>) -> Result<Self> {
        let rows = match self.shape().dimensions() {
            [rows, _] => to_usize(*rows, "multinomial rows")?,
            _ => 1,
        };
        let draws = rows.saturating_mul(to_usize(samples, "multinomial samples")?);
        sample(
            Operation::Multinomial,
            self.device(),
            random_attributes(&Shape::scalar(), self.data_type(), 0.0, 0.0, samples),
            &[self],
            random::blocks(draws, 4),
            generator,
        )
    }
}

fn random_attributes(
    shape: &Shape,
    data_type: DataType,
    first: f32,
    second: f32,
    samples: i64,
) -> RandomAttributes {
    RandomAttributes {
        dimensions: shape.dimensions().to_vec(),
        data_type,
        first,
        second,
        samples,
        offset: 0,
    }
}

/// Binds the generator state and dispatches a random operation that consumes `blocks` Philox blocks.
fn sample(
    operation: Operation,
    device: Device,
    mut attributes: RandomAttributes,
    inputs: &[&Tensor],
    blocks: u64,
    generator: Option<&Generator>,
) -> Result<Tensor> {
    let generator = match generator {
        Some(generator) => generator.clone(),
        None => Generator::default_for(device)?,
    };
    let (state, offset) = random::operand(&generator, device, blocks)?;
    attributes.offset = offset;
    let mut operands = inputs.to_vec();
    operands.push(&state);
    call_one(
        operation,
        OperationAttributes::Random(attributes),
        &operands,
    )
}

// Draws a tensor of `shape`, each Philox block giving `lanes` values.
fn sample_shaped(
    operation: Operation,
    lanes: usize,
    device: Device,
    shape: &Shape,
    first: f32,
    second: f32,
    data_type: DataType,
    generator: Option<&Generator>,
) -> Result<Tensor> {
    sample(
        operation,
        device,
        random_attributes(shape, data_type, first, second, 0),
        &[],
        random::blocks(shape.element_count(), lanes),
        generator,
    )
}
