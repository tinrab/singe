// Affine projections, and flattening what a projection reads.

use crate::{
    backend::registry::{FlattenAttributes, Operation, OperationAttributes},
    data_type::DataType,
    dispatch::call_one,
    error::{Error, Result},
    graph::capture,
    nn::{Layer, ModelState, RmsNorm, StateKey, joined_rows},
    tensor::Tensor,
};

/// Affine projection with checkpoint weights in `[out_features, in_features]` order.
#[derive(Debug, Clone)]
pub struct Linear {
    weight: StateKey,
    bias: Option<StateKey>,
    bit: Option<Box<BitLinear>>,
}

#[derive(Debug, Clone)]
pub struct BitLinear {
    pub norm: Option<RmsNorm>,
    pub scale: StateKey,
    // Whether the weight is `U8` `[R, C]` packing four ternary rows per byte, as `BitNet`'s offline checkpoints store it: bit pair `i` of row `r` holds row `i · R + r`'s value plus one.
    pub packed: bool,
}

impl BitLinear {
    // Multiplies `input` by the packed `weight`'s four row blocks in turn, each unpacked to the input's type just before its product, and joins the products.
    fn packed_product(input: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let mask = capture::scalar(weight.device(), 3_u8)?;
        let products = (0..4_u8)
            .map(|pair| {
                let rows = weight
                    .bitwise_right_shift(&capture::scalar(weight.device(), 2 * pair)?)?
                    .bitwise_and(&mask)?
                    .cast(input.data_type())?
                    .sub_scalar(1.0)?;
                input.matmul(&rows.transpose(0, 1)?)
            })
            .collect::<Result<Vec<_>>>()?;
        Tensor::concat(&products, -1)
    }

    fn quantized(&self, input: &Tensor, state: &ModelState) -> Result<Tensor> {
        let input = match &self.norm {
            Some(norm) => input.apply(norm, state)?,
            None => input.clone(),
        };
        let data_type = input.data_type();
        let input = input.cast(DataType::F32)?;
        let largest = input
            .abs()?
            .max_reduce(&[-1], true)?
            .clamp(1e-5, f32::MAX)?;
        let scale = largest.rdiv_scalar(127.0)?;
        input
            .mul(&scale)?
            .round()?
            .clamp(-128.0, 127.0)?
            .div(&scale)?
            .cast(data_type)
    }
}

impl Linear {
    /// Stores affine weight and optional bias keys without resolving their tensors.
    ///
    /// Weights are interpreted in `[out_features, in_features]` order when the layer runs.
    #[must_use]
    pub fn new(weight: StateKey, bias: Option<StateKey>) -> Self {
        Self {
            weight,
            bias,
            bit: None,
        }
    }

    #[must_use]
    pub fn with_bit_linear(mut self, bit: BitLinear) -> Self {
        self.bit = Some(Box::new(bit));
        self
    }

    #[must_use]
    pub const fn weight(&self) -> &StateKey {
        &self.weight
    }

    #[must_use]
    pub const fn bias(&self) -> Option<&StateKey> {
        self.bias.as_ref()
    }

    pub fn forward_with(
        &self,
        input: &Tensor,
        weight: &Tensor,
        state: &ModelState,
    ) -> Result<Tensor> {
        let quantized = self
            .bit
            .as_ref()
            .map(|bit| bit.quantized(input, state))
            .transpose()?;
        let input = quantized.as_ref().unwrap_or(input);
        let output = match &self.bit {
            Some(bit) if bit.packed => BitLinear::packed_product(input, weight)?,
            _ => input.matmul(&weight.transpose(0, 1)?)?,
        };
        let output = match &self.bit {
            Some(bit) => output.mul(state.tensor(&bit.scale)?)?,
            None => output,
        };
        match &self.bias {
            Some(bias) => output.add(state.tensor(bias)?),
            None => Ok(output),
        }
    }
}

impl Layer for Linear {
    fn forward(&self, input: &Tensor, state: &ModelState) -> Result<Tensor> {
        self.forward_with(input, state.tensor(&self.weight)?, state)
    }
}

// Runs `linears`, which read one `input`, as one matmul when their weights, and biases if they all have one, are adjacent rows of one tensor, as `ModelState::join_projection_rows` binds them, and returns each one's output as a view of the joined product.
// A capture records the linears apart, `None` then, as when the weights are not joined; compiling merges them again (`merge_sibling_projections`) after it has seen the whole graph.
pub(crate) fn joined_forward<const N: usize>(
    linears: &[&Linear; N],
    input: &Tensor,
    state: &ModelState,
) -> Result<Option<[Tensor; N]>> {
    if capture::is_active() || linears.iter().any(|linear| linear.bit.is_some()) {
        return Ok(None);
    }

    let weights = linears
        .iter()
        .map(|linear| state.tensor(&linear.weight))
        .collect::<Result<Vec<_>>>()?;
    let Some(weight) = joined_rows(&weights) else {
        return Ok(None);
    };
    let bias = if linears.iter().all(|linear| linear.bias.is_none()) {
        None
    } else {
        let Some(biases) = linears
            .iter()
            .map(|linear| linear.bias.as_ref().map(|bias| state.tensor(bias)))
            .collect::<Option<Result<Vec<_>>>>()
            .transpose()?
        else {
            return Ok(None);
        };
        let Some(bias) = joined_rows(&biases) else {
            return Ok(None);
        };
        Some(bias)
    };

    let mut product = input.matmul(&weight.transpose(0, 1)?)?;
    if let Some(bias) = bias {
        product = product.add(&bias)?;
    }

    let mut start = 0;
    let mut outputs = Vec::with_capacity(N);
    for weight in weights {
        let &[rows, ..] = weight.shape().dimensions() else {
            return Err(Error::invalid_argument("joined projection weight"));
        };
        outputs.push(product.narrow(-1, start, rows)?);
        start += rows;
    }

    let outputs = <[Tensor; N]>::try_from(outputs)
        .map_err(|_| Error::missing_output("joined projections"))?;
    Ok(Some(outputs))
}

/// A stateless layer combining an inclusive consecutive range of tensor axes.
#[derive(Debug, Clone, Copy)]
pub struct Flatten {
    start_dim: i64,
    end_dim: i64,
}

impl Flatten {
    /// Stores inclusive start and end dimensions for flattening.
    ///
    /// Negative axes count from the end.
    /// Validation occurs when the layer runs.
    #[must_use]
    pub fn new(start_dim: i64, end_dim: i64) -> Self {
        Self { start_dim, end_dim }
    }
}

impl Layer for Flatten {
    fn forward(&self, input: &Tensor, _state: &ModelState) -> Result<Tensor> {
        call_one(
            Operation::Flatten,
            OperationAttributes::Flatten(FlattenAttributes {
                start_dim: self.start_dim,
                end_dim: self.end_dim,
            }),
            &[input],
        )
    }
}
