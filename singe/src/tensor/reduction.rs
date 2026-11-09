use crate::{
    backend::{
        registry::{Operation, OperationAttributes, TopKAttributes},
        spec::reduction_problem,
    },
    dispatch::call,
    error::{Error, Result},
    tensor::Tensor,
};

impl Tensor {
    /// Returns the top `k` values and their indices along `dim`.
    ///
    /// This records a graph operation when called inside graph capture.
    ///
    /// # Errors
    ///
    /// Returns an error if `k`, `dim`, data type, layout, or backend support is invalid.
    pub fn topk(&self, k: i64, dim: i64, largest: bool, sorted: bool) -> Result<(Self, Self)> {
        let mut outputs = call(
            Operation::TopK,
            OperationAttributes::TopK(TopKAttributes {
                k,
                dim,
                largest,
                sorted,
            }),
            &[self],
        )?
        .into_iter();
        match (outputs.next(), outputs.next()) {
            (Some(values), Some(indices)) => Ok((values, indices)),
            _ => Err(Error::missing_output("topk")),
        }
    }

    /// Sorts each slice along `dim`, ascending or `descending`, returning the sorted values and their `I64` positions in this tensor.
    ///
    /// The sort is stable, and NaN sorts as the largest value.
    ///
    /// # Errors
    ///
    /// Returns an error if `dim` is out of range or the backend does not support the data type.
    pub fn sort(&self, dim: i64, descending: bool) -> Result<(Self, Self)> {
        let indices = self.argsort(dim, descending)?;
        let values = if self.shape().rank() == 0 {
            self.clone()
        } else {
            self.gather(dim, &indices)?
        };
        Ok((values, indices))
    }
}

// The order of a vector norm: `Finite(p)` sums `|x|^p`, and the infinite orders take the largest or smallest magnitude.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormOrder {
    Finite(f64),
    Infinity,
    NegativeInfinity,
}

// Reductions composed from registry reductions and element-wise operations.
impl Tensor {
    // The variance over `dimensions`, dividing the summed squared deviations by the point count less `correction`.
    pub fn var(&self, dimensions: &[i64], correction: i64, keepdim: bool) -> Result<Self> {
        let problem = reduction_problem(self.shape(), dimensions, keepdim)?;
        let points = problem
            .dimensions
            .iter()
            .map(|&axis| self.shape().dimensions()[axis])
            .product::<i64>();
        let deviation = |x: &Self| x.sub(&x.mean(dimensions, true)?);
        // Too few points leave no degrees of freedom, which divides by zero or a negative count, as `PyTorch` does.
        #[expect(
            clippy::cast_precision_loss,
            reason = "point counts are far below 2^53"
        )]
        let degrees = (points - correction).max(0) as f64;
        self.rounding_once(|x| {
            let deviation = deviation(x)?;
            deviation
                .mul(&deviation)?
                .sum(dimensions, keepdim)?
                .div_scalar(degrees)
        })
    }

    pub fn std(&self, dimensions: &[i64], correction: i64, keepdim: bool) -> Result<Self> {
        self.rounding_once(|x| x.var(dimensions, correction, keepdim)?.sqrt())
    }

    // The vector norm of `order` over `dimensions`.
    pub fn vector_norm(&self, order: NormOrder, dimensions: &[i64], keepdim: bool) -> Result<Self> {
        self.rounding_once(|x| {
            let magnitude = x.abs()?;
            match order {
                NormOrder::Infinity => magnitude.max_reduce(dimensions, keepdim),
                NormOrder::NegativeInfinity => magnitude.min_reduce(dimensions, keepdim),
                NormOrder::Finite(1.0) => magnitude.sum(dimensions, keepdim),
                NormOrder::Finite(2.0) => x.mul(x)?.sum(dimensions, keepdim)?.sqrt(),
                NormOrder::Finite(order) if order > 0.0 => magnitude
                    .pow_scalar(order)?
                    .sum(dimensions, keepdim)?
                    .pow_scalar(1.0 / order),
                NormOrder::Finite(_) => Err(Error::invalid_argument(
                    "vector norms take a positive or an infinite order",
                )),
            }
        })
    }
}
