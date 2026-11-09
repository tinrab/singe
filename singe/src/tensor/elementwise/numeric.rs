use crate::{
    backend::registry::{Operation, OperationAttributes},
    data_type::DataType,
    dispatch::call_one,
    error::{Error, Result},
    graph::capture,
    shape::Shape,
    tensor::Tensor,
};

impl Tensor {
    /// Selects values from `x` or `y` using a Bool condition and broadcasting.
    /// The selected values retain their dtype.
    /// All operands must share a device.
    ///
    /// # Errors
    ///
    /// Returns an error if the condition is not boolean, operand devices differ, value data types differ, shapes cannot broadcast, capture fails, or the backend cannot allocate or execute selection.
    pub fn where_select(condition: &Self, x: &Self, y: &Self) -> Result<Self> {
        if condition.data_type() != DataType::Bool {
            return Err(Error::unsupported_data_type("where condition"));
        }

        for input in [condition, y] {
            if input.device() != x.device() {
                return Err(Error::device_mismatch(
                    "where operands",
                    x.device(),
                    input.device(),
                ));
            }
        }

        if x.data_type() != y.data_type() {
            return Err(Error::data_type_mismatch(
                "where values",
                x.data_type(),
                y.data_type(),
            ));
        }
        call_one(
            Operation::Where,
            OperationAttributes::None,
            &[condition, x, y],
        )
    }
}

impl Tensor {
    // −1, 0, or 1 by the sign of each element, 0 for NaN as `torch.sign` gives, and each element itself for `Bool`.
    pub fn sign(&self) -> Result<Self> {
        let data_type = self.data_type();
        if data_type == DataType::Bool {
            return self.copy();
        }

        let zero = capture::literal(self.device(), |device| {
            Self::zeroed_on(device, data_type, Shape::scalar())
        })?;
        self.greater(&zero)?
            .cast(data_type)?
            .sub(&self.less(&zero)?.cast(data_type)?)
    }
}

impl Tensor {
    // A copy with NaN replaced by `nan`, and infinities by `infinity` and `negative_infinity`, which default to the data type's largest and smallest finite values.
    pub fn nan_to_num(
        &self,
        nan: f64,
        infinity: Option<f64>,
        negative_infinity: Option<f64>,
    ) -> Result<Self> {
        let data_type = self.data_type();
        if !data_type.is_float() {
            return self.copy();
        }

        let largest = largest_finite(data_type);
        let zero = capture::literal(self.device(), |device| {
            Self::zeroed_on(device, data_type, Shape::scalar())
        })?;
        let constant = |value: f64| zero.add_scalar(value);
        let positive = constant(f64::INFINITY)?;
        let negative = constant(f64::NEG_INFINITY)?;
        let replaced = Self::where_select(
            &self.equal(&positive)?,
            &constant(infinity.unwrap_or(largest))?,
            self,
        )?;
        let replaced = Self::where_select(
            &self.equal(&negative)?,
            &constant(negative_infinity.unwrap_or(-largest))?,
            &replaced,
        )?;
        Self::where_select(&self.isnan()?, &constant(nan)?, &replaced)
    }
}

// The largest finite value of floating `data_type`, which `nan_to_num` puts in place of positive infinity.
fn largest_finite(data_type: DataType) -> f64 {
    match data_type {
        DataType::F16 => 65_504.0,
        DataType::Bf16 => 3.389_531_389_251_535_5e38,
        DataType::F32 => f64::from(f32::MAX),
        DataType::F8E4M3 => 448.0,
        DataType::F8E5M2 => 57_344.0,
        DataType::F4E2M1 => 6.0,
        _ => f64::MAX,
    }
}
