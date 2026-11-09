use crate::{data_type::DataType, error::Result, tensor::Tensor};

/// The mathematical GELU formula, retained in captured graph opcodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GeluApproximation {
    /// Uses the error-function GELU formula, evaluated with the backend's numerical implementation.
    Exact,
    /// Uses the tanh approximation of the GELU formula.
    Tanh,
}

impl Tensor {
    /// Selects exact erf-based GELU or the tanh approximation explicitly.
    /// `gelu()` uses the tanh approximation; `gelu_exact()` uses erf.
    ///
    /// # Errors
    ///
    /// Returns an error if the selected GELU operation is unsupported for this tensor, graph capture fails, or backend allocation or execution fails.
    pub fn gelu_with_approximation(&self, approximation: GeluApproximation) -> Result<Self> {
        match approximation {
            GeluApproximation::Exact => self.gelu_exact(),
            GeluApproximation::Tanh => self.gelu(),
        }
    }
}

// SELU's constants, as `torch.nn.functional.selu` defines them.
const SELU_ALPHA: f64 = 1.673_263_242_354_377_3;
const SELU_SCALE: f64 = 1.050_700_987_355_480_5;

// Activations composed from element-wise operations, which fuse into one generated kernel in compiled graphs.
impl Tensor {
    // Computes a composite of half-precision values in `F32` and rounds its result once, as PyTorch's kernels compute the whole function, since rounding each step to 16 bits loses far more than one.
    pub(crate) fn rounding_once(
        &self,
        compute: impl FnOnce(&Self) -> Result<Self>,
    ) -> Result<Self> {
        let data_type = self.data_type();
        if matches!(data_type, DataType::F16 | DataType::Bf16) {
            compute(&self.cast(DataType::F32)?)?.cast(data_type)
        } else {
            compute(self)
        }
    }

    // `x` where it is positive and `negative_slope · x` elsewhere.
    pub fn leaky_relu(&self, negative_slope: f64) -> Result<Self> {
        self.rounding_once(|x| x.relu()?.sub(&x.neg()?.relu()?.mul_scalar(negative_slope)?))
    }

    // `x` where it is positive and `alpha · (exp(x) − 1)` elsewhere.
    pub fn elu(&self, alpha: f64) -> Result<Self> {
        self.rounding_once(|x| {
            x.relu()?
                .add(&x.negative_part()?.expm1()?.mul_scalar(alpha)?)
        })
    }

    pub fn selu(&self) -> Result<Self> {
        self.rounding_once(|x| x.elu(SELU_ALPHA)?.mul_scalar(SELU_SCALE))
    }

    // `x` where it is positive and `alpha · (exp(x / alpha) − 1)` elsewhere.
    pub fn celu(&self, alpha: f64) -> Result<Self> {
        self.rounding_once(|x| {
            x.relu()?.add(
                &x.negative_part()?
                    .div_scalar(alpha)?
                    .expm1()?
                    .mul_scalar(alpha)?,
            )
        })
    }

    // `x / (1 + |x|)`.
    pub fn softsign(&self) -> Result<Self> {
        self.rounding_once(|x| x.div(&x.abs()?.add_scalar(1.0)?))
    }

    // `clamp(x / 6 + 1 / 2, 0, 1)`.
    pub fn hardsigmoid(&self) -> Result<Self> {
        self.rounding_once(|x| x.div_scalar(6.0)?.add_scalar(0.5)?.clamp(0.0, 1.0))
    }

    pub fn hardswish(&self) -> Result<Self> {
        self.rounding_once(|x| x.mul(&x.hardsigmoid()?))
    }

    pub fn hardtanh(&self, min: f32, max: f32) -> Result<Self> {
        self.clamp(min, max)
    }

    // `log(1 + exp(beta · x)) / beta`, written as `max(beta · x, 0) + log1p(exp(−|beta · x|))` so no exponential overflows.
    pub fn softplus(&self, beta: f64) -> Result<Self> {
        self.rounding_once(|x| {
            let scaled = x.mul_scalar(beta)?;
            scaled
                .relu()?
                .add(&scaled.abs()?.neg()?.exp()?.log1p()?)?
                .div_scalar(beta)
        })
    }

    // `x · tanh(softplus(x))`.
    pub fn mish(&self) -> Result<Self> {
        self.rounding_once(|x| x.mul(&x.softplus(1.0)?.tanh()?))
    }

    // `min(x, 0)`, which keeps NaN.
    fn negative_part(&self) -> Result<Self> {
        self.neg()?.relu()?.neg()
    }
}
