use crate::{
    backend::registry::{Operation, OperationAttributes},
    dispatch::call_one,
    error::Result,
    tensor::Tensor,
};

// Complex values are pairs along a trailing axis of two, `[real, imaginary]`, as PyTorch's `view_as_real` lays them out.
impl Tensor {
    // The `n / 2 + 1` frequencies of the last axis's length-`n` real signal, unnormalized, as `torch.fft.rfft` computes them.
    pub fn rfft(&self) -> Result<Self> {
        call_one(Operation::Rfft, OperationAttributes::None, &[self])
    }

    // The length-`length` real signal of `[..., k, 2]` frequencies, as `torch.fft.irfft` computes it: the imaginary parts of the constant and, for even lengths, the highest frequency are ignored, and the result is divided by `length`.
    pub fn irfft(&self, length: i64) -> Result<Self> {
        call_one(
            Operation::Irfft,
            OperationAttributes::Length(length),
            &[self],
        )
    }
}
