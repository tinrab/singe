#[cfg(feature = "cuda")]
use crate::{
    backend::cuda::{CudaTransfer, runtime::runtime_for_device},
    data_type::DataType,
};
use crate::{error::Result, tensor::Tensor};

#[derive(Debug, Default)]
pub struct TokenReader {
    #[cfg(feature = "cuda")]
    pinned: Option<Tensor>,
}

#[derive(Debug)]
pub enum PendingTokens {
    Read(Vec<i64>),
    #[cfg(feature = "cuda")]
    Copying(CudaTransfer),
}

impl TokenReader {
    pub fn start(&mut self, tokens: &Tensor) -> Result<PendingTokens> {
        // On the device the copy is enqueued now, so work enqueued after it does not hold it back.
        #[cfg(feature = "cuda")]
        if tokens.device().is_cuda() {
            let destination = match self.pinned.take() {
                Some(pinned) if pinned.shape() == tokens.shape() => pinned,
                _ => Tensor::zeros_pinned(DataType::I64, tokens.shape().clone())?,
            };
            let runtime = runtime_for_device(tokens.device())?;
            return Ok(PendingTokens::Copying(
                tokens.copy_to_non_blocking(destination, runtime.eager_stream()?)?,
            ));
        }

        Ok(PendingTokens::Read(tokens.to_vec()?))
    }

    pub fn finish(&mut self, pending: PendingTokens) -> Result<Vec<i64>> {
        match pending {
            PendingTokens::Read(tokens) => Ok(tokens),
            #[cfg(feature = "cuda")]
            PendingTokens::Copying(transfer) => {
                let destination = transfer.wait()?;
                let tokens = destination.to_vec()?;
                self.pinned = Some(destination);
                Ok(tokens)
            }
        }
    }
}
