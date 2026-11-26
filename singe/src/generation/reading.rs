#[cfg(feature = "cuda")]
use crate::backend::cuda::{CudaTransfer, runtime::runtime_for_device};
use crate::{
    error::Result,
    tensor::{Tensor, TensorDataType},
};

#[derive(Debug, Default)]
pub struct HostReader {
    #[cfg(feature = "cuda")]
    pinned: Option<Tensor>,
}

#[derive(Debug)]
pub enum PendingRead {
    Ready(Tensor),
    #[cfg(feature = "cuda")]
    Copying(CudaTransfer),
}

impl HostReader {
    pub fn start(&mut self, values: &Tensor) -> Result<PendingRead> {
        // On the device the copy is enqueued now, so work enqueued after it does not hold it back.
        #[cfg(feature = "cuda")]
        if values.device().is_cuda() {
            let destination = match self.pinned.take() {
                Some(pinned)
                    if pinned.shape() == values.shape()
                        && pinned.data_type() == values.data_type() =>
                {
                    pinned
                }
                _ => Tensor::zeros_pinned(values.data_type(), values.shape().clone())?,
            };
            let runtime = runtime_for_device(values.device())?;
            return Ok(PendingRead::Copying(
                values.copy_to_non_blocking(destination, runtime.eager_stream()?)?,
            ));
        }

        Ok(PendingRead::Ready(values.clone()))
    }

    pub fn finish<T: TensorDataType>(&mut self, pending: PendingRead) -> Result<Vec<T>> {
        match pending {
            PendingRead::Ready(values) => values.to_vec(),
            #[cfg(feature = "cuda")]
            PendingRead::Copying(transfer) => {
                let destination = transfer.wait()?;
                let values = destination.to_vec()?;
                self.pinned = Some(destination);
                Ok(values)
            }
        }
    }
}
