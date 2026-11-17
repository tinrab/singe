use singe_cublas::context::Context as CublasContext;

use singe_cudnn::context::Context as CudnnContext;

use singe_cutensor::context::Context as CutensorContext;

use crate::{
    backend::cuda::{
        context::{CudaExecutionContext, with_cublas_numerics},
        stream::CudaStream,
    },
    error::{Error, Result},
    utility::Locked,
};

impl CudaExecutionContext {
    pub(crate) fn with_cublas<T>(&self, f: impl FnOnce(&CublasContext) -> Result<T>) -> Result<T> {
        let mut cublas = self.runtime.cublas().locked("cuda runtime cache")?;
        cublas.set_stream(Some(self.stream.raw()))?;
        let result = with_cublas_numerics(&mut cublas, f);
        let reset = cublas.set_stream(None);
        drop(cublas);

        match (result, reset) {
            (Ok(value), Ok(())) => Ok(value),
            (Ok(_), Err(error)) => Err(Error::from(error)),
            (Err(error), Ok(()) | Err(_)) => Err(error),
        }
    }

    pub(crate) fn with_cudnn<T>(&self, f: impl FnOnce(&CudnnContext) -> Result<T>) -> Result<T> {
        let mut cudnn = self.runtime.cudnn().locked("cuda runtime cache")?;
        cudnn.set_stream(Some(self.stream.raw()))?;
        let result = f(&cudnn);
        let reset = cudnn.set_stream(None);
        drop(cudnn);

        match (result, reset) {
            (Ok(value), Ok(())) => Ok(value),
            (Ok(_), Err(error)) => Err(Error::from(error)),
            (Err(error), Ok(()) | Err(_)) => Err(error),
        }
    }

    pub(crate) fn with_cutensor<T>(
        &self,
        f: impl FnOnce(&CutensorContext, &CudaStream) -> Result<T>,
    ) -> Result<T> {
        let cutensor = self.runtime.cutensor().locked("cuda runtime cache")?;
        f(&cutensor, &self.stream)
    }
}
