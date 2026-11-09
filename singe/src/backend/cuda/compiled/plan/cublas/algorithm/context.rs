use singe_cublas::lt::context::Context as CublasLtContext;

use crate::{backend::cuda::runtime::CudaRuntime, error::Result, utility::Locked};

pub fn with_cublaslt<T>(
    runtime: &CudaRuntime,
    f: impl FnOnce(&CublasLtContext) -> Result<T>,
) -> Result<T> {
    let cublaslt = runtime.cublaslt().locked("cuda runtime cache")?;
    f(&cublaslt)
}
