use singe_cublas::context::Context as CublasContext;

use crate::{
    backend::cuda::{context::with_cublas_numerics, runtime::CudaRuntime, stream::current_stream},
    device::Device,
    error::{Error, Result},
    utility::Locked,
};

pub fn with_cublas_stream<T>(
    runtime: &CudaRuntime,
    device: Device,
    f: impl FnOnce(&CublasContext) -> Result<T>,
) -> Result<T> {
    let stream = current_stream(device)?;
    let cublas = runtime.cublas().locked("cuda runtime cache")?;
    cublas.set_stream(stream.as_deref())?;
    // Captured GEMMs must not borrow the handle's mutable default workspace pool.
    // SAFETY: no external workspace is installed; restoring the stream restores the default pool.
    let result = unsafe { cublas.set_workspace(None) }
        .map_err(Error::from)
        .and_then(|()| with_cublas_numerics(&cublas, f));
    let reset = cublas.set_stream(None);
    drop(cublas);

    match (result, reset) {
        (Ok(value), Ok(())) => Ok(value),
        (Ok(_), Err(error)) => Err(Error::from(error)),
        (Err(error), Ok(()) | Err(_)) => Err(error),
    }
}
