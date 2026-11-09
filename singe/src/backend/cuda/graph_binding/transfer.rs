use singe_cuda::{
    error::Error as CudaError,
    memory::{DeviceMemory, MemoryCopyKind},
};

use crate::{
    backend::cuda::{
        graph_value::CudaGraphValue, stream::CudaStream, tensor_region::CudaTensorSource,
    },
    error::{Error, Result},
    utility::checked_sub,
};

/// Copies a matching physical layout, including its stride gaps.
///
/// # Safety
/// The source bytes must be initialized when the stream reads them.
/// Both owners must survive completion, prior producers must precede the copy, and accesses to the destination must be exclusive until completion.
pub unsafe fn copy_cuda_tensor_data_on(
    dst: &CudaGraphValue,
    src: &(impl CudaTensorSource + ?Sized),
    stream: &CudaStream,
) -> Result<()> {
    if dst.data_type() != src.data_type() {
        return Err(Error::data_type_mismatch(
            "cuda copy datatype",
            dst.data_type(),
            src.data_type(),
        ));
    }

    if dst.shape().dimensions() != src.shape().dimensions()
        || (!dst.is_contiguous() || !src.is_contiguous())
            && dst.layout().strides() != src.layout().strides()
    {
        return Err(Error::invalid_argument("cuda copy physical layout"));
    }

    let dst = dst.region()?.region();
    let src = src.tensor_region()?.region();
    if dst.context() != stream.raw().context() || src.context() != stream.raw().context() {
        return Err(CudaError::MemoryContextMismatch.into());
    }

    if src.overlaps(dst) {
        return Err(Error::invalid_argument(
            "cuda copy source and destination overlap",
        ));
    }

    if let Some(ready) = src.stream()
        && ready.as_raw() != stream.raw().as_raw()
    {
        src.order_on(stream.raw())?;
    }

    let bytes = checked_sub(src.byte_len(), dst.byte_len(), "cuda copy source capacity")?;
    if bytes != 0 {
        return Err(Error::invalid_argument("cuda copy physical span"));
    }

    stream.raw().context().bind()?;
    // SAFETY: validated ranges and actual contexts; caller retains ordered owners.
    Ok(unsafe {
        DeviceMemory::<u8>::copy_async(
            dst.as_ptr().cast(),
            src.as_ptr().cast(),
            dst.byte_len(),
            MemoryCopyKind::DeviceToDevice,
            stream.raw(),
        )
    }?)
}
