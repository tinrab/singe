use singe_cuda::data_type::DataTypeLike;
use singe_cusparse_sys as sys;

use crate::{
    context::Context,
    error::Result,
    scalar::ScalarMut,
    try_ffi,
    types::Operation,
    utility::to_usize,
    vector::{DenseVectorDescriptor, SparseVectorDescriptor},
};

/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including the
/// selected operation's descriptor ABI, sparse index bounds, and resource lifetimes.
///
/// # Errors
///
/// Returns an error if an operand or analysis descriptor belongs to another CUDA context.
/// A device result scalar must belong to the same CUDA context.
/// Also propagates output pointer-mode query or selection errors.
/// Returned workspace sizes must fit in `usize`.
/// Also propagates CUDA context-binding errors and any failure status returned by cuSPARSE.
pub unsafe fn spvv_buffer_size<Compute: DataTypeLike>(
    context: &Context,
    operation_x: Operation,
    x: &SparseVectorDescriptor,
    y: &DenseVectorDescriptor,
    result: &mut ScalarMut<'_, Compute>,
) -> Result<usize> {
    x.ensure_context(context)?;
    y.ensure_context(context)?;
    context.bind()?;
    context.require_output_scalar_pointer_mode(result)?;

    let mut size = 0;
    // SAFETY: `cusparseSpVV_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusparseSpVV_bufferSize(
            context.as_raw(),
            operation_x.into(),
            x.as_raw_const(),
            y.as_raw_const(),
            result.as_mut_ptr().cast(),
            Compute::data_type().into(),
            &raw mut size,
        ))?;
    }
    to_usize(size, "spvv buffer size")
}
