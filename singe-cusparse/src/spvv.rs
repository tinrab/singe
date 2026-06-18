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

pub fn spvv_buffer_size<Compute: DataTypeLike>(
    ctx: &Context,
    operation_x: Operation,
    x: &SparseVectorDescriptor,
    y: &DenseVectorDescriptor,
    result: &mut ScalarMut<'_, Compute>,
) -> Result<usize> {
    x.ensure_context(ctx)?;
    y.ensure_context(ctx)?;
    ctx.bind()?;
    if ctx.scalar_pointer_mode()? != result.pointer_mode() {
        ctx.set_scalar_pointer_mode(result.pointer_mode())?;
    }

    let mut size = 0;
    unsafe {
        try_ffi!(sys::cusparseSpVV_bufferSize(
            ctx.as_raw(),
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
