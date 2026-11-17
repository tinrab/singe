use std::ptr;

use singe_cuda::{
    types::{Complex32, Complex64},
    view::{DeviceOutputView, DeviceViewMut},
};

use crate::{
    context::Context,
    dense::validation::{
        require_info_buffer, require_info_entries, require_workspace,
        validate_batched_square_matrix_pointers, validate_batched_vector_pointers,
        validate_square_matrix,
    },
    error::{Error, Result},
    layout::{BatchedMatrixRef, BatchedVectorRef, MatrixLayout},
    sys, try_ffi,
    types::FillMode,
    utility::{to_i32, to_usize},
};

macro_rules! potrf_batched_by_precision {
    ($($name:ident, $ty:ty, $sys:ident;)+) => {
        $(
            pub(crate) unsafe fn $name(
                context: &Context,
                fill_mode: FillMode,
                n: usize,
                a: BatchedMatrixRef<'_, $ty>,
                mut info: DeviceOutputView<'_, i32>,
            ) -> Result<()> {
                context.bind()?;
                context.prepare_region(&info)?;
                context.prepare_region(&a.pointers)?;
                validate_batched_square_matrix_pointers(n, a)?;
                require_info_entries(info.len(), a.len())?;
                // SAFETY: the context handle is live and bound, the pointer tables and info buffer were checked against `n` and the batch size above, and their regions belong to the context.
                // The caller retains every pointee through completion under the raw execution contract.
                unsafe {
                    try_ffi!(sys::$sys(
                        context.as_raw(),
                        fill_mode.into(),
                        to_i32(n, "n")?,
                        a.as_mut_ptr().cast(),
                        to_i32(a.leading_dimension, "lda")?,
                        info.as_mut_ptr().cast(),
                        to_i32(a.len(), "batch_size")?,
                    ))?;
                }
                Ok(())
            }
        )+
    };
}

potrf_batched_by_precision! {
    spotrf_batched, f32, cusolverDnSpotrfBatched;
    dpotrf_batched, f64, cusolverDnDpotrfBatched;
    cpotrf_batched, Complex32, cusolverDnCpotrfBatched;
    zpotrf_batched, Complex64, cusolverDnZpotrfBatched;
}

macro_rules! potrs_batched_by_precision {
    ($($name:ident, $ty:ty, $sys:ident;)+) => {
        $(
            pub(crate) unsafe fn $name(
                context: &Context,
                fill_mode: FillMode,
                n: usize,
                a: BatchedMatrixRef<'_, $ty>,
                b: BatchedVectorRef<'_, $ty>,
                mut info: DeviceOutputView<'_, i32>,
            ) -> Result<()> {
                context.bind()?;
                context.prepare_region(&info)?;
                context.prepare_region(&a.pointers)?;
                context.prepare_region(&b.pointers)?;
                validate_batched_square_matrix_pointers(n, a)?;
                validate_batched_vector_pointers(n, b)?;
                require_info_buffer(info.len())?;
                if a.len() != b.len() {
                    return Err(Error::InvalidMatrixShape);
                }

                // SAFETY: the context handle is live and bound, the pointer tables and info buffer were checked against `n` and the batch size above, and their regions belong to the context.
                // The caller retains every pointee through completion under the raw execution contract.
                unsafe {
                    try_ffi!(sys::$sys(
                        context.as_raw(),
                        fill_mode.into(),
                        to_i32(n, "n")?,
                        1,
                        a.as_mut_ptr().cast(),
                        to_i32(a.leading_dimension, "lda")?,
                        b.as_mut_ptr().cast(),
                        to_i32(b.leading_dimension, "ldb")?,
                        info.as_mut_ptr(),
                        to_i32(a.len(), "batch_size")?,
                    ))?;
                }
                Ok(())
            }
        )+
    };
}

potrs_batched_by_precision! {
    spotrs_batched, f32, cusolverDnSpotrsBatched;
    dpotrs_batched, f64, cusolverDnDpotrsBatched;
    cpotrs_batched, Complex32, cusolverDnCpotrsBatched;
    zpotrs_batched, Complex64, cusolverDnZpotrsBatched;
}

macro_rules! potri_buffer_size_by_precision {
    ($($name:ident, $ty:ty, $sys:ident;)+) => {
        $(
            pub(crate) fn $name(
                context: &Context,
                fill_mode: FillMode,
                n: usize,
                lda: usize,
            ) -> Result<usize> {
                context.bind()?;
                MatrixLayout::new(lda).validate(n, n)?;
                let mut lwork = 0;
                // SAFETY: the context handle is live and bound, the dimensions were validated above, and the call writes only the local `lwork`.
                unsafe {
                    try_ffi!(sys::$sys(
                        context.as_raw(),
                        fill_mode.into(),
                        to_i32(n, "n")?,
                        ptr::null_mut::<u8>().cast(),
                        to_i32(lda, "lda")?,
                        &raw mut lwork,
                    ))?;
                }
                to_usize(lwork, "lwork")
            }
        )+
    };
}

potri_buffer_size_by_precision! {
    spotri_buffer_size, f32, cusolverDnSpotri_bufferSize;
    dpotri_buffer_size, f64, cusolverDnDpotri_bufferSize;
    cpotri_buffer_size, Complex32, cusolverDnCpotri_bufferSize;
    zpotri_buffer_size, Complex64, cusolverDnZpotri_bufferSize;
}

macro_rules! potri_by_precision {
    ($($name:ident, $ty:ty, $sys:ident, $buffer_size:ident;)+) => {
        $(
            pub(crate) unsafe fn $name(
                context: &Context,
                fill_mode: FillMode,
                n: usize,
                mut a: DeviceViewMut<'_, $ty>,
                lda: usize,
                mut workspace: DeviceOutputView<'_, $ty>,
                mut dev_info: DeviceOutputView<'_, i32>,
            ) -> Result<()> {
                context.bind()?;
                context.prepare_region(&a)?;
                context.prepare_region(&workspace)?;
                context.prepare_region(&dev_info)?;
                validate_square_matrix(n, a.len(), lda)?;
                require_info_buffer(dev_info.len())?;
                let lwork = $buffer_size(context, fill_mode, n, lda)?;
                require_workspace(workspace.len(), lwork)?;
                // SAFETY: the context handle is live and bound, every buffer was checked above against the dimensions and the required workspace and belongs to the context.
                // The caller retains the buffers through completion under the raw execution contract.
                unsafe {
                    try_ffi!(sys::$sys(
                        context.as_raw(),
                        fill_mode.into(),
                        to_i32(n, "n")?,
                        a.as_mut_ptr().cast(),
                        to_i32(lda, "lda")?,
                        workspace.as_mut_ptr().cast(),
                        to_i32(lwork, "lwork")?,
                        dev_info.as_mut_ptr().cast(),
                    ))?;
                }
                Ok(())
            }
        )+
    };
}

potri_by_precision! {
    spotri, f32, cusolverDnSpotri, spotri_buffer_size;
    dpotri, f64, cusolverDnDpotri, dpotri_buffer_size;
    cpotri, Complex32, cusolverDnCpotri, cpotri_buffer_size;
    zpotri, Complex64, cusolverDnZpotri, zpotri_buffer_size;
}
