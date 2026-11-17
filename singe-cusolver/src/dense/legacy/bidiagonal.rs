use std::ptr;

use singe_cuda::{
    types::{Complex32, Complex64},
    view::{DeviceOutputView, DeviceView, DeviceViewMut},
};

use crate::{
    context::Context,
    dense::validation::{
        require_info_buffer, require_workspace, validate_bidiagonal_buffers,
        validate_bidiagonal_dims, validate_orgbr_inputs, validate_orgbr_shape,
    },
    error::Result,
    layout::MatrixLayout,
    sys, try_ffi,
    types::SideMode,
    utility::{to_i32, to_usize},
};

macro_rules! gebrd_buffer_size_by_precision {
    ($($name:ident, $ty:ty, $sys:ident;)+) => {
        $(
            pub(crate) fn $name(context: &Context, m: usize, n: usize) -> Result<usize> {
                context.bind()?;
                validate_bidiagonal_dims(m, n)?;
                let mut lwork = 0;
                // SAFETY: the context handle is live and bound, the dimensions were validated above, and the call writes only the local `lwork`.
                unsafe {
                    try_ffi!(sys::$sys(
                        context.as_raw(),
                        to_i32(m, "m")?,
                        to_i32(n, "n")?,
                        &raw mut lwork,
                    ))?;
                }
                to_usize(lwork, "lwork")
            }
        )+
    };
}

gebrd_buffer_size_by_precision! {
    sgebrd_buffer_size, f32, cusolverDnSgebrd_bufferSize;
    dgebrd_buffer_size, f64, cusolverDnDgebrd_bufferSize;
    cgebrd_buffer_size, Complex32, cusolverDnCgebrd_bufferSize;
    zgebrd_buffer_size, Complex64, cusolverDnZgebrd_bufferSize;
}

macro_rules! gebrd_by_precision {
    ($($name:ident, $ty:ty, $real:ty, $sys:ident, $buffer_size:ident;)+) => {
        $(
            pub(crate) unsafe fn $name(
                context: &Context,
                m: usize,
                n: usize,
                mut a: DeviceViewMut<'_, $ty>,
                lda: usize,
                mut d: DeviceOutputView<'_, $real>,
                mut e: DeviceOutputView<'_, $real>,
                mut tauq: DeviceOutputView<'_, $ty>,
                mut taup: DeviceOutputView<'_, $ty>,
                mut workspace: DeviceOutputView<'_, $ty>,
                mut dev_info: DeviceOutputView<'_, i32>,
            ) -> Result<()> {
                context.bind()?;
                context.prepare_region(&a)?;
                context.prepare_region(&d)?;
                context.prepare_region(&e)?;
                context.prepare_region(&tauq)?;
                context.prepare_region(&taup)?;
                context.prepare_region(&workspace)?;
                context.prepare_region(&dev_info)?;
                validate_bidiagonal_buffers(m, n, a.len(), lda, d.len(), e.len(), tauq.len(), taup.len())?;
                require_info_buffer(dev_info.len())?;
                let lwork = $buffer_size(context, m, n)?;
                require_workspace(workspace.len(), lwork)?;
                // SAFETY: the context handle is live and bound, every buffer was checked above against the dimensions and the required workspace and belongs to the context.
                // The caller retains the buffers through completion under the raw execution contract.
                unsafe {
                    try_ffi!(sys::$sys(
                        context.as_raw(),
                        to_i32(m, "m")?,
                        to_i32(n, "n")?,
                        a.as_mut_ptr().cast(),
                        to_i32(lda, "lda")?,
                        d.as_mut_ptr().cast(),
                        e.as_mut_ptr().cast(),
                        tauq.as_mut_ptr().cast(),
                        taup.as_mut_ptr().cast(),
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

gebrd_by_precision! {
    sgebrd, f32, f32, cusolverDnSgebrd, sgebrd_buffer_size;
    dgebrd, f64, f64, cusolverDnDgebrd, dgebrd_buffer_size;
    cgebrd, Complex32, f32, cusolverDnCgebrd, cgebrd_buffer_size;
    zgebrd, Complex64, f64, cusolverDnZgebrd, zgebrd_buffer_size;
}

macro_rules! orgbr_buffer_size_by_precision {
    ($($name:ident, $ty:ty, $sys:ident;)+) => {
        $(
            pub(crate) fn $name(
                context: &Context,
                side: SideMode,
                m: usize,
                n: usize,
                k: usize,
                lda: usize,
            ) -> Result<usize> {
                context.bind()?;
                validate_orgbr_shape(side, m, n, k)?;
                MatrixLayout::new(lda).validate(m, n)?;
                let mut lwork = 0;
                // SAFETY: the context handle is live and bound, the dimensions were validated above, and the call writes only the local `lwork`.
                unsafe {
                    try_ffi!(sys::$sys(
                        context.as_raw(),
                        side.into(),
                        to_i32(m, "m")?,
                        to_i32(n, "n")?,
                        to_i32(k, "k")?,
                        ptr::null_mut::<u8>().cast(),
                        to_i32(lda, "lda")?,
                        ptr::null_mut::<u8>().cast(),
                        &raw mut lwork,
                    ))?;
                }
                to_usize(lwork, "lwork")
            }
        )+
    };
}

orgbr_buffer_size_by_precision! {
    sorgbr_buffer_size, f32, cusolverDnSorgbr_bufferSize;
    dorgbr_buffer_size, f64, cusolverDnDorgbr_bufferSize;
    cungbr_buffer_size, Complex32, cusolverDnCungbr_bufferSize;
    zungbr_buffer_size, Complex64, cusolverDnZungbr_bufferSize;
}

macro_rules! orgbr_by_precision {
    ($($name:ident, $ty:ty, $sys:ident, $buffer_size:ident;)+) => {
        $(
            pub(crate) unsafe fn $name(
                context: &Context,
                side: SideMode,
                m: usize,
                n: usize,
                k: usize,
                mut a: DeviceViewMut<'_, $ty>,
                lda: usize,
                tau: DeviceView<'_, $ty>,
                mut workspace: DeviceOutputView<'_, $ty>,
                mut dev_info: DeviceOutputView<'_, i32>,
            ) -> Result<()> {
                context.bind()?;
                context.prepare_region(&a)?;
                context.prepare_region(&tau)?;
                context.prepare_region(&workspace)?;
                context.prepare_region(&dev_info)?;
                validate_orgbr_inputs(side, m, n, k, a.len(), lda, tau.len())?;
                require_info_buffer(dev_info.len())?;
                let lwork = $buffer_size(context, side, m, n, k, lda)?;
                require_workspace(workspace.len(), lwork)?;
                // SAFETY: the context handle is live and bound, every buffer was checked above against the dimensions and the required workspace and belongs to the context.
                // The caller retains the buffers through completion under the raw execution contract.
                unsafe {
                    try_ffi!(sys::$sys(
                        context.as_raw(),
                        side.into(),
                        to_i32(m, "m")?,
                        to_i32(n, "n")?,
                        to_i32(k, "k")?,
                        a.as_mut_ptr().cast(),
                        to_i32(lda, "lda")?,
                        tau.as_ptr().cast(),
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

orgbr_by_precision! {
    sorgbr, f32, cusolverDnSorgbr, sorgbr_buffer_size;
    dorgbr, f64, cusolverDnDorgbr, dorgbr_buffer_size;
    cungbr, Complex32, cusolverDnCungbr, cungbr_buffer_size;
    zungbr, Complex64, cusolverDnZungbr, zungbr_buffer_size;
}
