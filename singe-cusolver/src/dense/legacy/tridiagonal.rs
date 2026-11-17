use std::ptr;

use singe_cuda::{
    types::{Complex32, Complex64},
    view::{DeviceOutputView, DeviceView, DeviceViewMut},
};

use crate::{
    context::Context,
    dense::validation::{
        require_info_buffer, require_workspace, tridiagonal_order, validate_orgtr_inputs,
        validate_ormtr_inputs, validate_sytrd_inputs,
    },
    error::Result,
    layout::MatrixLayout,
    sys, try_ffi,
    types::{FillMode, Operation, SideMode},
    utility::{to_i32, to_usize},
};

macro_rules! sytrd_buffer_size_by_precision {
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
                        ptr::null_mut::<u8>().cast(),
                        ptr::null_mut::<u8>().cast(),
                        ptr::null_mut::<u8>().cast(),
                        &raw mut lwork,
                    ))?;
                }
                to_usize(lwork, "lwork")
            }
        )+
    };
}

sytrd_buffer_size_by_precision! {
    ssytrd_buffer_size, f32, cusolverDnSsytrd_bufferSize;
    dsytrd_buffer_size, f64, cusolverDnDsytrd_bufferSize;
    chetrd_buffer_size, Complex32, cusolverDnChetrd_bufferSize;
    zhetrd_buffer_size, Complex64, cusolverDnZhetrd_bufferSize;
}

macro_rules! sytrd_by_precision {
    ($($name:ident, $ty:ty, $real:ty, $sys:ident, $buffer_size:ident;)+) => {
        $(
            pub(crate) unsafe fn $name(
                context: &Context,
                fill_mode: FillMode,
                n: usize,
                mut a: DeviceViewMut<'_, $ty>,
                lda: usize,
                mut d: DeviceOutputView<'_, $real>,
                mut e: DeviceOutputView<'_, $real>,
                mut tau: DeviceOutputView<'_, $ty>,
                mut workspace: DeviceOutputView<'_, $ty>,
                mut dev_info: DeviceOutputView<'_, i32>,
            ) -> Result<()> {
                context.bind()?;
                context.prepare_region(&a)?;
                context.prepare_region(&d)?;
                context.prepare_region(&e)?;
                context.prepare_region(&tau)?;
                context.prepare_region(&workspace)?;
                context.prepare_region(&dev_info)?;
                validate_sytrd_inputs(n, a.len(), lda, d.len(), e.len(), tau.len())?;
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
                        d.as_mut_ptr().cast(),
                        e.as_mut_ptr().cast(),
                        tau.as_mut_ptr().cast(),
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

sytrd_by_precision! {
    ssytrd, f32, f32, cusolverDnSsytrd, ssytrd_buffer_size;
    dsytrd, f64, f64, cusolverDnDsytrd, dsytrd_buffer_size;
    chetrd, Complex32, f32, cusolverDnChetrd, chetrd_buffer_size;
    zhetrd, Complex64, f64, cusolverDnZhetrd, zhetrd_buffer_size;
}

macro_rules! orgtr_buffer_size_by_precision {
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
                        ptr::null_mut::<u8>().cast(),
                        &raw mut lwork,
                    ))?;
                }
                to_usize(lwork, "lwork")
            }
        )+
    };
}

orgtr_buffer_size_by_precision! {
    sorgtr_buffer_size, f32, cusolverDnSorgtr_bufferSize;
    dorgtr_buffer_size, f64, cusolverDnDorgtr_bufferSize;
    cungtr_buffer_size, Complex32, cusolverDnCungtr_bufferSize;
    zungtr_buffer_size, Complex64, cusolverDnZungtr_bufferSize;
}

macro_rules! orgtr_by_precision {
    ($($name:ident, $ty:ty, $sys:ident, $buffer_size:ident;)+) => {
        $(
            pub(crate) unsafe fn $name(
                context: &Context,
                fill_mode: FillMode,
                n: usize,
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
                validate_orgtr_inputs(n, a.len(), lda, tau.len())?;
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

orgtr_by_precision! {
    sorgtr, f32, cusolverDnSorgtr, sorgtr_buffer_size;
    dorgtr, f64, cusolverDnDorgtr, dorgtr_buffer_size;
    cungtr, Complex32, cusolverDnCungtr, cungtr_buffer_size;
    zungtr, Complex64, cusolverDnZungtr, zungtr_buffer_size;
}

macro_rules! ormtr_buffer_size_by_precision {
    ($($name:ident, $ty:ty, $sys:ident;)+) => {
        $(
            pub(crate) fn $name(
                context: &Context,
                side: SideMode,
                fill_mode: FillMode,
                operation: Operation,
                m: usize,
                n: usize,
                lda: usize,
                ldc: usize,
            ) -> Result<usize> {
                context.bind()?;
                MatrixLayout::new(lda)
                    .validate(tridiagonal_order(side, m, n), tridiagonal_order(side, m, n))
                    .and_then(|()| MatrixLayout::new(ldc).validate(m, n))?;
                let mut lwork = 0;
                // SAFETY: the context handle is live and bound, the dimensions were validated above, and the call writes only the local `lwork`.
                unsafe {
                    try_ffi!(sys::$sys(
                        context.as_raw(),
                        side.into(),
                        fill_mode.into(),
                        operation.into(),
                        to_i32(m, "m")?,
                        to_i32(n, "n")?,
                        ptr::null_mut::<u8>().cast(),
                        to_i32(lda, "lda")?,
                        ptr::null_mut::<u8>().cast(),
                        ptr::null_mut::<u8>().cast(),
                        to_i32(ldc, "ldc")?,
                        &raw mut lwork,
                    ))?;
                }
                to_usize(lwork, "lwork")
            }
        )+
    };
}

ormtr_buffer_size_by_precision! {
    sormtr_buffer_size, f32, cusolverDnSormtr_bufferSize;
    dormtr_buffer_size, f64, cusolverDnDormtr_bufferSize;
    cunmtr_buffer_size, Complex32, cusolverDnCunmtr_bufferSize;
    zunmtr_buffer_size, Complex64, cusolverDnZunmtr_bufferSize;
}

macro_rules! ormtr_by_precision {
    ($($name:ident, $ty:ty, $sys:ident, $buffer_size:ident;)+) => {
        $(
            pub(crate) unsafe fn $name(
                context: &Context,
                side: SideMode,
                fill_mode: FillMode,
                operation: Operation,
                m: usize,
                n: usize,
                a: DeviceView<'_, $ty>,
                lda: usize,
                tau: DeviceView<'_, $ty>,
                mut c: DeviceViewMut<'_, $ty>,
                ldc: usize,
                mut workspace: DeviceOutputView<'_, $ty>,
                mut dev_info: DeviceOutputView<'_, i32>,
            ) -> Result<()> {
                context.bind()?;
                context.prepare_region(&a)?;
                context.prepare_region(&tau)?;
                context.prepare_region(&c)?;
                context.prepare_region(&workspace)?;
                context.prepare_region(&dev_info)?;
                validate_ormtr_inputs(side, m, n, a.len(), lda, tau.len(), c.len(), ldc)?;
                require_info_buffer(dev_info.len())?;
                let lwork = $buffer_size(context, side, fill_mode, operation, m, n, lda, ldc)?;
                require_workspace(workspace.len(), lwork)?;
                // SAFETY: the context handle is live and bound, every buffer was checked above against the dimensions and the required workspace and belongs to the context.
                // The caller retains the buffers through completion under the raw execution contract.
                unsafe {
                    try_ffi!(sys::$sys(
                        context.as_raw(),
                        side.into(),
                        fill_mode.into(),
                        operation.into(),
                        to_i32(m, "m")?,
                        to_i32(n, "n")?,
                        a.as_ptr().cast_mut().cast(),
                        to_i32(lda, "lda")?,
                        tau.as_ptr().cast_mut().cast(),
                        c.as_mut_ptr().cast(),
                        to_i32(ldc, "ldc")?,
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

ormtr_by_precision! {
    sormtr, f32, cusolverDnSormtr, sormtr_buffer_size;
    dormtr, f64, cusolverDnDormtr, dormtr_buffer_size;
    cunmtr, Complex32, cusolverDnCunmtr, cunmtr_buffer_size;
    zunmtr, Complex64, cusolverDnZunmtr, zunmtr_buffer_size;
}
