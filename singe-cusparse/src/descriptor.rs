//! Storage metadata and raw ownership for cuSPARSE data descriptors.

use std::sync::Arc;

use singe_cuda::{
    context::Context as CudaContext,
    view::{DeviceInput, DeviceOutputView, DeviceRegion, DeviceRepr},
};

use crate::{
    context::Context,
    error::{Error, Result},
};

/// A transferred opaque native resource and the CUDA context needed to destroy it.
/// External analysis workspaces remain subject to the raw execution contract.
#[derive(Debug)]
pub struct RawResource<H> {
    /// Native resource handle transferred to the caller.
    pub handle: H,
    /// CUDA context required to use and destroy the native resource.
    pub cuda_context: Arc<CudaContext>,
}

/// A transferred native descriptor and its actual CUDA context and buffer borrows.
///
/// Destroy `handle` with its corresponding cuSPARSE destroy function before
/// releasing these fields, or transfer the complete value back to its wrapper.
/// Regions describe available storage, including pending values and stride gaps.
/// Descriptors adopted without storage metadata have an empty region list;
/// their original unsafe lifetime and access obligations still apply.
#[derive(Debug)]
pub struct RawDescriptor<'a, H> {
    /// Native descriptor handle transferred to the caller.
    pub handle: H,
    /// Actual CUDA context associated with the descriptor and its storage.
    pub cuda_context: Arc<CudaContext>,
    /// Borrowed storage ranges retained by the descriptor, including absent optional arrays.
    pub regions: Vec<Option<DeviceRegion<'a>>>,
}

pub(crate) fn output_region<'a, T: DeviceRepr>(
    context: &Context,
    output: DeviceOutputView<'a, T>,
    minimum: usize,
    name: &str,
) -> Result<DeviceRegion<'a>> {
    output_region_for(context.cuda_context(), output, minimum, name)
}

pub(crate) fn output_region_for<'a, T: DeviceRepr>(
    context: &Arc<CudaContext>,
    output: DeviceOutputView<'a, T>,
    minimum: usize,
    name: &str,
) -> Result<DeviceRegion<'a>> {
    let () = T::LAYOUT;
    if output.context() != context {
        return Err(Error::PlanContextMismatch);
    }
    if output.len() < minimum {
        return Err(Error::LengthMismatch {
            name: name.to_owned().into(),
            expected: minimum,
            actual: output.len(),
        });
    }
    Ok(output.into_region())
}

pub(crate) fn input_region<'a, T: DeviceRepr>(
    context: &Context,
    input: &'a (impl DeviceInput<T> + ?Sized),
    minimum: usize,
    name: &str,
) -> Result<DeviceRegion<'a>> {
    let () = T::LAYOUT;
    if input.context() != context.cuda_context() {
        return Err(Error::PlanContextMismatch);
    }
    if input.len() < minimum {
        return Err(Error::LengthMismatch {
            name: name.to_owned().into(),
            expected: minimum,
            actual: input.len(),
        });
    }
    Ok(input.region())
}

pub(crate) fn matrix_extent(rows: usize, cols: usize, ld: usize, row_major: bool) -> Result<usize> {
    let (major, minor) = if row_major {
        (rows, cols)
    } else {
        (cols, rows)
    };
    if major == 0 || minor == 0 {
        return Ok(0);
    }
    if ld < minor {
        return Err(Error::out_of_range("leading dimension"));
    }
    (major - 1)
        .checked_mul(ld)
        .and_then(|value| value.checked_add(minor))
        .ok_or_else(|| Error::out_of_range("matrix extent"))
}

pub(crate) fn checked_add(value: usize, extra: usize, name: &str) -> Result<usize> {
    value
        .checked_add(extra)
        .ok_or_else(|| Error::out_of_range(name.to_owned()))
}

pub(crate) fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| Error::out_of_range(name.to_owned()))
}

pub(crate) fn workspace<'a>(
    context: &Context,
    buffer: Option<DeviceOutputView<'a, u8>>,
) -> Result<Option<DeviceRegion<'a>>> {
    buffer
        .map(|view| output_region(context, view, 0, "workspace"))
        .transpose()
}
