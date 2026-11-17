use std::sync::Arc;

use singe_core::{checked_arithmetic, checked_conversions};
use singe_cuda::{
    context::Context as CudaContext,
    error::Error as CudaError,
    view::{DeviceOutputView, DeviceRange, DeviceRegion, DeviceRepr},
};

use crate::{
    context::Context,
    error::{Error, Result},
    sys,
    types::Order,
};

checked_conversions!(Error {
    to_i64 -> i64,
    to_usize -> usize,
    to_i32 -> i32,
    to_u64 -> u64,
    to_size_t -> sys::size_t,
});

checked_arithmetic!(Error {
    checked_add,
    checked_mul,
    checked_pitched_extent,
});

pub(crate) fn output_region<'a, T: DeviceRepr>(
    context: &Context,
    output: DeviceOutputView<'a, T>,
    minimum: usize,
    name: &'static str,
) -> Result<DeviceRegion<'a>> {
    output_region_for(context.cuda_context(), output, minimum, name)
}

pub(crate) fn output_region_for<'a, T: DeviceRepr>(
    context: &Arc<CudaContext>,
    output: DeviceOutputView<'a, T>,
    minimum: usize,
    name: &'static str,
) -> Result<DeviceRegion<'a>> {
    let () = T::LAYOUT;
    if output.context() != context {
        return Err(Error::Cuda(CudaError::MemoryContextMismatch));
    }

    if output.len() < minimum {
        return Err(Error::BufferTooSmall {
            name: name.into(),
            required: minimum,
            actual: output.len(),
        });
    }
    Ok(output.into_region())
}

pub(crate) fn input_region<'a, T: DeviceRepr>(
    context: &Context,
    input: &'a (impl DeviceRange<T> + ?Sized),
    minimum: usize,
    name: &'static str,
) -> Result<DeviceRegion<'a>> {
    let () = T::LAYOUT;
    if input.context() != context.cuda_context() {
        return Err(Error::Cuda(CudaError::MemoryContextMismatch));
    }

    if input.len() < minimum {
        return Err(Error::BufferTooSmall {
            name: name.into(),
            required: minimum,
            actual: input.len(),
        });
    }
    Ok(input.region())
}

pub(crate) fn matrix_extent(rows: usize, cols: usize, ld: usize, order: Order) -> Result<usize> {
    let (major, minor) = match order {
        Order::Row => (rows, cols),
        Order::Column => (cols, rows),
    };
    if major == 0 || minor == 0 {
        return Ok(0);
    }

    if ld < minor {
        return Err(Error::out_of_range("leading dimension"));
    }
    checked_pitched_extent(major, ld, minor, "matrix extent")
}

pub(crate) fn workspace_region<'a>(
    context: &Context,
    workspace: Option<DeviceOutputView<'a, u8>>,
) -> Result<Option<DeviceRegion<'a>>> {
    workspace
        .map(|view| output_region(context, view, 0, "workspace"))
        .transpose()
}
