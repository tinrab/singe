#![allow(dead_code)]

use singe_cuda::view::{DeviceRepr, DeviceSlice};

use crate::{error::Result, utility::ensure_len};

pub fn ensure_binary_lengths<T: DeviceRepr>(
    out: &impl DeviceSlice<T>,
    lhs: &impl DeviceSlice<T>,
    rhs: &impl DeviceSlice<T>,
) -> Result<usize> {
    let len = out.len();
    ensure_len(lhs.len(), len)?;
    ensure_len(rhs.len(), len)?;
    Ok(len)
}

pub fn ensure_unary_lengths<T: DeviceRepr>(
    out: &impl DeviceSlice<T>,
    input: &impl DeviceSlice<T>,
) -> Result<usize> {
    let len = out.len();
    ensure_len(input.len(), len)?;
    Ok(len)
}
