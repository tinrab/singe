use std::{cmp::Ordering, mem::size_of};

use singe_cuda::view::{DeviceOutputView, DeviceRegionProvider, DeviceRepr, DeviceView};

use crate::{
    data_type::SolverReal,
    eigen::EigenSelection,
    error::{Error, Result},
    layout::MatrixLayout,
    types::EigenRange,
    utility::{to_i32, to_usize},
};

pub fn batch_layout(
    n: usize,
    layout: MatrixLayout,
    batch: usize,
    native_limit: bool,
) -> Result<usize> {
    if batch == 0 {
        return Err(Error::InvalidMatrixShape);
    }
    let values = n.checked_mul(batch).ok_or(Error::InvalidMatrixShape)?;
    layout.validate(n, values)?;
    if native_limit {
        let elements = layout
            .leading_dimension
            .checked_mul(values)
            .ok_or(Error::InvalidMatrixShape)?;
        to_i32(elements, "batched problem size")?;
    }
    Ok(values)
}

pub fn output<T: DeviceRepr>(value: &DeviceOutputView<'_, T>, required: usize) -> Result<()> {
    if value.len() < required {
        return Err(Error::InvalidVectorShape);
    }
    Ok(())
}

pub fn workspace(actual: usize, required: usize) -> Result<()> {
    if actual < required {
        return Err(Error::InsufficientWorkspaceSize { required, actual });
    }
    Ok(())
}

pub fn native_selection<T: SolverReal>(
    selection: &EigenSelection<T>,
    n: usize,
) -> Result<(EigenRange, &T, &T, usize, usize)> {
    match selection {
        EigenSelection::All => Ok((EigenRange::All, T::zero_ref(), T::zero_ref(), 0, 0)),
        EigenSelection::ByValue {
            lower_exclusive,
            upper_inclusive,
        } => {
            if lower_exclusive.partial_cmp(upper_inclusive) != Some(Ordering::Less) {
                return Err(Error::InvalidEigenRange);
            }
            Ok((EigenRange::Value, lower_exclusive, upper_inclusive, 0, 0))
        }
        EigenSelection::ByIndex(range) => {
            if range.start >= range.end || range.end > n {
                return Err(Error::InvalidEigenRange);
            }
            let first = range.start.checked_add(1).ok_or(Error::InvalidEigenRange)?;
            Ok((
                EigenRange::Index,
                T::zero_ref(),
                T::zero_ref(),
                first,
                range.end,
            ))
        }
    }
}

pub fn check_completed_status(info: &DeviceOutputView<'_, i32>) -> Result<()> {
    // SAFETY: a completed successful native call initializes its first status entry.
    let value =
        unsafe { DeviceView::<i32>::from_raw_parts(info.region().slice(..size_of::<i32>())?, 1)? };
    let info = value.copy_to_host_vec()?[0];
    if info != 0 {
        return Err(Error::ComputationFailed { info });
    }
    Ok(())
}

pub fn selected_count(value: impl TryInto<usize>, maximum: usize) -> Result<usize> {
    let actual = to_usize(value, "selected eigenvalue count")?;
    if actual > maximum {
        return Err(Error::InvalidEigenvalueCount { actual, maximum });
    }
    Ok(actual)
}
