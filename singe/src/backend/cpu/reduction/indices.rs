use crate::{
    error::{Error, Result},
    utility::{checked_add, checked_mul, to_i64, to_usize},
};

pub fn enumerate_indices(dimensions: &[i64]) -> Result<Vec<Vec<i64>>> {
    if dimensions.is_empty() {
        return Ok(vec![Vec::new()]);
    }

    let dims = dimensions
        .iter()
        .copied()
        .enumerate()
        .map(|(axis, dim)| to_usize(dim, &format!("cpu enumerate dimension {axis}")))
        .collect::<Result<Vec<_>>>()?;

    let total = dims.iter().try_fold(1usize, |acc, &dim| {
        checked_mul(acc, dim, "cpu enumerate element count")
    })?;
    let mut all = Vec::with_capacity(total);

    for linear in 0..total {
        let mut remainder = linear;
        let mut indices = vec![0i64; dims.len()];
        for axis in (0..dims.len()).rev() {
            let dim = dims[axis];
            if dim == 0 {
                return Err(Error::invalid_argument(format!(
                    "cpu enumerate dimension {axis}"
                )));
            }

            indices[axis] = to_i64(remainder % dim, &format!("cpu enumerate index axis {axis}"))?;
            remainder /= dim;
        }

        all.push(indices);
    }

    Ok(all)
}

pub fn flatten_indices(indices: &[i64], strides: &[i64]) -> Result<usize> {
    indices
        .iter()
        .zip(strides.iter())
        .try_fold(0usize, |acc, (&index, &stride)| {
            let index = to_usize(index, "cpu reduction index")?;
            let stride = usize::try_from(stride).map_err(|_| Error::InvalidStride)?;
            let offset = checked_mul(index, stride, "cpu reduction offset")?;
            checked_add(acc, offset, "cpu reduction offset")
        })
}
