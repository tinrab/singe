use crate::{
    backend::{
        cuda::{
            cutensor::validate_cutensor_tensor_compatibility,
            tensor_region::{CudaTensorRegion, CudaTensorSource},
        },
        spec,
    },
    error::{Error, Result},
};

pub fn validate_cutensor_contraction_inputs(
    a: &dyn CudaTensorSource,
    a_mode: &[i64],
    b: &dyn CudaTensorSource,
    b_mode: &[i64],
    c: &dyn CudaTensorSource,
    c_mode: &[i64],
    d_mode: &[i64],
    output: CudaTensorRegion<'_>,
) -> Result<()> {
    validate_cutensor_tensor_compatibility(a, b, c, output)?;
    spec::contraction_problem(
        a.shape(),
        a_mode,
        b.shape(),
        b_mode,
        c.shape(),
        c_mode,
        d_mode,
    )?;
    if output.shape() != c.shape() {
        return Err(Error::shape_mismatch(
            "cuda cutensor contraction output shape",
            c.shape().dimensions(),
            output.shape().dimensions(),
        ));
    }
    Ok(())
}
