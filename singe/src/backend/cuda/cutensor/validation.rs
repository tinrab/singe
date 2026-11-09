use crate::{
    backend::cuda::{
        tensor_region::{CudaTensorRegion, CudaTensorSource},
        utility::{ensure_f32, ensure_same_device},
    },
    data_type::DataType,
    error::{Error, Result},
};

pub fn validate_cutensor_tensor_compatibility(
    a: &dyn CudaTensorSource,
    b: &dyn CudaTensorSource,
    c: &dyn CudaTensorSource,
    output: CudaTensorRegion<'_>,
) -> Result<()> {
    ensure_same_device(a, b)?;
    ensure_same_device(a, c)?;
    if a.device() != output.device() {
        return Err(Error::device_mismatch(
            "cuTENSOR output device",
            a.device(),
            output.device(),
        ));
    }

    ensure_f32(a)?;
    ensure_f32(b)?;
    ensure_f32(c)?;
    if output.data_type() != DataType::F32 {
        return Err(Error::unsupported_data_type("cuTENSOR output"));
    }
    Ok(())
}
