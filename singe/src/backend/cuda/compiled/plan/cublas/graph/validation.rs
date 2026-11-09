use crate::{
    backend::cuda::graph_value::CudaGraphValue,
    error::{Error, Result},
    graph::ir::{Graph, ValueId},
};

pub fn validate_cublaslt_matmul_input(
    ir: &Graph,
    id: ValueId,
    tensor: &CudaGraphValue,
) -> Result<()> {
    let value = id;
    if tensor.data_type() != ir.required_value_data_type(value)? {
        return Err(Error::unsupported_data_type("cuda graph cuBLASLt matmul"));
    }

    if tensor.shape().dimensions() != ir.required_value_shape(value)?.dimensions() {
        return Err(Error::shape_mismatch(
            "cuda graph cuBLASLt matmul input dimensions",
            ir.required_value_shape(value)?.dimensions().to_vec(),
            tensor.shape().dimensions().to_vec(),
        ));
    }

    if tensor.device() != ir.required_value(value)?.device() {
        return Err(Error::device_mismatch(
            "cuda graph cuBLASLt matmul input device",
            ir.required_value(value)?.device(),
            tensor.device(),
        ));
    }

    tensor.region()?;
    Ok(())
}
