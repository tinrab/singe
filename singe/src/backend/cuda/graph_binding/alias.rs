use crate::{
    backend::cuda::{
        graph_binding::validation::validate_graph_tensor_metadata, tensor_region::CudaTensorSource,
    },
    data_type::DataType,
    device::Device,
    error::Result,
    layout::Layout,
};

pub fn validate_graph_alias_tensor_metadata(
    data_type: DataType,
    layout: &Layout,
    device: Device,
    tensor: &(impl CudaTensorSource + ?Sized),
) -> Result<()> {
    validate_graph_tensor_metadata("alias", data_type, layout, device, tensor)
}
