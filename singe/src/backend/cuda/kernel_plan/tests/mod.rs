mod compiled_graph;
mod neural;
mod rejections;
mod shape;
mod support;
mod unsupported;

use std::collections::HashMap;

use singe_core::assert_close;

pub use self::support::*;
use crate::{
    backend::{
        cuda::{self, CudaGraphProvider, kernels as cuda_kernels},
        registry::{Operation, OperationAttributes, operation_name},
    },
    data_type::DataType,
    device::Device,
    error::{Error, UnsupportedTarget},
    graph::{Graph, GraphOperationOutput, GraphOperationRequest},
    shape::Shape,
    tensor::Tensor,
    testing::{maybe_lock_device, skip_if_stub},
};
