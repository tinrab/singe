use std::collections::HashMap;

use singe_core::assert_close;

use crate::{
    backend::{
        cuda::{self, eager::tensor_algebra::contraction, test_support::assert_shape_mismatch},
        registry::{Operation, operation_name},
    },
    data_type::DataType,
    device::Device,
    error::Error,
    graph::{Graph, GraphOperationOutput, GraphOperationRequest},
    shape::Shape,
    tensor::Tensor,
    testing::{maybe_lock_device, skip_if_stub},
};

fn operation_output(output: &Tensor) -> GraphOperationOutput {
    GraphOperationOutput::new(output.data_type(), output.shape().clone(), output.device())
}

mod compiled;
mod eager;
mod sync;
mod validation;
