mod base;
mod copy_to;
mod permute;
mod slice;

use std::collections::HashMap;

use crate::{
    backend::cuda::{self, runtime::runtime_for_device, test_support::assert_device_mismatch},
    data_type::DataType,
    device::Device,
    error::Error,
    graph::Graph,
    shape::Shape,
    tensor::Tensor,
    testing::{maybe_lock_device, skip_if_stub},
};
