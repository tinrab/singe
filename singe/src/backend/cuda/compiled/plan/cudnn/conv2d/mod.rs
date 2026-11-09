mod bias_relu;
mod executable;
mod plain;
mod validation;

pub use self::{
    bias_relu::{CudnnConv2dGeometry, build_cudnn_conv2d_bias_relu_graph},
    plain::build_cudnn_convolution_graph,
};
