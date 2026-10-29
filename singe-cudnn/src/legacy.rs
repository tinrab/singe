//! Legacy cuDNN descriptor and operation APIs.

/// Connectionist temporal classification loss descriptors, workspace queries, costs, and gradients.
pub mod ctc;
/// Dropout descriptors retaining RNG state, reserve storage, and forward/backward operations.
pub mod dropout;
/// Local response and divisive normalization descriptors and tensor operations.
pub mod lrn;
/// Standalone softmax and log-softmax modes and forward/backward operations.
pub mod softmax;
/// Spatial transformation descriptors, coordinate-grid generation, and bilinear sampling.
pub mod spatial_transformer;
