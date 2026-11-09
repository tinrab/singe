//! CPU kernels grouped by operation family.

pub mod attention;
pub mod audio;
pub mod conv;
pub mod fft;
pub mod fused;
pub mod matmul;
pub mod moe;
pub mod normalization;
pub mod positional;
pub mod quantization;
pub mod recurrent;
pub mod shape;
