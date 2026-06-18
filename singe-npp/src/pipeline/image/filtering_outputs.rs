use singe_cuda::memory::DeviceMemory;

use crate::{
    error::Result,
    image::{memory::Image, view::C1},
    types::PointPolar,
};

#[derive(Debug)]
pub struct HoughLines {
    pub lines: DeviceMemory<PointPolar>,
    pub line_count: DeviceMemory<i32>,
}

impl HoughLines {
    pub fn create(max_line_count: usize) -> Result<Self> {
        Ok(Self {
            lines: DeviceMemory::create(max_line_count)?,
            line_count: DeviceMemory::create(1)?,
        })
    }
}

#[derive(Debug)]
pub struct GradientVector<T> {
    pub x: Image<T, C1>,
    pub y: Image<T, C1>,
    pub magnitude: Image<T, C1>,
    pub angle: Image<f32, C1>,
}

#[derive(Debug)]
pub struct HistogramOfGradients {
    pub descriptors: DeviceMemory<f32>,
    pub descriptor_bytes: usize,
}
