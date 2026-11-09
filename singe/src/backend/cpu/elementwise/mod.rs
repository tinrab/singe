mod conversion;
mod lerp;
mod selection;

pub use self::{
    conversion::{cast, dequantize, quantize},
    lerp::lerp,
    selection::where_op,
};
use crate::{
    backend::{
        cpu::utility::{cpu_device_mismatch, get_f32_data},
        spec,
    },
    device::Device,
    error::Result,
    tensor::Tensor,
};

pub fn clamp(input: &Tensor, min: f32, max: f32) -> Result<Tensor> {
    if input.device() != Device::Cpu {
        return Err(cpu_device_mismatch("cpu operation input", input.device()));
    }

    let problem = spec::clamp_problem(input.shape(), min, max)?;

    let output_data: Vec<f32> = get_f32_data(input)?
        .iter()
        .map(|&value| {
            if value.is_nan() || min.is_nan() || max.is_nan() {
                f32::NAN
            } else if value < min {
                min
            } else if value > max {
                max
            } else {
                value
            }
        })
        .collect();

    Tensor::from_vec(Device::Cpu, problem.output_shape, output_data)
}
