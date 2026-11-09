mod execute;
mod indices;
mod modes;

use self::{
    execute::{Contraction, ContractionOperand, execute_contraction},
    indices::enumerate_indices,
    modes::{contraction_modes, mode_dims},
};
use crate::{
    backend::{
        cpu::utility::{cpu_device_mismatch, get_f32_data},
        spec::contraction_problem,
    },
    device::Device,
    error::{Error, Result},
    tensor::Tensor,
};

pub fn contraction(
    a: &Tensor,
    a_mode: &[i64],
    b: &Tensor,
    b_mode: &[i64],
    c: &Tensor,
    c_mode: &[i64],
    d_mode: &[i64],
    alpha: f32,
    beta: f32,
) -> Result<Tensor> {
    let device = a.device();
    if device != b.device() || device != c.device() {
        let actual = [b.device(), c.device()]
            .into_iter()
            .find(|candidate| *candidate != device)
            .unwrap_or(device);
        return Err(Error::device_mismatch(
            "cpu contraction input device",
            device,
            actual,
        ));
    }

    if !device.is_cpu() {
        return Err(cpu_device_mismatch("cpu contraction input", device));
    }

    let spec = contraction_problem(
        a.shape(),
        a_mode,
        b.shape(),
        b_mode,
        c.shape(),
        c_mode,
        d_mode,
    )?;

    let a = a.contiguous()?;
    let b = b.contiguous()?;
    let c = c.contiguous()?;

    let a_data = get_f32_data(&a)?;
    let b_data = get_f32_data(&b)?;
    let c_data = get_f32_data(&c)?;

    let output_shape = spec.output_shape;
    let output_element_count = output_shape.element_count();
    if output_element_count == 0 {
        return Tensor::from_vec(Device::Cpu, output_shape, Vec::<f32>::new());
    }

    let mut output = vec![0.0f32; output_element_count];

    let mode_dims = mode_dims(a.shape(), a_mode, b.shape(), b_mode, c.shape(), c_mode)?;
    let reduction_modes = contraction_modes(a_mode, b_mode, d_mode);
    let reduction_shape = reduction_modes
        .iter()
        .map(|mode| {
            mode_dims.get(mode).copied().ok_or_else(|| {
                Error::out_of_range(format!("cpu contraction reduction mode {mode}"))
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let output_indices = enumerate_indices(output_shape.dimensions())?;
    let reduction_indices = enumerate_indices(&reduction_shape)?;

    execute_contraction(
        &Contraction {
            a: ContractionOperand {
                modes: a_mode,
                strides: a.layout().strides(),
                data: &a_data,
            },
            b: ContractionOperand {
                modes: b_mode,
                strides: b.layout().strides(),
                data: &b_data,
            },
            c: ContractionOperand {
                modes: c_mode,
                strides: c.layout().strides(),
                data: &c_data,
            },
            d_modes: d_mode,
            reduction_modes: &reduction_modes,
            output_indices: &output_indices,
            reduction_indices: &reduction_indices,
            alpha,
            beta,
        },
        &mut output,
    )?;

    Tensor::from_vec(Device::Cpu, output_shape, output)
}
