use crate::{
    backend::{
        cpu::utility::{
            broadcast_access, broadcast_access_swapped, broadcast_offset_for_spec,
            cpu_device_mismatch, get_f32_data, linear_to_indices,
        },
        spec,
    },
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    tensor::Tensor,
};

pub fn lerp(a: &Tensor, b: &Tensor, t: f32) -> Result<Tensor> {
    if a.device() != Device::Cpu || b.device() != Device::Cpu {
        let actual = if a.device() == Device::Cpu {
            b.device()
        } else {
            a.device()
        };
        return Err(cpu_device_mismatch("cpu lerp input", actual));
    }

    if a.data_type() != DataType::F32 || b.data_type() != DataType::F32 {
        return Err(Error::unsupported_data_type("cpu tensor operation"));
    }

    let a = a.contiguous()?;
    let b = b.contiguous()?;
    let problem = spec::lerp_problem(a.shape(), b.shape(), t)?;
    let a_data = get_f32_data(&a)?;
    let b_data = get_f32_data(&b)?;
    let broadcast = spec::BinaryBroadcastProblem {
        output_shape: problem.output_shape.clone(),
        lhs_padded_dimensions: problem.lhs_padded_dimensions,
        rhs_padded_dimensions: problem.rhs_padded_dimensions,
    };
    let a_access = broadcast_access(a.layout(), &broadcast);
    let b_access = broadcast_access_swapped(b.layout(), &broadcast);
    let out_shape = problem.output_shape;

    let output_element_count = out_shape.element_count();
    let mut output_data = Vec::with_capacity(output_element_count);
    for linear in 0..output_element_count {
        let indices = linear_to_indices(linear, out_shape.dimensions())?;
        let a_val = a_data[broadcast_offset_for_spec(&indices, &a_access)?];
        let b_val = b_data[broadcast_offset_for_spec(&indices, &b_access)?];
        output_data.push(t.mul_add(b_val - a_val, a_val));
    }

    drop(b_data);
    drop(a_data);

    Tensor::from_vec(Device::Cpu, out_shape, output_data)
}
