use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory};
use singe_cutensor::{
    context::Context,
    error::Result,
    operation::{ComputeDescriptor, OperationDescriptor, TensorOperand},
    plan::{Plan, PlanPreference},
    tensor::TensorDescriptor,
    types::WorkspacePreference,
};

fn packed_offset(indices: &[u64], extents: &[u64]) -> usize {
    let mut stride = 1_usize;
    let mut offset = 0_usize;

    for (&index, &extent) in indices.iter().zip(extents) {
        offset += index as usize * stride;
        stride *= extent as usize;
    }

    offset
}

fn main() -> Result<()> {
    let cuda_context = CudaContext::create()?;
    let context = Context::create(&cuda_context)?;
    let stream = cuda_context.create_stream()?;

    let mode_e = vec!['m'.into(), 'n'.into(), 'b'.into(), 'r'.into(), 'a'.into()];
    let mode_a = vec![
        'm'.into(),
        'k'.into(),
        'a'.into(),
        'j'.into(),
        'b'.into(),
        'i'.into(),
    ];
    let mode_b = vec!['k'.into(), 'n'.into(), 'i'.into()];
    let mode_c = vec!['r'.into(), 'j'.into()];

    let extent_e = vec![2, 2, 2, 2, 2];
    let extent_a = vec![2, 2, 2, 2, 2, 2];
    let extent_b = vec![2, 2, 2];
    let extent_c = vec![2, 2];

    let host_a = (0..extent_a.iter().product::<u64>())
        .map(|index| 0.25_f32 * (index as f32 + 1.0))
        .collect::<Vec<_>>();
    let host_b = (0..extent_b.iter().product::<u64>())
        .map(|index| -0.5_f32 + index as f32 * 0.1)
        .collect::<Vec<_>>();
    let host_c = (0..extent_c.iter().product::<u64>())
        .map(|index| 0.75_f32 + index as f32 * 0.2)
        .collect::<Vec<_>>();
    let host_d = (0..extent_e.iter().product::<u64>())
        .map(|index| -0.125_f32 * (index as f32 + 1.0))
        .collect::<Vec<_>>();

    let device_a = DeviceMemory::from_slice(&host_a)?;
    let device_b = DeviceMemory::from_slice(&host_b)?;
    let device_c = DeviceMemory::from_slice(&host_c)?;
    let device_d_input = DeviceMemory::from_slice(&host_d)?;
    let mut device_e = DeviceMemory::from_slice(&host_d)?;

    const ALIGNMENT: u32 = 128;

    let descriptor_a = TensorDescriptor::create_for::<f32>(&context, &extent_a, ALIGNMENT)?;
    let descriptor_b = TensorDescriptor::create_for::<f32>(&context, &extent_b, ALIGNMENT)?;
    let descriptor_c = TensorDescriptor::create_for::<f32>(&context, &extent_c, ALIGNMENT)?;
    let descriptor_e = TensorDescriptor::create_for::<f32>(&context, &extent_e, ALIGNMENT)?;

    let contraction = OperationDescriptor::contraction_trinary(
        &context,
        TensorOperand::identity(&descriptor_a, &mode_a),
        TensorOperand::identity(&descriptor_b, &mode_b),
        TensorOperand::identity(&descriptor_c, &mode_c),
        TensorOperand::identity(&descriptor_e, &mode_e),
        TensorOperand::identity(&descriptor_e, &mode_e),
        ComputeDescriptor::f32(),
    )?;
    let preference = PlanPreference::create_default(&context)?;
    let workspace_size = Plan::estimate_workspace_size(
        &context,
        &contraction,
        &preference,
        WorkspacePreference::Default,
    )?;
    let plan = Plan::create(&context, &contraction, &preference, workspace_size)?;
    let mut workspace = if plan.required_workspace_size() > 0 {
        Some(DeviceMemory::<u8>::create(
            plan.required_workspace_size_bytes()?,
        )?)
    } else {
        None
    };

    let alpha = 1.25_f32;
    let beta = -0.75_f32;
    plan.contract_trinary(
        &alpha,
        &device_a,
        &device_b,
        &device_c,
        &beta,
        &device_d_input,
        &mut device_e,
        workspace.as_mut(),
        &stream,
    )?;
    stream.synchronize()?;

    let result = device_e.copy_to_host_vec()?;
    let mut expected = vec![0.0_f32; result.len()];

    for m in 0..extent_e[0] {
        for n in 0..extent_e[1] {
            for b in 0..extent_e[2] {
                for r in 0..extent_e[3] {
                    for a in 0..extent_e[4] {
                        let mut sum = 0.0_f32;
                        for k in 0..extent_a[1] {
                            for j in 0..extent_a[3] {
                                for i in 0..extent_a[5] {
                                    let a_offset = packed_offset(&[m, k, a, j, b, i], &extent_a);
                                    let b_offset = packed_offset(&[k, n, i], &extent_b);
                                    let c_offset = packed_offset(&[r, j], &extent_c);
                                    sum += host_a[a_offset] * host_b[b_offset] * host_c[c_offset];
                                }
                            }
                        }

                        let e_offset = packed_offset(&[m, n, b, r, a], &extent_e);
                        expected[e_offset] = alpha * sum + beta * host_d[e_offset];
                    }
                }
            }
        }
    }

    for (actual, reference) in result.iter().zip(&expected) {
        assert!((actual - reference).abs() < 1.0e-4);
    }

    println!(
        "trinary contraction output verified for {} elements",
        result.len()
    );
    Ok(())
}
