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

    let mode_c = vec!['m'.into(), 'u'.into(), 'n'.into(), 'v'.into()];
    let mode_a = vec!['m'.into(), 'h'.into(), 'k'.into(), 'n'.into()];
    let mode_b = vec!['u'.into(), 'k'.into(), 'v'.into(), 'h'.into()];

    let extent_c = vec![2, 2, 3, 2];
    let extent_a = vec![2, 2, 2, 3];
    let extent_b = vec![2, 2, 2, 2];

    let elements_a = extent_a.iter().product::<u64>();
    let elements_b = extent_b.iter().product::<u64>();
    let elements_c = extent_c.iter().product::<u64>();

    let host_a = (0..elements_a)
        .map(|index| (index as f32 + 1.0) * 0.25)
        .collect::<Vec<_>>();
    let host_b = (0..elements_b)
        .map(|index| ((index % 7) as f32 - 3.0) * 0.5)
        .collect::<Vec<_>>();
    let host_c = (0..elements_c)
        .map(|index| (index as f32 - 4.0) * 0.125)
        .collect::<Vec<_>>();

    let device_a = DeviceMemory::from_slice(&host_a)?;
    let device_b = DeviceMemory::from_slice(&host_b)?;
    let device_c_input = DeviceMemory::from_slice(&host_c)?;
    let mut device_c = DeviceMemory::from_slice(&host_c)?;

    const ALIGNMENT: u32 = 128;

    let descriptor_a = TensorDescriptor::create_for::<f32>(&context, &extent_a, ALIGNMENT)?;
    let descriptor_b = TensorDescriptor::create_for::<f32>(&context, &extent_b, ALIGNMENT)?;
    let descriptor_c = TensorDescriptor::create_for::<f32>(&context, &extent_c, ALIGNMENT)?;

    let contraction = OperationDescriptor::contraction(
        &context,
        TensorOperand::identity(&descriptor_a, &mode_a),
        TensorOperand::identity(&descriptor_b, &mode_b),
        TensorOperand::identity(&descriptor_c, &mode_c),
        TensorOperand::identity(&descriptor_c, &mode_c),
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

    let alpha = 1.1_f32;
    let beta = -0.25_f32;
    plan.contract(
        &alpha,
        &device_a,
        &device_b,
        &beta,
        &device_c_input,
        &mut device_c,
        workspace.as_mut(),
        &stream,
    )?;
    stream.synchronize()?;

    let result = device_c.copy_to_host_vec()?;
    let mut expected = vec![0.0_f32; elements_c as usize];

    for m in 0..extent_c[0] {
        for u in 0..extent_c[1] {
            for n in 0..extent_c[2] {
                for v in 0..extent_c[3] {
                    let mut sum = 0.0_f32;
                    for h in 0..extent_a[1] {
                        for k in 0..extent_a[2] {
                            let a_offset = packed_offset(&[m, h, k, n], &extent_a);
                            let b_offset = packed_offset(&[u, k, v, h], &extent_b);
                            sum += host_a[a_offset] * host_b[b_offset];
                        }
                    }

                    let c_offset = packed_offset(&[m, u, n, v], &extent_c);
                    expected[c_offset] = alpha * sum + beta * host_c[c_offset];
                }
            }
        }
    }

    assert_eq!(result.len(), expected.len());
    for (actual, reference) in result.iter().zip(&expected) {
        assert!((actual - reference).abs() < 1.0e-5);
    }

    println!("contraction output verified for {} elements", result.len());
    Ok(())
}
