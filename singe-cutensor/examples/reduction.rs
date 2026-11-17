use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory, stream::StreamOptions};

use singe_cutensor::{
    context::Context,
    operation::{ComputeDescriptor, OperationDescriptor, TensorOperand},
    plan::{Plan, PlanPreference},
    tensor::TensorDescriptor,
    types::{Operator, WorkspacePreference},
};

fn packed_offset(indices: &[u64], extents: &[u64]) -> usize {
    let mut stride = 1_usize;
    let mut offset = 0_usize;

    for (&index, &extent) in indices.iter().zip(extents) {
        offset += usize::try_from(index).unwrap() * stride;
        stride *= usize::try_from(extent).unwrap();
    }

    offset
}

fn main() -> anyhow::Result<()> {
    const ALIGNMENT: u32 = 128;

    let cuda_context = CudaContext::create()?;
    let context = Context::create(&cuda_context)?;
    let stream = cuda_context.create_stream(StreamOptions::default())?;

    let extent_a = vec![2, 3, 2];
    let extent_c = vec![2, 2];
    let mode_a = vec!['m'.into(), 'h'.into(), 'v'.into()];
    let mode_c = vec!['m'.into(), 'v'.into()];

    let host_a = (0..extent_a.iter().product::<u64>())
        .map(|index| index as f32 + 1.0)
        .collect::<Vec<_>>();
    let host_c = vec![0.5_f32, -1.0, 1.5, -2.0];

    let device_a = DeviceMemory::from_slice(&cuda_context, &host_a)?;
    let device_c_input = DeviceMemory::from_slice(&cuda_context, &host_c)?;
    let mut device_c = DeviceMemory::from_slice(&cuda_context, &host_c)?;

    let descriptor_a = TensorDescriptor::create_for::<f32>(&context, &extent_a, ALIGNMENT)?;
    let descriptor_c = TensorDescriptor::create_for::<f32>(&context, &extent_c, ALIGNMENT)?;

    let reduction = OperationDescriptor::reduction(
        &context,
        TensorOperand::identity(&descriptor_a, &mode_a),
        TensorOperand::identity(&descriptor_c, &mode_c),
        TensorOperand::identity(&descriptor_c, &mode_c),
        Operator::Add,
        ComputeDescriptor::f32(),
    )?;
    let preference = PlanPreference::create_default(&context)?;
    let workspace_size = Plan::estimate_workspace_size(
        &context,
        &reduction,
        &preference,
        WorkspacePreference::Default,
    )?;
    let plan = Plan::create(&context, &reduction, &preference, workspace_size)?;
    let mut workspace = if plan.workspace_size() > 0 {
        Some(DeviceMemory::<u8>::create_uninit(
            &cuda_context,
            plan.workspace_size(),
        )?)
    } else {
        None
    };

    let alpha = 1.25_f32;
    let beta = -0.5_f32;
    plan.reduce(
        &alpha,
        &device_a,
        &beta,
        &device_c_input,
        &mut device_c,
        workspace.as_mut().map(|memory| memory.output()),
        &stream,
    )?;

    let result = device_c.copy_to_host_vec()?;
    let mut expected = vec![0.0_f32; usize::try_from(extent_c.iter().product::<u64>())?];

    for m in 0..usize::try_from(extent_c[0])? {
        for v in 0..usize::try_from(extent_c[1])? {
            let mut sum = 0.0_f32;
            for h in 0..extent_a[1] {
                let a_offset = packed_offset(&[m as u64, h, v as u64], &extent_a);
                sum += host_a[a_offset];
            }

            let c_offset = packed_offset(&[m as u64, v as u64], &extent_c);
            expected[c_offset] = beta.mul_add(host_c[c_offset], alpha * sum);
        }
    }

    assert_eq!(result.len(), expected.len());
    for (actual, reference) in result.iter().zip(&expected) {
        assert!((actual - reference).abs() < 1.0e-5);
    }

    println!("reduction output verified for {} elements", result.len());
    Ok(())
}
