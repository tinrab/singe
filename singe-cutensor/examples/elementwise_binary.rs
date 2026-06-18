use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory};
use singe_cutensor::{
    context::Context,
    error::Result,
    operation::{ComputeDescriptor, OperationDescriptor, TensorOperand},
    plan::{Plan, PlanPreference},
    tensor::TensorDescriptor,
    types::{Operator, WorkspacePreference},
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

    let extent_a = vec![2, 3, 2];
    let extent_c = vec![2, 3, 2];
    let mode_a = vec!['c'.into(), 'b'.into(), 'a'.into()];
    let mode_c = vec!['a'.into(), 'b'.into(), 'c'.into()];

    let host_a = (0..extent_a.iter().product::<u64>())
        .map(|index| index as f32 + 1.0)
        .collect::<Vec<_>>();
    let host_c = (0..extent_c.iter().product::<u64>())
        .map(|index| -0.25_f32 * (index as f32 + 1.0))
        .collect::<Vec<_>>();

    let device_a = DeviceMemory::from_slice(&host_a)?;
    let device_c = DeviceMemory::from_slice(&host_c)?;
    let mut device_d = DeviceMemory::from_slice(&host_c)?;

    const ALIGNMENT: u32 = 128;

    let descriptor_a = TensorDescriptor::create_for::<f32>(&context, &extent_a, ALIGNMENT)?;
    let descriptor_c = TensorDescriptor::create_for::<f32>(&context, &extent_c, ALIGNMENT)?;

    let operation = OperationDescriptor::elementwise_binary(
        &context,
        TensorOperand::identity(&descriptor_a, &mode_a),
        TensorOperand::identity(&descriptor_c, &mode_c),
        TensorOperand::identity(&descriptor_c, &mode_c),
        Operator::Add,
        ComputeDescriptor::f32(),
    )?;
    let preference = PlanPreference::create_default(&context)?;
    let workspace_size =
        Plan::estimate_workspace_size(&context, &operation, &preference, WorkspacePreference::Min)?;
    let plan = Plan::create(&context, &operation, &preference, workspace_size)?;

    let alpha = 1.1_f32;
    let gamma = 1.2_f32;
    plan.elementwise_binary(&alpha, &device_a, &gamma, &device_c, &mut device_d, &stream)?;
    stream.synchronize()?;

    let result = device_d.copy_to_host_vec()?;
    let mut expected = vec![0.0_f32; result.len()];

    for a in 0..extent_c[0] {
        for b in 0..extent_c[1] {
            for c in 0..extent_c[2] {
                let c_offset = packed_offset(&[a, b, c], &extent_c);
                let a_offset = packed_offset(&[c, b, a], &extent_a);
                expected[c_offset] = alpha * host_a[a_offset] + gamma * host_c[c_offset];
            }
        }
    }

    for (actual, reference) in result.iter().zip(&expected) {
        assert!((actual - reference).abs() < 1.0e-5);
    }

    println!(
        "elementwise binary output verified for {} elements",
        result.len()
    );
    Ok(())
}
