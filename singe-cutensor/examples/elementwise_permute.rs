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

    let extent_a = vec![2, 3, 2, 2];
    let extent_c = vec![2, 2, 3, 2];
    let mode_a = vec!['w'.into(), 'h'.into(), 'c'.into(), 'n'.into()];
    let mode_c = vec!['c'.into(), 'w'.into(), 'h'.into(), 'n'.into()];

    let host_a = (0..extent_a.iter().product::<u64>())
        .map(|index| index as f32 + 1.0)
        .collect::<Vec<_>>();

    let device_a = DeviceMemory::from_slice(&host_a)?;
    let mut device_c = DeviceMemory::<f32>::create(host_a.len())?;

    const ALIGNMENT: u32 = 128;

    let descriptor_a = TensorDescriptor::create_for::<f32>(&context, &extent_a, ALIGNMENT)?;
    let descriptor_c = TensorDescriptor::create_for::<f32>(&context, &extent_c, ALIGNMENT)?;

    let operation = OperationDescriptor::permutation(
        &context,
        TensorOperand::identity(&descriptor_a, &mode_a),
        TensorOperand::identity(&descriptor_c, &mode_c),
        ComputeDescriptor::f32(),
    )?;
    let preference = PlanPreference::create_default(&context)?;
    let workspace_size =
        Plan::estimate_workspace_size(&context, &operation, &preference, WorkspacePreference::Min)?;
    let plan = Plan::create(&context, &operation, &preference, workspace_size)?;

    let alpha = 1.0_f32;
    plan.permute(&alpha, &device_a, &mut device_c, &stream)?;
    stream.synchronize()?;

    let result = device_c.copy_to_host_vec()?;
    let mut expected = vec![0.0_f32; result.len()];

    for w in 0..extent_a[0] {
        for h in 0..extent_a[1] {
            for c in 0..extent_a[2] {
                for n in 0..extent_a[3] {
                    let a_offset = packed_offset(&[w, h, c, n], &extent_a);
                    let c_offset = packed_offset(&[c, w, h, n], &extent_c);
                    expected[c_offset] = alpha * host_a[a_offset];
                }
            }
        }
    }

    for (actual, reference) in result.iter().zip(&expected) {
        assert!((actual - reference).abs() < 1.0e-5);
    }

    println!("permutation output verified for {} elements", result.len());
    Ok(())
}
