use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory, types::DevicePtr};
use singe_cutensor::{
    context::Context,
    error::Result,
    operation::{BlockSparseTensorOperand, ComputeDescriptor, OperationDescriptor},
    plan::{Plan, PlanPreference},
    tensor::BlockSparseTensorDescriptor,
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

fn block_elements(section_extents: &[Vec<u64>], modes: &[char], coordinates: &[i32]) -> usize {
    modes
        .iter()
        .zip(coordinates)
        .fold(1_usize, |product, (mode, &coordinate)| {
            let mode_index = match mode {
                'k' => 0,
                'i' => 1,
                'l' => 2,
                _ => unreachable!(),
            };
            product * section_extents[mode_index][coordinate as usize] as usize
        })
}

fn make_block_pointers<T>(memory: &DeviceMemory<T>, offsets: &[usize]) -> Vec<DevicePtr> {
    offsets
        .iter()
        .map(|&offset| unsafe { DevicePtr::from_raw(memory.as_mut_ptr().add(offset).cast::<()>()) })
        .collect()
}

fn main() -> Result<()> {
    let cuda_context = CudaContext::create()?;
    let context = Context::create(&cuda_context)?;
    let stream = cuda_context.create_stream()?;

    let mode_a = vec!['k'.into(), 'i'.into(), 'l'.into()];
    let mode_b = vec!['k'.into(), 'l'.into()];
    let mode_c = vec!['i'.into()];

    let section_counts = vec![2_u32, 2, 2];
    let section_extents_flat = vec![2_u64, 1, 2, 1, 3, 1];
    let section_extents = vec![vec![2_u64, 1], vec![2_u64, 1], vec![3_u64, 1]];

    let coordinates_a = vec![0, 0, 0, 1, 1, 0];
    let coordinates_b = vec![0, 0, 1, 0];
    let coordinates_c = vec![0, 1];

    let a_block_sizes = vec![
        block_elements(&section_extents, &['k', 'i', 'l'], &coordinates_a[0..3]),
        block_elements(&section_extents, &['k', 'i', 'l'], &coordinates_a[3..6]),
    ];
    let b_block_sizes = vec![
        block_elements(&section_extents, &['k', 'l'], &coordinates_b[0..2]),
        block_elements(&section_extents, &['k', 'l'], &coordinates_b[2..4]),
    ];
    let c_block_sizes = vec![
        block_elements(&section_extents, &['i'], &coordinates_c[0..1]),
        block_elements(&section_extents, &['i'], &coordinates_c[1..2]),
    ];

    let mut a_offsets = vec![0_usize];
    let mut b_offsets = vec![0_usize];
    let mut c_offsets = vec![0_usize];
    for &size in &a_block_sizes {
        a_offsets.push(a_offsets.last().copied().unwrap() + size);
    }
    for &size in &b_block_sizes {
        b_offsets.push(b_offsets.last().copied().unwrap() + size);
    }
    for &size in &c_block_sizes {
        c_offsets.push(c_offsets.last().copied().unwrap() + size);
    }

    let host_a = (0..a_offsets.last().copied().unwrap())
        .map(|index| index as f32 + 1.0)
        .collect::<Vec<_>>();
    let host_b = (0..b_offsets.last().copied().unwrap())
        .map(|index| 0.5_f32 * (index as f32 + 1.0))
        .collect::<Vec<_>>();
    let host_c = vec![0.0_f32; c_offsets.last().copied().unwrap()];

    let device_a = DeviceMemory::from_slice(&host_a)?;
    let device_b = DeviceMemory::from_slice(&host_b)?;
    let device_c_input = DeviceMemory::from_slice(&host_c)?;
    let device_c_output = DeviceMemory::from_slice(&host_c)?;

    let descriptor_a = BlockSparseTensorDescriptor::create(
        &context,
        &section_counts,
        &section_extents_flat,
        2,
        &coordinates_a,
        None,
        singe_cuda::data_type::DataType::F32,
    )?;
    let descriptor_b = BlockSparseTensorDescriptor::create(
        &context,
        &[2, 2],
        &[2, 1, 3, 1],
        2,
        &coordinates_b,
        None,
        singe_cuda::data_type::DataType::F32,
    )?;
    let descriptor_c = BlockSparseTensorDescriptor::create(
        &context,
        &[2],
        &[2, 1],
        2,
        &coordinates_c,
        None,
        singe_cuda::data_type::DataType::F32,
    )?;

    let operation = OperationDescriptor::block_sparse_contraction(
        &context,
        BlockSparseTensorOperand::identity(&descriptor_a, &mode_a),
        BlockSparseTensorOperand::identity(&descriptor_b, &mode_b),
        BlockSparseTensorOperand::identity(&descriptor_c, &mode_c),
        BlockSparseTensorOperand::identity(&descriptor_c, &mode_c),
        ComputeDescriptor::f32(),
    )?;
    let preference = PlanPreference::create_default(&context)?;
    let workspace_size = Plan::estimate_workspace_size(
        &context,
        &operation,
        &preference,
        WorkspacePreference::Default,
    )?;
    let plan = Plan::create(&context, &operation, &preference, workspace_size)?;
    let mut workspace = if plan.required_workspace_size() > 0 {
        Some(DeviceMemory::<u8>::create(
            plan.required_workspace_size_bytes()?,
        )?)
    } else {
        None
    };

    let pointers_a = make_block_pointers(&device_a, &a_offsets[..a_offsets.len() - 1]);
    let pointers_b = make_block_pointers(&device_b, &b_offsets[..b_offsets.len() - 1]);
    let pointers_c_input = make_block_pointers(&device_c_input, &c_offsets[..c_offsets.len() - 1]);
    let mut pointers_c_output =
        make_block_pointers(&device_c_output, &c_offsets[..c_offsets.len() - 1]);

    let alpha = 1.0_f32;
    let beta = 0.0_f32;
    plan.block_sparse_contract(
        &alpha,
        &pointers_a,
        &pointers_b,
        &beta,
        &pointers_c_input,
        &mut pointers_c_output,
        workspace.as_mut(),
        &stream,
    )?;
    stream.synchronize()?;

    let result = device_c_output.copy_to_host_vec()?;
    let mut expected = vec![0.0_f32; result.len()];

    for i_block in 0..2 {
        let i_extent = section_extents[1][i_block];
        for i in 0..i_extent {
            let c_index = c_offsets[i_block] + i as usize;
            if i_block == 0 {
                for k in 0..section_extents[0][0] {
                    for l in 0..section_extents[2][0] {
                        let a_index = a_offsets[0] + packed_offset(&[k, i, l], &[2, 2, 3]);
                        let b_index = b_offsets[0] + packed_offset(&[k, l], &[2, 3]);
                        expected[c_index] += host_a[a_index] * host_b[b_index];
                    }
                }
            } else {
                for k in 0..section_extents[0][1] {
                    for l in 0..section_extents[2][0] {
                        let a_index = a_offsets[1] + packed_offset(&[k, i, l], &[1, 1, 3]);
                        let b_index = b_offsets[1] + packed_offset(&[k, l], &[1, 3]);
                        expected[c_index] += host_a[a_index] * host_b[b_index];
                    }
                }
            }
        }
    }

    for (actual, reference) in result.iter().zip(&expected) {
        assert!((actual - reference).abs() < 1.0e-5);
    }

    println!(
        "block-sparse contraction verified for {} stored values",
        result.len()
    );
    Ok(())
}
