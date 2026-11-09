use crate::{
    backend::{
        cpu::{
            shape::select,
            utility::{cpu_device_mismatch, get_f32_data},
        },
        registry::Operation,
        spec,
    },
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    storage::cpu::CpuStorage,
    tensor::Tensor,
    utility::{checked_add, checked_mul, to_usize},
};

pub fn concat(tensors: &[Tensor], dim: i64) -> Result<Tensor> {
    if tensors.is_empty() {
        return Err(Error::invalid_argument("cpu concat tensors"));
    }

    let data_type = tensors[0].data_type();
    let element_bytes = data_type.size_bytes();
    for t in tensors {
        if t.device() != Device::Cpu {
            return Err(cpu_device_mismatch("cpu concat input", t.device()));
        }
        if t.data_type() != data_type {
            return Err(Error::data_type_mismatch(
                "cpu concat input",
                data_type,
                t.data_type(),
            ));
        }
    }

    let shapes = tensors.iter().map(Tensor::shape).collect::<Vec<_>>();
    let problem = spec::concat_problem(&shapes, dim, "cpu concat")?;

    let bytes = checked_mul(
        problem.output_shape.element_count(),
        element_bytes,
        "cpu concat output bytes",
    )?;
    let mut output_data = vec![0_u8; bytes];
    for outer_index in 0..problem.outer_size {
        let mut output_dim_offset = 0usize;
        for (tensor_index, tensor) in tensors.iter().enumerate() {
            let tensor = tensor.contiguous()?;
            let tensor_data = tensor.to_bytes()?;
            let tensor_dim = problem.axis_sizes[tensor_index];
            let chunk = checked_mul(tensor_dim, problem.inner_size, "cpu concat chunk size")?;
            let src_start = checked_mul(outer_index, chunk, "cpu concat source offset")?;
            let outer_offset = checked_mul(
                checked_mul(
                    outer_index,
                    problem.output_axis_size,
                    "cpu concat destination offset",
                )?,
                problem.inner_size,
                "cpu concat destination offset",
            )?;
            let axis_offset = checked_mul(
                output_dim_offset,
                problem.inner_size,
                "cpu concat destination offset",
            )?;
            let dst_start =
                checked_add(outer_offset, axis_offset, "cpu concat destination offset")?;
            let src_start = checked_mul(src_start, element_bytes, "cpu concat source bytes")?;
            let dst_start = checked_mul(dst_start, element_bytes, "cpu concat destination bytes")?;
            let chunk = checked_mul(chunk, element_bytes, "cpu concat chunk bytes")?;
            let dst_end = checked_add(dst_start, chunk, "cpu concat destination offset")?;
            output_data[dst_start..dst_end]
                .copy_from_slice(&tensor_data[src_start..src_start + chunk]);
            output_dim_offset = checked_add(
                output_dim_offset,
                tensor_dim,
                "cpu concat output dimension offset",
            )?;
        }
    }

    let storage = CpuStorage::from_bytes(data_type, output_data)?;
    // SAFETY: the storage was built from values computed for exactly this shape and data type.
    Ok(unsafe { Tensor::from_parts_unchecked(storage, data_type, problem.output_shape) })
}

pub fn index_select(input: &Tensor, dim: i64, indices: &Tensor) -> Result<Tensor> {
    select(Operation::IndexSelect, input, indices, dim)
}

pub fn repeat_kv(input: &Tensor, repeats: i64) -> Result<Tensor> {
    if input.device() != Device::Cpu {
        return Err(cpu_device_mismatch("cpu repeat_kv input", input.device()));
    }

    if input.data_type() != DataType::F32 {
        return Err(Error::unsupported_data_type("cpu repeat_kv"));
    }

    let problem = spec::repeat_kv_problem(input.shape(), repeats, "cpu repeat_kv")?;
    let output_shape = problem.output_shape;
    let input = input.contiguous()?;
    let input_data = get_f32_data(&input)?;
    let mut output_data = vec![0.0_f32; output_shape.element_count()];

    let heads = to_usize(problem.heads, "cpu repeat_kv heads")?;
    let repeats = to_usize(problem.repeats, "cpu repeat_kv repeats")?;
    let sequence = to_usize(problem.sequence, "cpu repeat_kv sequence")?;
    let head_dim = to_usize(problem.head_dim, "cpu repeat_kv head dimension")?;
    let output_heads = checked_mul(heads, repeats, "cpu repeat_kv output heads")?;
    let batch = to_usize(problem.batch, "cpu repeat_kv batch")?;

    for batch_index in 0..batch {
        for output_head in 0..output_heads {
            let input_head = output_head / repeats;
            for sequence_index in 0..sequence {
                let input_row = checked_add(
                    checked_mul(
                        checked_add(
                            checked_mul(batch_index, heads, "cpu repeat_kv input offset")?,
                            input_head,
                            "cpu repeat_kv input offset",
                        )?,
                        sequence,
                        "cpu repeat_kv input offset",
                    )?,
                    sequence_index,
                    "cpu repeat_kv input offset",
                )?;
                let input_start = checked_mul(input_row, head_dim, "cpu repeat_kv input offset")?;
                let output_row = checked_add(
                    checked_mul(
                        checked_add(
                            checked_mul(batch_index, output_heads, "cpu repeat_kv output offset")?,
                            output_head,
                            "cpu repeat_kv output offset",
                        )?,
                        sequence,
                        "cpu repeat_kv output offset",
                    )?,
                    sequence_index,
                    "cpu repeat_kv output offset",
                )?;
                let output_start =
                    checked_mul(output_row, head_dim, "cpu repeat_kv output offset")?;
                output_data[output_start..output_start + head_dim]
                    .copy_from_slice(&input_data[input_start..input_start + head_dim]);
            }
        }
    }

    drop(input_data);

    Tensor::from_vec(Device::Cpu, output_shape, output_data)
}
