use std::{collections::HashMap, env, time::Instant};

use singe_cuda::{context::Context as CudaContext, data_type::DataType, device::Device};
use singe_cutensor::{
    error::Result,
    mg::{
        context::Context,
        contraction::{ContractionDescriptor, ContractionFind, ContractionPlan},
        memory::DistributedDeviceMemory,
        tensor::{
            DevicePartition, TensorDescriptor, TensorDescriptorConfig, TensorDevice, TensorMode,
        },
        types::Algorithm,
    },
    types::{ComputeType, Mode, WorkspacePreference},
};

fn product<T>(values: &[T]) -> T
where
    T: Copy + From<u8> + std::ops::Mul<Output = T>,
{
    values
        .iter()
        .copied()
        .fold(T::from(1), |acc, value| acc * value)
}

fn multiply_i64_i32(lhs: &[i64], rhs: &[i32]) -> Vec<i64> {
    lhs.iter()
        .zip(rhs)
        .map(|(&lhs, &rhs)| lhs * i64::from(rhs))
        .collect()
}

fn discretize(values: &[i64], block: &[i64]) -> Vec<i64> {
    values
        .iter()
        .zip(block)
        .map(|(&value, &block)| block * ((value + block - 1) / block))
        .collect()
}

fn collect_i64(map: &HashMap<i32, i64>, modes: &[i32]) -> Vec<i64> {
    modes.iter().map(|mode| map[mode]).collect()
}

fn collect_i32(map: &HashMap<i32, i32>, modes: &[i32]) -> Vec<i32> {
    modes.iter().map(|mode| map[mode]).collect()
}

fn fill_up(devices: &[i32], count: i32) -> Vec<TensorDevice> {
    (0..count as usize)
        .map(|index| TensorDevice::Device(devices[index % devices.len()]))
        .collect()
}

fn mode_vec(modes: &[i32]) -> Vec<Mode> {
    modes.iter().copied().map(Mode::new).collect()
}

fn create_streams_for_devices(
    devices: &[i32],
) -> singe_cuda::error::Result<(
    Vec<singe_cuda::stream::Stream>,
    Vec<singe_cuda::stream::StreamBinding>,
)> {
    let mut streams = Vec::with_capacity(devices.len());
    for &device in devices {
        let cuda_context = CudaContext::retain_primary_for_device(Device::new(device))?;
        streams.push(cuda_context.create_stream()?);
    }

    let bindings = streams
        .iter()
        .map(|stream| singe_cuda::stream::StreamBinding::Borrowed(stream.to_borrowed()))
        .collect();
    Ok((streams, bindings))
}

fn main() -> Result<()> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let devices = if args.is_empty() {
        (0..Device::count()?).collect::<Vec<_>>()
    } else {
        args.iter()
            .map(|arg| arg.parse::<i32>().expect("device id must be an integer"))
            .collect::<Vec<_>>()
    };

    assert!(!devices.is_empty(), "at least one CUDA device is required");
    println!("using devices: {devices:?}");

    let context = Context::create(&devices)?;

    let mut extent = HashMap::new();
    extent.insert('i' as i32, 4096);
    extent.insert('j' as i32, 4096);
    extent.insert('k' as i32, 4096);

    let mut block_size = HashMap::new();
    block_size.insert('i' as i32, 2048);
    block_size.insert('j' as i32, 2048);
    block_size.insert('k' as i32, 2048);

    let mut device_count = HashMap::new();
    device_count.insert('i' as i32, 2);
    device_count.insert('j' as i32, 2);
    device_count.insert('k' as i32, 2);

    let modes_a = vec!['i' as i32, 'k' as i32];
    let modes_b = vec!['k' as i32, 'j' as i32];
    let modes_c = vec!['i' as i32, 'j' as i32];

    let extent_a = collect_i64(&extent, &modes_a);
    let block_size_a = collect_i64(&block_size, &modes_a);
    let device_count_a = collect_i32(&device_count, &modes_a);
    let devices_a = fill_up(&devices, product(&device_count_a));

    let extent_b = collect_i64(&extent, &modes_b);
    let block_size_b = collect_i64(&block_size, &modes_b);
    let device_count_b = collect_i32(&device_count, &modes_b);
    let devices_b = fill_up(&devices, product(&device_count_b));

    let extent_c = collect_i64(&extent, &modes_c);
    let block_size_c = collect_i64(&block_size, &modes_c);
    let device_count_c = collect_i32(&device_count, &modes_c);
    let devices_c = fill_up(&devices, product(&device_count_c));
    let tensor_modes_a = extent_a
        .iter()
        .zip(&block_size_a)
        .map(|(&shape, &block_size)| TensorMode {
            shape: shape as u64,
            element_stride: None,
            block_size: Some(block_size as u64),
            block_stride: None,
        })
        .collect::<Vec<_>>();
    let tensor_modes_b = extent_b
        .iter()
        .zip(&block_size_b)
        .map(|(&shape, &block_size)| TensorMode {
            shape: shape as u64,
            element_stride: None,
            block_size: Some(block_size as u64),
            block_stride: None,
        })
        .collect::<Vec<_>>();
    let tensor_modes_c = extent_c
        .iter()
        .zip(&block_size_c)
        .map(|(&shape, &block_size)| TensorMode {
            shape: shape as u64,
            element_stride: None,
            block_size: Some(block_size as u64),
            block_stride: None,
        })
        .collect::<Vec<_>>();

    let descriptor_a = TensorDescriptor::create(
        &context,
        TensorDescriptorConfig {
            modes: &tensor_modes_a,
            partition: DevicePartition {
                devices: &devices_a,
                count_per_mode: Some(&device_count_a),
            },
            data_type: DataType::F32,
        },
    )?;
    let descriptor_b = TensorDescriptor::create(
        &context,
        TensorDescriptorConfig {
            modes: &tensor_modes_b,
            partition: DevicePartition {
                devices: &devices_b,
                count_per_mode: Some(&device_count_b),
            },
            data_type: DataType::F32,
        },
    )?;
    let descriptor_c = TensorDescriptor::create(
        &context,
        TensorDescriptorConfig {
            modes: &tensor_modes_c,
            partition: DevicePartition {
                devices: &devices_c,
                count_per_mode: Some(&device_count_c),
            },
            data_type: DataType::F32,
        },
    )?;

    let contraction = ContractionDescriptor::create(
        &context,
        &descriptor_a,
        &mode_vec(&modes_a),
        &descriptor_b,
        &mode_vec(&modes_b),
        &descriptor_c,
        &mode_vec(&modes_c),
        &descriptor_c,
        &mode_vec(&modes_c),
        ComputeType::F32,
    )?;
    let find = ContractionFind::create(&context, Algorithm::Default)?;
    let workspace = contraction.workspace(&find, WorkspacePreference::Default)?;
    let plan = ContractionPlan::create(&context, &contraction, &find, workspace)?;

    let elements_a = product(&discretize(
        &extent_a,
        &multiply_i64_i32(&block_size_a, &device_count_a),
    )) / i64::from(product(&device_count_a));
    let elements_b = product(&discretize(
        &extent_b,
        &multiply_i64_i32(&block_size_b, &device_count_b),
    )) / i64::from(product(&device_count_b));
    let elements_c = product(&discretize(
        &extent_c,
        &multiply_i64_i32(&block_size_c, &device_count_c),
    )) / i64::from(product(&device_count_c));

    let memory_a = DistributedDeviceMemory::<f32>::create(&devices_a, elements_a as usize)?;
    let memory_b = DistributedDeviceMemory::<f32>::create(&devices_b, elements_b as usize)?;
    let mut memory_c = DistributedDeviceMemory::<f32>::create(&devices_c, elements_c as usize)?;
    let mut workspace = plan.create_workspace_memory()?;
    let (streams, stream_bindings) = create_streams_for_devices(&devices)?;

    let alpha = 1.0_f32;
    let beta = 0.0_f32;

    let mut best_ms = f64::INFINITY;
    for _ in 0..3 {
        let start = Instant::now();
        plan.contract_in_place(
            &alpha,
            &memory_a,
            &memory_b,
            &beta,
            &mut memory_c,
            &mut workspace,
            &stream_bindings,
        )?;

        for stream in &streams {
            stream.synchronize()?;
        }

        best_ms = best_ms.min(start.elapsed().as_secs_f64() * 1.0e3);
    }

    println!("execution took: {best_ms:.2e} ms");
    println!("done: everything completed successfully");
    Ok(())
}
