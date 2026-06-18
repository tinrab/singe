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

fn ceil_div(lhs: i64, rhs: i64) -> i64 {
    (lhs + rhs - 1) / rhs
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

fn create_tensor(
    context: &Context,
    modes: &[i32],
    extent_map: &HashMap<i32, i64>,
    block_size_map: &HashMap<i32, i64>,
    num_devices: i32,
) -> Result<(TensorDescriptor, DistributedDeviceMemory<f32>)> {
    let mut extent = Vec::with_capacity(modes.len());
    let mut block_size = Vec::with_capacity(modes.len());
    let mut device_count = vec![1_i32; modes.len()];

    for mode in modes {
        extent.push(extent_map[mode]);
        block_size.push(block_size_map[mode]);
    }

    let mut remaining_devices = num_devices;
    let mut changed = true;
    while changed {
        changed = false;
        for index in (0..modes.len()).rev() {
            if remaining_devices <= 1 {
                break;
            }

            let max_device_count = extent_map[&modes[index]] / block_size_map[&modes[index]];
            if i64::from(device_count[index]) < max_device_count {
                device_count[index] *= 2;
                remaining_devices /= 2;
                changed = true;
            }
        }
    }

    let mut elements = 1_i64;
    for index in 0..modes.len() {
        let num_blocks = ceil_div(extent[index], block_size[index]);
        let num_blocks_per_device = ceil_div(num_blocks, i64::from(device_count[index]));
        elements *= num_blocks_per_device * block_size[index];
    }

    println!("elements per tensor pointer: {elements}");

    let descriptor_devices = (0..num_devices / remaining_devices)
        .map(TensorDevice::Device)
        .collect::<Vec<_>>();
    let tensor_modes = extent
        .iter()
        .zip(&block_size)
        .map(|(&shape, &block_size)| TensorMode {
            shape: shape as u64,
            element_stride: None,
            block_size: Some(block_size as u64),
            block_stride: None,
        })
        .collect::<Vec<_>>();
    let descriptor = TensorDescriptor::create(
        context,
        TensorDescriptorConfig {
            modes: &tensor_modes,
            partition: DevicePartition {
                devices: &descriptor_devices,
                count_per_mode: Some(&device_count),
            },
            data_type: DataType::F32,
        },
    )?;

    let memory = DistributedDeviceMemory::<f32>::create(&descriptor_devices, elements as usize)?;

    Ok((descriptor, memory))
}

fn largest_power_of_two_at_most(value: i32) -> i32 {
    let mut power: i32 = 1;
    while power.saturating_mul(2) <= value {
        power *= 2;
    }
    power
}

fn main() -> Result<()> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let available_devices = Device::count()?;
    assert!(
        available_devices > 0,
        "at least one CUDA device is required"
    );

    let num_devices = args
        .first()
        .map(|arg| arg.parse::<i32>().expect("numDevices must be an integer"))
        .unwrap_or_else(|| largest_power_of_two_at_most(available_devices));
    assert!(num_devices >= 1, "numDevices must be positive");
    assert!(
        num_devices <= available_devices,
        "requested {num_devices} devices but only {available_devices} are available"
    );
    assert!(
        num_devices > 0 && (num_devices & (num_devices - 1)) == 0,
        "numDevices must be a power of two"
    );

    let scaling = args
        .get(1)
        .map(|arg| arg.parse::<i64>().expect("scaling must be an integer"))
        .unwrap_or(2);
    assert!(scaling >= 1, "scaling must be positive");

    let devices = (0..num_devices).collect::<Vec<_>>();
    println!("using devices: {devices:?}, scaling: {scaling}");

    let context = Context::create(&devices)?;

    let m0 = 0;
    let m1 = 1;
    let m2 = 2;
    let n0 = 3;
    let n1 = 4;
    let n2 = 5;
    let k0 = 6;
    let k1 = 7;
    let k2 = 8;

    let mut extent = HashMap::new();
    extent.insert(m0, 16);
    extent.insert(m1, 8 * scaling);
    extent.insert(m2, 8);
    extent.insert(n0, 16);
    extent.insert(n1, 8 * scaling);
    extent.insert(n2, 8);
    extent.insert(k0, 16);
    extent.insert(k1, 32);
    extent.insert(k2, 8);

    let num_devices_m = if num_devices >= 4 {
        num_devices / 2
    } else {
        num_devices
    };
    let num_devices_n = num_devices / num_devices_m;
    let m = extent[&m0] * extent[&m1] * extent[&m2];
    let n = extent[&n0] * extent[&n1] * extent[&n2];

    let mut block_size = HashMap::new();
    block_size.insert(m0, 16);
    block_size.insert(
        m1,
        ((m as f64 / (m as f64 / 4096.0 / f64::from(num_devices_m)).ceil()).ceil()
            / f64::from(num_devices_m)
            / extent[&m0] as f64
            / extent[&m2] as f64)
            .ceil() as i64,
    );
    block_size.insert(m2, 8);
    block_size.insert(n0, 16);
    block_size.insert(
        n1,
        ((n as f64 / (n as f64 / 4096.0 / f64::from(num_devices_n)).ceil()).ceil()
            / f64::from(num_devices_n)
            / extent[&n0] as f64
            / extent[&n1] as f64)
            .ceil() as i64,
    );
    block_size.insert(n2, 8);
    block_size.insert(k0, 16);
    block_size.insert(k1, 16);
    block_size.insert(k2, 8);

    let modes_a = vec![k0, m0, m1, k1, m2, k2];
    let modes_b = vec![k0, n0, k1, n1, k2, n2];
    let modes_c = vec![m0, n0, m1, n1, m2, n2];

    let (descriptor_a, memory_a) =
        create_tensor(&context, &modes_a, &extent, &block_size, num_devices)?;
    let (descriptor_b, memory_b) =
        create_tensor(&context, &modes_b, &extent, &block_size, num_devices)?;
    let (descriptor_c, mut memory_c) =
        create_tensor(&context, &modes_c, &extent, &block_size, num_devices)?;

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

    let mut workspace = plan.create_workspace_memory()?;
    let (streams, stream_bindings) = create_streams_for_devices(&devices)?;

    let alpha = 1.0_f32;
    let beta = 0.0_f32;

    let mut best_ms = f64::INFINITY;
    for rep in 0..3 {
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

        let elapsed_ms = start.elapsed().as_secs_f64() * 1.0e3;
        println!("rep {rep}: {elapsed_ms:.3} ms");
        best_ms = best_ms.min(elapsed_ms);
    }

    let flops = extent
        .values()
        .fold(2.0, |acc, &extent| acc * extent as f64)
        / (best_ms * 1.0e-3)
        * 1.0e-9;

    println!("execution time: {best_ms:.2e} ms");
    println!("execution perf: {flops:.2e} GFLOP/s");
    println!("done: everything completed successfully");
    Ok(())
}
