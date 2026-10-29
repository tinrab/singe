//! Port of the cuRAND "Host API Example": generate uniform floats on the device
//! with a pseudorandom generator, copy them back, and repeat the same sequence on
//! the host generator.

use std::{error::Error, sync::Arc};

use singe_cuda::{context::Context as CudaContext, device::Device, memory::DeviceMemory};
use singe_curand::{
    generator::{Generator, HostGenerator},
    types::RngType,
};

const COUNT: usize = 100;
const SEED: u64 = 1234;

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::create_for_device(Device::new(0))?;
    let stream = Arc::new(context.create_stream()?);

    let mut device_output = DeviceMemory::<f32>::zeroed(&context, COUNT)?;
    let mut generator = Generator::create(RngType::PseudoDefault, &stream)?;
    generator.set_seed(SEED)?;
    generator.generate_uniform_f32(&mut device_output)?;
    let device_values = device_output.copy_to_host_vec()?;

    let mut host_generator = HostGenerator::create(RngType::PseudoDefault)?;
    host_generator.set_seed(SEED)?;
    let mut host_values = vec![0.0_f32; COUNT];
    host_generator.generate_uniform_f32(&mut host_values)?;

    for (index, value) in device_values.iter().take(8).enumerate() {
        println!("device[{index}] = {value:.6}");
    }
    assert!(
        device_values
            .iter()
            .all(|value| *value > 0.0 && *value <= 1.0)
    );
    assert!(
        host_values
            .iter()
            .all(|value| *value > 0.0 && *value <= 1.0)
    );

    let mean = device_values.iter().sum::<f32>() / COUNT as f32;
    println!("device mean = {mean:.4}");
    Ok(())
}
