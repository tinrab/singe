#![allow(dead_code)]

use std::{env, io};

use anyhow::Result;
use singe_cuda::device::Device;
use singe_ptx::emulator::{EmulatorOptions, Topology};

use singe_nccl::local::LocalCommunicatorGroup;

fn install_emulator() -> Result<()> {
    let Ok(devices) = env::var("SINGE_EMULATOR") else {
        return Ok(());
    };
    singe_cuda_emulator::install(EmulatorOptions {
        topology: Topology::node(devices.parse()?),
        ..Default::default()
    })?;
    Ok(())
}

pub(crate) fn create_rank_contexts(min_rank_count: usize) -> Result<LocalCommunicatorGroup> {
    install_emulator()?;
    let count = usize::try_from(Device::count()?)?;
    if count < min_rank_count {
        return Err(io::Error::other(format!(
            "this example needs at least {min_rank_count} CUDA devices, found {count}"
        ))
        .into());
    }
    Ok(LocalCommunicatorGroup::create(&first_devices(count)?)?)
}

pub(crate) fn create_power_of_two_rank_contexts(
    min_rank_count: usize,
) -> Result<LocalCommunicatorGroup> {
    install_emulator()?;
    let count = usize::try_from(Device::count()?)?;
    if count < min_rank_count {
        return Err(io::Error::other(format!(
            "this example needs at least {min_rank_count} CUDA devices, found {count}"
        ))
        .into());
    }

    let rank_count = 1usize << (count.bit_width() - 1);
    Ok(LocalCommunicatorGroup::create(&first_devices(rank_count)?)?)
}

pub(crate) fn first_devices(count: usize) -> Result<Vec<Device>> {
    install_emulator()?;
    (0..count)
        .map(|id| -> Result<Device> { Ok(Device::new(i32::try_from(id)?)) })
        .collect()
}

pub(crate) fn rank_indexed_values(rank: i32, len: usize) -> Vec<f32> {
    let rank = rank as f32;
    (0..len)
        .map(|index| rank.mul_add(1000.0, index as f32))
        .collect()
}

pub(crate) fn check_all_equal(actual: &[f32], expected: &[f32], label: &str) -> Result<()> {
    if actual.len() != expected.len() {
        return Err(io::Error::other(format!(
            "{label} length mismatch: expected {}, got {}",
            expected.len(),
            actual.len()
        ))
        .into());
    }

    if let Some(index) = actual
        .iter()
        .zip(expected)
        .position(|(&actual, &expected)| (actual - expected).abs() > f32::EPSILON)
    {
        return Err(io::Error::other(format!(
            "{label} verification failed at index {index}: expected {:.2}, got {:.2}",
            expected[index], actual[index]
        ))
        .into());
    }

    Ok(())
}
