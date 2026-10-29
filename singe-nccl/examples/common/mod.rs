#![allow(dead_code)]

use std::{error::Error, io};

use singe_cuda::device::Device;

use singe_nccl::local::LocalCommunicatorGroup;

pub type ExampleResult<T = ()> = Result<T, Box<dyn Error>>;

pub fn create_rank_contexts(min_rank_count: usize) -> ExampleResult<LocalCommunicatorGroup> {
    let count = usize::try_from(Device::count()?)?;
    if count < min_rank_count {
        return Err(io::Error::other(format!(
            "this example needs at least {min_rank_count} CUDA devices, found {count}"
        ))
        .into());
    }
    Ok(LocalCommunicatorGroup::create_all_visible(count)?)
}

pub fn create_power_of_two_rank_contexts(
    min_rank_count: usize,
) -> ExampleResult<LocalCommunicatorGroup> {
    let count = usize::try_from(Device::count()?)?;
    if count < min_rank_count {
        return Err(io::Error::other(format!(
            "this example needs at least {min_rank_count} CUDA devices, found {count}"
        ))
        .into());
    }
    let rank_count = 1usize << (usize::BITS - count.leading_zeros() - 1);
    Ok(LocalCommunicatorGroup::create_all_visible(rank_count)?)
}

pub fn rank_indexed_values(rank: i32, len: usize) -> Vec<f32> {
    let rank = rank as f32;
    (0..len).map(|index| rank * 1000.0 + index as f32).collect()
}

pub fn check_all_equal(actual: &[f32], expected: &[f32], label: &str) -> ExampleResult {
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
