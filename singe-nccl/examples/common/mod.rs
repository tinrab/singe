#![allow(dead_code)]

use std::{error::Error, io};

use singe_cuda::{device::Device, stream::Stream};
use singe_nccl::communicator::Communicator;

pub type ExampleResult<T = ()> = Result<T, Box<dyn Error>>;

#[derive(Debug)]
pub struct RankContext {
    communicator: Communicator,
    stream: Stream,
}

impl RankContext {
    pub fn communicator(&self) -> &Communicator {
        &self.communicator
    }

    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    pub fn bind(&self) -> ExampleResult {
        self.communicator.bind()?;
        Ok(())
    }

    pub fn rank(&self) -> ExampleResult<i32> {
        Ok(self.communicator.rank()?)
    }

    pub fn rank_count(&self) -> ExampleResult<i32> {
        Ok(self.communicator.rank_count()?)
    }
}

pub fn create_rank_contexts(min_rank_count: usize) -> ExampleResult<Vec<RankContext>> {
    let available_rank_count = usize::try_from(Device::count()?)?;
    if available_rank_count < min_rank_count {
        return Err(io::Error::other(format!(
            "this example needs at least {min_rank_count} CUDA devices, found {available_rank_count}"
        ))
        .into());
    }

    let device_ids: Vec<_> = (0..available_rank_count)
        .map(i32::try_from)
        .collect::<Result<_, _>>()?;
    let communicators = Communicator::create_all(&device_ids)?;
    let mut ranks = Vec::with_capacity(communicators.len());

    for communicator in communicators {
        let stream = communicator.cuda_context().create_stream()?;
        ranks.push(RankContext {
            communicator,
            stream,
        });
    }

    Ok(ranks)
}

pub fn create_power_of_two_rank_contexts(min_rank_count: usize) -> ExampleResult<Vec<RankContext>> {
    let available_rank_count = usize::try_from(Device::count()?)?;
    if available_rank_count < min_rank_count {
        return Err(io::Error::other(format!(
            "this example needs at least {min_rank_count} CUDA devices, found {available_rank_count}"
        ))
        .into());
    }

    let rank_count = (usize::BITS - available_rank_count.leading_zeros() - 1) as usize;
    let rank_count = 1usize << rank_count;
    let rank_count = rank_count.max(min_rank_count.next_power_of_two());
    if rank_count > available_rank_count {
        return Err(io::Error::other(format!(
            "this example needs at least {} CUDA devices, found {available_rank_count}",
            min_rank_count.next_power_of_two()
        ))
        .into());
    }

    let device_ids: Vec<_> = (0..rank_count)
        .map(i32::try_from)
        .collect::<Result<_, _>>()?;
    let communicators = Communicator::create_all(&device_ids)?;
    let mut ranks = Vec::with_capacity(communicators.len());

    for communicator in communicators {
        let stream = communicator.cuda_context().create_stream()?;
        ranks.push(RankContext {
            communicator,
            stream,
        });
    }

    Ok(ranks)
}

pub fn destroy_rank_contexts(ranks: Vec<RankContext>) -> ExampleResult {
    for rank in ranks {
        rank.communicator.destroy()?;
    }
    Ok(())
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
