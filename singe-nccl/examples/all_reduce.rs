mod common;

use std::process;

use singe_cuda::memory::DeviceMemory;
use singe_nccl::{
    local::{LocalCommunicatorGroup, RankVec},
    types::ReductionOperator,
};

use crate::common::{ExampleResult, check_all_equal, rank_indexed_values};

const ELEMENT_COUNT: usize = 1024;

fn run() -> ExampleResult {
    let group = LocalCommunicatorGroup::create_all_visible(2)?;
    let rank_count = group.rank_count();

    let send_buffers = group.map_ranks(|rank| {
        rank.bind()?;
        let host = rank_indexed_values(rank.id().get(), ELEMENT_COUNT);
        Ok(DeviceMemory::from_slice(
            rank.communicator().cuda_context(),
            &host,
        )?)
    })?;
    let mut recv_buffers = group.map_ranks(|rank| {
        Ok(DeviceMemory::<f32>::create_uninit(
            rank.communicator().cuda_context(),
            ELEMENT_COUNT,
        )?)
    })?;

    group.all_reduce(
        &send_buffers.map(DeviceMemory::view),
        &mut recv_buffers.map_mut(DeviceMemory::output),
        ReductionOperator::Sum,
    )?;
    // SAFETY: the completed collective wrote every compact receive output.
    let recv_buffers = RankVec::from_vec(
        recv_buffers
            .into_vec()
            .into_iter()
            // SAFETY: the call above completed and wrote every element.
            .map(|buffer| unsafe { buffer.assume_init() })
            .collect(),
    );

    let rank_sum = (0..rank_count)
        .map(|rank| rank as f32 * 1000.0)
        .sum::<f32>();
    let expected: Vec<_> = (0..ELEMENT_COUNT)
        .map(|index| rank_sum + index as f32 * rank_count as f32)
        .collect();

    for (index, (rank, recv)) in group.ranks().zip(&recv_buffers).enumerate() {
        rank.bind()?;
        let actual = recv.copy_to_host_vec()?;
        check_all_equal(&actual, &expected, &format!("rank {index} all-reduce"))?;
    }

    println!(
        "all-reduce sum succeeded across {rank_count} rank(s) with {ELEMENT_COUNT} f32 values per rank"
    );

    drop(recv_buffers);
    drop(send_buffers);
    Ok(group.destroy()?)
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        process::exit(1);
    }
}
