mod common;

use std::process;

use singe_cuda::memory::DeviceMemory;
use singe_nccl::{
    local::{LocalCommunicatorGroup, RootRank},
    types::ReductionOperator,
};

use crate::common::{ExampleResult, check_all_equal, rank_indexed_values};

const ELEMENT_COUNT: usize = 16;
const ROOT_RANK: i32 = 0;

fn run() -> ExampleResult {
    let group = LocalCommunicatorGroup::create_all_visible(2)?;
    let rank_count = group.rank_count();
    let root = RootRank::create(ROOT_RANK)?;

    let send_buffers = group.map_ranks(|rank| {
        rank.bind()?;
        Ok(DeviceMemory::from_slice(
            rank.communicator().cuda_context(),
            &rank_indexed_values(rank.id().get(), ELEMENT_COUNT),
        )?)
    })?;

    group.rank(root.id())?.bind()?;
    let mut recv = DeviceMemory::<f32>::create_uninit(
        group.rank(root.id())?.communicator().cuda_context(),
        ELEMENT_COUNT,
    )?;
    group.reduce_into(
        &send_buffers.map(DeviceMemory::view),
        root,
        recv.output(),
        ReductionOperator::Sum,
    )?;
    // SAFETY: the completed collective wrote the entire compact root output.
    let recv = unsafe { recv.assume_init() };

    let rank_sum = (0..rank_count)
        .map(|rank| rank as f32 * 1000.0)
        .sum::<f32>();
    let expected: Vec<_> = (0..ELEMENT_COUNT)
        .map(|index| rank_sum + index as f32 * rank_count as f32)
        .collect();
    let root_rank = group.rank(root.id())?;
    root_rank.bind()?;
    let actual = recv.copy_to_host_vec()?;
    check_all_equal(&actual, &expected, "reduce root")?;

    println!("reduce sum to root {ROOT_RANK} succeeded across {rank_count} rank(s)");

    drop(send_buffers);
    drop(recv);
    Ok(group.destroy()?)
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        process::exit(1);
    }
}
