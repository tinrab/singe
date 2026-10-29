mod common;

use std::process;

use singe_cuda::memory::DeviceMemory;
use singe_nccl::local::{LocalCommunicatorGroup, RankVec};

use crate::common::{ExampleResult, check_all_equal, rank_indexed_values};

const ELEMENTS_PER_RANK: usize = 8;

fn run() -> ExampleResult {
    let group = LocalCommunicatorGroup::create_all_visible(2)?;
    let rank_count = group.rank_count();

    let send_buffers = group.map_ranks(|rank| {
        rank.bind()?;
        Ok(DeviceMemory::from_slice(
            rank.communicator().cuda_context(),
            &rank_indexed_values(rank.id().get(), ELEMENTS_PER_RANK),
        )?)
    })?;
    let mut recv_buffers = group.map_ranks(|rank| {
        Ok(DeviceMemory::<f32>::create_uninit(
            rank.communicator().cuda_context(),
            ELEMENTS_PER_RANK * rank_count,
        )?)
    })?;

    group.all_gather(
        &send_buffers.map(DeviceMemory::view),
        &mut recv_buffers.map_mut(DeviceMemory::output),
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

    let expected: Vec<_> = (0..rank_count)
        .flat_map(|rank| rank_indexed_values(rank as i32, ELEMENTS_PER_RANK))
        .collect();

    for (index, (rank, recv)) in group.ranks().zip(&recv_buffers).enumerate() {
        rank.bind()?;
        let actual = recv.copy_to_host_vec()?;
        check_all_equal(&actual, &expected, &format!("rank {index} all-gather"))?;
    }

    println!(
        "all-gather succeeded across {rank_count} rank(s) with {ELEMENTS_PER_RANK} f32 values per rank"
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
