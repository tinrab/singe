mod common;

use std::process;

use singe_cuda::memory::DeviceMemory;
use singe_nccl::local::{LocalCommunicatorGroup, RankVec, RootRank};

use crate::common::{ExampleResult, check_all_equal, rank_indexed_values};

const ELEMENTS_PER_RANK: usize = 8;
const ROOT_RANK: i32 = 0;

fn run() -> ExampleResult {
    let group = LocalCommunicatorGroup::create_all_visible(2)?;
    let rank_count = group.rank_count();
    let root = RootRank::create(ROOT_RANK)?;
    let root_values: Vec<_> = (0..rank_count)
        .flat_map(|rank| rank_indexed_values(rank as i32, ELEMENTS_PER_RANK))
        .collect();

    group.rank(root.id())?.bind()?;
    let root_send = DeviceMemory::from_slice(
        group.rank(root.id())?.communicator().cuda_context(),
        &root_values,
    )?;
    let mut recv_buffers = group.map_ranks(|rank| {
        rank.bind()?;
        Ok(DeviceMemory::<f32>::create_uninit(
            rank.communicator().cuda_context(),
            ELEMENTS_PER_RANK,
        )?)
    })?;

    group.scatter_from(
        root_send.view(),
        &mut recv_buffers.map_mut(DeviceMemory::output),
        root,
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

    for (index, (rank, recv)) in group.ranks().zip(&recv_buffers).enumerate() {
        let expected = rank_indexed_values(index as i32, ELEMENTS_PER_RANK);
        rank.bind()?;
        let actual = recv.copy_to_host_vec()?;
        check_all_equal(&actual, &expected, &format!("rank {index} scatter"))?;
    }

    println!("scatter from root {ROOT_RANK} succeeded across {rank_count} rank(s)");

    drop(recv_buffers);
    drop(root_send);
    Ok(group.destroy()?)
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        process::exit(1);
    }
}
