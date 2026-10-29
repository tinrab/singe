mod common;

use std::process;

use singe_cuda::memory::DeviceMemory;
use singe_nccl::local::{LocalCommunicatorGroup, RankVec, RootRank};

use crate::common::{ExampleResult, check_all_equal};

const ELEMENT_COUNT: usize = 16;
const ROOT_RANK: i32 = 0;

fn run() -> ExampleResult {
    let group = LocalCommunicatorGroup::create_all_visible(2)?;
    let rank_count = group.rank_count();
    let root = RootRank::create(ROOT_RANK)?;
    let root_index = usize::try_from(ROOT_RANK)?;
    let root_values: Vec<_> = (0..ELEMENT_COUNT).map(|index| index as f32 + 0.5).collect();

    let send_buffers = group.map_ranks(|rank| {
        rank.bind()?;
        let index = rank.id().get() as usize;
        let send_host = if index == root_index {
            root_values.clone()
        } else {
            vec![0.0; ELEMENT_COUNT]
        };
        Ok(DeviceMemory::from_slice(
            rank.communicator().cuda_context(),
            &send_host,
        )?)
    })?;
    let mut recv_buffers = group.map_ranks(|rank| {
        Ok(DeviceMemory::<f32>::create_uninit(
            rank.communicator().cuda_context(),
            ELEMENT_COUNT,
        )?)
    })?;

    group.broadcast(
        &send_buffers.map(DeviceMemory::view),
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
        rank.bind()?;
        let actual = recv.copy_to_host_vec()?;
        check_all_equal(&actual, &root_values, &format!("rank {index} broadcast"))?;
    }

    println!("broadcast from root {ROOT_RANK} succeeded across {rank_count} rank(s)");

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
