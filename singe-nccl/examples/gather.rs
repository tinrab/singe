mod common;

use singe_cuda::memory::DeviceMemory;
use singe_nccl::local::{LocalCommunicatorGroup, RootRank};

use crate::common::{ExampleResult, check_all_equal, rank_indexed_values};

const ELEMENTS_PER_RANK: usize = 8;
const ROOT_RANK: i32 = 0;

fn run() -> ExampleResult {
    let group = LocalCommunicatorGroup::create_all_visible(2)?;
    let rank_count = group.rank_count();
    let root = RootRank::create(ROOT_RANK)?;

    let send_buffers = group.map_ranks(|rank| {
        rank.bind()?;
        Ok(DeviceMemory::from_slice(&rank_indexed_values(
            rank.id().get(),
            ELEMENTS_PER_RANK,
        ))?)
    })?;
    group.rank(root.id())?.bind()?;
    let mut recv = DeviceMemory::<f32>::zeroes(ELEMENTS_PER_RANK * rank_count)?;

    group.gather_into(&send_buffers, root, &mut recv)?;
    group.synchronize()?;

    let expected: Vec<_> = (0..rank_count)
        .flat_map(|rank| rank_indexed_values(rank as i32, ELEMENTS_PER_RANK))
        .collect();

    let root_rank = group.rank(root.id())?;
    root_rank.bind()?;
    let actual = recv.copy_to_host_vec()?;
    check_all_equal(&actual, &expected, "gather root")?;

    println!("gather to root {ROOT_RANK} succeeded across {rank_count} rank(s)");

    drop(send_buffers);
    drop(recv);
    Ok(group.destroy()?)
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
