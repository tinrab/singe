mod common;

use singe_cuda::memory::DeviceMemory;
use singe_nccl::local::LocalCommunicatorGroup;

use crate::common::{ExampleResult, check_all_equal, rank_indexed_values};

const ELEMENTS_PER_RANK: usize = 8;

fn run() -> ExampleResult {
    let group = LocalCommunicatorGroup::create_all_visible(2)?;
    let rank_count = group.rank_count();

    let send_buffers = group.map_ranks(|rank| {
        rank.bind()?;
        Ok(DeviceMemory::from_slice(&rank_indexed_values(
            rank.id().get(),
            ELEMENTS_PER_RANK,
        ))?)
    })?;
    let mut recv_buffers =
        group.map_ranks(|_| Ok(DeviceMemory::<f32>::zeroes(ELEMENTS_PER_RANK * rank_count)?))?;

    group.all_gather(&send_buffers, &mut recv_buffers)?;
    group.synchronize()?;

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
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
