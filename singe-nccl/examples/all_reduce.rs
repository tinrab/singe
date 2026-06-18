mod common;

use singe_cuda::memory::DeviceMemory;
use singe_nccl::{local::LocalCommunicatorGroup, types::ReductionOperator};

use crate::common::{ExampleResult, check_all_equal, rank_indexed_values};

const ELEMENT_COUNT: usize = 1024;

fn run() -> ExampleResult {
    let group = LocalCommunicatorGroup::create_all_visible(2)?;
    let rank_count = group.rank_count();

    let send_buffers = group.map_ranks(|rank| {
        rank.bind()?;
        let host = rank_indexed_values(rank.id().get(), ELEMENT_COUNT);
        Ok(DeviceMemory::from_slice(&host)?)
    })?;
    let mut recv_buffers = group.map_ranks(|_| Ok(DeviceMemory::<f32>::zeroes(ELEMENT_COUNT)?))?;

    group.all_reduce(&send_buffers, &mut recv_buffers, ReductionOperator::Sum)?;
    group.synchronize()?;

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
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
