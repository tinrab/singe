mod common;

use singe_cuda::memory::DeviceMemory;
use singe_nccl::{local::LocalCommunicatorGroup, types::ReductionOperator};

use crate::common::{ExampleResult, check_all_equal};

const ELEMENTS_PER_RANK: usize = 4;

fn run() -> ExampleResult {
    let group = LocalCommunicatorGroup::create_all_visible(2)?;
    let rank_count = group.rank_count();

    let send_buffers = group.map_ranks(|rank| {
        rank.bind()?;
        let rank_index = rank.id().get() as usize;
        let send_host: Vec<_> = (0..rank_count)
            .flat_map(|segment| {
                (0..ELEMENTS_PER_RANK).map(move |offset| {
                    rank_index as f32 * 1000.0 + segment as f32 * 100.0 + offset as f32
                })
            })
            .collect();
        Ok(DeviceMemory::from_slice(&send_host)?)
    })?;
    let mut recv_buffers =
        group.map_ranks(|_| Ok(DeviceMemory::<f32>::zeroes(ELEMENTS_PER_RANK)?))?;

    group.reduce_scatter(
        &send_buffers,
        &mut recv_buffers,
        ELEMENTS_PER_RANK,
        ReductionOperator::Sum,
    )?;
    group.synchronize()?;

    for (index, (rank, recv)) in group.ranks().zip(&recv_buffers).enumerate() {
        let rank_sum = (0..rank_count)
            .map(|rank| rank as f32 * 1000.0)
            .sum::<f32>();
        let expected: Vec<_> = (0..ELEMENTS_PER_RANK)
            .map(|offset| {
                rank_sum
                    + index as f32 * 100.0 * rank_count as f32
                    + offset as f32 * rank_count as f32
            })
            .collect();
        rank.bind()?;
        let actual = recv.copy_to_host_vec()?;
        check_all_equal(&actual, &expected, &format!("rank {index} reduce-scatter"))?;
    }

    println!(
        "reduce-scatter sum succeeded across {rank_count} rank(s) with {ELEMENTS_PER_RANK} f32 values per rank"
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
