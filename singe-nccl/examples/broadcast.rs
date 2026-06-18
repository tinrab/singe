mod common;

use singe_cuda::memory::DeviceMemory;
use singe_nccl::local::{LocalCommunicatorGroup, RootRank};

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
        Ok(DeviceMemory::from_slice(&send_host)?)
    })?;
    let mut recv_buffers = group.map_ranks(|_| Ok(DeviceMemory::<f32>::zeroes(ELEMENT_COUNT)?))?;

    group.broadcast(&send_buffers, &mut recv_buffers, root)?;
    group.synchronize()?;

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
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
