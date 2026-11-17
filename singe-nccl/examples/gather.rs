mod common;

use singe_cuda::memory::DeviceMemory;
use singe_nccl::{local::LocalCommunicatorGroup, types::RootRank};

use crate::common::{check_all_equal, first_devices, rank_indexed_values};

const ELEMENTS_PER_RANK: usize = 8;
const ROOT_RANK: i32 = 0;

fn main() -> anyhow::Result<()> {
    let group = LocalCommunicatorGroup::create(&first_devices(2)?)?;
    let rank_count = group.rank_count();
    let root = RootRank::create(ROOT_RANK)?;

    let send_buffers = group.map_ranks(|rank| {
        rank.bind()?;
        Ok(DeviceMemory::from_slice(
            rank.communicator().cuda_context(),
            &rank_indexed_values(rank.id().get(), ELEMENTS_PER_RANK),
        )?)
    })?;
    group.rank(root.id())?.bind()?;
    let mut recv = DeviceMemory::<f32>::create_uninit(
        group.rank(root.id())?.communicator().cuda_context(),
        ELEMENTS_PER_RANK * rank_count,
    )?;

    group.gather_into(&send_buffers.map(DeviceMemory::view), root, recv.output())?;
    // SAFETY: the completed collective wrote the entire compact root output.
    let recv = unsafe { recv.assume_init() };

    let expected: Vec<_> = group
        .ranks()
        .flat_map(|rank| rank_indexed_values(rank.id().get(), ELEMENTS_PER_RANK))
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
