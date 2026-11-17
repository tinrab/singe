mod common;

use singe_cuda::memory::DeviceMemory;
use singe_nccl::local::{LocalCommunicatorGroup, RankVec};

use crate::common::{check_all_equal, first_devices};

const ELEMENTS_PER_PEER: usize = 4;

fn main() -> anyhow::Result<()> {
    let group = LocalCommunicatorGroup::create(&first_devices(2)?)?;
    let rank_count = group.rank_count();
    let element_count = ELEMENTS_PER_PEER * rank_count;

    let send_buffers = group.map_ranks(|rank| {
        rank.bind()?;
        let rank_index = rank.id().get() as usize;
        let send_host: Vec<_> = (0..rank_count)
            .flat_map(|peer| {
                (0..ELEMENTS_PER_PEER).map(move |offset| {
                    (peer as f32).mul_add(100.0, rank_index as f32 * 1000.0) + offset as f32
                })
            })
            .collect();
        Ok(DeviceMemory::from_slice(
            rank.communicator().cuda_context(),
            &send_host,
        )?)
    })?;
    let mut recv_buffers = group.map_ranks(|rank| {
        Ok(DeviceMemory::<f32>::create_uninit(
            rank.communicator().cuda_context(),
            element_count,
        )?)
    })?;

    group.all_to_all(
        &send_buffers.map(DeviceMemory::view),
        &mut recv_buffers.map_mut(DeviceMemory::output),
        ELEMENTS_PER_PEER,
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
        let expected: Vec<_> = (0..rank_count)
            .flat_map(|peer| {
                (0..ELEMENTS_PER_PEER).map(move |offset| {
                    (index as f32).mul_add(100.0, peer as f32 * 1000.0) + offset as f32
                })
            })
            .collect();
        rank.bind()?;
        let actual = recv.copy_to_host_vec()?;
        check_all_equal(&actual, &expected, &format!("rank {index} all-to-all"))?;
    }

    println!(
        "all-to-all succeeded across {rank_count} rank(s) with {ELEMENTS_PER_PEER} f32 values per peer"
    );

    drop(recv_buffers);
    drop(send_buffers);
    Ok(group.destroy()?)
}
