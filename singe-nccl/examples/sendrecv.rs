mod common;

use singe_cuda::memory::DeviceMemory;

use crate::common::{check_all_equal, create_rank_contexts, rank_indexed_values};

const ELEMENT_COUNT: usize = 16;

fn main() -> anyhow::Result<()> {
    let group = create_rank_contexts(2)?;
    let ranks: Vec<_> = group.ranks().collect();
    let rank_count = group.rank_count();
    let peer_pairs: Vec<_> = (0..rank_count)
        .map(|index| {
            Ok((
                i32::try_from((index + 1) % rank_count)?,
                i32::try_from((index + rank_count - 1) % rank_count)?,
            ))
        })
        .collect::<anyhow::Result<_>>()?;

    let mut send_buffers = Vec::with_capacity(rank_count);
    let mut recv_buffers = Vec::with_capacity(rank_count);

    for rank in &ranks {
        rank.bind()?;
        send_buffers.push(DeviceMemory::from_slice(
            rank.communicator().cuda_context(),
            &rank_indexed_values(rank.id().get(), ELEMENT_COUNT),
        )?);
        recv_buffers.push(DeviceMemory::<f32>::create_uninit(
            rank.communicator().cuda_context(),
            ELEMENT_COUNT,
        )?);
    }

    // SAFETY: the example keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        group.completed_group(|| {
            for (((index, rank), send), (send_peer, recv_peer)) in ranks
                .iter()
                .enumerate()
                .zip(&send_buffers)
                .zip(peer_pairs.iter().copied())
            {
                rank.communicator()
                    .send_async(send.view(), send_peer, rank.stream())?;
                rank.communicator().recv_async(
                    recv_buffers[index].output(),
                    recv_peer,
                    rank.stream(),
                )?;
            }
            Ok(())
        })
    }?;

    // SAFETY: all point-to-point receives completed and filled their entire buffers.
    let recv_buffers: Vec<_> = recv_buffers
        .into_iter()
        .map(|buffer| unsafe { buffer.assume_init() })
        .collect();

    for (index, (rank, recv)) in ranks.iter().zip(&recv_buffers).enumerate() {
        let peer = i32::try_from((index + rank_count - 1) % rank_count)?;
        let expected = rank_indexed_values(peer, ELEMENT_COUNT);
        rank.bind()?;
        let actual = recv.copy_to_host_vec()?;
        check_all_equal(&actual, &expected, &format!("rank {index} sendrecv"))?;
    }

    println!("ring send/recv succeeded across {rank_count} rank(s)");

    drop(recv_buffers);
    drop(send_buffers);
    Ok(group.destroy()?)
}
