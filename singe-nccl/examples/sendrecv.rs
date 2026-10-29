mod common;

use std::process;

use singe_cuda::memory::DeviceMemory;

use crate::common::{ExampleResult, check_all_equal, create_rank_contexts, rank_indexed_values};

const ELEMENT_COUNT: usize = 16;

fn run() -> ExampleResult {
    let group = create_rank_contexts(2)?;
    let ranks: Vec<_> = group.ranks().collect();
    let rank_count = group.rank_count();
    let peer_pairs: Vec<_> = (0..rank_count)
        .map(|index| {
            (
                (index + 1) % rank_count as usize,
                (index + rank_count - 1) % rank_count,
            )
        })
        .collect();

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
        group.sync_scope(|| {
            for (((index, rank), send), (send_peer, recv_peer)) in ranks
                .iter()
                .enumerate()
                .zip(&send_buffers)
                .zip(peer_pairs.iter().copied())
            {
                rank.communicator()
                    .send(send.view(), send_peer as i32, rank.stream())?;
                rank.communicator().recv(
                    recv_buffers[index].output(),
                    recv_peer as i32,
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
        let peer = ((index + rank_count - 1) % rank_count) as i32;
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

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        process::exit(1);
    }
}
