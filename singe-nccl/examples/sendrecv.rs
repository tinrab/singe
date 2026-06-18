mod common;

use singe_cuda::memory::DeviceMemory;

use crate::common::{
    ExampleResult, check_all_equal, create_rank_contexts, destroy_rank_contexts,
    rank_indexed_values,
};

const ELEMENT_COUNT: usize = 16;

fn run() -> ExampleResult {
    let ranks = create_rank_contexts(2)?;
    let rank_count = usize::try_from(ranks[0].rank_count()?)?;
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
        send_buffers.push(DeviceMemory::from_slice(&rank_indexed_values(
            rank.rank()?,
            ELEMENT_COUNT,
        ))?);
        recv_buffers.push(DeviceMemory::<f32>::zeroes(ELEMENT_COUNT)?);
    }

    singe_nccl::with_group(|| {
        for (((index, rank), send), (send_peer, recv_peer)) in ranks
            .iter()
            .enumerate()
            .zip(&send_buffers)
            .zip(peer_pairs.iter().copied())
        {
            rank.communicator()
                .send_memory(send, send_peer as i32, rank.stream())?;
            rank.communicator().recv_memory(
                &mut recv_buffers[index],
                recv_peer as i32,
                rank.stream(),
            )?;
        }
        Ok(())
    })?;

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
    destroy_rank_contexts(ranks)
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
