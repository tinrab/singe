mod common;

use std::process;

use singe_cuda::{memory::DeviceMemory, view::DeviceOutputView};

use crate::common::{
    ExampleResult, check_all_equal, create_power_of_two_rank_contexts, rank_indexed_values,
};

const ELEMENTS_PER_RANK: usize = 4;

fn run() -> ExampleResult {
    let group = create_power_of_two_rank_contexts(2)?;
    let ranks: Vec<_> = group.ranks().collect();
    let rank_count = group.rank_count();

    let total_elements = ELEMENTS_PER_RANK * rank_count;
    let mut recv_buffers = Vec::with_capacity(rank_count);
    let mut rank_indices = Vec::with_capacity(rank_count);

    for rank in &ranks {
        rank.bind()?;
        let rank_index = usize::try_from(rank.id().get())?;
        rank_indices.push(rank_index);
        let mut initial = vec![0.0; total_elements];
        let begin = rank_index * ELEMENTS_PER_RANK;
        let end = begin + ELEMENTS_PER_RANK;
        initial[begin..end]
            .copy_from_slice(&rank_indexed_values(rank.id().get(), ELEMENTS_PER_RANK));
        recv_buffers.push(DeviceMemory::from_slice(
            rank.communicator().cuda_context(),
            &initial,
        )?);
    }

    let mut mask = 1usize;
    while mask < rank_count {
        // SAFETY: the example keeps these buffers alive and synchronizes before reading the results.
        unsafe {
            group.sync_scope(|| {
                for (rank_index, rank) in rank_indices.iter().copied().zip(&ranks) {
                    let send_start = (rank_index & !(mask - 1)) * ELEMENTS_PER_RANK;
                    let recv_start = ((send_start / ELEMENTS_PER_RANK) ^ mask) * ELEMENTS_PER_RANK;
                    let len = ELEMENTS_PER_RANK * mask;
                    let peer = (rank_index ^ mask) as i32;

                    let buffer = &mut recv_buffers[rank_index];
                    let mut whole = buffer.view_mut();
                    let (mut first, mut second) = whole.split_at_mut(send_start.max(recv_start))?;
                    // Each step reads and writes disjoint regions of this allocation.
                    let (send, recv) = if send_start < recv_start {
                        (
                            first.slice(send_start..send_start + len)?,
                            DeviceOutputView::from_output(&mut second).slice(..len)?,
                        )
                    } else {
                        (
                            second.slice(..len)?,
                            DeviceOutputView::from_output(&mut first)
                                .slice(recv_start..recv_start + len)?,
                        )
                    };
                    rank.communicator().send(send, peer, rank.stream())?;
                    rank.communicator().recv(recv, peer, rank.stream())?;
                }
                Ok(())
            })
        }?;
        mask <<= 1;
    }

    let expected: Vec<_> = (0..rank_count)
        .flat_map(|rank| rank_indexed_values(rank as i32, ELEMENTS_PER_RANK))
        .collect();

    for (index, (rank, recv)) in ranks.iter().zip(&recv_buffers).enumerate() {
        rank.bind()?;
        let actual = recv.copy_to_host_vec()?;
        check_all_equal(&actual, &expected, &format!("rank {index} hypercube"))?;
    }

    println!(
        "hypercube all-gather succeeded across {rank_count} rank(s) with {ELEMENTS_PER_RANK} f32 values per rank"
    );

    drop(recv_buffers);
    Ok(group.destroy()?)
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        process::exit(1);
    }
}
