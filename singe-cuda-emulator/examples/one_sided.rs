//! Run NCCL's host one-sided operations on emulated GPUs: each rank puts its values into the next rank's window with a signal, and signals the previous rank without data.
//!
//! Pass the number of GPUs as the first argument; the default is 4.

use std::env;

use singe_cuda::{device::Device, memory::transfer};
use singe_nccl::{
    communicator::WaitSignal, group::Group, local::LocalCommunicatorGroup, memory::Memory,
    types::WindowFlags,
};
use singe_ptx::emulator::{EmulatorOptions, Topology};

const COUNT: usize = 512;
const PUT: i32 = 0;
const BARE: i32 = 1;

fn main() -> anyhow::Result<()> {
    run().map_err(|error| match singe_cuda_emulator::last_fault() {
        Some(fault) => error.context(format!("emulator fault: {fault}")),
        None => error,
    })
}

fn run() -> anyhow::Result<()> {
    let ranks: usize = env::args().nth(1).map_or(Ok(4), |count| count.parse())?;
    singe_cuda_emulator::install(EmulatorOptions {
        topology: Topology::node(ranks),
        ..Default::default()
    })?;
    let devices: Vec<Device> = (0..i32::try_from(ranks)?).map(Device::new).collect();
    let group = LocalCommunicatorGroup::create(&devices)?;

    // The first `COUNT` values are what a rank sends, and the next `COUNT` receive what the previous rank sends.
    let mut buffers = group.map_ranks(|rank| {
        let values: Vec<f32> = (0..2 * COUNT)
            .map(|index| {
                if index < COUNT {
                    (rank.id().get() as usize * COUNT + index) as f32
                } else {
                    -1.0
                }
            })
            .collect();
        Memory::from_slice(rank.communicator().cuda_context(), &values)
    })?;
    let registration = Group::start()?;
    let windows = group
        .ranks()
        .zip(buffers.iter_mut())
        .map(|(rank, buffer)| {
            rank.communicator()
                .register_window(buffer.view_mut(), WindowFlags::COLLECTIVE_SYMMETRIC)
        })
        .collect::<Result<Vec<_>, _>>()?;
    registration.end()?;

    let bytes = (COUNT * size_of::<f32>()) as u64;
    for (rank, window) in group.ranks().zip(&windows) {
        rank.bind()?;
        let id = rank.id().get() as usize;
        let (next, previous) = ((id + 1) % ranks, (id + ranks - 1) % ranks);
        // SAFETY: every rank's window holds `2 * COUNT` floats, and the windows stay registered until the synchronization below.
        unsafe {
            window.put_signal_async(
                0..bytes,
                window,
                i32::try_from(next)?,
                bytes,
                PUT,
                0,
                rank.stream(),
            )?;
            rank.communicator()
                .signal_async(i32::try_from(previous)?, BARE, 0, rank.stream())?;
        }
    }

    for rank in group.ranks() {
        rank.bind()?;
        let id = rank.id().get() as usize;
        let (next, previous) = ((id + 1) % ranks, (id + ranks - 1) % ranks);
        let waits = [
            WaitSignal {
                count: 1,
                peer: i32::try_from(previous)?,
                signal: PUT,
                context: 0,
            },
            WaitSignal {
                count: 1,
                peer: i32::try_from(next)?,
                signal: BARE,
                context: 0,
            },
        ];
        // SAFETY: the peers send these signals above.
        unsafe {
            rank.communicator()
                .wait_signals_async(&waits, rank.stream())?;
        }
    }

    group.synchronize()?;

    drop(windows);
    for (rank, buffer) in group.ranks().zip(&buffers) {
        rank.bind()?;
        let id = rank.id().get() as usize;
        let previous = (id + ranks - 1) % ranks;
        let values = transfer::copy_to_host_vec(buffer)?;
        let wrong =
            (0..COUNT).find(|&index| values[COUNT + index] != (previous * COUNT + index) as f32);
        anyhow::ensure!(
            wrong.is_none(),
            "rank {id} received a wrong value at {wrong:?}"
        );
    }

    println!("one-sided ring of {COUNT} floats per rank succeeded on {ranks} emulated GPUs");

    drop(buffers);
    Ok(group.destroy()?)
}
