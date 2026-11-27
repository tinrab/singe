//! Run NCCL device-API kernels on emulated GPUs: an all-reduce through symmetric memory and LSA barriers, compiled from `kernels/lsa.cu`.
//! Before the all-reduce, each rank reads the next rank's window through the pointer `ncclGetLsaDevicePointer` returns.
//!
//! Pass the number of GPUs as the first argument; the default is 4.

use std::env;

use singe_cuda::{
    device::Device,
    dim::Dim3,
    jit::JitOptions,
    memory::{DeviceMemory, transfer},
    module::{KernelParameters, LaunchConfig, ModuleImage},
};
use singe_nccl::{
    communicator::device::{DeviceCommunicatorBuilder, DeviceRequirements},
    group::Group,
    local::LocalCommunicatorGroup,
    memory::Memory,
    types::WindowFlags,
};
use singe_ptx::emulator::{EmulatorOptions, Topology};

const PTX: &str = include_str!("kernels/lsa.ptx");
const COUNT: usize = 1000;
const BLOCKS: u32 = 2;
const THREADS: u32 = 64;
const ROUNDS: u32 = 5;

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

    let mut buffers = group.map_ranks(|rank| {
        let values: Vec<f32> = (0..COUNT)
            .map(|index| (rank.id().get() as f32).mul_add(100.0, index as f32))
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
    let creation = Group::start()?;
    let requirements = DeviceRequirements {
        lsa_barriers: i32::try_from(BLOCKS)?,
        ..DeviceRequirements::default()
    };
    let communicators = group
        .ranks()
        .map(|rank| {
            DeviceCommunicatorBuilder::new(rank.communicator(), requirements.clone()).create()
        })
        .collect::<Result<Vec<_>, _>>()?;
    creation.end()?;

    // Kernels on different ranks wait for each other, so everything a launch needs is ready before the first one.
    let mut prepared = Vec::new();
    for rank in group.ranks() {
        rank.bind()?;
        let context = rank.communicator().cuda_context();
        let module = context.load_module(
            &ModuleImage::from_ptx(PTX.as_bytes())?,
            JitOptions::default(),
        )?;
        prepared.push((
            module,
            DeviceMemory::<u32>::zeroed(context, BLOCKS as usize)?,
            DeviceMemory::<f32>::zeroed(context, COUNT)?,
        ));
    }

    let config = LaunchConfig::create(Dim3::new(BLOCKS, 1, 1), Dim3::new(THREADS, 1, 1), 0)?;
    for ((rank, window), (module, _, neighbor)) in group.ranks().zip(&windows).zip(&mut prepared) {
        rank.bind()?;
        let next = (rank.id().get() as usize + 1) % ranks;
        let peer = window.lsa_pointer(0, i32::try_from(next)?)?;
        let mut parameters = KernelParameters::default();
        parameters
            .device_slice_mut(neighbor)
            .owned_arg(peer.addr() as u64)
            .owned_arg(COUNT as u64);
        // SAFETY: `peer` points at the next rank's `COUNT` floats, which nothing writes until the synchronization below.
        unsafe {
            module
                .function("add_peer")?
                .launch_async(&config, &mut parameters, rank.stream())?;
        }
    }

    group.synchronize()?;

    for (((rank, communicator), window), (module, counts, _)) in group
        .ranks()
        .zip(&communicators)
        .zip(&windows)
        .zip(&mut prepared)
    {
        rank.bind()?;
        let mut parameters = KernelParameters::default();
        parameters
            .arg(communicator.as_raw())
            .owned_arg(window.as_raw())
            .owned_arg(0u64)
            .owned_arg(COUNT as u64);
        // SAFETY: the window holds `COUNT` floats on every rank, and everything the kernel reads stays alive until the synchronization below.
        unsafe {
            module.function("lsa_all_reduce")?.launch_async(
                &config,
                &mut parameters,
                rank.stream(),
            )?;
        }

        let mut parameters = KernelParameters::default();
        parameters
            .arg(communicator.as_raw())
            .device_slice_mut(counts)
            .owned_arg(ROUNDS);
        // SAFETY: `counts` holds one counter per block and outlives the synchronization below.
        unsafe {
            module.function("lsa_barrier")?.launch_async(
                &config,
                &mut parameters,
                rank.stream(),
            )?;
        }
    }

    group.synchronize()?;

    drop(communicators);
    drop(windows);
    let offsets: f32 = (0..ranks).map(|rank| rank as f32 * 100.0).sum();
    for ((rank, buffer), (_, counts, neighbor)) in group.ranks().zip(&buffers).zip(&prepared) {
        rank.bind()?;
        let id = rank.id().get() as usize;
        let counts = counts.copy_to_host_vec()?;
        anyhow::ensure!(
            counts.iter().all(|count| *count == ROUNDS),
            "rank {id} counted {counts:?}"
        );

        let next = (id + 1) % ranks;
        let neighbor = neighbor.copy_to_host_vec()?;
        let wrong = (0..COUNT).find(|&index| neighbor[index] != (next * 100 + index) as f32);
        anyhow::ensure!(
            wrong.is_none(),
            "rank {id} read a wrong value of rank {next} at {wrong:?}"
        );

        let sums = transfer::copy_to_host_vec(buffer)?;
        let wrong = (0..COUNT).find(|&index| sums[index] != (index * ranks) as f32 + offsets);
        anyhow::ensure!(wrong.is_none(), "rank {id} has a wrong sum at {wrong:?}");
    }

    println!(
        "lsa all-reduce of {COUNT} floats and {ROUNDS} barrier rounds succeeded on {ranks} emulated GPUs"
    );

    drop(buffers);
    Ok(group.destroy()?)
}
