//! Run an NCCL device-API all-to-all through GIN puts and signals on emulated GPUs, compiled from `kernels/gin.cu`.
//!
//! The emulator performs GIN puts the way NCCL's host proxy backend does, so the ranks may sit on different nodes.
//! Pass the number of nodes and the GPUs per node as arguments; the default is 2 nodes of 2 GPUs.

use std::env;

use singe_cuda::{
    device::Device,
    dim::Dim3,
    jit::JitOptions,
    memory::transfer,
    module::{KernelParameters, LaunchConfig, ModuleImage},
};
use singe_nccl::{
    communicator::device::{DeviceCommunicatorBuilder, DeviceRequirements, GinConnection},
    group::Group,
    local::LocalCommunicatorGroup,
    memory::Memory,
    types::WindowFlags,
};
use singe_ptx::emulator::{EmulatorOptions, Topology};

const PTX: &str = include_str!("kernels/gin.ptx");
const COUNT: usize = 256;
const BLOCKS: u32 = 2;
const THREADS: u32 = 64;

fn main() -> anyhow::Result<()> {
    run().map_err(|error| match singe_cuda_emulator::last_fault() {
        Some(fault) => error.context(format!("emulator fault: {fault}")),
        None => error,
    })
}

/// Returns the value rank `from` sends to rank `to` at `index`.
fn value(from: usize, to: usize, index: usize) -> f32 {
    (from * 10_000 + to * 1000 + index) as f32
}

fn run() -> anyhow::Result<()> {
    let mut arguments = env::args().skip(1);
    let nodes: usize = arguments.next().map_or(Ok(2), |count| count.parse())?;
    let devices_per_node: usize = arguments.next().map_or(Ok(2), |count| count.parse())?;
    let topology = Topology {
        nodes,
        devices_per_node,
    };
    let ranks = topology.device_count();
    singe_cuda_emulator::install(EmulatorOptions {
        topology,
        ..Default::default()
    })?;
    let devices: Vec<Device> = (0..i32::try_from(ranks)?).map(Device::new).collect();
    let group = LocalCommunicatorGroup::create(&devices)?;

    let mut sends = group.map_ranks(|rank| {
        let from = rank.id().get() as usize;
        let values: Vec<f32> = (0..ranks * COUNT)
            .map(|index| value(from, index / COUNT, index % COUNT))
            .collect();
        Memory::from_slice(rank.communicator().cuda_context(), &values)
    })?;
    let mut receives = group.map_ranks(|rank| {
        Memory::from_slice(
            rank.communicator().cuda_context(),
            &vec![0.0f32; ranks * COUNT],
        )
    })?;
    let registration = Group::start()?;
    let windows = group
        .ranks()
        .zip(sends.iter_mut().zip(receives.iter_mut()))
        .map(|(rank, (send, receive))| {
            let communicator = rank.communicator();
            Ok((
                communicator.register_window(send.view_mut(), WindowFlags::COLLECTIVE_SYMMETRIC)?,
                communicator
                    .register_window(receive.view_mut(), WindowFlags::COLLECTIVE_SYMMETRIC)?,
            ))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    registration.end()?;
    let creation = Group::start()?;
    let requirements = DeviceRequirements {
        gin_connection: GinConnection::Full,
        gin_contexts: 1,
        gin_signals: i32::try_from(BLOCKS)?,
        world_gin_barriers: i32::try_from(BLOCKS)?,
        ..DeviceRequirements::default()
    };
    let communicators = group
        .ranks()
        .map(|rank| {
            DeviceCommunicatorBuilder::new(rank.communicator(), requirements.clone()).create()
        })
        .collect::<Result<Vec<_>, _>>()?;
    creation.end()?;

    // Kernels on different ranks wait for each other, so every module is loaded before the first launch.
    let mut modules = Vec::new();
    for rank in group.ranks() {
        rank.bind()?;
        let context = rank.communicator().cuda_context();
        modules.push(context.load_module(
            &ModuleImage::from_ptx(PTX.as_bytes())?,
            JitOptions::default(),
        )?);
    }

    let config = LaunchConfig::create(Dim3::new(BLOCKS, 1, 1), Dim3::new(THREADS, 1, 1), 0)?;
    for (((rank, communicator), (send, receive)), module) in group
        .ranks()
        .zip(&communicators)
        .zip(&windows)
        .zip(&modules)
    {
        rank.bind()?;
        let mut parameters = KernelParameters::default();
        parameters
            .owned_arg(send.as_raw())
            .owned_arg(0u64)
            .owned_arg(receive.as_raw())
            .owned_arg(0u64)
            .owned_arg(COUNT as u64)
            .arg(communicator.as_raw());
        // SAFETY: both windows hold `COUNT` floats per rank on every rank, and everything the kernel reads stays alive until the synchronization below.
        unsafe {
            module.function("gin_all_to_all")?.launch_async(
                &config,
                &mut parameters,
                rank.stream(),
            )?;
        }
    }

    group.synchronize()?;

    drop(communicators);
    drop(windows);
    for (rank, receive) in group.ranks().zip(&receives) {
        rank.bind()?;
        let to = rank.id().get() as usize;
        let received = transfer::copy_to_host_vec(receive)?;
        let wrong = (0..ranks * COUNT)
            .find(|&index| received[index] != value(index / COUNT, to, index % COUNT));
        anyhow::ensure!(
            wrong.is_none(),
            "rank {to} received a wrong value at {wrong:?}"
        );
    }

    println!(
        "gin all-to-all of {COUNT} floats per pair succeeded on {ranks} emulated GPUs, {devices_per_node} per node"
    );

    drop(sends);
    drop(receives);
    Ok(group.destroy()?)
}
