# singe-cuda-emulator

[![Latest version](https://img.shields.io/crates/v/singe-cuda-emulator.svg)](https://crates.io/crates/singe-cuda-emulator)
[![Documentation](https://docs.rs/singe-cuda-emulator/badge.svg)](https://docs.rs/singe-cuda-emulator)
![License](https://img.shields.io/crates/l/singe-cuda-emulator.svg)

Emulated CUDA and NCCL on the CPU.

Programs built on `singe-cuda` and `singe-nccl` run unchanged on emulated GPUs, with no GPU, CUDA, or NCCL installed.
Call `install` before the first CUDA or NCCL call.

This program runs an NCCL all-reduce across eight emulated GPUs:

```rust,no_run
use singe_cuda::{device::Device, memory::DeviceMemory};
use singe_nccl::{local::LocalCommunicatorGroup, types::ReductionOperator};
use singe_ptx::emulator::{EmulatorOptions, Topology};

fn main() -> anyhow::Result<()> {
    // Eight GPUs, emulated on the CPU. No GPU, CUDA, or NCCL installed.
    singe_cuda_emulator::install(EmulatorOptions {
        topology: Topology::node(8),
        ..Default::default()
    })?;

    let devices: Vec<_> = (0..Device::count()?).map(Device::new).collect();
    let group = LocalCommunicatorGroup::create(&devices)?;

    // GPU n starts with [n, n, n, n].
    let mut buffers = group.map_ranks(|rank| {
        rank.bind()?;
        let values = [rank.id().get() as f32; 4];
        Ok(DeviceMemory::from_slice(
            rank.communicator().cuda_context(),
            &values,
        )?)
    })?;

    // NCCL all-reduce across all eight GPUs.
    group.all_reduce_in_place(
        &mut buffers.map_mut(DeviceMemory::view_mut),
        ReductionOperator::Sum,
    )?;

    for (rank, buffer) in group.ranks().zip(buffers.iter()) {
        rank.bind()?;
        println!("GPU {}: {:?}", rank.id().get(), buffer.copy_to_host_vec()?);
    }
    Ok(())
}
```

Every GPU ends with the sum 0 + 1 + ... + 7:

```text
GPU 0: [28.0, 28.0, 28.0, 28.0]
GPU 1: [28.0, 28.0, 28.0, 28.0]
...
GPU 7: [28.0, 28.0, 28.0, 28.0]
```

## Coverage

- CUDA: devices, contexts, streams, events, memory, virtual memory, and PTX kernels.
- NCCL: collectives, send and receive, host signals, and the device API with LSA, multimem, and GIN.

Kernels must be PTX.
Kernels that use GIN (`ncclGin`) must be compiled with `-DNCCL_GIN_GDAKI_ENABLE=0 -DNCCL_GIN_GPI_ENABLE=0 -DNCCL_GIN_EFA_GDA_ENABLE=0`, which leaves only NCCL's proxy backend, the one the emulator serves.
A failed kernel or NCCL call faults its device, as a CUDA error does, and `last_fault` returns the cause.

Not emulated: timing, weak memory ordering, ranks in separate processes, and the other NVIDIA libraries.
