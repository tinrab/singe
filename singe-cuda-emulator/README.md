# singe-cuda-emulator

[![Latest version](https://img.shields.io/crates/v/singe-cuda-emulator.svg)](https://crates.io/crates/singe-cuda-emulator)
[![Documentation](https://docs.rs/singe-cuda-emulator/badge.svg)](https://docs.rs/singe-cuda-emulator)
![License](https://img.shields.io/crates/l/singe-cuda-emulator.svg)

Emulated CUDA and NCCL on the CPU.

Programs built on `singe-cuda` and `singe-nccl` run unchanged on emulated GPUs, with no GPU, CUDA, or NCCL installed.
Call `install` before the first CUDA or NCCL call:

```rust,no_run
use singe_ptx::emulator::{EmulatorOptions, Topology};

fn main() -> anyhow::Result<()> {
    singe_cuda_emulator::install(EmulatorOptions {
        topology: Topology::node(4),
        ..Default::default()
    })?;
    // singe-cuda and singe-nccl now see four devices.
    Ok(())
}
```

## Coverage

- CUDA: devices, contexts, streams, events, memory, virtual memory, and PTX kernels.
- NCCL: collectives, send and receive, host signals, and the device API with LSA, multimem, and GIN.

Kernels must be PTX.
Kernels that use GIN (`ncclGin`) must be compiled with `-DNCCL_GIN_GDAKI_ENABLE=0 -DNCCL_GIN_GPI_ENABLE=0 -DNCCL_GIN_EFA_GDA_ENABLE=0`, which leaves only NCCL's proxy backend, the one the emulator serves.
A failed kernel or NCCL call faults its device, as a CUDA error does, and `last_fault` returns the cause.

Not emulated: timing, weak memory ordering, ranks in separate processes, and the other NVIDIA libraries.
