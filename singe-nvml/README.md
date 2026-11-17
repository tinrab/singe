# singe-nvml

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-nvml.svg)](https://crates.io/crates/singe-nvml)
[![Documentation](https://docs.rs/singe-nvml/badge.svg)](https://docs.rs/singe-nvml)
![License](https://img.shields.io/crates/l/singe-nvml.svg)

Safe Rust wrappers for NVML, the NVIDIA Management Library, for monitoring and managing GPUs.
NVML ships with the NVIDIA driver and needs no CUDA Toolkit or CUDA runtime.

This crate provides:

- **Library:** `Library` owns an NVML initialization and reports driver and CUDA driver versions, enumerates devices and units, and creates event sets and GPM samples.
- **Devices:** `Device` queries memory, utilization, temperature, fans, clocks, power, ECC, PCI, topology, and processes, and changes the settings NVML exposes for them.
- **MIG:** GPU instances and compute instances on partitioned GPUs.
- **vGPU:** vGPU types and instances.
- **Operational events:** subscriptions to GPU operational events.

## Example

Print each GPU's name and free memory, then the NVML version:

```rust,no_run
use singe_nvml::{library::Library, types::InitFlags};

fn main() -> anyhow::Result<()> {
    let nvml = Library::create(InitFlags::empty())?;
    for index in 0..nvml.device_count()? {
        let device = nvml.device(index)?;
        let memory = device.memory_info()?;
        println!("{}: {} MiB free", device.name()?, memory.free / 1_048_576);
    }

    println!("NVML version: {}", singe_nvml::version()?);
    Ok(())
}
```

`Library::create` initializes NVML without initializing any GPU; NVML initializes a GPU when a handle to it is first requested.
Devices and other objects keep their `Library` alive, and NVML shuts down after the last `Library` clone and every object that holds one are dropped.
The queries and settings available depend on the GPU and driver, and changing a setting needs root or administrator permissions.

## More examples

- [Fan curve](./examples/fan_curve.rs): choose fan speeds from GPU temperature readings; it only prints them unless its `APPLY_CHANGES` constant is `true`, and then it restores default fan control before it exits.
