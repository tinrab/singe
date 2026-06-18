singe-nvml
===

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->
[![Latest version](https://img.shields.io/crates/v/singe-nvml.svg)](https://crates.io/crates/singe-nvml)
[![Documentation](https://docs.rs/singe-nvml/badge.svg)](https://docs.rs/singe-nvml)
![License](https://img.shields.io/crates/l/singe-nvml.svg)

Safe NVIDIA Management Library wrappers for monitoring, device management, MIG, and vGPU APIs.

This crate wraps NVML initialization and shutdown, system queries, device and
unit handles, MIG GPU and compute instances, vGPU types and instances, event
sets, clocks, power, thermals, memory, utilization, and other management
interfaces exposed by NVIDIA drivers.

## Setup

NVML is supplied by the NVIDIA driver rather than the CUDA Toolkit.
The build and runtime environment must be able to locate the driver NVML library, for example through the platform library path.

## Examples

```rust
use singe_nvml::library::Library;

let nvml = Library::create()?;
let count = nvml.device_count()?;

for index in 0..count {
    let device = nvml.device(index)?;
    let memory = device.memory_info()?;
    println!("{}: {} MiB free", device.name()?, memory.free / 1_048_576);
}

println!("NVML version: {}", singe_nvml::version()?);
```
