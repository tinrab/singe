# singe-nvml

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-nvml.svg)](https://crates.io/crates/singe-nvml)
[![Documentation](https://docs.rs/singe-nvml/badge.svg)](https://docs.rs/singe-nvml)
![License](https://img.shields.io/crates/l/singe-nvml.svg)

Safe Rust wrappers for the NVIDIA Management Library (NVML), for monitoring and managing GPUs.

This crate provides:

- Driver, system, and device information.
- Memory, utilization, temperatures, clocks, and power queries.
- Device management, events, MIG instances, and vGPU APIs.
- NVML initialization and shutdown through an owned `Library`.

## Example

Print each GPU's name and available memory:

```rust
use singe_nvml::library::Library;

fn main() -> singe_nvml::error::Result<()> {
	let nvml = Library::create()?;
	for index in 0..nvml.device_count()? {
		let device = nvml.device(index)?;
		let memory = device.memory_info()?;
		println!("{}: {} MiB free", device.name()?, memory.free / 1_048_576);
	}
	Ok(())
}
```

`Library` initializes NVML and shuts it down when dropped.
Which queries and settings are available depends on the GPU and driver, and changing settings needs elevated permissions.

NVML ships with the NVIDIA driver, so monitoring does not need the CUDA Toolkit.

## More examples

- [Fan curve](examples/fan_curve.rs): control supported fans from temperature readings. This example changes device settings.
- [API reference](https://docs.rs/singe-nvml): monitoring, management, MIG, and vGPU interfaces.
