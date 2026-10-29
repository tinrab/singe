# singe-ptx

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-ptx.svg)](https://crates.io/crates/singe-ptx)
[![Documentation](https://docs.rs/singe-ptx/badge.svg)](https://docs.rs/singe-ptx)
![License](https://img.shields.io/crates/l/singe-ptx.svg)

A PTX parser and a CPU emulator that runs PTX kernels without an NVIDIA GPU.

This crate provides:

- A PTX parser that builds a syntax tree and records where each node is in the source.
- Kernel validation, and inspection of a kernel's parameters and launch requirements.
- A CPU emulator with buffers, streams, events, and blocking or `async` submissions.
- Errors that name the faulting instruction and lane, and `brkpt` breakpoints that let you read a lane's registers.

## Example

Run a vector addition kernel on the CPU emulator.
`PTX` holds the kernel source; the [complete example](examples/vector_add.rs) includes it:

```rust,ignore
use singe_ptx::emulator::{Emulator, EmulatorOptions, LaunchConfig, ParsedModule, ShutdownMode};

let device = Emulator::create(EmulatorOptions::default())?;
let module = device.load(ParsedModule::parse(PTX)?)?.wait()?;
let kernel = module.kernel("add")?;

let a = device.upload(&[1u32, 2, 3, 4])?.wait()?;
let b = device.upload(&[10u32, 20, 30, 40])?.wait()?;
let out = device.zeroed::<u32>(4)?.wait()?;
let config = LaunchConfig::create([1, 1, 1], [32, 1, 1], 0)?;
kernel
	.launch(config, [(&a).into(), (&b).into(), (&out).into(), 4u32.into()])?
	.wait()?;

assert_eq!(device.download(&out)?.wait()?, [11, 22, 33, 44]);
device.shutdown(ShutdownMode::Drain).wait()?;
```

Every submission can also be awaited instead of calling `wait()`.

## Parsing PTX

Parsing and compilation do not start an emulator:

```rust,ignore
use singe_ptx::emulator::ParsedModule;

let parsed = ParsedModule::parse(PTX)?;
println!("{} top-level items", parsed.syntax().items.len());
let kernel = parsed.compile("add")?;
for parameter in kernel.parameters() {
	println!("{}: {} bytes at offset {}", parameter.name(), parameter.storage().byte_len(), parameter.byte_offset());
}
kernel.launch_requirements().validate_block([128, 1, 1])?;
```

## More examples

- [CPU emulator tour](examples/cpu_emulator.rs): parameter inspection, streams and events, `async` submissions, and debugging a kernel that faults.

See the [API reference](https://docs.rs/singe-ptx).
