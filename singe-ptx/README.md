# singe-ptx

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-ptx.svg)](https://crates.io/crates/singe-ptx)
[![Documentation](https://docs.rs/singe-ptx/badge.svg)](https://docs.rs/singe-ptx)
![License](https://img.shields.io/crates/l/singe-ptx.svg)

A PTX parser, and a CPU emulator that runs `sm_80` PTX kernels without an NVIDIA GPU or CUDA installation.

This crate provides:

- **Parser:** `parser::parse_module` builds an owned syntax tree (`ast`) that keeps the source span of every node, and its `ParseError` names the offending bytes.
- **Kernel inspection:** `emulator::ParsedModule` gives the syntax tree and compiles an entry point into a `CompiledKernel`, which reports each parameter's name, size, and offset and checks a block shape, without starting an emulator.
- **Emulator:** `emulator::Emulator` runs kernels on a worker thread, with buffers, streams, events, and submissions that can be waited on or awaited.
- **Debugging:** errors name the faulting instruction and lane, `brkpt` breakpoints let you read a lane's registers, and `diagnostic` prints an error with the source line it points at.

## Example

Inspect a vector addition kernel, then run it on the emulator and check the output:

```rust,no_run
use singe_ptx::emulator::{Emulator, EmulatorOptions, LaunchConfig, ParsedModule, ShutdownMode};

const PTX: &str = r"
.version 9.4
.target sm_80
.address_size 64

.visible .entry add(.param .u64 a, .param .u64 b, .param .u64 out, .param .u32 n)
{
    .reg .pred %p;
    .reg .b32 %r<4>;
    .reg .b64 %rd<7>;

    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n];
    setp.ge.u32 %p, %r0, %r1;
    @%p bra done;
    ld.param.u64 %rd0, [a];
    ld.param.u64 %rd1, [b];
    ld.param.u64 %rd2, [out];
    mul.wide.u32 %rd3, %r0, 4;
    add.u64 %rd4, %rd0, %rd3;
    add.u64 %rd5, %rd1, %rd3;
    add.u64 %rd6, %rd2, %rd3;
    ld.u32 %r2, [%rd4];
    ld.u32 %r3, [%rd5];
    add.u32 %r2, %r2, %r3;
    st.u32 [%rd6], %r2;
done:
    ret;
}
";

fn main() -> anyhow::Result<()> {
    let parsed = ParsedModule::parse(PTX)?;
    let compiled = parsed.compile("add")?;
    for parameter in compiled.parameters() {
        println!(
            "{}: {} bytes at offset {}",
            parameter.name(),
            parameter.storage().byte_len(),
            parameter.byte_offset()
        );
    }
    compiled.launch_requirements().validate_block([32, 1, 1])?;

    let device = Emulator::create(EmulatorOptions::default())?;
    let module = device.load(parsed)?.wait()?;
    let kernel = module.kernel("add")?;

    let a = device.upload(&[1_u32, 2, 3, 4])?.wait()?;
    let b = device.upload(&[10_u32, 20, 30, 40])?.wait()?;
    let out = device.zeroed::<u32>(4)?.wait()?;
    let config = LaunchConfig::create([1, 1, 1], [32, 1, 1], 0)?;
    kernel
        .launch(
            config,
            [(&a).into(), (&b).into(), (&out).into(), 4_u32.into()],
        )?
        .wait()?;

    assert_eq!(device.download(&out)?.wait()?, [11, 22, 33, 44]);
    device.shutdown(ShutdownMode::Drain).wait()?;
    Ok(())
}
```

Each emulator command returns a submission; `wait()` blocks until it finishes, and every submission can be awaited instead.

## More examples

- [Vector addition](./examples/vector_add.rs): run a PTX kernel on the CPU emulator and check its output.
- [CPU emulator tour](./examples/cpu_emulator.rs): parameter inspection, streams and events, `async` submissions, and debugging a kernel that faults.

Run one with `cargo run -p singe-ptx --example cpu_emulator`.
