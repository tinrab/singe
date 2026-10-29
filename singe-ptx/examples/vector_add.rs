//! Run a PTX kernel on the CPU emulator and check its output.

use std::error::Error;

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

fn main() -> Result<(), Box<dyn Error>> {
    let device = Emulator::create(EmulatorOptions::default())?;
    let module = device.load(ParsedModule::parse(PTX)?)?.wait()?;
    let kernel = module.kernel("add")?;

    let a = device.upload(&[1u32, 2, 3, 4])?.wait()?;
    let b = device.upload(&[10u32, 20, 30, 40])?.wait()?;
    let out = device.zeroed::<u32>(4)?.wait()?;
    let config = LaunchConfig::create([1, 1, 1], [32, 1, 1], 0)?;
    kernel
        .launch(
            config,
            [(&a).into(), (&b).into(), (&out).into(), 4u32.into()],
        )?
        .wait()?;

    let result = device.download(&out)?.wait()?;
    assert_eq!(result, [11, 22, 33, 44]);
    println!("{result:?}");
    device.shutdown(ShutdownMode::Drain).wait()?;
    Ok(())
}
