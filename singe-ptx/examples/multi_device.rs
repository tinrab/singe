//! Run kernels on two emulated devices that read each other's memory, share a mapped range, and reduce through a multicast object.
//! Then fault one device and keep using the other.

use std::task::Poll;

use anyhow::{Result, anyhow};
use singe_ptx::emulator::{
    Emulator, EmulatorOptions, ExecutionError, LaunchConfig, ParsedModule, ShutdownMode,
    SubmitError, Topology,
};

const PAGE: u64 = 1 << 21;

const ADD_PEER: &str = r"
.version 9.4
.target sm_80
.address_size 64

.visible .entry add_peer(.param .u64 local, .param .u64 peer)
{
    .reg .b32 %r<3>;
    .reg .b64 %rd<5>;

    ld.param.u64 %rd0, [local];
    ld.param.u64 %rd1, [peer];
    mov.u32 %r0, %tid.x;
    mul.wide.u32 %rd2, %r0, 4;
    add.u64 %rd3, %rd0, %rd2;
    add.u64 %rd4, %rd1, %rd2;
    ld.global.u32 %r1, [%rd3];
    ld.global.u32 %r2, [%rd4];
    add.u32 %r1, %r1, %r2;
    st.global.u32 [%rd3], %r1;
    ret;
}
";

const TRAP: &str = r"
.version 9.4
.target sm_80
.address_size 64

.visible .entry trap()
{
    trap;
}
";

const COUNT: &str = r"
.version 9.4
.target sm_90
.address_size 64

.visible .entry count(.param .u64 counter)
{
    .reg .b64 %rd0;

    ld.param.u64 %rd0, [counter];
    multimem.red.relaxed.sys.global.add.u32 [%rd0], 1;
    ret;
}
";

fn main() -> Result<()> {
    let emulator = Emulator::create(EmulatorOptions {
        topology: Topology::node(2),
        ..Default::default()
    })?;
    let [first, second] =
        <[_; 2]>::try_from(emulator.devices()).map_err(|_| anyhow!("expected two devices"))?;

    // Device 0 adds device 1's buffer into its own once peer access is enabled.
    let local = first.upload(&[1u32, 2, 3, 4])?.wait()?;
    let peer = second.upload(&[10u32, 20, 30, 40])?.wait()?;
    first.enable_peer_access(&second)?.wait()?;
    let module = first.load(ParsedModule::parse(ADD_PEER)?)?.wait()?;
    let config = LaunchConfig::create([1, 1, 1], [4, 1, 1], 0)?;
    let single = LaunchConfig::create([1, 1, 1], [1, 1, 1], 0)?;
    module
        .kernel("add_peer")?
        .launch(config, [(&local).into(), (&peer).into()])?
        .wait()?;
    let sums = first.download(&local)?.wait()?;
    assert_eq!(sums, [11, 22, 33, 44]);
    println!("peer sums: {sums:?}");

    // One reserved range maps a page of each device's physical memory back to back, as NCCL's symmetric windows do.
    let (flat, counter, copies) = emulator
        .host_task(move |memory| {
            let flat = memory.reserve(2 * PAGE, PAGE)?;
            for device in 0..2 {
                let physical = memory.allocate_physical(device, PAGE as usize)?;
                let address = flat + device as u64 * PAGE;
                memory.map(address, PAGE, physical, 0)?;
                memory.set_access(address, PAGE, 0, true)?;
                memory.set_access(address, PAGE, 1, true)?;
            }

            for (index, value) in [5u32, 6, 7, 8].into_iter().enumerate() {
                memory.write_value(flat + 4 * index as u64, value)?;
            }

            let multicast = memory.create_multicast(&[0, 1], PAGE)?;
            let mut copies = Vec::new();
            for device in 0..2 {
                let physical = memory.allocate_physical(device, PAGE as usize)?;
                memory.bind_multicast(multicast, 0, device, physical, 0, PAGE)?;
                let address = memory.reserve(PAGE, PAGE)?;
                memory.map(address, PAGE, physical, 0)?;
                copies.push(address);
            }
            Poll::Ready(Ok((flat, memory.map_multicast(multicast)?, copies)))
        })?
        .wait()?;

    // Device 1 adds device 0's page into its own page through the flat range.
    let module = second.load(ParsedModule::parse(ADD_PEER)?)?.wait()?;
    module
        .kernel("add_peer")?
        .launch(config, [(flat + PAGE).into(), flat.into()])?
        .wait()?;
    let mapped = emulator
        .host_task(move |memory| {
            Poll::Ready(
                (0..4)
                    .map(|index| memory.read_value::<u32>(flat + PAGE + 4 * index))
                    .collect::<Result<Vec<_>, _>>(),
            )
        })?
        .wait()?;
    assert_eq!(mapped, [5, 6, 7, 8]);
    println!("mapped page of device 1: {mapped:?}");

    // Each device adds one through the multicast address, so every device's copy counts both.
    for device in [&first, &second] {
        let module = device.load(ParsedModule::parse(COUNT)?)?.wait()?;
        module
            .kernel("count")?
            .launch(single, [counter.into()])?
            .wait()?;
    }

    let counts = emulator
        .host_task(move |memory| {
            Poll::Ready(
                copies
                    .iter()
                    .map(|&copy| memory.read_value::<u32>(copy))
                    .collect::<Result<Vec<_>, _>>(),
            )
        })?
        .wait()?;
    assert_eq!(counts, [2, 2]);
    println!("multicast counts per device: {counts:?}");

    // A trap faults only device 0, as a sticky error poisons only its CUDA context.
    let trap = first.load(ParsedModule::parse(TRAP)?)?.wait()?;
    let Err(fault) = trap.kernel("trap")?.launch(single, [])?.wait() else {
        return Err(anyhow!("expected the trap kernel to fault"));
    };
    assert!(matches!(fault.root_cause(), ExecutionError::Trap));
    assert!(matches!(first.upload(&[1u32]), Err(SubmitError::Faulted)));
    assert!(second.fault().is_none());
    let module = second.load(ParsedModule::parse(ADD_PEER)?)?.wait()?;
    let local = second.upload(&[1u32, 2, 3, 4])?.wait()?;
    module
        .kernel("add_peer")?
        .launch(config, [(&local).into(), (&local).into()])?
        .wait()?;
    let doubled = second.download(&local)?.wait()?;
    assert_eq!(doubled, [2, 4, 6, 8]);
    println!("device 1 still runs after device 0 faulted: {doubled:?}");

    let shutdown = emulator.shutdown(ShutdownMode::Drain).wait();
    assert!(matches!(shutdown, Err(error) if matches!(error.root_cause(), ExecutionError::Trap)));
    Ok(())
}
