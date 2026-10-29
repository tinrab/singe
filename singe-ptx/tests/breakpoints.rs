mod support;

use std::slice;

use anyhow::Result;
use singe_ptx::emulator::{
    Emulator, EmulatorOptions, ExecutionError, LaunchConfig, ParsedModule, ShutdownMode,
    SubmitError,
};
use support::Observe;

const SOURCE: &str = ".version 9.3 .target sm_80 .address_size 64
    .global .u32 value; .global .u32 other;
    .func helper() { brkpt; ret; }
    .entry k() { st.global.u32 [value],1; call helper;
        st.global.u32 [value],2; brkpt; st.global.u32 [value],3; ret; }
    .entry independent() { st.global.u32 [other],7; ret; }";

#[test]
fn test_breakpoints_preserve_frames_and_writes_while_other_streams_progress() -> Result<()> {
    for asynchronous in [false, true] {
        let device = Emulator::create(EmulatorOptions {
            queued_commands: 1,
            ..Default::default()
        })?;
        let module = device
            .load(ParsedModule::parse(SOURCE)?)?
            .observe(asynchronous)?;
        let compute = device.create_stream()?;
        let diagnostic = device.create_stream()?;
        let config = LaunchConfig::create([1; 3], [1; 3], 0)?;
        let first = device.next_breakpoint()?;
        let launch = compute.launch(&module.kernel("k")?, config, [])?;
        let id = launch.launch_id().unwrap();
        let first = first.observe(asynchronous)?;
        assert_eq!(first.location.launch_id, id);
        assert_eq!(first.location.function, "helper");
        assert_eq!(first.location.call_stack, ["k", "helper"]);
        assert_eq!(
            diagnostic
                .download(&module.global("value")?)?
                .observe(asynchronous)?,
            1u32.to_le_bytes()
        );
        diagnostic
            .launch(&module.kernel("independent")?, config, [])?
            .observe(asynchronous)?;
        assert_eq!(
            diagnostic
                .download(&module.global("other")?)?
                .observe(asynchronous)?,
            7u32.to_le_bytes()
        );
        let fence = compute.synchronize()?;
        assert!(matches!(compute.synchronize(), Err(SubmitError::QueueFull)));
        let second = device.next_breakpoint()?;
        device.resume(&first.id)?.observe(asynchronous)?;
        let second = second.observe(asynchronous)?;
        assert_ne!(first.id, second.id);
        assert_eq!(second.location.function, "k");
        assert!(matches!(
            device.resume(&first.id),
            Err(SubmitError::InvalidArgument(_))
        ));
        let foreign = Emulator::create(EmulatorOptions::default())?;
        assert!(matches!(
            foreign.resume(&second.id),
            Err(SubmitError::ForeignResource)
        ));
        foreign
            .shutdown(ShutdownMode::Drain)
            .observe(asynchronous)?;
        device.resume(&second.id)?.observe(asynchronous)?;
        let report = launch.observe(asynchronous)?;
        assert_eq!(report.completed_lanes, 1);
        fence.observe(asynchronous)?;
        assert_eq!(
            diagnostic
                .download(&module.global("value")?)?
                .observe(asynchronous)?,
            3u32.to_le_bytes()
        );
        device.shutdown(ShutdownMode::Drain).observe(asynchronous)?;
    }
    Ok(())
}

#[test]
fn test_drain_keeps_breakpoints_resumable_and_cancel_terminates_observers() -> Result<()> {
    for mode in [ShutdownMode::Drain, ShutdownMode::Cancel] {
        let device = Emulator::create(EmulatorOptions::default())?;
        let module = device
            .load(ParsedModule::parse(
                ".version 9.3 .target sm_80 .address_size 64
            .entry k() {brkpt;ret;}",
            )?)?
            .wait()?;
        let breakpoint = device.next_breakpoint()?;
        let launch = module
            .kernel("k")?
            .launch(LaunchConfig::create([1; 3], [1; 3], 0)?, [])?;
        let breakpoint = breakpoint.wait()?;
        let unused = device.next_breakpoint()?;
        let shutdown = device.shutdown(mode);
        if mode == ShutdownMode::Drain {
            device.resume(&breakpoint.id)?.wait()?;
            launch.wait()?;
        } else {
            assert!(matches!(
                launch.wait().unwrap_err().root_cause(),
                ExecutionError::Cancelled
            ));
        }
        assert!(matches!(unused.wait(), Err(ExecutionError::Cancelled)));
        shutdown.wait()?;
        assert!(device.paused_breakpoints().is_empty());
    }
    Ok(())
}

#[test]
fn test_abandoned_notifications_leave_a_recoverable_suspension() -> Result<()> {
    let device = Emulator::create(EmulatorOptions::default())?;
    let module = device
        .load(ParsedModule::parse(
            ".version 9.3 .target sm_80 .address_size 64
        .entry k() {brkpt;ret;}",
        )?)?
        .wait()?;
    drop(device.next_breakpoint()?);
    let launch = module
        .kernel("k")?
        .launch(LaunchConfig::create([1; 3], [32, 1, 1], 0)?, [])?;
    let breakpoint = device.next_breakpoint()?.wait()?;
    let id = breakpoint.id.clone();
    drop(breakpoint);
    assert_eq!(device.paused_breakpoints(), slice::from_ref(&id));
    device.resume(&id)?.wait()?;
    assert_eq!(launch.wait()?.completed_lanes, 32);
    device.shutdown(ShutdownMode::Drain).wait()?;
    Ok(())
}

#[test]
fn test_last_public_owner_cancels_a_paused_launch() -> Result<()> {
    let device = Emulator::create(EmulatorOptions::default())?;
    let module = device
        .load(ParsedModule::parse(
            ".version 9.3 .target sm_80 .address_size 64
        .entry k() {brkpt;ret;}",
        )?)?
        .wait()?;
    let launch = module
        .kernel("k")?
        .launch(LaunchConfig::create([1; 3], [1; 3], 0)?, [])?;
    let breakpoint = device.next_breakpoint()?.wait()?;
    let shutdown = device.shutdown(ShutdownMode::Drain);
    drop(launch);
    drop(module);
    drop(device);
    shutdown.wait()?;
    drop(breakpoint);
    Ok(())
}

#[test]
fn test_another_grid_fault_terminates_paused_launches_and_debug_observers() -> Result<()> {
    let device = Emulator::create(EmulatorOptions::default())?;
    let module = device
        .load(ParsedModule::parse(
            ".version 9.3 .target sm_80 .address_size 64
        .global .u32 value;
        .entry paused() {st.global.u32 [value],13; brkpt; ret;}
        .entry fault() {trap;}",
        )?)?
        .wait()?;
    let stream = device.create_stream()?;
    let config = LaunchConfig::create([1; 3], [1; 3], 0)?;
    let paused = module.kernel("paused")?.launch(config, [])?;
    let breakpoint = device.next_breakpoint()?.wait()?;
    let observer = device.next_breakpoint()?;
    let fault = stream
        .launch(&module.kernel("fault")?, config, [])?
        .wait()
        .unwrap_err();
    assert!(matches!(fault.root_cause(), ExecutionError::Trap));
    assert!(matches!(
        paused.wait().unwrap_err().root_cause(),
        ExecutionError::Faulted
    ));
    assert!(matches!(observer.wait(), Err(ExecutionError::Faulted)));
    assert!(matches!(
        device.resume(&breakpoint.id),
        Err(SubmitError::Faulted)
    ));
    assert!(matches!(
        device.next_breakpoint(),
        Err(SubmitError::Faulted)
    ));
    assert_eq!(
        device.download(&module.global("value")?)?.wait()?,
        13u32.to_le_bytes()
    );
    assert!(device.shutdown(ShutdownMode::Drain).wait().is_err());
    Ok(())
}
