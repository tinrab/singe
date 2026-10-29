//! Read-only observations through the breakpoint control path.

use anyhow::Result;
use singe_ptx::emulator::{
    Emulator, EmulatorOptions, ExecutionError, FrameSnapshot, LaneSnapshot, LaunchConfig,
    ParsedModule, RegisterSnapshot, ShutdownMode, SubmitError,
};

mod support;

use support::Observe;

const NESTED: &str = ".version 9.4 .target sm_80 .address_size 64
    .global .u32 output[2];
    .func helper() {
        .reg .u32 value; .reg .u32 carry; .reg .u32 shadow;
        .reg .b32 packed<2>; .reg .u64 address; .reg .u64 offset;
        mov.u32 value,%tid.x; add.u32 value,value,11;
        mov.u32 shadow,5; mov.b32 packed0,value;
        add.cc.u32 carry,0xffffffff,1;
        { .reg .u32 shadow; mov.u32 shadow,9; brkpt; }
        add.u32 value,value,100;
        mov.u32 carry,%tid.x; mul.wide.u32 offset,carry,4;
        mov.u64 address,output; add.u64 address,address,offset;
        st.global.u32 [address],value; ret;
    }
    .entry k() { .reg .u32 outer; mov.u32 outer,7; call helper; ret; }";

fn register<'a>(snapshot: &'a LaneSnapshot, function: &str, name: &str) -> &'a RegisterSnapshot {
    snapshot
        .frames()
        .iter()
        .find(|frame| frame.function() == function)
        .unwrap()
        .registers()
        .iter()
        .find(|register| register.name() == name)
        .unwrap()
}

#[test]
fn test_nested_lane_snapshots_survive_resume_and_shutdown_and_bypass_a_full_queue() -> Result<()> {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<LaneSnapshot>();
    for asynchronous in [false, true] {
        let device = Emulator::create(EmulatorOptions {
            queued_commands: 1,
            ..Default::default()
        })?;
        let module = device
            .load(ParsedModule::parse(NESTED)?)?
            .observe(asynchronous)?;
        let compute = device.create_stream()?;
        let launch = module.kernel("k")?.launch_on(
            &compute,
            LaunchConfig::create([1; 3], [2, 1, 1], 0)?,
            [],
        )?;
        let breakpoint = device.next_breakpoint()?.observe(asynchronous)?;
        let queued = compute.synchronize()?;
        assert!(matches!(compute.synchronize(), Err(SubmitError::QueueFull)));
        let first = device
            .inspect_lane(&breakpoint.id, [0; 3], [1, 0, 0])?
            .observe(asynchronous)?;
        assert_eq!(first.cta(), [0; 3]);
        assert_eq!(first.lane(), [1, 0, 0]);
        assert_eq!(first.carry(), Some(true));
        assert!(!first.is_exited());
        assert_eq!(
            first
                .frames()
                .iter()
                .map(FrameSnapshot::function)
                .collect::<Vec<_>>(),
            ["k", "helper"]
        );
        assert_eq!(register(&first, "k", "outer").value(), Some(7));
        assert_eq!(register(&first, "helper", "value").value(), Some(12));
        assert_eq!(register(&first, "helper", "packed0").value(), Some(12));
        assert_eq!(register(&first, "helper", "packed1").value(), None);
        let shadows: Vec<_> = first.frames()[1]
            .registers()
            .iter()
            .filter(|register| register.name() == "shadow")
            .collect();
        assert_eq!(
            shadows
                .iter()
                .map(|register| register.value())
                .collect::<Vec<_>>(),
            [Some(5), Some(9)]
        );
        assert_ne!(shadows[0].declaration(), shadows[1].declaration());
        let span = first.frames()[1].span().unwrap();
        assert!(NESTED[span.start..span.end].contains("add.u32 value"));
        let error = device
            .inspect_lane(&breakpoint.id, [1, 0, 0], [0; 3])?
            .observe(asynchronous)
            .unwrap_err();
        assert!(matches!(error, ExecutionError::Invalid(_)));
        let foreign = Emulator::create(EmulatorOptions::default())?;
        assert!(matches!(
            foreign.inspect_lane(&breakpoint.id, [0; 3], [0; 3]),
            Err(SubmitError::ForeignResource)
        ));
        foreign
            .shutdown(ShutdownMode::Drain)
            .observe(asynchronous)?;
        let shutdown = device.shutdown(ShutdownMode::Drain);
        let second = device
            .inspect_lane(&breakpoint.id, [0; 3], [0; 3])?
            .observe(asynchronous)?;
        assert_eq!(register(&second, "helper", "value").value(), Some(11));
        device.resume(&breakpoint.id)?.observe(asynchronous)?;
        // The draining shutdown may close the device as soon as the launch resumes.
        assert!(matches!(
            device.inspect_lane(&breakpoint.id, [0; 3], [0; 3]),
            Err(SubmitError::InvalidArgument(_) | SubmitError::Closed)
        ));
        assert_eq!(launch.observe(asynchronous)?.completed_lanes, 2);
        queued.observe(asynchronous)?;
        shutdown.observe(asynchronous)?;
        assert_eq!(register(&first, "helper", "value").value(), Some(12));
        assert_eq!(register(&second, "helper", "value").value(), Some(11));
    }
    Ok(())
}

#[test]
fn test_inspection_selects_multidimensional_coordinates_and_rejects_invalid_lanes() -> Result<()> {
    let parsed = ParsedModule::parse(
        ".version 9.4 .target sm_80 .address_size 64
         .global .u32 ready=0;
         .entry k() {
             .reg .u32 value; .reg .u32 t; .reg .u32 x; .reg .u32 old; .reg .pred waiting;
             mov.u32 value,%ctaid.y; mov.u32 x,%ctaid.x; mad.lo.u32 value,value,2,x;
             mul.lo.u32 value,value,4; mov.u32 t,%tid.y; mov.u32 x,%tid.x;
             mad.lo.u32 t,t,2,x; add.u32 value,value,t;
             atom.global.add.u32 old,[ready],1;
             again: ld.global.u32 old,[ready]; setp.ne.u32 waiting,old,16; @waiting bra again;
             brkpt; ret;
         }",
    )?;
    for asynchronous in [false, true] {
        let device = Emulator::create(EmulatorOptions::default())?;
        let module = device.load(parsed.clone())?.observe(asynchronous)?;
        let launch = module
            .kernel("k")?
            .launch(LaunchConfig::create([2, 2, 1], [2, 2, 1], 0)?, [])?;
        for pause in 0..4 {
            let breakpoint = device.next_breakpoint()?.observe(asynchronous)?;
            if pause == 0 {
                for (cta, lane, expected) in [
                    ([0, 0, 0], [1, 0, 0], 1),
                    ([1, 1, 0], [1, 1, 0], 15),
                    ([0, 1, 0], [0, 1, 0], 10),
                ] {
                    let snapshot = device
                        .inspect_lane(&breakpoint.id, cta, lane)?
                        .observe(asynchronous)?;
                    assert_eq!(snapshot.cta(), cta);
                    assert_eq!(snapshot.lane(), lane);
                    assert_eq!(register(&snapshot, "k", "value").value(), Some(expected));
                }
                let error = device
                    .inspect_lane(&breakpoint.id, [0; 3], [0, 2, 0])?
                    .observe(asynchronous)
                    .unwrap_err();
                assert!(matches!(error, ExecutionError::Invalid(_)));
            }
            device.resume(&breakpoint.id)?.observe(asynchronous)?;
        }
        assert_eq!(launch.observe(asynchronous)?.completed_lanes, 16);
        device.shutdown(ShutdownMode::Drain).observe(asynchronous)?;
    }
    Ok(())
}

#[test]
fn test_exited_lanes_have_no_retained_frames() -> Result<()> {
    let parsed = ParsedModule::parse(
        ".version 9.4 .target sm_80 .address_size 64
         .entry k() { .reg .u32 t; .reg .pred p; mov.u32 t,%tid.x;
         setp.eq.u32 p,t,0; @p ret; brkpt; ret; }",
    )?;
    let device = Emulator::create(EmulatorOptions::default())?;
    let module = device.load(parsed)?.wait()?;
    let launch = module
        .kernel("k")?
        .launch(LaunchConfig::create([1; 3], [2, 1, 1], 0)?, [])?;
    let breakpoint = device.next_breakpoint()?.wait()?;
    assert!(
        device
            .inspect_lane(&breakpoint.id, [0; 3], [0; 3])?
            .wait()?
            .is_exited()
    );
    assert!(
        !device
            .inspect_lane(&breakpoint.id, [0; 3], [1, 0, 0])?
            .wait()?
            .is_exited()
    );
    device.resume(&breakpoint.id)?.wait()?;
    launch.wait()?;
    device.shutdown(ShutdownMode::Drain).wait()?;
    Ok(())
}
