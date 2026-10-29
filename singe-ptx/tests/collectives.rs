use singe_ptx::emulator::{Emulator, EmulatorOptions, LaunchConfig, ParsedModule, ShutdownMode};

fn run(body: &str) -> Vec<u32> {
    run_cta(body, 32)
}

fn run_cta(body: &str, threads: u32) -> Vec<u32> {
    let source = format!(
        ".version 9.3 .target sm_80 .address_size 64 .global .u32 output[{threads}]; .entry k() {{ .reg .u32 t; .reg .u32 r; .reg .u32 b; .reg .u64 a; .reg .u64 base; .reg .pred p; .reg .pred q; mov.u32 t,%tid.x; {body} mul.wide.u32 a,t,4; mov.u64 base,output; add.u64 a,a,base; st.global.u32 [a],r; ret; }}"
    );
    let device = Emulator::create(EmulatorOptions::default()).unwrap();
    let module = device
        .load(ParsedModule::parse(&source).unwrap())
        .unwrap()
        .wait()
        .unwrap();
    module
        .kernel("k")
        .unwrap()
        .launch(
            LaunchConfig::create([1, 1, 1], [threads, 1, 1], 0).unwrap(),
            [],
        )
        .unwrap()
        .wait()
        .unwrap();
    let output = device
        .download(&module.global("output").unwrap())
        .unwrap()
        .wait()
        .unwrap();
    device.shutdown(ShutdownMode::Drain).wait().unwrap();
    output
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| u32::from_le_bytes(*v))
        .collect()
}

#[test]
fn test_shuffle_snapshots_and_clamp_predicates() {
    for (mode, control, delta) in [
        ("up", 0, 1),
        ("down", 31, 1),
        ("bfly", 31, 1),
        ("idx", 0x181f, 3),
        ("up", 8, 1),
    ] {
        let actual = run(&format!(
            "shfl.sync.{mode}.b32 r|p,t,{delta},{control},0xffffffff; selp.u32 b,0x80000000,0,p; or.b32 r,r,b;"
        ));
        let expected: Vec<_> = (0u32..32)
            .map(|lane| {
                let (source, valid) = match mode {
                    "up" => (lane.saturating_sub(1), lane > control),
                    "down" => (lane + 1, lane < 31),
                    "bfly" => (lane ^ 1, true),
                    "idx" => ((lane / 8) * 8 + 3, true),
                    _ => unreachable!(),
                };
                (if valid { source } else { lane }) | (if valid { 1 << 31 } else { 0 })
            })
            .collect();
        assert_eq!(actual, expected, "{mode}, clamp {control}");
    }
}

#[test]
fn test_votes_and_predicated_active_masks() {
    for (mode, expected) in [("all", 0), ("any", 1), ("uni", 0)] {
        assert_eq!(
            run(&format!(
                "and.b32 b,t,1; setp.ne.u32 p,b,0; vote.sync.{mode}.pred q,p,0xffffffff; selp.u32 r,1,0,q;"
            )),
            [expected; 32]
        );
    }
    assert_eq!(
        run("and.b32 b,t,1; setp.ne.u32 p,b,0; vote.sync.ballot.b32 r,p,0xffffffff;"),
        [0xaaaa_aaaa; 32]
    );
    let masks = run("setp.lt.u32 p,t,16; mov.u32 r,0; @p activemask.b32 r;");
    assert_eq!(&masks[..16], [0xffff; 16]);
    assert_eq!(&masks[16..], [0; 16]);
}

#[test]
fn test_cta_barriers_preserve_partial_warps_and_repeated_generations() {
    for threads in [1, 17, 33, 1024] {
        assert_eq!(
            run_cta(
                "mov.u32 b,0; repeat: bar.sync 0; add.u32 b,b,1; setp.lt.u32 p,b,4; @p bra repeat; mov.u32 r,b;",
                threads
            ),
            vec![4; threads as usize]
        );
    }
    for (threads, count) in [(17, 32), (33, 64), (64, 32)] {
        assert_eq!(
            run_cta(&format!("bar.sync 0,{count}; mov.u32 r,9;"), threads),
            vec![9; threads as usize]
        );
    }
}

#[test]
fn test_cta_barriers_preserve_exited_lanes_and_divergent_arrival() {
    let output = run_cta("setp.lt.u32 p,t,32; @p ret; bar.sync 0; mov.u32 r,7;", 64);
    assert_eq!(&output[..32], [0; 32]);
    assert_eq!(&output[32..], [7; 32]);
    let output = run(
        "setp.lt.u32 p,t,16; @p bra left; mov.u32 r,2; bar.sync 0; bra done; left: mov.u32 r,1; bar.sync 0; done:",
    );
    assert_eq!(&output[..16], [1; 16]);
    assert_eq!(&output[16..], [2; 16]);
}

#[test]
fn test_cta_barrier_counts_remain_diagnostic() {
    use singe_ptx::emulator::ExecutionError;

    for (body, inconsistent) in [
        (
            "mov.u32 t,%tid.x; setp.lt.u32 p,t,16; selp.u32 n,32,64,p; bar.sync 0,n;",
            true,
        ),
        (
            "mov.u32 t,%tid.x; setp.lt.u32 p,t,32; @p ret; bar.sync 0,64;",
            false,
        ),
    ] {
        let device = Emulator::create(EmulatorOptions::default()).unwrap();
        let source = format!(
            ".version 9.3 .target sm_80 .address_size 64 .entry k() {{ .reg .u32 t; .reg .u32 n; .reg .pred p; {body} ret; }}"
        );
        let module = device
            .load(ParsedModule::parse(&source).unwrap())
            .unwrap()
            .wait()
            .unwrap();
        let error = module
            .kernel("k")
            .unwrap()
            .launch(LaunchConfig::create([1; 3], [64, 1, 1], 0).unwrap(), [])
            .unwrap()
            .wait()
            .unwrap_err();
        if inconsistent {
            assert!(
                matches!(error.root_cause(), ExecutionError::Invalid(message) if message == "inconsistent barrier counts")
            );
        } else {
            assert!(matches!(error.root_cause(), ExecutionError::Stall(_)));
        }
        assert!(device.shutdown(ShutdownMode::Drain).wait().is_err());
    }
}

#[test]
fn test_cancellation_releases_a_launch_with_cta_waiters() {
    use singe_ptx::emulator::ExecutionError;

    let device = Emulator::create(EmulatorOptions::default()).unwrap();
    let module = device.load(ParsedModule::parse(".version 9.3 .target sm_80 .address_size 64 .entry k() { .reg .u32 t; .reg .pred p; mov.u32 t,%tid.x; setp.lt.u32 p,t,32; @p bra waiting; brkpt; ret; waiting: bar.sync 0; ret; }").unwrap()).unwrap().wait().unwrap();
    let launch = module
        .kernel("k")
        .unwrap()
        .launch(LaunchConfig::create([1; 3], [64, 1, 1], 0).unwrap(), [])
        .unwrap();
    let breakpoint = device.next_breakpoint().unwrap().wait().unwrap();
    assert_eq!(breakpoint.location.lane[0], 32);
    let shutdown = device.shutdown(ShutdownMode::Cancel);
    assert!(matches!(
        launch.wait().unwrap_err().root_cause(),
        ExecutionError::Cancelled
    ));
    shutdown.wait().unwrap();
}
