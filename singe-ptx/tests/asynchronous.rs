//! Deferred PTX copies and producer/consumer synchronization on `sm_80`.

use anyhow::Result;
use singe_ptx::emulator::{
    Emulator, EmulatorOptions, ExecutionError, LaunchConfig, ParsedModule, ShutdownMode,
};

mod support;

use support::Observe;

fn source(body: &str) -> String {
    format!(
        ".version 9.4 .target sm_80 .address_size 64
         .global .align 16 .u32 data[16] = {{ 42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57 }};
         .global .u32 output[32];
         .entry k() {{
             .shared .align 16 .b8 tile[512];
             .shared .align 8 .b64 ready;
             .shared .align 8 .b64 other;
             .reg .u32 r; .reg .u32 n; .reg .u32 t;
             .reg .b64 state; .reg .b64 saved;
             .reg .u64 a; .reg .u64 b; .reg .u64 c;
             .reg .pred p; .reg .pred q;
             {body}
             ret;
         }}"
    )
}

fn run(body: &str, asynchronous: bool) -> Result<Vec<u32>> {
    let device = Emulator::create(EmulatorOptions::default())?;
    let module = device
        .load(ParsedModule::parse(&source(body))?)?
        .observe(asynchronous)?;
    module
        .kernel("k")?
        .launch(LaunchConfig::create([1; 3], [1; 3], 0)?, [])?
        .observe(asynchronous)?;
    let bytes = device
        .download(&module.global("output")?)?
        .observe(asynchronous)?;
    device.shutdown(ShutdownMode::Drain).observe(asynchronous)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| u32::from_le_bytes(*bytes))
        .collect())
}

#[test]
fn test_copy_groups_defer_visibility_and_wait_only_for_the_selected_groups() -> Result<()> {
    let body = "
        st.shared.u32 [tile],10; st.shared.u32 [tile+16],11;
        st.shared.u32 [tile+32],12;
        cp.async.ca.shared.global [tile],[data],4;
        cp.async.commit_group;
        cp.async.cg.shared.global [tile+16],[data+16],16;
        cp.async.commit_group;
        ld.shared.u32 r,[tile]; st.global.u32 [output],r;
        cp.async.wait_group 1;
        ld.shared.u32 r,[tile]; st.global.u32 [output+4],r;
        ld.shared.u32 r,[tile+16]; st.global.u32 [output+8],r;
        cp.async.ca.shared.global [tile+32],[data+32],8;
        cp.async.wait_group 0;
        ld.shared.u32 r,[tile+16]; st.global.u32 [output+12],r;
        ld.shared.u32 r,[tile+32]; st.global.u32 [output+16],r;
        cp.async.wait_all;
        ld.shared.u32 r,[tile+32]; st.global.u32 [output+20],r;
    ";
    for asynchronous in [false, true] {
        assert_eq!(&run(body, asynchronous)?[..6], &[10, 42, 11, 46, 12, 50]);
        // Empty groups count among the most recent N groups, even though
        // they need no storage and are already complete.
        let empty = "cp.async.ca.shared.global [tile],[data],4;
            cp.async.commit_group; cp.async.commit_group; cp.async.commit_group;
            cp.async.wait_group 1;
            ld.shared.u32 r,[tile]; st.global.u32 [output],r;";
        assert_eq!(run(empty, asynchronous)?[0], 42);
    }
    Ok(())
}

#[test]
fn test_tiled_transforms_match_independent_host_references() -> Result<()> {
    let parsed = ParsedModule::parse(include_str!("fixtures/execution/async_tile.ptx"))?;
    for asynchronous in [false, true] {
        let device = Emulator::create(EmulatorOptions::default())?;
        let module = device.load(parsed.clone())?.observe(asynchronous)?;
        for block in [1u32, 17, 32, 33, 64, 127, 128] {
            for n in [0usize, 1, 17, 31, 32, 33, 127, 128, 129, 385, 1151] {
                let values: Vec<u32> = (0..n)
                    .map(|i| u32::try_from(i).unwrap().wrapping_mul(0x9e37_79b9))
                    .collect();
                let input = device.upload(&values)?.observe(asynchronous)?;
                let output = device.upload(&vec![0u32; n])?.observe(asynchronous)?;
                let grid = u32::try_from(n).unwrap().div_ceil(2 * block).max(1);
                module
                    .kernel("groups")?
                    .launch(
                        LaunchConfig::create([grid, 1, 1], [block, 1, 1], 0)?,
                        [
                            (&input).into(),
                            (&output).into(),
                            u32::try_from(n).unwrap().into(),
                        ],
                    )?
                    .observe(asynchronous)?;
                let expected: Vec<_> = values
                    .iter()
                    .map(|v| v.wrapping_mul(3).wrapping_add(1))
                    .collect();
                assert_eq!(
                    device.download(&output)?.observe(asynchronous)?,
                    expected,
                    "groups: n={n}, block={block}"
                );

                let grid = u32::try_from(n).unwrap().div_ceil(3 * block).max(1);
                module
                    .kernel("barrier")?
                    .launch(
                        LaunchConfig::create([grid, 1, 1], [block, 1, 1], 0)?,
                        [
                            (&input).into(),
                            (&output).into(),
                            u32::try_from(n).unwrap().into(),
                        ],
                    )?
                    .observe(asynchronous)?;
                let expected: Vec<_> = values
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        let other = i / block as usize * block as usize + (i + 1) % block as usize;
                        v.wrapping_add(values.get(other).copied().unwrap_or(0))
                    })
                    .collect();
                assert_eq!(
                    device.download(&output)?.observe(asynchronous)?,
                    expected,
                    "barrier: n={n}, block={block}"
                );
            }
        }
        device.shutdown(ShutdownMode::Drain).observe(asynchronous)?;
    }
    Ok(())
}

#[test]
fn test_copy_zero_fill_and_ignored_sources_preserve_exact_bytes() -> Result<()> {
    for asynchronous in [false, true] {
        for cache in ["ca", "cg"] {
            for shared in ["shared", "shared::cta"] {
                for input in ["", ",n", ",p"] {
                    let body = format!(
                        "mov.u32 n,16; mov.pred p,0;
                        cp.async.{cache}.{shared}.global [tile],[data],16{input};
                        cp.async.wait_all; ld.shared.u32 r,[tile]; st.global.u32 [output],r;"
                    );
                    assert_eq!(run(&body, asynchronous)?[0], 42);
                }
            }
        }
        for size in [4usize, 8, 16] {
            for count in 0..=size {
                let body =
                    format!("mov.u32 n,{count};
                    cp.async.ca.shared::cta.global [tile],[data],{size},n;
                    cp.async.wait_all;
                    ld.shared.u32 r,[tile]; st.global.u32 [output],r;
                    {}", (4..size).step_by(4).map(|offset| format!(
                        "ld.shared.u32 r,[tile+{offset}]; st.global.u32 [output+{offset}],r;"
                    )).collect::<Vec<_>>().concat());
                let result = run(&body, asynchronous)?;
                let actual: Vec<_> = result
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .take(size)
                    .collect();
                let mut expected: Vec<_> = [42u32, 43, 44, 45]
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .take(count)
                    .collect();
                expected.resize(size, 0);
                assert_eq!(actual, expected, "size={size}, count={count}");
            }
        }
        let body = "mov.pred p,1;
            cp.async.cg.shared.global [tile],[a],16,p;
            cp.async.wait_all;
            ld.shared.u32 r,[tile]; st.global.u32 [output],r;";
        // The ignored address register is indeterminate. It is never read.
        assert_eq!(run(body, asynchronous)?[0], 0);
        assert_eq!(
            run(
                "mov.pred p,0; cp.async.ca.shared.global [tile],[data],4,p;
            cp.async.wait_all; ld.shared.u32 r,[tile]; st.global.u32 [output],r;",
                asynchronous
            )?[0],
            42
        );
        // Disabled copies do not read operands or change carry state.
        assert_eq!(
            run(
                "mov.pred p,0; add.cc.u32 r,0xffffffff,1;
            @p cp.async.ca.shared.global [a],[b],4,n;
            addc.u32 r,0,0; st.global.u32 [output],r;",
                asynchronous
            )?[0],
            1
        );
    }
    Ok(())
}

#[test]
fn test_barrier_phases_tokens_invalidation_and_partial_producers() -> Result<()> {
    for asynchronous in [false, true] {
        for shared in ["", ".shared", ".shared::cta"] {
            let address = if shared.is_empty() { "a" } else { "ready" };
            for noinc in [false, true] {
                let body = format!(
                    "cvta.shared.u64 a,ready;
                    mbarrier.init{shared}.b64 [{address}],{};
                    cp.async.ca.shared.global [tile],[data],4;
                    cp.async.mbarrier.arrive{}{shared}.b64 [{address}];
                    mbarrier.arrive{shared}.b64 state,[{address}];
                    spin: mbarrier.test_wait{shared}.b64 p,[{address}],state; @!p bra spin;
                    mbarrier.test_wait.parity{shared}.b64 p,[{address}],0;
                    @p ld.shared.u32 r,[tile]; @p st.global.u32 [output],r;
                    mbarrier.inval{shared}.b64 [{address}];",
                    if noinc { 2 } else { 1 },
                    if noinc { ".noinc" } else { "" }
                );
                assert_eq!(run(&body, asynchronous)?[0], 42);
            }
        }
        let body = "mbarrier.init.shared.b64 [ready],2;
            mbarrier.test_wait.parity.shared.b64 p,[ready],0;
            selp.u32 r,1,0,p; st.global.u32 [output],r;
            mbarrier.arrive.shared.b64 state,[ready];
            mbarrier.test_wait.shared.b64 p,[ready],state;
            selp.u32 r,1,0,p; st.global.u32 [output+4],r;
            mbarrier.arrive.shared.b64 saved,[ready];
            mbarrier.test_wait.shared.b64 p,[ready],state;
            selp.u32 r,1,0,p; st.global.u32 [output+8],r;
            mbarrier.arrive.shared.b64 state,[ready];
            mbarrier.arrive.shared.b64 saved,[ready];
            mbarrier.test_wait.parity.shared.b64 p,[ready],1;
            selp.u32 r,1,0,p; st.global.u32 [output+12],r;
            mbarrier.inval.shared.b64 [ready];
            mbarrier.init.shared.b64 [ready],1;
            mbarrier.arrive.shared.b64 state,[ready];
            mbarrier.test_wait.shared.b64 p,[ready],state;
            selp.u32 r,1,0,p; st.global.u32 [output+16],r;
            mbarrier.inval.shared.b64 [ready];";
        assert_eq!(&run(body, asynchronous)?[..5], &[0, 0, 1, 1, 1]);

        for noinc in [false, true] {
            let body = source(&format!(
                "mov.u32 t,%tid.x; setp.eq.u32 p,t,0;
                @p mbarrier.init.shared.b64 [ready],{}; bar.sync 0;
                setp.lt.u32 p,t,3; mul.wide.u32 a,t,4;
                mov.u64 b,tile; add.u64 b,b,a; mov.u64 c,data; add.u64 c,c,a;
                @p cp.async.ca.shared.global [b],[c],4;
                @p cp.async.mbarrier.arrive{}.shared.b64 [ready];
                mbarrier.arrive.shared.b64 state,[ready];
                spin: mbarrier.test_wait.shared.b64 q,[ready],state; @!q bra spin;
                @p ld.shared.u32 r,[b]; mov.u64 c,output; add.u64 c,c,a;
                @p st.global.u32 [c],r; bar.sync 0;
                setp.eq.u32 p,t,0; @p mbarrier.inval.shared.b64 [ready];",
                if noinc { 20 } else { 17 },
                if noinc { ".noinc" } else { "" }
            ));
            let device = Emulator::create(EmulatorOptions::default())?;
            let module = device
                .load(ParsedModule::parse(&body)?)?
                .observe(asynchronous)?;
            module
                .kernel("k")?
                .launch(LaunchConfig::create([1; 3], [17, 1, 1], 0)?, [])?
                .observe(asynchronous)?;
            let bytes = device
                .download(&module.global("output")?)?
                .observe(asynchronous)?;
            assert_eq!(
                &bytes[..12],
                &[42u32, 43, 44]
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect::<Vec<_>>()
            );
            device.shutdown(ShutdownMode::Drain).observe(asynchronous)?;
        }
    }
    Ok(())
}

#[test]
fn test_asynchronous_forms_enforce_versions_and_reject_unimplemented_protocols() -> Result<()> {
    for (body, version) in [
        (
            "mov.pred p,1; cp.async.ca.shared.global [tile],[data],4,p;",
            "7.5",
        ),
        ("cp.async.ca.shared::cta.global [tile],[data],4;", "7.8"),
        ("mbarrier.init.shared::cta.b64 [ready],1;", "7.8"),
        ("mbarrier.arrive.shared.b64 _,[ready];", "7.1"),
        ("mbarrier.test_wait.parity.shared.b64 p,[ready],0;", "7.1"),
    ] {
        let source = source(body).replace(".version 9.4", ".version 7.0");
        let error = ParsedModule::parse(&source)?.compile("k").unwrap_err();
        assert!(error.to_string().contains(version), "{body}: {error}");
    }
    for body in [
        "cp.async.bulk.shared.global [tile],[data],16;",
        "cp.async.ca.shared.global.L2::128B [tile],[data],4;",
        "mbarrier.arrive.noComplete.shared.b64 state,[ready],2;",
        "mbarrier.arrive_drop.shared.b64 state,[ready];",
        "mbarrier.arrive.expect_tx.shared.b64 state,[ready],4;",
    ] {
        assert!(
            ParsedModule::parse(&source(body))?.compile("k").is_err(),
            "{body}"
        );
    }
    Ok(())
}

#[test]
fn test_deferred_faults_retain_the_issuing_function_after_its_lane_exits() -> Result<()> {
    let text = ".version 9.4 .target sm_80 .address_size 64
        .func helper() { .shared .align 16 .b8 tile[64]; .shared .align 8 .b64 ready;
            .reg .pred ignore; .reg .u64 unknown;
            mov.pred ignore,1; mbarrier.init.shared.b64 [ready],1;
            cp.async.cg.shared.global [tile],[unknown],16,ignore;
            cp.async.cg.shared.global [tile+16],[unknown],16,ignore;
            cp.async.cg.shared.global [tile+32],[unknown],16,ignore;
            cp.async.mbarrier.arrive.shared.b64 [ready];
            st.shared.b64 [ready],0; ret;
        }
        .entry k() {call helper; ret;}";
    for asynchronous in [false, true] {
        let device = Emulator::create(EmulatorOptions::default())?;
        let module = device
            .load(ParsedModule::parse(text)?)?
            .observe(asynchronous)?;
        let error = module
            .kernel("k")?
            .launch(LaunchConfig::create([1; 3], [1; 3], 0)?, [])?
            .observe(asynchronous)
            .unwrap_err();
        let ExecutionError::Launch { cause, report } = &error else {
            panic!("{error}");
        };
        let ExecutionError::Lane { location, cause } = cause.as_ref() else {
            panic!("{error}");
        };
        assert!(cause.to_string().contains("overwritten"), "{error}");
        assert_eq!(location.function, "helper");
        assert_eq!(location.call_stack, ["k", "helper"]);
        assert_eq!(
            &text[location.span.start..location.span.end],
            "cp.async.mbarrier.arrive.shared.b64 [ready];"
        );
        assert_eq!(report.completed_lanes, 1);
        assert!(
            device
                .shutdown(ShutdownMode::Drain)
                .observe(asynchronous)
                .is_err()
        );
    }
    Ok(())
}

#[cfg(feature = "cuda_tests")]
#[test]
fn test_tiled_fixture_assembles_for_sm80() -> Result<()> {
    use std::{
        env, fs,
        process::{self, Command},
    };
    let output = env::temp_dir().join(format!("singe-ptx-async-tile-{}.cubin", process::id()));
    let result = Command::new("ptxas")
        .args([
            "-arch=sm_80",
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/execution/async_tile.ptx"
            ),
            "-o",
        ])
        .arg(&output)
        .output()?;
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    fs::remove_file(output)?;
    Ok(())
}

#[test]
fn test_invalid_async_protocols_and_addresses_preserve_prior_writes() -> Result<()> {
    for asynchronous in [false, true] {
        for (body, expected) in [
            ("mbarrier.init.shared.b64 [ready],1; cp.async.ca.shared.global [tile],[data],4;
              cp.async.ca.shared.global [tile+4],[data+4],4;
              cp.async.mbarrier.arrive.shared.b64 [ready]; mbarrier.inval.shared.b64 [ready];", "still tracks"),
            ("mbarrier.init.shared.b64 [ready],1; cp.async.ca.shared.global [tile],[data],4;
              cp.async.ca.shared.global [tile+4],[data+4],4;
              cp.async.mbarrier.arrive.noinc.shared.b64 [ready]; mbarrier.arrive.shared.b64 state,[ready];", "reserved"),
            ("mbarrier.init.shared.b64 [ready],1048575; cp.async.mbarrier.arrive.shared.b64 [ready];", "exceeds"),
            ("mbarrier.init.shared.b64 [ready],1; mbarrier.arrive.shared.b64 saved,[ready];
              mbarrier.test_wait.shared.b64 p,[ready],saved; mbarrier.arrive.shared.b64 state,[ready];
              mbarrier.test_wait.shared.b64 p,[ready],saved;", "stale"),
            ("mbarrier.init.shared.b64 [ready],1; mbarrier.test_wait.parity.shared.b64 p,[ready],1;", "initial phase"),
            (
                "cp.async.ca.shared.global [tile],[data],4; bar.sync 0; ld.shared.u32 r,[tile];",
                "indeterminate",
            ),
            ("cp.async.ca.shared.global [tile+1],[data],4;", "misaligned"),
            ("cp.async.ca.shared.global [tile],[data+1],4;", "misaligned"),
            ("cp.async.ca.shared.global [tile+4096],[data],4;", "bounds"),
            ("cp.async.ca.shared.global [tile],[data+4096],4;", "bounds"),
            (
                "mov.u32 n,5; cp.async.ca.shared.global [tile],[data],4,n;",
                "source size",
            ),
            (
                "cp.async.ca.shared.global [tile],[data],8; cp.async.ca.shared.global [tile+4],[data],4;",
                "overlapping",
            ),
            (
                "mbarrier.arrive.shared.b64 state,[ready];",
                "not initialized",
            ),
            (
                "mbarrier.init.shared.b64 [ready],1; mbarrier.init.shared.b64 [ready],1;",
                "invalidated",
            ),
            ("mbarrier.init.shared.b64 [tile+4],1;", "misaligned"),
            (
                "mov.u32 n,0; mbarrier.init.shared.b64 [ready],n;",
                "arrival count",
            ),
            (
                "mbarrier.init.shared.b64 [ready],1; mbarrier.arrive.shared.b64 state,[ready]; mbarrier.arrive.shared.b64 state,[ready];",
                "not observed",
            ),
            (
                "mbarrier.init.shared.b64 [ready],1; mbarrier.init.shared.b64 [other],1; mbarrier.arrive.shared.b64 state,[ready]; mbarrier.test_wait.shared.b64 p,[other],state;",
                "stale",
            ),
            (
                "mbarrier.init.shared.b64 [ready],1; mbarrier.arrive.shared.b64 state,[ready]; mbarrier.inval.shared.b64 [ready]; mbarrier.init.shared.b64 [ready],1; mbarrier.test_wait.shared.b64 p,[ready],state;",
                "stale",
            ),
            (
                "mbarrier.init.shared.b64 [ready],1; st.shared.b64 [ready],0; mbarrier.arrive.shared.b64 state,[ready];",
                "overwritten",
            ),
        ] {
            let device = Emulator::create(EmulatorOptions::default())?;
            let body = source(&format!("st.global.u32 [output],99; {body}"));
            let module = device
                .load(ParsedModule::parse(&body)?)?
                .observe(asynchronous)?;
            let error = module
                .kernel("k")?
                .launch(LaunchConfig::create([1; 3], [1; 3], 0)?, [])?
                .observe(asynchronous)
                .unwrap_err();
            assert!(error.to_string().contains(expected), "{body}: {error}");
            assert!(
                matches!(
                    error.root_cause(),
                    ExecutionError::Invalid(_)
                        | ExecutionError::Memory(_)
                        | ExecutionError::Indeterminate(_)
                ),
                "{error}"
            );
            assert!(error.launch_report().is_some());
            let output = device
                .download(&module.global("output")?)?
                .observe(asynchronous)?;
            assert_eq!(&output[..4], &99u32.to_le_bytes());
            assert!(
                device
                    .shutdown(ShutdownMode::Drain)
                    .observe(asynchronous)
                    .is_err()
            );
        }
    }
    Ok(())
}
