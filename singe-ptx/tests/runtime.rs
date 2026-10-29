use std::{error::Error, result::Result as StdResult};

use singe_ptx::emulator::{
    Emulator, EmulatorOptions, ExecutionError, KernelArg, LaunchConfig, ParsedModule, ShutdownMode,
    SubmitError,
};

mod support;

use support::{Observe, block_on};

type Result = StdResult<(), Box<dyn Error>>;

const PREFIX: &str = ".version 9.3\n.target sm_80\n.address_size 64\n";

fn parse(source: &str) -> ParsedModule {
    ParsedModule::parse(&format!("{PREFIX}{source}")).unwrap()
}

fn one() -> LaunchConfig {
    LaunchConfig::create([1, 1, 1], [1, 1, 1], 0).unwrap()
}

#[test]
fn test_returning_from_a_noreturn_function_reports_the_return_instruction() -> Result {
    for asynchronous in [false, true] {
        for continuation in ["", "st.global.u32 [output],99; ret;"] {
            let text = format!(
                ".version 9.4 .target sm_80 .address_size 64 .global .u32 output;
                .func stop() .noreturn {{ st.global.u32 [output],42; ret; }}
                .entry k() {{ call stop; {continuation} }}"
            );
            let device = Emulator::create(EmulatorOptions::default())?;
            let module = device
                .load(ParsedModule::parse(&text)?)?
                .observe(asynchronous)?;
            let error = module
                .kernel("k")?
                .launch(LaunchConfig::create([1; 3], [1; 3], 0)?, [])?
                .observe(asynchronous)
                .unwrap_err();
            assert!(
                matches!(error.root_cause(),ExecutionError::Invalid(reason) if reason.contains("noreturn")),
                "{error}"
            );
            let ExecutionError::Launch { cause, .. } = &error else {
                panic!("{error}");
            };
            let ExecutionError::Lane { location, .. } = cause.as_ref() else {
                panic!("{error}");
            };
            assert_eq!(location.function, "stop");
            assert_eq!(&text[location.span.start..location.span.end], "ret;");
            let output = device
                .download(&module.global("output")?)?
                .observe(asynchronous)?;
            assert_eq!(output, 42u32.to_le_bytes());
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

#[test]
fn test_noreturn_functions_can_exit_and_skip_predicated_returns() -> Result {
    for asynchronous in [false, true] {
        let device = Emulator::create(EmulatorOptions::default())?;
        let module = device
            .load(ParsedModule::parse(
                ".version 9.4 .target sm_80 .address_size 64
            .func stop() .noreturn { .reg .pred p; mov.pred p,0; @p ret; exit; }
            .entry k() { call stop; }",
            )?)?
            .observe(asynchronous)?;
        let report = module
            .kernel("k")?
            .launch(LaunchConfig::create([1; 3], [1; 3], 0)?, [])?
            .observe(asynchronous)?;
        assert_eq!(report.completed_lanes, 1);
        device.shutdown(ShutdownMode::Drain).observe(asynchronous)?;
    }
    Ok(())
}

#[test]
fn test_transfers_views_async_and_foreign_handles() -> Result {
    block_on(async {
        let device = Emulator::create(EmulatorOptions::default())?;
        let mut input = vec![1u32, 2, 3, 4];
        let pending = device.upload(&input)?;
        input.fill(9);
        let buffer = pending.await?;
        let view = buffer.view(1..3)?;
        let pending = device.write(&view, &[8, 7])?;
        drop(view);
        pending.await?;
        assert_eq!(device.download(&buffer)?.await?, [1, 8, 7, 4]);
        assert!(buffer.view(3..5).is_err());
        assert!(device.write(&buffer, &[1]).is_err());
        let other = Emulator::create(EmulatorOptions::default())?;
        assert!(matches!(
            other.download(&buffer),
            Err(SubmitError::ForeignResource)
        ));
        device.shutdown(ShutdownMode::Drain).await?;
        assert!(matches!(device.download(&buffer), Err(SubmitError::Closed)));
        other.shutdown(ShutdownMode::Cancel).await?;
        Ok(())
    })
}

#[test]
fn test_globals_relocations_and_fault_inspection() -> Result {
    let device = Emulator::create(EmulatorOptions::default())?;
    let parsed = parse(
        ".global .u32 value = 7; .const .u32 constant = 19; .global .u64 ptr = generic(value); .entry run() { .reg .b64 a; .reg .b32 b; ld.global.u64 a,[ptr]; ld.u32 b,[a]; add.u32 b,b,1; st.u32 [a],b; trap; }",
    );
    let a = device.load(parsed.clone())?.wait()?;
    let b = device.load(parsed)?.wait()?;
    assert!(device.write(&a.global("constant")?, &[0; 4]).is_err());
    let error = a.kernel("run")?.launch(one(), [])?.wait().unwrap_err();
    assert!(
        matches!(&error, ExecutionError::Launch { cause: launch_cause, report } if report.instructions > 0 && matches!(&**launch_cause, ExecutionError::Lane { location, cause } if location.function == "run" && matches!(**cause, ExecutionError::Trap)))
    );
    assert_eq!(
        device.download(&a.global("value")?)?.wait()?,
        8u32.to_le_bytes()
    );
    assert_eq!(
        device.download(&b.global("value")?)?.wait()?,
        7u32.to_le_bytes()
    );
    assert!(matches!(device.zeroed::<u8>(1), Err(SubmitError::Faulted)));
    assert!(device.shutdown(ShutdownMode::Drain).wait().is_err());
    Ok(())
}

#[test]
fn test_resource_failures_are_local_and_results_release_staging() -> Result {
    let device = Emulator::create(EmulatorOptions {
        device_storage_bytes: 4096,
        staging_storage_bytes: 16,
        // Include command and completion metadata and the precharged fault diagnostic.
        execution_storage_bytes: 8192,
        ..Default::default()
    })?;
    assert!(matches!(
        device.zeroed::<u8>(4096)?.wait(),
        Err(ExecutionError::MemoryLimit)
    ));
    let buffer = device.upload(&[1u32, 2, 3, 4])?.wait()?;
    let download = device.download(&buffer)?;
    // The reservation remains charged whether the worker has finished or not.
    assert!(matches!(
        device.upload(&[1u8]),
        Err(SubmitError::StagingLimit)
    ));
    assert_eq!(download.wait()?, [1, 2, 3, 4]);
    drop(device.upload(&[1u8])?.wait()?);
    let module = device.load(parse(".entry empty() {ret;}"))?.wait()?;
    assert!(matches!(
        module
            .kernel("empty")?
            .launch(LaunchConfig::create([100, 1, 1], [1024, 1, 1], 0)?, [])?
            .wait(),
        Err(ExecutionError::ExecutionStateLimit)
    ));
    module.kernel("empty")?.launch(one(), [])?.wait()?;
    device.shutdown(ShutdownMode::Drain).observe(false)?;
    Ok(())
}

#[test]
fn test_aggregate_argument_staging_accounts_for_retained_capacity() -> Result {
    let device = Emulator::create(EmulatorOptions {
        staging_storage_bytes: 128,
        ..Default::default()
    })?;
    let loaded = device
        .load(parse(".entry k(.param .b8 bytes[1]) { ret; }"))?
        .wait()?;
    let kernel = loaded.kernel("k")?;
    let mut oversized = Vec::with_capacity(4096);
    oversized.push(0);
    assert!(matches!(
        kernel.launch(one(), [KernelArg::aggregate(oversized, 1)?]),
        Err(SubmitError::StagingLimit)
    ));
    kernel
        .launch(one(), [KernelArg::aggregate(vec![0], 1)?])?
        .wait()?;
    device.shutdown(ShutdownMode::Drain).wait()?;
    Ok(())
}

#[test]
fn test_stale_addresses_do_not_retain_allocations() -> Result {
    let device = Emulator::create(EmulatorOptions::default())?;
    let module = device.load(parse(".global .u64 saved; .entry save(.param .u64 p) { .reg .b64 a; ld.param.u64 a,[p]; st.global.u64 [saved],a; ret; } .entry read() { .reg .b64 a; .reg .b32 b; ld.global.u64 a,[saved]; ld.global.u32 b,[a]; ret; }"))?.wait()?;
    let data = device.upload(&[42u32])?.wait()?;
    module
        .kernel("save")?
        .launch(one(), [(&data).into()])?
        .wait()?;
    drop(data);
    // FIFO command acceptance is independent of reclamation; the worker processes reclamation first.
    let replacement = device.upload(&[99u32])?.wait()?;
    assert!(module.kernel("read")?.launch(one(), [])?.wait().is_err());
    assert_eq!(device.download(&replacement)?.wait()?, [99]);
    assert!(device.shutdown(ShutdownMode::Drain).wait().is_err());
    Ok(())
}

#[test]
fn test_argument_layout_and_constant_pointer_annotations() -> Result {
    let device = Emulator::create(EmulatorOptions::default())?;
    let module = device.load(parse(".const .u32 c = 23; .global .u32 output; .entry run(.param .u64 .ptr .const .align 4 p, .param .align 8 .b8 v[8]) { .reg .b64 a; .reg .b32 b; .reg .b32 d; ld.param.u64 a,[p]; ld.const.u32 b,[a]; ld.param.u32 d,[v+4]; add.u32 b,b,d; st.global.u32 [output],b; ret; }"))?.wait()?;
    let args = [
        (&module.global("c")?).into(),
        KernelArg::aggregate(vec![0, 0, 0, 0, 7, 0, 0, 0], 8)?,
    ];
    module.kernel("run")?.launch(one(), args)?.wait()?;
    assert_eq!(
        device.download(&module.global("output")?)?.wait()?,
        30u32.to_le_bytes()
    );
    device.shutdown(ShutdownMode::Drain).wait()?;
    Ok(())
}

#[test]
fn test_caller_local_pointers_recursion_limits_and_indeterminate_reads() -> Result {
    let device = Emulator::create(EmulatorOptions::default())?;
    let module = device.load(parse(".global .u32 output; .func helper(.param .u64 arg) { .reg .b64 a; ld.param.u64 a,[arg]; st.u32 [a],41; ret; } .entry run() { .local .u32 local; .param .u64 arg; .reg .b64 a; .reg .b32 b; mov.u64 a,local; cvta.local.u64 a,a; st.param.u64 [arg],a; call helper,(arg); ld.local.u32 b,[local]; st.global.u32 [output],b; ret; }"))?.wait()?;
    module.kernel("run")?.launch(one(), [])?.wait()?;
    assert_eq!(
        device.download(&module.global("output")?)?.wait()?,
        41u32.to_le_bytes()
    );
    device.shutdown(ShutdownMode::Drain).wait()?;
    for body in [
        ".reg .u32 r; add.u32 r,r,1;",
        ".local .u32 x; .reg .u32 r; ld.local.u32 r,[x];",
    ] {
        let device = Emulator::create(EmulatorOptions::default())?;
        let loaded = device
            .load(parse(&format!(".entry run() {{ {body} ret; }}")))?
            .wait()?;
        let error = loaded.kernel("run")?.launch(one(), [])?.wait().unwrap_err();
        assert!(error.to_string().contains("indeterminate"));
        assert!(device.shutdown(ShutdownMode::Drain).wait().is_err());
    }
    let device = Emulator::create(EmulatorOptions {
        call_depth: 3,
        ..Default::default()
    })?;
    let loaded = device
        .load(parse(
            ".func recurse() {call recurse,(); ret;} .entry run() {call recurse,(); ret;}",
        ))?
        .wait()?;
    assert!(
        loaded
            .kernel("run")?
            .launch(one(), [])?
            .wait()
            .unwrap_err()
            .to_string()
            .contains("call depth")
    );
    assert!(device.shutdown(ShutdownMode::Drain).wait().is_err());
    Ok(())
}

#[test]
fn test_cross_cta_polling_progress_and_instruction_limit() -> Result {
    let device = Emulator::create(EmulatorOptions::default())?;
    let loaded = device.load(parse(".global .u32 flag; .global .u32 output; .entry run() { .reg .u32 r; .reg .pred p; mov.u32 r,%ctaid.x; setp.eq.u32 p,r,1; @p bra producer; loop: ld.volatile.global.u32 r,[flag]; setp.eq.u32 p,r,0; @p bra loop; st.global.u32 [output],91; ret; producer: st.volatile.global.u32 [flag],1; ret; }"))?.wait()?;
    loaded
        .kernel("run")?
        .launch(LaunchConfig::create([2, 1, 1], [1, 1, 1], 0)?, [])?
        .wait()?;
    assert_eq!(
        device.download(&loaded.global("output")?)?.wait()?,
        91u32.to_le_bytes()
    );
    device.shutdown(ShutdownMode::Drain).wait()?;
    let device = Emulator::create(EmulatorOptions {
        launch_instructions: 10,
        ..Default::default()
    })?;
    let loaded = device
        .load(parse(".entry run() { loop: bra loop; }"))?
        .wait()?;
    assert!(
        loaded
            .kernel("run")?
            .launch(one(), [])?
            .wait()
            .unwrap_err()
            .to_string()
            .contains("instruction budget")
    );
    assert!(device.shutdown(ShutdownMode::Drain).wait().is_err());
    Ok(())
}

#[test]
fn test_divergent_collectives_repeated_generations_and_partial_warps() -> Result {
    let device = Emulator::create(EmulatorOptions::default())?;
    let loaded = device.load(parse(".global .u32 output[32]; .entry run() { .reg .u32 t; .reg .u32 r; .reg .u32 n; .reg .b64 a; .reg .b64 base; .reg .pred p; mov.u64 base,output; mov.u32 t,%tid.x; mov.u32 n,0; loop: setp.lt.u32 p,t,8; @p bra left; shfl.sync.bfly.b32 r,t,1,31,0xffff; vote.sync.ballot.b32 r,p,0xffff; bar.warp.sync 0xffff; bra joined; left: shfl.sync.bfly.b32 r,t,1,31,0xffff; vote.sync.ballot.b32 r,p,0xffff; bar.warp.sync 0xffff; joined: bar.sync 0; add.u32 n,n,1; setp.lt.u32 p,n,3; @p bra loop; mul.wide.u32 a,t,4; add.u64 a,a,base; st.global.u32 [a],r; ret; }"))?.wait()?;
    loaded
        .kernel("run")?
        .launch(LaunchConfig::create([1, 1, 1], [16, 1, 1], 0)?, [])?
        .wait()?;
    let result = device.download(&loaded.global("output")?)?.wait()?;
    assert_eq!(
        &result[..64],
        [255u32; 16]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>()
    );
    device.shutdown(ShutdownMode::Drain).wait()?;
    Ok(())
}

#[test]
fn test_barrier_stalls_and_memory_alignment_are_diagnostic() -> Result {
    for (body, message) in [
        (
            ".reg .pred p; .reg .u32 t; mov.u32 t,%tid.x; setp.eq.u32 p,t,0; @p bar.warp.sync 3; wait: bra wait;",
            "instruction budget",
        ),
        (
            ".reg .pred p; .reg .u32 t; mov.u32 t,%tid.x; setp.eq.u32 p,t,0; @p bra left; bar.warp.sync 3; ret; left: bar.sync 0;",
            "scheduler stalled",
        ),
        (
            ".local .align 4 .b8 bytes[8]; .reg .u32 r; ld.local.u32 r,[bytes+1];",
            "misaligned",
        ),
    ] {
        let device = Emulator::create(EmulatorOptions {
            launch_instructions: 100,
            ..Default::default()
        })?;
        let loaded = device
            .load(parse(&format!(".entry run() {{ {body} ret; }}")))?
            .wait()?;
        let error = loaded
            .kernel("run")?
            .launch(LaunchConfig::create([1, 1, 1], [2, 1, 1], 0)?, [])?
            .wait()
            .unwrap_err();
        assert!(error.to_string().contains(message), "{error}");
        assert!(device.shutdown(ShutdownMode::Drain).wait().is_err());
    }
    Ok(())
}

#[test]
fn test_failed_load_rolls_back_and_views_keep_allocations_alive() -> Result {
    let device = Emulator::create(EmulatorOptions {
        device_storage_bytes: 4096,
        ..Default::default()
    })?;
    let oversized = parse(".global .b8 a[2048]; .global .b8 b[2048]; .entry k() {ret;}");
    assert!(matches!(
        device.load(oversized)?.wait(),
        Err(ExecutionError::MemoryLimit)
    ));
    let buffer = device.upload(&[5u32, 6, 7, 8])?.wait()?;
    let view = buffer.view(1..2)?;
    drop(buffer);
    assert_eq!(device.download(&view)?.wait()?, [6]);
    // PTX addresses are checked against the allocation, while host transfers use the view.
    let loaded = device.load(parse(".entry k(.param .u64 p) { .reg .u64 a; ld.param.u64 a,[p]; st.global.u32 [a+4],19; ret; }"))?.wait()?;
    loaded
        .kernel("k")?
        .launch(one(), [(&view).into()])?
        .wait()?;
    assert!(view.view(0..2).is_err());
    let too_large = device.zeroed::<u8>(4096)?.wait();
    assert!(matches!(too_large, Err(ExecutionError::MemoryLimit)));
    device.shutdown(ShutdownMode::Drain).wait()?;
    Ok(())
}

#[test]
fn test_aggregate_call_padding_remains_indeterminate_until_read() -> Result {
    for read_padding in [false, true] {
        let device = Emulator::create(EmulatorOptions::default())?;
        let offset = if read_padding { 4 } else { 0 };
        let source = format!(
            ".global .u32 output; .func (.param .align 8 .b8 result[8]) helper(.param .align 8 .b8 input[8]) {{ .reg .u32 r; ld.param.u32 r,[input]; st.param.u32 [result],r; ret; }} .entry k() {{ .param .align 8 .b8 arg[8]; .param .align 8 .b8 result[8]; .reg .u32 r; st.param.u32 [arg],37; call (result),helper,(arg); ld.param.u32 r,[result+{offset}]; st.global.u32 [output],r; ret; }}"
        );
        let loaded = device.load(parse(&source))?.wait()?;
        let result = loaded.kernel("k")?.launch(one(), [])?.wait();
        if read_padding {
            assert!(result.unwrap_err().to_string().contains("indeterminate"));
            assert!(device.shutdown(ShutdownMode::Drain).wait().is_err());
        } else {
            result?;
            assert_eq!(
                device.download(&loaded.global("output")?)?.wait()?,
                37u32.to_le_bytes()
            );
            device.shutdown(ShutdownMode::Drain).wait()?;
        }
    }
    Ok(())
}

#[test]
fn test_generic_addresses_survive_integer_width_conversion() -> Result {
    let device = Emulator::create(EmulatorOptions::default())?;
    let loaded = device.load(parse(".global .u32 output; .entry k() { .shared .u32 value; .reg .u32 a; .reg .u32 r; .reg .u64 wide; .reg .u64 other; .reg .pred p; st.shared.u32 [value],29; mov.u32 a,value; cvta.shared.u32 a,a; cvt.u64.u32 wide,a; ld.u32 r,[wide]; mov.u64 other,value; cvta.shared.u64 other,other; setp.eq.u64 p,wide,other; @!p trap; cvta.to.shared.u64 wide,wide; ld.shared.u32 r,[wide]; st.global.u32 [output],r; ret; }"))?.wait()?;
    loaded.kernel("k")?.launch(one(), [])?.wait()?;
    assert_eq!(
        device.download(&loaded.global("output")?)?.wait()?,
        29u32.to_le_bytes()
    );
    device.shutdown(ShutdownMode::Drain).wait()?;
    Ok(())
}

#[test]
fn test_expired_shared_and_local_addresses_cannot_alias_a_later_launch() -> Result {
    for space in ["shared", "local"] {
        let device = Emulator::create(EmulatorOptions::default())?;
        let loaded = device.load(parse(&format!(
            ".global .u64 saved; .entry save() {{ .{space} .u32 scratch; .reg .u64 a; st.{space}.u32 [scratch],19; mov.u64 a,scratch; cvta.{space}.u64 a,a; st.global.u64 [saved],a; ret; }} .entry read() {{ .{space} .u32 scratch; .reg .u64 a; .reg .u32 r; st.{space}.u32 [scratch],37; ld.global.u64 a,[saved]; ld.u32 r,[a]; ret; }}"
        )))?.wait()?;
        loaded.kernel("save")?.launch(one(), [])?.wait()?;
        let error = loaded
            .kernel("read")?
            .launch(one(), [])?
            .wait()
            .unwrap_err();
        assert!(
            error.to_string().contains("invalid memory access"),
            "{error}"
        );
        assert!(device.shutdown(ShutdownMode::Drain).wait().is_err());
    }
    Ok(())
}

#[test]
fn test_repeated_launches_reuse_expired_local_address_ranges() -> Result {
    // Each launch uses 64 MiB of local addresses, so 72 launches exceed one 4 GiB window.
    let device = Emulator::create(EmulatorOptions::default())?;
    let module = device
        .load(parse(
            ".entry k() { .local .align 8 .b8 scratch[65536]; ret; }",
        ))?
        .wait()?;
    let kernel = module.kernel("k")?;
    let config = LaunchConfig::create([1, 1, 1], [1024, 1, 1], 0)?;
    for _ in 0..72 {
        kernel.launch(config, [])?.wait()?;
    }
    device.shutdown(ShutdownMode::Drain).wait()?;
    Ok(())
}

#[test]
fn test_integer_division_by_zero_leaves_an_indeterminate_destination() -> Result {
    for (operation, consume) in [
        ("div.u32", false),
        ("rem.s32", false),
        ("div.s64", true),
        ("rem.u64", true),
    ] {
        let ty = &operation[operation.len() - 3..];
        let store = if consume {
            format!("st.global.{ty} [output],d;")
        } else {
            String::new()
        };
        let device = Emulator::create(EmulatorOptions::default())?;
        let module = device
            .load(parse(&format!(
                ".global .{ty} output; .entry k() {{ .reg .{ty} d; {operation} d,7,0; {store} ret; }}"
            )))?
            .wait()?;
        let result = module.kernel("k")?.launch(one(), [])?.wait();
        if consume {
            let error = result.unwrap_err();
            assert!(
                matches!(error.root_cause(), ExecutionError::Indeterminate(_)),
                "{operation}: {error}"
            );
            assert!(device.shutdown(ShutdownMode::Drain).wait().is_err());
        } else {
            result?;
            device.shutdown(ShutdownMode::Drain).wait()?;
        }
    }
    Ok(())
}
