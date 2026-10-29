//! Opt-in, result-checked CPU benchmarks. Run with --release --ignored --nocapture.

use std::{error::Error, result::Result as StdResult, time::Instant};

use singe_ptx::emulator::{Emulator, EmulatorOptions, LaunchConfig, ParsedModule, ShutdownMode};

type Result<T = ()> = StdResult<T, Box<dyn Error>>;

fn require_release() -> Result {
    if cfg!(debug_assertions) {
        Err("run performance tests with --release".into())
    } else {
        Ok(())
    }
}

fn benchmark_dispatch(name: &str, body: &str, threads: u32, rounds: u32) -> Result {
    let source = format!(
        ".version 9.3 .target sm_80 .address_size 64
        .global .u32 output[{threads}];
        .global .align 16 .b8 scratch[{}];
        .func (.param .u32 result) increment(.param .u32 input) {{
            .reg .u32 r; ld.param.u32 r,[input]; add.u32 r,r,1;
            st.param.u32 [result],r; ret;
        }}
        .entry k() {{
            .reg .u32 t; .reg .u32 r; .reg .u32 i; .reg .u32 r1; .reg .u32 r2;
            .reg .u32 r3; .reg .u32 r4;
            .reg .u64 address; .reg .u64 v; .reg .u64 offset;
            .reg .pred p; .param .u32 argument; .param .u32 returned;
            mov.u32 t,%tid.x; mov.u32 r,0; mov.u32 i,0;
            mul.wide.u32 offset,t,16; mov.u64 v,scratch; add.u64 v,v,offset;
        loop:
            {body}
            add.u32 i,i,1; setp.lt.u32 p,i,{rounds}; @p bra loop;
            mul.wide.u32 offset,t,4; mov.u64 address,output;
            add.u64 address,address,offset; st.global.u32 [address],r; ret;
        }}",
        threads * 16,
    );
    let device = Emulator::create(EmulatorOptions::default())?;
    let module = device.load(ParsedModule::parse(&source)?)?.wait()?;
    let kernel = module.kernel("k")?;
    let config = LaunchConfig::create([1; 3], [threads, 1, 1], 0)?;
    kernel.launch(config, [])?.wait()?;
    let start = Instant::now();
    let mut instructions = 0;
    for _ in 0..3 {
        let report = kernel.launch(config, [])?.wait()?;
        assert_eq!(report.completed_lanes, u64::from(threads));
        instructions += report.instructions;
    }
    let elapsed = start.elapsed();
    let output = device.download(&module.global("output")?)?.wait()?;
    for value in output.as_chunks::<4>().0 {
        assert_eq!(u32::from_le_bytes(*value), rounds);
    }
    println!(
        "{name}: {:.3} ms, {:.3} million lane instructions/s ({instructions} instructions)",
        elapsed.as_secs_f64() * 1000.0,
        f64::from(u32::try_from(instructions)?) / elapsed.as_secs_f64() / 1_000_000.0,
    );
    device.shutdown(ShutdownMode::Drain).wait()?;
    Ok(())
}

#[test]
#[ignore = "release-mode throughput benchmark; no timing thresholds"]
fn dispatch_throughput() -> Result {
    require_release()?;
    benchmark_dispatch("arithmetic", "add.u32 r,r,1;", 32, 2000)?;
    benchmark_dispatch(
        "vector memory",
        "st.global.v4.u32 [v],{r,r,r,r}; ld.global.v4.u32 {r1,r2,r3,r4},[v]; add.u32 r,r1,1;",
        32,
        1000,
    )?;
    benchmark_dispatch(
        "device calls",
        "st.param.u32 [argument],r; call (returned),increment,(argument); ld.param.u32 r,[returned];",
        32,
        1000,
    )
}

#[test]
#[ignore = "release-mode throughput benchmark; no timing thresholds"]
fn barrier_throughput() -> Result {
    require_release()?;
    benchmark_dispatch(
        "1024-lane barriers",
        "bar.sync 0; add.u32 r,r,1; bar.sync 0;",
        1024,
        32,
    )
}

#[test]
#[ignore = "release-mode throughput benchmark; no timing thresholds"]
fn grid_retirement_throughput() -> Result {
    require_release()?;
    let parsed = ParsedModule::parse(
        ".version 9.3 .target sm_80 .address_size 64
        .entry k() { .local .align 16 .b8 scratch[16]; st.local.u32 [scratch],1; ret; }",
    )?;
    for retained in [0, 1024, 8192] {
        let device = Emulator::create(EmulatorOptions::default())?;
        let mut unrelated = Vec::with_capacity(retained);
        for _ in 0..retained {
            unrelated.push(device.zeroed::<u32>(1)?.wait()?);
        }
        let module = device.load(parsed.clone())?.wait()?;
        let kernel = module.kernel("k")?;
        let config = LaunchConfig::create([1; 3], [1; 3], 0)?;
        kernel.launch(config, [])?.wait()?;
        let start = Instant::now();
        for _ in 0..500 {
            assert_eq!(kernel.launch(config, [])?.wait()?.completed_lanes, 1);
        }
        let elapsed = start.elapsed();
        for buffer in &unrelated {
            assert_eq!(device.download(buffer)?.wait()?, [0]);
        }
        println!(
            "grid retirement, {retained} unrelated buffers: {:.3} ms, {:.0} launches/s (500 launches)",
            elapsed.as_secs_f64() * 1000.0,
            500.0 / elapsed.as_secs_f64(),
        );
        device.shutdown(ShutdownMode::Drain).wait()?;
    }
    Ok(())
}

#[test]
#[ignore = "release-mode throughput benchmark; no timing thresholds"]
fn grid_scaling_throughput() -> Result {
    require_release()?;
    let parsed = ParsedModule::parse(
        ".version 9.3 .target sm_80 .address_size 64
        .entry k() { .reg .u32 t; mov.u32 t,%tid.x; add.u32 t,t,1; bar.sync 0; ret; }",
    )?;
    let device = Emulator::create(EmulatorOptions::default())?;
    let module = device.load(parsed)?.wait()?;
    let kernel = module.kernel("k")?;
    // Waiter resolution should scale with the changed CTAs and warps, not the grid size.
    for ctas in [64, 256, 1024] {
        let start = Instant::now();
        let report = kernel
            .launch(LaunchConfig::create([ctas, 1, 1], [128, 1, 1], 0)?, [])?
            .wait()?;
        assert_eq!(report.completed_ctas, u64::from(ctas));
        println!(
            "grid scaling, {ctas} CTAs of 128 lanes: {:.3} ms",
            start.elapsed().as_secs_f64() * 1000.0,
        );
    }
    device.shutdown(ShutdownMode::Drain).wait()?;
    Ok(())
}
