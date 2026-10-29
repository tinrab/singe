//! Inspect, execute, compare, and diagnose PTX without CUDA.

use std::{
    error::Error,
    future::Future,
    pin::pin,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
    thread,
};

use singe_ptx::emulator::{
    CompiledKernel, Emulator, EmulatorOptions, ExecutionError, LaunchConfig, ParsedModule,
    ShutdownMode,
};

fn main() -> Result<(), Box<dyn Error>> {
    let parsed = ParsedModule::parse(include_str!("../tests/fixtures/frontend/kernels.ptx"))?;
    println!("syntax: {} module items", parsed.syntax().items.len());
    let layout = parsed.layout()?;
    println!(
        "storage: {} module declarations, {:?} addresses",
        layout.variables().len(),
        layout.address_size()
    );
    let compiled = parsed.compile("vector_add")?;
    compiled.launch_requirements().validate_block([32, 1, 1])?;
    for parameter in compiled.parameters() {
        println!(
            "parameter {}: {} bytes at offset {}",
            parameter.name(),
            parameter.storage().byte_len(),
            parameter.byte_offset()
        );
    }

    let device = Emulator::create(EmulatorOptions::default())?;
    let module = device.load(parsed.clone())?.wait()?;
    let kernel = module.bind(&compiled)?;
    let left_values = [1i32, 2, 3, 4, 5];
    let right_values = [10i32, 20, 30, 40, 50];
    let left = device.upload(&left_values)?.wait()?;
    let right = device.upload(&right_values)?.wait()?;
    let output = device.zeroed::<i32>(5)?.wait()?;
    let config = LaunchConfig::create([1, 1, 1], [32, 1, 1], 0)?;

    let report = kernel
        .launch(
            config,
            [
                (&left).into(),
                (&right).into(),
                (&output).into(),
                5u32.into(),
            ],
        )?
        .wait()?;
    let values = device.download(&output)?.wait()?;
    let expected: Vec<_> = left_values
        .iter()
        .zip(right_values)
        .map(|(left, right)| left.wrapping_add(right))
        .collect();
    assert_eq!(values, expected);
    println!(
        "blocking: {values:?}, {} lane instructions",
        report.instructions
    );

    reuse_compiled(
        parsed,
        &compiled,
        config,
        &left_values,
        &right_values,
        &expected,
    )?;

    block_on(async {
        let transfer = device.create_stream()?;
        let compute = device.create_stream()?;
        let updated = [100, 200, 300, 400, 500];
        let write = transfer.write(&right, &updated)?;
        let (ready, recorded) = transfer.record_event()?;
        let dependency = compute.wait_event(&ready)?;
        let report = kernel
            .launch_on(
                &compute,
                config,
                [
                    (&left).into(),
                    (&right).into(),
                    (&output).into(),
                    5u32.into(),
                ],
            )?
            .await?;
        write.await?;
        recorded.await?;
        dependency.await?;
        let values = compute.download(&output)?.await?;
        let expected: Vec<_> = left_values
            .iter()
            .zip(updated)
            .map(|(left, right)| left.wrapping_add(right))
            .collect();
        assert_eq!(values, expected);
        println!(
            "async: {values:?}, {} lane instructions",
            report.instructions
        );
        device.shutdown(ShutdownMode::Drain).await?;
        Ok::<_, Box<dyn Error>>(())
    })?;
    inspect_fault()
}

/// Binds already compiled code to a separate emulator with independent storage.
fn reuse_compiled(
    parsed: ParsedModule,
    compiled: &CompiledKernel,
    config: LaunchConfig,
    left_values: &[i32],
    right_values: &[i32],
    expected: &[i32],
) -> Result<(), Box<dyn Error>> {
    let other = Emulator::create(EmulatorOptions::default())?;
    let other_module = other.load(parsed)?.wait()?;
    let other_kernel = other_module.bind(compiled)?;
    let other_left = other.upload(left_values)?.wait()?;
    let other_right = other.upload(right_values)?.wait()?;
    let other_output = other.zeroed::<i32>(5)?.wait()?;
    other_kernel
        .launch(
            config,
            [
                (&other_left).into(),
                (&other_right).into(),
                (&other_output).into(),
                5u32.into(),
            ],
        )?
        .wait()?;
    assert_eq!(other.download(&other_output)?.wait()?, expected);
    other.shutdown(ShutdownMode::Drain).wait()?;
    println!("reused compiled code on an independent emulator");
    Ok(())
}

fn inspect_fault() -> Result<(), Box<dyn Error>> {
    let source = r".version 9.4
.target sm_80
.address_size 64
.global .u32 output = 0;
.entry fail() {
    .reg .u32 value;
    .reg .b32 unused;
    mov.u32 value, 42;
    st.global.u32 [output], value;
    brkpt;
    add.u32 value, value, 1;
    trap;
}";
    let parsed = ParsedModule::parse(source)?;
    let device = Emulator::create(EmulatorOptions::default())?;
    let module = device.load(parsed)?.wait()?;
    let diagnostic = device.create_stream()?;
    let launch = module
        .kernel("fail")?
        .launch(LaunchConfig::create([1; 3], [1; 3], 0)?, [])?;
    let breakpoint = device.next_breakpoint()?.wait()?;
    let snapshot = device
        .inspect_lane(
            &breakpoint.id,
            breakpoint.location.cta,
            breakpoint.location.lane,
        )?
        .wait()?;
    for frame in snapshot.frames() {
        println!(
            "paused frame: {} at instruction {}",
            frame.function(),
            frame.instruction()
        );
        if let Some(span) = frame.span() {
            println!("next instruction: {}", source[span.start..span.end].trim());
        }
        for register in frame.registers() {
            if let Some(value) = register.value() {
                println!("register {}: {value:#x}", register.name());
            } else {
                println!("register {}: uninitialized", register.name());
            }
        }
    }
    // This stream can read memory while the default stream's launch is suspended.
    assert_eq!(
        diagnostic.download(&module.global("output")?)?.wait()?,
        42u32.to_le_bytes()
    );
    device.resume(&breakpoint.id)?.wait()?;
    let result = launch.wait();
    let Err(error) = result else {
        return Err("expected the demonstration kernel to trap".into());
    };
    assert!(matches!(error.root_cause(), ExecutionError::Trap));
    println!("fault: {}", error.with_source(source));
    if let Some(report) = error.launch_report() {
        println!(
            "fault report: launch {}, {} completed lanes, {} lane instructions",
            report.launch_id, report.completed_lanes, report.instructions
        );
    }
    let output = device.download(&module.global("output")?)?.wait()?;
    assert_eq!(output, 42u32.to_le_bytes());
    println!("completed write: {output:?}");
    // Shutdown also reports the original kernel fault.
    let Err(error) = device.shutdown(ShutdownMode::Drain).wait() else {
        return Err("expected shutdown to retain the kernel fault".into());
    };
    assert!(matches!(error.root_cause(), ExecutionError::Trap));
    Ok(())
}

// A single-future executor is sufficient for this example.
// Applications can await submissions on their existing executor instead.
struct Notify(thread::Thread);

impl Wake for Notify {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

fn block_on<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(Notify(thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(result) => return result,
            Poll::Pending => thread::park(),
        }
    }
}
