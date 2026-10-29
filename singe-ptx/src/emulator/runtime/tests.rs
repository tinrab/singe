use super::*;

mod observation;

use std::{
    collections::BTreeSet,
    pin::{self, Pin},
    sync::{Barrier, atomic, mpsc},
    task::{Context, Poll, Wake},
    thread,
    time::{Duration, Instant},
};

use observation::Observe;

use crate::{
    ast::StateSpace,
    emulator::{
        LaneSnapshot, ParsedModule, RegisterSnapshot,
        engine::{Engine, GridBuilder, LaunchTraceEntry},
        memory::{CHUNK_BYTES, Memory, Owner},
        runtime::submit::submit_poll,
    },
};

/// Admits an immediate command that runs `build`'s closure once on the worker.
fn submit_to<T: Send + 'static, F, B>(
    client: &Arc<Client>,
    stream: u64,
    diagnostic: bool,
    size: usize,
    build: B,
) -> Result<Submission<T>, SubmitError>
where
    F: FnOnce(&mut Worker) -> Result<Pending<T>, ExecutionError> + Send + 'static,
    B: FnOnce(Staging) -> F,
{
    submit_poll(
        client,
        stream,
        CommandKind::Immediate,
        diagnostic,
        size,
        move |staging| {
            let mut run = Some(build(staging));
            move |worker: &mut Worker, terminal| {
                Poll::Ready(match terminal {
                    Some(error) => Err(error),
                    None => run.take().ok_or_else(|| {
                        ExecutionError::Invalid("command already consumed".into())
                    })?(worker),
                })
            }
        },
    )
}

fn inspection_launch(
    device: &Emulator,
    declarations: &str,
) -> (Module, Submission<ExecutionReport>, Breakpoint) {
    let parsed = ParsedModule::parse(&format!(
        ".version 9.4 .target sm_80 .address_size 64 .entry k() {{ {declarations} brkpt; ret; }}"
    ))
    .unwrap();
    let module = device.load(parsed).unwrap().wait().unwrap();
    let stream = device.create_stream().unwrap();
    let launch = module
        .kernel("k")
        .unwrap()
        .launch_on(
            &stream,
            LaunchConfig::create([1; 3], [1; 3], 0).unwrap(),
            [],
        )
        .unwrap();
    let breakpoint = device.next_breakpoint().unwrap().wait().unwrap();
    (module, launch, breakpoint)
}

#[test]
fn test_inspection_results_and_abandoned_requests_release_their_reservations() {
    struct Signal(Sender<()>);

    impl Wake for Signal {
        fn wake(self: Arc<Self>) {
            self.0.send(()).unwrap();
        }
    }
    let device = Emulator::create(EmulatorOptions::default()).unwrap();
    let (_module, launch, breakpoint) = inspection_launch(&device, ".reg .b32 r<128>;");
    let budget = &device.client.shared.execution_budget;
    let baseline = budget.used();
    let (blocked, release) = pause(&device);
    drop(device.inspect_lane(&breakpoint.id, [0; 3], [0; 3]).unwrap());
    release.send(()).unwrap();
    blocked.wait().unwrap();
    device.synchronize().unwrap().wait().unwrap();
    assert_eq!(budget.used(), baseline);

    let (blocked, release) = pause(&device);
    let mut inspection = Box::pin(device.inspect_lane(&breakpoint.id, [0; 3], [0; 3]).unwrap());
    let (wake_tx, wake_rx) = mpsc::channel();
    let waker = Waker::from(Arc::new(Signal(wake_tx)));
    assert!(
        inspection
            .as_mut()
            .poll(&mut Context::from_waker(&waker))
            .is_pending()
    );
    release.send(()).unwrap();
    blocked.wait().unwrap();
    wake_rx.recv().unwrap();
    assert!(budget.used() > baseline);
    drop(inspection);
    device.synchronize().unwrap().wait().unwrap();
    assert_eq!(budget.used(), baseline);
    let snapshot = device
        .inspect_lane(&breakpoint.id, [0; 3], [0; 3])
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(budget.used(), baseline);
    let shutdown = device.shutdown(ShutdownMode::Cancel);
    assert!(matches!(
        launch.wait().unwrap_err().root_cause(),
        ExecutionError::Cancelled
    ));
    shutdown.wait().unwrap();
    assert_eq!(budget.used(), 0);
    assert_eq!(snapshot.frames()[0].registers().len(), 128);
}

#[test]
fn test_inspection_budget_exhaustion_is_local_and_a_snapshot_can_be_retried() {
    let device = Emulator::create(EmulatorOptions::default()).unwrap();
    let (_module, launch, breakpoint) = inspection_launch(&device, ".reg .b32 r<8192>;");
    let budget = &device.client.shared.execution_budget;
    let baseline = budget.used();
    let full = budget
        .reserve(device.client.shared.options.execution_storage_bytes - budget.used())
        .unwrap();
    assert!(matches!(
        device.inspect_lane(&breakpoint.id, [0; 3], [0; 3]),
        Err(SubmitError::ExecutionStateLimit)
    ));
    drop(full);
    let (blocked, release) = pause(&device);
    let inspection = device.inspect_lane(&breakpoint.id, [0; 3], [0; 3]).unwrap();
    // Even releasing every transient request/command charge cannot fit this result.
    assert!(8192 * size_of::<RegisterSnapshot>() > budget.used() - baseline);
    let full = budget
        .reserve(device.client.shared.options.execution_storage_bytes - budget.used())
        .unwrap();
    release.send(()).unwrap();
    blocked.wait().unwrap();
    assert!(matches!(
        inspection.wait(),
        Err(ExecutionError::ExecutionStateLimit)
    ));
    drop(full);
    let snapshot = device
        .inspect_lane(&breakpoint.id, [0; 3], [0; 3])
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(snapshot.frames()[0].registers().len(), 8192);
    assert_eq!(budget.used(), baseline);
    device.resume(&breakpoint.id).unwrap().wait().unwrap();
    launch.wait().unwrap();
    device.shutdown(ShutdownMode::Drain).wait().unwrap();
    assert_eq!(budget.used(), 0);
}

#[test]
fn test_inspection_resume_and_cancel_races_resolve_every_accepted_request() {
    for cancelling in [false, true] {
        let device = Emulator::create(EmulatorOptions::default()).unwrap();
        let (_module, launch, breakpoint) = inspection_launch(&device, ".reg .u32 value;");
        let (blocked, release) = pause(&device);
        let inspection = device.inspect_lane(&breakpoint.id, [0; 3], [0; 3]).unwrap();
        if cancelling {
            let shutdown = device.shutdown(ShutdownMode::Cancel);
            release.send(()).unwrap();
            blocked.wait().unwrap();
            assert!(matches!(inspection.wait(), Err(ExecutionError::Cancelled)));
            assert!(matches!(
                launch.wait().unwrap_err().root_cause(),
                ExecutionError::Cancelled
            ));
            shutdown.wait().unwrap();
        } else {
            device.resume(&breakpoint.id).unwrap().wait().unwrap();
            release.send(()).unwrap();
            blocked.wait().unwrap();
            assert!(matches!(inspection.wait(), Err(ExecutionError::Invalid(_))));
            launch.wait().unwrap();
            device.shutdown(ShutdownMode::Drain).wait().unwrap();
        }
        assert_eq!(device.client.shared.execution_budget.used(), 0);
    }
}

#[test]
fn test_inspection_faults_and_worker_unwind_resolve_pending_observations() {
    for unwind in [false, true] {
        let device = Emulator::create(EmulatorOptions::default()).unwrap();
        let (_module, launch, breakpoint) = inspection_launch(&device, "");
        let fault = device
            .load(
                ParsedModule::parse(".version 9.4 .target sm_80 .entry fail() { trap; }").unwrap(),
            )
            .unwrap()
            .wait()
            .unwrap();
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        *lock(&device.client.shared.hooks.launch) = Some(Box::new(move |_| {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            assert!(!unwind, "injected inspection worker failure");
        }));
        let failure = fault
            .kernel("fail")
            .unwrap()
            .launch(LaunchConfig::create([1; 3], [1; 3], 0).unwrap(), [])
            .unwrap();
        started_rx.recv().unwrap();
        let inspection = device.inspect_lane(&breakpoint.id, [0; 3], [0; 3]).unwrap();
        release_tx.send(()).unwrap();
        let error = inspection.wait().unwrap_err();
        if unwind {
            assert!(matches!(error, ExecutionError::WorkerFailed));
        } else {
            assert!(matches!(error, ExecutionError::Faulted));
        }
        assert!(failure.wait().is_err());
        assert!(launch.wait().is_err());
        assert!(device.shutdown(ShutdownMode::Drain).wait().is_err());
        assert_only_fault_storage(&device);
    }
}

#[test]
fn test_inspection_publication_can_reenter_control_admission() {
    struct InspectAgain {
        device: Emulator,
        id: BreakpointId,
        next: Sender<Submission<LaneSnapshot>>,
        unlocked: AtomicBool,
    }

    impl Wake for InspectAgain {
        fn wake(self: Arc<Self>) {
            self.unlocked.store(
                self.device.client.shared.admission.try_lock().is_ok(),
                Ordering::Release,
            );
            self.next
                .send(self.device.inspect_lane(&self.id, [0; 3], [0; 3]).unwrap())
                .unwrap();
        }
    }
    let device = Emulator::create(EmulatorOptions::default()).unwrap();
    let (_module, launch, breakpoint) = inspection_launch(&device, ".reg .u32 value;");
    let (blocked, release) = pause(&device);
    let mut first = Box::pin(device.inspect_lane(&breakpoint.id, [0; 3], [0; 3]).unwrap());
    let (next_tx, next_rx) = mpsc::channel();
    let wake = Arc::new(InspectAgain {
        device: device.clone(),
        id: breakpoint.id.clone(),
        next: next_tx,
        unlocked: AtomicBool::new(false),
    });
    let waker = Waker::from(Arc::clone(&wake));
    assert!(
        first
            .as_mut()
            .poll(&mut Context::from_waker(&waker))
            .is_pending()
    );
    release.send(()).unwrap();
    blocked.wait().unwrap();
    let next = next_rx.recv().unwrap();
    let first = Pin::into_inner(first).wait().unwrap();
    assert_eq!(next.wait().unwrap(), first);
    assert!(wake.unlocked.load(Ordering::Acquire));
    device.resume(&breakpoint.id).unwrap().wait().unwrap();
    launch.wait().unwrap();
    device.shutdown(ShutdownMode::Drain).wait().unwrap();
    assert_eq!(device.client.shared.execution_budget.used(), 0);
}

#[test]
fn test_owned_snapshot_does_not_keep_an_orphaned_suspension_or_device_alive() {
    let device = Emulator::create(EmulatorOptions::default()).unwrap();
    let (module, launch, breakpoint) = inspection_launch(&device, ".reg .u32 value;");
    let snapshot = device
        .inspect_lane(&breakpoint.id, [0; 3], [0; 3])
        .unwrap()
        .wait()
        .unwrap();
    let budget = Arc::clone(&device.client.shared.execution_budget);
    let (closed_tx, closed_rx) = mpsc::channel();
    *lock(&device.client.shared.closed_notify) = Some(closed_tx);
    drop(launch);
    drop(module);
    drop(device);
    assert!(closed_rx.recv().is_err());
    assert_eq!(budget.used(), 0);
    assert_eq!(snapshot.frames()[0].function(), "k");
    assert_eq!(snapshot.frames()[0].registers()[0].name(), "value");
    assert_eq!(snapshot.frames()[0].registers()[0].value(), None);
}

fn pause(device: &Emulator) -> (Submission<()>, Sender<()>) {
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let submission = submit_to(&device.client, 0, false, 0, |_| {
        move |_| {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            Ok(Box::new(|_| ()) as Pending<()>)
        }
    })
    .unwrap();
    started_rx.recv().unwrap();
    (submission, release_tx)
}

#[test]
fn test_pending_copies_yield_to_other_streams_and_release_storage_on_terminal_paths() {
    for asynchronous in [false, true] {
        for terminal in ["wait", "cancel", "trap", "unwind"] {
            let device = Emulator::create(EmulatorOptions::default()).unwrap();
            let module = device
                .load(
                    ParsedModule::parse(&format!(
                        ".version 9.4 .target sm_80 .address_size 64
                 .global .u32 output;
                 .entry k() {{ .shared .align 16 .b8 tile[64];
                    .reg .b64 a; .reg .pred ignore; mov.pred ignore,1;
                    cp.async.cg.shared.global [tile],[a],16,ignore;
                    cp.async.commit_group; st.global.u32 [output],77; brkpt;
                    {} ret; }}",
                        if terminal == "trap" {
                            "trap;"
                        } else {
                            "cp.async.wait_all;"
                        }
                    ))
                    .unwrap(),
                )
                .unwrap()
                .observe(asynchronous)
                .unwrap();
            let stream = device.create_stream().unwrap();
            let launch = module
                .kernel("k")
                .unwrap()
                .launch_on(
                    &stream,
                    LaunchConfig::create([1; 3], [1; 3], 0).unwrap(),
                    [],
                )
                .unwrap();
            let breakpoint = device
                .next_breakpoint()
                .unwrap()
                .observe(asynchronous)
                .unwrap();
            // Default-stream work completes while this launch owns pending
            // copies and remains suspended on the independent stream.
            let values = device
                .upload(&[5u32, 6, 7])
                .unwrap()
                .observe(asynchronous)
                .unwrap();
            assert_eq!(
                device
                    .download(&values)
                    .unwrap()
                    .observe(asynchronous)
                    .unwrap(),
                [5, 6, 7]
            );
            assert_eq!(
                device
                    .download(&module.global("output").unwrap())
                    .unwrap()
                    .observe(asynchronous)
                    .unwrap(),
                77u32.to_le_bytes()
            );
            if terminal == "cancel" {
                let shutdown = device.shutdown(ShutdownMode::Cancel);
                assert!(matches!(
                    launch.observe(asynchronous).unwrap_err().root_cause(),
                    ExecutionError::Cancelled
                ));
                shutdown.observe(asynchronous).unwrap();
                assert_eq!(device.client.shared.execution_budget.used(), 0);
            } else {
                if terminal == "unwind" {
                    *lock(&device.client.shared.hooks.launch) =
                        Some(Box::new(|_| panic!("pending-copy worker unwind")));
                }
                device
                    .resume(&breakpoint.id)
                    .unwrap()
                    .observe(asynchronous)
                    .unwrap();
                let result = launch.observe(asynchronous);
                if terminal == "wait" {
                    result.unwrap();
                    device
                        .shutdown(ShutdownMode::Drain)
                        .observe(asynchronous)
                        .unwrap();
                    assert_eq!(device.client.shared.execution_budget.used(), 0);
                } else {
                    let error = result.unwrap_err();
                    if terminal == "trap" {
                        assert!(matches!(error.root_cause(), ExecutionError::Trap));
                    } else {
                        assert!(matches!(error.root_cause(), ExecutionError::WorkerFailed));
                    }
                    assert!(
                        device
                            .shutdown(ShutdownMode::Drain)
                            .observe(asynchronous)
                            .is_err()
                    );
                    assert_only_fault_storage(&device);
                }
            }
        }
    }
}

#[test]
fn test_partial_matrix_rendezvous_supports_inspection_resumption_and_terminal_cleanup() {
    let text = ".version 9.4 .target sm_80 .address_size 64 .global .f32 output[128];
        .entry k() { .reg .f32 c<4>; .reg .u32 t; .reg .u32 count; .reg .pred p;
            .reg .u64 offset; .reg .u64 base; .reg .u64 address;
            mov.u32 t,%tid.x; mov.f32 c0,0f00000000; mov.f32 c1,c0; mov.f32 c2,c0; mov.f32 c3,c0;
            setp.lt.u32 p,t,16; @p bra join;
            mov.u32 count,100;
            delay: sub.u32 count,count,1; setp.ne.u32 p,count,0; @p bra delay;
            brkpt;
            join: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
                {c0,c1,c2,c3},{0x3c003c00,0x3c003c00,0x3c003c00,0x3c003c00},
                {0x3c003c00,0x3c003c00},{c0,c1,c2,c3};
            mul.wide.u32 offset,t,16; mov.u64 base,output; add.u64 address,base,offset;
            st.global.v4.f32 [address],{c0,c1,c2,c3}; ret;
        }";
    for asynchronous in [false, true] {
        for terminal in ["resume", "cancel", "unwind"] {
            let device = Emulator::create(EmulatorOptions::default()).unwrap();
            let module = device
                .load(ParsedModule::parse(text).unwrap())
                .unwrap()
                .observe(asynchronous)
                .unwrap();
            let stream = device.create_stream().unwrap();
            let launch = module
                .kernel("k")
                .unwrap()
                .launch_on(
                    &stream,
                    LaunchConfig::create([1; 3], [32, 1, 1], 0).unwrap(),
                    [],
                )
                .unwrap();
            let breakpoint = device
                .next_breakpoint()
                .unwrap()
                .observe(asynchronous)
                .unwrap();
            let snapshot = device
                .inspect_lane(&breakpoint.id, [0; 3], [0; 3])
                .unwrap()
                .observe(asynchronous)
                .unwrap();
            let span = snapshot.frames()[0].span().unwrap();
            assert!(text[span.start..span.end].starts_with("mma.sync.aligned"));
            let other = device
                .upload(&[7u32])
                .unwrap()
                .observe(asynchronous)
                .unwrap();
            assert_eq!(
                device
                    .download(&other)
                    .unwrap()
                    .observe(asynchronous)
                    .unwrap(),
                [7]
            );
            if terminal == "cancel" {
                let shutdown = device.shutdown(ShutdownMode::Cancel);
                assert!(matches!(
                    launch.observe(asynchronous).unwrap_err().root_cause(),
                    ExecutionError::Cancelled
                ));
                shutdown.observe(asynchronous).unwrap();
                assert_eq!(device.client.shared.execution_budget.used(), 0);
            } else {
                if terminal == "unwind" {
                    *lock(&device.client.shared.hooks.launch) =
                        Some(Box::new(|_| panic!("matrix worker unwind")));
                }
                device
                    .resume(&breakpoint.id)
                    .unwrap()
                    .observe(asynchronous)
                    .unwrap();
                let result = launch.observe(asynchronous);
                if terminal == "resume" {
                    assert_eq!(result.unwrap().completed_lanes, 32);
                    let output = device
                        .download(&module.global("output").unwrap())
                        .unwrap()
                        .observe(asynchronous)
                        .unwrap();
                    let expected: Vec<_> = (0..128).flat_map(|_| 16f32.to_le_bytes()).collect();
                    assert_eq!(output, expected);
                    device
                        .shutdown(ShutdownMode::Drain)
                        .observe(asynchronous)
                        .unwrap();
                    assert_eq!(device.client.shared.execution_budget.used(), 0);
                } else {
                    assert!(matches!(
                        result.unwrap_err().root_cause(),
                        ExecutionError::WorkerFailed
                    ));
                    assert!(
                        device
                            .shutdown(ShutdownMode::Drain)
                            .observe(asynchronous)
                            .is_err()
                    );
                    assert_only_fault_storage(&device);
                }
            }
            assert_eq!(
                snapshot.frames()[0]
                    .registers()
                    .iter()
                    .find(|register| register.name() == "c0")
                    .unwrap()
                    .value(),
                Some(0)
            );
        }
    }
}

fn empty_module() -> ParsedModule {
    ParsedModule::parse(".version 9.3 .target sm_80 .address_size 64 .entry k() {ret;}").unwrap()
}

fn assert_only_fault_storage(device: &Emulator) {
    let retained = lock(&device.client.shared.admission)
        .shutdown_result
        .as_ref()
        .unwrap()
        .as_ref()
        .unwrap_err()
        .retained_bytes();
    assert_eq!(device.client.shared.execution_budget.used(), retained);
}

#[test]
fn test_retained_reports_release_on_consumption_and_abandonment() {
    for asynchronous in [false, true] {
        let device = Emulator::create(EmulatorOptions::default()).unwrap();
        let module = device
            .load(empty_module())
            .unwrap()
            .observe(asynchronous)
            .unwrap();
        let kernel = module.kernel("k").unwrap();
        let baseline = device.client.shared.execution_budget.used();
        let config = LaunchConfig::create([1; 3], [1; 3], 0).unwrap();
        let first = kernel.launch(config, []).unwrap();
        device.synchronize().unwrap().observe(asynchronous).unwrap();
        let retained = device.client.shared.execution_budget.used();
        assert!(retained > baseline);
        let report = first.observe(asynchronous).unwrap();
        assert_eq!(report.completed_lanes, 1);
        assert_eq!(device.client.shared.execution_budget.used(), baseline);
        let second = kernel.launch(config, []).unwrap();
        device.synchronize().unwrap().observe(asynchronous).unwrap();
        assert_eq!(device.client.shared.execution_budget.used(), retained);
        drop(second);
        assert_eq!(device.client.shared.execution_budget.used(), baseline);
        let (blocker, release) = pause(&device);
        drop(kernel.launch(config, []).unwrap());
        release.send(()).unwrap();
        blocker.observe(asynchronous).unwrap();
        device.synchronize().unwrap().observe(asynchronous).unwrap();
        assert_eq!(device.client.shared.execution_budget.used(), baseline);
        device
            .shutdown(ShutdownMode::Drain)
            .observe(asynchronous)
            .unwrap();
    }
}

#[test]
fn test_retained_fault_reports_share_storage_until_the_closed_device_is_released() {
    for asynchronous in [false, true] {
        let device = Emulator::create(EmulatorOptions::default()).unwrap();
        let (closed_tx, closed_rx) = mpsc::channel();
        *lock(&device.client.shared.closed_notify) = Some(closed_tx);
        let module = device
            .load(
                ParsedModule::parse(
                    ".version 9.3 .target sm_80 .address_size 64 .global .u32 output;
             .func helper() {trap;} .entry k() {st.global.u32 [output],17;call helper;ret;}",
                )
                .unwrap(),
            )
            .unwrap()
            .observe(asynchronous)
            .unwrap();
        let config = LaunchConfig::create([1; 3], [1; 3], 0).unwrap();
        let launch = module.kernel("k").unwrap().launch(config, []).unwrap();
        // A diagnostic download fences the failed launch without consuming its result.
        assert_eq!(
            device
                .download(&module.global("output").unwrap())
                .unwrap()
                .observe(asynchronous)
                .unwrap(),
            17u32.to_le_bytes()
        );
        let budget = Arc::clone(&device.client.shared.execution_budget);
        let before = budget.used();
        let error = launch.observe(asynchronous).unwrap_err();
        assert!(matches!(error.root_cause(), ExecutionError::Trap));
        assert!(budget.used() < before); // Only the completion cell was released.
        let retained = budget.used();
        assert!(retained > 0);
        let shutdown_error = device
            .shutdown(ShutdownMode::Drain)
            .observe(asynchronous)
            .unwrap_err();
        assert_eq!(budget.used(), retained);
        let (
            ExecutionError::Launch {
                report: a,
                cause: x,
            },
            ExecutionError::Launch {
                report: b,
                cause: y,
            },
        ) = (&error, &shutdown_error)
        else {
            panic!("missing fault reports")
        };
        assert!(Arc::ptr_eq(a, b));
        assert!(Arc::ptr_eq(x, y));
        drop(module);
        drop(device);
        assert!(closed_rx.recv().is_err());
        assert_eq!(budget.used(), 0);
        // Caller-owned diagnostics remain usable after all emulator accounting is released.
        assert_eq!(error.launch_report().unwrap().completed_lanes, 0);
    }
}

#[test]
fn test_completion_metadata_is_retained_only_while_the_emulator_owns_it() {
    for asynchronous in [false, true] {
        let device = Emulator::create(EmulatorOptions::default()).unwrap();
        let budget = &device.client.shared.execution_budget;
        let (event, observation) = device.default_stream().record_event().unwrap();
        observation.observe(asynchronous).unwrap();
        assert!(budget.used() > 0);
        assert!(matches!(event.status(), Some(Ok(()))));
        drop(event);
        // Observation can finish before the worker drops its event producer.
        let deadline = Instant::now() + Duration::from_secs(5);
        while budget.used() != 0 && Instant::now() < deadline {
            thread::yield_now();
        }
        assert_eq!(budget.used(), 0);
        let completed = device.synchronize().unwrap();
        device.synchronize().unwrap().observe(asynchronous).unwrap();
        assert!(budget.used() > 0);
        drop(completed);
        assert_eq!(budget.used(), 0);
        device
            .shutdown(ShutdownMode::Drain)
            .observe(asynchronous)
            .unwrap();
    }
}

#[test]
fn test_fault_diagnostics_use_precharged_storage_when_the_budget_is_full() {
    for asynchronous in [false, true] {
        let device = Emulator::create(EmulatorOptions::default()).unwrap();
        let module = device
            .load(
                ParsedModule::parse(
                    ".version 9.3 .target sm_80 .address_size 64 .global .u32 output;
             .entry k() {st.global.u32 [output],23;trap;}",
                )
                .unwrap(),
            )
            .unwrap()
            .observe(asynchronous)
            .unwrap();
        let (started_tx, started_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        *lock(&device.client.shared.hooks.launch) = Some(Box::new(move |engine| {
            if engine.trace.len() == 1 {
                started_tx.send(()).unwrap();
                resume_rx.recv().unwrap();
            }
        }));
        let launch = module
            .kernel("k")
            .unwrap()
            .launch(LaunchConfig::create([1; 3], [1; 3], 0).unwrap(), [])
            .unwrap();
        started_rx.recv().unwrap();
        let budget = &device.client.shared.execution_budget;
        let full = budget
            .reserve(device.client.shared.options.execution_storage_bytes - budget.used())
            .unwrap();
        resume_tx.send(()).unwrap();
        let error = launch.observe(asynchronous).unwrap_err();
        assert!(matches!(error.root_cause(), ExecutionError::Trap));
        assert_eq!(error.launch_report().unwrap().instructions, 2);
        drop(full);
        assert_eq!(
            device
                .download(&module.global("output").unwrap())
                .unwrap()
                .observe(asynchronous)
                .unwrap(),
            23u32.to_le_bytes()
        );
        assert!(
            device
                .shutdown(ShutdownMode::Drain)
                .observe(asynchronous)
                .is_err()
        );
        assert_only_fault_storage(&device);
    }
}

#[test]
fn test_cancelled_active_launch_results_release_without_retaining_grid_storage() {
    for asynchronous in [false, true] {
        for abandon in [false, true] {
            let device = Emulator::create(EmulatorOptions::default()).unwrap();
            let module = device
                .load(empty_module())
                .unwrap()
                .observe(asynchronous)
                .unwrap();
            let (started_tx, started_rx) = mpsc::channel();
            let (resume_tx, resume_rx) = mpsc::channel();
            *lock(&device.client.shared.hooks.launch) = Some(Box::new(move |_| {
                started_tx.send(()).unwrap();
                resume_rx.recv().unwrap();
            }));
            let launch = module
                .kernel("k")
                .unwrap()
                .launch(LaunchConfig::create([1; 3], [1; 3], 0).unwrap(), [])
                .unwrap();
            started_rx.recv().unwrap();
            let shutdown = device.shutdown(ShutdownMode::Cancel);
            resume_tx.send(()).unwrap();
            shutdown.observe(asynchronous).unwrap();
            assert!(device.client.shared.execution_budget.used() > 0);
            if abandon {
                drop(launch);
            } else {
                let error = launch.observe(asynchronous).unwrap_err();
                assert!(matches!(error.root_cause(), ExecutionError::Cancelled));
                assert_eq!(error.launch_report().unwrap().instructions, 0);
            }
            assert_eq!(device.client.shared.execution_budget.used(), 0);
        }
    }
}

#[test]
fn test_queue_capacity_cancel_and_abandoned_work() {
    let device = Emulator::create(EmulatorOptions {
        queued_commands: 1,
        ..Default::default()
    })
    .unwrap();
    let (paused, release) = pause(&device);
    let queued = device.upload(&[1u32]).unwrap();
    assert!(matches!(
        device.zeroed::<u8>(1),
        Err(SubmitError::QueueFull)
    ));
    let drain = device.shutdown(ShutdownMode::Drain);
    let cancel = device.shutdown(ShutdownMode::Cancel);
    assert!(matches!(device.zeroed::<u8>(1), Err(SubmitError::Closed)));
    release.send(()).unwrap();
    paused.wait().unwrap();
    assert!(matches!(queued.wait(), Err(ExecutionError::Cancelled)));
    drain.wait().unwrap();
    cancel.wait().unwrap();
    assert_eq!(lock(&device.client.shared.admission).staged, 0);
}

#[test]
fn test_abandoned_resource_results_and_last_owner_drain() {
    let device = Emulator::create(EmulatorOptions {
        device_storage_bytes: 4096,
        ..Default::default()
    })
    .unwrap();
    let (paused, release) = pause(&device);
    drop(device.zeroed::<u8>(1200).unwrap());
    let live = device.zeroed::<u8>(1200).unwrap();
    release.send(()).unwrap();
    paused.wait().unwrap();
    let live = live.wait().unwrap();
    let completed = device.zeroed::<u8>(1).unwrap();
    // Fence through the queue while leaving the completed result unconsumed.
    device.load(empty_module()).unwrap().wait().unwrap();
    drop(completed);
    let observer = Shutdown::new(Arc::clone(&device.client.shared));
    drop(device);
    drop(live);
    observer.wait().unwrap();
}

#[test]
fn test_worker_unwind_resolves_current_and_queued_submissions() {
    let device = Emulator::create(EmulatorOptions::default()).unwrap();
    let (paused, release) = pause(&device);
    let breakpoint = device.next_breakpoint().unwrap();
    let crash = submit_to::<(), _, _>(&device.client, 0, false, 0, |_| {
        |_| panic!("injected worker failure")
    })
    .unwrap();
    let queued = device.upload(&[7u32]).unwrap();
    release.send(()).unwrap();
    paused.wait().unwrap();
    assert!(matches!(crash.wait(), Err(ExecutionError::WorkerFailed)));
    assert!(matches!(queued.wait(), Err(ExecutionError::WorkerFailed)));
    assert!(matches!(
        breakpoint.wait(),
        Err(ExecutionError::WorkerFailed)
    ));
    assert!(matches!(
        device.shutdown(ShutdownMode::Drain).wait(),
        Err(ExecutionError::WorkerFailed)
    ));
    assert_eq!(lock(&device.client.shared.admission).staged, 0);
}

#[test]
fn test_abandoning_a_published_breakpoint_releases_its_result_and_keeps_resume_available() {
    struct Notify(Sender<()>);

    impl Wake for Notify {
        fn wake(self: Arc<Self>) {
            self.0.send(()).unwrap();
        }
    }
    let device = Emulator::create(EmulatorOptions::default()).unwrap();
    let module = device
        .load(
            ParsedModule::parse(
                ".version 9.3 .target sm_80 .address_size 64
        .entry k() {brkpt;ret;}",
            )
            .unwrap(),
        )
        .unwrap()
        .wait()
        .unwrap();
    let (notified_tx, notified_rx) = mpsc::channel();
    let mut observation = Box::pin(device.next_breakpoint().unwrap());
    let waker = Waker::from(Arc::new(Notify(notified_tx)));
    assert!(
        observation
            .as_mut()
            .poll(&mut Context::from_waker(&waker))
            .is_pending()
    );
    let launch = module
        .kernel("k")
        .unwrap()
        .launch(LaunchConfig::create([1; 3], [1; 3], 0).unwrap(), [])
        .unwrap();
    notified_rx.recv().unwrap();
    drop(observation);
    let ids = device.paused_breakpoints();
    assert_eq!(ids.len(), 1);
    device.resume(&ids[0]).unwrap().wait().unwrap();
    launch.wait().unwrap();
    device.shutdown(ShutdownMode::Drain).wait().unwrap();
    assert_eq!(device.client.shared.execution_budget.used(), 0);
}

#[test]
fn test_queued_resources_do_not_keep_public_ownership() {
    let device = Emulator::create(EmulatorOptions::default()).unwrap();
    let buffer = device.upload(&[1u32, 2]).unwrap().wait().unwrap();
    let (paused, release) = pause(&device);
    drop(device.write(&buffer, &[3, 4]).unwrap());
    let download = device.download(&buffer).unwrap();
    let observer = Shutdown::new(Arc::clone(&device.client.shared));
    drop(buffer);
    drop(device);
    drop(paused);
    release.send(()).unwrap();
    assert_eq!(download.wait().unwrap(), [3, 4]);
    observer.wait().unwrap();
}

struct CountWake(atomic::AtomicUsize);

impl Wake for CountWake {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn test_shutdown_waker_clone_and_drop_can_reenter_admission() {
    use std::{
        mem::ManuallyDrop,
        sync::atomic::AtomicUsize,
        task::{RawWaker, RawWakerVTable},
    };

    struct Probe {
        shared: Arc<Shared>,
        clones: AtomicUsize,
        drops: AtomicUsize,
    }

    impl Probe {
        fn reenter(&self) {
            // The worker is paused, so only the callback could be holding admission.
            drop(
                self.shared
                    .admission
                    .try_lock()
                    .expect("admission held during waker callback"),
            );
            self.shared.shutdown(ShutdownMode::Drain);
        }
    }

    // Each raw waker owns one Arc; cloning borrows it and creates another owner.
    unsafe fn clone_waker(data: *const ()) -> RawWaker {
        // SAFETY: the waker data is an `Arc` leaked by this vtable; `ManuallyDrop` keeps its count unchanged.
        let probe = ManuallyDrop::new(unsafe { Arc::<Probe>::from_raw(data.cast()) });
        probe.reenter();
        probe.clones.fetch_add(1, Ordering::SeqCst);
        RawWaker::new(Arc::into_raw(Arc::clone(&probe)).cast(), &VTABLE)
    }

    unsafe fn drop_waker(data: *const ()) {
        // SAFETY: this waker owns one Arc created by into_raw; from_raw consumes that ownership.
        let probe = unsafe { Arc::<Probe>::from_raw(data.cast()) };
        probe.reenter();
        probe.drops.fetch_add(1, Ordering::SeqCst);
    }

    unsafe fn wake_by_ref(data: *const ()) {
        // SAFETY: the waker data is an `Arc` leaked by this vtable; `ManuallyDrop` keeps its count unchanged.
        let probe = ManuallyDrop::new(unsafe { Arc::<Probe>::from_raw(data.cast()) });
        probe.reenter();
    }

    static VTABLE: RawWakerVTable =
        RawWakerVTable::new(clone_waker, drop_waker, wake_by_ref, drop_waker);

    let device = Emulator::create(EmulatorOptions::default()).unwrap();
    let (paused, release) = pause(&device);
    let mut shutdown = Box::pin(device.shutdown(ShutdownMode::Drain));
    let probe = Arc::new(Probe {
        shared: Arc::clone(&device.client.shared),
        clones: 0.into(),
        drops: 0.into(),
    });
    let raw = RawWaker::new(Arc::into_raw(Arc::clone(&probe)).cast(), &VTABLE);
    // SAFETY: raw owns one Arc<Probe>, and VTABLE clones or releases that ownership on each callback.
    let waker = unsafe { Waker::from_raw(raw) };
    for _ in 0..2 {
        assert!(
            shutdown
                .as_mut()
                .poll(&mut Context::from_waker(&waker))
                .is_pending()
        );
    }
    drop(shutdown);
    drop(waker);
    assert_eq!(probe.clones.load(Ordering::SeqCst), 2);
    assert_eq!(probe.drops.load(Ordering::SeqCst), 3);
    assert!(
        lock(&device.client.shared.admission)
            .shutdown_wakers
            .is_empty()
    );
    release.send(()).unwrap();
    paused.wait().unwrap();
    device.shutdown(ShutdownMode::Drain).wait().unwrap();
}

#[test]
fn test_shutdown_replaces_wakers_and_unregisters_abandoned_observers() {
    struct Notify {
        sender: Sender<u8>,
        id: u8,
        device: Emulator,
    }

    impl Wake for Notify {
        fn wake(self: Arc<Self>) {
            assert!(matches!(
                self.device.create_stream(),
                Err(SubmitError::Closed)
            ));
            self.sender.send(self.id).unwrap();
        }
    }
    let device = Emulator::create(EmulatorOptions::default()).unwrap();
    let (paused, release) = pause(&device);
    let mut first = Box::pin(device.shutdown(ShutdownMode::Drain));
    let mut second = Box::pin(device.shutdown(ShutdownMode::Drain));
    let obsolete = Arc::new(CountWake(0.into()));
    {
        let waker = Waker::from(Arc::clone(&obsolete));
        assert!(
            first
                .as_mut()
                .poll(&mut Context::from_waker(&waker))
                .is_pending()
        );
    }
    let (wake_tx, wake_rx) = mpsc::channel();
    let first_waker = Waker::from(Arc::new(Notify {
        sender: wake_tx.clone(),
        id: 1,
        device: device.clone(),
    }));
    let second_waker = Waker::from(Arc::new(Notify {
        sender: wake_tx,
        id: 2,
        device: device.clone(),
    }));
    for _ in 0..32 {
        assert!(
            first
                .as_mut()
                .poll(&mut Context::from_waker(&first_waker))
                .is_pending()
        );
    }
    assert_eq!(Arc::strong_count(&obsolete), 1);
    assert!(
        second
            .as_mut()
            .poll(&mut Context::from_waker(&second_waker))
            .is_pending()
    );
    let mut abandoned = Box::pin(device.shutdown(ShutdownMode::Drain));
    let abandoned_wake = Arc::new(CountWake(0.into()));
    {
        let waker = Waker::from(Arc::clone(&abandoned_wake));
        assert!(
            abandoned
                .as_mut()
                .poll(&mut Context::from_waker(&waker))
                .is_pending()
        );
    }
    drop(abandoned);
    assert_eq!(Arc::strong_count(&abandoned_wake), 1);
    assert_eq!(
        lock(&device.client.shared.admission).shutdown_wakers.len(),
        2
    );
    release.send(()).unwrap();
    paused.wait().unwrap();
    Pin::into_inner(first).wait().unwrap();
    Pin::into_inner(second).observe(true).unwrap();
    let mut notifications = [wake_rx.recv().unwrap(), wake_rx.recv().unwrap()];
    notifications.sort_unstable();
    assert_eq!(notifications, [1, 2]);
    assert_eq!(obsolete.0.load(Ordering::SeqCst), 0);
    assert_eq!(abandoned_wake.0.load(Ordering::SeqCst), 0);
    assert!(
        lock(&device.client.shared.admission)
            .shutdown_wakers
            .is_empty()
    );
}

#[test]
fn test_shutdown_registration_races_never_lose_a_wakeup() {
    struct Notify(Sender<()>);

    impl Wake for Notify {
        fn wake(self: Arc<Self>) {
            self.0.send(()).unwrap();
        }
    }
    for iteration in 0..30 {
        let device = Emulator::create(EmulatorOptions::default()).unwrap();
        let (blocker, release) = pause(&device);
        let observer = device.shutdown(ShutdownMode::Drain);
        let other = device.shutdown(ShutdownMode::Drain);
        let start = Arc::new(Barrier::new(2));
        let observer_start = Arc::clone(&start);
        let (wake_tx, wake_rx) = mpsc::channel();
        let observer = thread::spawn(move || {
            let waker = Waker::from(Arc::new(Notify(wake_tx)));
            let mut observer = Box::pin(observer);
            observer_start.wait();
            let pending = observer
                .as_mut()
                .poll(&mut Context::from_waker(&waker))
                .is_pending();
            Pin::into_inner(observer).wait().unwrap();
            pending
        });
        start.wait();
        if iteration % 3 == 0 {
            drop(device.shutdown(ShutdownMode::Cancel));
        }
        release.send(()).unwrap();
        blocker.wait().unwrap();
        other.observe(iteration % 2 == 0).unwrap();
        if observer.join().unwrap() {
            wake_rx.recv().unwrap();
        }
        let mut late = pin::pin!(device.shutdown(ShutdownMode::Drain));
        assert!(matches!(
            late.as_mut().poll(&mut Context::from_waker(Waker::noop())),
            Poll::Ready(Ok(()))
        ));
    }
}

#[test]
fn test_completion_before_poll_and_waker_replacement() {
    let cell = Completion::new();
    let first = Arc::new(CountWake(0.into()));
    let second = Arc::new(CountWake(0.into()));
    assert!(
        cell.poll(&Context::from_waker(&Waker::from(Arc::clone(&first))))
            .is_pending()
    );
    assert!(
        cell.poll(&Context::from_waker(&Waker::from(Arc::clone(&second))))
            .is_pending()
    );
    cell.publish(Ok(42));
    assert_eq!(first.0.load(Ordering::SeqCst), 0);
    assert_eq!(second.0.load(Ordering::SeqCst), 1);
    assert_eq!(cell.take().unwrap(), 42);
    let cell = Completion::new();
    cell.publish(Ok(7));
    assert!(matches!(
        cell.poll(&Context::from_waker(Waker::noop())),
        Poll::Ready(Ok(7))
    ));
}

#[test]
fn test_completion_releases_snapshot_capacity_before_a_waker_submits_more_work() {
    struct Resubmit {
        device: Emulator,
        output: Mutex<Option<Result<Submission<Buffer<u8>>, SubmitError>>>,
        ready: Sender<()>,
    }

    impl Wake for Resubmit {
        fn wake(self: Arc<Self>) {
            *lock(&self.output) = Some(self.device.upload(&[0u8; 16]));
            self.ready.send(()).unwrap();
        }
    }
    let device = Emulator::create(EmulatorOptions {
        staging_storage_bytes: 16,
        ..Default::default()
    })
    .unwrap();
    let (paused, release) = pause(&device);
    let completion = submit_poll::<(), _, _>(
        &device.client,
        0,
        CommandKind::Immediate,
        false,
        16,
        |staging| {
            move |_: &mut Worker, terminal| {
                let _held = &staging;
                Poll::Ready(terminal.map_or_else(|| Ok(Box::new(|_| ()) as Pending<()>), Err))
            }
        },
    )
    .unwrap();
    let (ready_tx, ready_rx) = mpsc::channel();
    let observer = Arc::new(Resubmit {
        device: device.clone(),
        output: Mutex::new(None),
        ready: ready_tx,
    });
    let waker = Waker::from(Arc::clone(&observer));
    let mut completion = Box::pin(completion);
    assert!(
        completion
            .as_mut()
            .poll(&mut Context::from_waker(&waker))
            .is_pending()
    );
    release.send(()).unwrap();
    paused.wait().unwrap();
    ready_rx.recv().unwrap();
    Pin::into_inner(completion).wait().unwrap();
    let output = lock(&observer.output)
        .take()
        .unwrap()
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(device.download(&output).unwrap().wait().unwrap(), [0; 16]);
    device.shutdown(ShutdownMode::Drain).wait().unwrap();
}

#[test]
fn test_completion_registration_races_never_lose_a_wakeup() {
    for _ in 0..100 {
        let cell = Arc::new(Completion::new());
        let start = Arc::new(Barrier::new(2));
        let producer = Arc::clone(&cell);
        let producer_start = Arc::clone(&start);
        let thread = thread::spawn(move || {
            producer_start.wait();
            producer.publish(Ok(11));
        });
        let wake = Arc::new(CountWake(0.into()));
        start.wait();
        let result = cell.poll(&Context::from_waker(&Waker::from(Arc::clone(&wake))));
        thread.join().unwrap();
        match result {
            Poll::Ready(value) => assert_eq!(value.unwrap(), 11),
            Poll::Pending => {
                assert_eq!(wake.0.load(Ordering::SeqCst), 1);
                assert_eq!(cell.take().unwrap(), 11);
            }
        }
    }
}

#[test]
fn test_shutdown_and_admission_share_an_ordering_point() {
    for _ in 0..50 {
        let device = Emulator::create(EmulatorOptions::default()).unwrap();
        let other = device.clone();
        let start = Arc::new(Barrier::new(2));
        let second = Arc::clone(&start);
        let host = thread::spawn(move || {
            second.wait();
            other.upload(&[3u8])
        });
        start.wait();
        let shutdown = device.shutdown(ShutdownMode::Drain);
        match host.join().unwrap() {
            Ok(submission) => {
                assert_eq!(submission.wait().unwrap().len(), 1);
            }
            Err(error) => assert!(matches!(error, SubmitError::Closed)),
        }
        shutdown.wait().unwrap();
    }
}

#[test]
fn test_handles_are_send_sync_and_submissions_are_send() {
    fn send_sync<T: Send + Sync>() {}

    fn send<T: Send>() {}
    send_sync::<Emulator>();
    send_sync::<Module>();
    send_sync::<Kernel>();
    send_sync::<Stream>();
    send_sync::<Event>();
    send_sync::<Buffer<u64>>();
    send_sync::<BufferView<f32>>();
    send::<Submission<Buffer<u32>>>();
}

#[test]
fn test_queued_stream_commands_wait_for_a_launch_slot() {
    for asynchronous in [false, true] {
        for cancel in [false, true] {
            let device = Emulator::create(EmulatorOptions {
                active_grids: 1,
                ..Default::default()
            })
            .unwrap();
            let module = device
                .load(
                    ParsedModule::parse(
                        ".version 9.3 .target sm_80 .address_size 64
                    .global .u32 output;
                    .entry hold() {brkpt;ret;}
                    .entry write() {st.global.u32 [output],42;ret;}",
                    )
                    .unwrap(),
                )
                .unwrap()
                .observe(asynchronous)
                .unwrap();
            let holding = device.create_stream().unwrap();
            let queued = device.create_stream().unwrap();
            let independent = device.create_stream().unwrap();
            let config = LaunchConfig::create([1; 3], [1; 3], 0).unwrap();
            let notification = device.next_breakpoint().unwrap();
            let hold = holding
                .launch(&module.kernel("hold").unwrap(), config, [])
                .unwrap();
            let breakpoint = notification.observe(asynchronous).unwrap();
            let launch = queued
                .launch(&module.kernel("write").unwrap(), config, [])
                .unwrap();
            let download = queued.download(&module.global("output").unwrap()).unwrap();
            let (event, observation) = queued.record_event().unwrap();
            let fence = queued.synchronize().unwrap();
            independent
                .synchronize()
                .unwrap()
                .observe(asynchronous)
                .unwrap();
            assert!(!launch.completion.is_published());
            assert!(!download.completion.is_published());
            assert!(event.status().is_none());
            assert!(!fence.completion.is_published());
            if cancel {
                let shutdown = device.shutdown(ShutdownMode::Cancel);
                for result in [hold.observe(asynchronous), launch.observe(asynchronous)] {
                    assert!(matches!(
                        result.unwrap_err().root_cause(),
                        ExecutionError::Cancelled
                    ));
                }
                assert!(matches!(
                    download.observe(asynchronous),
                    Err(ExecutionError::Cancelled)
                ));
                assert!(matches!(
                    observation.observe(asynchronous),
                    Err(ExecutionError::Cancelled)
                ));
                assert!(matches!(
                    fence.observe(asynchronous),
                    Err(ExecutionError::Cancelled)
                ));
                shutdown.observe(asynchronous).unwrap();
            } else {
                device
                    .resume(&breakpoint.id)
                    .unwrap()
                    .observe(asynchronous)
                    .unwrap();
                hold.observe(asynchronous).unwrap();
                launch.observe(asynchronous).unwrap();
                assert_eq!(download.observe(asynchronous).unwrap(), 42u32.to_le_bytes());
                observation.observe(asynchronous).unwrap();
                fence.observe(asynchronous).unwrap();
                device
                    .shutdown(ShutdownMode::Drain)
                    .observe(asynchronous)
                    .unwrap();
            }
        }
    }
}

#[test]
fn test_streams_interleave_grids_and_record_events_before_they_signal() {
    let device = Emulator::create(EmulatorOptions::default()).unwrap();
    let module = device.load(ParsedModule::parse(
        ".version 9.3 .target sm_80 .address_size 64 .global .u32 flag; .global .u32 output;
        .entry wait() { .shared .u32 scratch; .reg .u32 r; .reg .pred p;
            st.shared.u32 [scratch],17;
            loop: ld.global.u32 r,[flag]; setp.eq.u32 p,r,0; @p bra loop;
            ld.shared.u32 r,[scratch]; st.global.u32 [output],r; ret; }
        .entry signal() { .shared .u32 scratch; st.shared.u32 [scratch],99; st.global.u32 [flag],1; ret; }"
    ).unwrap()).unwrap().wait().unwrap();
    let a = device.create_stream().unwrap();
    let b = device.create_stream().unwrap();
    let (started_tx, started_rx) = mpsc::channel();
    let (resume_tx, resume_rx) = mpsc::channel();
    let mut paused = false;
    *lock(&device.client.shared.hooks.launch) = Some(Box::new(move |_| {
        if !paused {
            paused = true;
            started_tx.send(()).unwrap();
            resume_rx.recv().unwrap();
        }
    }));
    let config = LaunchConfig::create([1, 1, 1], [1, 1, 1], 0).unwrap();
    let waiting = a
        .launch(&module.kernel("wait").unwrap(), config, [])
        .unwrap();
    started_rx.recv().unwrap();
    let (event, observation) = a.record_event().unwrap();
    assert!(event.status().is_none());
    drop(observation);
    let signaling = b
        .launch(&module.kernel("signal").unwrap(), config, [])
        .unwrap();
    let dependency = b.wait_event(&event).unwrap();
    let output = b.download(&module.global("output").unwrap()).unwrap();
    resume_tx.send(()).unwrap();
    let first = waiting.wait().unwrap();
    let second = signaling.wait().unwrap();
    assert_ne!(first.launch_id, second.launch_id);
    dependency.wait().unwrap();
    assert_eq!(output.wait().unwrap(), 17u32.to_le_bytes());
    assert!(matches!(event.status(), Some(Ok(()))));
    device.shutdown(ShutdownMode::Drain).wait().unwrap();
}

#[test]
fn test_stream_dependencies_preserve_snapshots_and_reject_foreign_handles() {
    let device = Emulator::create(EmulatorOptions::default()).unwrap();
    let data = device.upload(&[0u32]).unwrap().wait().unwrap();
    let a = device.create_stream().unwrap();
    let b = device.create_stream().unwrap();
    let (paused, release) = pause(&device);
    let mut input = [41u32];
    drop(a.write(&data, &input).unwrap());
    input[0] = 99;
    let (event, observation) = a.record_event().unwrap();
    let wait = b.wait_event(&event).unwrap();
    let output = b.download(&data).unwrap();
    let foreign = Emulator::create(EmulatorOptions::default()).unwrap();
    assert!(matches!(
        foreign.default_stream().wait_event(&event),
        Err(SubmitError::ForeignResource)
    ));
    assert!(matches!(
        foreign.default_stream().download(&data),
        Err(SubmitError::ForeignResource)
    ));
    release.send(()).unwrap();
    paused.wait().unwrap();
    observation.wait().unwrap();
    wait.wait().unwrap();
    assert_eq!(output.wait().unwrap(), [41]);
    assert_eq!(input, [99]);
    device.shutdown(ShutdownMode::Drain).wait().unwrap();
    foreign.shutdown(ShutdownMode::Drain).wait().unwrap();
}

#[test]
fn test_cancellation_reaches_multiple_active_grids_events_and_a_full_queue() {
    let device = Emulator::create(EmulatorOptions {
        queued_commands: 3,
        ..Default::default()
    })
    .unwrap();
    let module = device
        .load(
            ParsedModule::parse(
                ".version 9.3 .target sm_80 .address_size 64 .entry k() { again: bra again; }",
            )
            .unwrap(),
        )
        .unwrap()
        .wait()
        .unwrap();
    let a = device.create_stream().unwrap();
    let b = device.create_stream().unwrap();
    let c = device.create_stream().unwrap();
    let (paused, release) = pause(&device);
    let (started_tx, started_rx) = mpsc::channel();
    let (resume_tx, resume_rx) = mpsc::channel();
    let mut reached = BTreeSet::new();
    let mut stopped = false;
    *lock(&device.client.shared.hooks.launch) = Some(Box::new(move |engine| {
        if !engine.trace.is_empty() {
            reached.insert(engine.launch_id());
        }
        if reached.len() == 2 && !stopped {
            stopped = true;
            started_tx.send(()).unwrap();
            resume_rx.recv().unwrap();
        }
    }));
    let config = LaunchConfig::create([1, 1, 1], [1, 1, 1], 0).unwrap();
    let first = a.launch(&module.kernel("k").unwrap(), config, []).unwrap();
    let second = b.launch(&module.kernel("k").unwrap(), config, []).unwrap();
    let first_id = first.launch_id().unwrap();
    let second_id = second.launch_id().unwrap();
    let (event, marker) = a.record_event().unwrap();
    release.send(()).unwrap();
    paused.wait().unwrap();
    started_rx.recv().unwrap();
    let dependency = c.wait_event(&event).unwrap();
    let queued = device.upload(&[7u32]).unwrap();
    assert!(matches!(
        device.zeroed::<u8>(1),
        Err(SubmitError::QueueFull)
    ));
    let draining = device.shutdown(ShutdownMode::Drain);
    let cancelling = device.shutdown(ShutdownMode::Cancel);
    resume_tx.send(()).unwrap();
    for (submission, id) in [(first, first_id), (second, second_id)] {
        let error = submission.wait().unwrap_err();
        assert!(matches!(error.root_cause(), ExecutionError::Cancelled));
        let report = error.launch_report().unwrap();
        assert_eq!(report.launch_id, id);
        assert!(report.instructions > 0);
    }
    assert!(matches!(marker.wait(), Err(ExecutionError::Cancelled)));
    assert!(matches!(dependency.wait(), Err(ExecutionError::Cancelled)));
    assert!(matches!(queued.wait(), Err(ExecutionError::Cancelled)));
    assert!(matches!(
        event.status(),
        Some(Err(ExecutionError::Cancelled))
    ));
    draining.wait().unwrap();
    cancelling.wait().unwrap();
    assert_eq!(lock(&device.client.shared.admission).staged, 0);
}

#[test]
fn test_chunked_upload_snapshots_inputs_and_yields_to_an_independent_grid() {
    use std::sync::atomic::AtomicUsize;
    let device = Emulator::create(EmulatorOptions::default()).unwrap();
    let module = device.load(empty_module()).unwrap().wait().unwrap();
    let transfer = device.create_stream().unwrap();
    let compute = device.create_stream().unwrap();
    let completed = Arc::new(AtomicUsize::new(0));
    let progress = Arc::clone(&completed);
    let (started_tx, started_rx) = mpsc::channel();
    let (resume_tx, resume_rx) = mpsc::channel();
    let mut paused = false;
    *lock(&device.client.shared.hooks.transfer) = Some(Box::new(move |copied, total| {
        progress.store(copied, Ordering::Release);
        if !paused && copied < total {
            paused = true;
            started_tx.send(()).unwrap();
            resume_rx.recv().unwrap();
        }
    }));
    let observed = Arc::new(AtomicBool::new(false));
    let observed_worker = Arc::clone(&observed);
    let progress = Arc::clone(&completed);
    let total = CHUNK_BYTES * 8;
    *lock(&device.client.shared.hooks.launch) = Some(Box::new(move |_| {
        if progress.load(Ordering::Acquire) < total {
            observed_worker.store(true, Ordering::Release);
        }
    }));
    let mut input = vec![17u8; total];
    let upload = transfer.upload(&input).unwrap();
    started_rx.recv().unwrap();
    input.fill(99);
    let mut upload = Box::pin(upload);
    assert!(
        upload
            .as_mut()
            .poll(&mut Context::from_waker(Waker::noop()))
            .is_pending()
    );
    assert_eq!(completed.load(Ordering::Acquire), CHUNK_BYTES);
    let launch = compute
        .launch(
            &module.kernel("k").unwrap(),
            LaunchConfig::create([1, 1, 1], [1, 1, 1], 0).unwrap(),
            [],
        )
        .unwrap();
    resume_tx.send(()).unwrap();
    let buffer = Pin::into_inner(upload).wait().unwrap();
    launch.wait().unwrap();
    assert!(observed.load(Ordering::Acquire));
    *lock(&device.client.shared.hooks.transfer) = None;
    assert_eq!(
        transfer.download(&buffer).unwrap().wait().unwrap(),
        vec![17; total]
    );
    device.shutdown(ShutdownMode::Drain).wait().unwrap();
    assert_eq!(device.client.shared.execution_budget.used(), 0);
}

#[test]
fn test_another_stream_fault_stops_a_transfer_and_preserves_completed_chunks() {
    use std::sync::atomic::AtomicUsize;
    let device = Emulator::create(EmulatorOptions::default()).unwrap();
    let module = device
        .load(
            ParsedModule::parse(".version 9.3 .target sm_80 .address_size 64 .entry k() {trap;}")
                .unwrap(),
        )
        .unwrap()
        .wait()
        .unwrap();
    let total = CHUNK_BYTES * 8;
    let buffer = device.zeroed::<u8>(total).unwrap().wait().unwrap();
    let transfer = device.create_stream().unwrap();
    let compute = device.create_stream().unwrap();
    let progress = Arc::new(AtomicUsize::new(0));
    let copied = Arc::clone(&progress);
    let (started_tx, started_rx) = mpsc::channel();
    let (resume_tx, resume_rx) = mpsc::channel();
    let mut paused = false;
    *lock(&device.client.shared.hooks.transfer) = Some(Box::new(move |completed, total| {
        copied.store(completed, Ordering::Release);
        if !paused && completed < total {
            paused = true;
            started_tx.send(()).unwrap();
            resume_rx.recv().unwrap();
        }
    }));
    let write = transfer.write(&buffer, &vec![71; total]).unwrap();
    started_rx.recv().unwrap();
    let fault = compute
        .launch(
            &module.kernel("k").unwrap(),
            LaunchConfig::create([1, 1, 1], [1, 1, 1], 0).unwrap(),
            [],
        )
        .unwrap();
    resume_tx.send(()).unwrap();
    assert!(matches!(
        fault.wait().unwrap_err().root_cause(),
        ExecutionError::Trap
    ));
    assert!(matches!(write.wait(), Err(ExecutionError::Faulted)));
    let written = progress.load(Ordering::Acquire);
    assert!(written > 0 && written < total);
    *lock(&device.client.shared.hooks.transfer) = None;
    let bytes = device.download(&buffer).unwrap().wait().unwrap();
    assert!(bytes[..written].iter().all(|byte| *byte == 71));
    assert!(bytes[written..].iter().all(|byte| *byte == 0));
    assert!(device.shutdown(ShutdownMode::Drain).wait().is_err());
    assert_only_fault_storage(&device);
}

#[test]
fn test_execution_budget_is_reserved_before_building_a_command_snapshot() {
    let device = Emulator::create(EmulatorOptions {
        execution_storage_bytes: 0,
        ..Default::default()
    })
    .unwrap();
    let copied = Arc::new(AtomicBool::new(false));
    let observe_copy = Arc::clone(&copied);
    let result = submit_to::<(), _, _>(&device.client, 0, false, 0, move |_| {
        observe_copy.store(true, Ordering::Release);
        |_| Ok(Box::new(|_| ()) as Pending<()>)
    });
    assert!(matches!(result, Err(SubmitError::ExecutionStateLimit)));
    assert!(!copied.load(Ordering::Acquire));
    device.shutdown(ShutdownMode::Drain).wait().unwrap();
}

#[test]
fn test_initial_grid_construction_yields_without_executing_and_cancels_with_a_full_queue() {
    let device = Emulator::create(EmulatorOptions {
        queued_commands: 2,
        ..Default::default()
    })
    .unwrap();
    let module = device
        .load(
            ParsedModule::parse(
                ".version 9.3 .target sm_80 .address_size 64
         .global .u32 value; .entry k() { st.global.u32 [value], 1; ret; }",
            )
            .unwrap(),
        )
        .unwrap()
        .wait()
        .unwrap();
    let compute = device.create_stream().unwrap();
    let transfer = device.create_stream().unwrap();
    let (checkpoint_tx, checkpoint_rx) = mpsc::channel();
    let (resume_tx, resume_rx) = mpsc::channel();
    let mut turns = 0;
    *lock(&device.client.shared.hooks.initialization) = Some(Box::new(move |_| {
        turns += 1;
        if turns == 1 || turns == 16 {
            checkpoint_tx.send(turns).unwrap();
            resume_rx.recv().unwrap();
        }
    }));
    let launch = compute
        .launch(
            &module.kernel("k").unwrap(),
            LaunchConfig::create([1, 1, 1], [512, 1, 1], 0).unwrap(),
            [],
        )
        .unwrap();
    assert_eq!(checkpoint_rx.recv().unwrap(), 1);
    let uploaded = transfer.upload(&[7u32]).unwrap();
    let untouched = transfer.download(&module.global("value").unwrap()).unwrap();
    resume_tx.send(()).unwrap();
    assert_eq!(checkpoint_rx.recv().unwrap(), 16);
    let buffer = uploaded.wait().unwrap();
    assert_eq!(untouched.wait().unwrap(), [0; 4]);
    let first = compute.synchronize().unwrap();
    let second = compute.synchronize().unwrap();
    assert!(matches!(compute.synchronize(), Err(SubmitError::QueueFull)));
    let shutdown = device.shutdown(ShutdownMode::Cancel);
    resume_tx.send(()).unwrap();
    let error = launch.wait().unwrap_err();
    assert!(matches!(error, ExecutionError::Cancelled));
    assert!(error.launch_report().is_none());
    assert!(matches!(first.wait(), Err(ExecutionError::Cancelled)));
    assert!(matches!(second.wait(), Err(ExecutionError::Cancelled)));
    drop(buffer);
    shutdown.wait().unwrap();
    assert_eq!(device.client.shared.execution_budget.used(), 0);
    assert_eq!(lock(&device.client.shared.admission).staged, 0);
}

#[test]
fn test_breakpoint_publication_wakes_outside_admission_and_cancel_releases_control_storage() {
    struct Reenter {
        shared: Weak<Shared>,
        unlocked: AtomicBool,
        finished: Sender<()>,
    }

    impl Wake for Reenter {
        fn wake(self: Arc<Self>) {
            self.unlocked.store(
                self.shared.upgrade().unwrap().admission.try_lock().is_ok(),
                Ordering::Release,
            );
            self.finished.send(()).unwrap();
        }
    }
    let device = Emulator::create(EmulatorOptions {
        queued_commands: 1,
        ..Default::default()
    })
    .unwrap();
    let module = device
        .load(
            ParsedModule::parse(
                ".version 9.3 .target sm_80 .address_size 64
        .entry k() {brkpt;ret;}",
            )
            .unwrap(),
        )
        .unwrap()
        .wait()
        .unwrap();
    let (paused, release) = pause(&device);
    let mut observation = Box::pin(device.next_breakpoint().unwrap());
    let (finished_tx, finished_rx) = mpsc::channel();
    let wake = Arc::new(Reenter {
        shared: Arc::downgrade(&device.client.shared),
        unlocked: AtomicBool::new(false),
        finished: finished_tx,
    });
    let waker = Waker::from(Arc::clone(&wake));
    assert!(
        observation
            .as_mut()
            .poll(&mut Context::from_waker(&waker))
            .is_pending()
    );
    let launch = module
        .kernel("k")
        .unwrap()
        .launch(LaunchConfig::create([1; 3], [1; 3], 0).unwrap(), [])
        .unwrap();
    release.send(()).unwrap();
    finished_rx.recv().unwrap();
    paused.wait().unwrap();
    let breakpoint = Pin::into_inner(observation).wait().unwrap();
    // Publication finishes the waker before another independent queue command can execute.
    let diagnostic = device.create_stream().unwrap();
    diagnostic.synchronize().unwrap().wait().unwrap();
    assert!(wake.unlocked.load(Ordering::Acquire));
    let queued = device.synchronize().unwrap();
    assert!(matches!(device.synchronize(), Err(SubmitError::QueueFull)));
    let unused = device.next_breakpoint().unwrap();
    let shutdown = device.shutdown(ShutdownMode::Cancel);
    assert!(matches!(
        launch.wait().unwrap_err().root_cause(),
        ExecutionError::Cancelled
    ));
    assert!(matches!(queued.wait(), Err(ExecutionError::Cancelled)));
    assert!(matches!(unused.wait(), Err(ExecutionError::Cancelled)));
    shutdown.wait().unwrap();
    assert_eq!(device.client.shared.execution_budget.used(), 0);
    drop(breakpoint);
}

#[test]
fn test_owner_drop_racing_with_a_new_breakpoint_does_not_leave_an_orphaned_worker() {
    let device = Emulator::create(EmulatorOptions::default()).unwrap();
    let module = device
        .load(
            ParsedModule::parse(
                ".version 9.3 .target sm_80 .address_size 64
        .entry k() {brkpt;ret;}",
            )
            .unwrap(),
        )
        .unwrap()
        .wait()
        .unwrap();
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    *lock(&device.client.shared.hooks.launch) = Some(Box::new(move |_| {
        started_tx.send(()).unwrap();
        release_rx.recv().unwrap();
    }));
    let launch = module
        .kernel("k")
        .unwrap()
        .launch(LaunchConfig::create([1; 3], [1; 3], 0).unwrap(), [])
        .unwrap();
    started_rx.recv().unwrap();
    let shutdown = Shutdown::new(Arc::clone(&device.client.shared));
    let shared = Arc::clone(&device.client.shared);
    drop(launch);
    drop(module);
    drop(device);
    release_tx.send(()).unwrap();
    shutdown.wait().unwrap();
    assert_eq!(shared.execution_budget.used(), 0);
}

#[test]
fn test_breakpoint_notification_exhaustion_faults_at_the_issuer_and_preserves_writes() {
    let device = Emulator::create(EmulatorOptions::default()).unwrap();
    let module = device
        .load(
            ParsedModule::parse(
                ".version 9.3 .target sm_80 .address_size 64
        .global .u32 value; .entry k() {st.global.u32 [value],7; brkpt; ret;}",
            )
            .unwrap(),
        )
        .unwrap()
        .wait()
        .unwrap();
    let observer = device.next_breakpoint().unwrap();
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let mut turns = 0;
    *lock(&device.client.shared.hooks.launch) = Some(Box::new(move |_| {
        turns += 1;
        if turns == 2 {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        }
    }));
    let launch = module
        .kernel("k")
        .unwrap()
        .launch(LaunchConfig::create([1; 3], [1; 3], 0).unwrap(), [])
        .unwrap();
    started_rx.recv().unwrap();
    let budget = &device.client.shared.execution_budget;
    let exhausted = budget
        .reserve(device.client.shared.options.execution_storage_bytes - budget.used())
        .unwrap();
    release_tx.send(()).unwrap();
    let error = launch.wait().unwrap_err();
    assert!(matches!(
        error.root_cause(),
        ExecutionError::ExecutionStateLimit
    ));
    let ExecutionError::Launch { cause, .. } = error else {
        panic!("missing launch report")
    };
    let ExecutionError::Lane { location, .. } = cause.as_ref() else {
        panic!("missing issuer location")
    };
    assert_eq!(location.function, "k");
    assert_eq!(location.instruction, 1);
    assert!(matches!(observer.wait(), Err(ExecutionError::Faulted)));
    drop(exhausted);
    assert_eq!(
        device
            .download(&module.global("value").unwrap())
            .unwrap()
            .wait()
            .unwrap(),
        7u32.to_le_bytes()
    );
    assert!(device.shutdown(ShutdownMode::Drain).wait().is_err());
    // The closed device retains its shared fault diagnostic, not worker execution storage.
    assert_only_fault_storage(&device);
}

#[test]
fn test_breakpoint_resume_and_cancellation_reproduce_through_both_drivers() {
    let source = ".version 9.3 .target sm_80 .address_size 64 .global .u32 output;
        .func helper() {brkpt;ret;}
        .entry k() {st.global.u32 [output],1; call helper;
            st.global.u32 [output],2; brkpt; st.global.u32 [output],3; ret;}";
    for cancel_second in [false, true] {
        let parsed = ParsedModule::parse(source).unwrap();
        let code = parsed.compile("k").unwrap();
        let options = EmulatorOptions::default();
        let config = LaunchConfig::create([1; 3], [1; 3], 0).unwrap();
        let flag = AtomicBool::new(false);
        let mut memory = Memory::new(4096, Arc::new(Budget::new(options.execution_storage_bytes)));
        let output = memory
            .allocate(4, 4, StateSpace::Global, Owner::Device, true, &flag)
            .unwrap();
        let mut engine =
            Engine::create(&code, &[output], config, &[], &options, &flag, &mut memory).unwrap();
        let mut expected_breakpoints = Vec::new();
        let expected = loop {
            match engine.step(&mut memory, &flag) {
                Ok(None) => {
                    if let Some(location) = engine.breakpoint_location() {
                        expected_breakpoints.push(location);
                        if cancel_second && expected_breakpoints.len() == 2 {
                            flag.store(true, Ordering::Release);
                        } else {
                            engine.resume();
                        }
                    }
                }
                result => break result.map(Option::unwrap),
            }
        };
        engine.release(&mut memory);
        let device = Emulator::create(options).unwrap();
        let module = device.load(parsed).unwrap().wait().unwrap();
        let launch = module.kernel("k").unwrap().launch(config, []).unwrap();
        for (index, expected) in expected_breakpoints.into_iter().enumerate() {
            let breakpoint = device.next_breakpoint().unwrap().wait().unwrap();
            assert_eq!(breakpoint.location, expected);
            if cancel_second && index == 1 {
                device.client.shared.cancel.store(true, Ordering::Release);
                device.client.shared.changed.notify_all();
            } else {
                device.resume(&breakpoint.id).unwrap().wait().unwrap();
            }
        }
        let actual = launch.wait();
        assert_eq!(format!("{expected:?}"), format!("{actual:?}"));
        assert_eq!(engine.trace, *lock(&device.client.shared.hooks.trace));
        device.client.shared.cancel.store(false, Ordering::Release);
        assert_eq!(
            device
                .download(&module.global("output").unwrap())
                .unwrap()
                .wait()
                .unwrap(),
            memory.read(output, 4, 4, None, None).unwrap()
        );
        device.shutdown(ShutdownMode::Drain).wait().unwrap();
        assert_eq!(device.client.shared.execution_budget.used(), 0);
    }
}

#[test]
fn test_abandoned_module_initialization_yields_and_releases_unpublished_storage() {
    let total = CHUNK_BYTES * 8;
    let device = Emulator::create(EmulatorOptions {
        device_storage_bytes: Memory::charge(total).unwrap() + 1024,
        ..Default::default()
    })
    .unwrap();
    let runnable = device.load(empty_module()).unwrap().wait().unwrap();
    let compute = device.create_stream().unwrap();
    let (started_tx, started_rx) = mpsc::channel();
    let (resume_tx, resume_rx) = mpsc::channel();
    let mut paused = false;
    let initialized = Arc::new(AtomicBool::new(false));
    let finished = Arc::clone(&initialized);
    *lock(&device.client.shared.hooks.transfer) = Some(Box::new(move |completed, size| {
        if size == total && completed == size {
            finished.store(true, Ordering::Release);
        }
        if size == total && !paused && completed < size {
            paused = true;
            started_tx.send(()).unwrap();
            resume_rx.recv().unwrap();
        }
    }));
    let observed = Arc::new(AtomicBool::new(false));
    let observed_worker = Arc::clone(&observed);
    let finished = Arc::clone(&initialized);
    *lock(&device.client.shared.hooks.launch) = Some(Box::new(move |_| {
        if !finished.load(Ordering::Acquire) {
            observed_worker.store(true, Ordering::Release);
        }
    }));
    let module = device.load(ParsedModule::parse(&format!(
        ".version 9.3 .target sm_80 .address_size 64 .global .b8 storage[{total}]; .entry k() {{ ret; }}"
    )).unwrap()).unwrap();
    started_rx.recv().unwrap();
    drop(module);
    let launch = compute
        .launch(
            &runnable.kernel("k").unwrap(),
            LaunchConfig::create([1, 1, 1], [1, 1, 1], 0).unwrap(),
            [],
        )
        .unwrap();
    let fence = device.synchronize().unwrap();
    resume_tx.send(()).unwrap();
    launch.wait().unwrap();
    fence.wait().unwrap();
    assert!(observed.load(Ordering::Acquire));
    *lock(&device.client.shared.hooks.transfer) = None;
    // The abandoned module would exhaust this limit if its private result leaked.
    let buffer = device.zeroed::<u8>(total).unwrap().wait().unwrap();
    assert_eq!(buffer.len(), total);
    device.shutdown(ShutdownMode::Drain).wait().unwrap();
    assert_eq!(device.client.shared.execution_budget.used(), 0);
}

#[test]
fn test_cancel_an_active_infinite_kernel_with_a_full_queue() {
    let device = Emulator::create(EmulatorOptions {
        queued_commands: 1,
        ..Default::default()
    })
    .unwrap();
    let module = device
        .load(
            ParsedModule::parse(
                ".version 9.3 .target sm_80 .address_size 64 .entry k() { loop: bra loop; }",
            )
            .unwrap(),
        )
        .unwrap()
        .wait()
        .unwrap();
    let (reached_tx, reached_rx) = mpsc::channel();
    let (resume_tx, resume_rx) = mpsc::channel();
    let mut reached = false;
    *lock(&device.client.shared.hooks.launch) = Some(Box::new(move |engine| {
        if !reached && engine.trace.len() >= 5 {
            reached = true;
            reached_tx.send(()).unwrap();
            resume_rx.recv().unwrap();
        }
    }));
    let launch = module
        .kernel("k")
        .unwrap()
        .launch(LaunchConfig::create([1, 1, 1], [1, 1, 1], 0).unwrap(), [])
        .unwrap();
    reached_rx.recv().unwrap();
    let queued = device.upload(&[7u32]).unwrap();
    let drain = device.shutdown(ShutdownMode::Drain);
    let cancel = device.shutdown(ShutdownMode::Cancel);
    resume_tx.send(()).unwrap();
    assert!(matches!(
        launch.wait().unwrap_err().root_cause(),
        ExecutionError::Cancelled
    ));
    assert!(matches!(queued.wait(), Err(ExecutionError::Cancelled)));
    drain.wait().unwrap();
    cancel.wait().unwrap();
}

#[test]
fn test_direct_and_worker_drivers_reproduce_values_faults_and_bounded_traces() {
    struct Scenario {
        name: &'static str,
        source: &'static str,
        bytes: usize,
        cancel_at: usize,
        expected: Vec<u8>,
    }
    let scenarios = [
        Scenario {
            name: "matrix multiply",
            source: r".version 9.4 .target sm_80 .address_size 64 .global .u32 output[128];
                .entry k() { .reg .b32 a<4>; .reg .b32 b<2>; .reg .f32 c<4>;
                    .reg .u32 t; .reg .u64 offset; .reg .u64 address; .reg .u64 base;
                    mov.b32 a0,0x3c003c00; mov.b32 a1,0x3c003c00;
                    mov.b32 a2,0x3c003c00; mov.b32 a3,0x3c003c00;
                    mov.b32 b0,0x3c003c00; mov.b32 b1,0x3c003c00;
                    mov.f32 c0,0f00000000; mov.f32 c1,c0; mov.f32 c2,c0; mov.f32 c3,c0;
                    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
                        {c0,c1,c2,c3},{a0,a1,a2,a3},{b0,b1},{c0,c1,c2,c3};
                    mov.u32 t,%tid.x; mul.wide.u32 offset,t,16;
                    mov.u64 base,output; add.u64 address,base,offset;
                    st.global.v4.f32 [address],{c0,c1,c2,c3}; ret;
                }",
            bytes: 512,
            cancel_at: 11,
            expected: (0..128).flat_map(|_| 16f32.to_le_bytes()).collect(),
        },
        Scenario {
            name: "matrix load",
            source: r".version 9.4 .target sm_80 .address_size 64 .global .u32 output[128];
                .entry k() { .shared .align 16 .b8 tile[512]; .reg .b32 a<4>;
                    .reg .u32 t; .reg .u32 value; .reg .u64 offset; .reg .u64 address; .reg .u64 base;
                    mov.u32 t,%tid.x; mul.wide.u32 offset,t,16;
                    mov.u64 base,tile; add.u64 address,base,offset;
                    mov.u32 value,0x00010001; st.shared.v4.u32 [address],{value,value,value,value};
                    bar.sync 0;
                    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {a0,a1,a2,a3},[address];
                    mov.u64 base,output; add.u64 address,base,offset;
                    st.global.v4.b32 [address],{a0,a1,a2,a3}; ret;
                }",
            bytes: 512,
            cancel_at: 8,
            expected: (0..128)
                .flat_map(|_| 0x0001_0001_u32.to_le_bytes())
                .collect(),
        },
        Scenario {
            name: "asynchronous groups",
            source: r".version 9.4 .target sm_80 .address_size 64 .global .u32 output[32];
                .entry k() { .shared .align 16 .b8 tile[128];
                    .reg .u32 t; .reg .u32 r; .reg .u64 offset; .reg .u64 a; .reg .u64 b;
                    mov.u32 t,%tid.x; mul.wide.u32 offset,t,4;
                    mov.u64 a,output; add.u64 a,a,offset;
                    add.u32 r,t,1; st.global.u32 [a],r; bar.sync 0;
                    mov.u64 b,tile; add.u64 b,b,offset;
                    cp.async.ca.shared.global [b],[a],4;
                    cp.async.commit_group; cp.async.wait_group 0; bar.sync 0;
                    ld.shared.u32 r,[b]; add.u32 r,r,17;
                    st.global.u32 [a],r; ret;
                }",
            bytes: 128,
            cancel_at: 11,
            expected: (18u32..50).flat_map(u32::to_le_bytes).collect(),
        },
        Scenario {
            name: "asynchronous barrier",
            source: r".version 9.4 .target sm_80 .address_size 64 .global .u32 output[32];
                .entry k() { .shared .align 16 .b8 tile[128]; .shared .align 8 .b64 ready;
                    .reg .u32 t; .reg .u32 r; .reg .u64 offset; .reg .u64 a; .reg .u64 b;
                    .reg .b64 state; .reg .pred p;
                    mov.u32 t,%tid.x; setp.eq.u32 p,t,0;
                    @p mbarrier.init.shared.b64 [ready],32; bar.sync 0;
                    mul.wide.u32 offset,t,4; mov.u64 a,output; add.u64 a,a,offset;
                    add.u32 r,t,1; st.global.u32 [a],r;
                    mov.u64 b,tile; add.u64 b,b,offset;
                    cp.async.ca.shared.global [b],[a],4;
                    cp.async.mbarrier.arrive.shared.b64 [ready];
                    mbarrier.arrive.shared.b64 state,[ready];
                    loop: mbarrier.test_wait.shared.b64 p,[ready],state; @!p bra loop;
                    ld.shared.u32 r,[b]; add.u32 r,r,17; st.global.u32 [a],r; ret;
                }",
            bytes: 128,
            cancel_at: 14,
            expected: (18u32..50).flat_map(u32::to_le_bytes).collect(),
        },
        Scenario {
            name: "integer",
            source: r".version 9.3 .target sm_80 .address_size 64 .global .u32 output;
                .entry k() { .reg .u32 r;
                mov.u32 r,%tid.x;
                atom.global.add.u32 r,[output],r;
                bar.sync 0;
                ld.global.u32 r,[output];
                add.u32 r,r,1;
                st.global.u32 [output],r;
                ret;
                }",
            bytes: 4,
            cancel_at: 5,
            expected: 497u32.to_le_bytes().to_vec(),
        },
        Scenario {
            name: "scalar",
            source: r".version 9.3 .target sm_80 .address_size 64 .global .align 16 .b8 output[128];
                .entry k() { .reg .u32 r;
                .reg .u32 t;
                .reg .pred p;
                .reg .b64 a;
                .reg .b64 offset;
                .reg .b16 h;
                mov.u32 t,%tid.x;
                and.b32 r,t,1;
                setp.eq.u32 p,r,0;
                @p add.cc.u32 r,0xffffffff,1;
                @!p add.cc.u32 r,0,0;
                @!p bra odd;
                addc.u32 r,1,0;
                bra done;
                odd: addc.u32 r,2,0;
                done: cvt.rn.f16.u32 h,r;
                fma.rn.f16 h,h,h,h;
                cvt.rzi.u32.f16 r,h;
                mul.wide.u32 offset,t,4;
                mov.u64 a,output;
                add.u64 a,a,offset;
                st.global.u32 [a],r;
                ret;
                }",
            bytes: 128,
            cancel_at: 7,
            expected: [6u32; 32].into_iter().flat_map(u32::to_le_bytes).collect(),
        },
        Scenario {
            name: "quantized",
            source: r".version 9.3 .target sm_80 .address_size 64 .global .align 16 .b8 output[128];
                .func helper() {.reg .b32 r;dp2a.hi.s32.u32 r,0x00010002,0x03040000,0;ret;} .entry k() { .reg .b32 r;
                .reg .b32 x;
                .reg .b32 y;
                .reg .u32 t;
                .reg .pred p;
                .reg .pred q;
                .reg .b64 a;
                .reg .b64 offset;
                mov.u32 t,%tid.x;
                and.b32 r,t,1;
                setp.eq.u32 p,r,0;
                @p add.cc.u32 r,0xffffffff,1;
                @!p add.cc.u32 r,0,0;
                dp4a.s32.u32 r,0x01010101,0x01010101,0;
                cvt.pack.sat.s8.s32.b32 r,r,r,0;
                mov.pred q,0;
                lop3.or.b32 r|q,r,0,0,0xf0,q;
                addc.u32 r,0,0;
                @p bra even;
                mov.b32 x,0x40003c00;
                mov.b32 y,0x3c004000;
                set.lt.f16x2.f16x2 r,x,y;
                setp.lt.f16x2 p|q,x,y;
                bra done;
                even: mad24.hi.sat.s32 r,0x800000,0xffffff,0;
                done: call helper,();
                mul.wide.u32 offset,t,4;
                mov.u64 a,output;
                add.u64 a,a,offset;
                st.global.u32 [a],r;
                ret;
                }",
            bytes: 128,
            cancel_at: 17,
            expected: (0..32)
                .flat_map(|lane| if lane % 2 == 0 { 128u32 } else { 0x3c00 }.to_le_bytes())
                .collect(),
        },
    ];
    for scenario in scenarios {
        for cancel_at in [None, Some(scenario.cancel_at)] {
            let device = Emulator::create(EmulatorOptions::default()).unwrap();
            let parsed = ParsedModule::parse(scenario.source).unwrap();
            let module = device.load(parsed.clone()).unwrap().wait().unwrap();
            let code = parsed.compile("k").unwrap();
            let config = LaunchConfig::create([1, 1, 1], [32, 1, 1], 0).unwrap();
            let flag = AtomicBool::new(false);
            let mut memory = Memory::new(
                4096,
                Arc::new(Budget::new(
                    EmulatorOptions::default().execution_storage_bytes,
                )),
            );
            let global = memory
                .allocate(
                    scenario.bytes,
                    16,
                    StateSpace::Global,
                    Owner::Device,
                    true,
                    &flag,
                )
                .unwrap();
            let mut engine = Engine::create(
                &code,
                &[global],
                config,
                &[],
                &EmulatorOptions::default(),
                &flag,
                &mut memory,
            )
            .unwrap();
            let mut count = 0;
            let direct = loop {
                if cancel_at == Some(count) {
                    flag.store(true, Ordering::Release);
                }
                count += 1;
                match engine.step(&mut memory, &flag) {
                    Ok(None) => {}
                    result => break result,
                }
            };
            let weak = Arc::downgrade(&device.client.shared);
            let mut count = 0;
            *lock(&device.client.shared.hooks.launch) = Some(Box::new(move |_| {
                if cancel_at == Some(count) {
                    weak.upgrade()
                        .unwrap()
                        .cancel
                        .store(true, Ordering::Release);
                }
                count += 1;
            }));
            let worker = module
                .kernel("k")
                .unwrap()
                .launch(config, [])
                .unwrap()
                .wait();
            assert_eq!(
                format!("{:?}", direct.map(|v| v.unwrap())),
                format!("{worker:?}")
            );
            assert_eq!(engine.trace, *lock(&device.client.shared.hooks.trace));
            // The synthetic cancellation hook leaves admission open for a diagnostic comparison.
            device.client.shared.cancel.store(false, Ordering::Release);
            let bytes = memory.read(global, scenario.bytes, 16, None, None).unwrap();
            let actual = device
                .download(&module.global("output").unwrap())
                .unwrap()
                .wait()
                .unwrap();
            assert_eq!(actual, bytes, "{} at {cancel_at:?}", scenario.name);
            if cancel_at.is_none() {
                assert_eq!(bytes, scenario.expected, "{}", scenario.name);
            } else if scenario.name == "integer" {
                assert_eq!(bytes, 496u32.to_le_bytes());
            }
            // Reproduce the same allocation/release events through both drivers.
            flag.store(false, Ordering::Release);
            let abandoned = memory
                .allocate(128, 256, StateSpace::Global, Owner::Device, true, &flag)
                .unwrap();
            memory.release(abandoned);
            let retained = memory
                .allocate(128, 256, StateSpace::Global, Owner::Device, true, &flag)
                .unwrap();
            drop(device.zeroed::<u32>(32).unwrap());
            let owned = device.zeroed::<u32>(32).unwrap().wait().unwrap();
            assert_eq!(device.download(&owned).unwrap().wait().unwrap(), [0; 32]);
            assert_eq!(memory.read(retained, 128, 1, None, None).unwrap(), [0; 128]);
            memory.release(retained);
            drop(owned);
            device.shutdown(ShutdownMode::Drain).wait().unwrap();
        }
    }
}

#[test]
fn test_interleaved_grids_reproduce_reports_storage_lifetimes_and_bounded_traces() {
    let source = ".version 9.3 .target sm_80 .address_size 64
        .global .u32 flag; .global .align 16 .u32 output[64];
        .entry consumer() {
            .shared .u32 scratch[32]; .reg .u32 t; .reg .u32 r; .reg .pred p;
            .reg .u64 a; .reg .u64 offset; .reg .f32 f;
            mov.u32 t,%tid.x; mul.wide.u32 offset,t,4;
            mov.u64 a,scratch; add.u64 a,a,offset;
            sqrt.rn.f32 f,4.0; cvt.rzi.u32.f32 r,f; add.u32 r,r,t; st.shared.u32 [a],r;
            loop: ld.global.u32 r,[flag]; setp.eq.u32 p,r,0; @p bra loop;
            ld.shared.u32 r,[a]; mov.u64 a,output; add.u64 a,a,offset;
            st.global.u32 [a],r; ret;
        }
        .entry producer() {
            .reg .u32 t; .reg .u32 r; .reg .u64 a; .reg .u64 offset;
            mov.u32 t,%tid.x; mul.wide.u32 offset,t,4;
            mov.u64 a,output; add.u64 a,a,128; add.u64 a,a,offset;
            add.u32 r,t,100; st.global.u32 [a],r; bar.sync 0;
            st.global.u32 [flag],1; ret;
        }";
    for cancel_at in [None, Some(12)] {
        let parsed = ParsedModule::parse(source).unwrap();
        let options = EmulatorOptions::default();
        let config = LaunchConfig::create([1, 1, 1], [32, 1, 1], 0).unwrap();
        let cancellation = AtomicBool::new(false);
        let mut memory = Memory::new(4096, Arc::new(Budget::new(options.execution_storage_bytes)));
        let flag = memory
            .allocate(4, 4, StateSpace::Global, Owner::Device, true, &cancellation)
            .unwrap();
        let output = memory
            .allocate(
                256,
                16,
                StateSpace::Global,
                Owner::Device,
                true,
                &cancellation,
            )
            .unwrap();
        let codes = [
            parsed.compile("consumer").unwrap(),
            parsed.compile("producer").unwrap(),
        ];
        let mut builders = [
            Some(GridBuilder::create(&codes[0], 2, config, &options, &mut memory).unwrap()),
            None,
        ];
        let mut engines: [Option<Engine>; 2] = [None, None];
        let mut direct_trace = Vec::new();
        let mut terminal = [None, None];
        let mut next = 0;
        let mut turns = 0;
        let mut issued = 0;
        while terminal.iter().any(Option::is_none) {
            // Admission adds B before A's second turn, then rotates B and A.
            if turns == 1 {
                builders[1] =
                    Some(GridBuilder::create(&codes[1], 2, config, &options, &mut memory).unwrap());
            }
            let index = next;
            next = if turns == 0 { 0 } else { 1 - next };
            turns += 1;
            if terminal[index].is_some() {
                continue;
            }
            if let Some(builder) = &mut builders[index] {
                match builder.advance(&[flag, output], &[], &cancellation, &mut memory) {
                    Ok(false) => continue,
                    Ok(true) => {}
                    Err(error) => {
                        builder.release(&mut memory);
                        builders[index] = None;
                        terminal[index] = Some(Err(error));
                        continue;
                    }
                }
                let mut engine = builders[index].take().unwrap().finish();
                engine.identify(index as u64, index as u64 + 1);
                engines[index] = Some(engine);
            }
            if cancel_at == Some(issued) {
                cancellation.store(true, Ordering::Release);
            }
            issued += 1;
            let engine = engines[index].as_mut().unwrap();
            let start = engine.trace.len();
            let result = engine.step(&mut memory, &cancellation);
            let remaining = 4096usize.saturating_sub(direct_trace.len());
            direct_trace.extend(engine.trace[start..].iter().take(remaining).map(|&entry| {
                LaunchTraceEntry {
                    launch: index as u64,
                    entry,
                }
            }));
            match result {
                Ok(None) => {}
                Ok(Some(report)) => {
                    terminal[index] = Some(Ok(report));
                    engine.release(&mut memory);
                }
                Err(error) => {
                    terminal[index] = Some(Err(error));
                    engine.release(&mut memory);
                }
            }
        }
        let expected = memory.read(output, 256, 16, None, None).unwrap().to_vec();
        memory.release(flag);
        memory.release(output);
        drop(engines);
        drop(memory);

        let device = Emulator::create(options).unwrap();
        let module = device.load(parsed).unwrap().wait().unwrap();
        let view = module.global("output").unwrap();
        let a = device.create_stream().unwrap();
        let b = device.create_stream().unwrap();
        let (blocker, release) = pause(&device);
        let weak = Arc::downgrade(&device.client.shared);
        let mut issued = 0;
        *lock(&device.client.shared.hooks.launch) = Some(Box::new(move |_| {
            if cancel_at == Some(issued) {
                weak.upgrade()
                    .unwrap()
                    .cancel
                    .store(true, Ordering::Release);
            }
            issued += 1;
        }));
        let consumer = a
            .launch(&module.kernel("consumer").unwrap(), config, [])
            .unwrap();
        let producer = b
            .launch(&module.kernel("producer").unwrap(), config, [])
            .unwrap();
        drop(module);
        release.send(()).unwrap();
        blocker.wait().unwrap();
        let actual = [consumer.wait(), producer.wait()];
        for (direct, worker) in terminal.into_iter().zip(actual) {
            assert_eq!(format!("{:?}", direct.unwrap()), format!("{worker:?}"));
        }
        assert_eq!(
            direct_trace,
            *lock(&device.client.shared.hooks.scheduler_trace)
        );
        device.client.shared.cancel.store(false, Ordering::Release);
        assert_eq!(device.download(&view).unwrap().wait().unwrap(), expected);
        if cancel_at.is_none() {
            let expected: Vec<_> = (0u32..32)
                .map(|index| index + 2)
                .chain((0u32..32).map(|index| index + 100))
                .flat_map(u32::to_le_bytes)
                .collect();
            assert_eq!(device.download(&view).unwrap().wait().unwrap(), expected);
        }
        device.shutdown(ShutdownMode::Drain).wait().unwrap();
        assert_eq!(device.client.shared.execution_budget.used(), 0);
    }
}

#[test]
fn test_snapshot_reservation_does_not_block_polling_or_shutdown() {
    let device = Emulator::create(EmulatorOptions::default()).unwrap();
    let completed = device.upload(&[3u32]).unwrap().wait().unwrap();
    let download = device.download(&completed).unwrap();
    device.load(empty_module()).unwrap().wait().unwrap();
    let other = device.clone();
    let (started_tx, started_rx) = mpsc::channel();
    let (finish_tx, finish_rx) = mpsc::channel();
    let snapshot = thread::spawn(move || {
        submit_to::<(), _, _>(&other.client, 0, false, 4, |staging| {
            started_tx.send(()).unwrap();
            finish_rx.recv().unwrap();
            move |_| {
                drop(staging);
                Ok(Box::new(|_| ()) as Pending<()>)
            }
        })
    });
    started_rx.recv().unwrap();
    let mut download = pin::pin!(download);
    assert!(
        matches!(download.as_mut().poll(&mut Context::from_waker(Waker::noop())), Poll::Ready(Ok(values)) if values == [3])
    );
    device.shutdown(ShutdownMode::Drain).wait().unwrap();
    finish_tx.send(()).unwrap();
    assert!(matches!(snapshot.join().unwrap(), Err(SubmitError::Closed)));
    assert_eq!(lock(&device.client.shared.admission).staged, 0);
    assert_eq!(lock(&device.client.shared.admission).reservations, 0);
}

#[test]
fn test_queued_mutations_fail_after_a_fault_but_downloads_observe_prior_writes() {
    let device = Emulator::create(EmulatorOptions::default()).unwrap();
    let module = device.load(ParsedModule::parse(".version 9.3 .target sm_80 .address_size 64 .entry k(.param .u64 data) { .reg .u64 a; ld.param.u64 a,[data]; st.global.u32 [a],2; trap; }").unwrap()).unwrap().wait().unwrap();
    let data = device.upload(&[1u32]).unwrap().wait().unwrap();
    let (paused, resume) = pause(&device);
    let launch = module
        .kernel("k")
        .unwrap()
        .launch(
            LaunchConfig::create([1, 1, 1], [1, 1, 1], 0).unwrap(),
            [(&data).into()],
        )
        .unwrap();
    let mutation = device.write(&data, &[9]).unwrap();
    let inspection = device.download(&data).unwrap();
    resume.send(()).unwrap();
    paused.wait().unwrap();
    assert!(launch.wait().is_err());
    assert!(matches!(mutation.wait(), Err(ExecutionError::Faulted)));
    assert_eq!(inspection.wait().unwrap(), [2]);
    assert!(device.shutdown(ShutdownMode::Drain).wait().is_err());
}

#[test]
fn test_unleased_addresses_are_reclaimed_during_an_active_launch() {
    let device = Emulator::create(EmulatorOptions::default()).unwrap();
    let module = device.load(ParsedModule::parse(".version 9.3 .target sm_80 .address_size 64 .global .u64 ptr; .entry save(.param .u64 input) { .reg .u64 a; ld.param.u64 a,[input]; st.global.u64 [ptr],a; ret; } .entry read() { .reg .u64 a; .reg .u32 r; ld.global.u64 a,[ptr]; ld.global.u32 r,[a]; ret; }").unwrap()).unwrap().wait().unwrap();
    let data = device.upload(&[3u32]).unwrap().wait().unwrap();
    let config = LaunchConfig::create([1, 1, 1], [1, 1, 1], 0).unwrap();
    module
        .kernel("save")
        .unwrap()
        .launch(config, [(&data).into()])
        .unwrap()
        .wait()
        .unwrap();
    let (arrived_tx, arrived_rx) = mpsc::channel();
    let (resume_tx, resume_rx) = mpsc::channel();
    let mut paused = false;
    *lock(&device.client.shared.hooks.launch) = Some(Box::new(move |engine| {
        if !paused && engine.trace.len() == 1 {
            paused = true;
            arrived_tx.send(()).unwrap();
            resume_rx.recv().unwrap();
        }
    }));
    let launch = module.kernel("read").unwrap().launch(config, []).unwrap();
    arrived_rx.recv().unwrap();
    drop(data);
    resume_tx.send(()).unwrap();
    let error = launch.wait().unwrap_err();
    assert!(error.to_string().contains("memory access"), "{error}");
    assert!(device.shutdown(ShutdownMode::Drain).wait().is_err());
}

#[test]
fn test_consumed_submission_reports_an_error_and_preserves_event_status() {
    let device = Emulator::create(EmulatorOptions::default()).unwrap();
    let stream = device.default_stream();
    let (event, mut submission) = stream.record_event().unwrap();
    stream.synchronize().unwrap().wait().unwrap();
    let mut context = Context::from_waker(Waker::noop());
    assert!(matches!(
        Pin::new(&mut submission).poll(&mut context),
        Poll::Ready(Ok(()))
    ));
    assert!(matches!(
        Pin::new(&mut submission).poll(&mut context),
        Poll::Ready(Err(ExecutionError::ResultConsumed))
    ));
    assert!(matches!(
        submission.wait(),
        Err(ExecutionError::ResultConsumed)
    ));
    assert!(matches!(event.status(), Some(Ok(()))));
    device.shutdown(ShutdownMode::Drain).wait().unwrap();
}

#[test]
fn test_consumed_completion_releases_its_budget_once() {
    let budget = Arc::new(Budget::new(64));
    let completion = Completion::<u32>::create(Some(budget.reserve(64).unwrap()), false);
    completion.publish(Ok(7));
    let context = Context::from_waker(Waker::noop());
    assert!(matches!(completion.poll(&context), Poll::Ready(Ok(7))));
    assert_eq!(budget.used(), 0);
    assert!(matches!(
        completion.take(),
        Err(ExecutionError::ResultConsumed)
    ));
    assert!(matches!(
        completion.poll(&context),
        Poll::Ready(Err(ExecutionError::ResultConsumed))
    ));
    completion.abandon();
    assert_eq!(budget.used(), 0);
}

#[test]
fn test_failed_consumption_preserves_terminal_status() {
    let completion = Completion::<()>::create(None, true);
    completion.publish(Err(ExecutionError::Trap));
    assert!(matches!(completion.take(), Err(ExecutionError::Trap)));
    let context = Context::from_waker(Waker::noop());
    assert!(matches!(
        completion.poll(&context),
        Poll::Ready(Err(ExecutionError::ResultConsumed))
    ));
    assert!(matches!(
        completion.take(),
        Err(ExecutionError::ResultConsumed)
    ));
    assert!(matches!(
        completion.status(),
        Some(Err(ExecutionError::Trap))
    ));
}
