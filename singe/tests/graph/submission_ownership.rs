#![cfg(feature = "cuda")]

use std::{
    sync::{
        Arc,
        mpsc::{self, Sender},
    },
    thread,
    time::Duration,
};

use singe_cuda::{
    context::Context,
    error::Error as CudaError,
    memory::DeviceMemory,
    stream::Stream,
    submission::{AccessMode, Execution, Operation, Resources, transfer::DeviceValue},
    view::DeviceRegionProvider,
};

struct ReleaseStream(Sender<()>);

impl Drop for ReleaseStream {
    fn drop(&mut self) {
        let _ = self.0.send(());
    }
}

fn hold_stream(stream: &Stream) -> ReleaseStream {
    let (release, wait) = mpsc::channel();
    stream
        .launch_host_func(move || {
            let _ = wait.recv_timeout(Duration::from_secs(30));
        })
        .unwrap();
    ReleaseStream(release)
}

#[test]
fn test_dropped_submission_retains_storage_and_cross_stream_consumers_wait_for_the_writer() {
    let context = Context::create().unwrap();
    let producer = context.create_stream().unwrap();
    let consumer = context.create_stream().unwrap();
    let memory = DeviceMemory::from_slice(&context, &[0_u8; 1024]).unwrap();
    let pointer = memory.as_ptr().cast_mut();
    let owner = Arc::downgrade(memory.allocation());
    let mut resources = Resources::new();
    resources.retain(Arc::clone(memory.allocation()));
    resources.access(memory.region(), AccessMode::Write);
    let readback = DeviceValue::from_initialized(memory).readback().unwrap();
    let mut execution = Execution::create(&producer).unwrap();
    let gate = hold_stream(&producer);
    // SAFETY: the ledger retains the full allocation and declares its write.
    // The only reader below participates in the same range tracker.
    unsafe {
        execution
            .enqueue(resources, |stream| {
                DeviceMemory::<u8>::set_async(pointer, 7, 1024, stream)
            })
            .unwrap();
    }
    let submitted = execution.finish(()).unwrap();
    let completion = submitted.completion().clone();
    drop(submitted);
    let copied = readback.submit_on(&consumer).unwrap();
    assert!(!completion.query().unwrap());
    assert!(!copied.completion().query().unwrap());
    assert!(owner.upgrade().is_some());
    drop(gate);
    assert_eq!(copied.wait().unwrap(), [7_u8; 1024]);
    completion.wait().unwrap();
    assert!(owner.upgrade().is_none());
}

#[test]
fn test_failure_after_native_submission_drains_work_before_releasing_the_resource_ledger() {
    let context = Context::create().unwrap();
    let stream = context.create_stream().unwrap();
    let memory = Arc::new(DeviceMemory::from_slice(&context, &[0_u8; 1024]).unwrap());
    let pointer = memory.as_ptr().cast_mut();
    let owner = Arc::downgrade(&memory);
    let mut resources = Resources::new();
    resources.retain(Arc::clone(&memory));
    resources.access(memory.region(), AccessMode::Write);
    let mut execution = Execution::create(&stream).unwrap();
    let gate = hold_stream(&stream);
    // SAFETY: the ledger retains the allocation and its declared write even though host code reports failure after successfully enqueuing the memset.
    let result = unsafe {
        execution.enqueue(resources, |stream| {
            DeviceMemory::<u8>::set_async(pointer, 9, 1024, stream)?;
            Err::<(), CudaError>(CudaError::ZeroValue {
                name: "injected failure after submission".into(),
            })
        })
    };
    assert!(result.is_err());
    drop(memory);
    let (started, dropping) = mpsc::channel();
    let cleanup = thread::spawn(move || {
        started.send(()).unwrap();
        drop(execution);
    });
    dropping.recv_timeout(Duration::from_secs(5)).unwrap();
    let retained = owner
        .upgrade()
        .expect("failed execution must retain submitted storage");
    drop(gate);
    cleanup.join().unwrap();
    let mut output = [0_u8; 1024];
    retained.copy_to_host(&mut output).unwrap();
    assert_eq!(output, [9_u8; 1024]);
    drop(retained);
    assert!(owner.upgrade().is_none());
}
