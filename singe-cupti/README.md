# singe-cupti

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cupti.svg)](https://crates.io/crates/singe-cupti)
[![Documentation](https://docs.rs/singe-cupti/badge.svg)](https://docs.rs/singe-cupti)
![License](https://img.shields.io/crates/l/singe-cupti.svg)

Safe Rust wrappers for CUPTI, the CUDA Profiling Tools Interface, for tracing and profiling CUDA programs from inside the process.

This crate provides:

- **Callbacks:** `Subscriber` runs a closure on CUDA driver and runtime API calls and other CUPTI callback points, with decoded payloads.
- **Activity tracing:** records of kernels, memory copies, API calls, and other CUDA activity, collected in owned or borrowed buffers.
- **Hardware metrics:** metric configuration and range profiling through `ProfilerHost` and `RangeProfiler`.
- **Identifiers:** context, device, stream, and graph IDs as CUPTI reports them, and the driver and runtime callback IDs.

## Example

Print the CUPTI version, then count the CUDA driver API callbacks made while a context and a stream are created:

```rust,no_run
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use singe_cuda::{context::Context, stream::StreamOptions};
use singe_cupti::{
    callback::{Subscriber, SubscriberOptions},
    types::CallbackDomain,
};

fn main() -> anyhow::Result<()> {
    println!("CUPTI {}", singe_cupti::version()?);

    let callbacks = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&callbacks);
    let subscriber = Subscriber::create(
        move |_domain, _callback_id, _data| {
            counter.fetch_add(1, Ordering::Relaxed);
        },
        &SubscriberOptions::default(),
    )?;
    subscriber.set_domain_enabled(CallbackDomain::DriverApi, true)?;

    let context = Context::create()?;
    let _stream = context.create_stream(StreamOptions::default())?;
    subscriber.unsubscribe()?;

    println!("driver API callbacks: {}", callbacks.load(Ordering::Relaxed));
    Ok(())
}
```

A new `Subscriber` has every callback disabled until a domain or callback ID is enabled.
CUPTI calls the closure on the thread that makes the traced call, once on entry and once on exit of each API function.
After `unsubscribe` returns, CUPTI makes no more calls to the closure; dropping the `Subscriber` also unsubscribes.
CUPTI allows one subscriber per process unless every subscriber sets `SubscriberOptions::allow_multiple_subscribers`, so an attached profiler or debugger can make `Subscriber::create` fail.

Hardware metrics need a GPU that supports them, and the NVIDIA driver restricts performance counters to administrators unless it is configured to allow every user.

## More examples

- [Callback trace](./examples/callback_trace.rs): count the CUDA driver functions entered during a device transfer, by name.
