# singe-cupti

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cupti.svg)](https://crates.io/crates/singe-cupti)
[![Documentation](https://docs.rs/singe-cupti/badge.svg)](https://docs.rs/singe-cupti)
![License](https://img.shields.io/crates/l/singe-cupti.svg)

Safe Rust wrappers for NVIDIA CUPTI, the CUDA Profiling Tools Interface, for tracing and profiling CUDA applications.

This crate provides:

- **Callbacks:** `Subscriber` runs a closure on CUDA driver and runtime API calls and other CUPTI callback points, with decoded payloads.
- **Activity tracing:** records of kernels, memory copies, API calls, and other CUDA activity, collected in owned or borrowed buffers.
- **Hardware metrics:** metric configuration and range profiling through `ProfilerHost` and `RangeProfiler`.
- **Identifiers:** context, device, stream, and graph IDs as CUPTI reports them, and the driver and runtime callback IDs.

## Example

Trace which CUDA driver functions are called while copying an array to the GPU:

```rust,no_run
use std::sync::{Arc, Mutex};

use singe_cuda::{context::Context, memory::DeviceMemory};
use singe_cupti::{
    callback::{CallbackPayload, Subscriber, SubscriberOptions},
    types::{ApiCallbackSite, CallbackDomain},
};

fn main() -> anyhow::Result<()> {
    let context = Context::create()?;
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let calls = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&calls);
    let subscriber = Subscriber::create(
        move |_domain, _callback_id, data| {
            data.with_decoded(|data| {
                if let CallbackPayload::Api(api) = data
                    && api.site().is_ok_and(|site| site == ApiCallbackSite::Enter)
                    && let Some(name) = api.function_name()
                {
                    recorded.lock().unwrap().push(name.to_string_lossy().into_owned());
                }
            });
        },
        &SubscriberOptions::default(),
    )?;
    subscriber.set_domain_enabled(CallbackDomain::DriverApi, true)?;
    DeviceMemory::from_slice(&context, &input)?;
    subscriber.unsubscribe()?;

    for name in calls.lock().unwrap().iter() {
        println!("{name}");
    }

    Ok(())
}
```

This prints `cuMemAlloc_v2`, `cuMemcpyHtoDAsync_v2`, `cuStreamSynchronize`, and `cuMemFree_v2`, between a few context and stream lookups.
Only one subscriber may exist per process unless each sets `SubscriberOptions::allow_multiple_subscribers`, so an attached profiler can make `Subscriber::create` fail.

Hardware metrics need a GPU that supports them, and the NVIDIA driver restricts performance counters to administrators unless it is configured to allow every user.

## More examples

- [Callback trace](./examples/callback_trace.rs): count the driver functions called by name, from creating a context to reading a copy back.
