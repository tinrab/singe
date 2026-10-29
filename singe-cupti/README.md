# singe-cupti

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cupti.svg)](https://crates.io/crates/singe-cupti)
[![Documentation](https://docs.rs/singe-cupti/badge.svg)](https://docs.rs/singe-cupti)
![License](https://img.shields.io/crates/l/singe-cupti.svg)

Safe Rust wrappers for NVIDIA CUPTI, for tracing and profiling CUDA applications.

This crate provides:

- CUDA API callback subscriptions and decoded callback payloads.
- Activity tracing with owned or borrowed activity buffers.
- Event and metric collection.
- Host metric configuration and range profiling.

## Example

Observe CUDA driver calls made while creating a context:

```rust
use singe_cuda::context::Context;
use singe_cupti::{callback::Subscriber, types::CallbackDomain};

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let subscriber = Subscriber::create(|domain, callback_id, _data| {
		println!("{domain:?}: {callback_id:?}");
	})?;
	subscriber.enable_domain(true, CallbackDomain::DriverApi)?;
	let context = Context::create()?;
	drop(context);
	subscriber.unsubscribe()?;
	Ok(())
}
```

`Subscriber` owns the subscription.
CUPTI invokes its callback on the thread making the traced call.
The `activity` module collects activity records, and the `profiler` module collects metrics over code ranges.

## More examples

- [Callback trace](examples/callback_trace.rs): count named CUDA driver calls during a device transfer.

Some profiling features need elevated driver permissions or particular GPUs.
See the [API reference](https://docs.rs/singe-cupti).
