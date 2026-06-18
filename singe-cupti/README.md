singe-cupti
===

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->
[![Latest version](https://img.shields.io/crates/v/singe-cupti.svg)](https://crates.io/crates/singe-cupti)
[![Documentation](https://docs.rs/singe-cupti/badge.svg)](https://docs.rs/singe-cupti)
![License](https://img.shields.io/crates/l/singe-cupti.svg)

Safe Rust wrappers for NVIDIA CUPTI profiling, activity tracing, callbacks, events, and metrics.

## Examples

```rust,ignore
use singe_cupti::{
    collector::{self, ActivityBufferCallbackConfig},
    context::Context,
    types::{ActivityFlushFlag, ActivityKind},
};

let context = Context::create()?;
context.bind()?;

let _activity_callbacks = collector::register_callbacks(
    ActivityBufferCallbackConfig::create(16 * 1024),
    |buffer| {
        for record in buffer.records().flatten() {
            println!("activity record: {:?}", record.decode());
        }
    },
)?;

collector::enable(ActivityKind::Kernel)?;
collector::flush_all(ActivityFlushFlag::Forced)?;
```
