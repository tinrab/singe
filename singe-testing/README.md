# singe-testing

Test and benchmark helpers for Singe's own crates.
Tests compare Singe's results with PyTorch references, and benchmarks time operations and compare runs saved by name.

This crate provides:

- **References:** `reference!` loads a reference that `scripts/references` wrote under the calling crate's `tests/references`, and `Reference` returns its tensors on a chosen device and type, and its JSON metadata.
- **Comparison:** `assert_matches` compares two `f32` slices under a `Tolerance`, which is exact, absolute, or relative, and states why that bound holds; `host` reads a tensor back as `f32` values.
- **Benchmarks:** `Benches` runs each named closure for a warm-up and a measured period, prints the median and spread, and saves or compares results under `benchmarks/<name>.json`.
- **Inputs and snapshots:** `pattern_image` and `tone_and_chirp` build deterministic test inputs, and `snapshot_if_set` reads a model snapshot directory from `SINGE_SNAPSHOT_<NAME>`.
- **Measuring:** `wait` reads a tensor back so the work behind it finishes, `decode_step` steps a decoder one token per sequence, and `decode_step_or_restart` starts the decoder over when the step would fill it.

## Example

A test that runs a layer on a reference's input and compares the result with PyTorch's output:

```rust,ignore
use singe::{DataType, Device};
use singe_testing::{Tolerance, assert_matches, host, reference};

const F32: Tolerance = Tolerance::relative(2e-5, 2e-5, "F32 sums in another order than PyTorch");

#[test]
fn test_layer_matches_pytorch() {
    let reference = reference!("layer");
    let input = reference.tensor("input", Device::Cpu, DataType::F32);
    let output = layer(&input).unwrap();
    assert_matches(&host(&output), &reference.values("output"), F32, "layer output");
}
```

`just prepare-tests` writes the references; `reference!` panics with that hint when one is missing.

A benchmark target with `harness = false` drives `Benches` from its arguments:

```rust,ignore
use singe_testing::Benches;

fn main() {
    let mut benches = Benches::from_args();
    benches.run("matmul 1024", || {
        // Run the operation and wait for the device.
    });
    benches.finish();
}
```

`just bench --save NAME` saves the results, and `just bench --compare NAME` prints each benchmark's change against them.

## Features

- `cuda_13_4`: builds `singe` with CUDA, and re-exports `lock_device`, which locks CUDA device 0 so GPU tests run one at a time, or returns `None` when no CUDA driver loads.
  It also adds `profiled`, which runs work and returns, from CUPTI's activity records, each kernel's GPU time and launches by name and the copies' time.
