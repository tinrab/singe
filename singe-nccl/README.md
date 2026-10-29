# singe-nccl

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-nccl.svg)](https://crates.io/crates/singe-nccl)
[![Documentation](https://docs.rs/singe-nccl/badge.svg)](https://docs.rs/singe-nccl)
![License](https://img.shields.io/crates/l/singe-nccl.svg)

Safe Rust wrappers for NCCL, the NVIDIA Collective Communications Library, for exchanging data between GPUs.

This crate provides:

- Communicators and grouped collective calls.
- All-reduce, broadcast, gather/scatter, all-to-all, and point-to-point communication.
- Local rank groups with one communicator and stream per rank.
- NCCL memory allocation, and raw asynchronous operations.

## Example

Sum four values across two local GPU ranks:

```rust,ignore
use singe_cuda::{memory::DeviceMemory, view::DeviceOutputView};
use singe_nccl::{local::LocalCommunicatorGroup, types::ReductionOperator};

let group = LocalCommunicatorGroup::create_all_visible(2)?;
let inputs = group.map_ranks(|rank| {
	Ok(DeviceMemory::from_slice(rank.communicator().cuda_context(), &[1.0_f32; 4])?)
})?;
let mut outputs = group.map_ranks(|rank| {
	Ok(DeviceMemory::<f32>::zeroed(rank.communicator().cuda_context(), 4)?)
})?;

group.all_reduce(
	&inputs.map(DeviceMemory::view),
	&mut outputs.map_mut(DeviceOutputView::from_output),
	ReductionOperator::Sum,
)?;
for output in &outputs {
	assert_eq!(output.copy_to_host_vec()?, [2.0; 4]);
}
```

The group owns one communicator and stream per rank, and `all_reduce` completes before returning.
This example needs two visible GPUs.
The [complete example](examples/all_reduce.rs) also receives into uninitialized buffers and destroys the group explicitly.

## Memory and execution

NCCL uses each device's primary CUDA context, so allocate a rank's buffers in its communicator's context.
The group methods wait for the operation and check its status before returning.
Raw collective calls are `unsafe`: everything they use must stay valid until the work finishes.

Use `LocalCommunicatorGroup` for a single process controlling several GPUs.
Use `Communicator` for process-per-rank programs or custom scheduling.

## More examples

- [Broadcast](examples/broadcast.rs): distribute one rank's values.
- [Reduce-scatter](examples/reduce_scatter.rs): reduce and partition results.
- [All-to-all](examples/all_to_all.rs): exchange partitions between ranks.
- [Send/receive](examples/sendrecv.rs): group point-to-point communication.

See the [API reference](https://docs.rs/singe-nccl).
