# singe-nccl

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-nccl.svg)](https://crates.io/crates/singe-nccl)
[![Documentation](https://docs.rs/singe-nccl/badge.svg)](https://docs.rs/singe-nccl)
![License](https://img.shields.io/crates/l/singe-nccl.svg)

Safe Rust wrappers for NCCL, the NVIDIA Collective Communications Library, which moves and reduces data between GPUs.

This crate provides:

- **Local groups:** `LocalCommunicatorGroup` runs one rank per GPU in a single process, and its collectives return after every rank finishes.
- **Communicators:** `Communicator` owns one rank, for programs that run a process per GPU, with collectives as `unsafe` `*_async` methods or safe `*_scoped` methods.
- **Collectives:** all-reduce, broadcast, reduce, all-gather, gather, scatter, all-to-all, and reduce-scatter, plus point-to-point send and receive.
- **Registration and windows:** registered buffers, custom reduction operators, and one-sided signals on a `Window`.
- **Groups and configuration:** NCCL group calls, NCCL-owned allocations, socket encryption, and runtime parameters.

## Example

Create one rank per visible GPU, fill each rank's buffer with its index, and sum the buffers in place:

```rust,no_run
use singe_cuda::{device::Device, memory::DeviceMemory};
use singe_nccl::{local::LocalCommunicatorGroup, types::ReductionOperator};

fn main() -> anyhow::Result<()> {
    let devices: Vec<_> = (0..Device::count()?).map(Device::new).collect();
    let group = LocalCommunicatorGroup::create(&devices)?;

    let mut buffers = group.map_ranks(|rank| {
        let values = [rank.index() as f32; 4];
        Ok(DeviceMemory::from_slice(rank.communicator().cuda_context(), &values)?)
    })?;

    group.all_reduce_in_place(
        &mut buffers.map_mut(DeviceMemory::view_mut),
        ReductionOperator::Sum,
    )?;

    let expected = (0..group.rank_count()).sum::<usize>() as f32;
    for (rank, buffer) in group.ranks().zip(&buffers) {
        rank.bind()?;
        assert_eq!(buffer.copy_to_host_vec()?, [expected; 4]);
    }

    drop(buffers);
    group.destroy()?;
    Ok(())
}
```

`map_ranks` builds a `RankVec` with one value per rank, in device order.
NCCL uses each device's primary CUDA context, so each buffer is allocated in its rank's communicator context.

## Raw collectives

The `*_async` methods on `Communicator` take `CollectiveOptions` and a stream, and return once the work is enqueued.
Every rank must issue the matching call, and the buffers must stay valid until the stream finishes.
`LocalCommunicatorGroup::completed_group` runs such calls for every local rank in one NCCL group and waits for them.
The `*_scoped` forms take a `StreamScope` from `Stream::sync_scope` instead of a stream, and the scope waits for the work before it returns.

## More examples

Every example needs at least two visible GPUs.

- [All-reduce](./examples/all_reduce.rs): sum values across ranks.
- [Broadcast](./examples/broadcast.rs): copy one rank's values to every rank.
- [Reduce](./examples/reduce.rs): reduce onto one root rank.
- [All-gather](./examples/all_gather.rs): collect every rank's values on every rank.
- [Gather](./examples/gather.rs) and [scatter](./examples/scatter.rs): collect onto and split from one root rank.
- [Reduce-scatter](./examples/reduce_scatter.rs): reduce, then give each rank one partition of the result.
- [All-to-all](./examples/all_to_all.rs): exchange partitions between ranks.
- [Send/receive](./examples/sendrecv.rs): grouped point-to-point exchange in a ring.
- [Hypercube](./examples/hypercube.rs): an all-gather built from point-to-point exchanges over a power-of-two number of GPUs.
