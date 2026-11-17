//! Safe Rust wrappers for NCCL, the NVIDIA Collective Communications Library, which moves and reduces data between GPUs.
//!
//! This crate provides:
//!
//! - **Local groups:** [`LocalCommunicatorGroup`](local::LocalCommunicatorGroup) runs one rank per GPU in a single process, and its collectives return after every rank finishes.
//! - **Communicators:** [`Communicator`](communicator::Communicator) owns one rank, for programs that run a process per GPU, with collectives as `unsafe` `*_async` methods or safe `*_scoped` methods.
//! - **Collectives:** all-reduce, broadcast, reduce, all-gather, gather, scatter, all-to-all, and reduce-scatter, plus point-to-point send and receive.
//! - **Registration and windows:** registered buffers, custom reduction operators, and one-sided signals on a [`Window`](communicator::Window).
//! - **Groups and configuration:** NCCL group calls, NCCL-owned allocations, socket encryption, and runtime parameters.
//!
//! # Example
//!
//! Create one rank per visible GPU, fill each rank's buffer with its index, and sum the buffers in place:
//!
//! ```rust,no_run
//! use singe_cuda::{device::Device, memory::DeviceMemory};
//! use singe_nccl::{local::LocalCommunicatorGroup, types::ReductionOperator};
//!
//! fn main() -> anyhow::Result<()> {
//!     let devices: Vec<_> = (0..Device::count()?).map(Device::new).collect();
//!     let group = LocalCommunicatorGroup::create(&devices)?;
//!
//!     let mut buffers = group.map_ranks(|rank| {
//!         let values = [rank.index() as f32; 4];
//!         Ok(DeviceMemory::from_slice(rank.communicator().cuda_context(), &values)?)
//!     })?;
//!
//!     group.all_reduce_in_place(
//!         &mut buffers.map_mut(DeviceMemory::view_mut),
//!         ReductionOperator::Sum,
//!     )?;
//!
//!     let expected = (0..group.rank_count()).sum::<usize>() as f32;
//!     for (rank, buffer) in group.ranks().zip(&buffers) {
//!         rank.bind()?;
//!         assert_eq!(buffer.copy_to_host_vec()?, [expected; 4]);
//!     }
//!
//!     drop(buffers);
//!     group.destroy()?;
//!     Ok(())
//! }
//! ```
//!
//! [`map_ranks`](local::LocalCommunicatorGroup::map_ranks) builds a [`RankVec`](local::RankVec) with one value per rank, in device order.
//! NCCL uses each device's primary CUDA context, so each buffer is allocated in its rank's communicator context.
//!
//! # Raw execution
//!
//! The `unsafe` `*_async` methods of [`communicator::Communicator`] and [`communicator::Window`] check contexts, lengths, and communicator state, order `stream` after the producers of their buffers, enqueue the operation, and return without waiting.
//! They are allowed during stream capture.
//! The caller must:
//!
//! - issue the matching call on every participating rank, in the same order, with the same options;
//! - keep the communicator, stream, buffers, windows, custom operators, and launch-completion events alive until the work completes, including after an error or unwind, and through every replay of a captured graph;
//! - initialize every input, give the work exclusive access to every output, and keep separate inputs and outputs from overlapping;
//! - on a nonblocking communicator, poll [`communicator::Communicator::async_error`] until submission finishes before waiting on the stream, since until then the work may not be on the stream.
//!
//! The safe `*_scoped` forms take a [`StreamScope`](singe_cuda::stream::StreamScope), which keeps the operands borrowed until its stream drains; each waits for nonblocking submission and rejects an open group, whose native enqueue would run after the scope ends.
//! [`LocalCommunicatorGroup::completed_group`](local::LocalCommunicatorGroup::completed_group) runs the calls of every local rank in one NCCL group and waits for them.

#[cfg(not(feature = "nccl_2_32"))]
compile_error!("singe-nccl requires a version feature, such as `nccl_2_32`");

/// Per-collective options, vendor extensions, and launch-completion events.
pub mod collective;

/// Communicator creation, lifecycle, registration, and queued communication.
pub mod communicator;

pub mod encryption;

/// NCCL status codes, wrapper errors, and result types.
pub mod error;

/// Thread-local NCCL operation groups and simulated execution.
pub mod group;

/// Completed collectives across a group of ranks hosted in one process.
pub mod local;

pub mod memory;
pub mod parameter;

/// Typed NCCL identifiers, operations, flags, and communicator configuration.
pub mod types;

pub(crate) mod utility;

/// Serialized CUDA test setup for NCCL.
#[cfg(feature = "testing")]
pub mod testing;

pub use singe_core::LibraryVersion;
use singe_core::PackedVersion;
pub use singe_nccl_sys as sys;

use crate::{
    error::{Error, Result, try_ffi},
    types::UniqueId,
};

/// Returns the version of the currently linked NCCL library.
///
/// # Errors
///
/// Returns an error if NCCL cannot report its version or reports one that does not decode.
pub fn version() -> Result<LibraryVersion> {
    let mut value = 0;
    // SAFETY: `value` is a local `c_int` output slot.
    unsafe {
        try_ffi!(sys::ncclGetVersion(&raw mut value))?;
    }

    // NCCL encodes versions as `major * 10000 + minor * 100 + patch` since 2.9.
    LibraryVersion::from_packed(value, PackedVersion::Major10000Minor100)
        .ok_or_else(|| Error::out_of_range("nccl version"))
}

/// Generates a [`UniqueId`] for [`Communicator::create_rank`](crate::communicator::Communicator::create_rank) and [`Communicator::create_all`](crate::communicator::Communicator::create_all).
///
/// Call this once for a communicator clique and distribute the returned value to all ranks before creating per-rank communicators.
///
/// # Errors
///
/// Returns an error if NCCL cannot generate a unique ID.
pub fn unique_id() -> Result<UniqueId> {
    let mut value = sys::ncclUniqueId::default();
    // SAFETY: `value` is a local `ncclUniqueId` output slot.
    unsafe {
        try_ffi!(sys::ncclGetUniqueId(&raw mut value))?;
    }
    Ok(value.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it_works() -> Result<()> {
        assert_ne!(version()?.major(), 0);
        Ok(())
    }
}
