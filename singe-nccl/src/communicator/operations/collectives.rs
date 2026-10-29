use std::ptr;

use singe_cuda::{
    stream::Stream,
    view::{DeviceOutputView, DeviceView, DeviceViewMut},
};

use crate::{
    collective::CollectiveConfig,
    communicator::Communicator,
    error::{Error, Result},
    sys, try_ffi,
    types::{DataTypeLike, ReductionOperator},
    utility::{to_u64, to_usize},
};

impl Communicator {
    /// Reduces the values in `send` with `op` and writes the result to every rank's `recv` buffer.
    /// `send` and `recv` must have the same length.
    ///
    /// Input and output regions must be disjoint.
    /// Work is enqueued on `stream`; asynchronous failures may be reported later
    /// through [`Communicator::async_error`].
    ///
    /// # Errors
    ///
    /// Returns an error if the buffers have different lengths, the stream is not
    /// associated with the communicator's CUDA context, the element count is too
    /// large for NCCL, or NCCL rejects the collective.
    ///
    /// # Safety
    ///
    /// Follow [`Communicator`]'s raw collective contract.
    pub unsafe fn all_reduce<T: DataTypeLike>(
        &self,
        send: DeviceView<'_, T>,
        recv: DeviceOutputView<'_, T>,
        op: ReductionOperator,
        stream: &Stream,
    ) -> Result<()> {
        // SAFETY: the caller upholds the forwarded enqueue contract.
        unsafe { self.all_reduce_with_config(send, recv, op, stream, &CollectiveConfig::default()) }
    }

    /// Like [`Self::all_reduce`], with per-collective configuration.
    ///
    /// # Safety
    ///
    /// Follow [`Communicator`]'s raw collective contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not active, the stream or a buffer belongs to another CUDA context, or binding the CUDA context fails.
    /// Returns an error if buffer lengths do not match the collective or a required element-count calculation is out of range.
    /// Invalid configuration values, vendor extension keys, launch-event ownership, or inconsistent grouped configuration also produce errors.
    /// NCCL collective errors are propagated.
    pub unsafe fn all_reduce_with_config<T: DataTypeLike>(
        &self,
        send: DeviceView<'_, T>,
        mut recv: DeviceOutputView<'_, T>,
        op: ReductionOperator,
        stream: &Stream,
        config: &CollectiveConfig<'_>,
    ) -> Result<()> {
        self.ensure_buffer_context(send.context())?;
        self.ensure_buffer_context(recv.context())?;
        if send.len() != recv.len() {
            return Err(Error::LengthMismatch {
                name: "all_reduce buffers".into(),
            });
        }

        self.ensure_stream(stream)?;
        let config = config.prepare(self)?;
        let count = to_u64(send.len(), "all_reduce count")?;
        // SAFETY: `ncclAllReduceConfig` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::ncclAllReduceConfig(
                send.as_ptr().cast(),
                recv.as_mut_ptr().cast(),
                count,
                T::DATA_TYPE.into(),
                op.into(),
                self.as_raw(),
                stream.as_raw(),
                &raw const config.value,
            ))?;
        }
        Ok(())
    }

    /// Reduces and overwrites the same initialized buffer on every rank.
    ///
    /// # Safety
    /// Follow [`Communicator`]'s raw collective contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not active, the stream or a buffer belongs to another CUDA context, or binding the CUDA context fails.
    /// Returns an error if the element count cannot be represented by NCCL.
    /// NCCL collective errors are propagated.
    pub unsafe fn all_reduce_in_place<T: DataTypeLike>(
        &self,
        buffer: DeviceViewMut<'_, T>,
        op: ReductionOperator,
        stream: &Stream,
    ) -> Result<()> {
        // SAFETY: forwards the caller's collective access and retention contract.
        unsafe {
            self.all_reduce_in_place_with_config(buffer, op, stream, &CollectiveConfig::default())
        }
    }

    /// Like [`Self::all_reduce_in_place`], with per-collective configuration.
    ///
    /// # Safety
    /// Follow [`Communicator`]'s raw collective contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not active, the stream or a buffer belongs to another CUDA context, or binding the CUDA context fails.
    /// Returns an error if the element count cannot be represented by NCCL.
    /// Invalid configuration values, vendor extension keys, launch-event ownership, or inconsistent grouped configuration also produce errors.
    /// NCCL collective errors are propagated.
    pub unsafe fn all_reduce_in_place_with_config<T: DataTypeLike>(
        &self,
        mut buffer: DeviceViewMut<'_, T>,
        op: ReductionOperator,
        stream: &Stream,
        config: &CollectiveConfig<'_>,
    ) -> Result<()> {
        self.ensure_buffer_context(buffer.context())?;
        self.ensure_stream(stream)?;
        let config = config.prepare(self)?;
        let count = to_u64(buffer.len(), "all_reduce count")?;
        // SAFETY: NCCL supports identical input and output pointers for all-reduce.
        unsafe {
            try_ffi!(sys::ncclAllReduceConfig(
                buffer.as_ptr().cast(),
                buffer.as_mut_ptr().cast(),
                count,
                T::DATA_TYPE.into(),
                op.into(),
                self.as_raw(),
                stream.as_raw(),
                &raw const config.value,
            ))?;
        }
        Ok(())
    }

    /// Broadcasts the contents of `send` from `root` into `recv` on every rank.
    /// `send` is only read on the `root` rank. Non-root ranks still pass a buffer, but NCCL ignores its contents.
    /// `send` and `recv` must have the same length.
    ///
    /// Input and output regions must be disjoint.
    /// Work is enqueued on `stream`; asynchronous failures may be reported later
    /// through [`Communicator::async_error`].
    ///
    /// # Errors
    ///
    /// Returns an error if the buffers have different lengths, the stream is not
    /// associated with the communicator's CUDA context, the element count is too
    /// large for NCCL, or NCCL rejects the collective.
    ///
    /// # Safety
    ///
    /// Follow [`Communicator`]'s raw collective contract.
    pub unsafe fn broadcast<T: DataTypeLike>(
        &self,
        send: DeviceView<'_, T>,
        recv: DeviceOutputView<'_, T>,
        root: i32,
        stream: &Stream,
    ) -> Result<()> {
        // SAFETY: the caller upholds the forwarded enqueue contract.
        unsafe {
            self.broadcast_with_config(send, recv, root, stream, &CollectiveConfig::default())
        }
    }

    /// Like [`Self::broadcast`], with per-collective configuration.
    ///
    /// # Safety
    ///
    /// Follow [`Communicator`]'s raw collective contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not active, the stream or a buffer belongs to another CUDA context, or binding the CUDA context fails.
    /// Returns an error if buffer lengths do not match the collective or a required element-count calculation is out of range.
    /// The root rank and root-only buffer arguments must be valid for the calling rank.
    /// Invalid configuration values, vendor extension keys, launch-event ownership, or inconsistent grouped configuration also produce errors.
    /// NCCL collective errors are propagated.
    pub unsafe fn broadcast_with_config<T: DataTypeLike>(
        &self,
        send: DeviceView<'_, T>,
        mut recv: DeviceOutputView<'_, T>,
        root: i32,
        stream: &Stream,
        config: &CollectiveConfig<'_>,
    ) -> Result<()> {
        self.ensure_buffer_context(send.context())?;
        self.ensure_buffer_context(recv.context())?;
        if send.len() != recv.len() {
            return Err(Error::LengthMismatch {
                name: "broadcast buffers".into(),
            });
        }

        self.ensure_stream(stream)?;
        let config = config.prepare(self)?;
        let count = to_u64(send.len(), "broadcast count")?;
        // SAFETY: `ncclBroadcastConfig` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::ncclBroadcastConfig(
                send.as_ptr().cast(),
                recv.as_mut_ptr().cast(),
                count,
                T::DATA_TYPE.into(),
                root,
                self.as_raw(),
                stream.as_raw(),
                &raw const config.value,
            ))?;
        }
        Ok(())
    }

    /// Broadcasts `buffer` from `root` to every rank using the same buffer for both input and output.
    ///
    /// `buffer.len()` determines the element count.
    ///
    /// # Safety
    ///
    /// Follow [`Communicator`]'s raw collective contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not active, the stream or a buffer belongs to another CUDA context, or binding the CUDA context fails.
    /// Returns an error if the element count cannot be represented by NCCL.
    /// The root rank and root-only buffer arguments must be valid for the calling rank.
    /// NCCL collective errors are propagated.
    pub unsafe fn broadcast_in_place<T: DataTypeLike>(
        &self,
        buffer: DeviceViewMut<'_, T>,
        root: i32,
        stream: &Stream,
    ) -> Result<()> {
        // SAFETY: the caller upholds the forwarded enqueue contract.
        unsafe {
            self.broadcast_in_place_with_config(buffer, root, stream, &CollectiveConfig::default())
        }
    }

    /// Like [`Self::broadcast_in_place`], with per-collective configuration.
    ///
    /// # Safety
    ///
    /// Follow [`Communicator`]'s raw collective contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not active, the stream or a buffer belongs to another CUDA context, or binding the CUDA context fails.
    /// Returns an error if the element count cannot be represented by NCCL.
    /// The root rank and root-only buffer arguments must be valid for the calling rank.
    /// Invalid configuration values, vendor extension keys, launch-event ownership, or inconsistent grouped configuration also produce errors.
    /// NCCL collective errors are propagated.
    pub unsafe fn broadcast_in_place_with_config<T: DataTypeLike>(
        &self,
        mut buffer: DeviceViewMut<'_, T>,
        root: i32,
        stream: &Stream,
        config: &CollectiveConfig<'_>,
    ) -> Result<()> {
        self.ensure_buffer_context(buffer.context())?;
        self.ensure_stream(stream)?;
        let config = config.prepare(self)?;
        let count = to_u64(buffer.len(), "broadcast count")?;
        // SAFETY: `ncclBroadcastConfig` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::ncclBroadcastConfig(
                buffer.as_ptr().cast(),
                buffer.as_mut_ptr().cast(),
                count,
                T::DATA_TYPE.into(),
                root,
                self.as_raw(),
                stream.as_raw(),
                &raw const config.value,
            ))?;
        }
        Ok(())
    }

    /// Reduces the values in `send` with `op` and writes the result to `recv` on the `root` rank.
    /// On the `root` rank, `recv` must be `Some(...)` and must have the same length as `send`.
    /// On non-root ranks, pass `None` for `recv`.
    ///
    /// The operation is in-place when the root rank uses the same device buffer for both `send` and `recv`.
    ///
    /// # Safety
    ///
    /// Follow [`Communicator`]'s raw collective contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not active, the stream or a buffer belongs to another CUDA context, or binding the CUDA context fails.
    /// Returns an error if buffer lengths do not match the collective or a required element-count calculation is out of range.
    /// The root rank and root-only buffer arguments must be valid for the calling rank.
    /// NCCL collective errors are propagated.
    pub unsafe fn reduce<T: DataTypeLike>(
        &self,
        send: DeviceView<'_, T>,
        recv: Option<DeviceOutputView<'_, T>>,
        op: ReductionOperator,
        root: i32,
        stream: &Stream,
    ) -> Result<()> {
        // SAFETY: the caller upholds the forwarded enqueue contract.
        unsafe {
            self.reduce_with_config(send, recv, op, root, stream, &CollectiveConfig::default())
        }
    }

    /// Like [`Self::reduce`], with per-collective configuration.
    ///
    /// # Safety
    ///
    /// Follow [`Communicator`]'s raw collective contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not active, the stream or a buffer belongs to another CUDA context, or binding the CUDA context fails.
    /// Returns an error if buffer lengths do not match the collective or a required element-count calculation is out of range.
    /// The root rank and root-only buffer arguments must be valid for the calling rank.
    /// Invalid configuration values, vendor extension keys, launch-event ownership, or inconsistent grouped configuration also produce errors.
    /// NCCL collective errors are propagated.
    pub unsafe fn reduce_with_config<T: DataTypeLike>(
        &self,
        send: DeviceView<'_, T>,
        recv: Option<DeviceOutputView<'_, T>>,
        op: ReductionOperator,
        root: i32,
        stream: &Stream,
        config: &CollectiveConfig<'_>,
    ) -> Result<()> {
        self.ensure_buffer_context(send.context())?;
        if let Some(buffer) = &recv {
            self.ensure_buffer_context(buffer.context())?;
        }
        self.ensure_stream(stream)?;
        let config = config.prepare(self)?;
        let rank = self.rank()?;
        let mut recv = recv;
        if rank == root {
            let Some(ref recv_buffer) = recv else {
                return Err(Error::LengthMismatch {
                    name: "reduce recv buffer".into(),
                });
            };
            if send.len() != recv_buffer.len() {
                return Err(Error::LengthMismatch {
                    name: "reduce buffers".into(),
                });
            }
        }
        let count = to_u64(send.len(), "reduce count")?;
        // SAFETY: `ncclReduceConfig` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::ncclReduceConfig(
                send.as_ptr().cast(),
                recv.as_mut()
                    .map_or(ptr::null_mut(), |recv| recv.as_mut_ptr().cast()),
                count,
                T::DATA_TYPE.into(),
                op.into(),
                root,
                self.as_raw(),
                stream.as_raw(),
                &raw const config.value,
            ))?;
        }
        Ok(())
    }

    /// Gathers each rank's `send` buffer into `recv` on every rank.
    /// Rank `i` contributes its `send` slice to the `i`-th contiguous segment of `recv`.
    ///
    /// `recv` must have length `send.len() * self.rank_count()?`.
    ///
    /// Separate input and output views must not overlap.
    ///
    /// # Safety
    ///
    /// Follow [`Communicator`]'s raw collective contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not active, the stream or a buffer belongs to another CUDA context, or binding the CUDA context fails.
    /// Returns an error if buffer lengths do not match the collective or a required element-count calculation is out of range.
    /// NCCL collective errors are propagated.
    pub unsafe fn all_gather<T: DataTypeLike>(
        &self,
        send: DeviceView<'_, T>,
        recv: DeviceOutputView<'_, T>,
        stream: &Stream,
    ) -> Result<()> {
        // SAFETY: the caller upholds the forwarded enqueue contract.
        unsafe { self.all_gather_with_config(send, recv, stream, &CollectiveConfig::default()) }
    }

    /// Like [`Self::all_gather`], with per-collective configuration.
    ///
    /// # Safety
    ///
    /// Follow [`Communicator`]'s raw collective contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not active, the stream or a buffer belongs to another CUDA context, or binding the CUDA context fails.
    /// Returns an error if buffer lengths do not match the collective or a required element-count calculation is out of range.
    /// Invalid configuration values, vendor extension keys, launch-event ownership, or inconsistent grouped configuration also produce errors.
    /// NCCL collective errors are propagated.
    pub unsafe fn all_gather_with_config<T: DataTypeLike>(
        &self,
        send: DeviceView<'_, T>,
        mut recv: DeviceOutputView<'_, T>,
        stream: &Stream,
        config: &CollectiveConfig<'_>,
    ) -> Result<()> {
        self.ensure_buffer_context(send.context())?;
        self.ensure_buffer_context(recv.context())?;
        let rank_count = to_usize(self.rank_count()?, "rank_count")?;
        let expected_len = send
            .len()
            .checked_mul(rank_count)
            .ok_or_else(|| Error::out_of_range("all_gather recv length"))?;
        if recv.len() != expected_len {
            return Err(Error::LengthMismatch {
                name: "all_gather recv buffer".into(),
            });
        }

        self.ensure_stream(stream)?;
        let config = config.prepare(self)?;
        let send_count = to_u64(send.len(), "all_gather sendcount")?;
        // SAFETY: `ncclAllGatherConfig` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::ncclAllGatherConfig(
                send.as_ptr().cast(),
                recv.as_mut_ptr().cast(),
                send_count,
                T::DATA_TYPE.into(),
                self.as_raw(),
                stream.as_raw(),
                &raw const config.value,
            ))?;
        }
        Ok(())
    }

    /// Performs an all-to-all exchange.
    /// `send` is partitioned into one equally sized segment per rank; segment `j` is sent to rank `j`.
    /// `recv` is partitioned the same way, and segment `i` receives the data sent by rank `i`.
    ///
    /// `send` and `recv` must have the same length, and that length must be divisible by `self.rank_count()?`.
    ///
    /// In-place operation is not supported.
    ///
    /// # Safety
    ///
    /// Follow [`Communicator`]'s raw collective contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not active, the stream or a buffer belongs to another CUDA context, or binding the CUDA context fails.
    /// Returns an error if buffer lengths do not match the collective or a required element-count calculation is out of range.
    /// NCCL collective errors are propagated.
    pub unsafe fn all_to_all<T: DataTypeLike>(
        &self,
        send: DeviceView<'_, T>,
        recv: DeviceOutputView<'_, T>,
        stream: &Stream,
    ) -> Result<()> {
        // SAFETY: the caller upholds the forwarded enqueue contract.
        unsafe { self.all_to_all_with_config(send, recv, stream, &CollectiveConfig::default()) }
    }

    /// Like [`Self::all_to_all`], with per-collective configuration.
    ///
    /// # Safety
    ///
    /// Follow [`Communicator`]'s raw collective contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not active, the stream or a buffer belongs to another CUDA context, or binding the CUDA context fails.
    /// Returns an error if buffer lengths do not match the collective or a required element-count calculation is out of range.
    /// Invalid configuration values, vendor extension keys, launch-event ownership, or inconsistent grouped configuration also produce errors.
    /// NCCL collective errors are propagated.
    pub unsafe fn all_to_all_with_config<T: DataTypeLike>(
        &self,
        send: DeviceView<'_, T>,
        mut recv: DeviceOutputView<'_, T>,
        stream: &Stream,
        config: &CollectiveConfig<'_>,
    ) -> Result<()> {
        self.ensure_buffer_context(send.context())?;
        self.ensure_buffer_context(recv.context())?;
        if send.len() != recv.len() {
            return Err(Error::LengthMismatch {
                name: "all_to_all buffers".into(),
            });
        }

        let rank_count = to_usize(self.rank_count()?, "rank_count")?;
        if rank_count != 0 && !send.len().is_multiple_of(rank_count) {
            return Err(Error::LengthMismatch {
                name: "all_to_all buffer partition".into(),
            });
        }

        self.ensure_stream(stream)?;
        let config = config.prepare(self)?;
        let count = to_u64(send.len() / rank_count.max(1), "all_to_all count")?;
        // SAFETY: `ncclAlltoAllConfig` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::ncclAlltoAllConfig(
                send.as_ptr().cast(),
                recv.as_mut_ptr().cast(),
                count,
                T::DATA_TYPE.into(),
                self.as_raw(),
                stream.as_raw(),
                &raw const config.value,
            ))?;
        }
        Ok(())
    }

    /// Gathers `send` from every rank into `recv` on `root`.
    /// On the root rank, `recv` must be `Some(...)` and is laid out as one contiguous `send.len()` segment per rank, in rank order.
    /// On non-root ranks, pass `None` for `recv`.
    ///
    /// On the root rank, `recv.len()` must equal `send.len() * self.rank_count()?`.
    ///
    /// The operation is in-place when the root rank's `send` buffer aliases its own segment inside `recv`.
    ///
    /// # Safety
    ///
    /// Follow [`Communicator`]'s raw collective contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not active, the stream or a buffer belongs to another CUDA context, or binding the CUDA context fails.
    /// Returns an error if buffer lengths do not match the collective or a required element-count calculation is out of range.
    /// The root rank and root-only buffer arguments must be valid for the calling rank.
    /// NCCL collective errors are propagated.
    pub unsafe fn gather<T: DataTypeLike>(
        &self,
        send: DeviceView<'_, T>,
        recv: Option<DeviceOutputView<'_, T>>,
        root: i32,
        stream: &Stream,
    ) -> Result<()> {
        // SAFETY: the caller upholds the forwarded enqueue contract.
        unsafe { self.gather_with_config(send, recv, root, stream, &CollectiveConfig::default()) }
    }

    /// Like [`Self::gather`], with per-collective configuration.
    ///
    /// # Safety
    ///
    /// Follow [`Communicator`]'s raw collective contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not active, the stream or a buffer belongs to another CUDA context, or binding the CUDA context fails.
    /// Returns an error if buffer lengths do not match the collective or a required element-count calculation is out of range.
    /// The root rank and root-only buffer arguments must be valid for the calling rank.
    /// Invalid configuration values, vendor extension keys, launch-event ownership, or inconsistent grouped configuration also produce errors.
    /// NCCL collective errors are propagated.
    pub unsafe fn gather_with_config<T: DataTypeLike>(
        &self,
        send: DeviceView<'_, T>,
        recv: Option<DeviceOutputView<'_, T>>,
        root: i32,
        stream: &Stream,
        config: &CollectiveConfig<'_>,
    ) -> Result<()> {
        self.ensure_buffer_context(send.context())?;
        if let Some(buffer) = &recv {
            self.ensure_buffer_context(buffer.context())?;
        }
        self.ensure_stream(stream)?;
        let config = config.prepare(self)?;
        let rank = self.rank()?;
        let mut recv = recv;
        if rank == root {
            let rank_count = to_usize(self.rank_count()?, "rank_count")?;
            let expected_len = send
                .len()
                .checked_mul(rank_count)
                .ok_or_else(|| Error::out_of_range("gather recv length"))?;
            let Some(ref recv_buffer) = recv else {
                return Err(Error::LengthMismatch {
                    name: "gather recv buffer".into(),
                });
            };
            if recv_buffer.len() != expected_len {
                return Err(Error::LengthMismatch {
                    name: "gather recv buffer".into(),
                });
            }
        }
        let count = to_u64(send.len(), "gather count")?;
        // SAFETY: `ncclGatherConfig` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::ncclGatherConfig(
                send.as_ptr().cast(),
                recv.as_mut()
                    .map_or(ptr::null_mut(), |recv| recv.as_mut_ptr().cast()),
                count,
                T::DATA_TYPE.into(),
                root,
                self.as_raw(),
                stream.as_raw(),
                &raw const config.value,
            ))?;
        }
        Ok(())
    }

    /// Scatters data from `send` on `root` into `recv` on every rank.
    /// On the root rank, `send` must be `Some(...)` and is interpreted as one contiguous `recv.len()` segment per rank, in rank order.
    /// On non-root ranks, pass `None` for `send`.
    ///
    /// On the root rank, `send.len()` must equal `recv.len() * self.rank_count()?`.
    ///
    /// The operation is in-place when the root rank's `recv` buffer aliases its own segment inside `send`.
    ///
    /// # Safety
    ///
    /// Follow [`Communicator`]'s raw collective contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not active, the stream or a buffer belongs to another CUDA context, or binding the CUDA context fails.
    /// Returns an error if buffer lengths do not match the collective or a required element-count calculation is out of range.
    /// The root rank and root-only buffer arguments must be valid for the calling rank.
    /// NCCL collective errors are propagated.
    pub unsafe fn scatter<T: DataTypeLike>(
        &self,
        send: Option<DeviceView<'_, T>>,
        recv: DeviceOutputView<'_, T>,
        root: i32,
        stream: &Stream,
    ) -> Result<()> {
        // SAFETY: the caller upholds the forwarded enqueue contract.
        unsafe { self.scatter_with_config(send, recv, root, stream, &CollectiveConfig::default()) }
    }

    /// Like [`Self::scatter`], with per-collective configuration.
    ///
    /// # Safety
    ///
    /// Follow [`Communicator`]'s raw collective contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not active, the stream or a buffer belongs to another CUDA context, or binding the CUDA context fails.
    /// Returns an error if buffer lengths do not match the collective or a required element-count calculation is out of range.
    /// The root rank and root-only buffer arguments must be valid for the calling rank.
    /// Invalid configuration values, vendor extension keys, launch-event ownership, or inconsistent grouped configuration also produce errors.
    /// NCCL collective errors are propagated.
    pub unsafe fn scatter_with_config<T: DataTypeLike>(
        &self,
        send: Option<DeviceView<'_, T>>,
        mut recv: DeviceOutputView<'_, T>,
        root: i32,
        stream: &Stream,
        config: &CollectiveConfig<'_>,
    ) -> Result<()> {
        if let Some(buffer) = &send {
            self.ensure_buffer_context(buffer.context())?;
        }
        self.ensure_buffer_context(recv.context())?;
        self.ensure_stream(stream)?;
        let config = config.prepare(self)?;
        let rank = self.rank()?;
        if rank == root {
            let rank_count = to_usize(self.rank_count()?, "rank_count")?;
            let expected_len = recv
                .len()
                .checked_mul(rank_count)
                .ok_or_else(|| Error::out_of_range("scatter send length"))?;
            let Some(send_buffer) = send else {
                return Err(Error::LengthMismatch {
                    name: "scatter send buffer".into(),
                });
            };
            if send_buffer.len() != expected_len {
                return Err(Error::LengthMismatch {
                    name: "scatter send buffer".into(),
                });
            }
        }
        let count = to_u64(recv.len(), "scatter count")?;
        // SAFETY: `ncclScatterConfig` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::ncclScatterConfig(
                send.map_or(ptr::null(), |send| send.as_ptr().cast()),
                recv.as_mut_ptr().cast(),
                count,
                T::DATA_TYPE.into(),
                root,
                self.as_raw(),
                stream.as_raw(),
                &raw const config.value,
            ))?;
        }
        Ok(())
    }

    /// Reduces the values in `send` with `op` and scatters the reduced result across ranks.
    /// Rank `i` receives the `i`-th contiguous segment of the reduced output in `recv`.
    ///
    /// `send.len()` must equal `recv.len() * self.rank_count()?`.
    ///
    /// The operation is in-place when `recv` aliases the segment of `send` corresponding to the local rank.
    ///
    /// # Safety
    ///
    /// Follow [`Communicator`]'s raw collective contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not active, the stream or a buffer belongs to another CUDA context, or binding the CUDA context fails.
    /// Returns an error if buffer lengths do not match the collective or a required element-count calculation is out of range.
    /// NCCL collective errors are propagated.
    pub unsafe fn reduce_scatter<T: DataTypeLike>(
        &self,
        send: DeviceView<'_, T>,
        recv: DeviceOutputView<'_, T>,
        op: ReductionOperator,
        stream: &Stream,
    ) -> Result<()> {
        // SAFETY: the caller upholds the forwarded enqueue contract.
        unsafe {
            self.reduce_scatter_with_config(send, recv, op, stream, &CollectiveConfig::default())
        }
    }

    /// Like [`Self::reduce_scatter`], with per-collective configuration.
    ///
    /// # Safety
    ///
    /// Follow [`Communicator`]'s raw collective contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not active, the stream or a buffer belongs to another CUDA context, or binding the CUDA context fails.
    /// Returns an error if buffer lengths do not match the collective or a required element-count calculation is out of range.
    /// Invalid configuration values, vendor extension keys, launch-event ownership, or inconsistent grouped configuration also produce errors.
    /// NCCL collective errors are propagated.
    pub unsafe fn reduce_scatter_with_config<T: DataTypeLike>(
        &self,
        send: DeviceView<'_, T>,
        mut recv: DeviceOutputView<'_, T>,
        op: ReductionOperator,
        stream: &Stream,
        config: &CollectiveConfig<'_>,
    ) -> Result<()> {
        self.ensure_buffer_context(send.context())?;
        self.ensure_buffer_context(recv.context())?;
        let rank_count = to_usize(self.rank_count()?, "rank_count")?;
        let expected_len = recv
            .len()
            .checked_mul(rank_count)
            .ok_or_else(|| Error::out_of_range("reduce_scatter send length"))?;
        if send.len() != expected_len {
            return Err(Error::LengthMismatch {
                name: "reduce_scatter send buffer".into(),
            });
        }

        self.ensure_stream(stream)?;
        let config = config.prepare(self)?;
        let recv_count = to_u64(recv.len(), "reduce_scatter recvcount")?;
        // SAFETY: `ncclReduceScatterConfig` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::ncclReduceScatterConfig(
                send.as_ptr().cast(),
                recv.as_mut_ptr().cast(),
                recv_count,
                T::DATA_TYPE.into(),
                op.into(),
                self.as_raw(),
                stream.as_raw(),
                &raw const config.value,
            ))?;
        }
        Ok(())
    }

    /// Sends the contents of `send` to `peer`.
    ///
    /// Rank `peer` must call [`Communicator::recv`] with the same element type and the same buffer length.
    ///
    /// Blocks the GPU until the operation completes.
    /// If multiple [`Communicator::send`] and [`Communicator::recv`] operations must progress concurrently to complete, group them between [`group_start`](crate::group::start) and [`group_end`](crate::group::end).
    ///
    /// # Safety
    ///
    /// Follow [`Communicator`]'s raw collective contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not active, the stream or a buffer belongs to another CUDA context, or binding the CUDA context fails.
    /// Returns an error if the peer rank or element count is out of range, or NCCL rejects the transfer.
    pub unsafe fn send<T: DataTypeLike>(
        &self,
        send: DeviceView<'_, T>,
        peer: i32,
        stream: &Stream,
    ) -> Result<()> {
        self.ensure_buffer_context(send.context())?;
        self.ensure_stream(stream)?;
        let count = to_u64(send.len(), "send count")?;
        // SAFETY: `ncclSend` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::ncclSend(
                send.as_ptr().cast(),
                count,
                T::DATA_TYPE.into(),
                peer,
                self.as_raw(),
                stream.as_raw(),
            ))?;
        }
        Ok(())
    }

    /// Receives data from `peer` into `recv`.
    ///
    /// Rank `peer` must call [`Communicator::send`] with the same element type and the same buffer length.
    ///
    /// Blocks the GPU until the operation completes.
    /// If multiple [`Communicator::send`] and [`Communicator::recv`] operations must progress concurrently to complete, group them between [`group_start`](crate::group::start) and [`group_end`](crate::group::end).
    ///
    /// # Safety
    ///
    /// Follow [`Communicator`]'s raw collective contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not active, the stream or a buffer belongs to another CUDA context, or binding the CUDA context fails.
    /// Returns an error if the peer rank or element count is out of range, or NCCL rejects the transfer.
    pub unsafe fn recv<T: DataTypeLike>(
        &self,
        mut recv: DeviceOutputView<'_, T>,
        peer: i32,
        stream: &Stream,
    ) -> Result<()> {
        self.ensure_buffer_context(recv.context())?;
        self.ensure_stream(stream)?;
        let count = to_u64(recv.len(), "recv count")?;
        // SAFETY: `ncclRecv` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::ncclRecv(
                recv.as_mut_ptr().cast(),
                count,
                T::DATA_TYPE.into(),
                peer,
                self.as_raw(),
                stream.as_raw(),
            ))?;
        }
        Ok(())
    }
}
