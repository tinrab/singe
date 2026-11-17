use std::ptr;

use singe_cuda::{
    stream::Stream,
    view::{DeviceOutputView, DeviceRegionProvider, DeviceView, DeviceViewMut},
};

use crate::{
    collective::CollectiveOptions,
    communicator::{Communicator, Reduction},
    error::{Error, Result},
    sys, try_ffi,
    types::DataTypeLike,
    utility::{checked_mul, to_u64, to_usize},
};

impl Communicator {
    /// Enqueues a reduction of `send` with `op` across ranks, writing the result to every rank's `recv`.
    ///
    /// `send` and `recv` must have the same length and must not overlap; use [`Self::all_reduce_in_place_async`] to reduce one buffer in place.
    ///
    /// # Safety
    ///
    /// Follow the crate's [raw execution](crate#raw-execution) contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not active, a buffer or the stream belongs to another CUDA context, the lengths differ or exceed NCCL's range, `options` is invalid, a custom `op` belongs to another communicator, or NCCL rejects the collective.
    pub unsafe fn all_reduce_async<'r, T: DataTypeLike>(
        &self,
        send: DeviceView<'_, T>,
        mut recv: DeviceOutputView<'_, T>,
        op: impl Into<Reduction<'r, T>>,
        options: &CollectiveOptions<'_>,
        stream: &Stream,
    ) -> Result<()> {
        self.ensure_buffer_context(send.context())?;
        self.ensure_buffer_context(recv.context())?;
        if send.len() != recv.len() {
            return Err(Error::LengthMismatch {
                name: "all_reduce buffers".into(),
                expected: send.len(),
                actual: recv.len(),
            });
        }

        self.ensure_stream(stream)?;
        let op = self.reduction::<T>(op.into())?;
        let options = options.prepare(self)?;
        let count = to_u64(send.len(), "all_reduce count")?;
        stream.join_regions(&[send.region(), recv.region()])?;
        // SAFETY: both buffers belong to the communicator's context and hold `count` elements of `T`, the stream shares that context, and the caller keeps every operand alive through completion.
        unsafe {
            try_ffi!(sys::ncclAllReduceConfig(
                send.as_ptr().cast(),
                recv.as_mut_ptr().cast(),
                count,
                T::DATA_TYPE.into(),
                op,
                self.as_raw(),
                stream.as_raw(),
                &raw const options.value,
            ))?;
        }
        Ok(())
    }

    /// Enqueues a reduction across ranks that reads and overwrites the same initialized `buffer` on every rank.
    ///
    /// # Safety
    ///
    /// Follow the crate's [raw execution](crate#raw-execution) contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not active, the buffer or the stream belongs to another CUDA context, the length exceeds NCCL's range, `options` is invalid, a custom `op` belongs to another communicator, or NCCL rejects the collective.
    pub unsafe fn all_reduce_in_place_async<'r, T: DataTypeLike>(
        &self,
        mut buffer: DeviceViewMut<'_, T>,
        op: impl Into<Reduction<'r, T>>,
        options: &CollectiveOptions<'_>,
        stream: &Stream,
    ) -> Result<()> {
        self.ensure_buffer_context(buffer.context())?;
        self.ensure_stream(stream)?;
        let op = self.reduction::<T>(op.into())?;
        let options = options.prepare(self)?;
        let count = to_u64(buffer.len(), "all_reduce count")?;
        stream.join_regions(&[buffer.region()])?;
        // SAFETY: NCCL supports identical input and output pointers for all-reduce; the buffer holds `count` elements of `T` in the communicator's context, and the caller keeps it alive through completion.
        unsafe {
            try_ffi!(sys::ncclAllReduceConfig(
                buffer.as_ptr().cast(),
                buffer.as_mut_ptr().cast(),
                count,
                T::DATA_TYPE.into(),
                op,
                self.as_raw(),
                stream.as_raw(),
                &raw const options.value,
            ))?;
        }
        Ok(())
    }

    /// Enqueues a broadcast of `send` from `root` into `recv` on every rank.
    ///
    /// NCCL reads `send` only on the root rank, but every rank passes a buffer of the same length as `recv`.
    /// `send` and `recv` must not overlap; use [`Self::broadcast_in_place_async`] to broadcast one buffer in place.
    ///
    /// # Safety
    ///
    /// Follow the crate's [raw execution](crate#raw-execution) contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not active, a buffer or the stream belongs to another CUDA context, the lengths differ or exceed NCCL's range, `root` is not a rank of the communicator, `options` is invalid, or NCCL rejects the collective.
    pub unsafe fn broadcast_async<T: DataTypeLike>(
        &self,
        send: DeviceView<'_, T>,
        mut recv: DeviceOutputView<'_, T>,
        root: i32,
        options: &CollectiveOptions<'_>,
        stream: &Stream,
    ) -> Result<()> {
        self.ensure_buffer_context(send.context())?;
        self.ensure_buffer_context(recv.context())?;
        if send.len() != recv.len() {
            return Err(Error::LengthMismatch {
                name: "broadcast buffers".into(),
                expected: send.len(),
                actual: recv.len(),
            });
        }

        self.ensure_stream(stream)?;
        self.ensure_rank(root, "root")?;
        let options = options.prepare(self)?;
        let count = to_u64(send.len(), "broadcast count")?;
        stream.join_regions(&[send.region(), recv.region()])?;
        // SAFETY: both buffers belong to the communicator's context and hold `count` elements of `T`, the stream shares that context, and the caller keeps every operand alive through completion.
        unsafe {
            try_ffi!(sys::ncclBroadcastConfig(
                send.as_ptr().cast(),
                recv.as_mut_ptr().cast(),
                count,
                T::DATA_TYPE.into(),
                root,
                self.as_raw(),
                stream.as_raw(),
                &raw const options.value,
            ))?;
        }
        Ok(())
    }

    /// Enqueues a broadcast from `root` that uses the same `buffer` for input and output on every rank.
    ///
    /// # Safety
    ///
    /// Follow the crate's [raw execution](crate#raw-execution) contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not active, the buffer or the stream belongs to another CUDA context, the length exceeds NCCL's range, `root` is not a rank of the communicator, `options` is invalid, or NCCL rejects the collective.
    pub unsafe fn broadcast_in_place_async<T: DataTypeLike>(
        &self,
        mut buffer: DeviceViewMut<'_, T>,
        root: i32,
        options: &CollectiveOptions<'_>,
        stream: &Stream,
    ) -> Result<()> {
        self.ensure_buffer_context(buffer.context())?;
        self.ensure_stream(stream)?;
        self.ensure_rank(root, "root")?;
        let options = options.prepare(self)?;
        let count = to_u64(buffer.len(), "broadcast count")?;
        stream.join_regions(&[buffer.region()])?;
        // SAFETY: NCCL supports identical input and output pointers for broadcast; the buffer holds `count` elements of `T` in the communicator's context, and the caller keeps it alive through completion.
        unsafe {
            try_ffi!(sys::ncclBroadcastConfig(
                buffer.as_ptr().cast(),
                buffer.as_mut_ptr().cast(),
                count,
                T::DATA_TYPE.into(),
                root,
                self.as_raw(),
                stream.as_raw(),
                &raw const options.value,
            ))?;
        }
        Ok(())
    }

    /// Enqueues a reduction of `send` with `op` across ranks, writing the result to `recv` on the `root` rank.
    ///
    /// The root rank passes `Some` output of the same length as `send`; other ranks pass `None`.
    /// The root may pass the same allocation for `send` and `recv` to reduce in place.
    ///
    /// # Safety
    ///
    /// Follow the crate's [raw execution](crate#raw-execution) contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not active, a buffer or the stream belongs to another CUDA context, the root omits `recv` or passes one of another length, a length exceeds NCCL's range, `root` is not a rank of the communicator, `options` is invalid, a custom `op` belongs to another communicator, or NCCL rejects the collective.
    pub unsafe fn reduce_async<'r, T: DataTypeLike>(
        &self,
        send: DeviceView<'_, T>,
        mut recv: Option<DeviceOutputView<'_, T>>,
        op: impl Into<Reduction<'r, T>>,
        root: i32,
        options: &CollectiveOptions<'_>,
        stream: &Stream,
    ) -> Result<()> {
        self.ensure_buffer_context(send.context())?;
        if let Some(buffer) = &recv {
            self.ensure_buffer_context(buffer.context())?;
        }

        self.ensure_stream(stream)?;
        let op = self.reduction::<T>(op.into())?;
        self.ensure_rank(root, "root")?;
        let options = options.prepare(self)?;
        if self.rank()? == root {
            let Some(recv_buffer) = &recv else {
                return Err(Error::NullPointer {
                    name: "reduce recv buffer".into(),
                });
            };
            if send.len() != recv_buffer.len() {
                return Err(Error::LengthMismatch {
                    name: "reduce buffers".into(),
                    expected: send.len(),
                    actual: recv_buffer.len(),
                });
            }
        }

        let count = to_u64(send.len(), "reduce count")?;
        {
            let mut regions = vec![send.region()];
            regions.extend(recv.as_ref().map(DeviceRegionProvider::region));
            stream.join_regions(&regions)?;
        }

        // SAFETY: every buffer belongs to the communicator's context, the root's output holds `count` elements of `T`, the stream shares that context, and the caller keeps every operand alive through completion.
        unsafe {
            try_ffi!(sys::ncclReduceConfig(
                send.as_ptr().cast(),
                recv.as_mut()
                    .map_or(ptr::null_mut(), |recv| recv.as_mut_ptr().cast()),
                count,
                T::DATA_TYPE.into(),
                op,
                root,
                self.as_raw(),
                stream.as_raw(),
                &raw const options.value,
            ))?;
        }
        Ok(())
    }

    /// Enqueues a gather of every rank's `send` into `recv` on every rank, in rank order.
    ///
    /// `recv` holds `send.len() * rank_count` elements and must not overlap `send`.
    ///
    /// # Safety
    ///
    /// Follow the crate's [raw execution](crate#raw-execution) contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not active, a buffer or the stream belongs to another CUDA context, `recv` has the wrong length, a length exceeds NCCL's range, `options` is invalid, or NCCL rejects the collective.
    pub unsafe fn all_gather_async<T: DataTypeLike>(
        &self,
        send: DeviceView<'_, T>,
        mut recv: DeviceOutputView<'_, T>,
        options: &CollectiveOptions<'_>,
        stream: &Stream,
    ) -> Result<()> {
        self.ensure_buffer_context(send.context())?;
        self.ensure_buffer_context(recv.context())?;
        let rank_count = to_usize(self.rank_count()?, "rank_count")?;
        let expected_len = checked_mul(send.len(), rank_count, "all_gather recv length")?;
        if recv.len() != expected_len {
            return Err(Error::LengthMismatch {
                name: "all_gather recv buffer".into(),
                expected: expected_len,
                actual: recv.len(),
            });
        }

        self.ensure_stream(stream)?;
        let options = options.prepare(self)?;
        let send_count = to_u64(send.len(), "all_gather sendcount")?;
        stream.join_regions(&[send.region(), recv.region()])?;
        // SAFETY: both buffers belong to the communicator's context, `recv` holds `send_count * rank_count` elements of `T`, the stream shares that context, and the caller keeps every operand alive through completion.
        unsafe {
            try_ffi!(sys::ncclAllGatherConfig(
                send.as_ptr().cast(),
                recv.as_mut_ptr().cast(),
                send_count,
                T::DATA_TYPE.into(),
                self.as_raw(),
                stream.as_raw(),
                &raw const options.value,
            ))?;
        }
        Ok(())
    }

    /// Enqueues an all-to-all exchange: segment `j` of `send` goes to rank `j`, and segment `i` of `recv` receives rank `i`'s data.
    ///
    /// `send` and `recv` have the same length, divisible by the rank count, and must not overlap.
    ///
    /// # Safety
    ///
    /// Follow the crate's [raw execution](crate#raw-execution) contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not active, a buffer or the stream belongs to another CUDA context, the lengths differ or do not divide by the rank count, a length exceeds NCCL's range, `options` is invalid, or NCCL rejects the collective.
    pub unsafe fn all_to_all_async<T: DataTypeLike>(
        &self,
        send: DeviceView<'_, T>,
        mut recv: DeviceOutputView<'_, T>,
        options: &CollectiveOptions<'_>,
        stream: &Stream,
    ) -> Result<()> {
        self.ensure_buffer_context(send.context())?;
        self.ensure_buffer_context(recv.context())?;
        if send.len() != recv.len() {
            return Err(Error::LengthMismatch {
                name: "all_to_all buffers".into(),
                expected: send.len(),
                actual: recv.len(),
            });
        }

        let rank_count = to_usize(self.rank_count()?, "rank_count")?;
        if rank_count != 0 && !send.len().is_multiple_of(rank_count) {
            return Err(Error::out_of_range("all_to_all buffer partition"));
        }

        self.ensure_stream(stream)?;
        let options = options.prepare(self)?;
        let count = to_u64(send.len() / rank_count.max(1), "all_to_all count")?;
        stream.join_regions(&[send.region(), recv.region()])?;
        // SAFETY: both buffers belong to the communicator's context and hold `count * rank_count` elements of `T`, the stream shares that context, and the caller keeps every operand alive through completion.
        unsafe {
            try_ffi!(sys::ncclAlltoAllConfig(
                send.as_ptr().cast(),
                recv.as_mut_ptr().cast(),
                count,
                T::DATA_TYPE.into(),
                self.as_raw(),
                stream.as_raw(),
                &raw const options.value,
            ))?;
        }
        Ok(())
    }

    /// Enqueues a gather of every rank's `send` into `recv` on `root`, in rank order.
    ///
    /// The root rank passes `Some` output of `send.len() * rank_count` elements; other ranks pass `None`.
    /// The root's `send` may alias its own segment of `recv`.
    ///
    /// # Safety
    ///
    /// Follow the crate's [raw execution](crate#raw-execution) contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not active, a buffer or the stream belongs to another CUDA context, the root omits `recv` or passes one of the wrong length, a length exceeds NCCL's range, `root` is not a rank of the communicator, `options` is invalid, or NCCL rejects the collective.
    pub unsafe fn gather_async<T: DataTypeLike>(
        &self,
        send: DeviceView<'_, T>,
        mut recv: Option<DeviceOutputView<'_, T>>,
        root: i32,
        options: &CollectiveOptions<'_>,
        stream: &Stream,
    ) -> Result<()> {
        self.ensure_buffer_context(send.context())?;
        if let Some(buffer) = &recv {
            self.ensure_buffer_context(buffer.context())?;
        }

        self.ensure_stream(stream)?;
        self.ensure_rank(root, "root")?;
        let options = options.prepare(self)?;
        if self.rank()? == root {
            let rank_count = to_usize(self.rank_count()?, "rank_count")?;
            let expected_len = checked_mul(send.len(), rank_count, "gather recv length")?;
            let Some(recv_buffer) = &recv else {
                return Err(Error::NullPointer {
                    name: "gather recv buffer".into(),
                });
            };
            if recv_buffer.len() != expected_len {
                return Err(Error::LengthMismatch {
                    name: "gather recv buffer".into(),
                    expected: expected_len,
                    actual: recv_buffer.len(),
                });
            }
        }

        let count = to_u64(send.len(), "gather count")?;
        {
            let mut regions = vec![send.region()];
            regions.extend(recv.as_ref().map(DeviceRegionProvider::region));
            stream.join_regions(&regions)?;
        }

        // SAFETY: every buffer belongs to the communicator's context, the root's output holds `count * rank_count` elements of `T`, the stream shares that context, and the caller keeps every operand alive through completion.
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
                &raw const options.value,
            ))?;
        }
        Ok(())
    }

    /// Enqueues a scatter of `send` on `root` into `recv` on every rank, one `recv.len()` segment per rank in rank order.
    ///
    /// The root rank passes `Some` input of `recv.len() * rank_count` elements; other ranks pass `None`.
    /// The root's `recv` may alias its own segment of `send`.
    ///
    /// # Safety
    ///
    /// Follow the crate's [raw execution](crate#raw-execution) contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not active, a buffer or the stream belongs to another CUDA context, the root omits `send` or passes one of the wrong length, a length exceeds NCCL's range, `root` is not a rank of the communicator, `options` is invalid, or NCCL rejects the collective.
    pub unsafe fn scatter_async<T: DataTypeLike>(
        &self,
        send: Option<DeviceView<'_, T>>,
        mut recv: DeviceOutputView<'_, T>,
        root: i32,
        options: &CollectiveOptions<'_>,
        stream: &Stream,
    ) -> Result<()> {
        if let Some(buffer) = &send {
            self.ensure_buffer_context(buffer.context())?;
        }

        self.ensure_buffer_context(recv.context())?;
        self.ensure_stream(stream)?;
        self.ensure_rank(root, "root")?;
        let options = options.prepare(self)?;
        if self.rank()? == root {
            let rank_count = to_usize(self.rank_count()?, "rank_count")?;
            let expected_len = checked_mul(recv.len(), rank_count, "scatter send length")?;
            let Some(send_buffer) = &send else {
                return Err(Error::NullPointer {
                    name: "scatter send buffer".into(),
                });
            };
            if send_buffer.len() != expected_len {
                return Err(Error::LengthMismatch {
                    name: "scatter send buffer".into(),
                    expected: expected_len,
                    actual: send_buffer.len(),
                });
            }
        }

        let count = to_u64(recv.len(), "scatter count")?;
        {
            let mut regions = vec![recv.region()];
            regions.extend(send.as_ref().map(DeviceView::region));
            stream.join_regions(&regions)?;
        }

        // SAFETY: every buffer belongs to the communicator's context, the root's input holds `count * rank_count` elements of `T`, the stream shares that context, and the caller keeps every operand alive through completion.
        unsafe {
            try_ffi!(sys::ncclScatterConfig(
                send.map_or(ptr::null(), |send| send.as_ptr().cast()),
                recv.as_mut_ptr().cast(),
                count,
                T::DATA_TYPE.into(),
                root,
                self.as_raw(),
                stream.as_raw(),
                &raw const options.value,
            ))?;
        }
        Ok(())
    }

    /// Enqueues a reduction of `send` with `op` whose result is scattered across ranks: rank `i` receives segment `i` in `recv`.
    ///
    /// `send` holds `recv.len() * rank_count` elements; `recv` may alias the local rank's segment of `send`.
    ///
    /// # Safety
    ///
    /// Follow the crate's [raw execution](crate#raw-execution) contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not active, a buffer or the stream belongs to another CUDA context, `send` has the wrong length, a length exceeds NCCL's range, `options` is invalid, a custom `op` belongs to another communicator, or NCCL rejects the collective.
    pub unsafe fn reduce_scatter_async<'r, T: DataTypeLike>(
        &self,
        send: DeviceView<'_, T>,
        mut recv: DeviceOutputView<'_, T>,
        op: impl Into<Reduction<'r, T>>,
        options: &CollectiveOptions<'_>,
        stream: &Stream,
    ) -> Result<()> {
        self.ensure_buffer_context(send.context())?;
        self.ensure_buffer_context(recv.context())?;
        let rank_count = to_usize(self.rank_count()?, "rank_count")?;
        let expected_len = checked_mul(recv.len(), rank_count, "reduce_scatter send length")?;
        if send.len() != expected_len {
            return Err(Error::LengthMismatch {
                name: "reduce_scatter send buffer".into(),
                expected: expected_len,
                actual: send.len(),
            });
        }

        self.ensure_stream(stream)?;
        let op = self.reduction::<T>(op.into())?;
        let options = options.prepare(self)?;
        let recv_count = to_u64(recv.len(), "reduce_scatter recvcount")?;
        stream.join_regions(&[send.region(), recv.region()])?;
        // SAFETY: both buffers belong to the communicator's context, `send` holds `recv_count * rank_count` elements of `T`, the stream shares that context, and the caller keeps every operand alive through completion.
        unsafe {
            try_ffi!(sys::ncclReduceScatterConfig(
                send.as_ptr().cast(),
                recv.as_mut_ptr().cast(),
                recv_count,
                T::DATA_TYPE.into(),
                op,
                self.as_raw(),
                stream.as_raw(),
                &raw const options.value,
            ))?;
        }
        Ok(())
    }

    /// Enqueues a send of `send` to `peer`, which must enqueue a matching [`Self::recv_async`] of the same type and length.
    ///
    /// The transfer blocks the stream until the peer receives it.
    /// Enqueue sends and receives that must progress together inside one [`Group`](crate::group::Group).
    ///
    /// # Safety
    ///
    /// Follow the crate's [raw execution](crate#raw-execution) contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not active, the buffer or the stream belongs to another CUDA context, the length exceeds NCCL's range, `peer` is not a rank of the communicator, or NCCL rejects the transfer.
    pub unsafe fn send_async<T: DataTypeLike>(
        &self,
        send: DeviceView<'_, T>,
        peer: i32,
        stream: &Stream,
    ) -> Result<()> {
        self.ensure_buffer_context(send.context())?;
        self.ensure_stream(stream)?;
        self.ensure_rank(peer, "peer")?;
        let count = to_u64(send.len(), "send count")?;
        stream.join_regions(&[send.region()])?;
        // SAFETY: the buffer holds `count` elements of `T` in the communicator's context, the stream shares that context, and the caller keeps it alive through completion.
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

    /// Enqueues a receive from `peer` into `recv`; the peer must enqueue a matching [`Self::send_async`] of the same type and length.
    ///
    /// The transfer blocks the stream until the data arrives.
    /// Enqueue sends and receives that must progress together inside one [`Group`](crate::group::Group).
    ///
    /// # Safety
    ///
    /// Follow the crate's [raw execution](crate#raw-execution) contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not active, the buffer or the stream belongs to another CUDA context, the length exceeds NCCL's range, `peer` is not a rank of the communicator, or NCCL rejects the transfer.
    pub unsafe fn recv_async<T: DataTypeLike>(
        &self,
        mut recv: DeviceOutputView<'_, T>,
        peer: i32,
        stream: &Stream,
    ) -> Result<()> {
        self.ensure_buffer_context(recv.context())?;
        self.ensure_stream(stream)?;
        self.ensure_rank(peer, "peer")?;
        let count = to_u64(recv.len(), "recv count")?;
        stream.join_regions(&[recv.region()])?;
        // SAFETY: the buffer holds `count` elements of `T` in the communicator's context, the stream shares that context, and the caller keeps it alive through completion.
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
