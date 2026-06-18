use std::ptr;

use singe_cuda::{memory::DeviceMemory, stream::Stream};

use crate::{
    buffer::{BufferMut, BufferRef},
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
    /// The operation is in-place when `send` and `recv` refer to the same device buffer.
    /// Work is enqueued on `stream`; asynchronous failures may be reported later
    /// through [`Communicator::async_error`].
    ///
    /// # Errors
    ///
    /// Returns an error if the buffers have different lengths, the stream is not
    /// associated with the communicator's CUDA context, the element count is too
    /// large for NCCL, or NCCL rejects the collective.
    pub fn all_reduce<T: DataTypeLike>(
        &self,
        send: BufferRef<'_, T>,
        mut recv: BufferMut<'_, T>,
        op: ReductionOperator,
        stream: &Stream,
    ) -> Result<()> {
        if send.len() != recv.len() {
            return Err(Error::LengthMismatch {
                name: "all_reduce buffers".into(),
            });
        }

        self.ensure_stream(stream)?;
        let count = to_u64(send.len(), "all_reduce count")?;
        unsafe {
            try_ffi!(sys::ncclAllReduce(
                send.as_ptr().cast(),
                recv.as_mut_ptr().cast(),
                count,
                T::nccl_data_type().into(),
                op.into(),
                self.as_raw(),
                stream.as_raw(),
            ))?;
        }
        Ok(())
    }

    pub fn all_reduce_memory<T: DataTypeLike>(
        &self,
        send: &DeviceMemory<T>,
        recv: &mut DeviceMemory<T>,
        op: ReductionOperator,
        stream: &Stream,
    ) -> Result<()> {
        self.all_reduce(
            BufferRef::from_device_memory(send)?,
            BufferMut::from_device_memory(recv)?,
            op,
            stream,
        )
    }

    /// Broadcasts the contents of `send` from `root` into `recv` on every rank.
    /// `send` is only read on the `root` rank. Non-root ranks still pass a buffer, but NCCL ignores its contents.
    /// `send` and `recv` must have the same length.
    ///
    /// The operation is in-place when `send` and `recv` refer to the same device buffer.
    /// Work is enqueued on `stream`; asynchronous failures may be reported later
    /// through [`Communicator::async_error`].
    ///
    /// # Errors
    ///
    /// Returns an error if the buffers have different lengths, the stream is not
    /// associated with the communicator's CUDA context, the element count is too
    /// large for NCCL, or NCCL rejects the collective.
    pub fn broadcast<T: DataTypeLike>(
        &self,
        send: BufferRef<'_, T>,
        mut recv: BufferMut<'_, T>,
        root: i32,
        stream: &Stream,
    ) -> Result<()> {
        if send.len() != recv.len() {
            return Err(Error::LengthMismatch {
                name: "broadcast buffers".into(),
            });
        }

        self.ensure_stream(stream)?;
        let count = to_u64(send.len(), "broadcast count")?;
        unsafe {
            try_ffi!(sys::ncclBroadcast(
                send.as_ptr().cast(),
                recv.as_mut_ptr().cast(),
                count,
                T::nccl_data_type().into(),
                root,
                self.as_raw(),
                stream.as_raw(),
            ))?;
        }
        Ok(())
    }

    /// Broadcasts `buffer` from `root` to every rank using the same buffer for both input and output.
    ///
    /// `buffer.len()` determines the element count.
    pub fn broadcast_in_place<T: DataTypeLike>(
        &self,
        mut buffer: BufferMut<'_, T>,
        root: i32,
        stream: &Stream,
    ) -> Result<()> {
        self.ensure_stream(stream)?;
        let count = to_u64(buffer.len(), "broadcast count")?;
        unsafe {
            try_ffi!(sys::ncclBroadcast(
                buffer.as_ptr().cast(),
                buffer.as_mut_ptr().cast(),
                count,
                T::nccl_data_type().into(),
                root,
                self.as_raw(),
                stream.as_raw(),
            ))?;
        }
        Ok(())
    }

    pub fn broadcast_memory<T: DataTypeLike>(
        &self,
        send: &DeviceMemory<T>,
        recv: &mut DeviceMemory<T>,
        root: i32,
        stream: &Stream,
    ) -> Result<()> {
        self.broadcast(
            BufferRef::from_device_memory(send)?,
            BufferMut::from_device_memory(recv)?,
            root,
            stream,
        )
    }

    /// Reduces the values in `send` with `op` and writes the result to `recv` on the `root` rank.
    /// On the `root` rank, `recv` must be `Some(...)` and must have the same length as `send`.
    /// On non-root ranks, pass `None` for `recv`.
    ///
    /// The operation is in-place when the root rank uses the same device buffer for both `send` and `recv`.
    pub fn reduce<T: DataTypeLike>(
        &self,
        send: BufferRef<'_, T>,
        recv: Option<BufferMut<'_, T>>,
        op: ReductionOperator,
        root: i32,
        stream: &Stream,
    ) -> Result<()> {
        self.ensure_stream(stream)?;
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
        unsafe {
            try_ffi!(sys::ncclReduce(
                send.as_ptr().cast(),
                recv.as_mut()
                    .map_or(ptr::null_mut(), |recv| recv.as_mut_ptr().cast()),
                count,
                T::nccl_data_type().into(),
                op.into(),
                root,
                self.as_raw(),
                stream.as_raw(),
            ))?;
        }
        Ok(())
    }

    pub fn reduce_memory<T: DataTypeLike>(
        &self,
        send: &DeviceMemory<T>,
        recv: Option<&mut DeviceMemory<T>>,
        op: ReductionOperator,
        root: i32,
        stream: &Stream,
    ) -> Result<()> {
        self.reduce(
            BufferRef::from_device_memory(send)?,
            recv.map(BufferMut::from_device_memory).transpose()?,
            op,
            root,
            stream,
        )
    }

    /// Gathers each rank's `send` buffer into `recv` on every rank.
    /// Rank `i` contributes its `send` slice to the `i`-th contiguous segment of `recv`.
    ///
    /// `recv` must have length `send.len() * self.rank_count()?`.
    ///
    /// The operation is in-place when `send` aliases the segment of `recv` corresponding to the local rank.
    pub fn all_gather<T: DataTypeLike>(
        &self,
        send: BufferRef<'_, T>,
        mut recv: BufferMut<'_, T>,
        stream: &Stream,
    ) -> Result<()> {
        let rank_count = to_usize(self.rank_count()?, "rank_count")?;
        let expected_len = send
            .len()
            .checked_mul(rank_count)
            .ok_or(Error::OutOfRange {
                name: "all_gather recv length".into(),
            })?;
        if recv.len() != expected_len {
            return Err(Error::LengthMismatch {
                name: "all_gather recv buffer".into(),
            });
        }

        self.ensure_stream(stream)?;
        let send_count = to_u64(send.len(), "all_gather sendcount")?;
        unsafe {
            try_ffi!(sys::ncclAllGather(
                send.as_ptr().cast(),
                recv.as_mut_ptr().cast(),
                send_count,
                T::nccl_data_type().into(),
                self.as_raw(),
                stream.as_raw(),
            ))?;
        }
        Ok(())
    }

    pub fn all_gather_memory<T: DataTypeLike>(
        &self,
        send: &DeviceMemory<T>,
        recv: &mut DeviceMemory<T>,
        stream: &Stream,
    ) -> Result<()> {
        self.all_gather(
            BufferRef::from_device_memory(send)?,
            BufferMut::from_device_memory(recv)?,
            stream,
        )
    }

    /// Performs an all-to-all exchange.
    /// `send` is partitioned into one equally sized segment per rank; segment `j` is sent to rank `j`.
    /// `recv` is partitioned the same way, and segment `i` receives the data sent by rank `i`.
    ///
    /// `send` and `recv` must have the same length, and that length must be divisible by `self.rank_count()?`.
    ///
    /// In-place operation is not supported.
    pub fn all_to_all<T: DataTypeLike>(
        &self,
        send: BufferRef<'_, T>,
        mut recv: BufferMut<'_, T>,
        stream: &Stream,
    ) -> Result<()> {
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
        let count = to_u64(send.len() / rank_count.max(1), "all_to_all count")?;
        unsafe {
            try_ffi!(sys::ncclAlltoAll(
                send.as_ptr().cast(),
                recv.as_mut_ptr().cast(),
                count,
                T::nccl_data_type().into(),
                self.as_raw(),
                stream.as_raw(),
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
    pub fn gather<T: DataTypeLike>(
        &self,
        send: BufferRef<'_, T>,
        recv: Option<BufferMut<'_, T>>,
        root: i32,
        stream: &Stream,
    ) -> Result<()> {
        self.ensure_stream(stream)?;
        let rank = self.rank()?;
        let mut recv = recv;
        if rank == root {
            let rank_count = to_usize(self.rank_count()?, "rank_count")?;
            let expected_len = send
                .len()
                .checked_mul(rank_count)
                .ok_or(Error::OutOfRange {
                    name: "gather recv length".into(),
                })?;
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
        unsafe {
            try_ffi!(sys::ncclGather(
                send.as_ptr().cast(),
                recv.as_mut()
                    .map_or(ptr::null_mut(), |recv| recv.as_mut_ptr().cast()),
                count,
                T::nccl_data_type().into(),
                root,
                self.as_raw(),
                stream.as_raw(),
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
    pub fn scatter<T: DataTypeLike>(
        &self,
        send: Option<BufferRef<'_, T>>,
        mut recv: BufferMut<'_, T>,
        root: i32,
        stream: &Stream,
    ) -> Result<()> {
        self.ensure_stream(stream)?;
        let rank = self.rank()?;
        if rank == root {
            let rank_count = to_usize(self.rank_count()?, "rank_count")?;
            let expected_len = recv
                .len()
                .checked_mul(rank_count)
                .ok_or(Error::OutOfRange {
                    name: "scatter send length".into(),
                })?;
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
        unsafe {
            try_ffi!(sys::ncclScatter(
                send.map_or(ptr::null(), |send| send.as_ptr().cast()),
                recv.as_mut_ptr().cast(),
                count,
                T::nccl_data_type().into(),
                root,
                self.as_raw(),
                stream.as_raw(),
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
    pub fn reduce_scatter<T: DataTypeLike>(
        &self,
        send: BufferRef<'_, T>,
        mut recv: BufferMut<'_, T>,
        op: ReductionOperator,
        stream: &Stream,
    ) -> Result<()> {
        let rank_count = to_usize(self.rank_count()?, "rank_count")?;
        let expected_len = recv
            .len()
            .checked_mul(rank_count)
            .ok_or(Error::OutOfRange {
                name: "reduce_scatter send length".into(),
            })?;
        if send.len() != expected_len {
            return Err(Error::LengthMismatch {
                name: "reduce_scatter send buffer".into(),
            });
        }

        self.ensure_stream(stream)?;
        let recv_count = to_u64(recv.len(), "reduce_scatter recvcount")?;
        unsafe {
            try_ffi!(sys::ncclReduceScatter(
                send.as_ptr().cast(),
                recv.as_mut_ptr().cast(),
                recv_count,
                T::nccl_data_type().into(),
                op.into(),
                self.as_raw(),
                stream.as_raw(),
            ))?;
        }
        Ok(())
    }

    pub fn reduce_scatter_memory<T: DataTypeLike>(
        &self,
        send: &DeviceMemory<T>,
        recv: &mut DeviceMemory<T>,
        op: ReductionOperator,
        stream: &Stream,
    ) -> Result<()> {
        self.reduce_scatter(
            BufferRef::from_device_memory(send)?,
            BufferMut::from_device_memory(recv)?,
            op,
            stream,
        )
    }

    /// Sends the contents of `send` to `peer`.
    ///
    /// Rank `peer` must call [`Communicator::recv`] with the same element type and the same buffer length.
    ///
    /// Blocks the GPU until the operation completes.
    /// If multiple [`Communicator::send`] and [`Communicator::recv`] operations must progress concurrently to complete, group them between [`group_start`](crate::group_start) and [`group_end`](crate::group_end).
    pub fn send<T: DataTypeLike>(
        &self,
        send: BufferRef<'_, T>,
        peer: i32,
        stream: &Stream,
    ) -> Result<()> {
        self.ensure_stream(stream)?;
        let count = to_u64(send.len(), "send count")?;
        unsafe {
            try_ffi!(sys::ncclSend(
                send.as_ptr().cast(),
                count,
                T::nccl_data_type().into(),
                peer,
                self.as_raw(),
                stream.as_raw(),
            ))?;
        }
        Ok(())
    }

    pub fn send_memory<T: DataTypeLike>(
        &self,
        send: &DeviceMemory<T>,
        peer: i32,
        stream: &Stream,
    ) -> Result<()> {
        self.send(BufferRef::from_device_memory(send)?, peer, stream)
    }

    /// Receives data from `peer` into `recv`.
    ///
    /// Rank `peer` must call [`Communicator::send`] with the same element type and the same buffer length.
    ///
    /// Blocks the GPU until the operation completes.
    /// If multiple [`Communicator::send`] and [`Communicator::recv`] operations must progress concurrently to complete, group them between [`group_start`](crate::group_start) and [`group_end`](crate::group_end).
    pub fn recv<T: DataTypeLike>(
        &self,
        mut recv: BufferMut<'_, T>,
        peer: i32,
        stream: &Stream,
    ) -> Result<()> {
        self.ensure_stream(stream)?;
        let count = to_u64(recv.len(), "recv count")?;
        unsafe {
            try_ffi!(sys::ncclRecv(
                recv.as_mut_ptr().cast(),
                count,
                T::nccl_data_type().into(),
                peer,
                self.as_raw(),
                stream.as_raw(),
            ))?;
        }
        Ok(())
    }

    pub fn recv_memory<T: DataTypeLike>(
        &self,
        recv: &mut DeviceMemory<T>,
        peer: i32,
        stream: &Stream,
    ) -> Result<()> {
        self.recv(BufferMut::from_device_memory(recv)?, peer, stream)
    }
}
