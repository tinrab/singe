use singe_cuda::{
    stream::StreamScope,
    submission::{Access, AccessMode},
    view::{DeviceOutputView, DeviceRegionProvider, DeviceView, DeviceViewMut},
};

use crate::{
    collective::CollectiveOptions,
    communicator::{Communicator, Reduction},
    error::Result,
    types::DataTypeLike,
};

// Each `_scoped` form borrows the communicator, its operands, and its operator for `'env`, which the scope keeps alive until it drains.
// `enqueue_scoped` rejects an open group and finishes nonblocking submission before the scope can drain.
impl Communicator {
    /// Enqueues [`Self::all_reduce_async`] on the scope's stream, which the scope waits for.
    ///
    /// The collective blocks the stream until every rank runs its matching call, so the scope's wait returns only after all ranks participate.
    ///
    /// # Errors
    ///
    /// Returns an error if an NCCL group is open on this thread, or [`Self::all_reduce_async`] or nonblocking submission fails.
    pub fn all_reduce_scoped<'env, T: DataTypeLike>(
        &'env self,
        send: DeviceView<'env, T>,
        recv: DeviceOutputView<'env, T>,
        op: impl Into<Reduction<'env, T>>,
        options: &CollectiveOptions<'_>,
        scope: &StreamScope<'_, 'env>,
    ) -> Result<()> {
        let accesses = [
            Access::new(send.region(), AccessMode::Read),
            Access::new(recv.region(), AccessMode::Write),
        ];
        let op = op.into();
        self.enqueue_scoped(&accesses, scope, |stream| {
            // SAFETY: every operand is borrowed for `'env`, and `enqueue_scoped` keeps the work inside the scope.
            unsafe { self.all_reduce_async(send, recv, op, options, stream) }
        })
    }

    /// Enqueues [`Self::all_reduce_in_place_async`] on the scope's stream, which the scope waits for.
    ///
    /// The collective blocks the stream until every rank runs its matching call, so the scope's wait returns only after all ranks participate.
    ///
    /// # Errors
    ///
    /// Returns an error if an NCCL group is open on this thread, or [`Self::all_reduce_in_place_async`] or nonblocking submission fails.
    pub fn all_reduce_in_place_scoped<'env, T: DataTypeLike>(
        &'env self,
        buffer: DeviceViewMut<'env, T>,
        op: impl Into<Reduction<'env, T>>,
        options: &CollectiveOptions<'_>,
        scope: &StreamScope<'_, 'env>,
    ) -> Result<()> {
        let accesses = [Access::new(buffer.region(), AccessMode::ReadWrite)];
        let op = op.into();
        self.enqueue_scoped(&accesses, scope, |stream| {
            // SAFETY: every operand is borrowed for `'env`, and `enqueue_scoped` keeps the work inside the scope.
            unsafe { self.all_reduce_in_place_async(buffer, op, options, stream) }
        })
    }

    /// Enqueues [`Self::broadcast_async`] on the scope's stream, which the scope waits for.
    ///
    /// The collective blocks the stream until every rank runs its matching call, so the scope's wait returns only after all ranks participate.
    ///
    /// # Errors
    ///
    /// Returns an error if an NCCL group is open on this thread, or [`Self::broadcast_async`] or nonblocking submission fails.
    pub fn broadcast_scoped<'env, T: DataTypeLike>(
        &'env self,
        send: DeviceView<'env, T>,
        recv: DeviceOutputView<'env, T>,
        root: i32,
        options: &CollectiveOptions<'_>,
        scope: &StreamScope<'_, 'env>,
    ) -> Result<()> {
        let accesses = [
            Access::new(send.region(), AccessMode::Read),
            Access::new(recv.region(), AccessMode::Write),
        ];
        self.enqueue_scoped(&accesses, scope, |stream| {
            // SAFETY: every operand is borrowed for `'env`, and `enqueue_scoped` keeps the work inside the scope.
            unsafe { self.broadcast_async(send, recv, root, options, stream) }
        })
    }

    /// Enqueues [`Self::broadcast_in_place_async`] on the scope's stream, which the scope waits for.
    ///
    /// The collective blocks the stream until every rank runs its matching call, so the scope's wait returns only after all ranks participate.
    ///
    /// # Errors
    ///
    /// Returns an error if an NCCL group is open on this thread, or [`Self::broadcast_in_place_async`] or nonblocking submission fails.
    pub fn broadcast_in_place_scoped<'env, T: DataTypeLike>(
        &'env self,
        buffer: DeviceViewMut<'env, T>,
        root: i32,
        options: &CollectiveOptions<'_>,
        scope: &StreamScope<'_, 'env>,
    ) -> Result<()> {
        let accesses = [Access::new(buffer.region(), AccessMode::ReadWrite)];
        self.enqueue_scoped(&accesses, scope, |stream| {
            // SAFETY: every operand is borrowed for `'env`, and `enqueue_scoped` keeps the work inside the scope.
            unsafe { self.broadcast_in_place_async(buffer, root, options, stream) }
        })
    }

    /// Enqueues [`Self::reduce_async`] on the scope's stream, which the scope waits for.
    ///
    /// The collective blocks the stream until every rank runs its matching call, so the scope's wait returns only after all ranks participate.
    ///
    /// # Errors
    ///
    /// Returns an error if an NCCL group is open on this thread, or [`Self::reduce_async`] or nonblocking submission fails.
    pub fn reduce_scoped<'env, T: DataTypeLike>(
        &'env self,
        send: DeviceView<'env, T>,
        recv: Option<DeviceOutputView<'env, T>>,
        op: impl Into<Reduction<'env, T>>,
        root: i32,
        options: &CollectiveOptions<'_>,
        scope: &StreamScope<'_, 'env>,
    ) -> Result<()> {
        let mut accesses = vec![Access::new(send.region(), AccessMode::Read)];
        accesses.extend(
            recv.as_ref()
                .map(|recv| Access::new(recv.region(), AccessMode::Write)),
        );
        let op = op.into();
        self.enqueue_scoped(&accesses, scope, |stream| {
            // SAFETY: every operand is borrowed for `'env`, and `enqueue_scoped` keeps the work inside the scope.
            unsafe { self.reduce_async(send, recv, op, root, options, stream) }
        })
    }

    /// Enqueues [`Self::all_gather_async`] on the scope's stream, which the scope waits for.
    ///
    /// The collective blocks the stream until every rank runs its matching call, so the scope's wait returns only after all ranks participate.
    ///
    /// # Errors
    ///
    /// Returns an error if an NCCL group is open on this thread, or [`Self::all_gather_async`] or nonblocking submission fails.
    pub fn all_gather_scoped<'env, T: DataTypeLike>(
        &'env self,
        send: DeviceView<'env, T>,
        recv: DeviceOutputView<'env, T>,
        options: &CollectiveOptions<'_>,
        scope: &StreamScope<'_, 'env>,
    ) -> Result<()> {
        let accesses = [
            Access::new(send.region(), AccessMode::Read),
            Access::new(recv.region(), AccessMode::Write),
        ];
        self.enqueue_scoped(&accesses, scope, |stream| {
            // SAFETY: every operand is borrowed for `'env`, and `enqueue_scoped` keeps the work inside the scope.
            unsafe { self.all_gather_async(send, recv, options, stream) }
        })
    }

    /// Enqueues [`Self::all_to_all_async`] on the scope's stream, which the scope waits for.
    ///
    /// The collective blocks the stream until every rank runs its matching call, so the scope's wait returns only after all ranks participate.
    ///
    /// # Errors
    ///
    /// Returns an error if an NCCL group is open on this thread, or [`Self::all_to_all_async`] or nonblocking submission fails.
    pub fn all_to_all_scoped<'env, T: DataTypeLike>(
        &'env self,
        send: DeviceView<'env, T>,
        recv: DeviceOutputView<'env, T>,
        options: &CollectiveOptions<'_>,
        scope: &StreamScope<'_, 'env>,
    ) -> Result<()> {
        let accesses = [
            Access::new(send.region(), AccessMode::Read),
            Access::new(recv.region(), AccessMode::Write),
        ];
        self.enqueue_scoped(&accesses, scope, |stream| {
            // SAFETY: every operand is borrowed for `'env`, and `enqueue_scoped` keeps the work inside the scope.
            unsafe { self.all_to_all_async(send, recv, options, stream) }
        })
    }

    /// Enqueues [`Self::gather_async`] on the scope's stream, which the scope waits for.
    ///
    /// The collective blocks the stream until every rank runs its matching call, so the scope's wait returns only after all ranks participate.
    ///
    /// # Errors
    ///
    /// Returns an error if an NCCL group is open on this thread, or [`Self::gather_async`] or nonblocking submission fails.
    pub fn gather_scoped<'env, T: DataTypeLike>(
        &'env self,
        send: DeviceView<'env, T>,
        recv: Option<DeviceOutputView<'env, T>>,
        root: i32,
        options: &CollectiveOptions<'_>,
        scope: &StreamScope<'_, 'env>,
    ) -> Result<()> {
        let mut accesses = vec![Access::new(send.region(), AccessMode::Read)];
        accesses.extend(
            recv.as_ref()
                .map(|recv| Access::new(recv.region(), AccessMode::Write)),
        );
        self.enqueue_scoped(&accesses, scope, |stream| {
            // SAFETY: every operand is borrowed for `'env`, and `enqueue_scoped` keeps the work inside the scope.
            unsafe { self.gather_async(send, recv, root, options, stream) }
        })
    }

    /// Enqueues [`Self::scatter_async`] on the scope's stream, which the scope waits for.
    ///
    /// The collective blocks the stream until every rank runs its matching call, so the scope's wait returns only after all ranks participate.
    ///
    /// # Errors
    ///
    /// Returns an error if an NCCL group is open on this thread, or [`Self::scatter_async`] or nonblocking submission fails.
    pub fn scatter_scoped<'env, T: DataTypeLike>(
        &'env self,
        send: Option<DeviceView<'env, T>>,
        recv: DeviceOutputView<'env, T>,
        root: i32,
        options: &CollectiveOptions<'_>,
        scope: &StreamScope<'_, 'env>,
    ) -> Result<()> {
        let mut accesses = vec![Access::new(recv.region(), AccessMode::Write)];
        accesses.extend(
            send.as_ref()
                .map(|send| Access::new(send.region(), AccessMode::Read)),
        );
        self.enqueue_scoped(&accesses, scope, |stream| {
            // SAFETY: every operand is borrowed for `'env`, and `enqueue_scoped` keeps the work inside the scope.
            unsafe { self.scatter_async(send, recv, root, options, stream) }
        })
    }

    /// Enqueues [`Self::reduce_scatter_async`] on the scope's stream, which the scope waits for.
    ///
    /// The collective blocks the stream until every rank runs its matching call, so the scope's wait returns only after all ranks participate.
    ///
    /// # Errors
    ///
    /// Returns an error if an NCCL group is open on this thread, or [`Self::reduce_scatter_async`] or nonblocking submission fails.
    pub fn reduce_scatter_scoped<'env, T: DataTypeLike>(
        &'env self,
        send: DeviceView<'env, T>,
        recv: DeviceOutputView<'env, T>,
        op: impl Into<Reduction<'env, T>>,
        options: &CollectiveOptions<'_>,
        scope: &StreamScope<'_, 'env>,
    ) -> Result<()> {
        let accesses = [
            Access::new(send.region(), AccessMode::Read),
            Access::new(recv.region(), AccessMode::Write),
        ];
        let op = op.into();
        self.enqueue_scoped(&accesses, scope, |stream| {
            // SAFETY: every operand is borrowed for `'env`, and `enqueue_scoped` keeps the work inside the scope.
            unsafe { self.reduce_scatter_async(send, recv, op, options, stream) }
        })
    }

    /// Enqueues [`Self::send_async`] on the scope's stream, which the scope waits for.
    ///
    /// The send blocks the stream until `peer` receives it, so the matching receive must run on another thread or process.
    ///
    /// # Errors
    ///
    /// Returns an error if an NCCL group is open on this thread, or [`Self::send_async`] or nonblocking submission fails.
    pub fn send_scoped<'env, T: DataTypeLike>(
        &'env self,
        send: DeviceView<'env, T>,
        peer: i32,
        scope: &StreamScope<'_, 'env>,
    ) -> Result<()> {
        let accesses = [Access::new(send.region(), AccessMode::Read)];
        self.enqueue_scoped(&accesses, scope, |stream| {
            // SAFETY: every operand is borrowed for `'env`, and `enqueue_scoped` keeps the work inside the scope.
            unsafe { self.send_async(send, peer, stream) }
        })
    }

    /// Enqueues [`Self::recv_async`] on the scope's stream, which the scope waits for.
    ///
    /// The receive blocks the stream until `peer` sends, so the matching send must run on another thread or process.
    ///
    /// # Errors
    ///
    /// Returns an error if an NCCL group is open on this thread, or [`Self::recv_async`] or nonblocking submission fails.
    pub fn recv_scoped<'env, T: DataTypeLike>(
        &'env self,
        recv: DeviceOutputView<'env, T>,
        peer: i32,
        scope: &StreamScope<'_, 'env>,
    ) -> Result<()> {
        let accesses = [Access::new(recv.region(), AccessMode::Write)];
        self.enqueue_scoped(&accesses, scope, |stream| {
            // SAFETY: every operand is borrowed for `'env`, and `enqueue_scoped` keeps the work inside the scope.
            unsafe { self.recv_async(recv, peer, stream) }
        })
    }
}
