use std::{
    ops::{Index, IndexMut},
    process, slice, thread, vec,
};

use singe_cuda::{
    device::Device,
    memory::DeviceMemory,
    stream::Stream,
    view::{DeviceOutputView, DeviceView, DeviceViewMut},
};

use crate::{
    communicator::Communicator,
    error::{Error, Result, Status},
    group,
    group::Group,
    sys,
    types::{CommunicatorState, DataTypeLike, ReductionOperator},
    utility::{to_i32, to_usize},
};

/// A nonnegative communicator rank identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RankId(i32);

/// A nonnegative rank selected as the root of a collective.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RootRank(RankId);

/// A local rank group whose borrowed collectives return after successful completion.
/// Raw grouped scheduling remains available through [`Self::sync_scope`].
#[derive(Debug)]
pub struct LocalCommunicatorGroup {
    ranks: Vec<LocalRank>,
}

/// One local communicator rank and its dedicated CUDA stream.
#[derive(Debug)]
pub struct LocalRank {
    id: RankId,
    communicator: Communicator,
    stream: Stream,
}

/// Values stored in communicator-rank order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankVec<T> {
    values: Vec<T>,
}

/// An output value paired with the root rank that produced it.
#[derive(Debug)]
pub struct RootOutput<T> {
    root: RootRank,
    value: T,
}

impl RankId {
    /// Creates a nonnegative rank identifier.
    ///
    /// # Errors
    ///
    /// Returns an error if `value` is negative.
    pub fn create(value: i32) -> Result<Self> {
        if value < 0 {
            return Err(Error::out_of_range("rank"));
        }
        Ok(Self(value))
    }

    /// Returns the native rank number.
    #[must_use]
    pub const fn get(self) -> i32 {
        self.0
    }

    pub(crate) fn to_index(self) -> Result<usize> {
        to_usize(self.0, "rank")
    }
}

impl RootRank {
    /// Creates a nonnegative rank identifier.
    ///
    /// # Errors
    ///
    /// Returns an error if `value` is negative.
    pub fn create(value: i32) -> Result<Self> {
        Ok(Self(RankId::create(value)?))
    }

    /// Returns the root as a rank identifier.
    #[must_use]
    pub const fn id(self) -> RankId {
        self.0
    }

    /// Returns the native root rank number.
    #[must_use]
    pub const fn get(self) -> i32 {
        self.0.get()
    }
}

impl LocalCommunicatorGroup {
    /// Creates a local communicator group with one rank per supplied device ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested devices are unavailable, a count exceeds the native integer range, CUDA primary-context or stream setup fails, or NCCL cannot create the local clique.
    pub fn create_all(device_ids: &[i32]) -> Result<Self> {
        let communicators = Communicator::create_all(device_ids)?;
        Self::from_communicators(communicators)
    }

    /// Creates a local group using the first `device_count` CUDA devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested devices are unavailable, a count exceeds the native integer range, CUDA primary-context or stream setup fails, or NCCL cannot create the local clique.
    pub fn create_all_first_devices(device_count: i32) -> Result<Self> {
        let device_count = to_usize(device_count, "device_count")?;
        let device_ids = (0..device_count)
            .map(|device_id| to_i32(device_id, "device_id"))
            .collect::<Result<Vec<_>>>()?;
        Self::create_all(&device_ids)
    }

    /// Creates `rank_count` ranks on the first visible CUDA devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested devices are unavailable, a count exceeds the native integer range, CUDA primary-context or stream setup fails, or NCCL cannot create the local clique.
    pub fn create_all_visible(rank_count: usize) -> Result<Self> {
        let device_count = to_usize(Device::count()?, "device_count")?;
        if device_count < rank_count {
            return Err(Error::out_of_range("visible device count"));
        }
        let device_ids = (0..rank_count)
            .map(|device_id| to_i32(device_id, "device_id"))
            .collect::<Result<Vec<_>>>()?;
        Self::create_all(&device_ids)
    }

    /// Takes ownership of a complete local communicator clique and creates one stream per rank.
    ///
    /// Ranks are sorted into communicator-rank order.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator count exceeds the `i32` range, communicator sizes do not match, ranks do not form a complete clique, or querying NCCL metadata or creating CUDA streams fails.
    pub fn from_communicators(communicators: Vec<Communicator>) -> Result<Self> {
        let rank_count = to_i32(communicators.len(), "local rank count")?;
        let mut ranks = Vec::with_capacity(communicators.len());
        for communicator in communicators {
            if communicator.rank_count()? != rank_count {
                return Err(Error::LengthMismatch {
                    name: "local communicator clique".into(),
                });
            }
            let id = RankId::create(communicator.rank()?)?;
            let stream = communicator.cuda_context().create_stream()?;
            ranks.push(LocalRank {
                id,
                communicator,
                stream,
            });
        }
        ranks.sort_by_key(|rank| rank.id);
        for (index, rank) in ranks.iter().enumerate() {
            if rank.id.to_index()? != index {
                return Err(Error::out_of_range("communicator ranks"));
            }
        }
        Ok(Self { ranks })
    }

    /// Returns the number of local ranks.
    #[must_use]
    pub fn rank_count(&self) -> usize {
        self.ranks.len()
    }

    /// Returns whether the group contains no ranks.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ranks.is_empty()
    }

    /// Iterates over local ranks in communicator-rank order.
    pub fn ranks(&self) -> slice::Iter<'_, LocalRank> {
        self.ranks.iter()
    }

    /// Returns the local rank identified by `id`.
    ///
    /// # Errors
    ///
    /// Returns an error if the rank is outside the stored range.
    pub fn rank(&self, id: RankId) -> Result<&LocalRank> {
        let index = id.to_index()?;
        self.ranks
            .get(index)
            .ok_or_else(|| Error::out_of_range("rank"))
    }

    /// Applies `f` to each local rank and collects successful results in rank order.
    ///
    /// # Errors
    ///
    /// Returns the first error produced by `f`.
    pub fn map_ranks<T>(&self, mut f: impl FnMut(&LocalRank) -> Result<T>) -> Result<RankVec<T>> {
        self.ranks
            .iter()
            .map(&mut f)
            .collect::<Result<Vec<_>>>()
            .map(RankVec::from_vec)
    }

    /// Waits for every local rank stream to complete.
    ///
    /// # Errors
    ///
    /// Returns an error if synchronizing a rank stream fails.
    pub fn synchronize(&self) -> Result<()> {
        for rank in &self.ranks {
            rank.stream.synchronize()?;
        }
        Ok(())
    }

    /// Destroys every local communicator.
    ///
    /// # Errors
    ///
    /// Returns the first error produced while binding a CUDA context or destroying a communicator.
    pub fn destroy(self) -> Result<()> {
        for rank in self.ranks {
            rank.communicator.destroy()?;
        }
        Ok(())
    }

    /// Reduces equal-length inputs across ranks and writes the result to every rank.
    ///
    /// Returns after all local rank streams complete successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if per-rank counts or buffer lengths are incompatible with the collective, a buffer or stream belongs to another CUDA context.
    /// An already open NCCL group, communicator-state checks, CUDA context binding, NCCL grouping or execution, and stream synchronization can also produce errors.
    pub fn all_reduce<T: DataTypeLike>(
        &self,
        send: &RankVec<DeviceView<'_, T>>,
        recv: &mut RankVec<DeviceOutputView<'_, T>>,
        op: ReductionOperator,
    ) -> Result<()> {
        self.ensure_rank_vec(send, "all_reduce send")?;
        self.ensure_rank_vec(recv, "all_reduce recv")?;
        Self::ensure_same_len(send, "all_reduce send")?;
        Self::ensure_pair_lengths(send, recv, "all_reduce buffers")?;
        self.with_completed_group(|| {
            // SAFETY: all views remain borrowed until the completed group drains.
            unsafe {
                for ((rank, send), recv) in self.ranks.iter().zip(send.iter()).zip(recv.iter_mut())
                {
                    rank.communicator
                        .all_reduce(*send, recv.reborrow(), op, &rank.stream)?;
                }
                Ok(())
            }
        })
    }

    /// Reduces initialized buffers in place and waits for all ranks to complete.
    ///
    /// # Errors
    ///
    /// Returns an error if per-rank counts or buffer lengths are incompatible with the collective, a buffer or stream belongs to another CUDA context.
    /// An already open NCCL group, communicator-state checks, CUDA context binding, NCCL grouping or execution, and stream synchronization can also produce errors.
    pub fn all_reduce_in_place<T: DataTypeLike>(
        &self,
        buffers: &mut RankVec<DeviceViewMut<'_, T>>,
        op: ReductionOperator,
    ) -> Result<()> {
        self.ensure_rank_vec(buffers, "all_reduce buffers")?;
        let expected =
            buffers
                .iter()
                .next()
                .map(DeviceViewMut::len)
                .ok_or(Error::LengthMismatch {
                    name: "all_reduce buffers".into(),
                })?;
        ensure_lengths(
            buffers.iter().map(DeviceViewMut::len),
            expected,
            "all_reduce buffers",
        )?;
        self.with_completed_group(|| {
            for (rank, buffer) in self.ranks.iter().zip(buffers.iter_mut()) {
                // SAFETY: the completed group retains the exclusive borrows through execution.
                unsafe {
                    rank.communicator
                        .all_reduce_in_place(buffer.reborrow(), op, &rank.stream)?;
                }
            }
            Ok(())
        })
    }

    /// Copies the root input into every rank's output buffer.
    ///
    /// Returns after all local rank streams complete successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if per-rank counts or buffer lengths are incompatible with the collective, a buffer or stream belongs to another CUDA context.
    /// Returns an error if the root rank is outside this group.
    /// An already open NCCL group, communicator-state checks, CUDA context binding, NCCL grouping or execution, and stream synchronization can also produce errors.
    pub fn broadcast<T: DataTypeLike>(
        &self,
        send: &RankVec<DeviceView<'_, T>>,
        recv: &mut RankVec<DeviceOutputView<'_, T>>,
        root: RootRank,
    ) -> Result<()> {
        self.ensure_root(root)?;
        self.ensure_rank_vec(send, "broadcast send")?;
        self.ensure_rank_vec(recv, "broadcast recv")?;
        Self::ensure_same_len(send, "broadcast send")?;
        Self::ensure_pair_lengths(send, recv, "broadcast buffers")?;
        self.with_completed_group(|| {
            // SAFETY: all views remain borrowed until the completed group drains.
            unsafe {
                for ((rank, send), recv) in self.ranks.iter().zip(send.iter()).zip(recv.iter_mut())
                {
                    rank.communicator.broadcast(
                        *send,
                        recv.reborrow(),
                        root.get(),
                        &rank.stream,
                    )?;
                }
                Ok(())
            }
        })
    }

    /// Copies the root buffer into the buffers of all ranks.
    ///
    /// Returns after all local rank streams complete successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if per-rank counts or buffer lengths are incompatible with the collective, a buffer or stream belongs to another CUDA context.
    /// Returns an error if the root rank is outside this group.
    /// An already open NCCL group, communicator-state checks, CUDA context binding, NCCL grouping or execution, and stream synchronization can also produce errors.
    pub fn broadcast_in_place<T: DataTypeLike>(
        &self,
        buffers: &mut RankVec<DeviceViewMut<'_, T>>,
        root: RootRank,
    ) -> Result<()> {
        self.ensure_root(root)?;
        self.ensure_rank_vec(buffers, "broadcast buffers")?;
        let expected =
            buffers
                .iter()
                .next()
                .map(DeviceViewMut::len)
                .ok_or(Error::LengthMismatch {
                    name: "broadcast buffers".into(),
                })?;
        ensure_lengths(
            buffers.iter().map(DeviceViewMut::len),
            expected,
            "broadcast buffers",
        )?;
        self.with_completed_group(|| {
            // SAFETY: all views remain borrowed until the completed group drains.
            unsafe {
                for (rank, buffer) in self.ranks.iter().zip(buffers.iter_mut()) {
                    rank.communicator.broadcast_in_place(
                        buffer.reborrow(),
                        root.get(),
                        &rank.stream,
                    )?;
                }
                Ok(())
            }
        })
    }

    /// Reduces into a newly allocated root result and waits for completion.
    ///
    /// # Errors
    ///
    /// Returns an error if per-rank counts or buffer lengths are incompatible with the collective, a buffer or stream belongs to another CUDA context.
    /// Returns an error if the root rank is outside this group.
    /// Allocating the root output can also fail.
    /// An already open NCCL group, communicator-state checks, CUDA context binding, NCCL grouping or execution, and stream synchronization can also produce errors.
    pub fn reduce<T: DataTypeLike>(
        &self,
        send: &RankVec<DeviceView<'_, T>>,
        root: RootRank,
        op: ReductionOperator,
    ) -> Result<RootOutput<DeviceMemory<T>>> {
        self.ensure_root(root)?;
        self.ensure_rank_vec(send, "reduce send")?;
        let root_index = root.id().to_index()?;
        let mut result = DeviceMemory::<T>::create_uninit(
            self.ranks[root_index].communicator.cuda_context(),
            send[root_index].len(),
        )?;
        self.reduce_into(send, root, result.output(), op)?;
        // SAFETY: the completed collective initialized every compact root element.
        let result = unsafe { result.assume_init() };
        Ok(RootOutput {
            root,
            value: result,
        })
    }

    /// Reduces rank inputs into the provided root output buffer.
    ///
    /// Returns after all local rank streams complete successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if per-rank counts or buffer lengths are incompatible with the collective, a buffer or stream belongs to another CUDA context.
    /// Returns an error if the root rank is outside this group.
    /// An already open NCCL group, communicator-state checks, CUDA context binding, NCCL grouping or execution, and stream synchronization can also produce errors.
    pub fn reduce_into<T: DataTypeLike>(
        &self,
        send: &RankVec<DeviceView<'_, T>>,
        root: RootRank,
        mut root_recv: DeviceOutputView<'_, T>,
        op: ReductionOperator,
    ) -> Result<()> {
        self.ensure_root(root)?;
        self.ensure_rank_vec(send, "reduce send")?;
        Self::ensure_same_len(send, "reduce send")?;
        let send_len = Self::first_len(send, "reduce send")?;
        if root_recv.len() != send_len {
            return Err(Error::LengthMismatch {
                name: "reduce recv".into(),
            });
        }
        let root_index = root.id().to_index()?;
        self.with_completed_group(|| {
            // SAFETY: all views remain borrowed until the completed group drains.
            unsafe {
                let mut root_recv = Some(&mut root_recv);
                for (index, (rank, send)) in self.ranks.iter().zip(send.iter()).enumerate() {
                    let recv = if index == root_index {
                        root_recv.as_deref_mut()
                    } else {
                        None
                    };
                    rank.communicator.reduce(
                        *send,
                        recv.map(DeviceOutputView::reborrow),
                        op,
                        root.get(),
                        &rank.stream,
                    )?;
                }
                Ok(())
            }
        })
    }

    /// Concatenates every rank's input in rank order into each output buffer.
    ///
    /// Returns after all local rank streams complete successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if per-rank counts or buffer lengths are incompatible with the collective, a buffer or stream belongs to another CUDA context.
    /// An already open NCCL group, communicator-state checks, CUDA context binding, NCCL grouping or execution, and stream synchronization can also produce errors.
    pub fn all_gather<T: DataTypeLike>(
        &self,
        send: &RankVec<DeviceView<'_, T>>,
        recv: &mut RankVec<DeviceOutputView<'_, T>>,
    ) -> Result<()> {
        self.ensure_rank_vec(send, "all_gather send")?;
        self.ensure_rank_vec(recv, "all_gather recv")?;
        Self::ensure_same_len(send, "all_gather send")?;
        let expected_recv_len = Self::first_len(send, "all_gather send")?
            .checked_mul(self.rank_count())
            .ok_or_else(|| Error::out_of_range("all_gather recv"))?;
        Self::ensure_all_len(recv, expected_recv_len, "all_gather recv")?;
        self.with_completed_group(|| {
            // SAFETY: all views remain borrowed until the completed group drains.
            unsafe {
                for ((rank, send), recv) in self.ranks.iter().zip(send.iter()).zip(recv.iter_mut())
                {
                    rank.communicator
                        .all_gather(*send, recv.reborrow(), &rank.stream)?;
                }
                Ok(())
            }
        })
    }

    /// Concatenates rank inputs into the provided root output buffer.
    ///
    /// Returns after all local rank streams complete successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if per-rank counts or buffer lengths are incompatible with the collective, a buffer or stream belongs to another CUDA context.
    /// Returns an error if the root rank is outside this group.
    /// An already open NCCL group, communicator-state checks, CUDA context binding, NCCL grouping or execution, and stream synchronization can also produce errors.
    pub fn gather_into<T: DataTypeLike>(
        &self,
        send: &RankVec<DeviceView<'_, T>>,
        root: RootRank,
        mut root_recv: DeviceOutputView<'_, T>,
    ) -> Result<()> {
        self.ensure_root(root)?;
        self.ensure_rank_vec(send, "gather send")?;
        Self::ensure_same_len(send, "gather send")?;
        let expected_recv_len = Self::first_len(send, "gather send")?
            .checked_mul(self.rank_count())
            .ok_or_else(|| Error::out_of_range("gather recv"))?;
        if root_recv.len() != expected_recv_len {
            return Err(Error::LengthMismatch {
                name: "gather recv".into(),
            });
        }
        let root_index = root.id().to_index()?;
        self.with_completed_group(|| {
            // SAFETY: all views remain borrowed until the completed group drains.
            unsafe {
                let mut root_recv = Some(&mut root_recv);
                for (index, (rank, send)) in self.ranks.iter().zip(send.iter()).enumerate() {
                    let recv = if index == root_index {
                        root_recv.as_deref_mut()
                    } else {
                        None
                    };
                    rank.communicator.gather(
                        *send,
                        recv.map(DeviceOutputView::reborrow),
                        root.get(),
                        &rank.stream,
                    )?;
                }
                Ok(())
            }
        })
    }

    /// Distributes equal contiguous portions of the root input to rank outputs.
    ///
    /// Returns after all local rank streams complete successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if per-rank counts or buffer lengths are incompatible with the collective, a buffer or stream belongs to another CUDA context.
    /// Returns an error if the root rank is outside this group.
    /// An already open NCCL group, communicator-state checks, CUDA context binding, NCCL grouping or execution, and stream synchronization can also produce errors.
    pub fn scatter_from<T: DataTypeLike>(
        &self,
        root_send: DeviceView<'_, T>,
        recv: &mut RankVec<DeviceOutputView<'_, T>>,
        root: RootRank,
    ) -> Result<()> {
        self.ensure_root(root)?;
        self.ensure_rank_vec(recv, "scatter recv")?;
        let recv_len =
            recv.iter()
                .next()
                .map(DeviceOutputView::len)
                .ok_or(Error::LengthMismatch {
                    name: "scatter recv".into(),
                })?;
        Self::ensure_all_len(recv, recv_len, "scatter recv")?;
        let expected_send_len = recv_len
            .checked_mul(self.rank_count())
            .ok_or_else(|| Error::out_of_range("scatter send"))?;
        if root_send.len() != expected_send_len {
            return Err(Error::LengthMismatch {
                name: "scatter send".into(),
            });
        }
        let root_index = root.id().to_index()?;
        self.with_completed_group(|| {
            // SAFETY: all views remain borrowed until the completed group drains.
            unsafe {
                for (index, (rank, recv)) in self.ranks.iter().zip(recv.iter_mut()).enumerate() {
                    let send = (index == root_index).then_some(root_send);
                    rank.communicator
                        .scatter(send, recv.reborrow(), root.get(), &rank.stream)?;
                }
                Ok(())
            }
        })
    }

    /// Exchanges rank-indexed input portions so each rank receives a portion from every rank.
    ///
    /// Returns after all local rank streams complete successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if per-rank counts or buffer lengths are incompatible with the collective, a buffer or stream belongs to another CUDA context.
    /// An already open NCCL group, communicator-state checks, CUDA context binding, NCCL grouping or execution, and stream synchronization can also produce errors.
    pub fn all_to_all<T: DataTypeLike>(
        &self,
        send: &RankVec<DeviceView<'_, T>>,
        recv: &mut RankVec<DeviceOutputView<'_, T>>,
        elements_per_rank: usize,
    ) -> Result<()> {
        self.ensure_partitioned_rank_vec(send, elements_per_rank, "all_to_all send")?;
        self.ensure_rank_vec(recv, "all_to_all recv")?;
        Self::ensure_all_len(
            recv,
            partitioned_len(elements_per_rank, self.rank_count(), "all_to_all recv")?,
            "all_to_all recv",
        )?;
        self.with_completed_group(|| {
            // SAFETY: all views remain borrowed until the completed group drains.
            unsafe {
                for ((rank, send), recv) in self.ranks.iter().zip(send.iter()).zip(recv.iter_mut())
                {
                    rank.communicator
                        .all_to_all(*send, recv.reborrow(), &rank.stream)?;
                }
                Ok(())
            }
        })
    }

    /// Reduces rank inputs and distributes equal contiguous portions of the result to rank outputs.
    ///
    /// Returns after all local rank streams complete successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if per-rank counts or buffer lengths are incompatible with the collective, a buffer or stream belongs to another CUDA context.
    /// An already open NCCL group, communicator-state checks, CUDA context binding, NCCL grouping or execution, and stream synchronization can also produce errors.
    pub fn reduce_scatter<T: DataTypeLike>(
        &self,
        send: &RankVec<DeviceView<'_, T>>,
        recv: &mut RankVec<DeviceOutputView<'_, T>>,
        elements_per_rank: usize,
        op: ReductionOperator,
    ) -> Result<()> {
        self.ensure_partitioned_rank_vec(send, elements_per_rank, "reduce_scatter send")?;
        self.ensure_rank_vec(recv, "reduce_scatter recv")?;
        for buffer in recv.iter() {
            if buffer.len() != elements_per_rank {
                return Err(Error::LengthMismatch {
                    name: "reduce_scatter recv".into(),
                });
            }
        }
        self.with_completed_group(|| {
            // SAFETY: all views remain borrowed until the completed group drains.
            unsafe {
                for ((rank, send), recv) in self.ranks.iter().zip(send.iter()).zip(recv.iter_mut())
                {
                    rank.communicator
                        .reduce_scatter(*send, recv.reborrow(), op, &rank.stream)?;
                }
                Ok(())
            }
        })
    }

    fn ensure_root(&self, root: RootRank) -> Result<()> {
        let index = root.id().to_index()?;
        if index >= self.rank_count() {
            return Err(Error::out_of_range("root rank"));
        }
        Ok(())
    }

    fn ensure_rank_vec<T>(&self, values: &RankVec<T>, name: &str) -> Result<()> {
        if values.len() != self.rank_count() {
            return Err(Error::LengthMismatch {
                name: name.to_owned().into(),
            });
        }
        Ok(())
    }

    fn ensure_partitioned_rank_vec<T: DataTypeLike>(
        &self,
        values: &RankVec<DeviceView<'_, T>>,
        elements_per_rank: usize,
        name: &str,
    ) -> Result<()> {
        self.ensure_rank_vec(values, name)?;
        let expected = partitioned_len(elements_per_rank, self.rank_count(), name)?;
        ensure_lengths(values.iter().map(DeviceView::len), expected, name)
    }

    fn first_len<T: DataTypeLike>(
        values: &RankVec<DeviceView<'_, T>>,
        name: &str,
    ) -> Result<usize> {
        values
            .iter()
            .next()
            .map(DeviceView::len)
            .ok_or(Error::LengthMismatch {
                name: name.to_owned().into(),
            })
    }

    fn ensure_same_len<T: DataTypeLike>(
        values: &RankVec<DeviceView<'_, T>>,
        name: &str,
    ) -> Result<()> {
        let expected = Self::first_len(values, name)?;
        ensure_lengths(values.iter().map(DeviceView::len), expected, name)
    }

    fn ensure_all_len<T: DataTypeLike>(
        values: &RankVec<DeviceOutputView<'_, T>>,
        expected: usize,
        name: &str,
    ) -> Result<()> {
        ensure_lengths(values.iter().map(DeviceOutputView::len), expected, name)
    }

    fn ensure_pair_lengths<T: DataTypeLike>(
        left: &RankVec<DeviceView<'_, T>>,
        right: &RankVec<DeviceOutputView<'_, T>>,
        name: &str,
    ) -> Result<()> {
        if left
            .iter()
            .zip(right)
            .any(|(left, right)| left.len() != right.len())
        {
            return Err(Error::LengthMismatch {
                name: name.to_owned().into(),
            });
        }
        Ok(())
    }

    /// Runs raw operations for these ranks as one completed NCCL group.
    ///
    /// Polls native submission and device execution, checks asynchronous errors,
    /// and drains every stream before returning. An error or unwind cancels the
    /// group before releasing borrowed resources. Open groups and capture are rejected.
    ///
    /// # Safety
    /// The closure must issue matching collective or point-to-point calls using
    /// these communicators and streams, including on error and unwind paths.
    /// Retain backing buffers and referenced device resources outside the closure
    /// through this call, and uphold each raw operation's access rules.
    ///
    /// # Errors
    ///
    /// Returns an error if an NCCL group is already open, starting or ending the group fails, `f` fails, or asynchronous communicator status or stream synchronization reports a failure.
    pub unsafe fn sync_scope<R>(&self, f: impl FnOnce() -> Result<R>) -> Result<R> {
        self.with_completed_group(f)
    }

    fn with_completed_group<R>(&self, f: impl FnOnce() -> Result<R>) -> Result<R> {
        if group::is_active() {
            return Err(Error::GroupActive);
        }
        self.drain_streams(0, || {
            let mut drain = CollectiveDrain {
                ranks: &self.ranks,
                completed: false,
            };
            // End native grouping before aborting its communicators on every path.
            let group = Group::start()?;
            let result = f();
            let end = group.end();
            let value = result?;
            match end {
                Ok(())
                | Err(Error::Nccl {
                    code: Status::InProgress,
                    ..
                }) => {}
                Err(error) => return Err(error),
            }
            // Nonblocking native submission must finish before CUDA stream waits.
            drain.wait()?;
            drain.completed = true;
            Ok(value)
        })
    }

    fn drain_streams<R>(&self, index: usize, f: impl FnOnce() -> Result<R>) -> Result<R> {
        let Some(rank) = self.ranks.get(index) else {
            return f();
        };
        let mut result = None;
        rank.stream.sync_scope(|_| {
            result = Some(self.drain_streams(index + 1, f));
            Ok(())
        })?;
        result.expect("stream scope ran")
    }
}

fn partitioned_len(elements_per_rank: usize, rank_count: usize, name: &str) -> Result<usize> {
    elements_per_rank
        .checked_mul(rank_count)
        .ok_or_else(|| Error::out_of_range(name.to_owned()))
}

impl LocalRank {
    /// Returns this rank's typed identifier.
    pub const fn id(&self) -> RankId {
        self.id
    }

    /// Returns this rank's zero-based index.
    pub const fn index(&self) -> i32 {
        self.id.get()
    }

    /// Returns this rank's communicator.
    pub fn communicator(&self) -> &Communicator {
        &self.communicator
    }

    /// Returns this rank's CUDA stream.
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Binds the CUDA context associated with this rank.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator state disallows control operations or binding its CUDA context fails.
    pub fn bind(&self) -> Result<()> {
        self.communicator.bind()
    }
}

impl<T> RankVec<T> {
    /// Stores values in their existing rank order.
    #[must_use]
    pub fn from_vec(values: Vec<T>) -> Self {
        Self { values }
    }

    /// Maps values while preserving rank order.
    pub fn map<'a, U>(&'a self, f: impl FnMut(&'a T) -> U) -> RankVec<U> {
        RankVec::from_vec(self.values.iter().map(f).collect())
    }

    /// Maps mutable value borrows while preserving rank order.
    pub fn map_mut<'a, U>(&'a mut self, f: impl FnMut(&'a mut T) -> U) -> RankVec<U> {
        RankVec::from_vec(self.values.iter_mut().map(f).collect())
    }

    /// Returns the owned values in rank order.
    #[must_use]
    pub fn into_vec(self) -> Vec<T> {
        self.values
    }

    /// Returns the number of rank values.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns whether there are no rank values.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Iterates over values in rank order.
    pub fn iter(&self) -> slice::Iter<'_, T> {
        self.values.iter()
    }

    /// Iterates over mutable values in rank order.
    pub fn iter_mut(&mut self) -> slice::IterMut<'_, T> {
        self.values.iter_mut()
    }

    /// Returns the value for `rank`.
    ///
    /// # Errors
    ///
    /// Returns an error if the rank is outside the stored range.
    pub fn get(&self, rank: RankId) -> Result<&T> {
        self.values
            .get(rank.to_index()?)
            .ok_or_else(|| Error::out_of_range("rank"))
    }

    /// Returns a mutable borrow of the value for `rank`.
    ///
    /// # Errors
    ///
    /// Returns an error if the rank is outside the stored range.
    pub fn get_mut(&mut self, rank: RankId) -> Result<&mut T> {
        self.values
            .get_mut(rank.to_index()?)
            .ok_or_else(|| Error::out_of_range("rank"))
    }
}

impl<T> RootOutput<T> {
    /// Returns the root rank associated with the output.
    pub const fn root_rank(&self) -> RootRank {
        self.root
    }

    /// Borrows the root output value.
    pub fn root(&self) -> &T {
        &self.value
    }

    /// Mutably borrows the root output value.
    pub fn root_mut(&mut self) -> &mut T {
        &mut self.value
    }

    /// Returns the owned root output value.
    pub fn into_root(self) -> T {
        self.value
    }
}

impl<T> Index<usize> for RankVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

impl<T> IndexMut<usize> for RankVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.values[index]
    }
}

impl<T> IntoIterator for RankVec<T> {
    type IntoIter = vec::IntoIter<T>;

    type Item = T;

    fn into_iter(self) -> Self::IntoIter {
        self.values.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a RankVec<T> {
    type IntoIter = slice::Iter<'a, T>;

    type Item = &'a T;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut RankVec<T> {
    type IntoIter = slice::IterMut<'a, T>;

    type Item = &'a mut T;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

fn ensure_lengths(lengths: impl Iterator<Item = usize>, expected: usize, name: &str) -> Result<()> {
    if lengths.into_iter().any(|len| len != expected) {
        return Err(Error::LengthMismatch {
            name: name.to_owned().into(),
        });
    }
    Ok(())
}

/// Cancels incomplete native submission before scoped CUDA drains release borrows.
struct CollectiveDrain<'a> {
    ranks: &'a [LocalRank],
    completed: bool,
}

impl CollectiveDrain<'_> {
    fn wait(&self) -> Result<()> {
        loop {
            let mut pending = false;
            for rank in self.ranks {
                match rank.communicator.async_error()? {
                    None => {}
                    Some(Status::InProgress) => pending = true,
                    Some(status) => return Err(status.into()),
                }
            }
            if !pending {
                let mut ready = true;
                for rank in self.ranks {
                    ready &= rank.stream.query()?;
                }
                if ready {
                    // A native execution failure may be visible only after submission.
                    for rank in self.ranks {
                        if let Some(status) = rank.communicator.async_error()? {
                            return Err(status.into());
                        }
                    }
                    return Ok(());
                }
            }
            thread::yield_now();
        }
    }
}

impl Drop for CollectiveDrain<'_> {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        for rank in self.ranks {
            let communicator = &rank.communicator;
            if communicator.bind().is_err() {
                process::abort();
            }
            // SAFETY: this guard owns cancellation of the incomplete local group.
            let status = unsafe { sys::ncclCommAbort(communicator.as_raw()) };
            if status != sys::ncclResult_t::ncclSuccess {
                process::abort();
            }
            communicator.set_state(CommunicatorState::Aborted);
        }
    }
}
