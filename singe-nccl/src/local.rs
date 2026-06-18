use std::{
    ops::{Index, IndexMut},
    slice,
};

use singe_cuda::{device::Device, memory::DeviceMemory, stream::Stream};

use crate::{
    communicator::Communicator,
    error::{Error, Result},
    types::{DataTypeLike, ReductionOperator},
    utility::{to_i32, to_usize},
    with_group,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RankId(i32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RootRank(RankId);

#[derive(Debug)]
pub struct LocalCommunicatorGroup {
    ranks: Vec<LocalRank>,
}

#[derive(Debug)]
pub struct LocalRank {
    id: RankId,
    communicator: Communicator,
    stream: Stream,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankVec<T> {
    values: Vec<T>,
}

#[derive(Debug)]
pub struct RootOutput<T> {
    root: RootRank,
    values: RankVec<Option<T>>,
}

impl RankId {
    pub fn create(value: i32) -> Result<Self> {
        if value < 0 {
            return Err(Error::OutOfRange {
                name: "rank".into(),
            });
        }
        Ok(Self(value))
    }

    pub const fn get(self) -> i32 {
        self.0
    }

    pub(crate) fn to_index(self) -> Result<usize> {
        to_usize(self.0, "rank")
    }
}

impl RootRank {
    pub fn create(value: i32) -> Result<Self> {
        Ok(Self(RankId::create(value)?))
    }

    pub const fn id(self) -> RankId {
        self.0
    }

    pub const fn get(self) -> i32 {
        self.0.get()
    }
}

impl LocalCommunicatorGroup {
    pub fn create_all(device_ids: &[i32]) -> Result<Self> {
        let communicators = Communicator::create_all(device_ids)?;
        Self::from_communicators(communicators)
    }

    pub fn create_all_first_devices(device_count: i32) -> Result<Self> {
        let device_count = to_usize(device_count, "device_count")?;
        let device_ids = (0..device_count)
            .map(|device_id| to_i32(device_id, "device_id"))
            .collect::<Result<Vec<_>>>()?;
        Self::create_all(&device_ids)
    }

    pub fn create_all_visible(rank_count: usize) -> Result<Self> {
        let device_count = to_usize(Device::count()?, "device_count")?;
        if device_count < rank_count {
            return Err(Error::OutOfRange {
                name: "visible device count".into(),
            });
        }
        let device_ids = (0..rank_count)
            .map(|device_id| to_i32(device_id, "device_id"))
            .collect::<Result<Vec<_>>>()?;
        Self::create_all(&device_ids)
    }

    pub fn from_communicators(communicators: Vec<Communicator>) -> Result<Self> {
        let mut ranks = Vec::with_capacity(communicators.len());
        for communicator in communicators {
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
                return Err(Error::OutOfRange {
                    name: "communicator ranks".into(),
                });
            }
        }
        Ok(Self { ranks })
    }

    pub fn rank_count(&self) -> usize {
        self.ranks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ranks.is_empty()
    }

    pub fn ranks(&self) -> slice::Iter<'_, LocalRank> {
        self.ranks.iter()
    }

    pub fn rank(&self, id: RankId) -> Result<&LocalRank> {
        let index = id.to_index()?;
        self.ranks.get(index).ok_or(Error::OutOfRange {
            name: "rank".into(),
        })
    }

    pub fn map_ranks<T>(&self, mut f: impl FnMut(&LocalRank) -> Result<T>) -> Result<RankVec<T>> {
        self.ranks
            .iter()
            .map(&mut f)
            .collect::<Result<Vec<_>>>()
            .map(RankVec::from_vec)
    }

    pub fn synchronize(&self) -> Result<()> {
        for rank in &self.ranks {
            rank.stream.synchronize()?;
        }
        Ok(())
    }

    pub fn destroy(self) -> Result<()> {
        for rank in self.ranks {
            rank.communicator.destroy()?;
        }
        Ok(())
    }

    pub fn all_reduce<T: DataTypeLike>(
        &self,
        send: &RankVec<DeviceMemory<T>>,
        recv: &mut RankVec<DeviceMemory<T>>,
        op: ReductionOperator,
    ) -> Result<()> {
        self.ensure_rank_vec(send, "all_reduce send")?;
        self.ensure_rank_vec(recv, "all_reduce recv")?;
        self.ensure_same_len(send, "all_reduce send")?;
        self.ensure_same_len(recv, "all_reduce recv")?;
        self.ensure_pair_lengths(send, recv, "all_reduce buffers")?;
        with_group(|| {
            for ((rank, send), recv) in self.ranks.iter().zip(send.iter()).zip(recv.iter_mut()) {
                rank.communicator
                    .all_reduce_memory(send, recv, op, &rank.stream)?;
            }
            Ok(())
        })
    }

    pub fn broadcast<T: DataTypeLike>(
        &self,
        send: &RankVec<DeviceMemory<T>>,
        recv: &mut RankVec<DeviceMemory<T>>,
        root: RootRank,
    ) -> Result<()> {
        self.ensure_root(root)?;
        self.ensure_rank_vec(send, "broadcast send")?;
        self.ensure_rank_vec(recv, "broadcast recv")?;
        self.ensure_same_len(send, "broadcast send")?;
        self.ensure_same_len(recv, "broadcast recv")?;
        self.ensure_pair_lengths(send, recv, "broadcast buffers")?;
        with_group(|| {
            for ((rank, send), recv) in self.ranks.iter().zip(send.iter()).zip(recv.iter_mut()) {
                rank.communicator
                    .broadcast_memory(send, recv, root.get(), &rank.stream)?;
            }
            Ok(())
        })
    }

    pub fn broadcast_in_place<T: DataTypeLike>(
        &self,
        buffers: &mut RankVec<DeviceMemory<T>>,
        root: RootRank,
    ) -> Result<()> {
        self.ensure_root(root)?;
        self.ensure_rank_vec(buffers, "broadcast buffers")?;
        self.ensure_same_len(buffers, "broadcast buffers")?;
        with_group(|| {
            for (rank, buffer) in self.ranks.iter().zip(buffers.iter_mut()) {
                rank.communicator
                    .broadcast_in_place_memory(buffer, root.get(), &rank.stream)?;
            }
            Ok(())
        })
    }

    pub fn reduce<T: DataTypeLike>(
        &self,
        send: &RankVec<DeviceMemory<T>>,
        root: RootRank,
        op: ReductionOperator,
    ) -> Result<RootOutput<DeviceMemory<T>>> {
        self.ensure_root(root)?;
        self.ensure_rank_vec(send, "reduce send")?;

        let root_index = root.id().to_index()?;
        let mut values = (0..self.rank_count()).map(|_| None).collect::<Vec<_>>();
        self.ranks[root_index].bind()?;
        values[root_index] = Some(DeviceMemory::<T>::zeroes(send[root_index].len())?);
        let mut output = RootOutput {
            root,
            values: RankVec::from_vec(values),
        };
        self.reduce_into(
            send,
            root,
            output.root_mut().expect("root output exists"),
            op,
        )?;
        Ok(output)
    }

    pub fn reduce_into<T: DataTypeLike>(
        &self,
        send: &RankVec<DeviceMemory<T>>,
        root: RootRank,
        root_recv: &mut DeviceMemory<T>,
        op: ReductionOperator,
    ) -> Result<()> {
        self.ensure_root(root)?;
        self.ensure_rank_vec(send, "reduce send")?;
        self.ensure_same_len(send, "reduce send")?;
        let send_len = self.first_len(send, "reduce send")?;
        if root_recv.len() != send_len {
            return Err(Error::LengthMismatch {
                name: "reduce recv".into(),
            });
        }
        let root_index = root.id().to_index()?;
        with_group(|| {
            let mut root_recv = Some(root_recv);
            for (index, (rank, send)) in self.ranks.iter().zip(send.iter()).enumerate() {
                let recv = if index == root_index {
                    root_recv.as_deref_mut()
                } else {
                    None
                };
                rank.communicator
                    .reduce_memory(send, recv, op, root.get(), &rank.stream)?;
            }
            Ok(())
        })
    }

    pub fn all_gather<T: DataTypeLike>(
        &self,
        send: &RankVec<DeviceMemory<T>>,
        recv: &mut RankVec<DeviceMemory<T>>,
    ) -> Result<()> {
        self.ensure_rank_vec(send, "all_gather send")?;
        self.ensure_rank_vec(recv, "all_gather recv")?;
        self.ensure_same_len(send, "all_gather send")?;
        let expected_recv_len = self
            .first_len(send, "all_gather send")?
            .checked_mul(self.rank_count())
            .ok_or(Error::OutOfRange {
                name: "all_gather recv".into(),
            })?;
        self.ensure_all_len(recv, expected_recv_len, "all_gather recv")?;
        with_group(|| {
            for ((rank, send), recv) in self.ranks.iter().zip(send.iter()).zip(recv.iter_mut()) {
                rank.communicator
                    .all_gather_memory(send, recv, &rank.stream)?;
            }
            Ok(())
        })
    }

    pub fn gather_into<T: DataTypeLike>(
        &self,
        send: &RankVec<DeviceMemory<T>>,
        root: RootRank,
        root_recv: &mut DeviceMemory<T>,
    ) -> Result<()> {
        self.ensure_root(root)?;
        self.ensure_rank_vec(send, "gather send")?;
        self.ensure_same_len(send, "gather send")?;
        let expected_recv_len = self
            .first_len(send, "gather send")?
            .checked_mul(self.rank_count())
            .ok_or(Error::OutOfRange {
                name: "gather recv".into(),
            })?;
        if root_recv.len() != expected_recv_len {
            return Err(Error::LengthMismatch {
                name: "gather recv".into(),
            });
        }
        let root_index = root.id().to_index()?;
        with_group(|| {
            let mut root_recv = Some(root_recv);
            for (index, (rank, send)) in self.ranks.iter().zip(send.iter()).enumerate() {
                let recv = if index == root_index {
                    root_recv.as_deref_mut()
                } else {
                    None
                };
                rank.communicator
                    .gather_memory(send, recv, root.get(), &rank.stream)?;
            }
            Ok(())
        })
    }

    pub fn scatter_from<T: DataTypeLike>(
        &self,
        root_send: &DeviceMemory<T>,
        recv: &mut RankVec<DeviceMemory<T>>,
        root: RootRank,
    ) -> Result<()> {
        self.ensure_root(root)?;
        self.ensure_rank_vec(recv, "scatter recv")?;
        self.ensure_same_len(recv, "scatter recv")?;
        let expected_send_len = self
            .first_len(recv, "scatter recv")?
            .checked_mul(self.rank_count())
            .ok_or(Error::OutOfRange {
                name: "scatter send".into(),
            })?;
        if root_send.len() != expected_send_len {
            return Err(Error::LengthMismatch {
                name: "scatter send".into(),
            });
        }
        let root_index = root.id().to_index()?;
        with_group(|| {
            for (index, (rank, recv)) in self.ranks.iter().zip(recv.iter_mut()).enumerate() {
                let send = (index == root_index).then_some(root_send);
                rank.communicator
                    .scatter_memory(send, recv, root.get(), &rank.stream)?;
            }
            Ok(())
        })
    }

    pub fn all_to_all<T: DataTypeLike>(
        &self,
        send: &RankVec<DeviceMemory<T>>,
        recv: &mut RankVec<DeviceMemory<T>>,
        elements_per_rank: usize,
    ) -> Result<()> {
        self.ensure_partitioned_rank_vec(send, elements_per_rank, "all_to_all send")?;
        self.ensure_partitioned_rank_vec(recv, elements_per_rank, "all_to_all recv")?;
        with_group(|| {
            for ((rank, send), recv) in self.ranks.iter().zip(send.iter()).zip(recv.iter_mut()) {
                rank.communicator
                    .all_to_all_memory(send, recv, &rank.stream)?;
            }
            Ok(())
        })
    }

    pub fn reduce_scatter<T: DataTypeLike>(
        &self,
        send: &RankVec<DeviceMemory<T>>,
        recv: &mut RankVec<DeviceMemory<T>>,
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
        with_group(|| {
            for ((rank, send), recv) in self.ranks.iter().zip(send.iter()).zip(recv.iter_mut()) {
                rank.communicator
                    .reduce_scatter_memory(send, recv, op, &rank.stream)?;
            }
            Ok(())
        })
    }

    fn ensure_root(&self, root: RootRank) -> Result<()> {
        let index = root.id().to_index()?;
        if index >= self.rank_count() {
            return Err(Error::OutOfRange {
                name: "root rank".into(),
            });
        }
        Ok(())
    }

    fn ensure_rank_vec<T>(&self, values: &RankVec<T>, name: &str) -> Result<()> {
        if values.len() != self.rank_count() {
            return Err(Error::LengthMismatch { name: name.into() });
        }
        Ok(())
    }

    fn ensure_partitioned_rank_vec<T>(
        &self,
        values: &RankVec<DeviceMemory<T>>,
        elements_per_rank: usize,
        name: &str,
    ) -> Result<()> {
        self.ensure_rank_vec(values, name)?;
        let expected_len = partitioned_len(elements_per_rank, self.rank_count(), name)?;
        for buffer in values.iter() {
            if buffer.len() != expected_len {
                return Err(Error::LengthMismatch { name: name.into() });
            }
        }
        Ok(())
    }

    fn first_len<T>(&self, values: &RankVec<DeviceMemory<T>>, name: &str) -> Result<usize> {
        values
            .iter()
            .next()
            .map(DeviceMemory::len)
            .ok_or(Error::LengthMismatch { name: name.into() })
    }

    fn ensure_same_len<T>(&self, values: &RankVec<DeviceMemory<T>>, name: &str) -> Result<()> {
        let expected_len = self.first_len(values, name)?;
        self.ensure_all_len(values, expected_len, name)
    }

    fn ensure_all_len<T>(
        &self,
        values: &RankVec<DeviceMemory<T>>,
        expected_len: usize,
        name: &str,
    ) -> Result<()> {
        for buffer in values.iter() {
            if buffer.len() != expected_len {
                return Err(Error::LengthMismatch { name: name.into() });
            }
        }
        Ok(())
    }

    fn ensure_pair_lengths<T>(
        &self,
        left: &RankVec<DeviceMemory<T>>,
        right: &RankVec<DeviceMemory<T>>,
        name: &str,
    ) -> Result<()> {
        for (left, right) in left.iter().zip(right.iter()) {
            if left.len() != right.len() {
                return Err(Error::LengthMismatch { name: name.into() });
            }
        }
        Ok(())
    }
}

fn partitioned_len(elements_per_rank: usize, rank_count: usize, name: &str) -> Result<usize> {
    elements_per_rank
        .checked_mul(rank_count)
        .ok_or(Error::OutOfRange { name: name.into() })
}

impl LocalRank {
    pub const fn id(&self) -> RankId {
        self.id
    }

    pub const fn index(&self) -> i32 {
        self.id.get()
    }

    pub fn communicator(&self) -> &Communicator {
        &self.communicator
    }

    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    pub fn bind(&self) -> Result<()> {
        self.communicator.bind()
    }
}

impl<T> RankVec<T> {
    pub fn from_vec(values: Vec<T>) -> Self {
        Self { values }
    }

    pub fn into_vec(self) -> Vec<T> {
        self.values
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn iter(&self) -> slice::Iter<'_, T> {
        self.values.iter()
    }

    pub fn iter_mut(&mut self) -> slice::IterMut<'_, T> {
        self.values.iter_mut()
    }

    pub fn get(&self, rank: RankId) -> Result<&T> {
        self.values.get(rank.to_index()?).ok_or(Error::OutOfRange {
            name: "rank".into(),
        })
    }

    pub fn get_mut(&mut self, rank: RankId) -> Result<&mut T> {
        self.values
            .get_mut(rank.to_index()?)
            .ok_or(Error::OutOfRange {
                name: "rank".into(),
            })
    }
}

impl<T> RootOutput<T> {
    pub const fn root_rank(&self) -> RootRank {
        self.root
    }

    pub fn values(&self) -> &RankVec<Option<T>> {
        &self.values
    }

    pub fn root(&self) -> Option<&T> {
        self.values
            .values
            .get(self.root.0.to_index().ok()?)?
            .as_ref()
    }

    pub fn root_mut(&mut self) -> Option<&mut T> {
        self.values
            .values
            .get_mut(self.root.0.to_index().ok()?)?
            .as_mut()
    }

    pub fn into_root(self) -> Option<T> {
        let index = self.root.0.to_index().ok()?;
        self.values.into_vec().into_iter().nth(index).flatten()
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
    type IntoIter = std::vec::IntoIter<T>;
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
