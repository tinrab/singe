use std::{
    fmt::{self, Debug, Formatter},
    marker::PhantomData,
    ops::{Deref, Range},
    sync::Arc,
};

use crate::{
    ast::{ScalarType, StateSpace},
    emulator::{
        execution_error::SubmitError,
        runtime::{Client, Lease, scalar::PtxScalar},
    },
};

pub fn encode<T: PtxScalar>(values: &[T]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(size_of_val(values));
    for value in values {
        value.encode(&mut bytes);
    }
    bytes
}

pub fn byte_size<T: PtxScalar>(len: usize) -> Result<usize, SubmitError> {
    len.checked_mul(T::SIZE)
        .ok_or_else(|| SubmitError::InvalidArgument("byte size overflow".into()))
}

pub fn foreign(a: &Arc<Client>, b: &Arc<Client>) -> Result<(), SubmitError> {
    if Arc::ptr_eq(a, b) {
        Ok(())
    } else {
        Err(SubmitError::ForeignResource)
    }
}

#[derive(Debug, Clone)]
pub struct AllocationRef {
    pub _lease: Arc<Lease>,
    pub address: u64,
    pub writable: bool,
    pub space: StateSpace,
}

/// An emulator allocation of `T` elements.
///
/// Clones refer to the same allocation. A buffer dereferences to a [`BufferView`] of
/// all its elements.
#[derive(Debug, Clone)]
pub struct Buffer<T: PtxScalar> {
    pub(crate) view: BufferView<T>,
}

impl<T: PtxScalar> Deref for Buffer<T> {
    type Target = BufferView<T>;

    fn deref(&self) -> &Self::Target {
        &self.view
    }
}

/// A range of elements in an allocation. It keeps the allocation alive.
#[derive(Debug, Clone)]
pub struct BufferView<T: PtxScalar> {
    pub(crate) client: Arc<Client>,
    pub(crate) allocation: AllocationRef,
    pub(crate) offset: usize,
    pub(crate) len: usize,
    pub(crate) marker: PhantomData<T>,
}

impl<T: PtxScalar> BufferView<T> {
    /// Returns the number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the view has no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the elements in `range`, indexed relative to this view.
    ///
    /// # Errors
    ///
    /// Returns an error if `range` is reversed or extends past the end of this view, or if
    /// its byte offset overflows.
    pub fn view(&self, range: Range<usize>) -> Result<Self, SubmitError> {
        let len = range
            .end
            .checked_sub(range.start)
            .filter(|_| range.end <= self.len)
            .ok_or_else(|| SubmitError::InvalidArgument("view bounds".into()))?;
        Ok(Self {
            client: Arc::clone(&self.client),
            allocation: self.allocation.clone(),
            offset: self.offset + byte_size::<T>(range.start)?,
            len,
            marker: PhantomData,
        })
    }

    pub(crate) fn address(&self) -> u64 {
        self.allocation.address + self.offset as u64
    }
}

/// A kernel argument: a scalar, a pointer to a buffer, or aggregate bytes.
///
/// Scalars convert with `From`, as in `5u32.into()`, and pointers from `&Buffer` or
/// `&BufferView`.
#[derive(Debug, Clone)]
pub struct KernelArg {
    pub(crate) value: Argument,
}

#[derive(Clone)]
pub enum Argument {
    Scalar(ScalarType, Vec<u8>),
    Pointer {
        client: Arc<Client>,
        allocation: AllocationRef,
        address: u64,
    },
    Aggregate {
        bytes: Vec<u8>,
        alignment: u32,
    },
}

impl Debug for Argument {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Argument").finish_non_exhaustive()
    }
}

impl KernelArg {
    /// Creates an argument for an aggregate (array) parameter from its raw bytes.
    ///
    /// The launch checks that `bytes` has the parameter's size and that `alignment` is at
    /// least the parameter's alignment.
    ///
    /// # Errors
    ///
    /// Returns [`SubmitError::InvalidArgument`] if `alignment` is not a nonzero power of
    /// two.
    pub fn aggregate(bytes: Vec<u8>, alignment: u32) -> Result<Self, SubmitError> {
        if !alignment.is_power_of_two() {
            return Err(SubmitError::InvalidArgument("aggregate alignment".into()));
        }
        Ok(Self {
            value: Argument::Aggregate { bytes, alignment },
        })
    }
}

impl<T: PtxScalar> From<&BufferView<T>> for KernelArg {
    fn from(view: &BufferView<T>) -> Self {
        Self {
            value: Argument::Pointer {
                client: Arc::clone(&view.client),
                allocation: view.allocation.clone(),
                address: view.address(),
            },
        }
    }
}

impl<T: PtxScalar> From<&Buffer<T>> for KernelArg {
    fn from(buffer: &Buffer<T>) -> Self {
        Self::from(&buffer.view)
    }
}
