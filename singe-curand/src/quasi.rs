//! Sobol direction-vector and scramble-constant tables.
//!
//! These wrappers expose library-owned host-memory tables returned by cuRAND for
//! quasirandom number generation.

use std::{
    ops::{Index, RangeFull},
    ptr::{self, NonNull},
    slice,
};

use crate::{
    error::{Error, Result},
    try_ffi,
    types::DirectionVectorSet,
};

use singe_curand_sys as sys;

/// Number of Sobol dimensions in the direction-vector and scramble-constant tables.
///
/// cuRAND documents the Joe-Kuo direction-vector sets exposed by
/// `curandGetDirectionVectors32` and `curandGetDirectionVectors64` as containing 20,000 dimensions.
/// The same dimension limit is used by `curandSetQuasiRandomGeneratorDimensions`.
pub const MAX_QUASI_RANDOM_DIMENSIONS: usize = 20_000;

#[repr(transparent)]
/// Direction vector for one 32-bit Sobol dimension.
///
/// Each dimension has 32 `u32` words.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DirectionVector32 {
    words: [u32; Self::WORDS],
}

#[repr(transparent)]
/// Direction vector for one 64-bit Sobol dimension.
///
/// Each dimension has 64 `u64` words.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DirectionVector64 {
    words: [u64; Self::WORDS],
}

/// Library-owned table of 32-bit Sobol direction vectors.
///
/// The table contains one [`DirectionVector32`] per supported dimension.
#[derive(Debug, Clone, Copy)]
pub struct DirectionVectorTable32 {
    ptr: NonNull<DirectionVector32>,
    len: usize,
}

/// Library-owned table of 64-bit Sobol direction vectors.
///
/// The table contains one [`DirectionVector64`] per supported dimension.
#[derive(Debug, Clone, Copy)]
pub struct DirectionVectorTable64 {
    ptr: NonNull<DirectionVector64>,
    len: usize,
}

/// Library-owned 32-bit scramble constants for scrambled Sobol generation.
#[derive(Debug, Clone, Copy)]
pub struct ScrambleConstants32 {
    ptr: NonNull<u32>,
    len: usize,
}

/// Library-owned 64-bit scramble constants for scrambled Sobol generation.
#[derive(Debug, Clone, Copy)]
pub struct ScrambleConstants64 {
    ptr: NonNull<u64>,
    len: usize,
}

/// Returns a table of 32-bit Sobol direction vectors for `set`.
///
/// cuRAND returns a pointer to host memory owned by the library. The returned
/// wrapper borrows that stable table and exposes it as a Rust slice.
///
/// # Errors
///
/// Returns an error if `set` is not valid for 32-bit direction vectors or if
/// cuRAND does not return a table pointer.
pub fn direction_vectors_32(set: DirectionVectorSet) -> Result<DirectionVectorTable32> {
    let mut vectors = ptr::null_mut();
    unsafe {
        try_ffi!(sys::curandGetDirectionVectors32(
            &raw mut vectors,
            set.into(),
        ))?;
    }

    DirectionVectorTable32::from_raw(vectors.cast(), MAX_QUASI_RANDOM_DIMENSIONS)
}

/// Returns a table of 64-bit Sobol direction vectors for `set`.
///
/// cuRAND returns a pointer to host memory owned by the library. The returned
/// wrapper borrows that stable table and exposes it as a Rust slice.
///
/// # Errors
///
/// Returns an error if `set` is not valid for 64-bit direction vectors or if
/// cuRAND does not return a table pointer.
pub fn direction_vectors_64(set: DirectionVectorSet) -> Result<DirectionVectorTable64> {
    let mut vectors = ptr::null_mut();
    unsafe {
        try_ffi!(sys::curandGetDirectionVectors64(
            &raw mut vectors,
            set.into(),
        ))?;
    }

    DirectionVectorTable64::from_raw(vectors.cast(), MAX_QUASI_RANDOM_DIMENSIONS)
}

/// Returns 32-bit scramble constants for scrambled Sobol generation.
///
/// cuRAND returns a pointer to host memory owned by the library.
///
/// # Errors
///
/// Returns an error if cuRAND does not return a constants pointer.
pub fn scramble_constants_32() -> Result<ScrambleConstants32> {
    let mut constants = ptr::null_mut();
    unsafe {
        try_ffi!(sys::curandGetScrambleConstants32(&raw mut constants))?;
    }

    ScrambleConstants32::from_raw(constants, MAX_QUASI_RANDOM_DIMENSIONS)
}

/// Returns 64-bit scramble constants for scrambled Sobol generation.
///
/// cuRAND returns a pointer to host memory owned by the library.
///
/// # Errors
///
/// Returns an error if cuRAND does not return a constants pointer.
pub fn scramble_constants_64() -> Result<ScrambleConstants64> {
    let mut constants = ptr::null_mut();
    unsafe {
        try_ffi!(sys::curandGetScrambleConstants64(&raw mut constants))?;
    }

    ScrambleConstants64::from_raw(constants, MAX_QUASI_RANDOM_DIMENSIONS)
}

impl DirectionVectorTable32 {
    fn from_raw(ptr: *mut DirectionVector32, len: usize) -> Result<Self> {
        Ok(Self {
            ptr: NonNull::new(ptr).ok_or(Error::NullHandle)?,
            len,
        })
    }

    /// Returns the number of dimensions in the table.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the table contains no direction vectors.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the table as a slice indexed by dimension.
    pub fn as_slice(&self) -> &[DirectionVector32] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Returns the direction vector for `dimension`, if present.
    pub fn get(&self, dimension: usize) -> Option<&DirectionVector32> {
        self.as_slice().get(dimension)
    }
}

impl Index<usize> for DirectionVectorTable32 {
    type Output = DirectionVector32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl DirectionVectorTable64 {
    fn from_raw(ptr: *mut DirectionVector64, len: usize) -> Result<Self> {
        Ok(Self {
            ptr: NonNull::new(ptr).ok_or(Error::NullHandle)?,
            len,
        })
    }

    /// Returns the number of dimensions in the table.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the table contains no direction vectors.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the table as a slice indexed by dimension.
    pub fn as_slice(&self) -> &[DirectionVector64] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Returns the direction vector for `dimension`, if present.
    pub fn get(&self, dimension: usize) -> Option<&DirectionVector64> {
        self.as_slice().get(dimension)
    }
}

impl Index<usize> for DirectionVectorTable64 {
    type Output = DirectionVector64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl ScrambleConstants32 {
    fn from_raw(ptr: *mut u32, len: usize) -> Result<Self> {
        Ok(Self {
            ptr: NonNull::new(ptr).ok_or(Error::NullHandle)?,
            len,
        })
    }

    /// Returns the number of scramble constants in the table.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the table contains no constants.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the constants as a slice indexed by dimension.
    pub fn as_slice(&self) -> &[u32] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl ScrambleConstants64 {
    fn from_raw(ptr: *mut u64, len: usize) -> Result<Self> {
        Ok(Self {
            ptr: NonNull::new(ptr).ok_or(Error::NullHandle)?,
            len,
        })
    }

    /// Returns the number of scramble constants in the table.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the table contains no constants.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the constants as a slice indexed by dimension.
    pub fn as_slice(&self) -> &[u64] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl DirectionVector32 {
    /// Number of words in a 32-bit Sobol direction vector.
    pub const WORDS: usize = 32;

    /// Returns the direction vector as an array of words.
    pub const fn as_words(&self) -> &[u32; Self::WORDS] {
        &self.words
    }

    /// Returns the direction vector as a slice of words.
    pub const fn as_slice(&self) -> &[u32] {
        self.words.as_slice()
    }
}

impl AsRef<[u32]> for DirectionVector32 {
    fn as_ref(&self) -> &[u32] {
        self.as_slice()
    }
}

impl Index<usize> for DirectionVector32 {
    type Output = u32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.words[index]
    }
}

impl Index<RangeFull> for DirectionVector32 {
    type Output = [u32];

    fn index(&self, index: RangeFull) -> &Self::Output {
        &self.words[index]
    }
}

impl DirectionVector64 {
    /// Number of words in a 64-bit Sobol direction vector.
    pub const WORDS: usize = 64;

    /// Returns the direction vector as an array of words.
    pub const fn as_words(&self) -> &[u64; Self::WORDS] {
        &self.words
    }

    /// Returns the direction vector as a slice of words.
    pub const fn as_slice(&self) -> &[u64] {
        self.words.as_slice()
    }
}

impl AsRef<[u64]> for DirectionVector64 {
    fn as_ref(&self) -> &[u64] {
        self.as_slice()
    }
}

impl Index<usize> for DirectionVector64 {
    type Output = u64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.words[index]
    }
}

impl Index<RangeFull> for DirectionVector64 {
    type Output = [u64];

    fn index(&self, index: RangeFull) -> &Self::Output {
        &self.words[index]
    }
}

const _: () = {
    assert!(size_of::<DirectionVector32>() == size_of::<sys::curandDirectionVectors32_t>());
    assert!(align_of::<DirectionVector32>() == align_of::<sys::curandDirectionVectors32_t>());
    assert!(size_of::<DirectionVector64>() == size_of::<sys::curandDirectionVectors64_t>());
    assert!(align_of::<DirectionVector64>() == align_of::<sys::curandDirectionVectors64_t>());
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_direction_vector_tables() -> Result<()> {
        let vectors = direction_vectors_32(DirectionVectorSet::JoeKuo6_32)?;
        assert_eq!(vectors.len(), MAX_QUASI_RANDOM_DIMENSIONS);
        assert_eq!(vectors[0].as_slice().len(), DirectionVector32::WORDS);
        assert_eq!(vectors[0][..].len(), DirectionVector32::WORDS);

        let vectors = direction_vectors_64(DirectionVectorSet::JoeKuo6_64)?;
        assert_eq!(vectors.len(), MAX_QUASI_RANDOM_DIMENSIONS);
        assert_eq!(vectors[0].as_slice().len(), DirectionVector64::WORDS);
        assert_eq!(vectors[0][..].len(), DirectionVector64::WORDS);

        Ok(())
    }
}
