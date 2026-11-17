//! Sobol direction-vector and scramble-constant tables.
//!
//! cuRAND owns these host tables for the lifetime of the program, so they are returned as `&'static` slices.

use std::{
    ops::{Index, RangeFull},
    ptr, slice,
};

use singe_curand_sys as sys;

use crate::{
    error::{Error, Result},
    try_ffi,
    types::DirectionVectorSet,
};

/// Number of Sobol dimensions in the direction-vector and scramble-constant tables.
///
/// The Joe-Kuo direction-vector sets returned by [`direction_vectors_32`] and [`direction_vectors_64`] contain 20,000 dimensions.
/// [`Generator::set_quasi_random_dimensions`](crate::generator::Generator::set_quasi_random_dimensions) accepts at most this many dimensions.
pub const MAX_QUASI_RANDOM_DIMENSIONS: usize = 20_000;

/// Direction vector for one 32-bit Sobol dimension.
///
/// Each dimension has 32 `u32` words.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DirectionVector32 {
    words: [u32; Self::WORDS],
}

/// Direction vector for one 64-bit Sobol dimension.
///
/// Each dimension has 64 `u64` words.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DirectionVector64 {
    words: [u64; Self::WORDS],
}

/// Returns the library-owned table of 32-bit Sobol direction vectors for `set`, one per dimension.
///
/// # Errors
///
/// Returns an error if `set` is not valid for 32-bit direction vectors or if cuRAND returns a null table.
pub fn direction_vectors_32(set: DirectionVectorSet) -> Result<&'static [DirectionVector32]> {
    let mut vectors = ptr::null_mut();
    // SAFETY: `vectors` is a valid output location.
    unsafe {
        try_ffi!(sys::curandGetDirectionVectors32(
            &raw mut vectors,
            set.into(),
        ))?;
    }

    // SAFETY: cuRAND returns its static table of `MAX_QUASI_RANDOM_DIMENSIONS` vectors, and `DirectionVector32` has the native layout.
    unsafe { static_table(vectors.cast(), "curand direction vectors") }
}

/// Returns the library-owned table of 64-bit Sobol direction vectors for `set`, one per dimension.
///
/// # Errors
///
/// Returns an error if `set` is not valid for 64-bit direction vectors or if cuRAND returns a null table.
pub fn direction_vectors_64(set: DirectionVectorSet) -> Result<&'static [DirectionVector64]> {
    let mut vectors = ptr::null_mut();
    // SAFETY: `vectors` is a valid output location.
    unsafe {
        try_ffi!(sys::curandGetDirectionVectors64(
            &raw mut vectors,
            set.into(),
        ))?;
    }

    // SAFETY: cuRAND returns its static table of `MAX_QUASI_RANDOM_DIMENSIONS` vectors, and `DirectionVector64` has the native layout.
    unsafe { static_table(vectors.cast(), "curand direction vectors") }
}

/// Returns the library-owned 32-bit scramble constants for scrambled Sobol generation, one per dimension.
///
/// # Errors
///
/// Returns an error if cuRAND returns a null table.
pub fn scramble_constants_32() -> Result<&'static [u32]> {
    let mut constants = ptr::null_mut();
    // SAFETY: `constants` is a valid output location.
    unsafe {
        try_ffi!(sys::curandGetScrambleConstants32(&raw mut constants))?;
    }

    // SAFETY: cuRAND returns its static table of `MAX_QUASI_RANDOM_DIMENSIONS` constants.
    unsafe { static_table(constants, "curand scramble constants") }
}

/// Returns the library-owned 64-bit scramble constants for scrambled Sobol generation, one per dimension.
///
/// # Errors
///
/// Returns an error if cuRAND returns a null table.
pub fn scramble_constants_64() -> Result<&'static [u64]> {
    let mut constants = ptr::null_mut();
    // SAFETY: `constants` is a valid output location.
    unsafe {
        try_ffi!(sys::curandGetScrambleConstants64(&raw mut constants))?;
    }

    // SAFETY: cuRAND returns its static table of `MAX_QUASI_RANDOM_DIMENSIONS` constants.
    unsafe { static_table(constants, "curand scramble constants") }
}

// A non-null `ptr` must point to `MAX_QUASI_RANDOM_DIMENSIONS` initialized values that stay alive and unchanged for the rest of the program.
unsafe fn static_table<T>(ptr: *mut T, name: &'static str) -> Result<&'static [T]> {
    if ptr.is_null() {
        return Err(Error::NullPointer { name: name.into() });
    }

    // SAFETY: `ptr` is non-null, and the caller guarantees it points to `MAX_QUASI_RANDOM_DIMENSIONS` initialized values that live for the rest of the program.
    Ok(unsafe { slice::from_raw_parts(ptr, MAX_QUASI_RANDOM_DIMENSIONS) })
}

impl DirectionVector32 {
    /// Number of words in a 32-bit Sobol direction vector.
    pub const WORDS: usize = 32;

    /// Returns the direction vector as an array of words.
    #[must_use]
    pub const fn as_words(&self) -> &[u32; Self::WORDS] {
        &self.words
    }

    /// Returns the direction vector as a slice of words.
    #[must_use]
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
    #[must_use]
    pub const fn as_words(&self) -> &[u64; Self::WORDS] {
        &self.words
    }

    /// Returns the direction vector as a slice of words.
    #[must_use]
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
    fn test_wraps_direction_vector_tables() -> Result<()> {
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
