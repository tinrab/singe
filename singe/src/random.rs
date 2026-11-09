//! Seeded, counter-based random numbers shared by every backend.
//!
//! A [`Generator`] owns a seed and an offset counted in 128-bit Philox blocks.
//! Element `i` of a random operation takes lane `i % 4` of Philox 4x32-10 with key `seed` and counter `offset + i / 4`, so CPU and CUDA produce the same bits.

use std::{
    collections::HashMap,
    f32::consts::TAU,
    sync::{
        Arc, Mutex, MutexGuard, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

use crate::{
    device::Device,
    error::{Error, Result},
    graph::capture,
    shape::Shape,
    tensor::Tensor,
    utility::Locked,
};

const PHILOX_M0: u32 = 0xD251_1F53;
const PHILOX_M1: u32 = 0xCD9E_8D57;
const PHILOX_W0: u32 = 0x9E37_79B9;
const PHILOX_W1: u32 = 0xBB67_AE85;

/// Seed used by default generators before `manual_seed`.
pub const DEFAULT_SEED: u64 = 67_280_421_310_721;

/// A shared source of random numbers: a seed and the offset of the next unused Philox block.
///
/// Cloning a generator shares its state.
#[derive(Debug, Clone)]
pub struct Generator {
    id: u64,
    state: Arc<Mutex<GeneratorState>>,
}

/// The seed and next unused Philox block of a generator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GeneratorState {
    /// Philox key.
    pub seed: u64,
    /// Next unused Philox counter.
    pub offset: u64,
}

impl Generator {
    /// Creates an independent generator starting at offset zero.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            state: Arc::new(Mutex::new(GeneratorState { seed, offset: 0 })),
        }
    }

    /// Returns the default generator of `device`, shared by random operations that name no generator.
    ///
    /// # Errors
    ///
    /// Returns an error if a thread panicked while holding the default generator table.
    pub fn default_for(device: Device) -> Result<Self> {
        static DEFAULTS: OnceLock<Mutex<HashMap<Device, Generator>>> = OnceLock::new();
        let mut defaults = DEFAULTS.get_or_init(Mutex::default).locked("generator")?;
        Ok(defaults
            .entry(device)
            .or_insert_with(|| Self::new(DEFAULT_SEED))
            .clone())
    }

    /// Returns a process-local identity shared by clones of this generator.
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Returns the current seed and offset.
    ///
    /// # Errors
    ///
    /// Returns an error if a thread panicked while holding this generator.
    pub fn state(&self) -> Result<GeneratorState> {
        Ok(*self.lock()?)
    }

    /// Replaces the seed and offset.
    ///
    /// # Errors
    ///
    /// Returns an error if a thread panicked while holding this generator.
    pub fn set_state(&self, state: GeneratorState) -> Result<()> {
        *self.lock()? = state;
        Ok(())
    }

    /// Sets the seed and restarts at offset zero.
    ///
    /// # Errors
    ///
    /// Returns an error if a thread panicked while holding this generator.
    pub fn manual_seed(&self, seed: u64) -> Result<()> {
        self.set_state(GeneratorState { seed, offset: 0 })
    }

    /// Returns the state at which the next `blocks` Philox blocks start, and advances past them.
    ///
    /// # Errors
    ///
    /// Returns an error if a thread panicked while holding this generator or the offset overflows.
    pub(crate) fn reserve(&self, blocks: u64) -> Result<GeneratorState> {
        let mut state = self.lock()?;
        let reserved = *state;
        state.offset = state
            .offset
            .checked_add(blocks)
            .ok_or_else(|| Error::out_of_range("generator offset"))?;
        drop(state);
        Ok(reserved)
    }

    fn lock(&self) -> Result<MutexGuard<'_, GeneratorState>> {
        self.state.locked("generator")
    }
}

impl GeneratorState {
    /// Returns this state as the `I64` vector `[seed, offset]` that random operations read on `device`.
    ///
    /// # Errors
    ///
    /// Returns an error if tensor creation fails.
    pub fn tensor(self, device: Device) -> Result<Tensor> {
        Tensor::from_vec(
            device,
            Shape::create([2])?,
            vec![self.seed.cast_signed(), self.offset.cast_signed()],
        )
    }
}

/// Returns the generator state operand and the block offset a random operation uses.
///
/// Outside capture the state is a literal and the offset is zero.
/// During capture the generator becomes a graph input and the offset counts blocks used earlier in the graph.
pub(crate) fn operand(generator: &Generator, device: Device, blocks: u64) -> Result<(Tensor, u64)> {
    if let Some(bound) = capture::generator_input(generator, device, blocks)? {
        return Ok(bound);
    }

    let state = generator.reserve(blocks)?;
    Ok((capture::literal(device, |device| state.tensor(device))?, 0))
}

/// Returns the number of Philox blocks that `elements` values with `lanes` values per block consume.
#[must_use]
pub fn blocks(elements: usize, lanes: usize) -> u64 {
    u64::try_from(elements.div_ceil(lanes)).unwrap_or(u64::MAX)
}

/// Computes Philox 4x32-10 of a 64-bit counter under a 64-bit key.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    reason = "Philox splits 64-bit keys, counters, and products into 32-bit halves"
)]
pub fn philox(seed: u64, counter: u64) -> [u32; 4] {
    let mut key = [seed as u32, (seed >> 32) as u32];
    let mut lanes = [counter as u32, (counter >> 32) as u32, 0, 0];
    for _ in 0..10 {
        let product0 = u64::from(PHILOX_M0) * u64::from(lanes[0]);
        let product1 = u64::from(PHILOX_M1) * u64::from(lanes[2]);
        lanes = [
            (product1 >> 32) as u32 ^ lanes[1] ^ key[0],
            product1 as u32,
            (product0 >> 32) as u32 ^ lanes[3] ^ key[1],
            product0 as u32,
        ];
        key[0] = key[0].wrapping_add(PHILOX_W0);
        key[1] = key[1].wrapping_add(PHILOX_W1);
    }
    lanes
}

/// Maps 32 random bits to a value in `[0, 1)` with 24 bits of precision.
#[must_use]
pub fn unit_float(bits: u32) -> f32 {
    (bits >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// Returns uniform `[0, 1)` values for elements `0..count` at `state` plus `offset` blocks.
pub(crate) fn unit_values(state: GeneratorState, offset: u64, count: usize) -> Vec<f32> {
    (0..count)
        .map(|element| {
            let block = state.offset + offset + (element / 4) as u64;
            unit_float(philox(state.seed, block)[element % 4])
        })
        .collect()
}

/// Returns standard normal values for elements `0..count`, each from two lanes through Box-Muller.
pub(crate) fn normal_values(state: GeneratorState, offset: u64, count: usize) -> Vec<f32> {
    (0..count)
        .map(|element| {
            let block = state.offset + offset + (element / 2) as u64;
            let lanes = philox(state.seed, block);
            let pair = (element % 2) * 2;
            let radius = (-2.0 * (1.0 - unit_float(lanes[pair])).ln()).sqrt();
            radius * (TAU * unit_float(lanes[pair + 1])).cos()
        })
        .collect()
}
