use std::{
    collections::{HashMap, hash_map::Entry},
    fmt::{Debug, Formatter, Result as FmtResult},
    hash::Hash,
    result::Result as StdResult,
    sync::{Mutex, PoisonError},
};

use singe::{
    Result as SingeResult,
    generation::{CachedDecoder, Decoder, DecoderOptions},
    nn::{DecoderModel, Model},
};

/// Decoders a model or pipeline keeps across its runs, by `K`, since preparing a compiled decoder costs far more than a run of it.
/// [`DecoderCache::run`] resets a kept decoder before it reuses it.
pub(crate) struct DecoderCache<K, D> {
    held: Mutex<HashMap<K, D>>,
}

impl<K, D> Default for DecoderCache<K, D> {
    fn default() -> Self {
        Self {
            held: Mutex::new(HashMap::new()),
        }
    }
}

/// A clone starts empty and prepares decoders of its own.
impl<K, D> Clone for DecoderCache<K, D> {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl<K, D> Debug for DecoderCache<K, D> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        let held = self.held.lock().map_or(0, |held| held.len());
        formatter
            .debug_struct("DecoderCache")
            .field("held", &held)
            .finish()
    }
}

impl<K: Hash + Eq, D: Decoder> DecoderCache<K, D> {
    /// Runs `run` on the decoder kept for `key`, made by `create` the first time and reset before each later run.
    #[expect(
        clippy::significant_drop_tightening,
        reason = "a kept decoder serves one run at a time"
    )]
    pub(crate) fn run<T, E>(
        &self,
        key: K,
        create: impl FnOnce() -> StdResult<D, E>,
        run: impl FnOnce(&mut D) -> StdResult<T, E>,
    ) -> StdResult<T, E> {
        // A run that panicked left its decoder to be reset like any other.
        let mut held = self.held.lock().unwrap_or_else(PoisonError::into_inner);
        let decoder = match held.entry(key) {
            Entry::Occupied(entry) => {
                let decoder = entry.into_mut();
                decoder.reset();
                decoder
            }
            Entry::Vacant(entry) => entry.insert(create()?),
        };
        run(decoder)
    }
}

/// The key of a kept decoder: its rows, its capacity, and whether it compiles.
pub(crate) type DecoderKey = (usize, i64, bool);

/// Returns a capacity of at least `tokens`, rounded up to a power of two of at least 64, so runs of nearby lengths share one kept decoder.
#[must_use]
pub(crate) fn bucketed_capacity(tokens: usize) -> i64 {
    i64::try_from(tokens.max(64).next_power_of_two()).unwrap_or(i64::MAX)
}

pub(crate) fn cached_decoder<L: DecoderModel>(
    model: &Model<L>,
    options: DecoderOptions,
    compile: bool,
) -> SingeResult<CachedDecoder<L>> {
    let mut decoder = CachedDecoder::create(model, options)?;
    if compile {
        decoder.compile()?;
    }

    Ok(decoder)
}
