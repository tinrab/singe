//! Inference layers over checked operations, run on tensors and recorded by capture.
//! Models pair a layer with immutable named state; layers retain keys rather than independently replaceable tensors.

mod checkpoint;
mod decode;
mod encode;
mod layers;
mod model;
mod state;

pub(crate) use self::state::joined_rows;
pub use self::{
    checkpoint::{Checkpoint, CheckpointOptions, Derivation, LoadPlan, StatePiece, TensorSource},
    decode::{
        CacheSize, DecodeCache, DecodeStep, DecoderModel, DecodingLayer, EncoderDecoderCache,
        EncoderDecoderCacheSpec, HybridCache, HybridCacheSpec, InputGroup, KvCache, KvCacheSpec,
        LatentKvCache, LatentKvCacheSpec, PACKED_ORDER, PageLayout, PageTable, PositionCache,
        SourceCache, SourceCacheSpec, SourcedCache, SourcedCacheSpec, StateCache, StateCacheSpec,
        TensorGroup, caches_from_inputs, named_cache_inputs,
    },
    encode::{Placement, TokenBatch},
    layers::*,
    model::{Layer, Model, ModelProgram, Sequential},
    state::{ModelState, StateBuilder, StateKey, StateScope},
};

#[cfg(test)]
mod tests;
