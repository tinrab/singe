//! Inference layers over checked operations, run on tensors and recorded by capture.
//! Models pair a layer with immutable named state; layers retain keys rather than independently replaceable tensors.

mod checkpoint;
mod decode;
mod encode;
mod layers;
mod model;
mod state;

pub use self::{
    checkpoint::{CheckpointOptions, StateNameMap, StatePiece, StatePieceMap},
    decode::{
        CacheSize, DecodeCache, DecodeStep, DecoderModel, DecodingLayer, EncoderDecoderCache,
        EncoderDecoderCacheSpec, HybridCache, HybridCacheSpec, InputGroup, KvCache, KvCacheSpec,
        LatentKvCache, LatentKvCacheSpec, PageLayout, PageTable, SourceCache, SourceCacheSpec,
        SourcedCache, SourcedCacheSpec, StateCache, StateCacheSpec, TensorGroup,
        caches_from_inputs, named_cache_inputs,
    },
    encode::TokenBatch,
    layers::*,
    model::{Layer, Model, ModelProgram, Sequential},
    state::{ModelState, StateBuilder, StateKey, StateScope},
};

#[cfg(test)]
mod tests;
