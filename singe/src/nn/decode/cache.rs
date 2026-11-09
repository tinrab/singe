use std::fmt::Debug;

use crate::{
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    graph::capture,
    index::TensorIndex,
    nn::{DecodeStep, InputGroup, SourcePages, decode::input_group, pages_for},
    shape::Shape,
    tensor::Tensor,
};

// A paged cache a decoder's blocks read and extend, whose tensors a compiled step takes as inputs.
pub trait DecodeCache: InputGroup {
    // What sizes the cache besides its pages.
    type Spec: Copy + Debug + Send + Sync;

    // Allocates a zeroed cache of `size` on `device`.
    fn create(
        device: Device,
        data_type: DataType,
        size: CacheSize,
        spec: Self::Spec,
    ) -> Result<Self>;
}

// How much a decoder's caches hold: `pages` pages of `page_size` slots, shared by `sequences` sequences.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheSize {
    pub pages: i64,
    pub page_size: i64,
    pub sequences: i64,
}

// Names `caches`' tensors as the inputs of a compiled step.
// Preparing freezes bound state, so a cache is always an input, passed with `GraphPrepareOptions::with_inputs_in_place`.
pub fn named_cache_inputs<C: DecodeCache>(caches: &[C]) -> Vec<(String, Tensor)> {
    caches
        .iter()
        .enumerate()
        .flat_map(|(layer, cache)| {
            cache
                .named_tensors()
                .into_iter()
                .map(move |(name, tensor)| (format!("cache.{layer}.{name}"), tensor))
        })
        .collect()
}

// Rebuilds caches from `tensors`, in the order [`named_cache_inputs`] gives them, as a capture closure receives them.
pub fn caches_from_inputs<C: DecodeCache>(tensors: &[Tensor]) -> Result<Vec<C>> {
    if !tensors.len().is_multiple_of(C::COUNT) {
        return Err(Error::missing_input("decoder cache inputs"));
    }
    tensors
        .chunks_exact(C::COUNT)
        .map(C::from_tensors)
        .collect()
}

// What sizes a [`KvCache`] besides its pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCacheSpec {
    pub kv_heads: i64,
    pub head_dim: i64,
}

input_group! {
    // One attention layer's cached keys and values, `[P, S, G, D]`.
    pub struct KvCache {
        keys,
        values,
    }
}

impl KvCache {
    // Writes `keys` and `values`, `[T, G, D]`, at the step's slots.
    pub fn write(&self, step: &DecodeStep, keys: &Tensor, values: &Tensor) -> Result<()> {
        self.keys.index_put_(&step.slots(), keys)?;
        self.values.index_put_(&step.slots(), values)
    }

    // Returns the cached keys.
    #[must_use]
    pub const fn keys(&self) -> &Tensor {
        &self.keys
    }

    // Returns the cached values.
    #[must_use]
    pub const fn values(&self) -> &Tensor {
        &self.values
    }
}

impl DecodeCache for KvCache {
    type Spec = KvCacheSpec;

    fn create(
        device: Device,
        data_type: DataType,
        size: CacheSize,
        spec: KvCacheSpec,
    ) -> Result<Self> {
        let shape = Shape::create([size.pages, size.page_size, spec.kv_heads, spec.head_dim])?;
        Ok(Self {
            keys: Tensor::zeroed_on(device, data_type, shape.clone())?,
            values: Tensor::zeroed_on(device, data_type, shape)?,
        })
    }
}

// What sizes a [`LatentKvCache`] besides its pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatentKvCacheSpec {
    pub latent_dim: i64,
    pub positional_dim: i64,
}

input_group! {
    // One latent attention layer's compressed values and positional keys, `[P, S, C]` and `[P, S, R]`.
    pub struct LatentKvCache {
        latent,
        positional,
    }
}

impl LatentKvCache {
    // Writes `latent` and `positional`, `[T, C]` and `[T, R]`, at the step's slots.
    pub fn write(&self, step: &DecodeStep, latent: &Tensor, positional: &Tensor) -> Result<()> {
        self.latent.index_put_(&step.slots(), latent)?;
        self.positional.index_put_(&step.slots(), positional)
    }

    // Returns the cached compressed values.
    #[must_use]
    pub const fn latent(&self) -> &Tensor {
        &self.latent
    }

    // Returns the cached positional keys.
    #[must_use]
    pub const fn positional(&self) -> &Tensor {
        &self.positional
    }
}

impl DecodeCache for LatentKvCache {
    type Spec = LatentKvCacheSpec;

    fn create(
        device: Device,
        data_type: DataType,
        size: CacheSize,
        spec: LatentKvCacheSpec,
    ) -> Result<Self> {
        let cache = |dim: i64| {
            Tensor::zeroed_on(
                device,
                data_type,
                Shape::create([size.pages, size.page_size, dim])?,
            )
        };
        Ok(Self {
            latent: cache(spec.latent_dim)?,
            positional: cache(spec.positional_dim)?,
        })
    }
}

// What sizes an [`EncoderDecoderCache`] besides its pages and sequences.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncoderDecoderCacheSpec {
    pub heads: i64,
    pub head_dim: i64,
    // How many encoder outputs each sequence's cross-attention reads.
    pub sources: i64,
    // The slots of a cross-attention page, a power of two, as cuDNN's paged attention takes.
    pub source_page: i64,
}

input_group! {
    // One decoder layer's caches in an encoder-decoder: paged self-attention keys and values, `[P, S, H, D]`, and each sequence's cross-attention keys and values over its encoder's `F` outputs, in `k` pages of its own, `[N · k, page, H, D]`, which are written once per sequence and do not change while it decodes.
    pub struct EncoderDecoderCache {
        keys,
        values,
        cross_keys,
        cross_values,
    }
}

impl EncoderDecoderCache {
    // Returns the self-attention cache, which shares these tensors.
    #[must_use]
    pub fn self_attention(&self) -> KvCache {
        KvCache {
            keys: self.keys.clone(),
            values: self.values.clone(),
        }
    }

    #[must_use]
    pub const fn cross_keys(&self) -> &Tensor {
        &self.cross_keys
    }

    #[must_use]
    pub const fn cross_values(&self) -> &Tensor {
        &self.cross_values
    }

    // Writes one sequence's cross-attention keys and values, `[F, H, D]`, as its encoder computed them, to its pages.
    pub fn write_cross(&self, sequence: usize, keys: &Tensor, values: &Tensor) -> Result<()> {
        let sequence = i64::try_from(sequence).map_err(|_| Error::out_of_range("sequence"))?;
        let &[sources, heads, dim] = keys.shape().dimensions() else {
            return Err(Error::invalid_argument(
                "cross-attention keys and values are [sources, heads, dim]",
            ));
        };
        let layout = SourcePages::create(&self.cross_keys, sources)?;
        let (page, blocks, first) = (layout.page, layout.per_sequence, layout.first(sequence)?);
        let rows = [TensorIndex::Indices(capture::literal(
            self.cross_keys.device(),
            |device| {
                Tensor::from_vec(
                    device,
                    Shape::create([blocks])?,
                    (first..first + blocks).collect(),
                )
            },
        )?)];
        let paged = |tensor: &Tensor| {
            tensor
                .pad(&[(0, blocks * page - sources), (0, 0), (0, 0)], 0.0)?
                .reshape([blocks, page, heads, dim])
        };
        self.cross_keys.index_put_(&rows, &paged(keys)?)?;
        self.cross_values.index_put_(&rows, &paged(values)?)
    }
}

impl DecodeCache for EncoderDecoderCache {
    type Spec = EncoderDecoderCacheSpec;

    fn create(
        device: Device,
        data_type: DataType,
        size: CacheSize,
        spec: EncoderDecoderCacheSpec,
    ) -> Result<Self> {
        let zeroed =
            |dimensions: [i64; 4]| Tensor::zeroed_on(device, data_type, Shape::create(dimensions)?);
        let paged = [size.pages, size.page_size, spec.heads, spec.head_dim];
        if spec.source_page <= 0 || spec.source_page.count_ones() != 1 {
            return Err(Error::invalid_argument(
                "cross-attention pages hold a power of two slots",
            ));
        }

        let cross = [
            size.sequences * pages_for(spec.sources, spec.source_page),
            spec.source_page,
            spec.heads,
            spec.head_dim,
        ];
        Ok(Self {
            keys: zeroed(paged)?,
            values: zeroed(paged)?,
            cross_keys: zeroed(cross)?,
            cross_values: zeroed(cross)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::PageTable;

    const fn size(pages: i64, page_size: i64) -> CacheSize {
        CacheSize {
            pages,
            page_size,
            sequences: 1,
        }
    }

    #[test]
    fn test_caches_write_at_their_step_slots() {
        let mut table = PageTable::create(1, 3, 2).unwrap();
        let spec = KvCacheSpec {
            kv_heads: 1,
            head_dim: 1,
        };
        let cache = KvCache::create(Device::Cpu, DataType::F32, size(2, 2), spec).unwrap();
        let step = table.step(Device::Cpu, &[&[1, 2, 3]]).unwrap();
        let values = Tensor::from_vec(
            Device::Cpu,
            Shape::create([3, 1, 1]).unwrap(),
            vec![1_f32, 2., 3.],
        )
        .unwrap();
        cache.write(&step, &values, &values).unwrap();
        let written = cache.keys().to_vec::<f32>().unwrap();
        let mut found = written.clone();
        found.sort_by(f32::total_cmp);
        assert_eq!(found, [0., 1., 2., 3.]);
        let pages = step.paging().pages.to_vec::<i64>().unwrap();
        let at = |page: i64, slot: i64| written[usize::try_from(page * 2 + slot).unwrap()];
        assert_eq!(at(pages[0], 0), 1.);
        assert_eq!(at(pages[0], 1), 2.);
        assert_eq!(at(pages[1], 0), 3.);
    }

    #[test]
    fn test_caches_rebuild_from_their_named_inputs() {
        let spec = LatentKvCacheSpec {
            latent_dim: 2,
            positional_dim: 3,
        };
        let caches = [
            LatentKvCache::create(Device::Meta, DataType::F32, size(1, 2), spec).unwrap(),
            LatentKvCache::create(Device::Meta, DataType::F32, size(1, 2), spec).unwrap(),
        ];
        let named = named_cache_inputs(&caches);
        assert_eq!(named[3].0, "cache.1.positional");
        let tensors = named
            .into_iter()
            .map(|(_, tensor)| tensor)
            .collect::<Vec<_>>();
        let rebuilt = caches_from_inputs::<LatentKvCache>(&tensors).unwrap();
        assert_eq!(rebuilt.len(), 2);
        assert_eq!(rebuilt[1].positional().shape().dimensions(), [1, 2, 3]);
    }
}
