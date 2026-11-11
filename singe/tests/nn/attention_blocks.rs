//! Attention blocks against references of the kernel-dataset blocks, written at small shapes by `scripts/references/suites/blocks.py`.

use safetensors::SafeTensors;
#[cfg(all(feature = "testing", feature = "cuda"))]
use singe::testing::maybe_lock_device;
use singe::{
    DataType, Device, Shape, Tensor,
    nn::{
        Activation, AttentionInput, DecodeCache, DecodingLayer, DenseAttentionInput, GatedMlp,
        GroupedQueryAttention, GroupedQueryLayers, HeadProjection, KvCache, KvCacheSpec,
        LatentKvCache, LatentKvCacheSpec, LatentQuery, Layer, Linear, ModelState,
        MultiHeadLatentAttention, MultiHeadLatentLayers, PageTable, PreNormBlock, RmsNorm,
        RotaryFrequencies, RotaryLayout, RotarySpec, RotaryTable, StateBuilder, StateKey,
    },
    tensor::AttentionOptions,
};
use singe_testing::{Tolerance, assert_matches, host};

const EPS: f32 = 1e-6;
const CAPACITY: i64 = 8;
const PAGE_SIZE: i64 = 3;

struct Fixture<'a> {
    source: SafeTensors<'a>,
    device: Device,
    data_type: DataType,
}

impl Fixture<'_> {
    fn tensor(&self, name: &str) -> Tensor {
        let view = self.source.tensor(name).unwrap();
        let shape =
            Shape::create(view.shape().iter().map(|&dim| i64::try_from(dim).unwrap())).unwrap();
        Tensor::from_bytes(Device::Cpu, DataType::F32, &shape, view.data())
            .unwrap()
            .cast(self.data_type)
            .unwrap()
            .to_device(self.device)
            .unwrap()
    }

    fn expected(&self, name: &str) -> Vec<f32> {
        let view = self.source.tensor(name).unwrap();
        let shape =
            Shape::create(view.shape().iter().map(|&dim| i64::try_from(dim).unwrap())).unwrap();
        Tensor::from_bytes(Device::Cpu, DataType::F32, &shape, view.data())
            .unwrap()
            .to_vec::<f32>()
            .unwrap()
    }

    fn key(&self, state: &mut StateBuilder, name: &str) -> StateKey {
        state.parameter(name, self.tensor(name)).unwrap()
    }

    fn linear(&self, state: &mut StateBuilder, name: &str) -> Linear {
        Linear::new(self.key(state, name), None)
    }

    fn norm(&self, state: &mut StateBuilder, name: &str, width: i64) -> RmsNorm {
        RmsNorm::new(vec![width], Some(self.key(state, name)), EPS)
    }
}

fn rotary(fixture: &Fixture<'_>, head_dim: i64) -> RotaryTable {
    RotaryTable::create(
        fixture.device,
        fixture.data_type,
        RotarySpec {
            head_dim,
            frequencies: RotaryFrequencies::new(10000.0),
            layout: RotaryLayout::HalfSplit,
        },
        CAPACITY,
    )
    .unwrap()
}

fn grouped(
    fixture: &Fixture<'_>,
    state: &mut StateBuilder,
    norms: bool,
    causal: bool,
) -> GroupedQueryAttention {
    let (heads, kv_heads, dim) = (4, 2, 16);
    let query_norm = norms.then(|| fixture.norm(state, "gqa.q_norm", dim));
    let key_norm = norms.then(|| fixture.norm(state, "gqa.k_norm", dim));
    let normed = |projection: HeadProjection, norm: Option<_>| match norm {
        Some(norm) => projection.with_norm(norm),
        None => projection,
    };
    GroupedQueryAttention::create(
        GroupedQueryLayers {
            query: normed(
                HeadProjection::new(fixture.linear(state, "gqa.q"), heads, dim),
                query_norm,
            ),
            key: normed(
                HeadProjection::new(fixture.linear(state, "gqa.k"), kv_heads, dim),
                key_norm,
            ),
            value: HeadProjection::new(fixture.linear(state, "gqa.v"), kv_heads, dim),
            output: fixture.linear(state, "gqa.o"),
        },
        AttentionOptions {
            causal,
            ..AttentionOptions::default()
        },
    )
    .unwrap()
}

// Prefills three tokens, then decodes one, over a cache of pages too small to hold them in one.
fn check_gqa(fixture: &Fixture<'_>, label: &str, norms: bool, causal: bool, tolerance: Tolerance) {
    let mut builder = StateBuilder::new();
    let attention = grouped(fixture, &mut builder, norms, causal);
    let state = builder.finish().unwrap();
    let mut pages = PageTable::create(1, CAPACITY, PAGE_SIZE).unwrap();
    let cache = KvCache::create(
        fixture.device,
        fixture.data_type,
        pages.cache_size().unwrap(),
        KvCacheSpec {
            kv_heads: 2,
            head_dim: 16,
        },
    )
    .unwrap();
    let table = rotary(fixture, 16);
    let hidden = fixture.tensor("gqa.hidden");

    let step = pages.step(fixture.device, &[&[0, 0, 0]]).unwrap();
    let angles = table.angles(step.positions()).unwrap();
    let prefill = attention
        .decode(
            &AttentionInput {
                hidden: &hidden.narrow(0, 0, 3).unwrap(),
                step: &step,
                angles: Some(&angles),
                cache: &cache,
            },
            &state,
        )
        .unwrap();
    assert_matches(
        &host(&prefill),
        &fixture.expected(&format!("gqa.{label}.prefill")),
        tolerance,
        label,
    );
    // L1/018: the cache holds normed, rotated keys, where the step's slots say.
    let page_table = step.paging().pages;
    let first_page = page_table.to_vec::<i64>().unwrap()[0];
    assert_matches(
        &host(
            &cache
                .keys()
                .narrow(0, first_page, 1)
                .unwrap()
                .reshape([3, 2, 16])
                .unwrap(),
        ),
        &fixture.expected(&format!("gqa.{label}.keys")),
        tolerance,
        "cached keys",
    );

    let step = pages.step(fixture.device, &[&[0]]).unwrap();
    let angles = table.angles(step.positions()).unwrap();
    let decode = attention
        .decode(
            &AttentionInput {
                hidden: &hidden.narrow(0, 3, 1).unwrap(),
                step: &step,
                angles: Some(&angles),
                cache: &cache,
            },
            &state,
        )
        .unwrap();
    assert_matches(
        &host(&decode),
        &fixture.expected(&format!("gqa.{label}.decode")),
        tolerance,
        label,
    );
}

// L1/075: whole batches without a cache, causal for a decoder and open for an encoder.
fn check_dense(fixture: &Fixture<'_>, label: &str, causal: bool, tolerance: Tolerance) {
    let mut builder = StateBuilder::new();
    let attention = grouped(fixture, &mut builder, false, causal);
    let state = builder.finish().unwrap();
    let table = rotary(fixture, 16);
    let positions = Tensor::from_vec(
        fixture.device,
        Shape::create([4]).unwrap(),
        vec![0_i64, 1, 2, 3],
    )
    .unwrap();
    let angles = table.angles(&positions).unwrap();
    let output = attention
        .forward(
            &DenseAttentionInput {
                hidden: &fixture.tensor("gqa.dense.hidden"),
                angles: Some(&angles),
                key_lengths: None,
            },
            &state,
        )
        .unwrap();
    assert_matches(
        &host(&output),
        &fixture.expected(&format!("gqa.{label}")),
        tolerance,
        label,
    );
}

fn check_block(fixture: &Fixture<'_>, tolerance: Tolerance) {
    let mut builder = StateBuilder::new();
    let attention = grouped(fixture, &mut builder, true, true);
    let width = 32;
    let block = PreNormBlock::new(
        fixture.norm(&mut builder, "block.input_norm", width),
        attention,
        fixture.norm(&mut builder, "block.post_norm", width),
        GatedMlp::new(
            fixture.linear(&mut builder, "block.gate"),
            fixture.linear(&mut builder, "block.up"),
            fixture.linear(&mut builder, "block.down"),
            Activation::Silu,
        ),
    );
    let state: ModelState = builder.finish().unwrap();
    let mut pages = PageTable::create(1, CAPACITY, PAGE_SIZE).unwrap();
    let cache = KvCache::create(
        fixture.device,
        fixture.data_type,
        pages.cache_size().unwrap(),
        KvCacheSpec {
            kv_heads: 2,
            head_dim: 16,
        },
    )
    .unwrap();
    let table = rotary(fixture, 16);
    let step = pages.step(fixture.device, &[&[0, 0, 0]]).unwrap();
    let angles = table.angles(step.positions()).unwrap();
    let output = block
        .decode(
            &AttentionInput {
                hidden: &fixture.tensor("gqa.hidden").narrow(0, 0, 3).unwrap(),
                step: &step,
                angles: Some(&angles),
                cache: &cache,
            },
            &state,
        )
        .unwrap();
    assert_matches(
        &host(&output),
        &fixture.expected("block.output"),
        tolerance,
        "block",
    );
}

fn check_mla(fixture: &Fixture<'_>, tolerance: Tolerance) {
    let (heads, key_dim, rotary_dim, value_dim, latent_dim) = (2, 16, 8, 16, 32);
    let mut builder = StateBuilder::new();
    #[expect(
        clippy::cast_precision_loss,
        reason = "the scale is defined in F32, as the references compute it"
    )]
    let scale = 1.0 / ((key_dim + rotary_dim) as f32).sqrt();
    let query = LatentQuery::LowRank {
        compress: fixture.linear(&mut builder, "mla.q_a"),
        norm: fixture.norm(&mut builder, "mla.q_a_norm", 24),
        expand: fixture.linear(&mut builder, "mla.q_b"),
    };
    let layers = MultiHeadLatentLayers {
        query,
        compress: fixture.linear(&mut builder, "mla.kv_a"),
        latent_norm: fixture.norm(&mut builder, "mla.kv_a_norm", latent_dim),
        expand: fixture.key(&mut builder, "mla.kv_b"),
        output: fixture.linear(&mut builder, "mla.o"),
        heads,
        key_dim,
        rotary_dim,
        value_dim,
        latent_dim,
    };
    let attention = MultiHeadLatentAttention::create(
        layers,
        AttentionOptions {
            scale: Some(scale),
            causal: true,
            ..AttentionOptions::default()
        },
    )
    .unwrap();
    let state = builder.finish().unwrap();
    let mut pages = PageTable::create(1, CAPACITY, PAGE_SIZE).unwrap();
    let cache = LatentKvCache::create(
        fixture.device,
        fixture.data_type,
        pages.cache_size().unwrap(),
        LatentKvCacheSpec {
            latent_dim,
            positional_dim: rotary_dim,
        },
    )
    .unwrap();
    let table = rotary(fixture, rotary_dim);
    let hidden = fixture.tensor("mla.hidden");

    let step = pages.step(fixture.device, &[&[0, 0, 0]]).unwrap();
    let angles = table.angles(step.positions()).unwrap();
    let prefill = attention
        .decode(
            &AttentionInput {
                hidden: &hidden.narrow(0, 0, 3).unwrap(),
                step: &step,
                angles: Some(&angles),
                cache: &cache,
            },
            &state,
        )
        .unwrap();
    assert_matches(
        &host(&prefill),
        &fixture.expected("mla.prefill"),
        tolerance,
        "prefill",
    );

    let step = pages.step(fixture.device, &[&[0]]).unwrap();
    let angles = table.angles(step.positions()).unwrap();
    let decode = attention
        .decode(
            &AttentionInput {
                hidden: &hidden.narrow(0, 3, 1).unwrap(),
                step: &step,
                angles: Some(&angles),
                cache: &cache,
            },
            &state,
        )
        .unwrap();
    assert_matches(
        &host(&decode),
        &fixture.expected("mla.decode"),
        tolerance,
        "decode",
    );
}

fn check_all(device: Device, data_type: DataType, tolerance: Tolerance) {
    let bytes = include_bytes!("../references/blocks/blocks.safetensors");
    let fixture = Fixture {
        source: SafeTensors::deserialize(bytes).unwrap(),
        device,
        data_type,
    };
    check_gqa(&fixture, "causal_norms", true, true, tolerance);
    check_gqa(&fixture, "causal", false, true, tolerance);
    check_gqa(&fixture, "open_norms", true, false, tolerance);
    check_dense(&fixture, "dense_causal", true, tolerance);
    check_dense(&fixture, "dense_open", false, tolerance);
    check_block(&fixture, tolerance);
    check_mla(&fixture, tolerance);
}

// The absorbed and expanded forms of latent attention are equal only if the algebra is right, so F32 leaves room for arithmetic alone.
const F32: Tolerance = Tolerance::absolute(1e-3, "F32 leaves room for arithmetic alone");

#[test]
fn test_attention_blocks_match_references_in_f32_on_cpu() {
    check_all(Device::Cpu, DataType::F32, F32);
}

// `F32` runs the general kernels on the GPU.
#[cfg(all(feature = "testing", feature = "cuda"))]
#[test]
fn test_attention_blocks_match_references_in_f32_on_cuda() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    check_all(Device::Cuda(0), DataType::F32, F32);
}

#[cfg(all(feature = "testing", feature = "cuda"))]
#[test]
fn test_attention_blocks_match_references_in_half_precision_on_cuda() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    check_all(
        Device::Cuda(0),
        DataType::F16,
        Tolerance::absolute(0.02, "F16 keeps eleven mantissa bits through each block"),
    );
    check_all(
        Device::Cuda(0),
        DataType::Bf16,
        Tolerance::absolute(0.25, "BF16 keeps eight mantissa bits through each block"),
    );
}
