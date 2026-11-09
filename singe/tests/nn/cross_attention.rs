//! Cross-attention over paged encoder outputs against dense attention to the same outputs.

#[cfg(all(feature = "testing", feature = "cuda"))]
use singe::testing::maybe_lock_device;
use singe::{
    DataType, Device, Shape, Tensor,
    nn::{
        CacheSize, CrossAttention, CrossAttentionInput, DecodeCache, EncoderDecoderCache,
        EncoderDecoderCacheSpec, Layer, Linear, ModelState, PageTable, StateBuilder,
    },
    tensor::AttentionOptions,
};
use singe_testing::{Tolerance, assert_matches, host};

const WIDTH: i64 = 32;
const HEADS: i64 = 2;
const HEAD_DIM: i64 = WIDTH / HEADS;

// Values that differ everywhere and stay small, so no score saturates the softmax.
#[expect(clippy::cast_precision_loss, reason = "test values have few elements")]
fn values(shape: &[i64], seed: f32, device: Device, data_type: DataType) -> Tensor {
    let count = shape.iter().product::<i64>();
    let values = (0..count)
        .map(|index| (index as f32).mul_add(0.37, seed).sin() * 0.5)
        .collect::<Vec<_>>();
    Tensor::from_vec(device, Shape::create(shape.to_vec()).unwrap(), values)
        .unwrap()
        .cast(data_type)
        .unwrap()
}

struct Case {
    attention: CrossAttention,
    state: ModelState,
    // Each sequence's keys and values, `[F, H, D]`.
    sources: Vec<(Tensor, Tensor)>,
    query: Linear,
    output: Linear,
}

fn case(sources: i64, sequences: usize, device: Device, data_type: DataType) -> Case {
    let mut builder = StateBuilder::new();
    let mut linear = |name: &str, seed: f32| {
        let weight = builder
            .parameter(name, values(&[WIDTH, WIDTH], seed, device, data_type))
            .unwrap();
        Linear::new(weight, None)
    };
    let query = linear("query", 0.1);
    let key = linear("key", 0.2);
    let value = linear("value", 0.3);
    let output = linear("output", 0.4);
    let attention = CrossAttention::create(
        query.clone(),
        key,
        value,
        output.clone(),
        HEADS,
        WIDTH,
        sources,
    )
    .unwrap();
    let state = builder.finish().unwrap();
    #[expect(clippy::cast_precision_loss, reason = "a few sequences")]
    let sources = (0..sequences)
        .map(|sequence| {
            let encoded = values(&[sources, WIDTH], 1.0 + sequence as f32, device, data_type);
            attention.sources(&encoded, &state).unwrap()
        })
        .collect();
    Case {
        attention,
        state,
        sources,
        query,
        output,
    }
}

// Each token of `tokens` per sequence attends densely to its own sequence's keys and values.
fn dense(case: &Case, hidden: &Tensor, tokens: &[usize]) -> Vec<f32> {
    let queries = hidden.apply(&case.query, &case.state).unwrap();
    let mut rows = Vec::new();
    let mut first = 0;
    for (sequence, &count) in tokens.iter().enumerate() {
        let (keys, values) = &case.sources[sequence];
        let heads = |tensor: &Tensor| tensor.permute(&[1, 0, 2]).unwrap().unsqueeze(0).unwrap();
        let count = i64::try_from(count).unwrap();
        let query = queries
            .narrow(0, first, count)
            .unwrap()
            .reshape([count, HEADS, HEAD_DIM])
            .unwrap();
        let attended = heads(&query)
            .attend(
                &heads(keys),
                &heads(values),
                None,
                None,
                AttentionOptions::default(),
            )
            .unwrap()
            .squeeze(0)
            .unwrap()
            .permute(&[1, 0, 2])
            .unwrap()
            .reshape([count, WIDTH])
            .unwrap();
        rows.push(attended);
        first += count;
    }
    host(
        &Tensor::concat(&rows, 0)
            .unwrap()
            .apply(&case.output, &case.state)
            .unwrap(),
    )
}

// A batch whose sequences read different numbers of tokens attends through paged cross-attention as each token does densely, for encoder lengths that fill their last page and that do not, in one chunk of pages and in several.
fn check(device: Device, data_type: DataType, tolerance: Tolerance) {
    for (sources, page) in [(8, 4), (10, 4), (5, 8), (37, 4), (40, 4)] {
        let tokens = [3_usize, 1, 2];
        let case = case(sources, tokens.len(), device, data_type);
        let cache = EncoderDecoderCache::create(
            device,
            data_type,
            CacheSize {
                pages: 4,
                page_size: 4,
                sequences: 3,
            },
            EncoderDecoderCacheSpec {
                heads: HEADS,
                head_dim: HEAD_DIM,
                sources,
                source_page: page,
            },
        )
        .unwrap();
        for (sequence, (keys, values)) in case.sources.iter().enumerate() {
            cache.write_cross(sequence, keys, values).unwrap();
        }
        let ids = tokens
            .iter()
            .map(|&count| vec![0_i64; count])
            .collect::<Vec<_>>();
        let rows = ids.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let mut table = PageTable::create(tokens.len(), 4, 4).unwrap();
        let step = table.step(device, &rows).unwrap();
        let total = i64::try_from(tokens.iter().sum::<usize>()).unwrap();
        let hidden = values(&[total, WIDTH], 3.0, device, data_type);
        let paged = case
            .attention
            .forward(
                &CrossAttentionInput {
                    hidden: &hidden,
                    step: &step,
                    keys: cache.cross_keys(),
                    values: cache.cross_values(),
                },
                &case.state,
            )
            .unwrap();
        let expected = dense(&case, &hidden, &tokens);
        assert_matches(
            &host(&paged),
            &expected,
            tolerance,
            &format!("{sources} sources in pages of {page}, {data_type:?} on {device:?}"),
        );
    }
}

const F32: Tolerance = Tolerance::absolute(1e-5, "F32 leaves room for arithmetic alone");

#[test]
fn test_paged_cross_attention_matches_dense_attention_on_cpu() {
    check(Device::Cpu, DataType::F32, F32);
}

#[cfg(all(feature = "testing", feature = "cuda"))]
#[test]
fn test_paged_cross_attention_matches_dense_attention_on_cuda() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    check(Device::Cuda(0), DataType::F32, F32);
    check(
        Device::Cuda(0),
        DataType::Bf16,
        Tolerance::absolute(
            0.04,
            "each chunk of keys rounds its partial result to BF16 before the chunks combine",
        ),
    );
}
