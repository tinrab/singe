//! Eager cache writes land where the cache is, change only the step's slots, and allocate nothing.
#![cfg(feature = "testing")]

#[cfg(feature = "cuda")]
use singe::testing::maybe_lock_device;
use singe::{
    DataType, Device, Shape, Tensor,
    nn::{
        CacheSize, DecodeCache, EncoderDecoderCache, EncoderDecoderCacheSpec, KvCache, KvCacheSpec,
        PageTable,
    },
    testing::TensorActivity,
};

const PAGES: i64 = 4;
const PAGE_SIZE: i64 = 4;
const HEADS: i64 = 2;
const DIM: i64 = 8;

#[expect(clippy::cast_precision_loss, reason = "test values have few elements")]
fn values(shape: &[i64], seed: f32, device: Device) -> Tensor {
    let count = shape.iter().product::<i64>();
    let values = (0..count)
        .map(|index| index as f32 + seed)
        .collect::<Vec<_>>();
    Tensor::from_vec(device, Shape::create(shape.to_vec()).unwrap(), values).unwrap()
}

fn host(tensor: &Tensor) -> Vec<f32> {
    tensor.to_vec::<f32>().unwrap()
}

fn size(sequences: i64) -> CacheSize {
    CacheSize {
        pages: PAGES,
        page_size: PAGE_SIZE,
        sequences,
        capacity: PAGES * PAGE_SIZE,
    }
}

// A step of three tokens and one writes them in place, as the out-of-place `index_put` would, and requests no storage.
fn check_kv_cache(device: Device) {
    let spec = KvCacheSpec {
        kv_heads: HEADS,
        head_dim: DIM,
    };
    let cache = KvCache::create(device, DataType::F32, size(2), spec).unwrap();
    cache
        .keys()
        .assign(&values(&[PAGES, PAGE_SIZE, HEADS, DIM], 0.5, device))
        .unwrap();
    cache
        .values()
        .assign(&values(&[PAGES, PAGE_SIZE, HEADS, DIM], -0.5, device))
        .unwrap();
    let mut table = PageTable::create(2, 8, PAGE_SIZE).unwrap();
    let step = table.step(device, &[&[1, 2, 3], &[4]]).unwrap();
    let written = values(&[4, HEADS, DIM], 1_000.0, device);
    let expected = (
        host(
            &cache
                .keys()
                .index_put(&step.slots(), &written, false)
                .unwrap(),
        ),
        host(
            &cache
                .values()
                .index_put(&step.slots(), &written, false)
                .unwrap(),
        ),
    );
    let before = TensorActivity::current();
    cache.write(&step, &written, &written).unwrap();
    let requests = TensorActivity::current().storage_requests - before.storage_requests;
    assert_eq!(host(cache.keys()), expected.0, "keys on {device:?}");
    assert_eq!(host(cache.values()), expected.1, "values on {device:?}");
    assert_eq!(requests, 0, "storage requests on {device:?}");
}

// A sequence's cross-attention keys and values land in its own pages, and the other sequence's stay as they were.
fn check_cross_cache(device: Device) {
    let spec = EncoderDecoderCacheSpec {
        heads: HEADS,
        head_dim: DIM,
        sources: 6,
        source_page: 4,
    };
    let cache = EncoderDecoderCache::create(device, DataType::F32, size(2), spec).unwrap();
    let shape = cache.cross_keys().shape().dimensions().to_vec();
    cache
        .cross_keys()
        .assign(&values(&shape, 0.5, device))
        .unwrap();
    let untouched = host(cache.cross_keys());
    let written = values(&[6, HEADS, DIM], 1_000.0, device);
    cache.write_cross(1, &written, &written).unwrap();
    let after = host(cache.cross_keys());
    let page = usize::try_from(PAGE_SIZE * HEADS * DIM).unwrap();
    let row = usize::try_from(HEADS * DIM).unwrap();
    // Sequence 1 owns pages 2 and 3, whose first six slots hold its outputs and the rest zeros.
    assert_eq!(
        after[..2 * page],
        untouched[..2 * page],
        "sequence 0 on {device:?}"
    );
    assert_eq!(
        after[2 * page..2 * page + 6 * row],
        host(&written),
        "sequence 1 on {device:?}"
    );
    assert!(
        after[2 * page + 6 * row..]
            .iter()
            .all(|&value| value == 0.0),
        "padding on {device:?}"
    );
}

#[test]
fn test_cache_writes_land_in_place_on_cpu() {
    check_kv_cache(Device::Cpu);
    check_cross_cache(Device::Cpu);
}

#[cfg(feature = "cuda")]
#[test]
fn test_cache_writes_land_in_place_on_cuda() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    check_kv_cache(Device::Cuda(0));
    check_cross_cache(Device::Cuda(0));
}
