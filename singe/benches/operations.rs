// Times single operations eagerly at the shapes the models run them, as part of `just bench`.

use singe::{
    DataType, Device, Shape, Tensor, backend::cuda::synchronize, testing::maybe_lock_device,
};
use singe_testing::Benches;

const DEVICE: Device = Device::Cuda(0);

fn random(dimensions: &[i64], data_type: DataType) -> Tensor {
    Tensor::randn(
        DEVICE,
        Shape::create(dimensions.to_vec()).unwrap(),
        DataType::F32,
        None,
    )
    .unwrap()
    .cast(data_type)
    .unwrap()
}

// Runs `operation` and waits for the device, so each sample is one operation's whole time.
fn time(benches: &mut Benches, name: &str, operation: impl Fn() -> Tensor) {
    benches.run(name, || {
        operation();
        synchronize(DEVICE).unwrap();
    });
}

// Qwen3-0.6B's merged QKV projection for a decoding step and a 128-token prompt chunk, and its MLP's up projection.
fn matmul(benches: &mut Benches) {
    for (label, rows, inner, columns) in [
        ("qwen3-qkv/1", 1, 1024, 4096),
        ("qwen3-qkv/128", 128, 1024, 4096),
        ("qwen3-up/128", 128, 1024, 6144),
    ] {
        for data_type in [DataType::F32, DataType::Bf16] {
            let name = format!("matmul/{label}/{data_type:?}");
            if !benches.wants(&name) {
                continue;
            }

            let left = random(&[rows, inner], data_type);
            let right = random(&[inner, columns], data_type);
            time(benches, &name, || left.matmul(&right).unwrap());
        }
    }
}

// Whisper-tiny's and DINOv2-small's self-attention, six heads of 64 over their encoders' tokens.
fn attention(benches: &mut Benches) {
    for (label, tokens) in [("whisper-encoder", 1500), ("dinov2-518", 1370)] {
        for data_type in [DataType::F32, DataType::Bf16] {
            let name = format!("attention/{label}/{data_type:?}");
            if !benches.wants(&name) {
                continue;
            }

            let query = random(&[1, 6, tokens, 64], data_type);
            let key = random(&[1, 6, tokens, 64], data_type);
            let value = random(&[1, 6, tokens, 64], data_type);
            time(benches, &name, || {
                query
                    .scaled_dot_product_attention(&key, &value, None, false)
                    .unwrap()
            });
        }
    }
}

// TAESD's 64-channel 3 by 3 convolutions at 256 by 256 images' largest feature maps.
fn convolution(benches: &mut Benches) {
    for size in [64, 128] {
        for data_type in [DataType::F32, DataType::F16] {
            let name = format!("convolution/taesd-64x{size}/{data_type:?}");
            if !benches.wants(&name) {
                continue;
            }

            let input = random(&[1, 64, size, size], data_type);
            let weight = random(&[64, 64, 3, 3], data_type);
            let bias = random(&[64], data_type);
            time(benches, &name, || {
                input
                    .convolution2d(&weight, Some(&bias), (1, 1), (1, 1, 1, 1), (1, 1), 1)
                    .unwrap()
            });
        }
    }
}

fn main() {
    let Some(_lock) = maybe_lock_device() else {
        eprintln!("the benchmarks need a GPU");
        return;
    };
    let mut benches = Benches::from_args();
    matmul(&mut benches);
    attention(&mut benches);
    convolution(&mut benches);
    benches.finish();
}
