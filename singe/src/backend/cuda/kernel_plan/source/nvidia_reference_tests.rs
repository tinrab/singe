// NVIDIA's compiled `sm_120` attention, as a reference that shares no code with our kernels or with cuDNN.
// TensorRT's `bertQKVToContextPlugin` holds it as a C array.

use std::fs;

use half::f16;
use singe_cuda::{
    dim::Dim3,
    memory::DeviceMemory,
    module::{KernelParameters, LaunchConfig, ModuleImage},
};

use crate::{
    backend::cuda::runtime::runtime_for_device,
    data_type::DataType,
    device::Device,
    shape::Shape,
    tensor::{AttentionOptions, Tensor},
    testing::{maybe_lock_device, skip_if_stub},
};

const BATCH: usize = 2;
const HEADS: usize = 2;
const ROWS: usize = 64;
const DIM: usize = 64;

// One over the square root of `DIM`.
const SCALE: f32 = 0.125;

// `Fused_multihead_attention_params_v2` in `fused_multihead_attention_v2.h`, with its pointers as device addresses.
#[derive(Clone, Copy)]
#[repr(C)]
struct KernelArguments {
    qkv: u64,
    packed_mask: u64,
    output: u64,
    qkv_stride: i64,
    packed_mask_stride: i64,
    output_stride: i64,
    batch: i32,
    heads: i32,
    rows: i32,
    dim: i32,
    scale_scores: u32,
    scale_softmax: u32,
    scale_output: u32,
    enable_i2f_trick: bool,
    row_offsets: u64,
    interleaved: bool,
    ignore_single_batch: bool,
    force_unroll: bool,
    use_int8_scale_max: bool,
    heads_per_wave: i32,
    counters: u64,
    max_barriers: u64,
    sum_barriers: u64,
    locks: u64,
    max_scratch: u64,
    sum_scratch: u64,
    output_scratch: u64,
}

// The bytes of a C array of `0x..` values.
fn c_array_bytes(source: &str) -> Vec<u8> {
    let array = source.split_once('{').map_or(source, |(_, array)| array);
    array
        .split("0x")
        .skip(1)
        .map(|byte| u8::from_str_radix(&byte[..2], 16).unwrap())
        .collect()
}

fn half_bits(value: f32) -> u32 {
    u32::from(f16::from_f32(value).to_bits())
}

fn tensor(seed: u32) -> Tensor {
    let count = BATCH * HEADS * ROWS * DIM;
    let mut state = seed;
    let values = (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            f32::from(u16::try_from(state >> 16).unwrap()) / 32_768.0 - 1.0
        })
        .collect::<Vec<_>>();
    let shape =
        Shape::create([BATCH, HEADS, ROWS, DIM].map(|side| i64::try_from(side).unwrap())).unwrap();
    Tensor::from_vec(Device::Cuda(0), shape, values)
        .unwrap()
        .cast(DataType::F16)
        .unwrap()
}

#[test]
#[ignore = "reads NVIDIA's kernel from thirdparty/TensorRT, which is only for sm_120"]
fn test_attention_matches_nvidias_compiled_kernel() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let Some(runtime) = skip_if_stub(runtime_for_device(Device::Cuda(0))) else {
        return;
    };
    let source = fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../thirdparty/TensorRT/plugin/bertQKVToContextPlugin/fused_multihead_attention_v2/fused_multihead_attention_v2_fp16_64_64_kernel.sm120.cpp"
    ))
    .unwrap();
    let context = runtime.context();
    let module = context
        .load_module(&ModuleImage::new(&c_array_bytes(&source)))
        .unwrap();
    let function = module.function("fmha_v2_fp16_64_64_sm120_kernel").unwrap();

    let (query, key, value) = (tensor(1), tensor(2), tensor(3));
    let [query_bytes, key_bytes, value_bytes] =
        [&query, &key, &value].map(|operand| operand.to_bytes().unwrap());
    let row_bytes = DIM * 2;
    let mut qkv = Vec::with_capacity(3 * query_bytes.len());
    for batch in 0..BATCH {
        for row in 0..ROWS {
            // Each row holds each head's query, key, and value, side by side.
            for head in 0..HEADS {
                let start = ((batch * HEADS + head) * ROWS + row) * row_bytes;
                for operand in [&query_bytes, &key_bytes, &value_bytes] {
                    qkv.extend_from_slice(&operand[start..start + row_bytes]);
                }
            }
        }
    }

    let qkv_memory = DeviceMemory::from_slice(context, &qkv).unwrap();
    let output_memory = DeviceMemory::<u8>::zeroed(context, query_bytes.len()).unwrap();
    let row_offsets = DeviceMemory::from_slice(context, &[0_i32, 64, 128]).unwrap();
    let arguments = KernelArguments {
        qkv: qkv_memory.as_ptr() as u64,
        packed_mask: 0,
        output: output_memory.as_mut_ptr() as u64,
        qkv_stride: i64::try_from(3 * HEADS * row_bytes).unwrap(),
        packed_mask_stride: 0,
        output_stride: i64::try_from(HEADS * row_bytes).unwrap(),
        batch: i32::try_from(BATCH).unwrap(),
        heads: i32::try_from(HEADS).unwrap(),
        rows: i32::try_from(ROWS).unwrap(),
        dim: i32::try_from(DIM).unwrap(),
        // The kernel multiplies by these with `HMUL2`, so they hold half-precision bits.
        scale_scores: half_bits(SCALE),
        scale_softmax: half_bits(1.0),
        scale_output: half_bits(1.0),
        enable_i2f_trick: false,
        row_offsets: row_offsets.as_ptr() as u64,
        interleaved: false,
        ignore_single_batch: false,
        force_unroll: false,
        use_int8_scale_max: false,
        heads_per_wave: 0,
        counters: 0,
        max_barriers: 0,
        sum_barriers: 0,
        locks: 0,
        max_scratch: 0,
        sum_scratch: 0,
        output_scratch: 0,
    };
    // The table in `fused_multihead_attention_v2.h` gives this kernel 128 threads and 32,768 bytes of shared memory.
    let config = LaunchConfig::create(
        Dim3::new(
            u32::try_from(HEADS).unwrap(),
            u32::try_from(BATCH).unwrap(),
            1,
        ),
        Dim3::new(128, 1, 1),
        32_768,
    )
    .unwrap();
    let mut parameters = KernelParameters::new();
    parameters.owned_arg(arguments);
    // SAFETY: the arguments match the kernel's parameter struct, and every buffer outlives the synchronization below.
    unsafe { function.launch(&config, parameters) }.unwrap();
    context.synchronize().unwrap();
    let nvidia = output_memory
        .copy_to_host_vec()
        .unwrap()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|half| f16::from_le_bytes(*half).to_f32())
        .collect::<Vec<_>>();

    let ours = query
        .attend(&key, &value, None, None, AttentionOptions::default())
        .unwrap()
        .cast(DataType::F32)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();

    // Attention worked out on the host, which judges both kernels.
    let host = |operand: &Tensor| {
        operand
            .cast(DataType::F32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap()
    };
    let (query, key, value) = (host(&query), host(&key), host(&value));
    let scale = f64::from(SCALE);
    let (mut worst_ours, mut worst_nvidia) = (0.0_f64, 0.0_f64);
    for batch in 0..BATCH {
        for head in 0..HEADS {
            let at =
                |row: usize, column: usize| ((batch * HEADS + head) * ROWS + row) * DIM + column;
            for row in 0..ROWS {
                let scores = (0..ROWS)
                    .map(|other| {
                        scale
                            * (0..DIM)
                                .map(|column| {
                                    f64::from(query[at(row, column)])
                                        * f64::from(key[at(other, column)])
                                })
                                .sum::<f64>()
                    })
                    .collect::<Vec<_>>();
                let largest = scores.iter().copied().fold(f64::MIN, f64::max);
                let weights = scores
                    .iter()
                    .map(|score| (score - largest).exp())
                    .collect::<Vec<_>>();
                let total = weights.iter().sum::<f64>();
                for column in 0..DIM {
                    let expected = (0..ROWS)
                        .map(|other| weights[other] * f64::from(value[at(other, column)]))
                        .sum::<f64>()
                        / total;
                    let theirs = nvidia[((batch * ROWS + row) * HEADS + head) * DIM + column];
                    worst_ours =
                        worst_ours.max((f64::from(ours[at(row, column)]) - expected).abs());
                    worst_nvidia = worst_nvidia.max((f64::from(theirs) - expected).abs());
                }
            }
        }
    }
    eprintln!("worst difference from the reference: ours {worst_ours}, NVIDIA's {worst_nvidia}");
    assert!(worst_ours < 2e-3, "{worst_ours}");
    assert!(worst_nvidia < 0.2, "{worst_nvidia}");
}
