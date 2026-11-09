// The choices a hand-written kernel leaves open, stated outright so a tuning record never depends on a list that can change.
// Each kernel's `Default` is what it does untuned, and a planner reads a record only if the record's choice is among those valid for the operands and target.

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use singe_cuda::architecture::GpuArchitecture;

use crate::{
    backend::cuda::{
        kernel_plan::source::{KernelInput, KernelPlanner, mma::MmaTiling},
        tuning::TunedChoice,
    },
    data_type::DataType,
    error::Error,
    graph::NumericalPolicy,
};

// Fused attention on tensor cores, plain or over a paged cache: a block is `warps` warps of 16 query rows each, reading `key_tile` keys at a time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AttentionParameters {
    pub warps: usize,
    pub key_tile: usize,
    // How many tiles of keys and values shared memory holds, so that with more than one the next is on its way while a block folds this one, as FlashInfer's `NUM_STAGES_SMEM` does.
    pub stages: usize,
    // A launch of fewer blocks than this splits its keys to fill the GPU, and zero never splits.
    pub split_below_blocks: usize,
    pub min_blocks: Option<u32>,
}

impl Default for AttentionParameters {
    fn default() -> Self {
        Self {
            warps: 4,
            key_tile: 64,
            stages: 1,
            split_below_blocks: 64,
            min_blocks: None,
        }
    }
}

impl AttentionParameters {
    pub const fn query_rows(self) -> usize {
        self.warps * 16
    }

    // What the kernel's tiles of keys and values take, for rows of `row` values of two bytes.
    pub const fn shared_bytes(self, row: usize) -> usize {
        2 * self.stages * self.key_tile * row * 2
    }

    // Whether the kernel's tiles of keys and values fit in the shared memory it may ask for, for rows of `dim` values and a pad, two bytes each.
    pub const fn fits(self, row: usize) -> bool {
        self.shared_bytes(row) <= MAX_DYNAMIC_SHARED
    }
}

// Latent attention over a paged cache: `warps` warps per group of 16 rows, reading `key_tile` keys at a time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MlaParameters {
    pub warps: usize,
    pub key_tile: usize,
    // The blocks a few long sequences split their keys into.
    pub target_blocks: usize,
    pub min_blocks: Option<u32>,
}

impl Default for MlaParameters {
    fn default() -> Self {
        Self {
            warps: 4,
            key_tile: 32,
            target_blocks: 128,
            min_blocks: None,
        }
    }
}

// Grouped matmul on tensor cores, reading `depth` of the shared dimension at a time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GroupedParameters {
    pub depth: usize,
    pub min_blocks: Option<u32>,
}

impl Default for GroupedParameters {
    fn default() -> Self {
        Self {
            depth: 32,
            min_blocks: None,
        }
    }
}

// A matmul on tensor cores: how a block tiles the product, and when it splits its depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MmaParameters {
    pub tiling: MmaTiling,
    pub split_below_blocks: usize,
    pub min_blocks: Option<u32>,
}

// A matmul of a few rows as dot products: `warps` output columns per block, a warp per column, each lane reading one vector of the weight at a time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DotProductParameters {
    pub warps: usize,
    // Whether lanes read 32 bytes at a time, on targets that have 256-bit loads, rather than 16.
    pub wide_loads: bool,
    pub min_blocks: Option<u32>,
}

impl Default for DotProductParameters {
    fn default() -> Self {
        Self {
            warps: 8,
            wide_loads: false,
            min_blocks: None,
        }
    }
}

// A kernel that splits a long row among blocks for lack of rows to fill the GPU: whether it does, and with how many threads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SplitParameters {
    pub enabled: bool,
    pub threads: u32,
}

impl SplitParameters {
    pub const fn with_threads(threads: u32) -> Self {
        Self {
            enabled: true,
            threads,
        }
    }
}

// Query and key rotation with the cache write: the threads that take one head, the threads of a block, and the launch bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RotaryCacheParameters {
    pub threads_per_head: u32,
    pub threads_per_block: u32,
    pub min_blocks: Option<u32>,
}

impl RotaryCacheParameters {
    // The threads per head vLLM's kernel fixes from the head size.
    #[must_use]
    pub const fn for_head(head_dim: usize) -> Self {
        Self {
            threads_per_head: if head_dim <= 64 {
                8
            } else if head_dim <= 192 {
                16
            } else {
                32
            },
            threads_per_block: 128,
            min_blocks: None,
        }
    }
}

// What a kernel's parameters were, and so which kernel runs where a node has several.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KernelParameters {
    Attention(AttentionParameters),
    Mla(MlaParameters),
    Grouped(GroupedParameters),
    Mma(MmaParameters),
    DotProduct(DotProductParameters),
    Split(SplitParameters),
    RotaryCache(RotaryCacheParameters),
    // The kernel IR's general kernel, whose own choices are its schedule.
    General,
}

// The threads a target's SM holds at once, which bounds how many blocks of a size fit on one.
fn threads_per_sm(target: GpuArchitecture) -> u32 {
    match target.compute_capability() {
        (7, 5) => 1024,
        (8, 6 | 7 | 9) | (10, 1) | (11, 0) | (12, _) => 1536,
        _ => 2048,
    }
}

// The launch bound worth trying for blocks of `threads` on `target`: the blocks that fit on one SM, up to four, as vLLM's kernels ask for.
pub fn min_blocks_per_sm(target: Option<GpuArchitecture>, threads: usize) -> Option<u32> {
    let threads = u32::try_from(threads).ok().filter(|&threads| threads > 0)?;
    let blocks = (threads_per_sm(target?) / threads).clamp(1, 4);
    (blocks > 1).then_some(blocks)
}

// Whether `target` assembles 256-bit loads, as `ptxas` does from `sm_100` on, including `sm_120`.
pub fn has_wide_loads(target: Option<GpuArchitecture>) -> bool {
    target.is_some_and(|target| target.compute_capability() >= (10, 0))
}

// Whether `target` assembles `tcgen05`, as `ptxas` does only for these targets and not for plain `sm_100` or `sm_103`, nor any `sm_120`.
pub const fn has_tcgen05(target: Option<GpuArchitecture>) -> bool {
    matches!(
        target,
        Some(GpuArchitecture::Sm100a | GpuArchitecture::Sm100f | GpuArchitecture::Sm103a)
    )
}

// An input's type and the elements its layout addresses, as `value_key` writes them.
fn input_key(input: &KernelInput, key: &mut String) {
    let _ = write!(
        key,
        " {:?}{:?}{:?}+{}",
        input.data_type(),
        input.layout().dimensions(),
        input.layout().addressing_strides(),
        input.layout().addressing_offset() % 8,
    );
}

// The key of a kernel's parameters: what it computes, on what, under which numerical policy.
pub fn parameter_key(label: &str, inputs: &[KernelInput], attributes: &str) -> String {
    let mut key = format!(
        "kernel|{label}|{:?}|{attributes}",
        NumericalPolicy::current()
    );
    for input in inputs {
        input_key(input, &mut key);
    }
    key
}

impl KernelPlanner {
    // The parameters a record gives this kernel, if it gives any.
    pub(crate) fn tuned_parameters(&self, key: &str) -> Option<KernelParameters> {
        let records = self.context.records()?;
        match records.get(key) {
            Some(TunedChoice::Parameters(parameters)) => {
                records.note_used(key);
                Some(parameters)
            }
            _ => None,
        }
    }

    // Fails for a record whose parameters its kernel no longer takes, since planning the default instead would quietly run something tuning did not choose.
    pub(crate) fn stale_parameters(key: &str) -> Error {
        Error::StaleRecord {
            key: key.to_owned(),
        }
    }

    // The launch bound candidates of a kernel of `threads` threads on this planner's target.
    pub(crate) fn launch_bound_candidates(&self, threads: usize) -> Vec<Option<u32>> {
        let mut bounds = vec![None];
        bounds.extend(min_blocks_per_sm(self.context.target, threads).map(Some));
        bounds
    }
}

// The widest static shared memory a kernel declares.
pub const MAX_STATIC_SHARED: usize = 48 * 1024;

// The most shared memory an attention kernel asks a launch for, which every target of `mma.sync` offers.
// NVIDIA's own `sm_120` attention, in TensorRT, takes up to 73,728 bytes.
pub const MAX_DYNAMIC_SHARED: usize = 96 * 1024;

// The data types tensor-core kernels read, whose elements take two bytes.
pub const fn half_precision(data_type: DataType) -> bool {
    matches!(data_type, DataType::F16 | DataType::Bf16)
}
