use num_enum::{IntoPrimitive, TryFromPrimitive};
use serde::{Deserialize, Serialize};
use singe_cudnn_sys as sys;

use singe_core::{impl_enum_conversion, impl_enum_display};

/// Type of performance knob.
/// Performance knobs are runtime engine settings that affect performance.
/// Query performance knobs and their valid ranges from a finalized
/// [`BackendDescriptorType::Engine`](crate::descriptor::BackendDescriptorType::Engine) with
/// [`BackendDescriptor::attribute_descriptor_slice`](crate::descriptor::BackendDescriptor::attribute_descriptor_slice).
/// Configure the selected value for each knob on a
/// [`BackendDescriptorType::KnobChoice`](crate::descriptor::BackendDescriptorType::KnobChoice)
/// with [`BackendDescriptor`](crate::descriptor::BackendDescriptor) attribute methods.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive, Serialize, Deserialize,
)]
#[repr(u32)]
#[non_exhaustive]
pub enum BackendKnobType {
    SplitK = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_SPLIT_K as _,
    Swizzle = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_SWIZZLE as _,
    TileSize = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_TILE_SIZE as _,
    UseTex = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_USE_TEX as _,
    Edge = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_EDGE as _,
    KBlock = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_KBLOCK as _,
    Ldga = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_LDGA as _,
    Ldgb = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_LDGB as _,
    ChunkK = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_CHUNK_K as _,
    SplitH = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_SPLIT_H as _,
    WinoTile = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_WINO_TILE as _,
    Multiply = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_MULTIPLY as _,
    SplitKBuf = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_SPLIT_K_BUF as _,
    TileK = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_TILEK as _,
    Stages = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_STAGES as _,
    ReductionMode = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_REDUCTION_MODE as _,
    CtaSplitKMode = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_CTA_SPLIT_K_MODE as _,
    SplitKSlc = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_SPLIT_K_SLC as _,
    IdxMode = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_IDX_MODE as _,
    Sliced = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_SLICED as _,
    SplitRs = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_SPLIT_RS as _,
    SingleBuffer = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_SINGLEBUFFER as _,
    Ldgc = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_LDGC as _,
    SpecFilt = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_SPECFILT as _,
    KernelCfg = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_KERNEL_CFG as _,
    Workspace = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_WORKSPACE as _,
    TileCgaM = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_TILE_CGA_M as _,
    TileCgaN = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_TILE_CGA_N as _,
    BlockSize = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_BLOCK_SIZE as _,
    Occupancy = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_OCCUPANCY as _,
    ArraySizePerThread = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_ARRAY_SIZE_PER_THREAD as _,
    SplitCols = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_SPLIT_COLS as _,
    TileRows = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_TILE_ROWS as _,
    TileCols = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_TILE_COLS as _,
    LoadSize = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_LOAD_SIZE as _,
    CtaCount = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_CTA_COUNT as _,
    StreamK = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_STREAM_K as _,
    SplitPSlc = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_SPLIT_P_SLC as _,
    TileM = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_TILE_M as _,
    TileN = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_TILE_N as _,
    SwapAb = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_SWAP_AB as _,
    WarpSpecCfg = sys::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_WARP_SPEC_CFG as _,
}

impl_enum_conversion!(sys::cudnnBackendKnobType_t, BackendKnobType);

impl_enum_display!(BackendKnobType, {
    Self::SplitK => "CUDNN_KNOB_TYPE_SPLIT_K",
    Self::Swizzle => "CUDNN_KNOB_TYPE_SWIZZLE",
    Self::TileSize => "CUDNN_KNOB_TYPE_TILE_SIZE",
    Self::UseTex => "CUDNN_KNOB_TYPE_USE_TEX",
    Self::Edge => "CUDNN_KNOB_TYPE_EDGE",
    Self::KBlock => "CUDNN_KNOB_TYPE_KBLOCK",
    Self::Ldga => "CUDNN_KNOB_TYPE_LDGA",
    Self::Ldgb => "CUDNN_KNOB_TYPE_LDGB",
    Self::ChunkK => "CUDNN_KNOB_TYPE_CHUNK_K",
    Self::SplitH => "CUDNN_KNOB_TYPE_SPLIT_H",
    Self::WinoTile => "CUDNN_KNOB_TYPE_WINO_TILE",
    Self::Multiply => "CUDNN_KNOB_TYPE_MULTIPLY",
    Self::SplitKBuf => "CUDNN_KNOB_TYPE_SPLIT_K_BUF",
    Self::TileK => "CUDNN_KNOB_TYPE_TILEK",
    Self::Stages => "CUDNN_KNOB_TYPE_STAGES",
    Self::ReductionMode => "CUDNN_KNOB_TYPE_REDUCTION_MODE",
    Self::CtaSplitKMode => "CUDNN_KNOB_TYPE_CTA_SPLIT_K_MODE",
    Self::SplitKSlc => "CUDNN_KNOB_TYPE_SPLIT_K_SLC",
    Self::IdxMode => "CUDNN_KNOB_TYPE_IDX_MODE",
    Self::Sliced => "CUDNN_KNOB_TYPE_SLICED",
    Self::SplitRs => "CUDNN_KNOB_TYPE_SPLIT_RS",
    Self::SingleBuffer => "CUDNN_KNOB_TYPE_SINGLEBUFFER",
    Self::Ldgc => "CUDNN_KNOB_TYPE_LDGC",
    Self::SpecFilt => "CUDNN_KNOB_TYPE_SPECFILT",
    Self::KernelCfg => "CUDNN_KNOB_TYPE_KERNEL_CFG",
    Self::Workspace => "CUDNN_KNOB_TYPE_WORKSPACE",
    Self::TileCgaM => "CUDNN_KNOB_TYPE_TILE_CGA_M",
    Self::TileCgaN => "CUDNN_KNOB_TYPE_TILE_CGA_N",
    Self::BlockSize => "CUDNN_KNOB_TYPE_BLOCK_SIZE",
    Self::Occupancy => "CUDNN_KNOB_TYPE_OCCUPANCY",
    Self::ArraySizePerThread => "CUDNN_KNOB_TYPE_ARRAY_SIZE_PER_THREAD",
    Self::SplitCols => "CUDNN_KNOB_TYPE_SPLIT_COLS",
    Self::TileRows => "CUDNN_KNOB_TYPE_TILE_ROWS",
    Self::TileCols => "CUDNN_KNOB_TYPE_TILE_COLS",
    Self::LoadSize => "CUDNN_KNOB_TYPE_LOAD_SIZE",
    Self::CtaCount => "CUDNN_KNOB_TYPE_CTA_COUNT",
    Self::StreamK => "CUDNN_KNOB_TYPE_STREAM_K",
    Self::SplitPSlc => "CUDNN_KNOB_TYPE_SPLIT_P_SLC",
    Self::TileM => "CUDNN_KNOB_TYPE_TILE_M",
    Self::TileN => "CUDNN_KNOB_TYPE_TILE_N",
    Self::SwapAb => "CUDNN_KNOB_TYPE_SWAP_AB",
    Self::WarpSpecCfg => "CUDNN_KNOB_TYPE_WARP_SPEC_CFG",
});
