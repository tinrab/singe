use singe_cudnn_sys as sys;

use crate::metadata::backend_metadata_enum;

backend_metadata_enum! {
    /// Type of performance knob.
    /// Performance knobs are runtime engine settings that affect performance.
    /// Query performance knobs and their valid ranges from a finalized
    /// [`BackendDescriptorType::Engine`](crate::backend::descriptor::BackendDescriptorType::Engine) with
    /// [`BackendDescriptor::attribute_descriptor_slice`](crate::backend::descriptor::BackendDescriptor::attribute_descriptor_slice).
    /// Configure the selected value for each knob on a
    /// [`BackendDescriptorType::KnobChoice`](crate::backend::descriptor::BackendDescriptorType::KnobChoice)
    /// with [`BackendDescriptor`](crate::backend::descriptor::BackendDescriptor) attribute methods.
    pub enum BackendKnobType: sys::cudnnBackendKnobType_t {
        SplitK => CUDNN_KNOB_TYPE_SPLIT_K => "Deprecated: Split-K factor.",
        Swizzle => CUDNN_KNOB_TYPE_SWIZZLE => "Memory access swizzle pattern for coalescing.",
        TileSize => CUDNN_KNOB_TYPE_TILE_SIZE => "Thread block tile size for computation.",
        UseTex => CUDNN_KNOB_TYPE_USE_TEX => "Deprecated: Use texture memory.",
        Edge => CUDNN_KNOB_TYPE_EDGE => "Edge/boundary handling mode.",
        KBlock => CUDNN_KNOB_TYPE_KBLOCK => "Deprecated: K-block size.",
        Ldga => CUDNN_KNOB_TYPE_LDGA => "Deprecated: Load granularity A.",
        Ldgb => CUDNN_KNOB_TYPE_LDGB => "Deprecated: Load granularity B.",
        ChunkK => CUDNN_KNOB_TYPE_CHUNK_K => "Deprecated: Chunk-K size.",
        SplitH => CUDNN_KNOB_TYPE_SPLIT_H => "Deprecated: Split-H factor.",
        WinoTile => CUDNN_KNOB_TYPE_WINO_TILE => "Deprecated: Winograd tile size.",
        Multiply => CUDNN_KNOB_TYPE_MULTIPLY => "Multiplication factor for tiling.",
        SplitKBuf => CUDNN_KNOB_TYPE_SPLIT_K_BUF => "Split-K with separate output buffers.",
        TileK => CUDNN_KNOB_TYPE_TILEK => "Tile size along K (contraction) dimension.",
        Stages => CUDNN_KNOB_TYPE_STAGES => "Number of pipeline stages.",
        ReductionMode => CUDNN_KNOB_TYPE_REDUCTION_MODE => "Cross-thread/cross-block reduction strategy.",
        CtaSplitKMode => CUDNN_KNOB_TYPE_CTA_SPLIT_K_MODE => "Deprecated: CTA split-K mode.",
        SplitKSlc => CUDNN_KNOB_TYPE_SPLIT_K_SLC => "Split-K slice count.",
        IdxMode => CUDNN_KNOB_TYPE_IDX_MODE => "Index computation mode.",
        Sliced => CUDNN_KNOB_TYPE_SLICED => "Deprecated: Sliced mode.",
        SplitRs => CUDNN_KNOB_TYPE_SPLIT_RS => "Deprecated: Split-RS factor.",
        SingleBuffer => CUDNN_KNOB_TYPE_SINGLEBUFFER => "Deprecated: Single buffer mode.",
        Ldgc => CUDNN_KNOB_TYPE_LDGC => "Deprecated: Load granularity C.",
        SpecFilt => CUDNN_KNOB_TYPE_SPECFILT => "Specialized filter processing mode.",
        KernelCfg => CUDNN_KNOB_TYPE_KERNEL_CFG => "Kernel configuration selector.",
        Workspace => CUDNN_KNOB_TYPE_WORKSPACE => "Workspace size preference.",
        // Native engines still report this deprecated constant, matching the C++ knob vocabulary.
        #[expect(deprecated, reason = "wraps deprecated upstream API values that remain part of the supported surface")]
        TileCga => CUDNN_KNOB_TYPE_TILE_CGA => "Deprecated: CGA tile size.",
        TileCgaM => CUDNN_KNOB_TYPE_TILE_CGA_M => "CGA cluster tile size along M dimension.",
        TileCgaN => CUDNN_KNOB_TYPE_TILE_CGA_N => "CGA cluster tile size along N dimension.",
        BlockSize => CUDNN_KNOB_TYPE_BLOCK_SIZE => "Thread block size (threads per block).",
        Occupancy => CUDNN_KNOB_TYPE_OCCUPANCY => "Target occupancy level.",
        ArraySizePerThread => CUDNN_KNOB_TYPE_ARRAY_SIZE_PER_THREAD => "Register array size per thread.",
        SplitCols => CUDNN_KNOB_TYPE_SPLIT_COLS => "Column split factor for parallelism.",
        TileRows => CUDNN_KNOB_TYPE_TILE_ROWS => "Tile size along row dimension.",
        TileCols => CUDNN_KNOB_TYPE_TILE_COLS => "Tile size along column dimension.",
        LoadSize => CUDNN_KNOB_TYPE_LOAD_SIZE => "Memory load granularity.",
        CtaCount => CUDNN_KNOB_TYPE_CTA_COUNT => "Number of CTAs (thread blocks) to launch.",
        StreamK => CUDNN_KNOB_TYPE_STREAM_K => "Stream-K work distribution mode.",
        SplitPSlc => CUDNN_KNOB_TYPE_SPLIT_P_SLC => "Split-P slice count.",
        TileM => CUDNN_KNOB_TYPE_TILE_M => "Tile size along M (output rows) dimension.",
        TileN => CUDNN_KNOB_TYPE_TILE_N => "Tile size along N (output cols) dimension.",
        WarpSpecCfg => CUDNN_KNOB_TYPE_WARP_SPEC_CFG => "Warp specialization configuration.",
        SwapAb => CUDNN_KNOB_TYPE_SWAP_AB => "Swap A and B operands in matmul.",
        InputTmaEnable => CUDNN_KNOB_TYPE_INPUT_TMA_ENABLE => "Enable TMA for input tensors.",
        OutputTmaEnable => CUDNN_KNOB_TYPE_OUTPUT_TMA_ENABLE => "Enable TMA for output tensors.",
    }
}
