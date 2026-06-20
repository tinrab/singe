use std::ptr;

use singe_cuda_sys::{driver, runtime};

use crate::{
    graph::{Extent, Position},
    memory::{ArrayHandle, MemoryCopyKind},
    types::HostFunction,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PitchedPtr {
    ptr: *mut (),
    pub pitch: usize,
    pub x_size: usize,
    pub y_size: usize,
}

impl PitchedPtr {
    /// Creates pitched pointer parameters from a raw device or mapped host pointer.
    ///
    /// # Safety
    ///
    /// `ptr` must be valid for every row described by `pitch`, `x_size`, and `y_size` when CUDA evaluates the graph node using this value.
    pub const unsafe fn new(ptr: *mut (), pitch: usize, x_size: usize, y_size: usize) -> Self {
        Self {
            ptr,
            pitch,
            x_size,
            y_size,
        }
    }

    pub const fn ptr(self) -> *mut () {
        self.ptr
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MemoryCopy3DNodeParams {
    src_array: Option<ArrayHandle>,
    src_pos: Position,
    src_ptr: PitchedPtr,
    dst_array: Option<ArrayHandle>,
    dst_pos: Position,
    dst_ptr: PitchedPtr,
    extent: Extent,
    kind: MemoryCopyKind,
}

#[derive(Debug, Clone, Copy)]
pub struct MemoryCopyToSymbolNodeParams {
    symbol: *const (),
    src: *const (),
    count: usize,
    offset: usize,
    kind: MemoryCopyKind,
}

#[derive(Debug, Clone, Copy)]
pub struct MemoryCopyFromSymbolNodeParams {
    dst: *mut (),
    symbol: *const (),
    count: usize,
    offset: usize,
    kind: MemoryCopyKind,
}

#[derive(Debug, Clone, Copy)]
pub struct HostNodeParams {
    func: HostFunction,
    user_data: *mut (),
}

impl HostNodeParams {
    /// Creates host callback node parameters from a raw user-data pointer.
    ///
    /// # Safety
    ///
    /// `user_data` must remain valid for `func` according to CUDA host-node callback rules until no graph execution can invoke the callback.
    pub const unsafe fn new(func: HostFunction, user_data: *mut ()) -> Self {
        Self { func, user_data }
    }

    pub const fn function(self) -> HostFunction {
        self.func
    }

    pub const fn user_data(self) -> *mut () {
        self.user_data
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MemoryCopy1DNodeParams {
    dst: *mut (),
    src: *const (),
    count: usize,
    kind: MemoryCopyKind,
}

impl MemoryCopyToSymbolNodeParams {
    /// Creates symbol-copy node parameters from a raw source pointer.
    ///
    /// # Safety
    ///
    /// `symbol` must identify a valid CUDA symbol and `src` must be valid
    /// for reads of `count` bytes according to `kind` when CUDA evaluates the graph node using this value.
    pub const unsafe fn new(
        symbol: *const (),
        src: *const (),
        count: usize,
        offset: usize,
        kind: MemoryCopyKind,
    ) -> Self {
        Self {
            symbol,
            src,
            count,
            offset,
            kind,
        }
    }

    pub const fn symbol(self) -> *const () {
        self.symbol
    }

    pub const fn src(self) -> *const () {
        self.src
    }

    pub const fn count(self) -> usize {
        self.count
    }

    pub const fn offset(self) -> usize {
        self.offset
    }

    pub const fn kind(self) -> MemoryCopyKind {
        self.kind
    }
}

impl MemoryCopyFromSymbolNodeParams {
    /// Creates symbol-copy node parameters from a raw destination pointer.
    ///
    /// # Safety
    ///
    /// `symbol` must identify a valid CUDA symbol and `dst` must be valid
    /// for writes of `count` bytes according to `kind` when CUDA evaluates
    /// the graph node using this value.
    pub const unsafe fn new(
        dst: *mut (),
        symbol: *const (),
        count: usize,
        offset: usize,
        kind: MemoryCopyKind,
    ) -> Self {
        Self {
            dst,
            symbol,
            count,
            offset,
            kind,
        }
    }

    pub const fn dst(self) -> *mut () {
        self.dst
    }

    pub const fn symbol(self) -> *const () {
        self.symbol
    }

    pub const fn count(self) -> usize {
        self.count
    }

    pub const fn offset(self) -> usize {
        self.offset
    }

    pub const fn kind(self) -> MemoryCopyKind {
        self.kind
    }
}

impl MemoryCopy1DNodeParams {
    /// Creates one-dimensional memcpy node parameters from raw pointers.
    ///
    /// # Safety
    ///
    /// `dst` and `src` must be valid for `count` bytes according to `kind` when CUDA evaluates the graph node using this value.
    pub const unsafe fn new(
        dst: *mut (),
        src: *const (),
        count: usize,
        kind: MemoryCopyKind,
    ) -> Self {
        Self {
            dst,
            src,
            count,
            kind,
        }
    }

    pub const fn dst(self) -> *mut () {
        self.dst
    }

    pub const fn src(self) -> *const () {
        self.src
    }

    pub const fn count(self) -> usize {
        self.count
    }

    pub const fn kind(self) -> MemoryCopyKind {
        self.kind
    }
}

impl MemoryCopy3DNodeParams {
    /// Creates three-dimensional memcpy node parameters from raw array and
    /// pitched pointer operands.
    ///
    /// # Safety
    ///
    /// Every array and pointer operand selected by `kind` must remain
    /// valid for the region described by the positions, pitched pointers,
    /// and extent whenever CUDA evaluates a graph node using this value.
    pub const unsafe fn new(
        src_array: Option<ArrayHandle>,
        src_pos: Position,
        src_ptr: PitchedPtr,
        dst_array: Option<ArrayHandle>,
        dst_pos: Position,
        dst_ptr: PitchedPtr,
        extent: Extent,
        kind: MemoryCopyKind,
    ) -> Self {
        Self {
            src_array,
            src_pos,
            src_ptr,
            dst_array,
            dst_pos,
            dst_ptr,
            extent,
            kind,
        }
    }

    pub const fn src_array(self) -> Option<ArrayHandle> {
        self.src_array
    }

    pub const fn src_pos(self) -> Position {
        self.src_pos
    }

    pub const fn src_ptr(self) -> PitchedPtr {
        self.src_ptr
    }

    pub const fn dst_array(self) -> Option<ArrayHandle> {
        self.dst_array
    }

    pub const fn dst_pos(self) -> Position {
        self.dst_pos
    }

    pub const fn dst_ptr(self) -> PitchedPtr {
        self.dst_ptr
    }

    pub const fn extent(self) -> Extent {
        self.extent
    }

    pub const fn kind(self) -> MemoryCopyKind {
        self.kind
    }
}

impl From<PitchedPtr> for runtime::cudaPitchedPtr {
    fn from(value: PitchedPtr) -> Self {
        Self {
            ptr: value.ptr().cast(),
            pitch: value.pitch as _,
            xsize: value.x_size as _,
            ysize: value.y_size as _,
        }
    }
}

impl From<&MemoryCopy3DNodeParams> for runtime::cudaMemcpy3DParms {
    fn from(value: &MemoryCopy3DNodeParams) -> Self {
        Self {
            srcArray: value
                .src_array()
                .map_or(ptr::null_mut(), ArrayHandle::as_raw),
            srcPos: value.src_pos().into(),
            srcPtr: value.src_ptr().into(),
            dstArray: value
                .dst_array()
                .map_or(ptr::null_mut(), ArrayHandle::as_raw),
            dstPos: value.dst_pos().into(),
            dstPtr: value.dst_ptr().into(),
            extent: value.extent().into(),
            kind: value.kind().into(),
        }
    }
}

impl From<&HostNodeParams> for driver::CUDA_HOST_NODE_PARAMS {
    fn from(value: &HostNodeParams) -> Self {
        Self {
            fn_: value.function().as_raw(),
            userData: value.user_data().cast(),
        }
    }
}
