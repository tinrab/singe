use std::ptr;

use singe_cuda_sys::{driver, runtime};

use crate::{
    graph::{Extent, Position},
    memory::{ArrayHandle, MemoryCopyKind},
    types::HostFunction,
};

/// A raw base pointer with the byte pitch and logical dimensions of a pitched allocation.
///
/// This value does not retain the allocation or establish initialized access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PitchedPtr {
    ptr: *mut (),
    /// Byte distance between the start of adjacent rows.
    pub pitch: usize,
    /// Logical allocation width in bytes.
    pub x_size: usize,
    /// Logical allocation height in rows.
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

    /// Returns the raw allocation base address without granting host accessibility.
    #[must_use]
    pub const fn ptr(self) -> *mut () {
        self.ptr
    }
}

/// Raw array or pitched-pointer operands, positions, and extent for a three-dimensional graph copy.
///
/// This descriptor retains no operand owners; its constructor's storage contract applies to every graph execution that uses it.
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

/// Raw parameters for copying bytes to a CUDA symbol in a graph node.
///
/// The descriptor retains neither the symbol's code owner nor the source allocation.
#[derive(Debug, Clone, Copy)]
pub struct MemoryCopyToSymbolNodeParams {
    symbol: *const (),
    src: *const (),
    count: usize,
    offset: usize,
    kind: MemoryCopyKind,
}

/// Raw parameters for copying bytes from a CUDA symbol in a graph node.
///
/// The descriptor retains neither the symbol's code owner nor the destination allocation.
#[derive(Debug, Clone, Copy)]
pub struct MemoryCopyFromSymbolNodeParams {
    dst: *mut (),
    symbol: *const (),
    count: usize,
    offset: usize,
    kind: MemoryCopyKind,
}

/// A native host-node callback and its raw user-data pointer.
///
/// This descriptor owns neither callback code nor user data.
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

    /// Returns the native host callback wrapper.
    #[must_use]
    pub const fn function(self) -> HostFunction {
        self.func
    }

    /// Returns the unowned user-data pointer passed to the callback.
    #[must_use]
    pub const fn user_data(self) -> *mut () {
        self.user_data
    }
}

/// Raw pointers, a byte count, and a direction for a one-dimensional graph copy.
///
/// This descriptor retains no allocation owners.
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
    #[must_use]
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

    /// Returns the CUDA destination symbol address.
    #[must_use]
    pub const fn symbol(self) -> *const () {
        self.symbol
    }

    /// Returns the raw source address.
    #[must_use]
    pub const fn src(self) -> *const () {
        self.src
    }

    /// Returns the number of bytes to copy.
    #[must_use]
    pub const fn count(self) -> usize {
        self.count
    }

    /// Returns the destination byte offset within the symbol.
    #[must_use]
    pub const fn offset(self) -> usize {
        self.offset
    }

    /// Returns the configured memory-copy direction.
    #[must_use]
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

    /// Returns the raw destination address.
    #[must_use]
    pub const fn dst(self) -> *mut () {
        self.dst
    }

    /// Returns the CUDA source symbol address.
    #[must_use]
    pub const fn symbol(self) -> *const () {
        self.symbol
    }

    /// Returns the number of bytes to copy.
    #[must_use]
    pub const fn count(self) -> usize {
        self.count
    }

    /// Returns the source byte offset within the symbol.
    #[must_use]
    pub const fn offset(self) -> usize {
        self.offset
    }

    /// Returns the configured memory-copy direction.
    #[must_use]
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

    /// Returns the raw destination address.
    #[must_use]
    pub const fn dst(self) -> *mut () {
        self.dst
    }

    /// Returns the raw source address.
    #[must_use]
    pub const fn src(self) -> *const () {
        self.src
    }

    /// Returns the number of bytes to copy.
    #[must_use]
    pub const fn count(self) -> usize {
        self.count
    }

    /// Returns the configured memory-copy direction.
    #[must_use]
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
    #[must_use]
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

    /// Returns the source array handle, when the source is an array.
    #[must_use]
    pub const fn src_array(self) -> Option<ArrayHandle> {
        self.src_array
    }

    /// Returns the source starting position.
    #[must_use]
    pub const fn src_pos(self) -> Position {
        self.src_pos
    }

    /// Returns the pitched source parameters, used when no source array is selected.
    #[must_use]
    pub const fn src_ptr(self) -> PitchedPtr {
        self.src_ptr
    }

    /// Returns the destination array handle, when the destination is an array.
    #[must_use]
    pub const fn dst_array(self) -> Option<ArrayHandle> {
        self.dst_array
    }

    /// Returns the destination starting position.
    #[must_use]
    pub const fn dst_pos(self) -> Position {
        self.dst_pos
    }

    /// Returns the pitched destination parameters, used when no destination array is selected.
    #[must_use]
    pub const fn dst_ptr(self) -> PitchedPtr {
        self.dst_ptr
    }

    /// Returns the three-dimensional copy extent.
    #[must_use]
    pub const fn extent(self) -> Extent {
        self.extent
    }

    /// Returns the configured memory-copy direction.
    #[must_use]
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
