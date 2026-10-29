//! Neighborhood aggregation on CSC-format graphs.

use std::{marker::PhantomData, mem::MaybeUninit, ptr};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cuda::{
    error::Error as CudaError,
    memory::DeviceMemory,
    stream::{Stream, StreamBinding},
    view::{
        AnyBitPatternDeviceRepr, DeviceInput, DeviceInputView, DeviceOutput, DeviceOutputView,
        DeviceRegionProvider, DeviceRepr, DeviceView,
    },
};
use singe_cudnn_sys as sys;

use crate::{
    data_type::{DataTypeLike, bf16, f16},
    error::{Error, Result},
    try_ffi,
    utility::{to_i32, to_i64},
};

/// Neighborhood aggregation operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum AggregationOp {
    /// Sums features from source neighbors.
    Sum = sys::cudnnGnnAggOp_t::CUDNN_GNN_AGG_SUM as _,
    /// Averages features from source neighbors.
    Mean = sys::cudnnGnnAggOp_t::CUDNN_GNN_AGG_MEAN as _,
    /// Selects the maximum feature and records its source-neighbor position.
    Max = sys::cudnnGnnAggOp_t::CUDNN_GNN_AGG_MAX as _,
    /// Selects the minimum feature and records its source-neighbor position.
    Min = sys::cudnnGnnAggOp_t::CUDNN_GNN_AGG_MIN as _,
}

impl_enum_conversion!(sys::cudnnGnnAggOp_t, AggregationOp);

impl_enum_display!(AggregationOp, {Self::Sum=>"CUDNN_GNN_AGG_SUM",Self::Mean=>"CUDNN_GNN_AGG_MEAN",Self::Max=>"CUDNN_GNN_AGG_MAX",Self::Min=>"CUDNN_GNN_AGG_MIN"});

mod sealed {
    use crate::data_type::{bf16, f16};

    pub trait Index {}

    impl Index for i32 {}

    impl Index for i64 {}

    pub trait Feature {}

    impl Feature for f32 {}

    impl Feature for f16 {}

    impl Feature for bf16 {}
}

/// Supported CSC graph indices: signed 32-bit or 64-bit integers.
pub trait GraphIndex: DataTypeLike + Into<i64> + AnyBitPatternDeviceRepr + sealed::Index {}

impl GraphIndex for i32 {}

impl GraphIndex for i64 {}

/// Supported aggregation feature elements: float, half and bfloat16.
pub trait FeatureScalar: DataTypeLike + sealed::Feature {}

impl FeatureScalar for f32 {}

impl FeatureScalar for f16 {}

impl FeatureScalar for bf16 {}

/// CSC-format graph with validated device indices.
/// Remapping arrays identify positions in an edge-feature table when its order differs
/// from CSC or reversed-CSC order. All index arrays have the same integer type.
#[derive(Debug)]
pub struct CscGraph<I: GraphIndex> {
    offsets: DeviceMemory<I>,
    indices: DeviceMemory<I>,
    csc_to_coo: Option<DeviceMemory<I>>,
    reverse_to_coo: Option<DeviceMemory<I>>,
    source_nodes: usize,
    destination_nodes: usize,
    stream: Stream,
}

impl<I: GraphIndex> CscGraph<I> {
    /// Uploads a CSC graph. Offsets have length `destination_nodes + 1`, start at
    /// zero, are nondecreasing, and end at the number of indices.
    ///
    /// # Errors
    ///
    /// Returns an error if graph counts cannot fit native `i64` values, CSC offsets or source indices are invalid, an optional edge mapping has the wrong length or out-of-range entries, context binding fails, or device allocation or upload fails.
    pub fn create(
        stream: &Stream,
        source_nodes: usize,
        offsets: &[I],
        indices: &[I],
        csc_to_coo: Option<&[I]>,
        reverse_to_coo: Option<&[I]>,
    ) -> Result<Self> {
        let edges = to_i64(indices.len(), "edge count")?;
        let source_count = to_i64(source_nodes, "source nodes")?;
        if offsets.is_empty()
            || offsets[0].into() != 0
            || offsets.last().map(|v| (*v).into()) != Some(edges)
            || offsets.windows(2).any(|v| v[0].into() > v[1].into())
            || indices
                .iter()
                .any(|&i| i.into() < 0 || i.into() >= source_count)
        {
            return Err(Error::InvalidDataShape);
        }
        for mapping in [csc_to_coo, reverse_to_coo].into_iter().flatten() {
            if mapping.len() != indices.len()
                || mapping.iter().any(|&i| i.into() < 0 || i.into() >= edges)
            {
                return Err(Error::InvalidDataShape);
            }
        }
        stream.context().bind()?;
        Ok(Self {
            offsets: DeviceMemory::from_slice(stream.context(), offsets)?,
            indices: DeviceMemory::from_slice(stream.context(), indices)?,
            csc_to_coo: csc_to_coo
                .map(|values| DeviceMemory::from_slice(stream.context(), values))
                .transpose()?,
            reverse_to_coo: reverse_to_coo
                .map(|values| DeviceMemory::from_slice(stream.context(), values))
                .transpose()?,
            source_nodes,
            destination_nodes: offsets.len() - 1,
            stream: stream.clone(),
        })
    }

    fn raw(&self) -> sys::cudnnGnnCscGraph_t {
        sys::cudnnGnnCscGraph_t {
            cscOffsets: self.offsets.as_ptr().cast(),
            cscIndices: self.indices.as_ptr().cast(),
            mapCscToCoo: self
                .csc_to_coo
                .as_ref()
                .map_or(ptr::null(), |v| v.as_ptr().cast()),
            mapRevToCoo: self
                .reverse_to_coo
                .as_ref()
                .map_or(ptr::null(), |v| v.as_ptr().cast()),
            nSrcNodes: i64::try_from(self.source_nodes)
                .expect("graph dimensions fit the native i64 interface"),
            nDstNodes: i64::try_from(self.destination_nodes)
                .expect("graph dimensions fit the native i64 interface"),
            nIndices: i64::try_from(self.indices.len())
                .expect("graph dimensions fit the native i64 interface"),
            idxType: I::data_type().into(),
        }
    }

    /// Performs neighborhood aggregation on a CSC-format graph using NVRTC-compiled
    /// kernels (JIT-compiled on first call per (dataType, idxType), cached thereafter).
    /// The JIT cache is process-global and thread-safe.
    ///
    /// For each destination node, aggregates features from its source neighbors
    /// using the specified operation (sum, mean, max, min).
    ///
    /// Completes queued work before returning, including failure and unwind.
    /// Only the required output prefix is produced; tails remain unchanged.
    /// Requires SM 8.0 (Ampere) or later.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid feature widths, missing node and edge features, insufficient buffer capacity, overflowing sizes or native-width conversions, mismatched CUDA contexts, allocation or stream-dependency failure, native aggregation failure, or completion synchronization failure.
    pub fn forward<T: FeatureScalar>(
        &self,
        op: AggregationOp,
        node: Option<Features<DeviceView<'_, T>>>,
        edge: Option<Features<DeviceView<'_, T>>>,
        concat: Option<Features<DeviceView<'_, T>>>,
        output: &mut (impl DeviceOutput<T> + ?Sized),
    ) -> Result<Aggregation<'_, I, T>> {
        let dims = [
            node.as_ref().map_or(0, |f| f.width),
            edge.as_ref().map_or(0, |f| f.width),
            concat.as_ref().map_or(0, |f| f.width),
        ];
        let aggregated_width = dims[0]
            .checked_add(dims[1])
            .ok_or(Error::InvalidDataShape)?;
        let mut positions = if matches!(op, AggregationOp::Max | AggregationOp::Min) {
            Some(DeviceMemory::<I>::create_uninit(
                self.stream.context(),
                count(self.destination_nodes, aggregated_width)?,
            )?)
        } else {
            None
        };
        // Scratch owners are declared outside the scope so every queued failure
        // and unwind completes before their storage or caller borrows are released.
        StreamBinding::Borrowed(self.stream.to_borrowed()).sync_scope(|| {
            // SAFETY: initialized feature views and exclusive destinations stay
            // borrowed until completion; graph indices were checked on upload.
            unsafe {
                self.forward_async(
                    op,
                    node.as_ref().map(Features::input),
                    edge.as_ref().map(Features::input),
                    concat.as_ref().map(Features::input),
                    output,
                    positions.as_mut().map(|v| v.output()),
                )
            }
        })?;
        Ok(Aggregation {
            graph: self,
            op,
            dims,
            positions,
            _element: PhantomData,
        })
    }

    /// Enqueues aggregation with caller-owned output and optional max/min positions.
    ///
    /// # Safety
    /// Retain the graph, feature ranges, output and positions through completion,
    /// submission failure and replay. Writes must stay exclusive. Order input
    /// producers before this call; every input must be valid when read. Only the required contiguous prefixes are
    /// produced; tails remain unchanged. Positions are required only for max/min.
    ///
    /// # Errors
    ///
    /// Returns an error for missing node and edge features, zero or unrepresentable feature widths, overflowing sizes, insufficient buffer capacity, a positions buffer supplied for the wrong operation, mismatched CUDA contexts, context binding or stream-dependency failure, or native aggregation failure.
    pub unsafe fn forward_async<T: FeatureScalar>(
        &self,
        op: AggregationOp,
        node: Option<Features<DeviceInputView<'_, T>>>,
        edge: Option<Features<DeviceInputView<'_, T>>>,
        concat: Option<Features<DeviceInputView<'_, T>>>,
        output: &mut (impl DeviceOutput<T> + ?Sized),
        mut positions: Option<DeviceOutputView<'_, I>>,
    ) -> Result<()> {
        let dims = [
            node.as_ref().map_or(0, |f| f.width),
            edge.as_ref().map_or(0, |f| f.width),
            concat.as_ref().map_or(0, |f| f.width),
        ];
        if dims[0] == 0 && dims[1] == 0 {
            return Err(Error::InvalidDataShape);
        }
        for (feature, rows) in [
            (node.as_ref(), self.source_nodes),
            (edge.as_ref(), self.indices.len()),
            (concat.as_ref(), self.destination_nodes),
        ] {
            if let Some(feature) = feature {
                if feature.width == 0 {
                    return Err(Error::InvalidDataShape);
                }
                require(&feature.values, rows, feature.width)?;
            }
        }
        let width = dims.iter().try_fold(0usize, |a, &b| {
            a.checked_add(b).ok_or(Error::InvalidDataShape)
        })?;
        require(output, self.destination_nodes, width)?;
        self.stream.context().bind()?;
        if matches!(op, AggregationOp::Max | AggregationOp::Min) != positions.is_some() {
            return Err(Error::InvalidDataShape);
        }
        if let Some(positions) = &positions {
            require(positions, self.destination_nodes, dims[0] + dims[1])?;
        }
        let mut regions = vec![output.region()];
        for feature in [node.as_ref(), edge.as_ref(), concat.as_ref()]
            .into_iter()
            .flatten()
        {
            regions.push(feature.values.region());
        }
        if let Some(positions) = &positions {
            regions.push(positions.region());
        }
        if regions
            .iter()
            .any(|region| region.context() != self.stream.context())
        {
            return Err(CudaError::MemoryContextMismatch.into());
        }
        self.stream.join_regions(&regions)?;
        let raw = self.raw();
        // SAFETY: `cudnnGnnAggSimpleForward` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cudnnGnnAggSimpleForward(
                self.stream.as_raw().cast(),
                &raw const raw,
                node.map_or(ptr::null(), |f| f.values.as_ptr().cast()),
                edge.map_or(ptr::null(), |f| f.values.as_ptr().cast()),
                concat.map_or(ptr::null(), |f| f.values.as_ptr().cast()),
                output.as_mut_ptr().cast(),
                positions
                    .as_mut()
                    .map_or(ptr::null_mut(), |v| v.as_mut_ptr().cast()),
                to_i32(dims[0], "node width")?,
                to_i32(dims[1], "edge width")?,
                to_i32(dims[2], "concat width")?,
                T::data_type().into(),
                op.into()
            ))?;
        }
        Ok(())
    }

    /// Enqueues gradients using caller-owned positions from a matching forward call.
    /// `dimensions` contains the node, edge and concat feature widths, in that order.
    ///
    /// # Safety
    /// Retain the graph, positions, gradient and destinations through completion,
    /// failure and replay. Positions and dimensions must match the forward call;
    /// max/min require its valid position prefix. Order all reads and keep writes
    /// exclusive. Initialize every gradient destination before use, usually to zero:
    /// this SDK accumulates into concat gradients as well as node and edge gradients.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid or unrepresentable feature widths, missing or unexpected gradient destinations or position storage, overflowing sizes, insufficient buffer capacity, mismatched CUDA contexts, context binding or stream-dependency failure, or native gradient aggregation failure.
    pub unsafe fn backward_async<T: FeatureScalar>(
        &self,
        op: AggregationOp,
        dimensions: [usize; 3],
        positions: Option<DeviceInputView<'_, I>>,
        gradient: &(impl DeviceInput<T> + ?Sized),
        mut node: Option<DeviceOutputView<'_, T>>,
        mut edge: Option<DeviceOutputView<'_, T>>,
        mut concat: Option<DeviceOutputView<'_, T>>,
    ) -> Result<()> {
        let [n, e, c] = dimensions;
        let width = n
            .checked_add(e)
            .and_then(|v| v.checked_add(c))
            .ok_or(Error::InvalidDataShape)?;
        if n == 0 && e == 0 {
            return Err(Error::InvalidDataShape);
        }
        if matches!(op, AggregationOp::Max | AggregationOp::Min) != positions.is_some() {
            return Err(Error::InvalidDataShape);
        }
        if let Some(positions) = &positions {
            require(positions, self.destination_nodes, n + e)?;
        }
        require(gradient, self.destination_nodes, width)?;
        for (buffer, rows, width) in [
            (node.as_ref(), self.source_nodes, n),
            (edge.as_ref(), self.indices.len(), e),
            (concat.as_ref(), self.destination_nodes, c),
        ] {
            if (width == 0) != buffer.is_none() {
                return Err(Error::InvalidDataShape);
            }
            if let Some(buffer) = buffer {
                require(buffer, rows, width)?;
            }
        }
        let mut regions = vec![gradient.region()];
        for buffer in [node.as_ref(), edge.as_ref(), concat.as_ref()]
            .into_iter()
            .flatten()
        {
            regions.push(buffer.region());
        }
        if let Some(positions) = &positions {
            regions.push(positions.region());
        }
        if regions
            .iter()
            .any(|region| region.context() != self.stream.context())
        {
            return Err(CudaError::MemoryContextMismatch.into());
        }
        self.stream.join_regions(&regions)?;
        self.stream.context().bind()?;
        let raw = self.raw();
        // SAFETY: `cudnnGnnAggSimpleBackward` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cudnnGnnAggSimpleBackward(
                self.stream.as_raw().cast(),
                &raw const raw,
                gradient.as_device_ptr().cast(),
                positions
                    .as_ref()
                    .map_or(ptr::null(), |v| v.as_ptr().cast()),
                node.as_mut()
                    .map_or(ptr::null_mut(), |v| v.as_mut_ptr().cast()),
                edge.as_mut()
                    .map_or(ptr::null_mut(), |v| v.as_mut_ptr().cast()),
                concat
                    .as_mut()
                    .map_or(ptr::null_mut(), |v| v.as_mut_ptr().cast()),
                to_i32(n, "node width")?,
                to_i32(e, "edge width")?,
                to_i32(c, "concat width")?,
                T::data_type().into(),
                op.into()
            ))?;
        }
        Ok(())
    }
}

/// A contiguous feature table with the given number of features per row.
#[derive(Debug, Clone, Copy)]
pub struct Features<V> {
    /// Contiguous row-major feature storage; the required row count depends on the operand role.
    pub values: V,
    /// Number of feature elements per row.
    pub width: usize,
}

impl<T: DeviceRepr> Features<DeviceView<'_, T>> {
    /// Borrows feature metadata for a queued input without changing its shape.
    #[must_use]
    pub fn input(&self) -> Features<DeviceInputView<'_, T>> {
        Features {
            values: DeviceInputView::from_buffer(&self.values),
            width: self.width,
        }
    }
}

/// Forward aggregation state, retaining the graph, dimensions and max/min positions.
#[derive(Debug)]
pub struct Aggregation<'a, I: GraphIndex, T: FeatureScalar> {
    graph: &'a CscGraph<I>,
    op: AggregationOp,
    dims: [usize; 3],
    positions: Option<DeviceMemory<MaybeUninit<I>>>,
    _element: PhantomData<T>,
}

impl<I: GraphIndex, T: FeatureScalar> Aggregation<'_, I, T> {
    /// Computes gradients for [`CscGraph::forward`]. For each destination node, reads
    /// the upstream gradient and scatters contributions back to source neighbors via
    /// atomic-add (non-deterministic across runs; order of atomics is not guaranteed).
    ///
    /// Requires SM 8.0 (Ampere) or later. Returns `CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH`
    /// on older devices.
    ///
    /// # Safety
    /// Keep the aggregation state, gradient and destination ranges alive through
    /// completion, submission failure and replay. All gradient destinations must
    /// be initialized before this call, usually to zero. This SDK accumulates into
    /// concat gradients as well as node and edge gradients. Order all prior accesses
    /// and keep writes exclusive.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid or unrepresentable feature widths, missing or unexpected gradient destinations or position storage, overflowing sizes, insufficient buffer capacity, mismatched CUDA contexts, context binding or stream-dependency failure, or native gradient aggregation failure.
    pub unsafe fn backward_async(
        &self,
        gradient: &(impl DeviceInput<T> + ?Sized),
        node: Option<DeviceOutputView<'_, T>>,
        edge: Option<DeviceOutputView<'_, T>>,
        concat: Option<DeviceOutputView<'_, T>>,
    ) -> Result<()> {
        // SAFETY: the stored graph, operation, widths and positions came from a
        // completed forward. The caller supplies the queued range/access contract.
        unsafe {
            self.graph.backward_async(
                self.op,
                self.dims,
                self.positions.as_ref().map(DeviceInputView::from_buffer),
                gradient,
                node,
                edge,
                concat,
            )
        }
    }
}

fn count(rows: usize, width: usize) -> Result<usize> {
    rows.checked_mul(width).ok_or(Error::InvalidDataShape)
}

fn require<T: DeviceRepr>(
    buffer: &(impl DeviceInput<T> + ?Sized),
    rows: usize,
    width: usize,
) -> Result<()> {
    let expected = count(rows, width)?;
    if buffer.len() < expected {
        return Err(Error::LengthMismatch {
            name: "features".into(),
            expected,
            actual: buffer.len(),
        });
    }
    Ok(())
}
