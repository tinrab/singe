//! Plan execution over dense and block-sparse tensors.
//!
//! Each dense operation has an `unsafe` `*_async` form that validates its operands, orders itself after their producers, and enqueues, and a safe form over borrowed initialized inputs that also waits for the stream.
//! A safe `*_scoped` form enqueues on a [`StreamScope`] and records its operands' accesses there, so the scope keeps them borrowed until it drains.
//! The `*_async` forms follow the crate's [raw execution contract](crate#raw-execution).

use std::ptr;

use singe_cuda::{
    data_type::{DataType, DataTypeLike},
    error::Error as CudaError,
    stream::{Stream, StreamScope},
    submission::{Access, AccessMode},
    types::DevicePtr,
    view::{
        DeviceOutput, DeviceOutputView, DeviceRange, DeviceRegion, DeviceRegionProvider,
        DeviceSlice,
    },
};

use crate::{
    error::{Error, Result},
    operation::OperationKind,
    plan::Plan,
    sys, try_ffi,
    utility::to_usize,
};

// The element type and length of one dense tensor operand, checked against the plan.
#[derive(Debug, Clone, Copy)]
struct Operand {
    name: &'static str,
    data_type: DataType,
    len: usize,
    element_size: usize,
}

fn operand<T: DataTypeLike>(name: &'static str, range: &(impl DeviceRange<T> + ?Sized)) -> Operand {
    Operand {
        name,
        data_type: T::data_type(),
        len: range.len(),
        element_size: size_of::<T>(),
    }
}

// Checks one block-sparse operand's element type and block lengths against the plan's spans, and returns the spans after its blocks.
fn check_blocks<'s, T: DataTypeLike>(
    name: &'static str,
    expected: DataType,
    lengths: impl ExactSizeIterator<Item = usize>,
    spans: &'s [u64],
) -> Result<&'s [u64]> {
    if T::data_type() != expected {
        return Err(Error::TensorMemoryDataTypeMismatch {
            descriptor: expected,
            memory: T::data_type(),
        });
    }

    let (spans, rest) = spans
        .split_at_checked(lengths.len())
        .ok_or_else(|| Error::out_of_range("block spans"))?;
    for (length, &span) in lengths.zip(spans) {
        let required = to_usize(span, "block span")?
            .checked_mul(size_of::<T>())
            .ok_or_else(|| Error::out_of_range("block span"))?;
        let actual = length.saturating_mul(size_of::<T>());
        if actual < required {
            return Err(Error::BufferTooSmall {
                name: name.into(),
                required,
                actual,
            });
        }
    }
    Ok(rest)
}

fn wait_for<R>(stream: &Stream, f: impl FnOnce() -> Result<R>) -> Result<R> {
    stream.complete(f)
}

impl Plan {
    /// Enqueues the three-input element-wise tensor operation encoded by this plan.
    ///
    /// The operation has the form:
    ///
    /// $$ D\_{\Pi^C(i\_0,i\_1,...,i\_n)} = \Phi\_{ABC}(\Phi\_{AB}(\alpha op\_A(A\_{\Pi^A(i\_0,i\_1,...,i\_n)}), \beta op\_B(B\_{\Pi^B(i\_0,i\_1,...,i\_n)})), \gamma op\_C(C\_{\Pi^C(i\_0,i\_1,...,i\_n)})) $$
    ///
    /// See [`OperationDescriptor::elementwise_trinary`](crate::operation::OperationDescriptor::elementwise_trinary) for details.
    ///
    /// # Safety
    ///
    /// Follow the crate's [raw execution contract](crate#raw-execution).
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` or an operand belongs to another CUDA context, the plan encodes another operation, scalar or tensor types differ from the plan, a tensor is shorter than the storage its descriptor reaches, or cuTENSOR rejects the operation.
    pub unsafe fn elementwise_trinary_async<TA, TB, TC, TD, TScalar>(
        &self,
        alpha: &TScalar,
        a: &(impl DeviceRange<TA> + ?Sized),
        beta: &TScalar,
        b: &(impl DeviceRange<TB> + ?Sized),
        gamma: &TScalar,
        c: &(impl DeviceRange<TC> + ?Sized),
        d: &mut impl DeviceOutput<TD>,
        stream: &Stream,
    ) -> Result<()>
    where
        TA: DataTypeLike,
        TB: DataTypeLike,
        TC: DataTypeLike,
        TD: DataTypeLike,
        TScalar: DataTypeLike,
    {
        self.prepare(
            "elementwise_trinary",
            TScalar::data_type(),
            &[
                operand("a", a),
                operand("b", b),
                operand("c", c),
                operand::<TD>("d", d),
            ],
            &[a.region(), b.region(), c.region(), d.region()],
            stream,
        )?;

        // SAFETY: the operands match the plan's types and spans in the stream's context; the caller keeps them alive through completion.
        unsafe {
            try_ffi!(sys::cutensorElementwiseTrinaryExecute(
                self.context().as_raw(),
                self.as_raw(),
                ptr::from_ref(alpha).cast(),
                a.as_device_ptr().cast(),
                ptr::from_ref(beta).cast(),
                b.as_device_ptr().cast(),
                ptr::from_ref(gamma).cast(),
                c.as_device_ptr().cast(),
                d.as_mut_ptr().cast(),
                stream.as_raw(),
            ))
        }
    }

    /// Performs the three-input element-wise tensor operation encoded by this plan and waits for it.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` is capturing, the checks of [`Self::elementwise_trinary_async`] fail, cuTENSOR rejects the operation, or the stream reports an error.
    pub fn elementwise_trinary<TA, TB, TC, TD, TScalar>(
        &self,
        alpha: &TScalar,
        a: &(impl DeviceSlice<TA> + ?Sized),
        beta: &TScalar,
        b: &(impl DeviceSlice<TB> + ?Sized),
        gamma: &TScalar,
        c: &(impl DeviceSlice<TC> + ?Sized),
        d: &mut impl DeviceOutput<TD>,
        stream: &Stream,
    ) -> Result<()>
    where
        TA: DataTypeLike,
        TB: DataTypeLike,
        TC: DataTypeLike,
        TD: DataTypeLike,
        TScalar: DataTypeLike,
    {
        wait_for(stream, || {
            // SAFETY: the inputs are initialized, every operand stays borrowed until `wait_for` has drained the stream, and the output borrow is exclusive.
            unsafe {
                self.elementwise_trinary_async::<TA, TB, TC, TD, TScalar>(
                    alpha, a, beta, b, gamma, c, d, stream,
                )
            }
        })
    }

    /// Enqueues the two-input element-wise tensor operation encoded by this plan.
    ///
    /// The operation has the form:
    ///
    /// $$ D\_{\Pi^C(i\_0,i\_1,...,i\_n)} = \Phi\_{AC}(\alpha op\_A(A\_{\Pi^A(i\_0,i\_1,...,i\_n)}), \gamma op\_C(C\_{\Pi^C(i\_0,i\_1,...,i\_n)})) $$
    ///
    /// See [`OperationDescriptor::elementwise_binary`](crate::operation::OperationDescriptor::elementwise_binary) for details.
    ///
    /// # Safety
    ///
    /// Follow the crate's [raw execution contract](crate#raw-execution).
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` or an operand belongs to another CUDA context, the plan encodes another operation, scalar or tensor types differ from the plan, a tensor is shorter than the storage its descriptor reaches, or cuTENSOR rejects the operation.
    pub unsafe fn elementwise_binary_async<TA, TC, TD, TScalar>(
        &self,
        alpha: &TScalar,
        a: &(impl DeviceRange<TA> + ?Sized),
        gamma: &TScalar,
        c: &(impl DeviceRange<TC> + ?Sized),
        d: &mut impl DeviceOutput<TD>,
        stream: &Stream,
    ) -> Result<()>
    where
        TA: DataTypeLike,
        TC: DataTypeLike,
        TD: DataTypeLike,
        TScalar: DataTypeLike,
    {
        self.prepare(
            "elementwise_binary",
            TScalar::data_type(),
            &[operand("a", a), operand("c", c), operand::<TD>("d", d)],
            &[a.region(), c.region(), d.region()],
            stream,
        )?;

        // SAFETY: the operands match the plan's types and spans in the stream's context; the caller keeps them alive through completion.
        unsafe {
            try_ffi!(sys::cutensorElementwiseBinaryExecute(
                self.context().as_raw(),
                self.as_raw(),
                ptr::from_ref(alpha).cast(),
                a.as_device_ptr().cast(),
                ptr::from_ref(gamma).cast(),
                c.as_device_ptr().cast(),
                d.as_mut_ptr().cast(),
                stream.as_raw(),
            ))
        }
    }

    /// Performs the two-input element-wise tensor operation encoded by this plan and waits for it.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` is capturing, the checks of [`Self::elementwise_binary_async`] fail, cuTENSOR rejects the operation, or the stream reports an error.
    pub fn elementwise_binary<TA, TC, TD, TScalar>(
        &self,
        alpha: &TScalar,
        a: &(impl DeviceSlice<TA> + ?Sized),
        gamma: &TScalar,
        c: &(impl DeviceSlice<TC> + ?Sized),
        d: &mut impl DeviceOutput<TD>,
        stream: &Stream,
    ) -> Result<()>
    where
        TA: DataTypeLike,
        TC: DataTypeLike,
        TD: DataTypeLike,
        TScalar: DataTypeLike,
    {
        wait_for(stream, || {
            // SAFETY: the inputs are initialized, every operand stays borrowed until `wait_for` has drained the stream, and the output borrow is exclusive.
            unsafe {
                self.elementwise_binary_async::<TA, TC, TD, TScalar>(alpha, a, gamma, c, d, stream)
            }
        })
    }

    /// Enqueues the tensor permutation encoded by this plan.
    ///
    /// The operation has the form:
    ///
    /// $$ B\_{\Pi^B(i\_0,i\_1,...,i\_n)} = \alpha op\_A(A\_{\Pi^A(i\_0,i\_1,...,i\_n)}) $$
    ///
    /// Consequently, this performs an out-of-place tensor permutation.
    ///
    /// Where:
    ///
    /// * `A` and `B` are multi-mode tensors of arbitrary data types.
    /// * $\Pi^A$ and $\Pi^B$ permute the modes of `A` and `B`, respectively.
    /// * $op\_A$ is the unary element-wise operator specified when creating the operation descriptor with [`OperationDescriptor::permutation`](crate::operation::OperationDescriptor::permutation).
    ///
    /// # Safety
    ///
    /// Follow the crate's [raw execution contract](crate#raw-execution).
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` or an operand belongs to another CUDA context, the plan encodes another operation, scalar or tensor types differ from the plan, a tensor is shorter than the storage its descriptor and padding reach, or cuTENSOR rejects the operation.
    pub unsafe fn permute_async<TA, TB, TScalar>(
        &self,
        alpha: &TScalar,
        a: &(impl DeviceRange<TA> + ?Sized),
        b: &mut impl DeviceOutput<TB>,
        stream: &Stream,
    ) -> Result<()>
    where
        TA: DataTypeLike,
        TB: DataTypeLike,
        TScalar: DataTypeLike,
    {
        self.prepare(
            "permute",
            TScalar::data_type(),
            &[operand("a", a), operand::<TB>("b", b)],
            &[a.region(), b.region()],
            stream,
        )?;

        // SAFETY: the operands match the plan's types and spans in the stream's context; the caller keeps them alive through completion.
        unsafe {
            try_ffi!(sys::cutensorPermute(
                self.context().as_raw(),
                self.as_raw(),
                ptr::from_ref(alpha).cast(),
                a.as_device_ptr().cast(),
                b.as_mut_ptr().cast(),
                stream.as_raw(),
            ))
        }
    }

    /// Performs the tensor permutation encoded by this plan and waits for it.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` is capturing, the checks of [`Self::permute_async`] fail, cuTENSOR rejects the operation, or the stream reports an error.
    pub fn permute<TA, TB, TScalar>(
        &self,
        alpha: &TScalar,
        a: &(impl DeviceSlice<TA> + ?Sized),
        b: &mut impl DeviceOutput<TB>,
        stream: &Stream,
    ) -> Result<()>
    where
        TA: DataTypeLike,
        TB: DataTypeLike,
        TScalar: DataTypeLike,
    {
        wait_for(stream, || {
            // SAFETY: the input is initialized, every operand stays borrowed until `wait_for` has drained the stream, and the output borrow is exclusive.
            unsafe { self.permute_async::<TA, TB, TScalar>(alpha, a, b, stream) }
        })
    }

    /// Enqueues the tensor contraction $D = alpha \cdot A \cdot B + beta \cdot C$.
    ///
    /// $$ \mathcal{D}\_{{modes}\_\mathcal{D}} \gets \alpha \cdot \mathcal{A}\_{{modes}\_\mathcal{A}} B\_{{modes}\_\mathcal{B}} + \beta \mathcal{C}\_{{modes}\_\mathcal{C}} $$
    ///
    /// The active CUDA device must match the CUDA device that was active when the plan was created.
    ///
    /// See [NVIDIA/CUDALibrarySamples](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuTENSOR/contraction.cu) for a concrete example.
    ///
    /// # Safety
    ///
    /// Follow the crate's [raw execution contract](crate#raw-execution).
    ///
    /// # Errors
    ///
    /// Returns an error if `stream`, an operand, or the workspace belongs to another CUDA context, the plan encodes another operation, scalar or tensor types differ from the plan, a tensor is shorter than the storage its descriptor reaches, the workspace is smaller than [`Plan::workspace_size`] or misaligned, or cuTENSOR rejects or fails the operation.
    pub unsafe fn contract_async<TA, TB, TC, TD, TScalar>(
        &self,
        alpha: &TScalar,
        a: &(impl DeviceRange<TA> + ?Sized),
        b: &(impl DeviceRange<TB> + ?Sized),
        beta: &TScalar,
        c: &(impl DeviceRange<TC> + ?Sized),
        d: &mut impl DeviceOutput<TD>,
        workspace: Option<DeviceOutputView<'_, u8>>,
        stream: &Stream,
    ) -> Result<()>
    where
        TA: DataTypeLike,
        TB: DataTypeLike,
        TC: DataTypeLike,
        TD: DataTypeLike,
        TScalar: DataTypeLike,
    {
        let mut workspace = workspace;
        {
            let mut regions = vec![a.region(), b.region(), c.region(), d.region()];
            regions.extend(workspace.as_ref().map(DeviceRegionProvider::region));
            self.prepare(
                "contract",
                TScalar::data_type(),
                &[
                    operand("a", a),
                    operand("b", b),
                    operand("c", c),
                    operand::<TD>("d", d),
                ],
                &regions,
                stream,
            )?;
        }

        let workspace_ptr = self.workspace_ptr(workspace.as_mut())?;

        // SAFETY: the operands match the plan's types and spans and the workspace its size, all in the stream's context; the caller keeps them alive through completion.
        unsafe {
            try_ffi!(sys::cutensorContract(
                self.context().as_raw(),
                self.as_raw(),
                ptr::from_ref(alpha).cast(),
                a.as_device_ptr().cast(),
                b.as_device_ptr().cast(),
                ptr::from_ref(beta).cast(),
                c.as_device_ptr().cast(),
                d.as_mut_ptr().cast(),
                workspace_ptr.cast(),
                self.workspace_size_u64()?,
                stream.as_raw(),
            ))
        }
    }

    /// Computes the tensor contraction $D = alpha \cdot A \cdot B + beta \cdot C$ and waits for it.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` is capturing, the checks of [`Self::contract_async`] fail, cuTENSOR rejects or fails the operation, or the stream reports an error.
    pub fn contract<TA, TB, TC, TD, TScalar>(
        &self,
        alpha: &TScalar,
        a: &(impl DeviceSlice<TA> + ?Sized),
        b: &(impl DeviceSlice<TB> + ?Sized),
        beta: &TScalar,
        c: &(impl DeviceSlice<TC> + ?Sized),
        d: &mut impl DeviceOutput<TD>,
        workspace: Option<DeviceOutputView<'_, u8>>,
        stream: &Stream,
    ) -> Result<()>
    where
        TA: DataTypeLike,
        TB: DataTypeLike,
        TC: DataTypeLike,
        TD: DataTypeLike,
        TScalar: DataTypeLike,
    {
        wait_for(stream, || {
            // SAFETY: the inputs are initialized, every operand and the workspace stay borrowed until `wait_for` has drained the stream, and the output borrows are exclusive.
            unsafe {
                self.contract_async::<TA, TB, TC, TD, TScalar>(
                    alpha, a, b, beta, c, d, workspace, stream,
                )
            }
        })
    }

    /// Enqueues the tensor reduction that is encoded by this plan.
    ///
    /// See [`OperationDescriptor::reduction`](crate::operation::OperationDescriptor::reduction) for details.
    ///
    /// # Safety
    ///
    /// Follow the crate's [raw execution contract](crate#raw-execution).
    ///
    /// # Errors
    ///
    /// Returns an error if `stream`, an operand, or the workspace belongs to another CUDA context, the plan encodes another operation, scalar or tensor types differ from the plan, a tensor is shorter than the storage its descriptor reaches, the workspace is smaller than [`Plan::workspace_size`] or misaligned, or cuTENSOR rejects the operation.
    pub unsafe fn reduce_async<TA, TC, TD, TScalar>(
        &self,
        alpha: &TScalar,
        a: &(impl DeviceRange<TA> + ?Sized),
        beta: &TScalar,
        c: &(impl DeviceRange<TC> + ?Sized),
        d: &mut impl DeviceOutput<TD>,
        workspace: Option<DeviceOutputView<'_, u8>>,
        stream: &Stream,
    ) -> Result<()>
    where
        TA: DataTypeLike,
        TC: DataTypeLike,
        TD: DataTypeLike,
        TScalar: DataTypeLike,
    {
        let mut workspace = workspace;
        {
            let mut regions = vec![a.region(), c.region(), d.region()];
            regions.extend(workspace.as_ref().map(DeviceRegionProvider::region));
            self.prepare(
                "reduce",
                TScalar::data_type(),
                &[operand("a", a), operand("c", c), operand::<TD>("d", d)],
                &regions,
                stream,
            )?;
        }

        let workspace_ptr = self.workspace_ptr(workspace.as_mut())?;

        // SAFETY: the operands match the plan's types and spans and the workspace its size, all in the stream's context; the caller keeps them alive through completion.
        unsafe {
            try_ffi!(sys::cutensorReduce(
                self.context().as_raw(),
                self.as_raw(),
                ptr::from_ref(alpha).cast(),
                a.as_device_ptr().cast(),
                ptr::from_ref(beta).cast(),
                c.as_device_ptr().cast(),
                d.as_mut_ptr().cast(),
                workspace_ptr.cast(),
                self.workspace_size_u64()?,
                stream.as_raw(),
            ))
        }
    }

    /// Performs the tensor reduction encoded by this plan and waits for it.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` is capturing, the checks of [`Self::reduce_async`] fail, cuTENSOR rejects the operation, or the stream reports an error.
    pub fn reduce<TA, TC, TD, TScalar>(
        &self,
        alpha: &TScalar,
        a: &(impl DeviceSlice<TA> + ?Sized),
        beta: &TScalar,
        c: &(impl DeviceSlice<TC> + ?Sized),
        d: &mut impl DeviceOutput<TD>,
        workspace: Option<DeviceOutputView<'_, u8>>,
        stream: &Stream,
    ) -> Result<()>
    where
        TA: DataTypeLike,
        TC: DataTypeLike,
        TD: DataTypeLike,
        TScalar: DataTypeLike,
    {
        wait_for(stream, || {
            // SAFETY: the inputs are initialized, every operand and the workspace stay borrowed until `wait_for` has drained the stream, and the output borrows are exclusive.
            unsafe {
                self.reduce_async::<TA, TC, TD, TScalar>(alpha, a, beta, c, d, workspace, stream)
            }
        })
    }

    /// Enqueues the tensor contraction $E = alpha \cdot A \cdot B \cdot C + beta \cdot D$.
    ///
    /// $$ \mathcal{E}\_{{modes}\_\mathcal{E}} \gets \alpha \cdot \mathcal{A}\_{{modes}\_\mathcal{A}} \mathcal{B}\_{{modes}\_\mathcal{B}} \mathcal{C}\_{{modes}\_\mathcal{C}} + \beta \mathcal{D}\_{{modes}\_\mathcal{D}} $$
    ///
    /// The active CUDA device must match the CUDA device that was active when the plan was created.
    ///
    /// See [NVIDIA/CUDALibrarySamples](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuTENSOR/contraction_trinary.cu) for a concrete example.
    ///
    /// # Safety
    ///
    /// Follow the crate's [raw execution contract](crate#raw-execution).
    ///
    /// # Errors
    ///
    /// Returns an error if `stream`, an operand, or the workspace belongs to another CUDA context, the plan encodes another operation, scalar or tensor types differ from the plan, a tensor is shorter than the storage its descriptor reaches, the workspace is smaller than [`Plan::workspace_size`] or misaligned, or cuTENSOR rejects or fails the operation.
    pub unsafe fn contract_trinary_async<TA, TB, TC, TD, TE, TScalar>(
        &self,
        alpha: &TScalar,
        a: &(impl DeviceRange<TA> + ?Sized),
        b: &(impl DeviceRange<TB> + ?Sized),
        c: &(impl DeviceRange<TC> + ?Sized),
        beta: &TScalar,
        d: &(impl DeviceRange<TD> + ?Sized),
        e: &mut impl DeviceOutput<TE>,
        workspace: Option<DeviceOutputView<'_, u8>>,
        stream: &Stream,
    ) -> Result<()>
    where
        TA: DataTypeLike,
        TB: DataTypeLike,
        TC: DataTypeLike,
        TD: DataTypeLike,
        TE: DataTypeLike,
        TScalar: DataTypeLike,
    {
        let mut workspace = workspace;
        {
            let mut regions = vec![a.region(), b.region(), c.region(), d.region(), e.region()];
            regions.extend(workspace.as_ref().map(DeviceRegionProvider::region));
            self.prepare(
                "contract_trinary",
                TScalar::data_type(),
                &[
                    operand("a", a),
                    operand("b", b),
                    operand("c", c),
                    operand("d", d),
                    operand::<TE>("e", e),
                ],
                &regions,
                stream,
            )?;
        }

        let workspace_ptr = self.workspace_ptr(workspace.as_mut())?;

        // SAFETY: the operands match the plan's types and spans and the workspace its size, all in the stream's context; the caller keeps them alive through completion.
        unsafe {
            try_ffi!(sys::cutensorContractTrinary(
                self.context().as_raw(),
                self.as_raw(),
                ptr::from_ref(alpha).cast(),
                a.as_device_ptr().cast(),
                b.as_device_ptr().cast(),
                c.as_device_ptr().cast(),
                ptr::from_ref(beta).cast(),
                d.as_device_ptr().cast(),
                e.as_mut_ptr().cast(),
                workspace_ptr.cast(),
                self.workspace_size_u64()?,
                stream.as_raw(),
            ))
        }
    }

    /// Computes the tensor contraction $E = alpha \cdot A \cdot B \cdot C + beta \cdot D$ and waits for it.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` is capturing, the checks of [`Self::contract_trinary_async`] fail, cuTENSOR rejects or fails the operation, or the stream reports an error.
    pub fn contract_trinary<TA, TB, TC, TD, TE, TScalar>(
        &self,
        alpha: &TScalar,
        a: &(impl DeviceSlice<TA> + ?Sized),
        b: &(impl DeviceSlice<TB> + ?Sized),
        c: &(impl DeviceSlice<TC> + ?Sized),
        beta: &TScalar,
        d: &(impl DeviceSlice<TD> + ?Sized),
        e: &mut impl DeviceOutput<TE>,
        workspace: Option<DeviceOutputView<'_, u8>>,
        stream: &Stream,
    ) -> Result<()>
    where
        TA: DataTypeLike,
        TB: DataTypeLike,
        TC: DataTypeLike,
        TD: DataTypeLike,
        TE: DataTypeLike,
        TScalar: DataTypeLike,
    {
        wait_for(stream, || {
            // SAFETY: the inputs are initialized, every operand and the workspace stay borrowed until `wait_for` has drained the stream, and the output borrows are exclusive.
            unsafe {
                self.contract_trinary_async::<TA, TB, TC, TD, TE, TScalar>(
                    alpha, a, b, c, beta, d, e, workspace, stream,
                )
            }
        })
    }

    /// Enqueues the block-sparse tensor contraction $D = alpha \cdot A \cdot B + beta \cdot C$.
    ///
    /// $$ \mathcal{D}\_{{modes}\_\mathcal{D}} \gets \alpha \cdot \mathcal{A}\_{{modes}\_\mathcal{A}} B\_{{modes}\_\mathcal{B}} + \beta \mathcal{C}\_{{modes}\_\mathcal{C}} $$
    ///
    /// The active CUDA device must match the CUDA device that was active when the plan was created.
    ///
    /// The array parameters `A`, `B`, `C`, and `D` are host arrays containing pointers to GPU-accessible memory.
    /// For example, `A` is a host array whose size equals the number of non-zero blocks in tensor $\mathcal{A}$.
    /// `A\[i\]` is a pointer to the GPU-accessible memory location of block number `i` of $\mathcal{A}$.
    /// The blocks are numbered in the same way as in the construction of $\mathcal{A}$’s block-sparse tensor descriptor.
    /// The same applies to the other array parameters `B`, `C`, and `D`.
    ///
    /// # Safety
    ///
    /// Follow the crate's [raw execution contract](crate#raw-execution).
    /// Every block pointer must cover its declared native block extent in the plan's context, and its producers must be ordered before `stream`.
    /// Retain the host pointer tables through completion and every captured replay.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` or the workspace belongs to another CUDA context, the plan encodes another operation, the scalar type differs from the plan, the pointer slices do not match the block counts encoded in the plan, the workspace is smaller than [`Plan::workspace_size`] or misaligned, or cuTENSOR rejects or fails the operation.
    pub unsafe fn block_sparse_contract_async<TScalar>(
        &self,
        alpha: &TScalar,
        a: &[DevicePtr],
        b: &[DevicePtr],
        beta: &TScalar,
        c: &[DevicePtr],
        d: &mut [DevicePtr],
        workspace: Option<DeviceOutputView<'_, u8>>,
        stream: &Stream,
    ) -> Result<()>
    where
        TScalar: DataTypeLike,
    {
        let mut workspace = workspace;
        {
            let regions = workspace
                .as_ref()
                .map(DeviceRegionProvider::region)
                .into_iter()
                .collect::<Vec<_>>();
            self.prepare(
                "block_sparse_contract",
                TScalar::data_type(),
                &[],
                &regions,
                stream,
            )?;
        }

        self.validate_block_counts(a, b, c, d)?;
        let workspace_ptr = self.workspace_ptr(workspace.as_mut())?;

        // SAFETY: `DevicePtr` is transparent over a device address, the block counts match the plan, and the caller retains the tables and block allocations through completion.
        unsafe {
            try_ffi!(sys::cutensorBlockSparseContract(
                self.context().as_raw(),
                self.as_raw(),
                ptr::from_ref(alpha).cast(),
                a.as_ptr().cast(),
                b.as_ptr().cast(),
                ptr::from_ref(beta).cast(),
                c.as_ptr().cast(),
                d.as_mut_ptr().cast(),
                workspace_ptr.cast(),
                self.workspace_size_u64()?,
                stream.as_raw(),
            ))
        }
    }

    /// Computes the block-sparse tensor contraction $D = alpha \cdot A \cdot B + beta \cdot C$ and waits for it.
    ///
    /// Each slice holds one range per nonzero block, numbered as in the block's [`BlockSparseTensorDescriptor`](crate::tensor::BlockSparseTensorDescriptor).
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` is capturing or belongs to another CUDA context, the plan encodes another operation, a block's element type differs from its descriptor, a block count differs from the plan, a block is shorter than the storage its descriptor reaches or belongs to another CUDA context, the checks of [`Self::block_sparse_contract_async`] fail, cuTENSOR rejects or fails the operation, or the stream reports an error.
    pub fn block_sparse_contract<TA, TB, TC, TD, TScalar>(
        &self,
        alpha: &TScalar,
        a: &[impl DeviceSlice<TA>],
        b: &[impl DeviceSlice<TB>],
        beta: &TScalar,
        c: &[impl DeviceSlice<TC>],
        d: &mut [impl DeviceOutput<TD>],
        workspace: Option<DeviceOutputView<'_, u8>>,
        stream: &Stream,
    ) -> Result<()>
    where
        TA: DataTypeLike,
        TB: DataTypeLike,
        TC: DataTypeLike,
        TD: DataTypeLike,
        TScalar: DataTypeLike,
    {
        self.context().ensure_stream(stream)?;
        let OperationKind::BlockSparseContraction {
            a: a_type,
            b: b_type,
            c: c_type,
            d: d_type,
            ..
        } = self.signature().kind
        else {
            return Err(Error::PlanOperationMismatch {
                expected: "block_sparse_contract".into(),
                actual: self.signature().kind.name().into(),
            });
        };

        // SAFETY: each address comes from a borrowed block range, and the tables are used only while those borrows last.
        let pointer = |address: *const ()| unsafe { DevicePtr::from_raw(address.cast_mut()) };
        let a_pointers = a
            .iter()
            .map(|block| pointer(block.as_device_ptr().cast()))
            .collect::<Vec<_>>();
        let b_pointers = b
            .iter()
            .map(|block| pointer(block.as_device_ptr().cast()))
            .collect::<Vec<_>>();
        let c_pointers = c
            .iter()
            .map(|block| pointer(block.as_device_ptr().cast()))
            .collect::<Vec<_>>();
        let mut d_pointers = d
            .iter_mut()
            .map(|block| pointer(block.as_mut_ptr().cast_const().cast()))
            .collect::<Vec<_>>();
        self.validate_block_counts(&a_pointers, &b_pointers, &c_pointers, &d_pointers)?;

        let spans = self.block_spans();
        let spans = check_blocks::<TA>("a", a_type, a.iter().map(DeviceRange::len), spans)?;
        let spans = check_blocks::<TB>("b", b_type, b.iter().map(DeviceRange::len), spans)?;
        let spans = check_blocks::<TC>("c", c_type, c.iter().map(DeviceRange::len), spans)?;
        check_blocks::<TD>("d", d_type, d.iter().map(DeviceRange::len), spans)?;

        let mut regions = Vec::with_capacity(a.len() + b.len() + c.len() + d.len());
        regions.extend(a.iter().map(|block| block.region()));
        regions.extend(b.iter().map(|block| block.region()));
        regions.extend(c.iter().map(|block| block.region()));
        regions.extend(d.iter().map(|block| block.region()));
        wait_for(stream, || {
            stream.join_regions(&regions)?;
            // SAFETY: every block covers the span its descriptor reaches in the stream's context, `stream` is ordered after the blocks' producers, the inputs are initialized, the output borrows are exclusive, and the blocks, pointer tables, and workspace stay borrowed until `wait_for` has drained the stream.
            unsafe {
                self.block_sparse_contract_async::<TScalar>(
                    alpha,
                    &a_pointers,
                    &b_pointers,
                    beta,
                    &c_pointers,
                    &mut d_pointers,
                    workspace,
                    stream,
                )
            }
        })
    }

    // Checks the plan's operation, scalar type, and operands, then orders `stream` after the operands' producers.
    fn prepare(
        &self,
        kind: &'static str,
        scalar_type: DataType,
        operands: &[Operand],
        regions: &[DeviceRegion<'_>],
        stream: &Stream,
    ) -> Result<()> {
        self.context().ensure_stream(stream)?;
        let signature = self.signature();
        if signature.kind.name() != kind {
            return Err(Error::PlanOperationMismatch {
                expected: kind.into(),
                actual: signature.kind.name().into(),
            });
        }

        if signature.scalar_type != scalar_type {
            return Err(Error::ScalarDataTypeMismatch {
                descriptor: signature.scalar_type,
                memory: scalar_type,
            });
        }

        let types = signature.kind.tensor_types();
        let spans = self.tensor_spans();
        if types.len() != operands.len() || spans.len() != operands.len() {
            return Err(Error::LengthMismatch {
                name: "tensor operands".into(),
                expected: types.len(),
                actual: operands.len(),
            });
        }

        for ((operand, &expected), &span) in operands.iter().zip(&types).zip(spans) {
            if operand.data_type != expected {
                return Err(Error::TensorMemoryDataTypeMismatch {
                    descriptor: expected,
                    memory: operand.data_type,
                });
            }

            let required = to_usize(span, "tensor span")?
                .checked_mul(operand.element_size)
                .ok_or_else(|| Error::out_of_range("tensor span"))?;
            let actual = operand.len.saturating_mul(operand.element_size);
            if actual < required {
                return Err(Error::BufferTooSmall {
                    name: operand.name.into(),
                    required,
                    actual,
                });
            }
        }

        stream.join_regions(regions)?;
        Ok(())
    }

    fn validate_block_counts(
        &self,
        a: &[DevicePtr],
        b: &[DevicePtr],
        c: &[DevicePtr],
        d: &[DevicePtr],
    ) -> Result<()> {
        let OperationKind::BlockSparseContraction {
            a_non_zero_blocks,
            b_non_zero_blocks,
            c_non_zero_blocks,
            d_non_zero_blocks,
            ..
        } = self.signature().kind
        else {
            return Err(Error::PlanOperationMismatch {
                expected: "block_sparse_contract".into(),
                actual: self.signature().kind.name().into(),
            });
        };

        for (name, expected, actual) in [
            ("a", a_non_zero_blocks, a.len()),
            ("b", b_non_zero_blocks, b.len()),
            ("c", c_non_zero_blocks, c.len()),
            ("d", d_non_zero_blocks, d.len()),
        ] {
            let expected = to_usize(expected, "nonzero block count")?;
            if expected != actual {
                return Err(Error::LengthMismatch {
                    name: name.into(),
                    expected,
                    actual,
                });
            }
        }
        Ok(())
    }

    fn workspace_ptr(&self, workspace: Option<&mut DeviceOutputView<'_, u8>>) -> Result<*mut u8> {
        let required = self.workspace_size();
        let Some(workspace) = workspace else {
            if required == 0 {
                return Ok(ptr::null_mut());
            }

            return Err(Error::BufferTooSmall {
                name: "workspace".into(),
                required,
                actual: 0,
            });
        };

        if workspace.len() < required {
            return Err(Error::BufferTooSmall {
                name: "workspace".into(),
                required,
                actual: workspace.len(),
            });
        }

        let pointer = workspace.as_mut_ptr();
        if required != 0 && !pointer.addr().is_multiple_of(256) {
            return Err(CudaError::Misaligned {
                name: "workspace".into(),
                alignment: 256,
            }
            .into());
        }
        Ok(pointer)
    }

    fn workspace_size_u64(&self) -> Result<u64> {
        u64::try_from(self.workspace_size()).map_err(|_| Error::out_of_range("workspace size"))
    }
}

// Scoped forms declare their operands' accesses, so tracked work on the scope orders around them.
impl Plan {
    /// Enqueues the three-input element-wise tensor operation on the scope's stream.
    ///
    /// # Errors
    ///
    /// Returns an error if access tracking fails, the checks of [`Self::elementwise_trinary_async`] fail, or cuTENSOR rejects the operation.
    pub fn elementwise_trinary_scoped<'env, TA, TB, TC, TD, TScalar>(
        &'env self,
        alpha: &'env TScalar,
        a: &'env (impl DeviceSlice<TA> + ?Sized),
        beta: &'env TScalar,
        b: &'env (impl DeviceSlice<TB> + ?Sized),
        gamma: &'env TScalar,
        c: &'env (impl DeviceSlice<TC> + ?Sized),
        d: &'env mut impl DeviceOutput<TD>,
        scope: &StreamScope<'_, 'env>,
    ) -> Result<()>
    where
        TA: DataTypeLike,
        TB: DataTypeLike,
        TC: DataTypeLike,
        TD: DataTypeLike,
        TScalar: DataTypeLike,
    {
        let accesses = scoped_accesses(&[a.region(), b.region(), c.region()], &[d.region()], None);
        // SAFETY: the operation touches only the declared operands on the scope's stream, the inputs are initialized, and the plan, scalars, and operands are `'env` borrows the scope keeps alive until it drains.
        unsafe {
            scope.enqueue(&accesses, || {
                Ok(self.elementwise_trinary_async::<TA, TB, TC, TD, TScalar>(
                    alpha,
                    a,
                    beta,
                    b,
                    gamma,
                    c,
                    d,
                    scope.stream(),
                ))
            })
        }?
    }

    /// Enqueues the two-input element-wise tensor operation on the scope's stream.
    ///
    /// # Errors
    ///
    /// Returns an error if access tracking fails, the checks of [`Self::elementwise_binary_async`] fail, or cuTENSOR rejects the operation.
    pub fn elementwise_binary_scoped<'env, TA, TC, TD, TScalar>(
        &'env self,
        alpha: &'env TScalar,
        a: &'env (impl DeviceSlice<TA> + ?Sized),
        gamma: &'env TScalar,
        c: &'env (impl DeviceSlice<TC> + ?Sized),
        d: &'env mut impl DeviceOutput<TD>,
        scope: &StreamScope<'_, 'env>,
    ) -> Result<()>
    where
        TA: DataTypeLike,
        TC: DataTypeLike,
        TD: DataTypeLike,
        TScalar: DataTypeLike,
    {
        let accesses = scoped_accesses(&[a.region(), c.region()], &[d.region()], None);
        // SAFETY: the operation touches only the declared operands on the scope's stream, the inputs are initialized, and the plan, scalars, and operands are `'env` borrows the scope keeps alive until it drains.
        unsafe {
            scope.enqueue(&accesses, || {
                Ok(self.elementwise_binary_async::<TA, TC, TD, TScalar>(
                    alpha,
                    a,
                    gamma,
                    c,
                    d,
                    scope.stream(),
                ))
            })
        }?
    }

    /// Enqueues the tensor permutation on the scope's stream.
    ///
    /// # Errors
    ///
    /// Returns an error if access tracking fails, the checks of [`Self::permute_async`] fail, or cuTENSOR rejects the operation.
    pub fn permute_scoped<'env, TA, TB, TScalar>(
        &'env self,
        alpha: &'env TScalar,
        a: &'env (impl DeviceSlice<TA> + ?Sized),
        b: &'env mut impl DeviceOutput<TB>,
        scope: &StreamScope<'_, 'env>,
    ) -> Result<()>
    where
        TA: DataTypeLike,
        TB: DataTypeLike,
        TScalar: DataTypeLike,
    {
        let accesses = scoped_accesses(&[a.region()], &[b.region()], None);
        // SAFETY: the operation touches only the declared operands on the scope's stream, the input is initialized, and the plan, scalar, and operands are `'env` borrows the scope keeps alive until it drains.
        unsafe {
            scope.enqueue(&accesses, || {
                Ok(self.permute_async::<TA, TB, TScalar>(alpha, a, b, scope.stream()))
            })
        }?
    }

    /// Enqueues the tensor contraction $D = alpha \cdot A \cdot B + beta \cdot C$ on the scope's stream.
    ///
    /// # Errors
    ///
    /// Returns an error if access tracking fails, the checks of [`Self::contract_async`] fail, or cuTENSOR rejects or fails the operation.
    pub fn contract_scoped<'env, TA, TB, TC, TD, TScalar>(
        &'env self,
        alpha: &'env TScalar,
        a: &'env (impl DeviceSlice<TA> + ?Sized),
        b: &'env (impl DeviceSlice<TB> + ?Sized),
        beta: &'env TScalar,
        c: &'env (impl DeviceSlice<TC> + ?Sized),
        d: &'env mut impl DeviceOutput<TD>,
        workspace: Option<DeviceOutputView<'env, u8>>,
        scope: &StreamScope<'_, 'env>,
    ) -> Result<()>
    where
        TA: DataTypeLike,
        TB: DataTypeLike,
        TC: DataTypeLike,
        TD: DataTypeLike,
        TScalar: DataTypeLike,
    {
        let accesses = scoped_accesses(
            &[a.region(), b.region(), c.region()],
            &[d.region()],
            workspace.as_ref(),
        );
        // SAFETY: the operation touches only the declared operands and workspace on the scope's stream, the inputs are initialized, and the plan, scalars, operands, and workspace are `'env` borrows the scope keeps alive until it drains.
        unsafe {
            scope.enqueue(&accesses, || {
                Ok(self.contract_async::<TA, TB, TC, TD, TScalar>(
                    alpha,
                    a,
                    b,
                    beta,
                    c,
                    d,
                    workspace,
                    scope.stream(),
                ))
            })
        }?
    }

    /// Enqueues the tensor reduction on the scope's stream.
    ///
    /// # Errors
    ///
    /// Returns an error if access tracking fails, the checks of [`Self::reduce_async`] fail, or cuTENSOR rejects the operation.
    pub fn reduce_scoped<'env, TA, TC, TD, TScalar>(
        &'env self,
        alpha: &'env TScalar,
        a: &'env (impl DeviceSlice<TA> + ?Sized),
        beta: &'env TScalar,
        c: &'env (impl DeviceSlice<TC> + ?Sized),
        d: &'env mut impl DeviceOutput<TD>,
        workspace: Option<DeviceOutputView<'env, u8>>,
        scope: &StreamScope<'_, 'env>,
    ) -> Result<()>
    where
        TA: DataTypeLike,
        TC: DataTypeLike,
        TD: DataTypeLike,
        TScalar: DataTypeLike,
    {
        let accesses =
            scoped_accesses(&[a.region(), c.region()], &[d.region()], workspace.as_ref());
        // SAFETY: the operation touches only the declared operands and workspace on the scope's stream, the inputs are initialized, and the plan, scalars, operands, and workspace are `'env` borrows the scope keeps alive until it drains.
        unsafe {
            scope.enqueue(&accesses, || {
                Ok(self.reduce_async::<TA, TC, TD, TScalar>(
                    alpha,
                    a,
                    beta,
                    c,
                    d,
                    workspace,
                    scope.stream(),
                ))
            })
        }?
    }

    /// Enqueues the tensor contraction $E = alpha \cdot A \cdot B \cdot C + beta \cdot D$ on the scope's stream.
    ///
    /// # Errors
    ///
    /// Returns an error if access tracking fails, the checks of [`Self::contract_trinary_async`] fail, or cuTENSOR rejects or fails the operation.
    pub fn contract_trinary_scoped<'env, TA, TB, TC, TD, TE, TScalar>(
        &'env self,
        alpha: &'env TScalar,
        a: &'env (impl DeviceSlice<TA> + ?Sized),
        b: &'env (impl DeviceSlice<TB> + ?Sized),
        c: &'env (impl DeviceSlice<TC> + ?Sized),
        beta: &'env TScalar,
        d: &'env (impl DeviceSlice<TD> + ?Sized),
        e: &'env mut impl DeviceOutput<TE>,
        workspace: Option<DeviceOutputView<'env, u8>>,
        scope: &StreamScope<'_, 'env>,
    ) -> Result<()>
    where
        TA: DataTypeLike,
        TB: DataTypeLike,
        TC: DataTypeLike,
        TD: DataTypeLike,
        TE: DataTypeLike,
        TScalar: DataTypeLike,
    {
        let accesses = scoped_accesses(
            &[a.region(), b.region(), c.region(), d.region()],
            &[e.region()],
            workspace.as_ref(),
        );
        // SAFETY: the operation touches only the declared operands and workspace on the scope's stream, the inputs are initialized, and the plan, scalars, operands, and workspace are `'env` borrows the scope keeps alive until it drains.
        unsafe {
            scope.enqueue(&accesses, || {
                Ok(self.contract_trinary_async::<TA, TB, TC, TD, TE, TScalar>(
                    alpha,
                    a,
                    b,
                    c,
                    beta,
                    d,
                    e,
                    workspace,
                    scope.stream(),
                ))
            })
        }?
    }
}

// Declares reads of the inputs and writes of the outputs and workspace for a scoped operation.
fn scoped_accesses(
    inputs: &[DeviceRegion<'_>],
    outputs: &[DeviceRegion<'_>],
    workspace: Option<&DeviceOutputView<'_, u8>>,
) -> Vec<Access> {
    let outputs = outputs
        .iter()
        .copied()
        .chain(workspace.map(DeviceRegionProvider::region));
    inputs
        .iter()
        .map(|&region| Access::new(region, AccessMode::Read))
        .chain(outputs.map(|region| Access::new(region, AccessMode::Write)))
        .collect()
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use singe_core::assert_close;
    use singe_cuda::{data_type::DataType, memory::DeviceMemory, stream::StreamOptions};

    use super::*;
    use crate::{
        error::Error,
        operation::{ComputeDescriptor, OperationDescriptor, TensorOperand},
        plan::{Plan, PlanPreference},
        tensor::TensorDescriptor,
        testing::setup_context,
        types::{Operator, WorkspacePreference},
    };

    fn packed_offset(indices: &[u64], extents: &[u64]) -> usize {
        let mut stride = 1_usize;
        let mut offset = 0_usize;

        for (&index, &extent) in indices.iter().zip(extents) {
            offset += usize::try_from(index).unwrap() * stride;
            stride *= usize::try_from(extent).unwrap();
        }

        offset
    }

    #[test]
    fn test_permute_matches_cpu_reference() -> Result<()> {
        const ALIGNMENT: u32 = 128;

        let test = setup_context()?;
        let context = test.context();
        let stream = context
            .cuda_context()
            .create_stream(StreamOptions::default())?;

        let extent_a = vec![2, 3];
        let extent_b = vec![3, 2];
        let mode_a = vec!['i'.into(), 'j'.into()];
        let mode_b = vec!['j'.into(), 'i'.into()];

        let host_a = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let device_a = DeviceMemory::from_slice(context.cuda_context(), &host_a)?;
        let mut device_b =
            DeviceMemory::<f32>::create_uninit(context.cuda_context(), host_a.len())?;

        let descriptor_a = TensorDescriptor::create(context, &extent_a, DataType::F32, ALIGNMENT)?;
        let descriptor_b = TensorDescriptor::create(context, &extent_b, DataType::F32, ALIGNMENT)?;

        let permutation = OperationDescriptor::permutation(
            context,
            TensorOperand::identity(&descriptor_a, &mode_a),
            TensorOperand::identity(&descriptor_b, &mode_b),
            ComputeDescriptor::f32(),
        )?;
        let preference = PlanPreference::create_default(context)?;
        let workspace_estimate = Plan::estimate_workspace_size(
            context,
            &permutation,
            &preference,
            WorkspacePreference::Default,
        )?;
        let plan = Plan::create(context, &permutation, &preference, workspace_estimate)?;

        let alpha = 2.5_f32;
        plan.permute(&alpha, &device_a, &mut device_b.output(), &stream)?;

        // SAFETY: the completed operation produced the full contiguous descriptor output.
        let device_b = unsafe { device_b.assume_init() };
        let result = device_b.copy_to_host_vec()?;
        let mut expected = vec![0.0_f32; host_a.len()];

        for i in 0..extent_a[0] {
            for j in 0..extent_a[1] {
                let a_offset = packed_offset(&[i, j], &extent_a);
                let b_offset = packed_offset(&[j, i], &extent_b);
                expected[b_offset] = alpha * host_a[a_offset];
            }
        }

        assert_close!(&result, &expected);

        Ok(())
    }

    #[test]
    fn test_reduce_matches_cpu_reference() -> Result<()> {
        const ALIGNMENT: u32 = 128;

        let test = setup_context()?;
        let context = test.context();
        let stream = context
            .cuda_context()
            .create_stream(StreamOptions::default())?;

        let extent_a = vec![2, 3];
        let extent_cd = vec![2];
        let mode_a = vec!['i'.into(), 'j'.into()];
        let mode_cd = vec!['i'.into()];

        let host_a = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let host_c = vec![0.5_f32, -1.0];

        let device_a = DeviceMemory::from_slice(context.cuda_context(), &host_a)?;
        let device_c = DeviceMemory::from_slice(context.cuda_context(), &host_c)?;
        let mut device_d = DeviceMemory::from_slice(context.cuda_context(), &host_c)?;

        let descriptor_a = TensorDescriptor::create(context, &extent_a, DataType::F32, ALIGNMENT)?;
        let descriptor_cd =
            TensorDescriptor::create(context, &extent_cd, DataType::F32, ALIGNMENT)?;

        let reduction = OperationDescriptor::reduction(
            context,
            TensorOperand::identity(&descriptor_a, &mode_a),
            TensorOperand::identity(&descriptor_cd, &mode_cd),
            TensorOperand::identity(&descriptor_cd, &mode_cd),
            Operator::Add,
            ComputeDescriptor::f32(),
        )?;
        let preference = PlanPreference::create_default(context)?;
        let workspace_estimate = Plan::estimate_workspace_size(
            context,
            &reduction,
            &preference,
            WorkspacePreference::Default,
        )?;
        let plan = Plan::create(context, &reduction, &preference, workspace_estimate)?;

        let alpha = 1.25_f32;
        let beta = -0.5_f32;

        let mut workspace = (plan.workspace_size() > 0)
            .then(|| {
                DeviceMemory::<u8>::create_uninit(context.cuda_context(), plan.workspace_size())
            })
            .transpose()?;
        plan.reduce(
            &alpha,
            &device_a,
            &beta,
            &device_c,
            &mut device_d,
            workspace.as_mut().map(|memory| memory.output()),
            &stream,
        )?;

        let result = device_d.copy_to_host_vec()?;
        let mut expected = vec![0.0_f32; usize::try_from(extent_cd[0]).unwrap()];

        for i in 0..usize::try_from(extent_cd[0]).unwrap() {
            let mut sum = 0.0_f32;
            for j in 0..extent_a[1] {
                let a_offset = packed_offset(&[i as u64, j], &extent_a);
                sum += host_a[a_offset];
            }
            expected[i] = beta.mul_add(host_c[i], alpha * sum);
        }

        assert_close!(&result, &expected);

        Ok(())
    }

    #[test]
    fn test_padding_attributes_validate_inputs() -> Result<()> {
        const ALIGNMENT: u32 = 128;

        let test = setup_context()?;
        let context = test.context();

        let extent_a = vec![2, 3];
        let extent_b = vec![3, 2];
        let mode_a = vec!['i'.into(), 'j'.into()];
        let mode_b = vec!['j'.into(), 'i'.into()];

        let descriptor_a = TensorDescriptor::create(context, &extent_a, DataType::F32, ALIGNMENT)?;
        let descriptor_b = TensorDescriptor::create(context, &extent_b, DataType::F32, ALIGNMENT)?;

        let mut permutation = OperationDescriptor::permutation(
            context,
            TensorOperand::identity(&descriptor_a, &mode_a),
            TensorOperand::identity(&descriptor_b, &mode_b),
            ComputeDescriptor::f32(),
        )?;

        let error = permutation.set_padding_left(&[1]).unwrap_err();
        assert!(matches!(
            error,
            Error::LengthMismatch {
                expected: 2,
                actual: 1,
                ..
            }
        ));

        let error = permutation.set_padding_right(&[1]).unwrap_err();
        assert!(matches!(
            error,
            Error::LengthMismatch {
                expected: 2,
                actual: 1,
                ..
            }
        ));

        let error = permutation.set_padding_left(&[1, u32::MAX]).unwrap_err();
        assert!(matches!(error, Error::OutOfRange { .. }));

        let error = permutation.set_padding_right(&[1, u32::MAX]).unwrap_err();
        assert!(matches!(error, Error::OutOfRange { .. }));

        let error = permutation.set_padding_value(7_i32).unwrap_err();
        assert!(matches!(
            error,
            Error::ScalarDataTypeMismatch {
                descriptor: DataType::F32,
                memory: DataType::I32,
            }
        ));

        Ok(())
    }

    #[test]
    fn test_operation_creation_validates_matching_output_descriptors_and_modes() -> Result<()> {
        const ALIGNMENT: u32 = 128;

        let test = setup_context()?;
        let context = test.context();

        let extent_a = vec![2, 3];
        let extent_cd = vec![2, 3];
        let extent_e = vec![3, 2];

        let mode_a = vec!['i'.into(), 'j'.into()];
        let mode_cd = vec!['i'.into(), 'j'.into()];
        let mode_e = vec!['j'.into(), 'i'.into()];

        let descriptor_a = TensorDescriptor::create(context, &extent_a, DataType::F32, ALIGNMENT)?;
        let descriptor_cd =
            TensorDescriptor::create(context, &extent_cd, DataType::F32, ALIGNMENT)?;
        let descriptor_e = TensorDescriptor::create(context, &extent_e, DataType::F32, ALIGNMENT)?;

        let error = OperationDescriptor::elementwise_binary(
            context,
            TensorOperand::identity(&descriptor_a, &mode_a),
            TensorOperand::identity(&descriptor_cd, &mode_cd),
            TensorOperand::identity(&descriptor_e, &mode_e),
            Operator::Add,
            ComputeDescriptor::f32(),
        )
        .unwrap_err();
        assert!(matches!(error, Error::DescriptorMismatch { .. }));

        let error = OperationDescriptor::reduction(
            context,
            TensorOperand::identity(&descriptor_a, &mode_a),
            TensorOperand::identity(&descriptor_cd, &mode_cd),
            TensorOperand::identity(&descriptor_cd, &mode_e),
            Operator::Add,
            ComputeDescriptor::f32(),
        )
        .unwrap_err();
        assert!(matches!(error, Error::ModeMismatch { .. }));

        let error = OperationDescriptor::contraction(
            context,
            TensorOperand::identity(&descriptor_a, &mode_a),
            TensorOperand::identity(&descriptor_a, &mode_a),
            TensorOperand::identity(&descriptor_cd, &mode_cd),
            TensorOperand::identity(&descriptor_e, &mode_e),
            ComputeDescriptor::f32(),
        )
        .unwrap_err();
        assert!(matches!(error, Error::DescriptorMismatch { .. }));

        let error = OperationDescriptor::contraction_trinary(
            context,
            TensorOperand::identity(&descriptor_a, &mode_a),
            TensorOperand::identity(&descriptor_a, &mode_a),
            TensorOperand::identity(&descriptor_a, &mode_a),
            TensorOperand::identity(&descriptor_cd, &mode_cd),
            TensorOperand::identity(&descriptor_cd, &mode_e),
            ComputeDescriptor::f32(),
        )
        .unwrap_err();
        assert!(matches!(error, Error::ModeMismatch { .. }));

        Ok(())
    }

    #[test]
    fn test_contraction_matches_cpu_reference() -> Result<()> {
        const ALIGNMENT: u32 = 128;

        let test = setup_context()?;
        let context = test.context();
        let stream = context
            .cuda_context()
            .create_stream(StreamOptions::default())?;

        let mode_c = vec!['m'.into(), 'u'.into(), 'n'.into(), 'v'.into()];
        let mode_a = vec!['m'.into(), 'h'.into(), 'k'.into(), 'n'.into()];
        let mode_b = vec!['u'.into(), 'k'.into(), 'v'.into(), 'h'.into()];

        let extent_m = 2;
        let extent_u = 2;
        let extent_n = 3;
        let extent_v = 2;
        let extent_h = 2;
        let extent_k = 2;

        let extent_c = vec![extent_m, extent_u, extent_n, extent_v];
        let extent_a = vec![extent_m, extent_h, extent_k, extent_n];
        let extent_b = vec![extent_u, extent_k, extent_v, extent_h];

        let elements_a = extent_a.iter().product::<u64>();
        let elements_b = extent_b.iter().product::<u64>();
        let elements_c = extent_c.iter().product::<u64>();

        let host_a = (0..elements_a)
            .map(|index| (index as f32 + 1.0) * 0.25)
            .collect::<Vec<_>>();
        let host_b = (0..elements_b)
            .map(|index| ((index % 7) as f32 - 3.0) * 0.5)
            .collect::<Vec<_>>();
        let host_c = (0..elements_c)
            .map(|index| (index as f32 - 4.0) * 0.125)
            .collect::<Vec<_>>();

        let device_a = DeviceMemory::from_slice(context.cuda_context(), &host_a)?;
        let device_b = DeviceMemory::from_slice(context.cuda_context(), &host_b)?;
        let device_c_input = DeviceMemory::from_slice(context.cuda_context(), &host_c)?;
        let mut device_c =
            DeviceMemory::<f32>::create_uninit(context.cuda_context(), host_c.len())?;

        let descriptor_a = TensorDescriptor::create(context, &extent_a, DataType::F32, ALIGNMENT)?;
        let descriptor_b = TensorDescriptor::create(context, &extent_b, DataType::F32, ALIGNMENT)?;
        let descriptor_c = TensorDescriptor::create(context, &extent_c, DataType::F32, ALIGNMENT)?;

        let contraction = OperationDescriptor::contraction(
            context,
            TensorOperand::identity(&descriptor_a, &mode_a),
            TensorOperand::identity(&descriptor_b, &mode_b),
            TensorOperand::identity(&descriptor_c, &mode_c),
            TensorOperand::identity(&descriptor_c, &mode_c),
            ComputeDescriptor::f32(),
        )?;

        assert_eq!(contraction.scalar_type()?, DataType::F32);

        let preference = PlanPreference::create_default(context)?;
        let workspace_estimate = Plan::estimate_workspace_size(
            context,
            &contraction,
            &preference,
            WorkspacePreference::Default,
        )?;
        let plan = Plan::create(context, &contraction, &preference, workspace_estimate)?;

        let mut workspace =
            DeviceMemory::<u8>::create_uninit(context.cuda_context(), plan.workspace_size())?;

        let alpha = 1.1_f32;
        let beta = 0.0_f32;

        plan.contract(
            &alpha,
            &device_a,
            &device_b,
            &beta,
            &device_c_input,
            &mut device_c.output(),
            Some(workspace.output()),
            &stream,
        )?;

        // SAFETY: the packed output descriptor covers the whole allocation and contraction has completed.
        let device_c = unsafe { device_c.assume_init() };
        let result = device_c.copy_to_host_vec()?;
        let mut expected = vec![0.0_f32; usize::try_from(elements_c).unwrap()];

        for m in 0..extent_m {
            for u in 0..extent_u {
                for n in 0..extent_n {
                    for v in 0..extent_v {
                        let mut sum = 0.0_f32;
                        for h in 0..extent_h {
                            for k in 0..extent_k {
                                let a_offset = packed_offset(&[m, h, k, n], &extent_a);
                                let b_offset = packed_offset(&[u, k, v, h], &extent_b);
                                sum = f32::mul_add(host_a[a_offset], host_b[b_offset], sum);
                            }
                        }

                        let c_offset = packed_offset(&[m, u, n, v], &extent_c);
                        expected[c_offset] = beta.mul_add(host_c[c_offset], alpha * sum);
                    }
                }
            }
        }

        assert_close!(&result, &expected);

        Ok(())
    }
}
