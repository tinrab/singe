use std::ptr;

use singe_cuda::{data_type::DataTypeLike, memory::DeviceMemory, stream::Stream, types::DevicePtr};

use crate::{
    error::{Error, Result},
    operation::OperationKind,
    plan::Plan,
    sys, try_ffi,
};

impl Plan {
    /// Performs the three-input element-wise tensor operation encoded by this plan.
    ///
    /// The operation has the form:
    ///
    /// $$ D\_{\Pi^C(i\_0,i\_1,...,i\_n)} = \Phi\_{ABC}(\Phi\_{AB}(\alpha op\_A(A\_{\Pi^A(i\_0,i\_1,...,i\_n)}), \beta op\_B(B\_{\Pi^B(i\_0,i\_1,...,i\_n)})), \gamma op\_C(C\_{\Pi^C(i\_0,i\_1,...,i\_n)})) $$
    ///
    /// See [`OperationDescriptor::elementwise_trinary`](crate::operation::OperationDescriptor::elementwise_trinary) for details.
    ///
    /// The call enqueues asynchronous CUDA work on `stream`. cuTENSOR treats this
    /// operation as thread-safe but not reentrant.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to a different CUDA context, scalar
    /// or tensor types do not match the plan, tensor dimensions or modes are
    /// invalid, a data type or operation combination is unsupported, or
    /// cuTENSOR rejects the operation.
    pub fn elementwise_trinary<TA, TB, TC, TD, TScalar>(
        &self,
        alpha: &TScalar,
        a: &DeviceMemory<TA>,
        beta: &TScalar,
        b: &DeviceMemory<TB>,
        gamma: &TScalar,
        c: &DeviceMemory<TC>,
        d: &mut DeviceMemory<TD>,
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
        validate_elementwise_trinary_types::<TA, TB, TC, TD, TScalar>(self)?;

        unsafe {
            try_ffi!(sys::cutensorElementwiseTrinaryExecute(
                self.context().as_raw(),
                self.as_raw(),
                ptr::from_ref(alpha) as _,
                a.as_ptr() as _,
                ptr::from_ref(beta) as _,
                b.as_ptr() as _,
                ptr::from_ref(gamma) as _,
                c.as_ptr() as _,
                d.as_mut_ptr() as _,
                stream.as_raw(),
            ))?;
        }

        Ok(())
    }

    /// Performs the two-input element-wise tensor operation encoded by this plan.
    ///
    /// The operation has the form:
    ///
    /// $$ D\_{\Pi^C(i\_0,i\_1,...,i\_n)} = \Phi\_{AC}(\alpha op\_A(A\_{\Pi^A(i\_0,i\_1,...,i\_n)}), \gamma op\_C(C\_{\Pi^C(i\_0,i\_1,...,i\_n)})) $$
    ///
    /// See [`OperationDescriptor::elementwise_binary`](crate::operation::OperationDescriptor::elementwise_binary) for details.
    ///
    /// The call enqueues asynchronous CUDA work on `stream`. cuTENSOR treats this
    /// operation as thread-safe but not reentrant.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to a different CUDA context, scalar
    /// or tensor types do not match the plan, tensor dimensions or modes are
    /// invalid, a data type or operation combination is unsupported, or
    /// cuTENSOR rejects the operation.
    pub fn elementwise_binary<TA, TC, TD, TScalar>(
        &self,
        alpha: &TScalar,
        a: &DeviceMemory<TA>,
        gamma: &TScalar,
        c: &DeviceMemory<TC>,
        d: &mut DeviceMemory<TD>,
        stream: &Stream,
    ) -> Result<()>
    where
        TA: DataTypeLike,
        TC: DataTypeLike,
        TD: DataTypeLike,
        TScalar: DataTypeLike,
    {
        self.context().ensure_stream(stream)?;
        validate_elementwise_binary_types::<TA, TC, TD, TScalar>(self)?;

        unsafe {
            try_ffi!(sys::cutensorElementwiseBinaryExecute(
                self.context().as_raw(),
                self.as_raw(),
                ptr::from_ref(alpha) as _,
                a.as_ptr() as _,
                ptr::from_ref(gamma) as _,
                c.as_ptr() as _,
                d.as_mut_ptr() as _,
                stream.as_raw(),
            ))?;
        }

        Ok(())
    }

    /// Performs the tensor permutation encoded by this plan.
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
    /// * $op\_A$ is the unary element-wise operator specified when creating the
    ///   operation descriptor with [`OperationDescriptor::permutation`](crate::operation::OperationDescriptor::permutation).
    ///
    /// The call enqueues asynchronous CUDA work on `stream`. cuTENSOR treats this
    /// operation as thread-safe but not reentrant.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to a different CUDA context, scalar
    /// or tensor types do not match the plan, tensor dimensions or modes are
    /// invalid, a data type or operation combination is unsupported, or
    /// cuTENSOR rejects the operation.
    pub fn permute<TA, TB, TScalar>(
        &self,
        alpha: &TScalar,
        a: &DeviceMemory<TA>,
        b: &mut DeviceMemory<TB>,
        stream: &Stream,
    ) -> Result<()>
    where
        TA: DataTypeLike,
        TB: DataTypeLike,
        TScalar: DataTypeLike,
    {
        self.context().ensure_stream(stream)?;
        validate_permute_types::<TA, TB, TScalar>(self)?;

        unsafe {
            try_ffi!(sys::cutensorPermute(
                self.context().as_raw(),
                self.as_raw(),
                ptr::from_ref(alpha) as _,
                a.as_ptr() as _,
                b.as_mut_ptr() as _,
                stream.as_raw(),
            ))?;
        }

        Ok(())
    }

    /// Computes the tensor contraction $D = alpha \cdot A \cdot B + beta \cdot C$.
    ///
    /// $$ \mathcal{D}\_{{modes}\_\mathcal{D}} \gets \alpha \cdot \mathcal{A}\_{{modes}\_\mathcal{A}} B\_{{modes}\_\mathcal{B}} + \beta \mathcal{C}\_{{modes}\_\mathcal{C}} $$
    ///
    /// The active CUDA device must match the CUDA device that was active when
    /// the plan was created.
    ///
    /// See [NVIDIA/CUDALibrarySamples](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuTENSOR/contraction.cu) for a concrete example.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to a different CUDA context, scalar
    /// or tensor types do not match the plan, the active device does not match
    /// the plan, the provided workspace is smaller than
    /// [`Plan::required_workspace_size`], the CUDA driver is insufficient, or
    /// cuTENSOR rejects or fails the operation.
    pub fn contract<TA, TB, TC, TD, TScalar>(
        &self,
        alpha: &TScalar,
        a: &DeviceMemory<TA>,
        b: &DeviceMemory<TB>,
        beta: &TScalar,
        c: &DeviceMemory<TC>,
        d: &mut DeviceMemory<TD>,
        workspace: Option<&mut DeviceMemory<u8>>,
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
        validate_contraction_types::<TA, TB, TC, TD, TScalar>(self)?;
        let (workspace_ptr, workspace_size) =
            workspace_ptr_and_size(workspace, self.required_workspace_size())?;

        unsafe {
            try_ffi!(sys::cutensorContract(
                self.context().as_raw(),
                self.as_raw(),
                ptr::from_ref(alpha) as _,
                a.as_ptr() as _,
                b.as_ptr() as _,
                ptr::from_ref(beta) as _,
                c.as_ptr() as _,
                d.as_mut_ptr() as _,
                workspace_ptr as _,
                workspace_size,
                stream.as_raw(),
            ))?;
        }

        Ok(())
    }

    /// Performs the tensor reduction that is encoded by this plan.
    ///
    /// See [`OperationDescriptor::reduction`](crate::operation::OperationDescriptor::reduction) for details.
    /// The call enqueues asynchronous CUDA work on `stream`. The provided
    /// `workspace`, when required by the plan, must remain valid until the
    /// queued work has completed.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to a different CUDA context, the
    /// scalar and tensor types do not match the plan, the provided workspace is
    /// smaller than [`Plan::required_workspace_size`], or cuTENSOR rejects the
    /// operation.
    pub fn reduce<TA, TC, TD, TScalar>(
        &self,
        alpha: &TScalar,
        a: &DeviceMemory<TA>,
        beta: &TScalar,
        c: &DeviceMemory<TC>,
        d: &mut DeviceMemory<TD>,
        workspace: Option<&mut DeviceMemory<u8>>,
        stream: &Stream,
    ) -> Result<()>
    where
        TA: DataTypeLike,
        TC: DataTypeLike,
        TD: DataTypeLike,
        TScalar: DataTypeLike,
    {
        self.context().ensure_stream(stream)?;
        validate_reduction_types::<TA, TC, TD, TScalar>(self)?;
        let (workspace_ptr, workspace_size) =
            workspace_ptr_and_size(workspace, self.required_workspace_size())?;

        unsafe {
            try_ffi!(sys::cutensorReduce(
                self.context().as_raw(),
                self.as_raw(),
                ptr::from_ref(alpha) as _,
                a.as_ptr() as _,
                ptr::from_ref(beta) as _,
                c.as_ptr() as _,
                d.as_mut_ptr() as _,
                workspace_ptr as _,
                workspace_size,
                stream.as_raw(),
            ))?;
        }

        Ok(())
    }

    /// Computes the tensor contraction $E = alpha \cdot A \cdot B \cdot C + beta \cdot D$.
    ///
    /// $$ \mathcal{E}\_{{modes}\_\mathcal{E}} \gets \alpha \cdot \mathcal{A}\_{{modes}\_\mathcal{A}} \mathcal{B}\_{{modes}\_\mathcal{B}} \mathcal{C}\_{{modes}\_\mathcal{C}} + \beta \mathcal{D}\_{{modes}\_\mathcal{D}} $$
    ///
    /// The active CUDA device must match the CUDA device that was active when
    /// the plan was created.
    ///
    /// See [NVIDIA/CUDALibrarySamples](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuTENSOR/contraction_trinary.cu) for a concrete example.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to a different CUDA context, scalar
    /// or tensor types do not match the plan, the active device does not match
    /// the plan, the provided workspace is smaller than
    /// [`Plan::required_workspace_size`], the CUDA driver is insufficient, or
    /// cuTENSOR rejects or fails the operation.
    pub fn contract_trinary<TA, TB, TC, TD, TE, TScalar>(
        &self,
        alpha: &TScalar,
        a: &DeviceMemory<TA>,
        b: &DeviceMemory<TB>,
        c: &DeviceMemory<TC>,
        beta: &TScalar,
        d: &DeviceMemory<TD>,
        e: &mut DeviceMemory<TE>,
        workspace: Option<&mut DeviceMemory<u8>>,
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
        self.context().ensure_stream(stream)?;
        validate_contraction_trinary_types::<TA, TB, TC, TD, TE, TScalar>(self)?;
        let (workspace_ptr, workspace_size) =
            workspace_ptr_and_size(workspace, self.required_workspace_size())?;

        unsafe {
            try_ffi!(sys::cutensorContractTrinary(
                self.context().as_raw(),
                self.as_raw(),
                ptr::from_ref(alpha) as _,
                a.as_ptr() as _,
                b.as_ptr() as _,
                c.as_ptr() as _,
                ptr::from_ref(beta) as _,
                d.as_ptr() as _,
                e.as_mut_ptr() as _,
                workspace_ptr as _,
                workspace_size,
                stream.as_raw(),
            ))?;
        }

        Ok(())
    }

    /// Computes the block-sparse tensor contraction $D = alpha \cdot A \cdot B + beta \cdot C$.
    ///
    /// $$ \mathcal{D}\_{{modes}\_\mathcal{D}} \gets \alpha \cdot \mathcal{A}\_{{modes}\_\mathcal{A}} B\_{{modes}\_\mathcal{B}} + \beta \mathcal{C}\_{{modes}\_\mathcal{C}} $$
    ///
    /// The active CUDA device must match the CUDA device that was active when
    /// the plan was created.
    ///
    /// The array parameters `A`, `B`, `C`, and `D` are host arrays containing pointers to GPU-accessible memory.
    /// For example, `A` is a host array whose size equals the number of non-zero blocks in tensor $\mathcal{A}$.
    /// `A\[i\]` is a pointer to the GPU-accessible memory location of block number `i` of $\mathcal{A}$.
    /// The blocks are numbered in the same way as in the construction of $\mathcal{A}$’s block-sparse tensor descriptor.
    /// The same applies to the other array parameters `B`, `C`, and `D`.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to a different CUDA context, the
    /// pointer slices do not match the block counts encoded in the plan, the
    /// active device does not match the plan, the provided workspace is smaller
    /// than [`Plan::required_workspace_size`], the CUDA driver is insufficient,
    /// or cuTENSOR rejects or fails the operation.
    pub fn block_sparse_contract<TScalar>(
        &self,
        alpha: &TScalar,
        a: &[DevicePtr],
        b: &[DevicePtr],
        beta: &TScalar,
        c: &[DevicePtr],
        d: &mut [DevicePtr],
        workspace: Option<&mut DeviceMemory<u8>>,
        stream: &Stream,
    ) -> Result<()>
    where
        TScalar: DataTypeLike,
    {
        self.context().ensure_stream(stream)?;
        validate_block_sparse_contract_arguments::<TScalar>(self, a, b, c, d)?;
        let (workspace_ptr, workspace_size) =
            workspace_ptr_and_size(workspace, self.required_workspace_size())?;

        let a = a
            .iter()
            .map(|ptr| ptr.as_ptr().cast_const())
            .collect::<Vec<_>>();
        let b = b
            .iter()
            .map(|ptr| ptr.as_ptr().cast_const())
            .collect::<Vec<_>>();
        let c = c
            .iter()
            .map(|ptr| ptr.as_ptr().cast_const())
            .collect::<Vec<_>>();
        let mut d = d.iter().map(|ptr| ptr.as_ptr()).collect::<Vec<_>>();

        unsafe {
            try_ffi!(sys::cutensorBlockSparseContract(
                self.context().as_raw(),
                self.as_raw(),
                ptr::from_ref(alpha) as _,
                a.as_ptr() as _,
                b.as_ptr() as _,
                ptr::from_ref(beta) as _,
                c.as_ptr() as _,
                d.as_mut_ptr() as _,
                workspace_ptr as _,
                workspace_size,
                stream.as_raw(),
            ))?;
        }

        Ok(())
    }
}

fn validate_elementwise_trinary_types<TA, TB, TC, TD, TScalar>(plan: &Plan) -> Result<()>
where
    TA: DataTypeLike,
    TB: DataTypeLike,
    TC: DataTypeLike,
    TD: DataTypeLike,
    TScalar: DataTypeLike,
{
    let signature = plan.signature();
    let OperationKind::ElementwiseTrinary { a, b, c, d } = signature.kind else {
        return Err(Error::PlanOperationMismatch {
            expected: "elementwise_trinary".into(),
            actual: signature.kind.name().into(),
        });
    };
    validate_scalar_type(signature.scalar_type, TScalar::data_type())?;
    validate_tensor_type(a, TA::data_type())?;
    validate_tensor_type(b, TB::data_type())?;
    validate_tensor_type(c, TC::data_type())?;
    validate_tensor_type(d, TD::data_type())?;
    Ok(())
}

fn validate_elementwise_binary_types<TA, TC, TD, TScalar>(plan: &Plan) -> Result<()>
where
    TA: DataTypeLike,
    TC: DataTypeLike,
    TD: DataTypeLike,
    TScalar: DataTypeLike,
{
    let signature = plan.signature();
    let OperationKind::ElementwiseBinary { a, c, d } = signature.kind else {
        return Err(Error::PlanOperationMismatch {
            expected: "elementwise_binary".into(),
            actual: signature.kind.name().into(),
        });
    };
    validate_scalar_type(signature.scalar_type, TScalar::data_type())?;
    validate_tensor_type(a, TA::data_type())?;
    validate_tensor_type(c, TC::data_type())?;
    validate_tensor_type(d, TD::data_type())?;
    Ok(())
}

fn validate_permute_types<TA, TB, TScalar>(plan: &Plan) -> Result<()>
where
    TA: DataTypeLike,
    TB: DataTypeLike,
    TScalar: DataTypeLike,
{
    let signature = plan.signature();
    let OperationKind::Permutation { a, b } = signature.kind else {
        return Err(Error::PlanOperationMismatch {
            expected: "permute".into(),
            actual: signature.kind.name().into(),
        });
    };
    validate_scalar_type(signature.scalar_type, TScalar::data_type())?;
    validate_tensor_type(a, TA::data_type())?;
    validate_tensor_type(b, TB::data_type())?;
    Ok(())
}

fn validate_contraction_types<TA, TB, TC, TD, TScalar>(plan: &Plan) -> Result<()>
where
    TA: DataTypeLike,
    TB: DataTypeLike,
    TC: DataTypeLike,
    TD: DataTypeLike,
    TScalar: DataTypeLike,
{
    let signature = plan.signature();
    let OperationKind::Contraction { a, b, c, d } = signature.kind else {
        return Err(Error::PlanOperationMismatch {
            expected: "contract".into(),
            actual: signature.kind.name().into(),
        });
    };
    validate_scalar_type(signature.scalar_type, TScalar::data_type())?;
    validate_tensor_type(a, TA::data_type())?;
    validate_tensor_type(b, TB::data_type())?;
    validate_tensor_type(c, TC::data_type())?;
    validate_tensor_type(d, TD::data_type())?;
    Ok(())
}

fn validate_reduction_types<TA, TC, TD, TScalar>(plan: &Plan) -> Result<()>
where
    TA: DataTypeLike,
    TC: DataTypeLike,
    TD: DataTypeLike,
    TScalar: DataTypeLike,
{
    let signature = plan.signature();
    let OperationKind::Reduction { a, c, d } = signature.kind else {
        return Err(Error::PlanOperationMismatch {
            expected: "reduce".into(),
            actual: signature.kind.name().into(),
        });
    };
    validate_scalar_type(signature.scalar_type, TScalar::data_type())?;
    validate_tensor_type(a, TA::data_type())?;
    validate_tensor_type(c, TC::data_type())?;
    validate_tensor_type(d, TD::data_type())?;
    Ok(())
}

fn validate_contraction_trinary_types<TA, TB, TC, TD, TE, TScalar>(plan: &Plan) -> Result<()>
where
    TA: DataTypeLike,
    TB: DataTypeLike,
    TC: DataTypeLike,
    TD: DataTypeLike,
    TE: DataTypeLike,
    TScalar: DataTypeLike,
{
    let signature = plan.signature();
    let OperationKind::ContractionTrinary { a, b, c, d, e } = signature.kind else {
        return Err(Error::PlanOperationMismatch {
            expected: "contract_trinary".into(),
            actual: signature.kind.name().into(),
        });
    };
    validate_scalar_type(signature.scalar_type, TScalar::data_type())?;
    validate_tensor_type(a, TA::data_type())?;
    validate_tensor_type(b, TB::data_type())?;
    validate_tensor_type(c, TC::data_type())?;
    validate_tensor_type(d, TD::data_type())?;
    validate_tensor_type(e, TE::data_type())?;
    Ok(())
}

fn validate_block_sparse_contract_arguments<TScalar>(
    plan: &Plan,
    a: &[DevicePtr],
    b: &[DevicePtr],
    c: &[DevicePtr],
    d: &[DevicePtr],
) -> Result<()>
where
    TScalar: DataTypeLike,
{
    let signature = plan.signature();
    let OperationKind::BlockSparseContraction {
        a_non_zero_blocks,
        b_non_zero_blocks,
        c_non_zero_blocks,
        d_non_zero_blocks,
    } = signature.kind
    else {
        return Err(Error::PlanOperationMismatch {
            expected: "block_sparse_contract".into(),
            actual: signature.kind.name().into(),
        });
    };
    validate_scalar_type(signature.scalar_type, TScalar::data_type())?;
    validate_pointer_count("a", a_non_zero_blocks as usize, a.len())?;
    validate_pointer_count("b", b_non_zero_blocks as usize, b.len())?;
    validate_pointer_count("c", c_non_zero_blocks as usize, c.len())?;
    validate_pointer_count("d", d_non_zero_blocks as usize, d.len())?;
    Ok(())
}

fn validate_tensor_type(
    expected: singe_cuda::data_type::DataType,
    actual: singe_cuda::data_type::DataType,
) -> Result<()> {
    if expected != actual {
        return Err(Error::TensorMemoryDataTypeMismatch {
            descriptor: expected,
            memory: actual,
        });
    }
    Ok(())
}

fn validate_scalar_type(
    expected: singe_cuda::data_type::DataType,
    actual: singe_cuda::data_type::DataType,
) -> Result<()> {
    if expected != actual {
        return Err(Error::ScalarDataTypeMismatch {
            descriptor: expected,
            memory: actual,
        });
    }
    Ok(())
}

fn validate_pointer_count(name: &str, expected: usize, actual: usize) -> Result<()> {
    if expected != actual {
        return Err(Error::PointerCountMismatch {
            name: name.into(),
            expected,
            actual,
        });
    }
    Ok(())
}

fn workspace_ptr_and_size(
    workspace: Option<&mut DeviceMemory<u8>>,
    required: u64,
) -> Result<(*mut (), u64)> {
    let Some(workspace) = workspace else {
        if required == 0 {
            return Ok((ptr::null_mut(), 0));
        }

        return Err(Error::InsufficientWorkspaceSize {
            required,
            actual: 0,
        });
    };

    let actual = workspace.byte_len() as u64;
    if actual < required {
        return Err(Error::InsufficientWorkspaceSize { required, actual });
    }
    if required != 0
        && !workspace.as_ptr().is_null()
        && !(workspace.as_ptr() as usize).is_multiple_of(256)
    {
        return Err(Error::WorkspaceMisaligned {
            required_alignment: 256,
            size: workspace.byte_len(),
        });
    }

    Ok((workspace.as_mut_ptr() as _, required))
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;
    use crate::{
        error::Error,
        operation::{ComputeDescriptor, OperationDescriptor, TensorOperand},
        plan::{Plan, PlanPreference},
        tensor::TensorDescriptor,
        testing::setup_context,
        types::{Operator, WorkspacePreference},
    };
    use singe_core::assert_close;
    use singe_cuda::data_type::DataType;

    fn packed_offset(indices: &[u64], extents: &[u64]) -> usize {
        let mut stride = 1_usize;
        let mut offset = 0_usize;

        for (&index, &extent) in indices.iter().zip(extents) {
            offset += index as usize * stride;
            stride *= extent as usize;
        }

        offset
    }

    #[test]
    fn test_permute_matches_cpu_reference() -> Result<()> {
        let context = setup_context()?;
        let stream = context.cuda_context().create_stream()?;

        let extent_a = vec![2, 3];
        let extent_b = vec![3, 2];
        let mode_a = vec!['i'.into(), 'j'.into()];
        let mode_b = vec!['j'.into(), 'i'.into()];

        let host_a = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let device_a = DeviceMemory::from_slice(&host_a)?;
        let mut device_b = DeviceMemory::<f32>::create(host_a.len())?;

        const ALIGNMENT: u32 = 128;

        let descriptor_a = TensorDescriptor::create(&context, &extent_a, DataType::F32, ALIGNMENT)?;
        let descriptor_b = TensorDescriptor::create(&context, &extent_b, DataType::F32, ALIGNMENT)?;

        let permutation = OperationDescriptor::permutation(
            &context,
            TensorOperand::identity(&descriptor_a, &mode_a),
            TensorOperand::identity(&descriptor_b, &mode_b),
            ComputeDescriptor::f32(),
        )?;
        let preference = PlanPreference::create_default(&context)?;
        let workspace_estimate = Plan::estimate_workspace_size(
            &context,
            &permutation,
            &preference,
            WorkspacePreference::Default,
        )?;
        let plan = Plan::create(&context, &permutation, &preference, workspace_estimate)?;

        let alpha = 2.5_f32;
        plan.permute(&alpha, &device_a, &mut device_b, &stream)?;

        stream.synchronize()?;

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
        let context = setup_context()?;
        let stream = context.cuda_context().create_stream()?;

        let extent_a = vec![2, 3];
        let extent_cd = vec![2];
        let mode_a = vec!['i'.into(), 'j'.into()];
        let mode_cd = vec!['i'.into()];

        let host_a = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let host_c = vec![0.5_f32, -1.0];

        let device_a = DeviceMemory::from_slice(&host_a)?;
        let device_c = DeviceMemory::from_slice(&host_c)?;
        let mut device_d = DeviceMemory::from_slice(&host_c)?;

        const ALIGNMENT: u32 = 128;

        let descriptor_a = TensorDescriptor::create(&context, &extent_a, DataType::F32, ALIGNMENT)?;
        let descriptor_cd =
            TensorDescriptor::create(&context, &extent_cd, DataType::F32, ALIGNMENT)?;

        let reduction = OperationDescriptor::reduction(
            &context,
            TensorOperand::identity(&descriptor_a, &mode_a),
            TensorOperand::identity(&descriptor_cd, &mode_cd),
            TensorOperand::identity(&descriptor_cd, &mode_cd),
            Operator::Add,
            ComputeDescriptor::f32(),
        )?;
        let preference = PlanPreference::create_default(&context)?;
        let workspace_estimate = Plan::estimate_workspace_size(
            &context,
            &reduction,
            &preference,
            WorkspacePreference::Default,
        )?;
        let plan = Plan::create(&context, &reduction, &preference, workspace_estimate)?;

        let alpha = 1.25_f32;
        let beta = -0.5_f32;

        if plan.required_workspace_size() == 0 {
            plan.reduce(
                &alpha,
                &device_a,
                &beta,
                &device_c,
                &mut device_d,
                None,
                &stream,
            )?;
        } else {
            let mut workspace =
                DeviceMemory::<u8>::create(plan.required_workspace_size() as usize)?;
            plan.reduce(
                &alpha,
                &device_a,
                &beta,
                &device_c,
                &mut device_d,
                Some(&mut workspace),
                &stream,
            )?;
        }

        stream.synchronize()?;

        let result = device_d.copy_to_host_vec()?;
        let mut expected = vec![0.0_f32; extent_cd[0] as usize];

        for i in 0..extent_cd[0] as usize {
            let mut sum = 0.0_f32;
            for j in 0..extent_a[1] {
                let a_offset = packed_offset(&[i as u64, j], &extent_a);
                sum += host_a[a_offset];
            }
            expected[i] = alpha * sum + beta * host_c[i];
        }

        assert_close!(&result, &expected);

        Ok(())
    }

    #[test]
    fn test_padding_attributes_validate_inputs() -> Result<()> {
        let context = setup_context()?;

        let extent_a = vec![2, 3];
        let extent_b = vec![3, 2];
        let mode_a = vec!['i'.into(), 'j'.into()];
        let mode_b = vec!['j'.into(), 'i'.into()];
        const ALIGNMENT: u32 = 128;

        let descriptor_a = TensorDescriptor::create(&context, &extent_a, DataType::F32, ALIGNMENT)?;
        let descriptor_b = TensorDescriptor::create(&context, &extent_b, DataType::F32, ALIGNMENT)?;

        let mut permutation = OperationDescriptor::permutation(
            &context,
            TensorOperand::identity(&descriptor_a, &mode_a),
            TensorOperand::identity(&descriptor_b, &mode_b),
            ComputeDescriptor::f32(),
        )?;

        let err = permutation.set_padding_left(&[1]).unwrap_err();
        assert!(matches!(
            err,
            Error::LengthMismatch {
                expected: 2,
                actual: 1,
                ..
            }
        ));

        let err = permutation.set_padding_right(&[1]).unwrap_err();
        assert!(matches!(
            err,
            Error::LengthMismatch {
                expected: 2,
                actual: 1,
                ..
            }
        ));

        let err = permutation.set_padding_left(&[1, u32::MAX]).unwrap_err();
        assert!(matches!(err, Error::OutOfRange { .. }));

        let err = permutation.set_padding_right(&[1, u32::MAX]).unwrap_err();
        assert!(matches!(err, Error::OutOfRange { .. }));

        let err = permutation.set_padding_value(7_i32).unwrap_err();
        assert!(matches!(
            err,
            Error::ScalarDataTypeMismatch {
                descriptor: DataType::F32,
                memory: DataType::I32,
            }
        ));

        Ok(())
    }

    #[test]
    fn test_operation_creation_validates_matching_output_descriptors_and_modes() -> Result<()> {
        let context = setup_context()?;

        const ALIGNMENT: u32 = 128;

        let extent_a = vec![2, 3];
        let extent_cd = vec![2, 3];
        let extent_e = vec![3, 2];

        let mode_a = vec!['i'.into(), 'j'.into()];
        let mode_cd = vec!['i'.into(), 'j'.into()];
        let mode_e = vec!['j'.into(), 'i'.into()];

        let descriptor_a = TensorDescriptor::create(&context, &extent_a, DataType::F32, ALIGNMENT)?;
        let descriptor_cd =
            TensorDescriptor::create(&context, &extent_cd, DataType::F32, ALIGNMENT)?;
        let descriptor_e = TensorDescriptor::create(&context, &extent_e, DataType::F32, ALIGNMENT)?;

        let err = OperationDescriptor::elementwise_binary(
            &context,
            TensorOperand::identity(&descriptor_a, &mode_a),
            TensorOperand::identity(&descriptor_cd, &mode_cd),
            TensorOperand::identity(&descriptor_e, &mode_e),
            Operator::Add,
            ComputeDescriptor::f32(),
        )
        .unwrap_err();
        assert!(matches!(err, Error::DescriptorMismatch { .. }));

        let err = OperationDescriptor::reduction(
            &context,
            TensorOperand::identity(&descriptor_a, &mode_a),
            TensorOperand::identity(&descriptor_cd, &mode_cd),
            TensorOperand::identity(&descriptor_cd, &mode_e),
            Operator::Add,
            ComputeDescriptor::f32(),
        )
        .unwrap_err();
        assert!(matches!(err, Error::ModeMismatch { .. }));

        let err = OperationDescriptor::contraction(
            &context,
            TensorOperand::identity(&descriptor_a, &mode_a),
            TensorOperand::identity(&descriptor_a, &mode_a),
            TensorOperand::identity(&descriptor_cd, &mode_cd),
            TensorOperand::identity(&descriptor_e, &mode_e),
            ComputeDescriptor::f32(),
        )
        .unwrap_err();
        assert!(matches!(err, Error::DescriptorMismatch { .. }));

        let err = OperationDescriptor::contraction_trinary(
            &context,
            TensorOperand::identity(&descriptor_a, &mode_a),
            TensorOperand::identity(&descriptor_a, &mode_a),
            TensorOperand::identity(&descriptor_a, &mode_a),
            TensorOperand::identity(&descriptor_cd, &mode_cd),
            TensorOperand::identity(&descriptor_cd, &mode_e),
            ComputeDescriptor::f32(),
        )
        .unwrap_err();
        assert!(matches!(err, Error::ModeMismatch { .. }));

        Ok(())
    }

    #[test]
    fn test_contraction_matches_cpu_reference() -> Result<()> {
        let context = setup_context()?;
        let stream = context.cuda_context().create_stream()?;

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

        let device_a = DeviceMemory::from_slice(&host_a)?;
        let device_b = DeviceMemory::from_slice(&host_b)?;
        let device_c_input = DeviceMemory::from_slice(&host_c)?;
        let mut device_c = DeviceMemory::from_slice(&host_c)?;

        const ALIGNMENT: u32 = 128;

        let descriptor_a = TensorDescriptor::create(&context, &extent_a, DataType::F32, ALIGNMENT)?;
        let descriptor_b = TensorDescriptor::create(&context, &extent_b, DataType::F32, ALIGNMENT)?;
        let descriptor_c = TensorDescriptor::create(&context, &extent_c, DataType::F32, ALIGNMENT)?;

        let contraction = OperationDescriptor::contraction(
            &context,
            TensorOperand::identity(&descriptor_a, &mode_a),
            TensorOperand::identity(&descriptor_b, &mode_b),
            TensorOperand::identity(&descriptor_c, &mode_c),
            TensorOperand::identity(&descriptor_c, &mode_c),
            ComputeDescriptor::f32(),
        )?;

        assert_eq!(contraction.scalar_type()?, DataType::F32);

        let preference = PlanPreference::create_default(&context)?;
        let workspace_estimate = Plan::estimate_workspace_size(
            &context,
            &contraction,
            &preference,
            WorkspacePreference::Default,
        )?;
        let plan = Plan::create(&context, &contraction, &preference, workspace_estimate)?;

        let mut workspace = DeviceMemory::<u8>::create(plan.required_workspace_size() as usize)?;

        let alpha = 1.1_f32;
        let beta = 0.0_f32;

        plan.contract(
            &alpha,
            &device_a,
            &device_b,
            &beta,
            &device_c_input,
            &mut device_c,
            Some(&mut workspace),
            &stream,
        )?;

        stream.synchronize()?;

        let result = device_c.copy_to_host_vec()?;
        let mut expected = vec![0.0_f32; elements_c as usize];

        for m in 0..extent_m {
            for u in 0..extent_u {
                for n in 0..extent_n {
                    for v in 0..extent_v {
                        let mut sum = 0.0_f32;
                        for h in 0..extent_h {
                            for k in 0..extent_k {
                                let a_offset = packed_offset(&[m, h, k, n], &extent_a);
                                let b_offset = packed_offset(&[u, k, v, h], &extent_b);
                                sum += host_a[a_offset] * host_b[b_offset];
                            }
                        }

                        let c_offset = packed_offset(&[m, u, n, v], &extent_c);
                        expected[c_offset] = alpha * sum + beta * host_c[c_offset];
                    }
                }
            }
        }

        assert_close!(&result, &expected);

        Ok(())
    }
}
