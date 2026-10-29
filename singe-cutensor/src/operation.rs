use std::{
    mem,
    mem::{ManuallyDrop, MaybeUninit},
    ptr,
};

use singe_cuda::{
    data_type::{DataType, DataTypeLike},
    types::{Complex32, Complex64, bf16, f4e2m1, f6e2m3, f6e3m2, f8e4m3, f8e5m2, f8ue8m0, f16},
};

use crate::{
    context::{Context, ContextRef, validate_same_context},
    error::{Error, Result},
    sys,
    tensor::{BlockSparseTensorDescriptor, TensorDescriptor},
    try_ffi,
    types::{Mode, OperationDescriptorAttribute, Operator},
    utility::{modes_to_i32_vec, to_i32},
};

/// A borrowed native compute-mode descriptor selecting arithmetic precision and implementation.
///
/// Support depends on the operation, tensor data types, and GPU architecture. This value does not own or destroy the native descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ComputeDescriptor(sys::cutensorComputeDescriptor_t);

impl ComputeDescriptor {
    // SAFETY: TODO: Not sure if these extern values are always valid.
    /// Selects half-precision floating-point computation using `CUTENSOR_COMPUTE_DESC_16F`.
    #[must_use]
    pub const fn f16() -> Self {
        Self(unsafe { sys::CUTENSOR_COMPUTE_DESC_16F })
    }

    /// Selects bfloat16 computation using `CUTENSOR_COMPUTE_DESC_16BF`.
    #[must_use]
    pub const fn bf16() -> Self {
        // SAFETY: reads an immutable compute-descriptor constant exported by cuTENSOR.
        Self(unsafe { sys::CUTENSOR_COMPUTE_DESC_16BF })
    }

    /// Selects TensorFloat-32 computation using `CUTENSOR_COMPUTE_DESC_TF32`.
    #[must_use]
    pub const fn tf32() -> Self {
        // SAFETY: reads an immutable compute-descriptor constant exported by cuTENSOR.
        Self(unsafe { sys::CUTENSOR_COMPUTE_DESC_TF32 })
    }

    /// Selects the native `3XTF32` compute mode using `CUTENSOR_COMPUTE_DESC_3XTF32`.
    #[must_use]
    pub const fn tf32x3() -> Self {
        // SAFETY: reads an immutable compute-descriptor constant exported by cuTENSOR.
        Self(unsafe { sys::CUTENSOR_COMPUTE_DESC_3XTF32 })
    }

    /// Selects single-precision floating-point computation using `CUTENSOR_COMPUTE_DESC_32F`.
    #[must_use]
    pub const fn f32() -> Self {
        // SAFETY: reads an immutable compute-descriptor constant exported by cuTENSOR.
        Self(unsafe { sys::CUTENSOR_COMPUTE_DESC_32F })
    }

    /// Selects double-precision floating-point computation using `CUTENSOR_COMPUTE_DESC_64F`.
    #[must_use]
    pub const fn f64() -> Self {
        // SAFETY: reads an immutable compute-descriptor constant exported by cuTENSOR.
        Self(unsafe { sys::CUTENSOR_COMPUTE_DESC_64F })
    }

    /// Selects the native `9X16BF` compute mode using `CUTENSOR_COMPUTE_DESC_9X16BF`.
    #[must_use]
    pub const fn bf16x9() -> Self {
        // SAFETY: reads an immutable compute-descriptor constant exported by cuTENSOR.
        Self(unsafe { sys::CUTENSOR_COMPUTE_DESC_9X16BF })
    }

    /// Selects the native `8XINT8` compute mode using `CUTENSOR_COMPUTE_DESC_8XINT8`.
    #[must_use]
    pub const fn i8x8() -> Self {
        // SAFETY: reads an immutable compute-descriptor constant exported by cuTENSOR.
        Self(unsafe { sys::CUTENSOR_COMPUTE_DESC_8XINT8 })
    }

    /// Selects the native `4X16F` compute mode using `CUTENSOR_COMPUTE_DESC_4X16F`.
    #[must_use]
    pub const fn f16x4() -> Self {
        // SAFETY: reads an immutable compute-descriptor constant exported by cuTENSOR.
        Self(unsafe { sys::CUTENSOR_COMPUTE_DESC_4X16F })
    }

    /// Returns the borrowed native compute descriptor.
    #[must_use]
    pub const fn as_raw(self) -> sys::cutensorComputeDescriptor_t {
        self.0
    }
}

impl From<sys::cutensorComputeDescriptor_t> for ComputeDescriptor {
    fn from(value: sys::cutensorComputeDescriptor_t) -> Self {
        Self(value)
    }
}

impl From<ComputeDescriptor> for sys::cutensorComputeDescriptor_t {
    fn from(value: ComputeDescriptor) -> Self {
        value.0
    }
}

/// Borrows a dense tensor descriptor, symbolic mode labels, and a unary operator.
#[derive(Debug, Clone, Copy)]
pub struct TensorOperand<'a> {
    desc: &'a TensorDescriptor,
    modes: &'a [Mode],
    op: Operator,
}

impl<'a> TensorOperand<'a> {
    /// Pairs a dense descriptor with mode labels and a unary element-wise operator.
    ///
    /// Mode counts and descriptor compatibility are checked when constructing an operation.
    #[must_use]
    pub const fn new(desc: &'a TensorDescriptor, modes: &'a [Mode], op: Operator) -> Self {
        Self { desc, modes, op }
    }

    /// Pairs a dense descriptor and mode labels with the identity operator.
    #[must_use]
    pub fn identity(desc: &'a TensorDescriptor, modes: &'a [Mode]) -> Self {
        Self::new(desc, modes, Operator::Identity)
    }
}

/// Borrows a block-sparse tensor descriptor, symbolic mode labels, and a unary operator.
#[derive(Debug, Clone, Copy)]
pub struct BlockSparseTensorOperand<'a> {
    desc: &'a BlockSparseTensorDescriptor,
    modes: &'a [Mode],
    op: Operator,
}

impl<'a> BlockSparseTensorOperand<'a> {
    /// Pairs a block-sparse descriptor with mode labels and a unary element-wise operator.
    ///
    /// Mode counts and descriptor compatibility are checked when constructing an operation.
    #[must_use]
    pub const fn new(
        desc: &'a BlockSparseTensorDescriptor,
        modes: &'a [Mode],
        op: Operator,
    ) -> Self {
        Self { desc, modes, op }
    }

    /// Pairs a block-sparse descriptor and mode labels with the identity operator.
    #[must_use]
    pub fn identity(desc: &'a BlockSparseTensorDescriptor, modes: &'a [Mode]) -> Self {
        Self::new(desc, modes, Operator::Identity)
    }
}

/// Owns a tensor operation descriptor, its context, signature, and optional padding scalar.
#[derive(Debug)]
pub struct OperationDescriptor {
    handle: sys::cutensorOperationDescriptor_t,
    context: ContextRef,
    output_rank: u32,
    output_data_type: DataType,
    signature: OperationSignature,
    padding_value: Option<OwnedPaddingValue>,
}

/// Transferred native ownership with its original metadata and retained library context.
///
/// Destroy `handle` before releasing this record, or reclaim it with [`OperationDescriptor::from_raw`].
/// Dropping this record does not destroy the native handle.
#[derive(Debug)]
pub struct RawOperationDescriptor {
    /// Transferred native operation handle; destroy it before releasing the retained context and metadata.
    pub handle: sys::cutensorOperationDescriptor_t,
    context: ContextRef,
    output_rank: u32,
    output_data_type: DataType,
    signature: OperationSignature,
    padding_value: Option<OwnedPaddingValue>,
}

/// Operation family and operand metadata used to validate execution against a plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OperationKind {
    /// Element-wise combination of three scaled inputs into output D.
    ElementwiseTrinary {
        /// Native sample data type of tensor A.
        a: DataType,
        /// Native sample data type of tensor B.
        b: DataType,
        /// Native sample data type of tensor C.
        c: DataType,
        /// Native sample data type of tensor D.
        d: DataType,
    },
    /// Element-wise combination of scaled inputs A and C into output D.
    ElementwiseBinary {
        /// Native sample data type of tensor A.
        a: DataType,
        /// Native sample data type of tensor C.
        c: DataType,
        /// Native sample data type of tensor D.
        d: DataType,
    },
    /// Permutation and unary transformation of A into output B.
    Permutation {
        /// Native sample data type of tensor A.
        a: DataType,
        /// Native sample data type of tensor B.
        b: DataType,
    },
    /// Contraction of A and B with an accumulated C term, producing D.
    Contraction {
        /// Native sample data type of tensor A.
        a: DataType,
        /// Native sample data type of tensor B.
        b: DataType,
        /// Native sample data type of tensor C.
        c: DataType,
        /// Native sample data type of tensor D.
        d: DataType,
    },
    /// Reduction of A with an accumulated C term, producing D.
    Reduction {
        /// Native sample data type of tensor A.
        a: DataType,
        /// Native sample data type of tensor C.
        c: DataType,
        /// Native sample data type of tensor D.
        d: DataType,
    },
    /// Contraction of three inputs A, B, and C with an accumulated D term, producing E.
    ContractionTrinary {
        /// Native sample data type of tensor A.
        a: DataType,
        /// Native sample data type of tensor B.
        b: DataType,
        /// Native sample data type of tensor C.
        c: DataType,
        /// Native sample data type of tensor D.
        d: DataType,
        /// Native sample data type of tensor E.
        e: DataType,
    },
    /// Block-sparse contraction with recorded nonzero block counts.
    BlockSparseContraction {
        /// Nonzero block count for tensor A.
        a_non_zero_blocks: u64,
        /// Nonzero block count for tensor B.
        b_non_zero_blocks: u64,
        /// Nonzero block count for tensor C.
        c_non_zero_blocks: u64,
        /// Nonzero block count for tensor D.
        d_non_zero_blocks: u64,
    },
}

impl OperationKind {
    /// Returns the stable wrapper name used to identify this operation family.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::ElementwiseTrinary { .. } => "elementwise_trinary",
            Self::ElementwiseBinary { .. } => "elementwise_binary",
            Self::Permutation { .. } => "permute",
            Self::Contraction { .. } => "contract",
            Self::Reduction { .. } => "reduce",
            Self::ContractionTrinary { .. } => "contract_trinary",
            Self::BlockSparseContraction { .. } => "block_sparse_contract",
        }
    }
}

/// Operation family, operand metadata, and scalar type expected by plan execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperationSignature {
    /// Operation family and operand metadata recorded when the descriptor was created.
    pub kind: OperationKind,
    /// Native representation required for the operation's scaling factors.
    pub scalar_type: DataType,
}

#[derive(Debug, Clone, Copy)]
enum OwnedPaddingValue {
    F32(f32),
    F64(f64),
    F16(f16),
    Bf16(bf16),
    F8E4M3(f8e4m3),
    F8E5M2(f8e5m2),
    F8UE8M0(f8ue8m0),
    F6E2M3(f6e2m3),
    F6E3M2(f6e3m2),
    F4E2M1(f4e2m1),
    I8(i8),
    U8(u8),
    I32(i32),
    U32(u32),
    Complex32(Complex32),
    Complex64(Complex64),
}

impl OwnedPaddingValue {
    fn from_value<T: DataTypeLike>(value: T) -> Result<Self> {
        match T::data_type() {
            // SAFETY: `T::data_type()` selected this variant, so `T` has the same layout as its payload.
            DataType::F32 => Ok(Self::F32(unsafe { mem::transmute_copy(&value) })),
            // SAFETY: `T::data_type()` selected this variant, so `T` has the same layout as its payload.
            DataType::F64 => Ok(Self::F64(unsafe { mem::transmute_copy(&value) })),
            // SAFETY: `T::data_type()` selected this variant, so `T` has the same layout as its payload.
            DataType::F16 => Ok(Self::F16(unsafe { mem::transmute_copy(&value) })),
            // SAFETY: `T::data_type()` selected this variant, so `T` has the same layout as its payload.
            DataType::Bf16 => Ok(Self::Bf16(unsafe { mem::transmute_copy(&value) })),
            // SAFETY: `T::data_type()` selected this variant, so `T` has the same layout as its payload.
            DataType::F8E4M3 => Ok(Self::F8E4M3(unsafe { mem::transmute_copy(&value) })),
            // SAFETY: `T::data_type()` selected this variant, so `T` has the same layout as its payload.
            DataType::F8E5M2 => Ok(Self::F8E5M2(unsafe { mem::transmute_copy(&value) })),
            // SAFETY: `T::data_type()` selected this variant, so `T` has the same layout as its payload.
            DataType::F8UE8M0 => Ok(Self::F8UE8M0(unsafe { mem::transmute_copy(&value) })),
            // SAFETY: `T::data_type()` selected this variant, so `T` has the same layout as its payload.
            DataType::F6E2M3 => Ok(Self::F6E2M3(unsafe { mem::transmute_copy(&value) })),
            // SAFETY: `T::data_type()` selected this variant, so `T` has the same layout as its payload.
            DataType::F6E3M2 => Ok(Self::F6E3M2(unsafe { mem::transmute_copy(&value) })),
            // SAFETY: `T::data_type()` selected this variant, so `T` has the same layout as its payload.
            DataType::F4E2M1 => Ok(Self::F4E2M1(unsafe { mem::transmute_copy(&value) })),
            // SAFETY: `T::data_type()` selected this variant, so `T` has the same layout as its payload.
            DataType::I8 => Ok(Self::I8(unsafe { mem::transmute_copy(&value) })),
            // SAFETY: `T::data_type()` selected this variant, so `T` has the same layout as its payload.
            DataType::U8 => Ok(Self::U8(unsafe { mem::transmute_copy(&value) })),
            // SAFETY: `T::data_type()` selected this variant, so `T` has the same layout as its payload.
            DataType::I32 => Ok(Self::I32(unsafe { mem::transmute_copy(&value) })),
            // SAFETY: `T::data_type()` selected this variant, so `T` has the same layout as its payload.
            DataType::U32 => Ok(Self::U32(unsafe { mem::transmute_copy(&value) })),
            // SAFETY: `T::data_type()` selected this variant, so `T` has the same layout as its payload.
            DataType::ComplexF32 => Ok(Self::Complex32(unsafe { mem::transmute_copy(&value) })),
            // SAFETY: `T::data_type()` selected this variant, so `T` has the same layout as its payload.
            DataType::ComplexF64 => Ok(Self::Complex64(unsafe { mem::transmute_copy(&value) })),
            data_type => Err(Error::UnsupportedScalarDataType {
                name: T::rust_type_name().into(),
                data_type,
            }),
        }
    }

    fn host_ptr(&self) -> *const () {
        match self {
            Self::F32(value) => ptr::from_ref(value).cast(),
            Self::F64(value) => ptr::from_ref(value).cast(),
            Self::F16(value) => ptr::from_ref(value).cast(),
            Self::Bf16(value) => ptr::from_ref(value).cast(),
            Self::F8E4M3(value) => ptr::from_ref(value).cast(),
            Self::F8E5M2(value) => ptr::from_ref(value).cast(),
            Self::F8UE8M0(value) => ptr::from_ref(value).cast(),
            Self::F6E2M3(value) => ptr::from_ref(value).cast(),
            Self::F6E3M2(value) => ptr::from_ref(value).cast(),
            Self::F4E2M1(value) => ptr::from_ref(value).cast(),
            Self::I8(value) => ptr::from_ref(value).cast(),
            Self::U8(value) => ptr::from_ref(value).cast(),
            Self::I32(value) => ptr::from_ref(value).cast(),
            Self::U32(value) => ptr::from_ref(value).cast(),
            Self::Complex32(value) => ptr::from_ref(value).cast(),
            Self::Complex64(value) => ptr::from_ref(value).cast(),
        }
    }

    fn size_bytes(&self) -> usize {
        match self {
            Self::F32(_) => size_of::<f32>(),
            Self::F64(_) => size_of::<f64>(),
            Self::F16(_) => size_of::<f16>(),
            Self::Bf16(_) => size_of::<bf16>(),
            Self::F8E4M3(_) => size_of::<f8e4m3>(),
            Self::F8E5M2(_) => size_of::<f8e5m2>(),
            Self::F8UE8M0(_) => size_of::<f8ue8m0>(),
            Self::F6E2M3(_) => size_of::<f6e2m3>(),
            Self::F6E3M2(_) => size_of::<f6e3m2>(),
            Self::F4E2M1(_) => size_of::<f4e2m1>(),
            Self::I8(_) => size_of::<i8>(),
            Self::U8(_) => size_of::<u8>(),
            Self::I32(_) => size_of::<i32>(),
            Self::U32(_) => size_of::<u32>(),
            Self::Complex32(_) => size_of::<Complex32>(),
            Self::Complex64(_) => size_of::<Complex64>(),
        }
    }
}

impl OperationDescriptor {
    /// Creates an operation descriptor that encodes an element-wise trinary operation.
    ///
    /// The trinary operation has the following general form:
    ///
    /// $$ D\_{\Pi^C(i\_0,i\_1,...,i\_n)} = \Phi\_{ABC}(\Phi\_{AB}(\alpha op\_A(A\_{\Pi^A(i\_0,i\_1,...,i\_n)}), \beta op\_B(B\_{\Pi^B(i\_0,i\_1,...,i\_n)})), \gamma op\_C(C\_{\Pi^C(i\_0,i\_1,...,i\_n)})) $$
    ///
    /// Where:
    ///
    /// * A,B,C,D are multi-mode tensors (of arbitrary data types).
    /// * $\Pi^A, \Pi^B, \Pi^C$ are permutation operators that permute the modes of A, B, and C respectively.
    /// * $op\_{A},op\_{B},op\_{C}$ are unary element-wise operators, such as IDENTITY and CONJUGATE.
    /// * $\Phi\_{ABC}, \Phi\_{AB}$ are binary element-wise operators, such as ADD, MUL, MAX, and MIN.
    ///
    /// Broadcasting can be achieved by omitting that mode from the respective tensor.
    ///
    /// Modes may appear in any order.
    /// The only **restrictions** are:
    ///
    /// * modes that appear in A or B *must* also appear in the output tensor; a mode that only appears in the input would be contracted and such an operation would be covered by either [`Plan::contract`](crate::plan::Plan::contract) or [`Plan::reduce`](crate::plan::Plan::reduce).
    /// * each mode may appear in each tensor at most once.
    ///
    /// Input tensors may be read even if the value of the corresponding scalar is zero.
    ///
    /// Examples:
    ///
    /// * $D\_{a,b,c,d} = A\_{b,d,a,c}$
    /// * $D\_{a,b,c,d} = 2.2 \cdot A\_{b,d,a,c} + 1.3 \cdot B\_{c,b,d,a}$
    /// * $D\_{a,b,c,d} = 2.2 \cdot A\_{b,d,a,c} + 1.3 \cdot B\_{c,b,d,a} + C\_{a,b,c,d}$
    /// * $D\_{a,b,c,d} = min((2.2 \cdot A\_{b,d,a,c} + 1.3 \cdot B\_{c,b,d,a}), C\_{a,b,c,d})$
    ///
    /// Call [`Plan::elementwise_trinary`](crate::plan::Plan::elementwise_trinary) to perform the actual operation.
    ///
    /// The returned descriptor frees the cuTENSOR descriptor when dropped.
    ///
    /// Supported data-type combinations are:
    ///
    /// | A type | B type | C type | compute descriptor |
    /// | --- | --- | --- | --- |
    /// | [`DataType::F16`] | [`DataType::F16`] | [`DataType::F16`] | [`ComputeDescriptor::f16`] |
    /// | [`DataType::F16`] | [`DataType::F16`] | [`DataType::F16`] | [`ComputeDescriptor::f32`] |
    /// | [`DataType::Bf16`] | [`DataType::Bf16`] | [`DataType::Bf16`] | [`ComputeDescriptor::bf16`] |
    /// | [`DataType::Bf16`] | [`DataType::Bf16`] | [`DataType::Bf16`] | [`ComputeDescriptor::f32`] |
    /// | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | [`ComputeDescriptor::f32`] |
    /// | [`DataType::F64`] | [`DataType::F64`] | [`DataType::F64`] | [`ComputeDescriptor::f64`] |
    /// | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`ComputeDescriptor::f32`] |
    /// | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | [`ComputeDescriptor::f64`] |
    /// | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F16`] | [`ComputeDescriptor::f32`] |
    /// | [`DataType::F64`] | [`DataType::F64`] | [`DataType::F32`] | [`ComputeDescriptor::f64`] |
    /// | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | [`DataType::ComplexF32`] | [`ComputeDescriptor::f64`] |
    ///
    /// This may call asynchronous CUDA functions. cuTENSOR treats descriptor
    /// creation as thread-safe but not reentrant.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, tensor modes do not
    /// match descriptor ranks, the target architecture is unsupported or not
    /// ready, cuTENSOR rejects the operand descriptors, or cuTENSOR returns a
    /// null operation descriptor.
    pub fn elementwise_trinary(
        context: &Context,
        lhs: TensorOperand<'_>,
        rhs: TensorOperand<'_>,
        aux: TensorOperand<'_>,
        out: TensorOperand<'_>,
        op_ab: Operator,
        op_abc: Operator,
        compute_descriptor: ComputeDescriptor,
    ) -> Result<Self> {
        let context_ref = context.as_context_ref();
        validate_tensor_operand_context(&context_ref, lhs, "lhs")?;
        validate_tensor_operand_context(&context_ref, rhs, "rhs")?;
        validate_tensor_operand_context(&context_ref, aux, "aux")?;
        validate_tensor_operand_context(&context_ref, out, "out")?;
        validate_modes(lhs.desc.rank(), lhs.modes)?;
        validate_modes(rhs.desc.rank(), rhs.modes)?;
        validate_modes(aux.desc.rank(), aux.modes)?;
        validate_modes(out.desc.rank(), out.modes)?;

        let mode_a = modes_to_i32_vec(lhs.modes);
        let mode_b = modes_to_i32_vec(rhs.modes);
        let mode_c = modes_to_i32_vec(aux.modes);
        let mode_d = modes_to_i32_vec(out.modes);

        let mut raw = ptr::null_mut();
        context.bind()?;
        // SAFETY: `cutensorCreateElementwiseTrinary` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cutensorCreateElementwiseTrinary(
                context.as_raw(),
                &raw mut raw,
                lhs.desc.as_raw(),
                mode_a.as_ptr(),
                lhs.op.into(),
                rhs.desc.as_raw(),
                mode_b.as_ptr(),
                rhs.op.into(),
                aux.desc.as_raw(),
                mode_c.as_ptr(),
                aux.op.into(),
                out.desc.as_raw(),
                mode_d.as_ptr(),
                op_ab.into(),
                op_abc.into(),
                compute_descriptor.into(),
            ))?;
        }

        Self::from_native(
            raw,
            context_ref,
            out.desc.rank(),
            out.desc.data_type(),
            OperationKind::ElementwiseTrinary {
                a: lhs.desc.data_type(),
                b: rhs.desc.data_type(),
                c: aux.desc.data_type(),
                d: out.desc.data_type(),
            },
        )
    }

    /// Creates an operation descriptor for an element-wise binary operation.
    ///
    /// The binary operation has the following general form:
    ///
    /// $$ D\_{\Pi^C(i\_0,i\_1,...,i\_n)} = \Phi\_{AC}(\alpha op\_A(A\_{\Pi^A(i\_0,i\_1,...,i\_n)}), \gamma op\_C(C\_{\Pi^C(i\_0,i\_1,...,i\_n)})) $$
    ///
    /// Call [`Plan::elementwise_binary`](crate::plan::Plan::elementwise_binary) to perform the actual operation.
    ///
    /// Supported data-type combinations are:
    ///
    /// | A type | C type | compute descriptor |
    /// | --- | --- | --- |
    /// | [`DataType::F16`] | [`DataType::F16`] | [`ComputeDescriptor::f16`] |
    /// | [`DataType::F16`] | [`DataType::F16`] | [`ComputeDescriptor::f32`] |
    /// | [`DataType::Bf16`] | [`DataType::Bf16`] | [`ComputeDescriptor::bf16`] |
    /// | [`DataType::Bf16`] | [`DataType::Bf16`] | [`ComputeDescriptor::f32`] |
    /// | [`DataType::F32`] | [`DataType::F32`] | [`ComputeDescriptor::f32`] |
    /// | [`DataType::F64`] | [`DataType::F64`] | [`ComputeDescriptor::f64`] |
    /// | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`ComputeDescriptor::f32`] |
    /// | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | [`ComputeDescriptor::f64`] |
    /// | [`DataType::F32`] | [`DataType::F16`] | [`ComputeDescriptor::f32`] |
    /// | [`DataType::F64`] | [`DataType::F32`] | [`ComputeDescriptor::f64`] |
    /// | [`DataType::ComplexF64`] | [`DataType::ComplexF32`] | [`ComputeDescriptor::f64`] |
    ///
    /// This may call asynchronous CUDA functions. cuTENSOR treats descriptor
    /// creation as thread-safe but not reentrant.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, tensor modes do not
    /// match descriptor ranks, input and output descriptors or mode sets are
    /// incompatible, a data type or operator combination is unsupported,
    /// cuTENSOR rejects the operands, or cuTENSOR returns a null operation
    /// descriptor.
    pub fn elementwise_binary(
        context: &Context,
        a: TensorOperand<'_>,
        c: TensorOperand<'_>,
        d: TensorOperand<'_>,
        op_ac: Operator,
        compute_descriptor: ComputeDescriptor,
    ) -> Result<Self> {
        let context_ref = context.as_context_ref();
        validate_tensor_operand_context(&context_ref, a, "a")?;
        validate_tensor_operand_context(&context_ref, c, "c")?;
        validate_tensor_operand_context(&context_ref, d, "d")?;
        validate_modes(a.desc.rank(), a.modes)?;
        validate_modes(c.desc.rank(), c.modes)?;
        validate_modes(d.desc.rank(), d.modes)?;
        validate_tensor_descriptors_match(c.desc, d.desc, "elementwise binary output")?;
        validate_mode_names_match(c.modes, d.modes, "elementwise binary output")?;

        let mode_a = modes_to_i32_vec(a.modes);
        let mode_c = modes_to_i32_vec(c.modes);
        let mode_d = modes_to_i32_vec(d.modes);

        let mut raw = ptr::null_mut();
        context.bind()?;
        // SAFETY: `cutensorCreateElementwiseBinary` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cutensorCreateElementwiseBinary(
                context.as_raw(),
                &raw mut raw,
                a.desc.as_raw(),
                mode_a.as_ptr(),
                a.op.into(),
                c.desc.as_raw(),
                mode_c.as_ptr(),
                c.op.into(),
                d.desc.as_raw(),
                mode_d.as_ptr(),
                op_ac.into(),
                compute_descriptor.into(),
            ))?;
        }

        Self::from_native(
            raw,
            context_ref,
            d.desc.rank(),
            d.desc.data_type(),
            OperationKind::ElementwiseBinary {
                a: a.desc.data_type(),
                c: c.desc.data_type(),
                d: d.desc.data_type(),
            },
        )
    }

    /// Creates an operation descriptor for a tensor permutation.
    ///
    /// The tensor permutation has the following general form:
    ///
    /// $$ B\_{\Pi^B(i\_0,i\_1,...,i\_n)} = \alpha op\_A(A\_{\Pi^A(i\_0,i\_1,...,i\_n)}) $$
    ///
    /// Consequently, tensor permutation is an out-of-place operation and a specialization of [`OperationDescriptor::elementwise_binary`].
    ///
    /// Where:
    ///
    /// * `A` and `B` are multi-mode tensors of arbitrary data types.
    /// * $\Pi^A$ and $\Pi^B$ permute the modes of `A` and `B`, respectively.
    /// * $op\_A$ is a unary element-wise operator such as identity, square, or conjugate.
    ///
    /// Broadcasting can be achieved by omitting that mode from the respective tensor.
    ///
    /// Modes may appear in any order.
    /// The only **restrictions** are:
    ///
    /// * modes that appear in A *must* also appear in the output tensor.
    /// * each mode may appear in each tensor at most once.
    ///
    /// Supported data-type combinations are:
    ///
    /// | A type | B type | compute descriptor |
    /// | --- | --- | --- |
    /// | [`DataType::F16`] | [`DataType::F16`] | [`ComputeDescriptor::f16`] |
    /// | [`DataType::F16`] | [`DataType::F16`] | [`ComputeDescriptor::f32`] |
    /// | [`DataType::F16`] | [`DataType::F32`] | [`ComputeDescriptor::f32`] |
    /// | [`DataType::F32`] | [`DataType::F16`] | [`ComputeDescriptor::f32`] |
    /// | [`DataType::Bf16`] | [`DataType::Bf16`] | [`ComputeDescriptor::bf16`] |
    /// | [`DataType::Bf16`] | [`DataType::Bf16`] | [`ComputeDescriptor::f32`] |
    /// | [`DataType::F32`] | [`DataType::F32`] | [`ComputeDescriptor::f32`] |
    /// | [`DataType::F64`] | [`DataType::F64`] | [`ComputeDescriptor::f64`] |
    /// | [`DataType::F32`] | [`DataType::F64`] | [`ComputeDescriptor::f64`] |
    /// | [`DataType::F64`] | [`DataType::F32`] | [`ComputeDescriptor::f64`] |
    /// | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`ComputeDescriptor::f32`] |
    /// | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | [`ComputeDescriptor::f64`] |
    /// | [`DataType::ComplexF32`] | [`DataType::ComplexF64`] | [`ComputeDescriptor::f64`] |
    /// | [`DataType::ComplexF64`] | [`DataType::ComplexF32`] | [`ComputeDescriptor::f64`] |
    ///
    /// This may call asynchronous CUDA functions. cuTENSOR treats descriptor
    /// creation as thread-safe but not reentrant.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, tensor modes do not
    /// match descriptor ranks, a requested output mode is invalid, a data type
    /// or operator combination is unsupported, cuTENSOR rejects the operands,
    /// or cuTENSOR returns a null operation descriptor.
    pub fn permutation(
        context: &Context,
        a: TensorOperand<'_>,
        b: TensorOperand<'_>,
        compute_descriptor: ComputeDescriptor,
    ) -> Result<Self> {
        let context_ref = context.as_context_ref();
        validate_tensor_operand_context(&context_ref, a, "a")?;
        validate_tensor_operand_context(&context_ref, b, "b")?;
        validate_modes(a.desc.rank(), a.modes)?;
        validate_modes(b.desc.rank(), b.modes)?;

        let mode_a = modes_to_i32_vec(a.modes);
        let mode_b = modes_to_i32_vec(b.modes);

        let mut raw = ptr::null_mut();
        context.bind()?;
        // SAFETY: `cutensorCreatePermutation` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cutensorCreatePermutation(
                context.as_raw(),
                &raw mut raw,
                a.desc.as_raw(),
                mode_a.as_ptr(),
                a.op.into(),
                b.desc.as_raw(),
                mode_b.as_ptr(),
                compute_descriptor.into(),
            ))?;
        }

        Self::from_native(
            raw,
            context_ref,
            b.desc.rank(),
            b.desc.data_type(),
            OperationKind::Permutation {
                a: a.desc.data_type(),
                b: b.desc.data_type(),
            },
        )
    }

    /// Creates an operation descriptor for a tensor contraction of the form $D = \alpha \mathcal{A} \mathcal{B} + \beta \mathcal{C}$.
    ///
    /// The descriptor represents a tensor contraction of the form:
    ///
    /// $$ \mathcal{D}\_{{modes}\_\mathcal{D}} \gets \alpha op\_\mathcal{A}(\mathcal{A}\_{{modes}\_\mathcal{A}}) op\_\mathcal{B}(B\_{{modes}\_\mathcal{B}}) + \beta op\_\mathcal{C}(\mathcal{C}\_{{modes}\_\mathcal{C}}). $$
    /// Use [`Plan::create`](crate::plan::Plan::create) to select a kernel, then call [`Plan::contract`](crate::plan::Plan::contract) to execute the contraction.
    ///
    /// The returned descriptor frees the cuTENSOR operation descriptor when dropped.
    ///
    /// Supported data-type combinations are:
    ///
    /// | A type | B type | C type | compute descriptor | scalar type | Tensor Core |
    /// | --- | --- | --- | --- | --- | --- |
    /// | [`DataType::F16`] | [`DataType::F16`] | [`DataType::F16`] | [`ComputeDescriptor::f32`] | [`DataType::F32`] | Volta+ |
    /// | [`DataType::Bf16`] | [`DataType::Bf16`] | [`DataType::Bf16`] | [`ComputeDescriptor::f32`] | [`DataType::F32`] | Ampere+ |
    /// | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | [`ComputeDescriptor::f32`] | [`DataType::F32`] | No |
    /// | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | [`ComputeDescriptor::tf32`] | [`DataType::F32`] | Ampere+ |
    /// | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | [`ComputeDescriptor::tf32x3`] | [`DataType::F32`] | Ampere+ |
    /// | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | [`ComputeDescriptor::bf16`] | [`DataType::F32`] | Ampere+ |
    /// | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | [`ComputeDescriptor::f16`] | [`DataType::F32`] | Volta+ |
    /// | [`DataType::F64`] | [`DataType::F64`] | [`DataType::F64`] | [`ComputeDescriptor::f64`] | [`DataType::F64`] | Ampere+ |
    /// | [`DataType::F64`] | [`DataType::F64`] | [`DataType::F64`] | [`ComputeDescriptor::f32`] | [`DataType::F64`] | No |
    /// | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`ComputeDescriptor::f32`] | [`DataType::ComplexF32`] | No |
    /// | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`ComputeDescriptor::tf32`] | [`DataType::ComplexF32`] | Ampere+ |
    /// | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`ComputeDescriptor::tf32x3`] | [`DataType::ComplexF32`] | Ampere+ |
    /// | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | [`ComputeDescriptor::f64`] | [`DataType::ComplexF64`] | Ampere+ |
    /// | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | [`ComputeDescriptor::f32`] | [`DataType::ComplexF64`] | No |
    /// | [`DataType::F64`] | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | [`ComputeDescriptor::f64`] | [`DataType::ComplexF64`] | No |
    /// | [`DataType::ComplexF64`] | [`DataType::F64`] | [`DataType::ComplexF64`] | [`ComputeDescriptor::f64`] | [`DataType::ComplexF64`] | No |
    /// | [`DataType::F64`] | [`DataType::F64`] | [`DataType::F64`] | [`ComputeDescriptor::i8x8`] | [`DataType::F64`] | Hopper+ |
    /// | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | [`ComputeDescriptor::i8x8`] | [`DataType::ComplexF64`] | Hopper+ |
    /// | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | [`ComputeDescriptor::bf16x9`] | [`DataType::F32`] | Hopper+ |
    /// | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`ComputeDescriptor::bf16x9`] | [`DataType::ComplexF32`] | Hopper+ |
    /// | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | [`ComputeDescriptor::f16x4`] | [`DataType::F32`] | Blackwell+ |
    /// | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`ComputeDescriptor::f16x4`] | [`DataType::ComplexF32`] | Blackwell+ |
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, tensor modes do not
    /// match descriptor ranks, input and output descriptors or mode sets are
    /// incompatible, a data type or operator combination is unsupported,
    /// cuTENSOR rejects the operands, or cuTENSOR returns a null operation
    /// descriptor.
    pub fn contraction(
        context: &Context,
        a: TensorOperand<'_>,
        b: TensorOperand<'_>,
        c: TensorOperand<'_>,
        d: TensorOperand<'_>,
        compute_descriptor: ComputeDescriptor,
    ) -> Result<Self> {
        let context_ref = context.as_context_ref();
        validate_tensor_operand_context(&context_ref, a, "a")?;
        validate_tensor_operand_context(&context_ref, b, "b")?;
        validate_tensor_operand_context(&context_ref, c, "c")?;
        validate_tensor_operand_context(&context_ref, d, "d")?;
        validate_modes(a.desc.rank(), a.modes)?;
        validate_modes(b.desc.rank(), b.modes)?;
        validate_modes(c.desc.rank(), c.modes)?;
        validate_modes(d.desc.rank(), d.modes)?;
        validate_tensor_descriptors_match(c.desc, d.desc, "contraction output")?;
        validate_mode_names_match(c.modes, d.modes, "contraction output")?;

        let mode_a = modes_to_i32_vec(a.modes);
        let mode_b = modes_to_i32_vec(b.modes);
        let mode_c = modes_to_i32_vec(c.modes);
        let mode_d = modes_to_i32_vec(d.modes);

        let mut raw = ptr::null_mut();
        context.bind()?;
        // SAFETY: `cutensorCreateContraction` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cutensorCreateContraction(
                context.as_raw(),
                &raw mut raw,
                a.desc.as_raw(),
                mode_a.as_ptr(),
                a.op.into(),
                b.desc.as_raw(),
                mode_b.as_ptr(),
                b.op.into(),
                c.desc.as_raw(),
                mode_c.as_ptr(),
                c.op.into(),
                d.desc.as_raw(),
                mode_d.as_ptr(),
                compute_descriptor.into(),
            ))?;
        }

        Self::from_native(
            raw,
            context_ref,
            d.desc.rank(),
            d.desc.data_type(),
            OperationKind::Contraction {
                a: a.desc.data_type(),
                b: b.desc.data_type(),
                c: c.desc.data_type(),
                d: d.desc.data_type(),
            },
        )
    }

    /// Creates an operation descriptor for a tensor reduction of the form $D = \alpha \cdot op\_reduce(op\_A(A)) + \beta \cdot op\_C(C)$.
    ///
    /// For example, this can reduce an entire tensor to a scalar: `C[] = alpha * A[i,j,k]`.
    ///
    /// Can also perform partial reductions; for instance,
    /// `C[i,j] = alpha * A[k,j,i]`.
    /// In this case only elements along the `k` mode are contracted.
    ///
    /// The binary `op_reduce` operator controls the kind of reduction that is performed.
    /// For instance, setting `op_reduce` to [`Operator::Add`] reduces elements of A via a summation while [`Operator::Max`] would find the largest element in A.
    ///
    /// Supported data-type combinations are:
    ///
    /// | A type | B type | C type | compute type |
    /// | --- | --- | --- | --- |
    /// | [`DataType::F16`] | [`DataType::F16`] | [`DataType::F16`] | [`ComputeDescriptor::f16`] |
    /// | [`DataType::F16`] | [`DataType::F16`] | [`DataType::F16`] | [`ComputeDescriptor::f32`] |
    /// | [`DataType::Bf16`] | [`DataType::Bf16`] | [`DataType::Bf16`] | [`ComputeDescriptor::bf16`] |
    /// | [`DataType::Bf16`] | [`DataType::Bf16`] | [`DataType::Bf16`] | [`ComputeDescriptor::f32`] |
    /// | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | [`ComputeDescriptor::f32`] |
    /// | [`DataType::F64`] | [`DataType::F64`] | [`DataType::F64`] | [`ComputeDescriptor::f64`] |
    /// | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`ComputeDescriptor::f32`] |
    /// | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | [`ComputeDescriptor::f64`] |
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, tensor modes do not
    /// match descriptor ranks, input and output descriptors or mode sets are
    /// incompatible, a data type or reduction operator combination is
    /// unsupported, cuTENSOR rejects the operands, or cuTENSOR returns a null
    /// operation descriptor.
    pub fn reduction(
        context: &Context,
        a: TensorOperand<'_>,
        c: TensorOperand<'_>,
        d: TensorOperand<'_>,
        op_reduce: Operator,
        compute_descriptor: ComputeDescriptor,
    ) -> Result<Self> {
        let context_ref = context.as_context_ref();
        validate_tensor_operand_context(&context_ref, a, "a")?;
        validate_tensor_operand_context(&context_ref, c, "c")?;
        validate_tensor_operand_context(&context_ref, d, "d")?;
        validate_modes(a.desc.rank(), a.modes)?;
        validate_modes(c.desc.rank(), c.modes)?;
        validate_modes(d.desc.rank(), d.modes)?;
        validate_tensor_descriptors_match(c.desc, d.desc, "reduction output")?;
        validate_mode_names_match(c.modes, d.modes, "reduction output")?;

        let mode_a = modes_to_i32_vec(a.modes);
        let mode_c = modes_to_i32_vec(c.modes);
        let mode_d = modes_to_i32_vec(d.modes);

        let mut raw = ptr::null_mut();
        context.bind()?;
        // SAFETY: `cutensorCreateReduction` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cutensorCreateReduction(
                context.as_raw(),
                &raw mut raw,
                a.desc.as_raw(),
                mode_a.as_ptr(),
                a.op.into(),
                c.desc.as_raw(),
                mode_c.as_ptr(),
                c.op.into(),
                d.desc.as_raw(),
                mode_d.as_ptr(),
                op_reduce.into(),
                compute_descriptor.into(),
            ))?;
        }

        Self::from_native(
            raw,
            context_ref,
            d.desc.rank(),
            d.desc.data_type(),
            OperationKind::Reduction {
                a: a.desc.data_type(),
                c: c.desc.data_type(),
                d: d.desc.data_type(),
            },
        )
    }

    /// Creates an operation descriptor for a tensor contraction of the form $\mathcal{E} = \alpha \mathcal{A} \mathcal{B} \mathcal{C} + \beta \mathcal{D}$.
    ///
    /// The descriptor represents a tensor contraction of the form:
    ///
    /// $$ \mathcal{E}\_{{modes}\_\mathcal{E}} \gets \alpha op\_\mathcal{A}(\mathcal{A}\_{{modes}\_\mathcal{A}}) op\_\mathcal{B}(\mathcal{B}\_{{modes}\_\mathcal{B}}) op\_\mathcal{C}(\mathcal{C}\_{{modes}\_\mathcal{C}}) + \beta op\_\mathcal{D}(\mathcal{D}\_{{modes}\_\mathcal{D}}). $$
    /// Use [`Plan::create`](crate::plan::Plan::create) to select a kernel, then call [`Plan::contract_trinary`](crate::plan::Plan::contract_trinary) to execute the contraction.
    ///
    /// The returned descriptor frees the cuTENSOR operation descriptor when dropped.
    ///
    /// Performance improvements are currently especially high for host-resident
    /// data, also called out-of-core data, on Grace-based systems.
    ///
    /// Supported data-type combinations are:
    ///
    /// | A type | B type | C type | D type | compute descriptor | scalar type | Tensor Core |
    /// | --- | --- | --- | --- | --- | --- | --- |
    /// | [`DataType::F16`] | [`DataType::F16`] | [`DataType::F16`] | [`DataType::F16`] | [`ComputeDescriptor::f32`] | [`DataType::F32`] | Volta+ |
    /// | [`DataType::Bf16`] | [`DataType::Bf16`] | [`DataType::Bf16`] | [`DataType::Bf16`] | [`ComputeDescriptor::f32`] | [`DataType::F32`] | Ampere+ |
    /// | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | [`ComputeDescriptor::f32`] | [`DataType::F32`] | No |
    /// | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | [`ComputeDescriptor::tf32`] | [`DataType::F32`] | Ampere+ |
    /// | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | [`ComputeDescriptor::tf32x3`] | [`DataType::F32`] | Ampere+ |
    /// | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | [`ComputeDescriptor::bf16`] | [`DataType::F32`] | Ampere+ |
    /// | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | [`ComputeDescriptor::f16`] | [`DataType::F32`] | Volta+ |
    /// | [`DataType::F64`] | [`DataType::F64`] | [`DataType::F64`] | [`DataType::F64`] | [`ComputeDescriptor::f64`] | [`DataType::F64`] | Ampere+ |
    /// | [`DataType::F64`] | [`DataType::F64`] | [`DataType::F64`] | [`DataType::F64`] | [`ComputeDescriptor::f32`] | [`DataType::F64`] | No |
    /// | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`ComputeDescriptor::f32`] | [`DataType::ComplexF32`] | No |
    /// | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`ComputeDescriptor::tf32`] | [`DataType::ComplexF32`] | Ampere+ |
    /// | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`ComputeDescriptor::tf32x3`] | [`DataType::ComplexF32`] | Ampere+ |
    /// | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | [`ComputeDescriptor::f64`] | [`DataType::ComplexF64`] | Ampere+ |
    /// | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | [`ComputeDescriptor::f32`] | [`DataType::ComplexF64`] | No |
    /// | [`DataType::F64`] | [`DataType::F64`] | [`DataType::F64`] | [`DataType::F64`] | [`ComputeDescriptor::i8x8`] | [`DataType::F64`] | Hopper+ |
    /// | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | [`ComputeDescriptor::i8x8`] | [`DataType::ComplexF64`] | Hopper+ |
    /// | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | [`ComputeDescriptor::bf16x9`] | [`DataType::F32`] | Hopper+ |
    /// | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`ComputeDescriptor::bf16x9`] | [`DataType::ComplexF32`] | Hopper+ |
    /// | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | [`ComputeDescriptor::f16x4`] | [`DataType::F32`] | Blackwell+ |
    /// | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`ComputeDescriptor::f16x4`] | [`DataType::ComplexF32`] | Blackwell+ |
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, tensor modes do not
    /// match descriptor ranks, input and output descriptors or mode sets are
    /// incompatible, a data type or operator combination is unsupported,
    /// cuTENSOR rejects the operands, or cuTENSOR returns a null operation
    /// descriptor.
    pub fn contraction_trinary(
        context: &Context,
        lhs: TensorOperand<'_>,
        rhs: TensorOperand<'_>,
        aux: TensorOperand<'_>,
        input: TensorOperand<'_>,
        output: TensorOperand<'_>,
        compute_descriptor: ComputeDescriptor,
    ) -> Result<Self> {
        let context_ref = context.as_context_ref();
        validate_tensor_operand_context(&context_ref, lhs, "lhs")?;
        validate_tensor_operand_context(&context_ref, rhs, "rhs")?;
        validate_tensor_operand_context(&context_ref, aux, "aux")?;
        validate_tensor_operand_context(&context_ref, input, "input")?;
        validate_tensor_operand_context(&context_ref, output, "output")?;
        validate_modes(lhs.desc.rank(), lhs.modes)?;
        validate_modes(rhs.desc.rank(), rhs.modes)?;
        validate_modes(aux.desc.rank(), aux.modes)?;
        validate_modes(input.desc.rank(), input.modes)?;
        validate_modes(output.desc.rank(), output.modes)?;
        validate_tensor_descriptors_match(input.desc, output.desc, "trinary contraction output")?;
        validate_mode_names_match(input.modes, output.modes, "trinary contraction output")?;

        let mode_a = modes_to_i32_vec(lhs.modes);
        let mode_b = modes_to_i32_vec(rhs.modes);
        let mode_c = modes_to_i32_vec(aux.modes);
        let mode_d = modes_to_i32_vec(input.modes);
        let mode_e = modes_to_i32_vec(output.modes);

        let mut raw = ptr::null_mut();
        context.bind()?;
        // SAFETY: `cutensorCreateContractionTrinary` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cutensorCreateContractionTrinary(
                context.as_raw(),
                &raw mut raw,
                lhs.desc.as_raw(),
                mode_a.as_ptr(),
                lhs.op.into(),
                rhs.desc.as_raw(),
                mode_b.as_ptr(),
                rhs.op.into(),
                aux.desc.as_raw(),
                mode_c.as_ptr(),
                aux.op.into(),
                input.desc.as_raw(),
                mode_d.as_ptr(),
                input.op.into(),
                output.desc.as_raw(),
                mode_e.as_ptr(),
                compute_descriptor.into(),
            ))?;
        }

        Self::from_native(
            raw,
            context_ref,
            output.desc.rank(),
            output.desc.data_type(),
            OperationKind::ContractionTrinary {
                a: lhs.desc.data_type(),
                b: rhs.desc.data_type(),
                c: aux.desc.data_type(),
                d: input.desc.data_type(),
                e: output.desc.data_type(),
            },
        )
    }

    /// Creates an operation descriptor for a block-sparse tensor contraction of the form $D = \alpha \mathcal{A} \mathcal{B} + \beta \mathcal{C}$.
    ///
    /// The descriptor represents a block-sparse tensor contraction of the form:
    ///
    /// $$ \mathcal{D}\_{{modes}\_\mathcal{D}} \gets \alpha op\_\mathcal{A}(\mathcal{A}\_{{modes}\_\mathcal{A}}) op\_\mathcal{B}(B\_{{modes}\_\mathcal{B}}) + \beta op\_\mathcal{C}(\mathcal{C}\_{{modes}\_\mathcal{C}}). $$
    ///
    /// Only the predefined non-zero blocks of $\mathcal{D}$ that were specified in [`BlockSparseTensorDescriptor::create`] are actually computed.
    /// The other blocks are omitted, even if the true result of the contraction would be non-zero.
    /// Conversely, if a predefined non-zero block of $\mathcal{D}$ is present but the result of the contraction is zero for this block, explicit zeros are stored.
    ///
    /// Currently, the data-types for the tensors `A`, `B`, `C`, and `D`, as well as the scalars $\alpha$ and $\beta$ must all be identical, and the only supported types are [`DataType::ComplexF64`], [`DataType::ComplexF32`], [`DataType::F64`], and [`DataType::F32`].
    /// The compute type must match as well; currently supported combinations are:
    ///
    /// | A type | B type | C type | D type | compute descriptor | scalar type |
    /// | --- | --- | --- | --- | --- | --- |
    /// | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | [`ComputeDescriptor::f32`] | [`DataType::F32`] |
    /// | [`DataType::F64`] | [`DataType::F64`] | [`DataType::F64`] | [`DataType::F64`] | [`ComputeDescriptor::f64`] | [`DataType::F64`] |
    /// | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`ComputeDescriptor::f32`] | [`DataType::ComplexF32`] |
    /// | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | [`ComputeDescriptor::f64`] | [`DataType::ComplexF64`] |
    ///
    /// For every mode, the segmentation of that mode must be identical in all tensors that it occurs in.
    /// For example, if mode `i` of tensor `A` matches mode `j` of tensor `B`, then the section count for mode `i` in `A` must match the section count for mode `j` in `B`, and the corresponding section extents must be identical.
    ///
    /// For example, let A, B, and C be block matrices and consider the ordinary matrix-matrix product $C\_{mn}=A\_{mk}B\_{kn}$.
    /// Then:
    ///
    /// * Mode ‘m’: C and A must have the same number of block-rows, and each block-row of C must contain the same number of rows as the corresponding block-row of A.
    /// * Mode ‘n’: C and B must have the same number of block-columns of matching size.
    /// * Mode ‘k’: A must have the same number of block-columns as B has block-rows, and each block-column of A must contain the same number of columns as the number of rows in the corresponding block-row of B.
    ///
    /// `input` and `output` must use identical descriptors: the same opaque pointer must be passed, and the layouts of the input and output tensors must be identical.
    ///
    /// See [`sys::cutensorCreatePlan`] to create the plan, [`Plan::estimate_workspace_size`](crate::plan::Plan::estimate_workspace_size) to compute the required workspace, and finally [`Plan::block_sparse_contract`](crate::plan::Plan::block_sparse_contract) to perform the actual contraction.
    ///
    /// The returned descriptor frees the cuTENSOR operation descriptor when dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, block-sparse tensor
    /// modes do not match descriptor ranks, input and output descriptors or
    /// mode sets are incompatible, section sizes are invalid, a data type or
    /// operator combination is unsupported, cuTENSOR rejects the operands, or
    /// cuTENSOR returns a null operation descriptor.
    pub fn block_sparse_contraction(
        context: &Context,
        a: BlockSparseTensorOperand<'_>,
        b: BlockSparseTensorOperand<'_>,
        c: BlockSparseTensorOperand<'_>,
        d: BlockSparseTensorOperand<'_>,
        compute_descriptor: ComputeDescriptor,
    ) -> Result<Self> {
        let context_ref = context.as_context_ref();
        validate_block_sparse_tensor_operand_context(&context_ref, a, "a")?;
        validate_block_sparse_tensor_operand_context(&context_ref, b, "b")?;
        validate_block_sparse_tensor_operand_context(&context_ref, c, "c")?;
        validate_block_sparse_tensor_operand_context(&context_ref, d, "d")?;
        validate_modes(a.desc.mode_count(), a.modes)?;
        validate_modes(b.desc.mode_count(), b.modes)?;
        validate_modes(c.desc.mode_count(), c.modes)?;
        validate_modes(d.desc.mode_count(), d.modes)?;
        validate_block_sparse_tensor_descriptors_match(c.desc, d.desc, "block sparse output")?;
        validate_mode_names_match(c.modes, d.modes, "block sparse output")?;

        let mode_a = modes_to_i32_vec(a.modes);
        let mode_b = modes_to_i32_vec(b.modes);
        let mode_c = modes_to_i32_vec(c.modes);
        let mode_d = modes_to_i32_vec(d.modes);

        let mut raw = ptr::null_mut();
        context.bind()?;
        // SAFETY: `cutensorCreateBlockSparseContraction` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cutensorCreateBlockSparseContraction(
                context.as_raw(),
                &raw mut raw,
                a.desc.as_raw(),
                mode_a.as_ptr(),
                a.op.into(),
                b.desc.as_raw(),
                mode_b.as_ptr(),
                b.op.into(),
                c.desc.as_raw(),
                mode_c.as_ptr(),
                c.op.into(),
                d.desc.as_raw(),
                mode_d.as_ptr(),
                compute_descriptor.into(),
            ))?;
        }

        Self::from_native(
            raw,
            context_ref,
            d.desc.mode_count(),
            d.desc.data_type(),
            OperationKind::BlockSparseContraction {
                a_non_zero_blocks: a.desc.non_zero_block_count(),
                b_non_zero_blocks: b.desc.non_zero_block_count(),
                c_non_zero_blocks: c.desc.non_zero_block_count(),
                d_non_zero_blocks: d.desc.non_zero_block_count(),
            },
        )
    }

    fn set_attribute<T>(&mut self, attr: OperationDescriptorAttribute, value: &T) -> Result<()> {
        validate_operation_attribute_size(attr, size_of::<T>())?;

        self.context.bind()?;
        // SAFETY: `cutensorOperationDescriptorSetAttribute` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cutensorOperationDescriptorSetAttribute(
                self.context.as_raw(),
                self.handle,
                attr.into(),
                ptr::from_ref(value).cast(),
                size_of::<T>() as _,
            ))?;
        }
        Ok(())
    }

    fn attribute<T>(&self, attr: OperationDescriptorAttribute) -> Result<T> {
        validate_operation_attribute_size(attr, size_of::<T>())?;

        let mut value = MaybeUninit::<T>::uninit();
        self.context.bind()?;
        // SAFETY: the library wrote exactly `size_of::<T>()` bytes, checked above, initializing `value`.
        unsafe {
            try_ffi!(sys::cutensorOperationDescriptorGetAttribute(
                self.context.as_raw(),
                self.handle,
                attr.into(),
                value.as_mut_ptr().cast(),
                size_of::<T>() as _,
            ))?;
            Ok(value.assume_init())
        }
    }

    /// Returns the integer tag distinguishing otherwise identical problems in the software-managed plan cache.
    ///
    /// # Errors
    ///
    /// Returns an error if the native attribute byte size is incompatible, the CUDA context cannot be bound, or the native attribute query fails.
    pub fn tag(&self) -> Result<i32> {
        let value = self.attribute(OperationDescriptorAttribute::Tag)?;
        Ok(value)
    }

    /// Sets the integer plan-cache tag; the native default is zero.
    ///
    /// # Errors
    ///
    /// Returns an error if the native attribute byte size is incompatible, the CUDA context cannot be bound, or cuTENSOR rejects the attribute value.
    pub fn set_tag(&mut self, tag: i32) -> Result<()> {
        self.set_attribute(OperationDescriptorAttribute::Tag, &tag)
    }

    /// Returns the native data type of the operation's scaling factors.
    ///
    /// # Errors
    ///
    /// Returns an error if the native attribute byte size is incompatible, the CUDA context cannot be bound, or the native attribute query fails.
    pub fn scalar_type(&self) -> Result<DataType> {
        self.attribute(OperationDescriptorAttribute::ScalarType)
    }

    /// Sets the native scaling-factor type and updates the tracked execution signature after success.
    ///
    /// # Errors
    ///
    /// Returns an error if the native attribute byte size is incompatible, the CUDA context cannot be bound, or cuTENSOR rejects the attribute value.
    pub fn set_scalar_type(&mut self, data_type: DataType) -> Result<()> {
        self.set_attribute(OperationDescriptorAttribute::ScalarType, &data_type)?;
        self.signature.scalar_type = data_type;
        Ok(())
    }

    /// Queries the number of padded values on the left of each output mode.
    ///
    /// The returned vector has one entry per output mode, in output descriptor order.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or the native padding query fails.
    pub fn padding_left(&self) -> Result<Vec<u32>> {
        self.padding(OperationDescriptorAttribute::PaddingLeft)
    }

    /// Sets the number of padded values on the left of each output mode.
    ///
    /// Entries follow output descriptor order and count values, rather than bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice length differs from the output rank, a value cannot fit `i32`, the CUDA context cannot be bound, or cuTENSOR rejects the padding.
    pub fn set_padding_left(&mut self, padding: &[u32]) -> Result<()> {
        self.set_padding(OperationDescriptorAttribute::PaddingLeft, padding)
    }

    /// Queries the number of padded values on the right of each output mode.
    ///
    /// The returned vector has one entry per output mode, in output descriptor order.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or the native padding query fails.
    pub fn padding_right(&self) -> Result<Vec<u32>> {
        self.padding(OperationDescriptorAttribute::PaddingRight)
    }

    /// Sets the number of padded values on the right of each output mode.
    ///
    /// Entries follow output descriptor order and count values, rather than bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice length differs from the output rank, a value cannot fit `i32`, the CUDA context cannot be bound, or cuTENSOR rejects the padding.
    pub fn set_padding_right(&mut self, padding: &[u32]) -> Result<()> {
        self.set_padding(OperationDescriptorAttribute::PaddingRight, padding)
    }

    /// Sets and retains a host-side constant used to fill padded output elements.
    ///
    /// This uses the output tensor's data type, which can differ from the scaling-factor type returned by [`Self::scalar_type`]. The owned value is replaced only after the native update succeeds.
    ///
    /// # Errors
    ///
    /// Returns an error if `T` does not match the output tensor type, the wrapper cannot retain this scalar representation, the CUDA context cannot be bound, or cuTENSOR rejects the value.
    pub fn set_padding_value<T: DataTypeLike>(&mut self, value: T) -> Result<()> {
        if self.output_data_type != T::data_type() {
            return Err(Error::ScalarDataTypeMismatch {
                descriptor: self.output_data_type,
                memory: T::data_type(),
            });
        }

        let owned = OwnedPaddingValue::from_value(value)?;
        self.context.bind()?;
        // SAFETY: `cutensorOperationDescriptorSetAttribute` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cutensorOperationDescriptorSetAttribute(
                self.context.as_raw(),
                self.handle,
                OperationDescriptorAttribute::PaddingValue.into(),
                owned.host_ptr().cast(),
                owned.size_bytes() as _,
            ))?;
        }
        self.padding_value = Some(owned);
        Ok(())
    }

    /// Returns the native floating-point operation-count estimate as `f32`.
    ///
    /// The estimate assumes nonzero scaling factors unless the operation specifies otherwise.
    ///
    /// # Errors
    ///
    /// Returns an error if the native attribute byte size is incompatible, the CUDA context cannot be bound, or the native attribute query fails.
    pub fn flops(&self) -> Result<f32> {
        self.attribute(OperationDescriptorAttribute::Flops)
    }

    /// Returns the native estimate of minimum bytes transferred to or from global memory as `f32`.
    ///
    /// The estimate assumes nonzero scaling factors unless the operation specifies otherwise.
    ///
    /// # Errors
    ///
    /// Returns an error if the native attribute byte size is incompatible, the CUDA context cannot be bound, or the native attribute query fails.
    pub fn moved_bytes(&self) -> Result<f32> {
        self.attribute(OperationDescriptorAttribute::MovedBytes)
    }

    pub(crate) fn signature(&self) -> OperationSignature {
        self.signature
    }

    /// Returns the borrowed native operation descriptor.
    #[must_use]
    pub const fn as_raw(&self) -> sys::cutensorOperationDescriptor_t {
        self.handle
    }

    /// Consumes the descriptor and returns the raw cuTENSOR operation
    /// descriptor handle without destroying it.
    ///
    /// The returned record retains the native library owner and descriptor metadata.
    /// Destroy its handle before releasing the record, or reclaim it with `from_raw`.
    #[must_use]
    pub fn into_raw(self) -> RawOperationDescriptor {
        let owner = ManuallyDrop::new(self);
        // SAFETY: move each field exactly once while suppressing native destruction.
        unsafe {
            RawOperationDescriptor {
                handle: owner.handle,
                context: ptr::read(&raw const owner.context),
                output_rank: owner.output_rank,
                output_data_type: owner.output_data_type,
                signature: owner.signature,
                padding_value: ptr::read(&raw const owner.padding_value),
            }
        }
    }

    /// Reclaims a transferred resource with its original metadata and library owner.
    ///
    /// # Safety
    ///
    /// `handle` must remain valid and uniquely owned with its original descriptor metadata unchanged.
    /// Complete all borrowed queued accesses before replacing or releasing their resources.
    #[must_use]
    pub unsafe fn from_raw(raw: RawOperationDescriptor) -> Self {
        Self {
            handle: raw.handle,
            context: raw.context,
            output_rank: raw.output_rank,
            output_data_type: raw.output_data_type,
            signature: raw.signature,
            padding_value: raw.padding_value,
        }
    }

    fn from_native(
        handle: sys::cutensorOperationDescriptor_t,
        context: ContextRef,
        output_rank: u32,
        output_data_type: DataType,
        kind: OperationKind,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        let scalar_type = {
            let mut value = MaybeUninit::<DataType>::uninit();
            context.bind()?;
            // SAFETY: the library wrote exactly `size_of::<T>()` bytes, checked above, initializing `value`.
            unsafe {
                try_ffi!(sys::cutensorOperationDescriptorGetAttribute(
                    context.as_raw(),
                    handle,
                    OperationDescriptorAttribute::ScalarType.into(),
                    value.as_mut_ptr().cast(),
                    size_of::<DataType>() as _,
                ))?;
                value.assume_init()
            }
        };

        Ok(Self {
            handle,
            context,
            output_rank,
            output_data_type,
            signature: OperationSignature { kind, scalar_type },
            padding_value: None,
        })
    }

    fn padding(&self, attr: OperationDescriptorAttribute) -> Result<Vec<u32>> {
        debug_assert!(matches!(
            attr,
            OperationDescriptorAttribute::PaddingLeft | OperationDescriptorAttribute::PaddingRight
        ));

        let mut padding = vec![0_u32; self.output_rank as usize];
        self.context.bind()?;
        // SAFETY: `cutensorOperationDescriptorGetAttribute` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cutensorOperationDescriptorGetAttribute(
                self.context.as_raw(),
                self.handle,
                attr.into(),
                padding.as_mut_ptr().cast(),
                size_of_val(padding.as_slice()) as _,
            ))?;
        }

        Ok(padding)
    }

    fn set_padding(&mut self, attr: OperationDescriptorAttribute, padding: &[u32]) -> Result<()> {
        debug_assert!(matches!(
            attr,
            OperationDescriptorAttribute::PaddingLeft | OperationDescriptorAttribute::PaddingRight
        ));

        if padding.len() != self.output_rank as usize {
            return Err(Error::LengthMismatch {
                name: "padding".into(),
                expected: self.output_rank as usize,
                actual: padding.len(),
            });
        }
        for &value in padding {
            to_i32(value, "padding")?;
        }

        let padding = padding.to_vec();
        self.context.bind()?;
        // SAFETY: `cutensorOperationDescriptorSetAttribute` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cutensorOperationDescriptorSetAttribute(
                self.context.as_raw(),
                self.handle,
                attr.into(),
                padding.as_ptr().cast(),
                size_of_val(padding.as_slice()) as _,
            ))?;
        }
        Ok(())
    }
}

impl Drop for OperationDescriptor {
    fn drop(&mut self) {
        if let Err(error) = self.context.bind() {
            #[cfg(debug_assertions)]
            eprintln!(
                "failed to bind cutensor context before destroying operation descriptor: {error}"
            );
        }

        // SAFETY: `cutensorDestroyOperationDescriptor` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cutensorDestroyOperationDescriptor(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cutensor operation descriptor: {error}");
            }
        }
    }
}

fn validate_modes(rank: u32, modes: &[Mode]) -> Result<()> {
    if rank as usize != modes.len() {
        return Err(Error::TensorModeMismatch {
            rank,
            mode_length: modes.len(),
        });
    }

    Ok(())
}

fn validate_tensor_operand_context(
    expected: &ContextRef,
    operand: TensorOperand<'_>,
    name: &str,
) -> Result<()> {
    validate_same_context(expected, operand.desc.context(), name)
}

fn validate_block_sparse_tensor_operand_context(
    expected: &ContextRef,
    operand: BlockSparseTensorOperand<'_>,
    name: &str,
) -> Result<()> {
    validate_same_context(expected, operand.desc.context(), name)
}

fn validate_mode_names_match(lhs: &[Mode], rhs: &[Mode], name: &str) -> Result<()> {
    if lhs != rhs {
        return Err(Error::ModeMismatch {
            name: name.to_owned().into(),
        });
    }

    Ok(())
}

fn validate_tensor_descriptors_match(
    lhs: &TensorDescriptor,
    rhs: &TensorDescriptor,
    name: &str,
) -> Result<()> {
    if lhs.shape() != rhs.shape()
        || lhs.strides() != rhs.strides()
        || lhs.data_type() != rhs.data_type()
        || lhs.alignment_requirement() != rhs.alignment_requirement()
    {
        return Err(Error::DescriptorMismatch {
            name: name.to_owned().into(),
        });
    }

    Ok(())
}

fn validate_block_sparse_tensor_descriptors_match(
    lhs: &BlockSparseTensorDescriptor,
    rhs: &BlockSparseTensorDescriptor,
    name: &str,
) -> Result<()> {
    if lhs.mode_count() != rhs.mode_count()
        || lhs.non_zero_block_count() != rhs.non_zero_block_count()
        || lhs.data_type() != rhs.data_type()
    {
        return Err(Error::DescriptorMismatch {
            name: name.to_owned().into(),
        });
    }

    Ok(())
}

fn validate_operation_attribute_size(
    attr: OperationDescriptorAttribute,
    actual: usize,
) -> Result<()> {
    let expected = match attr {
        OperationDescriptorAttribute::Tag => size_of::<i32>(),
        OperationDescriptorAttribute::ScalarType => size_of::<sys::cutensorDataType_t>(),
        OperationDescriptorAttribute::Flops | OperationDescriptorAttribute::MovedBytes => {
            size_of::<f32>()
        }
        OperationDescriptorAttribute::PaddingLeft | OperationDescriptorAttribute::PaddingRight => {
            return Ok(());
        }
        OperationDescriptorAttribute::PaddingValue => return Ok(()),
    };

    if actual != expected {
        return Err(Error::OperationDescriptorInvalidAttributeSize {
            attr,
            expected,
            actual,
        });
    }

    Ok(())
}
