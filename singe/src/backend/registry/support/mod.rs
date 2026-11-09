mod formatting;
mod library;
mod profile;
mod report;

use std::fmt;

use serde::{Deserialize, Serialize};

pub use self::{
    library::{
        CublasFeature, CublasLtFeature, CudnnFeature, CufftFeature, CutensorFeature, LibraryFeature,
    },
    profile::OperationSupportProfile,
    report::operation_support_table,
};
use crate::{
    backend::registry::{AliasBehavior, OperandRole, Operation, OperationExecutorFamily, table},
    data_type::DataType,
    scalar::Scalar,
    value::ValueSpec,
};

/// Execution mode used to query operation support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendExecution {
    /// Immediate execution on the CPU.
    CpuEager,
    /// Immediate execution on a CUDA device.
    CudaEager,
    /// Execution as part of a prepared CUDA graph.
    CudaGraph,
}

/// Physical tensor layout accepted or produced by an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayoutSupport {
    /// Accepts tensor views with arbitrary supported strides.
    AnyStrided,
    /// Materializes tensor data into contiguous storage.
    MaterializesToContiguous,
    /// Requires contiguous tensor storage.
    RequiresContiguous,
}

/// Execution provider available when preparing a CUDA graph operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CudaGraphProvider {
    /// Reuses input storage through a metadata-only operation.
    Alias,
    /// Executes a generated CUDA kernel.
    Kernel,
    /// Executes through the cuDNN backend.
    CudnnBackend,
    /// Executes through cuBLASLt.
    CublasLt,
    /// Executes through cuBLAS.
    Cublas,
    /// Executes through cuTENSOR.
    Cutensor,
    /// Executes through cuFFT.
    Cufft,
    /// Executes a chain of element-wise operations as one generated kernel.
    Fusion,
}

impl CudaGraphProvider {
    // Whether this provider calls a library rather than running generated code.
    #[must_use]
    pub const fn is_library(self) -> bool {
        matches!(
            self,
            Self::CudnnBackend | Self::CublasLt | Self::Cublas | Self::Cutensor | Self::Cufft
        )
    }
}

// CUDA graph providers and what each library among them may run, as the graph passes read them, since they run before any backend does.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct CudaGraphProviderSet {
    providers: u16,
    features: u16,
}

impl CudaGraphProviderSet {
    const LIBRARIES: u16 = (1 << CudaGraphProvider::CudnnBackend as u16)
        | (1 << CudaGraphProvider::CublasLt as u16)
        | (1 << CudaGraphProvider::Cublas as u16)
        | (1 << CudaGraphProvider::Cutensor as u16)
        | (1 << CudaGraphProvider::Cufft as u16);

    #[must_use]
    pub const fn all() -> Self {
        Self {
            providers: u16::MAX,
            features: u16::MAX,
        }
    }

    #[must_use]
    pub const fn empty() -> Self {
        Self {
            providers: 0,
            features: 0,
        }
    }

    // Adds `provider`, with all it may run if it is a library.
    #[must_use]
    pub const fn with(self, provider: CudaGraphProvider) -> Self {
        let features = match provider {
            CudaGraphProvider::CudnnBackend => LibraryFeature::Cudnn(CudnnFeature::All).bits(),
            CudaGraphProvider::Cublas => LibraryFeature::Cublas(CublasFeature::All).bits(),
            CudaGraphProvider::CublasLt => LibraryFeature::CublasLt(CublasLtFeature::All).bits(),
            CudaGraphProvider::Cufft => LibraryFeature::Cufft(CufftFeature::All).bits(),
            CudaGraphProvider::Cutensor => LibraryFeature::Cutensor(CutensorFeature::All).bits(),
            CudaGraphProvider::Alias | CudaGraphProvider::Kernel | CudaGraphProvider::Fusion => 0,
        };
        Self {
            providers: self.providers | (1 << provider as u16),
            features: self.features | features,
        }
    }

    // Adds what `feature` names, which adds its library if this set has none of it yet.
    #[must_use]
    pub const fn with_feature(self, feature: LibraryFeature) -> Self {
        Self {
            providers: self.providers | (1 << feature.provider() as u16),
            features: self.features | feature.bits(),
        }
    }

    // Removes every library, leaving aliases, kernels, and fusion as they are.
    #[must_use]
    pub const fn without_libraries(self) -> Self {
        Self {
            providers: self.providers & !Self::LIBRARIES,
            features: 0,
        }
    }

    // Whether any of `provider` is in this set, which says nothing about what it may run.
    #[must_use]
    pub const fn contains(self, provider: CudaGraphProvider) -> bool {
        self.providers & (1 << provider as u16) != 0
    }

    // Whether `provider` may run `operation`.
    // A pair no feature names is only as allowed as its provider.
    #[must_use]
    pub const fn allows(self, provider: CudaGraphProvider, operation: Operation) -> bool {
        self.contains(provider)
            && match LibraryFeature::of(provider, operation) {
                Some(feature) => self.features & feature.bits() != 0,
                None => true,
            }
    }
}

impl fmt::Debug for CudaGraphProviderSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let plain = [
            CudaGraphProvider::Alias,
            CudaGraphProvider::Kernel,
            CudaGraphProvider::Fusion,
        ]
        .into_iter()
        .filter(|&provider| self.contains(provider));
        f.debug_struct("CudaGraphProviderSet")
            .field("providers", &plain.collect::<Vec<_>>())
            .field("libraries", &LibraryFeature::held_in(self.features))
            .finish()
    }
}

/// Kind of work required for an operation in a particular execution mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationExecutionFamily {
    /// The execution mode cannot run the operation.
    Unsupported,
    /// Changes tensor metadata while reusing storage.
    MetadataOnly,
    /// May copy a tensor view into the required physical layout.
    LayoutMaterialization,
    /// Copies tensor storage between devices.
    StorageCopy,
    /// Computes tensor values using a backend executor.
    BackendCompute,
}

impl OperationExecutionFamily {
    /// Returns the descriptive name of this execution family.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Unsupported => "unsupported",
            Self::MetadataOnly => "metadata-only",
            Self::LayoutMaterialization => "layout materialization",
            Self::StorageCopy => "storage copy",
            Self::BackendCompute => "backend compute",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OperationDataTypes {
    inputs: &'static [DataType],
    outputs: &'static [DataType],
}

impl OperationDataTypes {
    const fn new(inputs: &'static [DataType], outputs: &'static [DataType]) -> Self {
        Self { inputs, outputs }
    }
}

/// Supported execution modes and their distinct input and output encodings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperationSupport {
    operation: Operation,
    cpu_eager: Option<OperationDataTypes>,
    cuda_eager: Option<OperationDataTypes>,
    compiled: Option<OperationDataTypes>,
    layout: LayoutSupport,
    cuda_graph_providers: &'static [CudaGraphProvider],
}

/// Returns the registry's execution modes, data types, layouts, and CUDA graph providers for `operation`.
#[must_use]
pub fn operation_support(operation: Operation) -> OperationSupport {
    profile::operation_support(operation, table::operation_support_profile(operation))
}

impl OperationSupport {
    /// Checks a scalar operation after weak scalar promotion and operand conversion.
    /// Physical layouts and provider restrictions still require preparation checks.
    #[must_use]
    pub fn supports_scalar_argument(
        &self,
        execution: BackendExecution,
        input: &ValueSpec,
        scalar: Scalar,
    ) -> bool {
        if !self.operation.is_scalar_elementwise() {
            return false;
        }

        let Ok(computation) =
            scalar.promoted_type(input.data_type(), self.operation == Operation::RDivScalar)
        else {
            return false;
        };
        let casts = operation_support(Operation::Cast);
        self.supports_input_data_type(execution, computation)
            && self.supports_data_type(execution, computation)
            && (input.data_type() == computation
                || (casts.supports_input_data_type(execution, input.data_type())
                    && casts.supports_data_type(execution, computation)))
    }

    /// Returns the operation described by this support record.
    #[must_use]
    pub fn operation(&self) -> Operation {
        self.operation
    }

    fn execution_data_types(&self, execution: BackendExecution) -> Option<OperationDataTypes> {
        match execution {
            BackendExecution::CpuEager => self.cpu_eager,
            BackendExecution::CudaEager => self.cuda_eager,
            BackendExecution::CudaGraph => self.compiled,
        }
    }

    /// Returns whether the registry declares support for `execution`.
    ///
    /// This checks execution-mode availability without validating operands, layouts, or hardware.
    #[must_use]
    pub fn supports(&self, execution: BackendExecution) -> bool {
        self.execution_data_types(execution).is_some()
    }

    /// Returns the kind of work required in `execution`, or `Unsupported` when that mode is unavailable.
    #[must_use]
    pub fn execution_family(&self, execution: BackendExecution) -> OperationExecutionFamily {
        if !self.supports(execution) {
            return OperationExecutionFamily::Unsupported;
        }

        if table::operation_executor_family(self.operation) == OperationExecutorFamily::CopyTo {
            return OperationExecutionFamily::StorageCopy;
        }

        match table::operation_alias_behavior(self.operation) {
            AliasBehavior::View => OperationExecutionFamily::MetadataOnly,
            AliasBehavior::MayAliasInput => OperationExecutionFamily::LayoutMaterialization,
            AliasBehavior::None | AliasBehavior::WritesInput => {
                OperationExecutionFamily::BackendCompute
            }
        }
    }

    /// Returns the descriptive name of the execution family for `execution`.
    #[must_use]
    pub fn execution_family_name(&self, execution: BackendExecution) -> &'static str {
        self.execution_family(execution).name()
    }

    /// Returns whether `data_type` is a supported output encoding for `execution`.
    #[must_use]
    pub fn supports_data_type(&self, execution: BackendExecution, data_type: DataType) -> bool {
        self.data_types(execution).contains(&data_type)
    }

    /// Returns supported output encodings, or an empty slice for an unavailable execution mode.
    #[must_use]
    pub fn data_types(&self, execution: BackendExecution) -> &'static [DataType] {
        self.execution_data_types(execution)
            .map_or(&[], |types| types.outputs)
    }

    /// Returns whether `data_type` is a supported primary input encoding for `execution`.
    #[must_use]
    pub fn supports_input_data_type(
        &self,
        execution: BackendExecution,
        data_type: DataType,
    ) -> bool {
        self.input_data_types(execution).contains(&data_type)
    }

    /// Returns supported primary input encodings, independently of output encodings.
    #[must_use]
    pub fn input_data_types(&self, execution: BackendExecution) -> &'static [DataType] {
        self.execution_data_types(execution)
            .map_or(&[], |types| types.inputs)
    }

    /// Returns whether the ordered supported input and output data-type slices differ for `execution`.
    #[must_use]
    pub fn has_distinct_input_data_types(&self, execution: BackendExecution) -> bool {
        self.input_data_types(execution) != self.data_types(execution)
    }

    /// Checks each supplied operand type against its role and execution-mode support.
    ///
    /// Roles and types must have equal lengths.
    /// Indices require `I64`, with `I32` also accepted for gather, paged attention, and attention's key lengths.
    /// Conditions require `Bool`.
    /// Other operands use the supported primary input types.
    ///
    /// This method does not check that the supplied roles match the operation schema or that operand types agree with one another.
    #[must_use]
    pub fn supports_operand_data_types(
        &self,
        execution: BackendExecution,
        roles: &[OperandRole],
        data_types: &[DataType],
    ) -> bool {
        self.supports(execution)
            && roles.len() == data_types.len()
            && roles
                .iter()
                .zip(data_types)
                .all(|(role, data_type)| match role {
                    OperandRole::Indices => {
                        *data_type == DataType::I64
                            || (matches!(
                                self.operation,
                                Operation::Take
                                    | Operation::PagedAttention
                                    | Operation::PagedAttentionOutput
                                    | Operation::PagedAttentionLse
                                    | Operation::PagedMlaAttention
                                    | Operation::PagedMlaAttentionOutput
                                    | Operation::PagedMlaAttentionLse
                                    | Operation::GroupedMatMul
                                    | Operation::ScaledDotProductAttention
                            ) && *data_type == DataType::I32)
                    }
                    OperandRole::Condition => *data_type == DataType::Bool,
                    OperandRole::Scale => *data_type == DataType::F32,
                    OperandRole::ZeroPoint | OperandRole::Generator => *data_type == DataType::I64,
                    _ => self.supports_input_data_type(execution, *data_type),
                })
    }

    /// Returns the registry's physical layout requirement for this operation.
    #[must_use]
    pub fn layout(&self) -> LayoutSupport {
        self.layout
    }

    /// Returns the registered CUDA graph execution providers in registry order.
    #[must_use]
    pub fn cuda_graph_providers(&self) -> &'static [CudaGraphProvider] {
        self.cuda_graph_providers
    }

    /// Returns whether CUDA graph execution is supported and `provider` appears in the registered provider list.
    #[must_use]
    pub fn supports_cuda_graph_provider(&self, provider: CudaGraphProvider) -> bool {
        self.supports(BackendExecution::CudaGraph) && self.cuda_graph_providers.contains(&provider)
    }
}
