use std::{fmt, str::FromStr};

use singe_cuda::architecture::GpuArchitecture;
use singe_cudnn::backend::execution::EngineChoice;

use crate::{
    backend::{
        cuda::{
            codegen::KernelArtifact, cudnn_backend::CudnnPaging,
            runtime::CublasLtMatmulAlgorithmFingerprint, tuning::RecordCoverage,
        },
        registry::{
            CublasFeature, CublasLtFeature, CudnnFeature, CufftFeature, CutensorFeature,
            LibraryFeature, Operation,
        },
    },
    data_type::DataType,
    error::{Error, Result},
    graph::NumericalPolicy,
    layout::Layout,
    nn::PageLayout,
};

// A library a program may call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Library {
    Cublas,
    CublasLt,
    Cudnn,
    Cufft,
    Cutensor,
}

impl From<Library> for LibraryFeature {
    fn from(library: Library) -> Self {
        match library {
            Library::Cublas => Self::Cublas(CublasFeature::All),
            Library::CublasLt => Self::CublasLt(CublasLtFeature::All),
            Library::Cudnn => Self::Cudnn(CudnnFeature::All),
            Library::Cufft => Self::Cufft(CufftFeature::All),
            Library::Cutensor => Self::Cutensor(CutensorFeature::All),
        }
    }
}

impl Library {
    pub const ALL: [Self; 5] = [
        Self::Cublas,
        Self::CublasLt,
        Self::Cudnn,
        Self::Cufft,
        Self::Cutensor,
    ];

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cublas => "cublas",
            Self::CublasLt => "cublaslt",
            Self::Cudnn => "cudnn",
            Self::Cufft => "cufft",
            Self::Cutensor => "cutensor",
        }
    }
}

impl fmt::Display for Library {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

impl FromStr for Library {
    type Err = Error;

    fn from_str(name: &str) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|library| library.name() == name)
            .ok_or_else(|| Error::invalid_argument(format!("unknown program library `{name}`")))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(pub usize);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BufferKind {
    // The caller's buffer for the input with this index.
    Input(usize),
    // The caller's buffer for the output with this index.
    Output(usize),
    // Device memory the program allocates once and reuses across runs.
    Internal,
    // Device memory the program allocates zeroed, which a plan's kernels share.
    Scratch,
    // The word checking kernels report data errors through.
    Status,
    // Device memory holding the state tensor with this index.
    State(usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramBuffer {
    pub kind: BufferKind,
    pub bytes: u64,
}

// A buffer read as values of one type and layout, so a view such as a transpose needs no step of its own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferView {
    pub buffer: BufferId,
    pub data_type: DataType,
    pub layout: Layout,
}

// One of a cuBLAS GEMM's matrices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GemmOperand {
    pub transposed: bool,
    pub leading_dimension: usize,
    pub batch_stride: i64,
}

// An activation a library computes inside its call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Relu,
    // The tanh approximation, as `Operation::Gelu` computes.
    Gelu,
}

// A call the program makes to a library, with what the call needs as data.
// What cuDNN's attention reads, writes, and hides.
#[derive(Debug, Clone, PartialEq)]
pub struct CudnnAttentionCall {
    pub query: BufferView,
    pub key: BufferView,
    pub value: BufferView,
    pub output: BufferView,
    pub scale: f32,
    pub causal: bool,
    pub window: Option<i64>,
    pub paged: Option<CudnnPagedCall>,
    // Dense attention's query and key lengths over a padded batch, `[B]` each, which cuDNN reads as `[B, 1, 1, 1]`.
    pub lengths: Option<CudnnLengths>,
    // The engine and knobs a record states, tried before the heuristics.
    pub engine: Option<EngineChoice>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudnnLengths {
    pub query_lengths: BufferView,
    pub key_lengths: BufferView,
}

// The tensors cuDNN reads a paged cache through, which generated kernels write from the paging tensors, and the statistics they read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudnnPagedCall {
    pub paging: CudnnPaging,
    pub offsets: BufferView,
    pub query_lengths: BufferView,
    pub key_lengths: BufferView,
    pub pages: BufferView,
    pub stats: BufferView,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LibraryCall {
    // `cublasGemmStridedBatchedEx` of `rhs · lhs`, column-major, accumulating in `F32`.
    Gemm {
        rows: usize,
        columns: usize,
        inner: usize,
        batch_count: usize,
        lhs_operand: GemmOperand,
        rhs_operand: GemmOperand,
        lhs: BufferView,
        rhs: BufferView,
        output: BufferView,
    },
    // `cufftExec*` of a batch of real signals along the last axis, or of half spectra back to them.
    Fft {
        inverse: bool,
        double: bool,
        length: i64,
        batch: usize,
        input: BufferView,
        output: BufferView,
    },
    // `cublasLtMatmul` of FP8 operands, in the layout `spec::takes_scaled_matmul` accepts, with one scale per operand.
    ScaledMatmul {
        rows: i64,
        depth: i64,
        columns: i64,
        lhs: BufferView,
        rhs: BufferView,
        lhs_scale: BufferView,
        rhs_scale: BufferView,
        output: BufferView,
    },
    // `cutensorContract` of `F32` tensors: `output = alpha · a · b + beta · c`.
    Contraction {
        a: (BufferView, Vec<i64>),
        b: (BufferView, Vec<i64>),
        c: (BufferView, Vec<i64>),
        output: (BufferView, Vec<i64>),
        alpha: f32,
        beta: f32,
    },
    // A cross-correlation through cuDNN's backend, with an optional bias added in its operation graph.
    // `lhs · rhs` as a cuDNN graph, with `addend` and an activation in the same call.
    CudnnMatmul {
        lhs: BufferView,
        rhs: BufferView,
        addend: Option<BufferView>,
        activation: Option<Activation>,
        output: BufferView,
        // The engine and knobs a record states, tried before the heuristics.
        engine: Option<EngineChoice>,
    },
    // `cublasLtMatmul`, laid out as `Gemm`, with `addend` and an activation in the same call.
    CublasLtMatmul {
        rows: usize,
        columns: usize,
        inner: usize,
        batch_count: usize,
        lhs_operand: GemmOperand,
        rhs_operand: GemmOperand,
        lhs: BufferView,
        rhs: BufferView,
        addend: Option<BufferView>,
        activation: Option<Activation>,
        output: BufferView,
        // The configuration a record states, tried before the heuristic.
        config: Option<CublasLtMatmulAlgorithmFingerprint>,
    },
    // `cudnnBackend` attention, over the graph's own keys and values or over a paged cache.
    CudnnAttention(Box<CudnnAttentionCall>),
    Convolution {
        input: BufferView,
        weight: BufferView,
        bias: Option<BufferView>,
        activation: Option<Activation>,
        output: BufferView,
        stride: Vec<i64>,
        padding: Vec<(i64, i64)>,
        dilation: Vec<i64>,
    },
}

impl LibraryCall {
    // The provider's name in `model.json`.
    #[must_use]
    pub const fn library(&self) -> Library {
        match self {
            Self::Gemm { .. } => Library::Cublas,
            Self::Fft { .. } => Library::Cufft,
            Self::ScaledMatmul { .. } | Self::CublasLtMatmul { .. } => Library::CublasLt,
            Self::Contraction { .. } => Library::Cutensor,
            Self::CudnnAttention(_) | Self::Convolution { .. } | Self::CudnnMatmul { .. } => {
                Library::Cudnn
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum ProgramStep {
    Kernel {
        artifact: Box<KernelArtifact>,
        inputs: Vec<BufferView>,
        outputs: Vec<BufferView>,
        scratch: Vec<BufferId>,
        status: Option<BufferId>,
    },
    // Copies a contiguous value's bytes.
    Copy {
        source: BufferView,
        destination: BufferView,
    },
    // Zeroes a contiguous value.
    Fill {
        destination: BufferView,
    },
    // A library call, with the graph operations it computes, which a failure names.
    Library {
        call: Box<LibraryCall>,
        operations: String,
    },
}

// A value the caller passes or receives: its name, as C++ spells it, its type, and how the program reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramValue {
    pub name: String,
    // What the graph calls it, which C++ might not accept.
    pub graph_name: String,
    pub data_type: DataType,
    pub layout: Layout,
    // An input the program also writes.
    pub written: bool,
    // The output holding how many leading rows of this one a run wrote, when only a run decides that.
    pub count: Option<usize>,
}

// What a state tensor is in the graph it came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateKind {
    // An initializer, parameter, or buffer.
    Initializer,
    Constant,
    // A generator's `[seed, next block]`, which each run advances by `blocks`.
    Generator { blocks: u64 },
}

// A tensor the program holds across runs, with generated functions that read and replace it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramState {
    // As C++ spells it.
    pub name: String,
    // As the graph spells it.
    pub graph_name: String,
    pub kind: StateKind,
    pub data_type: DataType,
    pub layout: Layout,
    // What the tensor holds when the program is created, over its layout's storage span.
    pub values: Vec<u8>,
}

// What runs a graph operation in a program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepProvider {
    Kernel,
    Fusion,
    Alias,
    Library(Library),
}

impl StepProvider {
    #[must_use]
    pub const fn of(library: Option<Library>) -> Self {
        match library {
            Some(library) => Self::Library(library),
            None => Self::Kernel,
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Kernel => "kernel",
            Self::Fusion => "fusion",
            Self::Alias => "alias",
            Self::Library(library) => library.name(),
        }
    }
}

// The provider that runs the operation producing a graph value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperationProvider {
    pub operation: Option<Operation>,
    pub value: usize,
    pub provider: StepProvider,
}

// One step of a program of several, named as C++ accepts.
#[derive(Debug, Clone)]
pub struct ProgramEntry {
    pub name: String,
    pub program: CudaProgram,
    // For each state tensor, the earlier entry and its tensor that hold the same values, which this one reads.
    pub shared: Vec<Option<(usize, usize)>>,
}

// Several graphs of one model as one program, such as a decoder's steps of different sizes, whose common state exists once.
#[derive(Debug, Clone)]
pub struct CudaProgramSet {
    pub entries: Vec<ProgramEntry>,
    // The paged cache of its `decode` step, when the set has a generation loop through that step, after its `encode` step if it has one.
    pub decoding: Option<PageLayout>,
}

// A whole graph as ordered steps over planned buffers, which `render` writes as C++.
#[derive(Debug, Clone)]
pub struct CudaProgram {
    pub target: GpuArchitecture,
    pub buffers: Vec<ProgramBuffer>,
    pub inputs: Vec<ProgramValue>,
    pub state: Vec<ProgramState>,
    pub outputs: Vec<ProgramValue>,
    pub steps: Vec<ProgramStep>,
    // Which provider runs each graph operation that has one.
    pub providers: Vec<OperationProvider>,
    // The operations that run as decompositions because no allowed provider runs them.
    pub decomposed: Vec<Operation>,
    // The kernels that use a schedule tuning recorded, with that schedule.
    pub tuned_kernels: Vec<String>,
    // How the program's planning read its tuning records.
    pub record_coverage: RecordCoverage,
    // Whether each tensor in `state` has functions that read and replace it.
    pub state_functions: bool,
    // Whether its kernels overlap the kernels they follow, which they make safe themselves.
    pub dependent_launch: bool,
    // Whether what its records force on its libraries is read from a table given at creation rather than written as literals, so a tuning candidate needs no build of its own.
    pub library_table: bool,
    // What the program's libraries may do to its numbers, as the policy stood at export.
    pub numerics: NumericalPolicy,
    // Where a decoder step's tokens fall in its paged cache, when the export writes a generation loop.
    pub decoding: Option<PageLayout>,
}
