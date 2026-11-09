use crate::backend::registry::{CudaGraphProvider, Operation};

// What cuDNN may run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum CudnnFeature {
    #[default]
    All,
    // Two- and three-dimensional convolutions, with a bias and activation.
    Convolution,
    // Matrix products, with an add and activation.
    Matmul,
    // Scaled dot-product attention, paged or not.
    Attention,
}

// What cuBLAS may run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum CublasFeature {
    #[default]
    All,
    // Matrix products that fit one strided batched GEMM.
    Matmul,
}

// What cuBLASLt may run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum CublasLtFeature {
    #[default]
    All,
    // Matrix products, with an add and activation.
    Matmul,
    // Quantized `scaled_mm`.
    ScaledMatmul,
}

// What cuFFT may run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum CufftFeature {
    #[default]
    All,
    Rfft,
    Irfft,
}

// What cuTENSOR may run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum CutensorFeature {
    #[default]
    All,
    Contraction,
}

// One library, and what it may run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LibraryFeature {
    Cudnn(CudnnFeature),
    Cublas(CublasFeature),
    CublasLt(CublasLtFeature),
    Cufft(CufftFeature),
    Cutensor(CutensorFeature),
}

impl LibraryFeature {
    // Every library with all it may run.
    pub const ALL: [Self; 5] = [
        Self::Cudnn(CudnnFeature::All),
        Self::Cublas(CublasFeature::All),
        Self::CublasLt(CublasLtFeature::All),
        Self::Cufft(CufftFeature::All),
        Self::Cutensor(CutensorFeature::All),
    ];

    // Every narrower feature, for reading a set back.
    const NARROW: [Self; 9] = [
        Self::Cudnn(CudnnFeature::Convolution),
        Self::Cudnn(CudnnFeature::Matmul),
        Self::Cudnn(CudnnFeature::Attention),
        Self::Cublas(CublasFeature::Matmul),
        Self::CublasLt(CublasLtFeature::Matmul),
        Self::CublasLt(CublasLtFeature::ScaledMatmul),
        Self::Cufft(CufftFeature::Rfft),
        Self::Cufft(CufftFeature::Irfft),
        Self::Cutensor(CutensorFeature::Contraction),
    ];

    // The provider that calls this library.
    #[must_use]
    pub const fn provider(self) -> CudaGraphProvider {
        match self {
            Self::Cudnn(_) => CudaGraphProvider::CudnnBackend,
            Self::Cublas(_) => CudaGraphProvider::Cublas,
            Self::CublasLt(_) => CudaGraphProvider::CublasLt,
            Self::Cufft(_) => CudaGraphProvider::Cufft,
            Self::Cutensor(_) => CudaGraphProvider::Cutensor,
        }
    }

    // The narrowest feature that lets `provider` run `operation`, or `None` when no feature names that pair.
    #[must_use]
    pub const fn of(provider: CudaGraphProvider, operation: Operation) -> Option<Self> {
        Some(match (provider, operation) {
            (CudaGraphProvider::CudnnBackend, Operation::Conv2d | Operation::Conv3d) => {
                Self::Cudnn(CudnnFeature::Convolution)
            }
            (CudaGraphProvider::CudnnBackend, Operation::MatMul) => {
                Self::Cudnn(CudnnFeature::Matmul)
            }
            (
                CudaGraphProvider::CudnnBackend,
                Operation::ScaledDotProductAttention
                | Operation::PagedAttention
                | Operation::PagedAttentionOutput
                | Operation::PagedAttentionLse,
            ) => Self::Cudnn(CudnnFeature::Attention),
            (CudaGraphProvider::Cublas, Operation::MatMul) => Self::Cublas(CublasFeature::Matmul),
            (CudaGraphProvider::CublasLt, Operation::MatMul) => {
                Self::CublasLt(CublasLtFeature::Matmul)
            }
            (CudaGraphProvider::CublasLt, Operation::ScaledMatMul) => {
                Self::CublasLt(CublasLtFeature::ScaledMatmul)
            }
            (CudaGraphProvider::Cufft, Operation::Rfft) => Self::Cufft(CufftFeature::Rfft),
            (CudaGraphProvider::Cufft, Operation::Irfft) => Self::Cufft(CufftFeature::Irfft),
            (CudaGraphProvider::Cutensor, Operation::Contraction) => {
                Self::Cutensor(CutensorFeature::Contraction)
            }
            _ => return None,
        })
    }

    // The bits of a set this feature takes, which `All` makes those of every feature of its library.
    pub(crate) const fn bits(self) -> u16 {
        match self {
            Self::Cudnn(CudnnFeature::Convolution) => 1 << 0,
            Self::Cudnn(CudnnFeature::Matmul) => 1 << 1,
            Self::Cudnn(CudnnFeature::Attention) => 1 << 2,
            Self::Cudnn(CudnnFeature::All) => 0b111,
            Self::Cublas(CublasFeature::Matmul | CublasFeature::All) => 1 << 3,
            Self::CublasLt(CublasLtFeature::Matmul) => 1 << 4,
            Self::CublasLt(CublasLtFeature::ScaledMatmul) => 1 << 5,
            Self::CublasLt(CublasLtFeature::All) => 0b11 << 4,
            Self::Cufft(CufftFeature::Rfft) => 1 << 6,
            Self::Cufft(CufftFeature::Irfft) => 1 << 7,
            Self::Cufft(CufftFeature::All) => 0b11 << 6,
            Self::Cutensor(CutensorFeature::Contraction | CutensorFeature::All) => 1 << 8,
        }
    }

    // How `features` reads for a person: this library's `All` when it holds all of it, or else the narrower features it holds.
    pub(crate) fn held_in(features: u16) -> Vec<Self> {
        let mut held = Vec::new();
        for all in Self::ALL {
            if features & all.bits() == all.bits() {
                held.push(all);
            } else {
                held.extend(Self::NARROW.into_iter().filter(|feature| {
                    feature.provider() as u16 == all.provider() as u16
                        && features & feature.bits() != 0
                }));
            }
        }
        held
    }
}

impl From<CudnnFeature> for LibraryFeature {
    fn from(feature: CudnnFeature) -> Self {
        Self::Cudnn(feature)
    }
}

impl From<CublasFeature> for LibraryFeature {
    fn from(feature: CublasFeature) -> Self {
        Self::Cublas(feature)
    }
}

impl From<CublasLtFeature> for LibraryFeature {
    fn from(feature: CublasLtFeature) -> Self {
        Self::CublasLt(feature)
    }
}

impl From<CufftFeature> for LibraryFeature {
    fn from(feature: CufftFeature) -> Self {
        Self::Cufft(feature)
    }
}

impl From<CutensorFeature> for LibraryFeature {
    fn from(feature: CutensorFeature) -> Self {
        Self::Cutensor(feature)
    }
}
