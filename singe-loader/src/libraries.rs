// The NVIDIA libraries the `*-sys` crates load, by the file names each platform gives them, so a crate names its library and never a file.
// Each file name carries the major version of the bindings its crate is generated from.
// The Windows names follow NVIDIA's naming and are unverified until a Windows machine loads them; a library NVIDIA ships only for Linux has none.

use crate::Library;

// The file names of the current platform: Linux's, Windows', or none elsewhere, where loading fails at once.
const fn files(
    linux: &'static [&'static str],
    windows: &'static [&'static str],
) -> &'static [&'static str] {
    if cfg!(target_os = "linux") {
        linux
    } else if cfg!(target_os = "windows") {
        windows
    } else {
        &[]
    }
}

impl Library {
    #[must_use]
    pub const fn cuda_driver() -> Self {
        Self::new("the CUDA driver", files(&["libcuda.so.1"], &["nvcuda.dll"]))
    }

    #[must_use]
    pub const fn cuda_runtime() -> Self {
        Self::new(
            "the CUDA runtime",
            files(&["libcudart.so.13"], &["cudart64_13.dll"]),
        )
    }

    #[must_use]
    pub const fn nvrtc() -> Self {
        Self::new("NVRTC", files(&["libnvrtc.so.13"], &["nvrtc64_130_0.dll"]))
    }

    #[must_use]
    pub const fn nvjitlink() -> Self {
        Self::new(
            "nvJitLink",
            files(&["libnvJitLink.so.13"], &["nvJitLink_130_0.dll"]),
        )
    }

    #[must_use]
    pub const fn nvvm() -> Self {
        Self::new("NVVM", files(&["libnvvm.so.4"], &["nvvm64_40_0.dll"]))
    }

    // NVTX's headers implement it inline; this library exports the same functions for languages that cannot use them.
    #[must_use]
    pub const fn nvtx() -> Self {
        Self::new(
            "NVTX",
            files(&["libnvtx3interop.so.1"], &["nvtx3interop.dll"]),
        )
    }

    #[must_use]
    pub const fn nvml() -> Self {
        Self::new("NVML", files(&["libnvidia-ml.so.1"], &["nvml.dll"]))
    }

    #[must_use]
    pub const fn cublas() -> Self {
        Self::new("cuBLAS", files(&["libcublas.so.13"], &["cublas64_13.dll"]))
    }

    #[must_use]
    pub const fn cublas_lt() -> Self {
        Self::new(
            "cuBLASLt",
            files(&["libcublasLt.so.13"], &["cublasLt64_13.dll"]),
        )
    }

    // cuDNN loads its own sub-libraries.
    #[must_use]
    pub const fn cudnn() -> Self {
        Self::new("cuDNN", files(&["libcudnn.so.9"], &["cudnn64_9.dll"]))
    }

    #[must_use]
    pub const fn cudss() -> Self {
        Self::new("cuDSS", files(&["libcudss.so.0"], &["cudss64_0.dll"]))
    }

    #[must_use]
    pub const fn cufft() -> Self {
        Self::new("cuFFT", files(&["libcufft.so.12"], &["cufft64_12.dll"]))
    }

    // GPUDirect Storage is Linux-only.
    #[must_use]
    pub const fn cufile() -> Self {
        Self::new("cuFile", files(&["libcufile.so.0"], &[]))
    }

    // Windows' CUPTI carries its release in its name, as `cupti64_2025.4.0.dll`, which no fixed name matches.
    #[must_use]
    pub const fn cupti() -> Self {
        Self::new("CUPTI", files(&["libcupti.so.13"], &[]))
    }

    #[must_use]
    pub const fn curand() -> Self {
        Self::new("cuRAND", files(&["libcurand.so.10"], &["curand64_10.dll"]))
    }

    #[must_use]
    pub const fn cusolver() -> Self {
        Self::new(
            "cuSOLVER",
            files(&["libcusolver.so.12"], &["cusolver64_12.dll"]),
        )
    }

    #[must_use]
    pub const fn cusparse() -> Self {
        Self::new(
            "cuSPARSE",
            files(&["libcusparse.so.12"], &["cusparse64_12.dll"]),
        )
    }

    #[must_use]
    pub const fn cutensor() -> Self {
        Self::new("cuTENSOR", files(&["libcutensor.so.2"], &["cutensor.dll"]))
    }

    #[must_use]
    pub const fn cutensor_mg() -> Self {
        Self::new(
            "cuTENSORMg",
            files(&["libcutensorMg.so.2"], &["cutensorMg.dll"]),
        )
    }

    // cuTENSORMp runs over NCCL, which is Linux-only.
    #[must_use]
    pub const fn cutensor_mp() -> Self {
        Self::new("cuTENSORMp", files(&["libcutensorMp.so.2"], &[]))
    }

    // NCCL is Linux-only.
    #[must_use]
    pub const fn nccl() -> Self {
        Self::new("NCCL", files(&["libnccl.so.2"], &[]))
    }

    // NPP's functions are spread over its core library and ten by domain.
    #[must_use]
    pub const fn npp() -> Self {
        Self::new(
            "NPP",
            files(
                &[
                    "libnppc.so.13",
                    "libnppial.so.13",
                    "libnppicc.so.13",
                    "libnppidei.so.13",
                    "libnppif.so.13",
                    "libnppig.so.13",
                    "libnppim.so.13",
                    "libnppist.so.13",
                    "libnppisu.so.13",
                    "libnppitc.so.13",
                    "libnpps.so.13",
                ],
                &[
                    "nppc64_13.dll",
                    "nppial64_13.dll",
                    "nppicc64_13.dll",
                    "nppidei64_13.dll",
                    "nppif64_13.dll",
                    "nppig64_13.dll",
                    "nppim64_13.dll",
                    "nppist64_13.dll",
                    "nppisu64_13.dll",
                    "nppitc64_13.dll",
                    "npps64_13.dll",
                ],
            ),
        )
    }

    #[must_use]
    pub const fn nvcomp() -> Self {
        Self::new("nvCOMP", files(&["libnvcomp.so.6"], &["nvcomp64_6.dll"]))
    }

    #[must_use]
    pub const fn nvimagecodec() -> Self {
        Self::new(
            "nvImageCodec",
            files(&["libnvimgcodec.so.0"], &["nvimgcodec_0.dll"]),
        )
    }

    #[must_use]
    pub const fn nvjpeg() -> Self {
        Self::new("nvJPEG", files(&["libnvjpeg.so.13"], &["nvjpeg64_13.dll"]))
    }

    #[must_use]
    pub const fn nvjpeg2k() -> Self {
        Self::new(
            "nvJPEG2000",
            files(&["libnvjpeg2k.so.0"], &["nvjpeg2k_0.dll"]),
        )
    }

    #[must_use]
    pub const fn nvtiff() -> Self {
        Self::new("nvTIFF", files(&["libnvtiff.so.0"], &["nvtiff_0.dll"]))
    }
}
