// The NVIDIA libraries the `*-sys` crates load, by the file names each platform gives them, so a crate names its library and never a file.
// Each entry states its library's major version once, and every file name and doc comment is built from it, so a version bump is a one-token change.
// The major version is that of the bindings its crate is generated from, or the soname's version for a library without one, such as the driver.
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

// Builds a file name from literal parts, where the token `major` stands for the library's major version.
macro_rules! file_name {
    ($major:literal; $($part:tt)+) => {
        concat!($(file_name!(@part $major; $part)),+)
    };
    (@part $major:literal; major) => {
        $major
    };
    (@part $major:literal; $part:literal) => {
        $part
    };
}

// The doc comment line that introduces a platform's files, or says it has none.
macro_rules! platform_files {
    ($platform:literal;) => {
        concat!("It has no files on ", $platform, ".")
    };
    ($platform:literal; $($file:tt)+) => {
        concat!("Its files on ", $platform, ":")
    };
}

// Declares one constructor per library from its name, its major version, and the parts of its file names on Linux and Windows.
macro_rules! libraries {
    ($(
        $(#[doc = $doc:literal])*
        $constructor:ident => $name:literal, major $major:literal,
        linux [$([$($linux:tt)+]),* $(,)?],
        windows [$([$($windows:tt)+]),* $(,)?];
    )+) => {
        impl Library {
            $(
                #[doc = concat!("Creates the library for ", $name, ", whose files carry major version ", $major, ".")]
                $(#[doc = ""] #[doc = $doc])*
                #[doc = ""]
                #[doc = platform_files!("Linux"; $([$($linux)+])*)]
                #[doc = ""]
                $(#[doc = concat!("- `", file_name!($major; $($linux)+), "`")])*
                #[doc = ""]
                #[doc = platform_files!("Windows"; $([$($windows)+])*)]
                #[doc = ""]
                $(#[doc = concat!("- `", file_name!($major; $($windows)+), "`")])*
                #[must_use]
                pub const fn $constructor() -> Self {
                    Self::new(
                        $name,
                        files(
                            &[$(file_name!($major; $($linux)+)),*],
                            &[$(file_name!($major; $($windows)+)),*],
                        ),
                    )
                }
            )+
        }
    };
}

libraries! {
    cuda_driver => "the CUDA driver", major "1",
        linux [["libcuda.so." major]],
        windows [["nvcuda.dll"]];

    cuda_runtime => "the CUDA runtime", major "13",
        linux [["libcudart.so." major]],
        windows [["cudart64_" major ".dll"]];

    nvrtc => "NVRTC", major "13",
        linux [["libnvrtc.so." major]],
        windows [["nvrtc64_" major "0_0.dll"]];

    nvjitlink => "nvJitLink", major "13",
        linux [["libnvJitLink.so." major]],
        windows [["nvJitLink_" major "0_0.dll"]];

    nvvm => "NVVM", major "4",
        linux [["libnvvm.so." major]],
        windows [["nvvm64_" major "0_0.dll"]];

    /// NVTX's headers implement it inline; this library exports the same functions for languages that cannot use them.
    nvtx => "NVTX", major "1",
        linux [["libnvtx3interop.so." major]],
        windows [["nvtx3interop.dll"]];

    nvml => "NVML", major "1",
        linux [["libnvidia-ml.so." major]],
        windows [["nvml.dll"]];

    cublas => "cuBLAS", major "13",
        linux [["libcublas.so." major]],
        windows [["cublas64_" major ".dll"]];

    cublas_lt => "cuBLASLt", major "13",
        linux [["libcublasLt.so." major]],
        windows [["cublasLt64_" major ".dll"]];

    /// cuDNN loads its own sub-libraries.
    cudnn => "cuDNN", major "9",
        linux [["libcudnn.so." major]],
        windows [["cudnn64_" major ".dll"]];

    cudss => "cuDSS", major "1",
        linux [["libcudss.so." major]],
        windows [["cudss64_" major ".dll"]];

    cufft => "cuFFT", major "12",
        linux [["libcufft.so." major]],
        windows [["cufft64_" major ".dll"]];

    /// cuFile has no Windows release.
    cufile => "cuFile", major "0",
        linux [["libcufile.so." major]],
        windows [];

    /// Windows' CUPTI carries its release in its name, as `cupti64_2025.4.0.dll`, which no fixed name matches.
    cupti => "CUPTI", major "13",
        linux [["libcupti.so." major]],
        windows [];

    curand => "cuRAND", major "10",
        linux [["libcurand.so." major]],
        windows [["curand64_" major ".dll"]];

    cusolver => "cuSOLVER", major "12",
        linux [["libcusolver.so." major]],
        windows [["cusolver64_" major ".dll"]];

    cusparse => "cuSPARSE", major "12",
        linux [["libcusparse.so." major]],
        windows [["cusparse64_" major ".dll"]];

    cutensor => "cuTENSOR", major "2",
        linux [["libcutensor.so." major]],
        windows [["cutensor.dll"]];

    cutensor_mg => "cuTENSORMg", major "2",
        linux [["libcutensorMg.so." major]],
        windows [["cutensorMg.dll"]];

    /// cuTENSORMp runs over NCCL, which is Linux-only.
    cutensor_mp => "cuTENSORMp", major "2",
        linux [["libcutensorMp.so." major]],
        windows [];

    nccl => "NCCL", major "2",
        linux [["libnccl.so." major]],
        windows [];

    /// NPP's functions are spread over its core library and ten by domain.
    npp => "NPP", major "13",
        linux [
            ["libnppc.so." major],
            ["libnppial.so." major],
            ["libnppicc.so." major],
            ["libnppidei.so." major],
            ["libnppif.so." major],
            ["libnppig.so." major],
            ["libnppim.so." major],
            ["libnppist.so." major],
            ["libnppisu.so." major],
            ["libnppitc.so." major],
            ["libnpps.so." major],
        ],
        windows [
            ["nppc64_" major ".dll"],
            ["nppial64_" major ".dll"],
            ["nppicc64_" major ".dll"],
            ["nppidei64_" major ".dll"],
            ["nppif64_" major ".dll"],
            ["nppig64_" major ".dll"],
            ["nppim64_" major ".dll"],
            ["nppist64_" major ".dll"],
            ["nppisu64_" major ".dll"],
            ["nppitc64_" major ".dll"],
            ["npps64_" major ".dll"],
        ];

    nvcomp => "nvCOMP", major "6",
        linux [["libnvcomp.so." major]],
        windows [["nvcomp64_" major ".dll"]];

    nvimagecodec => "nvImageCodec", major "0",
        linux [["libnvimgcodec.so." major]],
        windows [["nvimgcodec_" major ".dll"]];

    nvjpeg => "nvJPEG", major "13",
        linux [["libnvjpeg.so." major]],
        windows [["nvjpeg64_" major ".dll"]];

    nvjpeg2k => "nvJPEG2000", major "0",
        linux [["libnvjpeg2k.so." major]],
        windows [["nvjpeg2k_" major ".dll"]];

    nvtiff => "nvTIFF", major "0",
        linux [["libnvtiff.so." major]],
        windows [["nvtiff_" major ".dll"]];
}
