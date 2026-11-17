//! Runtime compilation of CUDA source and retrieval of compilation artifacts.

mod cache;
mod context;
mod library;
mod options;
mod program;

use singe_core::LibraryVersion;
use singe_cuda_sys::nvrtc as sys;

pub use self::{
    cache::{CompiledModule, ModuleSource, compile_cached},
    library::library_path,
    options::{
        CompileOptions, CppDialect, FastCompileLevel, MacroDefinition, OptimizationInfoKind,
        TileCompilation, WarningAsErrorKind,
    },
    program::{CompilationArtifact, Header, OutputKind, Program, RawProgram},
};
use crate::{
    error::{Error, Result},
    try_nvrtc,
    utility::{to_c_ulong, to_usize},
};

singe_core::wrapper_status! {
    library = "nvrtc",
    error = Error::Nvrtc,
    raw = sys::nvrtcResult,
    success = NVRTC_SUCCESS,
    absence = singe_cuda_sys::take_absence,
    check = check_nvrtc,
    unknown = "UNKNOWN_NVRTC_STATUS",
    description = sys::nvrtcGetErrorString,
    /// NVRTC result code returned by compiler operations.
    pub enum Status: u32 {
        /// The operation completed successfully.
        Success = NVRTC_SUCCESS,
        /// NVRTC could not allocate the required memory.
        OutOfMemory = NVRTC_ERROR_OUT_OF_MEMORY,
        /// NVRTC could not create the program.
        ProgramCreationFailure = NVRTC_ERROR_PROGRAM_CREATION_FAILURE,
        /// An input argument is invalid.
        InvalidInput = NVRTC_ERROR_INVALID_INPUT,
        /// The program handle is invalid.
        InvalidProgram = NVRTC_ERROR_INVALID_PROGRAM,
        /// A compilation option is invalid.
        InvalidOption = NVRTC_ERROR_INVALID_OPTION,
        /// CUDA source compilation failed; the program log contains diagnostics.
        Compilation = NVRTC_ERROR_COMPILATION,
        /// An NVRTC built-in operation failed.
        BuiltinOperationFailure = NVRTC_ERROR_BUILTIN_OPERATION_FAILURE,
        /// A name expression cannot be added after compilation.
        NoNameExpressionsAfterCompilation = NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION,
        /// A lowered name was requested before compiling the program.
        NoLoweredNamesBeforeCompilation = NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION,
        /// The name expression is invalid.
        NameExpressionNotValid = NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID,
        /// An internal NVRTC error occurred.
        InternalError = NVRTC_ERROR_INTERNAL_ERROR,
        /// NVRTC could not write its compilation timing file.
        TimeFileWriteFailed = NVRTC_ERROR_TIME_FILE_WRITE_FAILED,
        /// No precompiled-header creation was attempted.
        /// PCH processing may not have been requested, or automatic PCH processing chose not to create a file.
        NoPchCreateAttempted = NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED,
        /// The precompiled-header heap was exhausted.
        /// Query the required size and enlarge the heap before retrying compilation with a new program.
        PchCreateHeapExhausted = NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED,
        /// An error prevented the precompiled-header file from being created.
        PchCreate = NVRTC_ERROR_PCH_CREATE,
        /// The operation was cancelled.
        Cancelled = NVRTC_ERROR_CANCELLED,
        /// NVRTC could not write its compilation time-trace file.
        TimeTraceFileWriteFailed = NVRTC_ERROR_TIME_TRACE_FILE_WRITE_FAILED,
    }
}

/// Returns the CUDA Runtime Compilation version.
///
/// # Errors
///
/// Returns an error if NVRTC cannot report its version.
pub fn version() -> Result<LibraryVersion> {
    let mut major = 0;
    let mut minor = 0;
    // SAFETY: `major` and `minor` are local outputs.
    unsafe {
        try_nvrtc!(sys::nvrtcVersion(&raw mut major, &raw mut minor))?;
    }
    LibraryVersion::from_properties(major, minor, 0)
        .ok_or_else(|| Error::out_of_range("nvrtc version"))
}

/// Returns the GPU architecture numbers supported by the loaded NVRTC library.
///
/// # Errors
///
/// Returns an error if NVRTC cannot report the supported architecture count or fill the architecture list.
pub fn supported_architectures() -> Result<Vec<i32>> {
    let mut count = 0;
    // SAFETY: `count` is a local output.
    unsafe {
        try_nvrtc!(sys::nvrtcGetNumSupportedArchs(&raw mut count))?;
    }

    let mut architectures = vec![0; to_usize(count, "NVRTC supported architecture count")?];
    if architectures.is_empty() {
        return Ok(Vec::new());
    }

    // SAFETY: `architectures` holds the `count` entries NVRTC reported.
    unsafe {
        try_nvrtc!(sys::nvrtcGetSupportedArchs(architectures.as_mut_ptr()))?;
    }
    Ok(architectures)
}

/// Returns the current size of the PCH heap.
///
/// # Errors
///
/// Returns an error if NVRTC cannot report the PCH heap size.
pub fn pch_heap_size() -> Result<usize> {
    let mut size = 0;
    // SAFETY: `size` is a local output.
    unsafe {
        try_nvrtc!(sys::nvrtcGetPCHHeapSize(&raw mut size))?;
    }
    to_usize(size, "NVRTC output byte count")
}

/// Sets the size of the PCH heap.
///
/// The requested size may be rounded up to a platform-dependent alignment, such as the page size.
/// If the PCH heap has already been allocated, NVRTC frees it and allocates a new heap.
///
/// # Errors
///
/// Returns an error if NVRTC rejects the requested PCH heap size.
pub fn set_pch_heap_size(size: usize) -> Result<()> {
    let size = to_c_ulong(size, "PCH heap size")?;
    // SAFETY: `nvrtcSetPCHHeapSize` takes only the size and reallocates NVRTC's own heap.
    unsafe { try_nvrtc!(sys::nvrtcSetPCHHeapSize(size)) }
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use std::sync::{Arc, atomic::AtomicBool};

    use super::*;
    use crate::{
        architecture::GpuArchitecture, device::Device, error::Result, jit::JitOptions,
        memory::DeviceMemory, module::LaunchConfig, stream::StreamOptions, testing,
    };

    fn current_device_sm_architecture() -> Result<GpuArchitecture> {
        let properties = Device::current()?.properties()?;
        Ok(match (properties.major, properties.minor) {
            (7, 5) => GpuArchitecture::Sm75,
            (8, 0) => GpuArchitecture::Sm80,
            (8, 6) => GpuArchitecture::Sm86,
            (8, 7) => GpuArchitecture::Sm87,
            (8, 9) => GpuArchitecture::Sm89,
            (9, 0) => GpuArchitecture::Sm90,
            (10, 0) => GpuArchitecture::Sm100,
            (10, 1) => GpuArchitecture::Sm101,
            (10, 3) => GpuArchitecture::Sm103,
            (12, 0) => GpuArchitecture::Sm120,
            (12, 1) => GpuArchitecture::Sm121,
            (major, minor) => panic!("unsupported device architecture sm_{major}{minor}"),
        })
    }

    #[test]
    fn test_version_is_available() {
        let version = version().unwrap();
        assert_ne!(version.major(), 0);
    }

    #[test]
    fn test_supported_architectures_are_sorted() {
        let architectures = supported_architectures().unwrap();
        assert!(!architectures.is_empty());
        assert!(
            architectures
                .windows(2)
                .all(|window| window[0] <= window[1])
        );
    }

    #[test]
    fn test_compile_options_build_expected_arguments() {
        let arguments = CompileOptions::new()
            .gpu_architecture(GpuArchitecture::Compute80)
            .device_debug(true)
            .generate_line_info(true)
            .define_macro(MacroDefinition::WithValue {
                name: "FOO",
                value: "42",
            })
            .include_path("include")
            .cpp_dialect(CppDialect::Cpp20)
            .warning_as_error(WarningAsErrorKind::Reorder)
            .diag_suppress(177)
            .raw_option("--custom-flag")
            .as_arguments();

        assert_eq!(
            arguments,
            vec![
                "--gpu-architecture=compute_80",
                "--device-debug",
                "--generate-line-info",
                "--define-macro=FOO=42",
                "--include-path=include",
                "--std=c++20",
                "--warning-as-error=reorder",
                "--diag-suppress=177",
                "--custom-flag",
            ]
        );
    }

    #[test]
    fn test_compiles_to_ptx() {
        let program = Program::from_source(
            r#"
            extern "C" __global__ void saxpy(float a, const float* x, const float* y, float* out, size_t n) {
                size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
                if (tid < n) {
                    out[tid] = a * x[tid] + y[tid];
                }
            }
            "#,
        )
        .with_name("saxpy.cu");
        let options = CompileOptions::new()
            .gpu_architecture(GpuArchitecture::Compute80)
            .generate_line_info(true);
        program.compile(&options).unwrap();

        let ptx = program.ptx_string().unwrap();
        assert!(ptx.contains(".visible .entry saxpy"));
    }

    #[test]
    fn test_compile_with_cancel_flag_succeeds_when_not_cancelled() {
        let _device = testing::bootstrap().unwrap();
        let cancel = Arc::new(AtomicBool::new(false));
        let program = Program::from_source(
            r#"
            extern "C" __global__ void noop() {}
            "#,
        )
        .with_name("noop.cu");
        let options = CompileOptions::new()
            .gpu_architecture(GpuArchitecture::Compute80)
            .cancel_flag(cancel);

        program.compile(&options).unwrap();
        assert!(program.ptx_string().unwrap().contains("noop"));
    }

    #[test]
    fn test_clear_flow_callback_is_allowed_before_compilation() {
        let program =
            Program::from_source("extern \"C\" __global__ void noop() {}").with_name("noop.cu");
        program.clear_flow_callback().unwrap();
    }

    #[test]
    fn test_cubin_artifact_loads_as_module() {
        let test_device = testing::bootstrap().unwrap();
        let context = Arc::clone(test_device.context());
        let program = Program::from_source(
            r#"
            extern "C" __global__ void saxpy(float a, const float* x, const float* y, float* out, size_t n) {
                size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
                if (tid < n) {
                    out[tid] = a * x[tid] + y[tid];
                }
            }
            "#,
        )
        .with_name("saxpy_module.cu");
        let architecture = current_device_sm_architecture().unwrap();
        let options = CompileOptions::new().gpu_architecture(architecture);
        program.compile(&options).unwrap();

        let module = context
            .load_nvrtc_module(&program, OutputKind::Cubin, JitOptions::default())
            .unwrap();
        assert!(module.function("saxpy").is_ok());
    }

    #[test]
    fn test_cubin_artifact_loads_as_module_with_jit_options() {
        let test_device = testing::bootstrap().unwrap();
        let context = Arc::clone(test_device.context());
        let program = Program::from_source(
            r#"
            extern "C" __global__ void noop() {}
            "#,
        )
        .with_name("noop_module_jit.cu");
        let architecture = current_device_sm_architecture().unwrap();
        let options = CompileOptions::new().gpu_architecture(architecture);
        program.compile(&options).unwrap();

        let mut info_log = [0u8; 1024];
        let mut error_log = [0u8; 1024];
        let jit_options = JitOptions::default()
            .with_generate_line_info(true)
            .with_info_log(&mut info_log)
            .with_error_log(&mut error_log);

        let module = context
            .load_nvrtc_module(&program, OutputKind::Cubin, jit_options)
            .unwrap();
        assert!(module.function("noop").is_ok());
    }

    #[test]
    fn test_compiles_loads_launches_and_reads_back_results() {
        let test_device = testing::bootstrap().unwrap();
        let context = Arc::clone(test_device.context());

        let input = vec![1.0f32, 2.0, 3.5, -4.0, 8.25];
        let mut output = vec![0.0f32; input.len()];
        let scalar = 2.5f32;
        let input_device = DeviceMemory::from_slice(&context, &input).unwrap();
        let output_device = DeviceMemory::<f32>::zeroed(&context, output.len()).unwrap();
        let length = input.len();

        let program = Program::from_source(
            r#"
            extern "C" __global__ void scale_add(const float* input, float* output, float alpha, size_t len) {
                size_t i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < len) {
                    output[i] = input[i] * alpha + 1.0f;
                }
            }
            "#,
        )
        .with_name("scale_add.cu");
        let architecture = current_device_sm_architecture().unwrap();
        let compile_options = CompileOptions::new().gpu_architecture(architecture);
        program.compile(&compile_options).unwrap();

        let module = context
            .load_nvrtc_module(&program, OutputKind::Cubin, JitOptions::default())
            .unwrap();
        let function = module.function("scale_add").unwrap();

        let config = LaunchConfig::for_1d_grid(input.len(), 128);
        let input_ptr = input_device.as_ptr();
        let mut output_ptr = output_device.as_mut_ptr();

        let stream = context.create_stream(StreamOptions::default()).unwrap();
        // SAFETY: the compiled declaration matches these arguments, both allocations cover `length` elements, and they stay exclusively available until the stream synchronizes.
        unsafe {
            function.launch_async(
                &config,
                (&input_ptr, &mut output_ptr, &scalar, &length),
                &stream,
            )
        }
        .unwrap();
        stream.synchronize().unwrap();
        output_device.copy_to_host(&mut output).unwrap();

        let expected = input
            .iter()
            .map(|value| value * scalar + 1.0)
            .collect::<Vec<_>>();
        assert_eq!(output, expected);
    }

    #[test]
    fn test_cubin_artifact_loads_as_library() {
        let test_device = testing::bootstrap().unwrap();
        let context = Arc::clone(test_device.context());
        let program = Program::from_source(
            r#"
            extern "C" __global__ void noop() {}
            "#,
        )
        .with_name("noop_library.cu");
        let architecture = current_device_sm_architecture().unwrap();
        let options = CompileOptions::new().gpu_architecture(architecture);
        program.compile(&options).unwrap();

        let library = context
            .load_nvrtc_library(&program, OutputKind::Cubin, JitOptions::default())
            .unwrap();
        assert!(library.kernel_count().unwrap() >= 1);
    }

    #[test]
    fn test_lto_ir_artifact_is_available_when_requested() {
        let program = Program::from_source(
            r#"
            extern "C" __global__ void noop() {}
            "#,
        )
        .with_name("noop_lto.cu");
        let options = CompileOptions::new().dlink_time_optimization(true);
        program.compile(&options).unwrap();

        let artifact = program.artifact(OutputKind::LtoIr).unwrap();
        assert!(!artifact.as_bytes().is_empty());
        let ptx = program.artifact(OutputKind::Ptx).unwrap();
        assert!(!ptx.as_bytes().is_empty());
    }

    #[test]
    fn test_optix_ir_artifact_is_available_when_requested() {
        let program = Program::from_source(
            r#"
            extern "C" __global__ void noop() {}
            "#,
        )
        .with_name("noop_optix.cu");
        let options = CompileOptions::new().optix_ir(true);
        program.compile(&options).unwrap();

        let artifact = program.artifact(OutputKind::OptixIr).unwrap();
        assert!(!artifact.as_bytes().is_empty());
        let ptx = program.artifact(OutputKind::Ptx).unwrap();
        assert!(!ptx.as_bytes().is_empty());
    }

    #[test]
    fn test_cubin_artifact_requires_real_architecture() {
        let program = Program::from_source(
            r#"
            extern "C" __global__ void noop() {}
            "#,
        )
        .with_name("noop_cubin.cu");
        let options = CompileOptions::new().gpu_architecture(GpuArchitecture::Compute80);
        program.compile(&options).unwrap();

        assert!(program.artifact(OutputKind::Cubin).is_err());
    }
}
