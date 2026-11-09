//! Compiles generated kernels to cubin for the device and keeps them in the runtime's bounded cache.

use std::{
    collections::HashSet,
    panic,
    sync::atomic::{AtomicUsize, Ordering},
    thread,
};

use sha2::{Digest, Sha256};
use singe_cuda::{
    architecture::GpuArchitecture,
    module::{KernelFunction, ModuleImage},
    nvrtc::CompileOptions,
};

use crate::{
    backend::cuda::{
        codegen::{
            disk_cache::{self, CubinOrigin},
            kernel::{CompileRequirements, KernelArtifact, KernelCode},
            render::KernelOutput,
        },
        runtime::CudaRuntime,
    },
    error::{Error, Result},
    graph::{ComputePrecision, NumericalPolicy},
    utility::{Locked, to_i32},
};

/// The most host threads that compile generated kernels at once; NVRTC contends beyond this on the development machine.
const COMPILE_THREADS: usize = 8;

impl CudaRuntime {
    /// Compiles and loads `kernel`, caching it by its code and compile options.
    ///
    /// The kernel compiles to cubin for this device's architecture, with the options the current [`ComputePrecision`] selects, through the kernel cache directory when one is enabled.
    /// Compilation runs without holding the cache lock, so threads can compile different kernels at once.
    ///
    /// # Errors
    ///
    /// Returns an error if the code fails to compile or load, or a cache cannot be read or written.
    pub(crate) fn kernel(&self, kernel: &KernelArtifact) -> Result<KernelFunction> {
        self.kernel_with(kernel, NumericalPolicy::current().precision)
    }

    /// Compiles the kernels that are not cached yet on up to [`COMPILE_THREADS`] host threads, under the current [`ComputePrecision`].
    ///
    /// NVRTC takes 8 to 50 ms per kernel, so a graph's kernels compile together instead of one after another as lowering reaches them.
    /// Kernels NVRTC has not compiled before compile about three times faster on eight threads; kernels its driver cache holds compile slightly slower in parallel, because cache hits serialize.
    ///
    /// # Errors
    ///
    /// Returns an error if a kernel fails to compile or load.
    pub(crate) fn prepare_kernels<'a>(
        &self,
        kernels: impl IntoIterator<Item = &'a KernelArtifact>,
    ) -> Result<()> {
        // Worker threads do not inherit the thread-local numerical policy, so it is read here.
        let precision = NumericalPolicy::current().precision;
        let mut seen = HashSet::new();
        let kernels = kernels
            .into_iter()
            .filter(|kernel| seen.insert(&kernel.code))
            // Kernels planned for another architecture fail when they launch, which names the mismatch.
            .filter(|kernel| {
                kernel
                    .target
                    .is_none_or(|target| target == self.architecture())
            })
            .collect::<Vec<_>>();
        let threads = thread::available_parallelism()
            .map_or(1, usize::from)
            .min(COMPILE_THREADS)
            .min(kernels.len());
        let next = AtomicUsize::new(0);
        thread::scope(|scope| {
            let workers = (0..threads)
                .map(|_| {
                    scope.spawn(|| {
                        while let Some(kernel) = kernels.get(next.fetch_add(1, Ordering::Relaxed)) {
                            self.kernel_with(kernel, precision)?;
                        }
                        Ok(())
                    })
                })
                .collect::<Vec<_>>();
            workers.into_iter().try_for_each(|worker| {
                worker
                    .join()
                    .unwrap_or_else(|payload| panic::resume_unwind(payload))
            })
        })
    }

    fn kernel_with(
        &self,
        kernel: &KernelArtifact,
        precision: ComputePrecision,
    ) -> Result<KernelFunction> {
        let name = kernel.name.as_str();
        // A kernel planned for another architecture renders, but this device cannot run it.
        if let Some(target) = kernel.target
            && target != self.architecture()
        {
            return Err(Error::unsupported_operation_with_details(
                "cuda kernel launch",
                format!(
                    "kernel `{name}` is planned for {target}, but the device is {}",
                    self.architecture()
                ),
            ));
        }

        let requirements = kernel.requirements;
        if let Some((major, minor)) = requirements.minimum_capability
            && self.architecture().compute_capability() < (major, minor)
        {
            return Err(Error::unsupported_operation_with_details(
                "cuda kernel launch",
                format!(
                    "kernel `{name}` needs compute capability {major}.{minor}, but the device is {}",
                    self.architecture()
                ),
            ));
        }

        let options = kernel_compile_options(self.architecture(), requirements, precision)?;
        // The options join a CUDA C++ key, so exact and relaxed builds of one source coexist; Tile IR has no such options.
        let input = CompilerInput::create(kernel)?;
        let key = match &input {
            CompilerInput::Nvrtc(source) => {
                format!("{}\n{source}", options.as_arguments().join(" "))
            }
            CompilerInput::Tileiras(bytecode) => {
                format!("tile\n{}", hex::encode(Sha256::digest(bytecode)))
            }
        };
        if let Some(function) = self
            .generated_kernels()
            .locked("cuda runtime cache")?
            .get(&key)
        {
            return Ok(function.clone());
        }

        self.context().bind()?;
        let architecture = self.architecture().to_string();
        let compile = || match &input {
            CompilerInput::Nvrtc(source) => disk_cache::cubin(name, source, &options),
            CompilerInput::Tileiras(bytecode) => {
                disk_cache::tile_cubin(name, bytecode, &architecture)
            }
        };
        let (cubin, origin) = compile()?;
        let module = match self.context().load_module(&ModuleImage::from_vec(cubin)) {
            Ok(module) => module,
            // A damaged cache entry compiles again.
            Err(_) if origin == CubinOrigin::Cached => {
                match &input {
                    CompilerInput::Nvrtc(source) => disk_cache::forget(name, source, &options)?,
                    CompilerInput::Tileiras(bytecode) => {
                        disk_cache::forget_tile(name, bytecode, &architecture)?;
                    }
                }

                let (cubin, _) = compile()?;
                self.context().load_module(&ModuleImage::from_vec(cubin))?
            }
            Err(error) => return Err(error.into()),
        };
        let function = module.function(name)?;
        // Kernels whose blocks take more than the default 48 KiB of dynamic shared memory must say so before they launch.
        if kernel.launch.shared_bytes > 48 * 1024 {
            function.set_max_dynamic_shared_memory_bytes(to_i32(
                kernel.launch.shared_bytes,
                "cuda kernel shared memory",
            )?)?;
        }

        let mut cache = self.generated_kernels().locked("cuda runtime cache")?;
        if let Some(function) = cache.get(&key) {
            return Ok(function.clone());
        }

        // Executables retain their native module independently of this bounded cache.
        cache.insert(key, function.clone());
        drop(cache);
        Ok(function)
    }
}

// The options CUDA C++ kernels compile with for `architecture` under `precision`, which NVRTC and `nvcc` both take, so a program rounds as Singe does.
//
// Exact kernels use IEEE division and square root and no fused multiply-add, so they round as the CPU reference does.
pub fn kernel_compile_options(
    architecture: GpuArchitecture,
    requirements: CompileRequirements,
    precision: ComputePrecision,
) -> Result<CompileOptions<'static>> {
    let gpu_architecture = Some(if requirements.architecture_specific {
        architecture_specific(architecture)?
    } else {
        architecture
    });
    if precision == ComputePrecision::Relaxed {
        return Ok(CompileOptions {
            gpu_architecture,
            use_fast_math: true,
            ..CompileOptions::default()
        });
    }
    Ok(CompileOptions {
        gpu_architecture,
        fmad: Some(false),
        flush_to_zero: Some(false),
        precise_division: Some(true),
        precise_square_root: Some(true),
        ..CompileOptions::default()
    })
}

// What the compiler of a kernel's code reads: NVRTC the text of CUDA C++, `tileiras` Tile IR bytecode.
enum CompilerInput<'a> {
    Nvrtc(String),
    Tileiras(&'a [u8]),
}

impl<'a> CompilerInput<'a> {
    fn create(kernel: &'a KernelArtifact) -> Result<Self> {
        match &kernel.code {
            // Rendering rejects source in a language other backends compile, which reaches no CUDA compiler.
            KernelCode::Source(_) => Ok(Self::Nvrtc(kernel.render(KernelOutput::Nvrtc)?)),
            KernelCode::TileIr(bytecode) => Ok(Self::Tileiras(bytecode)),
        }
    }
}

// The architecture-specific variant of `architecture`, such as `sm_120a` for `sm_120`, whose instructions only that architecture runs.
pub fn architecture_specific(architecture: GpuArchitecture) -> Result<GpuArchitecture> {
    let name = architecture.to_string();
    let specific = if name.ends_with(['a', 'f']) {
        name
    } else {
        format!(
            "{}a",
            name.trim_end_matches(|character: char| character.is_ascii_alphabetic())
        )
    };
    specific.parse().map_err(|_| {
        Error::unsupported_operation_with_details(
            "cuda kernel compilation",
            format!("{architecture} has no architecture-specific target"),
        )
    })
}
