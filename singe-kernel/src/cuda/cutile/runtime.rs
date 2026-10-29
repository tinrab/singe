//! Host integration for cuTile kernels loaded in the launch stream's actual context.

pub use ::cutile::*;

pub mod tile_kernel {
    use std::{
        collections::HashMap,
        fs,
        path::Path,
        sync::{Arc, Mutex, OnceLock, PoisonError},
    };

    pub use ::cutile::tile_kernel::*;
    use ::cutile::{
        error::{Error, KernelLaunchError},
        tensor::GridBound,
    };
    use cuda_core::Function;
    use cutile_compiler::{
        ast::Module,
        compiler::{CUDATileFunctionCompiler, CUDATileModules},
        cuda_tile_runtime_utils::{
            Stage2Source, TileirasOptions, ToolkitCapabilities,
            compile_bytecode_cached_with_toolkit, get_gpu_name,
            recompile_after_disk_rejection_with_toolkit, serialize_tile_ir_bytecode_for_version,
        },
        specialization::{DivHint, SpecializationBits},
    };
    use singe_cuda::context::Context;

    use crate::error::{Error as KernelError, Result as KernelResult};

    type Resolution = (Arc<Function>, Arc<Validator>);

    type Slot = Arc<Mutex<Option<Resolution>>>;

    type Cache = HashMap<(usize, TileFunctionKey), Slot>;

    static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();

    /// Removes compiled functions loaded in this CUDA context.
    ///
    /// # Safety
    /// Unowned raw launches must have completed.
    /// Raw captured graphs must be destroyed or retain their compiled function owners through all replays.
    /// Do not compile or submit unowned raw kernels concurrently with this call.
    /// Owned `TileOperation` submissions and graphs retain their functions independently.
    pub unsafe fn clear_context_kernel_cache(context: &Context) {
        if let Some(cache) = CACHE.get() {
            cache
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .retain(|(handle, _), _| *handle != context.as_raw().addr());
        }
    }

    pub fn prepared_function(
        context: &Context,
        key: &TileFunctionKey,
    ) -> KernelResult<Arc<Function>> {
        let cache = CACHE.get().ok_or(KernelError::KernelNotPrepared)?;
        let slot = cache
            .lock()
            .map_err(|_| KernelError::KernelCachePoisoned)?
            .get(&(context.as_raw().addr(), key.clone()))
            .cloned()
            .ok_or(KernelError::KernelNotPrepared)?;
        let resolution = slot.lock().map_err(|_| KernelError::KernelCachePoisoned)?;
        Ok(Arc::clone(
            &resolution.as_ref().ok_or(KernelError::KernelNotPrepared)?.0,
        ))
    }

    /// Configures a cuTile kernel whose modules are loaded in the actual stream context.
    pub trait TileKernel<ARGS: Send, DI, STORED: Send = ARGS>: DeviceOp<Output = ARGS>
    where
        DI: DeviceOp<Output = STORED>,
    {
        /// Compiles and caches the specialization for the supplied execution context.
        #[doc(hidden)]
        #[allow(
            clippy::too_many_arguments,
            reason = "mirrors the upstream kernel signature"
        )]
        #[expect(
            clippy::too_many_lines,
            reason = "the kernel keeps its complete indexing and operation dispatch in one auditable unit"
        )]
        fn jit_compile<F: Fn() -> Module>(
            &mut self,
            context: &ExecutionContext,
            kernel_ast: F,
            module_name: &str,
            function_name: &str,
            function_entry: &str,
            generics: Vec<String>,
            strides: Vec<(String, Vec<i32>)>,
            specs: Vec<(String, SpecializationBits)>,
            hints: Vec<(String, DivHint)>,
            grid: Option<(u32, u32, u32)>,
            options: CompileOptions,
            source_hash: &str,
        ) -> Result<Resolution, Error> {
            let specialization = _specialization_from_context(
                context,
                &kernel_ast,
                module_name,
                function_name,
                generics,
                strides,
                specs,
                hints,
                grid,
                options,
                source_hash,
            );
            let key = specialization.into_l1_cache_key();
            let device = context.get_cuda_stream().device();
            device.bind_to_thread()?;
            let slot = {
                let mut cache = CACHE.get_or_init(Mutex::default).lock().map_err(|_| {
                    Error::KernelLaunch(KernelLaunchError("kernel cache lock poisoned".into()))
                })?;
                Arc::clone(
                    cache
                        .entry((device.cu_ctx().addr(), key.clone()))
                        .or_default(),
                )
            };
            let mut slot = slot.lock().map_err(|_| {
                Error::KernelLaunch(KernelLaunchError("kernel compilation lock poisoned".into()))
            })?;
            if let Some(resolution) = slot.as_ref() {
                return Ok(resolution.clone());
            }

            let modules = CUDATileModules::from_kernel(kernel_ast())?;
            let capabilities = ToolkitCapabilities::for_device(context.get_device_id())?;
            let strides: Vec<_> = key
                .stride_args
                .iter()
                .map(|(name, value)| (name.as_str(), value.as_slice()))
                .collect();
            let specs: Vec<_> = key
                .spec_args
                .iter()
                .map(|(name, value)| (name.as_str(), value))
                .collect();
            let hints: Vec<_> = key
                .scalar_hints
                .iter()
                .map(|(name, value)| (name.as_str(), value))
                .collect();
            let compiler = CUDATileFunctionCompiler::new(
                &modules,
                module_name,
                function_name,
                &key.function_generics,
                &strides,
                &specs,
                &hints,
                key.grid,
                get_gpu_name(context.get_device_id()),
                &key.compile_options,
            )?
            .with_target_capabilities(capabilities.target.clone())?;
            let module = compiler.compile()?;
            // Launch checks are accumulated during compilation.
            let validator = Arc::new(compiler.get_validator());
            let print_ir = modules.get_entry_arg_bool_by_function_name(
                module_name,
                function_name,
                "print_ir",
            )?;
            let dump_dir = modules.get_entry_arg_string_by_function_name(
                module_name,
                function_name,
                "dump_mlir_dir",
            )?;
            if print_ir || dump_dir.is_some() {
                let ir = module.to_mlir_text();
                if print_ir {
                    println!("COMPILED IR: {module_name}::{function_name}\n{ir}");
                }
                if let Some(directory) = dump_dir {
                    fs::create_dir_all(&directory).map_err(|error| {
                        Error::KernelLaunch(KernelLaunchError(error.to_string()))
                    })?;
                    let path = Path::new(&directory).join(format!(
                        "{module_name}-{function_name}-{}.mlir",
                        key.display_hash()
                    ));
                    fs::write(path, ir).map_err(|error| {
                        Error::KernelLaunch(KernelLaunchError(error.to_string()))
                    })?;
                }
            }
            let (bytecode, _) = serialize_tile_ir_bytecode_for_version(
                &module,
                capabilities.target.bytecode_version,
            )?;
            let options = TileirasOptions::from_compile_options(&key.compile_options);
            let (cubin, source) =
                compile_bytecode_cached_with_toolkit(&bytecode, &options, &capabilities)?;
            // SAFETY: the compiler or its checksum-verified artifact store supplied a complete cubin.
            let module = match unsafe { device.load_module_from_bytes(&cubin) } {
                Ok(module) => module,
                Err(error) => match source {
                    Stage2Source::DiskCache { store, key } => {
                        let cubin = recompile_after_disk_rejection_with_toolkit(
                            store.as_ref(),
                            &key,
                            &bytecode,
                            &options,
                            &capabilities,
                        )?;
                        // SAFETY: this is the complete cubin returned by the compiler.
                        unsafe { device.load_module_from_bytes(&cubin) }?
                    }
                    Stage2Source::Tileiras => return Err(error.into()),
                },
            };
            let resolution = (Arc::new(module.load_function(function_entry)?), validator);
            *slot = Some(resolution.clone());
            drop(slot);
            Ok(resolution)
        }

        /// Sets type and constant generic arguments.
        #[must_use]
        fn generics(self, generics: Vec<String>) -> Self;

        /// Sets a constant launch grid used during compilation.
        #[must_use]
        fn const_grid(self, grid: (u32, u32, u32)) -> Self;

        /// Sets the runtime launch grid.
        #[must_use]
        fn grid(self, grid: (u32, u32, u32)) -> Self;

        /// Sets compiler options for this specialization.
        #[must_use]
        fn compile_options(self, options: CompileOptions) -> Self;

        /// Infers the launch grid from partitioned arguments.
        fn infer_launch_grid(&self, bounds: &[GridBound]) -> Result<(u32, u32, u32), Error> {
            infer_launch_grid(self.get_launch_grid(), bounds)
        }

        /// Returns the configured runtime grid.
        fn get_launch_grid(&self) -> (u32, u32, u32);

        /// Returns dynamic shared memory in bytes.
        fn get_launch_smem(&self) -> u32 {
            0
        }

        /// Returns thread block dimensions.
        fn get_launch_block(&self) -> (u32, u32, u32) {
            (1, 1, 1)
        }
    }

    // cuTile 0.4's generated shortcut receives only a device ordinal.
    // Bypass it and resolve every launch through the actual-context cache above.
    // Never infer the requested stream context from the calling thread's current context.
    #[derive(Default)]
    pub struct LaunchSite;

    #[expect(
        clippy::unused_self,
        reason = "cuTile-generated launches require this instance-method cache protocol"
    )]
    impl LaunchSite {
        pub const fn new() -> Self {
            Self
        }

        pub fn get(
            &self,
            _: usize,
            _: &[String],
            _: &[&SpecializationBits],
            _: &[DivHint],
            _: Option<(u32, u32, u32)>,
            _: &CompileOptions,
        ) -> Option<Resolution> {
            None
        }

        pub fn store(&self, _: SiteResolution) {}
    }

    pub struct SiteResolution;

    impl SiteResolution {
        #[allow(
            clippy::too_many_arguments,
            reason = "mirrors the upstream kernel signature"
        )]
        pub fn new(
            _: usize,
            _: Vec<String>,
            _: Vec<SpecializationBits>,
            _: Vec<DivHint>,
            _: Option<(u32, u32, u32)>,
            _: CompileOptions,
            _: Arc<Function>,
            _: Arc<Validator>,
        ) -> Self {
            Self
        }
    }
}
