use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock, Weak},
};

use super::{CompileOptions, Header, OutputKind, Program, supported_architectures};
use crate::{
    architecture::GpuArchitecture,
    context::Context,
    error::{Error, Result},
    jit::JitOptions,
    module::{KernelFunction, Module},
};

/// CUDA C++ source with its headers, NVRTC options, and kernel names, which [`compile_cached`] compiles once per context.
///
/// `cuda_module!` emits one as a constant; `key` identifies the whole description, so equal keys must describe equal sources.
#[derive(Debug, Clone, Copy)]
pub struct ModuleSource {
    /// Identifier of this description, such as a hash of every other field.
    pub key: &'static str,
    /// CUDA C++ source text.
    pub source: &'static str,
    /// Virtual headers the source includes.
    pub headers: &'static [Header<'static>],
    /// Native NVRTC options appended after the target architecture.
    pub options: &'static [&'static str],
    /// Kernel names whose lowered symbols the compiled module records.
    pub kernels: &'static [&'static str],
}

/// A module compiled from a [`ModuleSource`], with the lowered symbol of each of its kernels.
#[derive(Debug)]
pub struct CompiledModule {
    module: Module,
    symbols: HashMap<&'static str, String>,
}

impl CompiledModule {
    /// Returns the loaded module.
    #[must_use]
    pub fn module(&self) -> &Module {
        &self.module
    }

    /// Looks up a kernel by its source name, through its lowered symbol when the source listed it.
    ///
    /// # Errors
    ///
    /// Returns an error if the name contains an interior NUL, the context cannot be bound, or CUDA cannot find the function.
    pub fn function(&self, kernel: &str) -> Result<KernelFunction> {
        let symbol = self.symbols.get(kernel).map_or(kernel, String::as_str);
        self.module.function(symbol)
    }
}

type ModuleCache = HashMap<(usize, &'static str), Weak<CompiledModule>>;

static MODULE_CACHE: OnceLock<Mutex<ModuleCache>> = OnceLock::new();

/// Compiles `source` for the device of `context` and loads it, or returns the module a live caller already compiled there.
///
/// The target is the device's real architecture, emitted as cubin, when NVRTC supports it.
/// Otherwise it is the newest virtual architecture NVRTC supports at or below the device's, emitted as PTX for the driver to compile.
///
/// # Errors
///
/// Returns an error if the module cache lock is poisoned, the device properties or NVRTC's architectures cannot be queried, NVRTC supports no architecture at or below the device's, compilation fails, or the module cannot be loaded.
pub fn compile_cached(
    context: &Arc<Context>,
    source: &ModuleSource,
) -> Result<Arc<CompiledModule>> {
    // The native context handle keys the cache, so wrappers of one context share modules; a cached module retains its context, so a live entry's handle cannot be reused.
    let key = (context.as_raw().addr(), source.key);
    let cache = MODULE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let poisoned = |_| Error::lock_poisoned("cuda module cache");
    let cached = cache
        .lock()
        .map_err(poisoned)?
        .get(&key)
        .and_then(Weak::upgrade);
    if let Some(module) = cached {
        return Ok(module);
    }

    // Compiles without holding the lock, so a racing caller may compile the same source too; the first to insert wins.
    let module = Arc::new(compile(context, source)?);
    let mut entries = cache.lock().map_err(poisoned)?;
    if let Some(cached) = entries.get(&key).and_then(Weak::upgrade) {
        return Ok(cached);
    }

    entries.retain(|_, module| module.strong_count() > 0);
    entries.insert(key, Arc::downgrade(&module));
    drop(entries);
    Ok(module)
}

fn compile(context: &Arc<Context>, source: &ModuleSource) -> Result<CompiledModule> {
    let properties = context.device().properties()?;
    let (architecture, output) = target(properties.major, properties.minor)?;
    let program = Program::from_source(source.source)
        .with_name("cuda_module.cu")
        .with_headers(source.headers);
    let name_expressions = source
        .kernels
        .iter()
        .map(|name| format!("&{name}"))
        .collect::<Vec<_>>();
    for name_expression in &name_expressions {
        program.add_name_expression(name_expression)?;
    }

    let options = source.options.iter().fold(
        CompileOptions::new().gpu_architecture(architecture),
        |options, option| options.raw_option(option),
    );
    program.compile(&options)?;
    let symbols = source
        .kernels
        .iter()
        .zip(&name_expressions)
        .map(|(kernel, name_expression)| Ok((*kernel, program.lowered_name(name_expression)?)))
        .collect::<Result<_>>()?;
    let module = context.load_nvrtc_module(&program, output, JitOptions::default())?;
    Ok(CompiledModule { module, symbols })
}

// Picks the real architecture `sm_<major><minor>` as cubin when NVRTC supports it, or else the newest supported virtual architecture at or below it as PTX.
fn target(major: i32, minor: i32) -> Result<(GpuArchitecture, OutputKind)> {
    let requested = major * 10 + minor;
    let supported = supported_architectures()?;
    if supported.contains(&requested)
        && let Ok(architecture) = format!("sm_{requested}").parse()
    {
        return Ok((architecture, OutputKind::Cubin));
    }

    let architecture = supported
        .into_iter()
        .filter(|value| *value <= requested)
        .filter_map(|value| format!("compute_{value}").parse().ok())
        .max_by_key(|architecture: &GpuArchitecture| architecture.compute_capability())
        .ok_or_else(|| Error::UnsupportedArchitecture {
            target: format!("sm_{requested}"),
        })?;
    Ok((architecture, OutputKind::Ptx))
}
