//! Keeps compiled generated kernels and built exported programs in a directory across processes.
//!
//! NVRTC spends about 12 ms per kernel parsing C++ even when the driver's cache holds its output, while loading a cubin takes about 0.05 ms.
//! The directory defaults to the user's cache directory for `singe`, [`enable_disk_cache`] names another, and [`disable_disk_cache`] turns it off.
//! Entries live under `v1/kernels` and `v1/programs`, each named by a hash of everything that built it, including the compiler's version, so a toolkit or option change builds again instead of loading a stale entry.
//! Reading and writing entries is best-effort: a failed read builds again, and a failed write keeps the result in memory only.

use std::{
    fs, io,
    path::{Path, PathBuf},
    process,
    sync::{
        LazyLock, PoisonError, RwLock,
        atomic::{AtomicUsize, Ordering},
    },
};

use directories::ProjectDirs;
use sha2::{Digest, Sha256};
use singe_cuda::{
    architecture::GpuArchitecture,
    nvrtc::{self, CompileOptions, Program},
};

use crate::{
    backend::cuda::codegen::{
        kernel::{KernelArtifact, KernelCode},
        render::KernelOutput,
        tile,
    },
    error::{CompileError, Result},
};

const VERSION: &str = "v1";

#[derive(Debug, Clone)]
enum CacheSetting {
    Default,
    On(PathBuf),
    Off,
}

static SETTING: RwLock<CacheSetting> = RwLock::new(CacheSetting::Default);

static DEFAULT_ROOT: LazyLock<Option<PathBuf>> = LazyLock::new(|| {
    ProjectDirs::from("", "", "singe").map(|directories| directories.cache_dir().to_owned())
});

fn root() -> Option<PathBuf> {
    match &*SETTING.read().unwrap_or_else(PoisonError::into_inner) {
        CacheSetting::Default => DEFAULT_ROOT.clone(),
        CacheSetting::On(root) => Some(root.clone()),
        CacheSetting::Off => None,
    }
}

/// Keeps compiled generated kernels and built programs in `directory`, creating it, so later processes load them instead of building them.
///
/// # Errors
///
/// Returns an error if `directory` cannot be created.
pub fn enable_disk_cache(directory: impl Into<PathBuf>) -> Result<()> {
    let directory = directory.into();
    fs::create_dir_all(&directory).map_err(|source| CompileError::CacheIo {
        path: directory.clone(),
        source,
    })?;
    *SETTING.write().unwrap_or_else(PoisonError::into_inner) = CacheSetting::On(directory);
    Ok(())
}

/// Stops reading and writing the disk cache; compiled kernels stay cached in memory.
pub fn disable_disk_cache() {
    *SETTING.write().unwrap_or_else(PoisonError::into_inner) = CacheSetting::Off;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EntryKind {
    Kernel,
    Program,
}

impl EntryKind {
    const fn directory(self) -> &'static str {
        match self {
            Self::Kernel => "kernels",
            Self::Program => "programs",
        }
    }

    const fn extension(self) -> &'static str {
        match self {
            Self::Kernel => "cubin",
            Self::Program => "so",
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Entry {
    path: PathBuf,
}

impl Entry {
    pub(crate) fn read(&self) -> Option<Vec<u8>> {
        fs::read(&self.path).ok().filter(|bytes| !bytes.is_empty())
    }

    pub(crate) fn copy_to(&self, destination: &Path) -> bool {
        fs::copy(&self.path, destination).is_ok_and(|bytes| bytes > 0)
    }

    pub(crate) fn write(&self, bytes: &[u8]) {
        self.publish(|partial| fs::write(partial, bytes));
    }

    pub(crate) fn store(&self, file: &Path) {
        self.publish(|partial| fs::copy(file, partial).map(drop));
    }

    // Writes through a file private to this call, then renames it into place, so readers never see a partial entry.
    fn publish(&self, write: impl FnOnce(&Path) -> io::Result<()>) {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let partial = self.path.with_extension(format!(
            "{}.{}.partial",
            process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let published = self
            .path
            .parent()
            .map_or(Ok(()), fs::create_dir_all)
            .and_then(|()| write(&partial))
            .and_then(|()| fs::rename(&partial, &self.path));
        if published.is_err() {
            fs::remove_file(&partial).ok();
        }
    }
}

// The entry of `kind` that `parts` build, or nothing when the cache is off.
pub(crate) fn entry<'a>(
    kind: EntryKind,
    parts: impl IntoIterator<Item = &'a [u8]>,
) -> Option<Entry> {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part.len().to_le_bytes());
        hasher.update(part);
    }

    let name = format!("{}.{}", hex::encode(hasher.finalize()), kind.extension());
    let path = root()?.join(VERSION).join(kind.directory()).join(name);
    Some(Entry { path })
}

/// Where a compiled kernel came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CubinOrigin {
    /// NVRTC compiled the source in this call.
    Compiled,
    /// The cache directory held the cubin.
    Cached,
}

// What a kernel's compiler reads: NVRTC the text of CUDA C++ with its options, `tileiras` Tile IR bytecode for an architecture.
pub(crate) enum CompilerInput<'a> {
    Cuda {
        source: String,
        options: Box<CompileOptions<'static>>,
    },
    Tile {
        bytecode: &'a [u8],
        architecture: String,
    },
}

impl<'a> CompilerInput<'a> {
    pub(crate) fn create(
        kernel: &'a KernelArtifact,
        options: CompileOptions<'static>,
        architecture: GpuArchitecture,
    ) -> Result<Self> {
        match &kernel.code {
            // Rendering rejects source in a language other backends compile, which reaches no CUDA compiler.
            KernelCode::Source(_) => Ok(Self::Cuda {
                source: kernel.render(KernelOutput::Nvrtc)?,
                options: Box::new(options),
            }),
            KernelCode::TileIr(bytecode) => Ok(Self::Tile {
                bytecode,
                architecture: architecture.to_string(),
            }),
        }
    }

    fn entry(&self, name: &str) -> Result<Option<Entry>> {
        Ok(match self {
            Self::Cuda { source, options } => {
                let version = nvrtc::version()?.to_string();
                let arguments = options.as_arguments();
                entry(
                    EntryKind::Kernel,
                    [
                        b"nvrtc".as_slice(),
                        version.as_bytes(),
                        name.as_bytes(),
                        source.as_bytes(),
                    ]
                    .into_iter()
                    .chain(arguments.iter().map(String::as_bytes)),
                )
            }
            Self::Tile {
                bytecode,
                architecture,
            } => {
                let version = tile::version()?;
                entry(
                    EntryKind::Kernel,
                    [
                        b"tileiras".as_slice(),
                        version.as_bytes(),
                        name.as_bytes(),
                        architecture.as_bytes(),
                        bytecode,
                    ],
                )
            }
        })
    }

    fn compile(&self, name: &str) -> Result<Vec<u8>> {
        match self {
            Self::Cuda { source, options } => compile_cuda(name, source, options),
            Self::Tile {
                bytecode,
                architecture,
            } => tile::compile(name, bytecode, architecture),
        }
    }
}

/// Returns the cubin of kernel `name` that `input` defines, from the cache directory when it holds one.
///
/// # Errors
///
/// Returns an error if the compiler rejects the input or cannot run.
pub(crate) fn cubin(name: &str, input: &CompilerInput<'_>) -> Result<(Vec<u8>, CubinOrigin)> {
    let entry = input.entry(name)?;
    if let Some(bytes) = entry.as_ref().and_then(Entry::read) {
        return Ok((bytes, CubinOrigin::Cached));
    }

    let bytes = input.compile(name)?;
    if let Some(entry) = entry {
        entry.write(&bytes);
    }
    Ok((bytes, CubinOrigin::Compiled))
}

// Compiles kernel `name` again and replaces its entry, whose cubin failed to load.
pub(crate) fn recompile(name: &str, input: &CompilerInput<'_>) -> Result<Vec<u8>> {
    let bytes = input.compile(name)?;
    if let Some(entry) = input.entry(name)? {
        entry.write(&bytes);
    }
    Ok(bytes)
}

fn compile_cuda(name: &str, source: &str, options: &CompileOptions<'_>) -> Result<Vec<u8>> {
    let program = Program::from_source(source).with_name(&format!("{name}.cu"));
    if let Err(source) = program.compile(options) {
        return Err(CompileError::Nvrtc {
            kernel: name.to_owned(),
            log: program.log().unwrap_or_default(),
            source: Box::new(source),
        }
        .into());
    }
    Ok(program.cubin()?)
}

#[cfg(test)]
mod tests;
