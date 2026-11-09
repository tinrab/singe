//! Compiles generated kernels to cubin for one GPU architecture, and keeps the results in a directory across processes.
//!
//! NVRTC spends about 12 ms per kernel parsing C++ even when the driver's cache holds its output, while loading a cubin takes about 0.05 ms.
//! The directory defaults to `singe/kernels` under the user's cache directory, [`enable_kernel_cache`] names another, and [`disable_kernel_cache`] turns it off.
//! Each entry is named by a hash of the kernel's code, its name, the compile options or target architecture, and the compiler's version, NVRTC's or `tileiras`', so a toolkit or option change compiles again instead of loading a stale kernel.

use std::{
    env, fs,
    io::ErrorKind,
    path::{Path, PathBuf},
    process,
    sync::RwLock,
};

use sha2::{Digest, Sha256};
use singe_cuda::nvrtc::{self, CompileOptions, Program};

use crate::{
    backend::cuda::codegen::tile,
    error::{CompileError, Error, Result},
};

/// Increments when the entry format or key changes, so old entries stop matching.
const FORMAT: &str = "singe-kernel-cache-2";

// What the cache directory is.
#[derive(Debug, Clone)]
enum CacheSetting {
    // Nothing has chosen, so the first use resolves the default.
    Default,
    On(PathBuf),
    Off,
}

static DIRECTORY: RwLock<CacheSetting> = RwLock::new(CacheSetting::Default);

// `$XDG_CACHE_HOME/singe/kernels` or `~/.cache/singe/kernels`, or nothing when neither is set.
fn default_directory() -> Option<PathBuf> {
    env::var_os("XDG_CACHE_HOME")
        .filter(|path| !path.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            env::var_os("HOME")
                .filter(|path| !path.is_empty())
                .map(|home| PathBuf::from(home).join(".cache"))
        })
        .map(|cache| cache.join("singe/kernels"))
        .filter(|directory| fs::create_dir_all(directory).is_ok())
}

// Returns the directory in force, resolving the default once.
fn directory() -> Result<Option<PathBuf>> {
    let chosen = DIRECTORY
        .read()
        .map_err(|_| Error::poisoned("cuda kernel cache setting"))?
        .clone();
    match chosen {
        CacheSetting::On(directory) => return Ok(Some(directory)),
        CacheSetting::Off => return Ok(None),
        CacheSetting::Default => {}
    }

    let directory = default_directory();
    *DIRECTORY
        .write()
        .map_err(|_| Error::poisoned("cuda kernel cache setting"))? = directory
        .clone()
        .map_or(CacheSetting::Off, CacheSetting::On);
    Ok(directory)
}

/// Keeps compiled generated kernels in `directory`, creating it, so later processes load them instead of compiling them.
///
/// # Errors
///
/// Returns an error if `directory` cannot be created or the cache setting is poisoned.
pub fn enable_kernel_cache(directory: impl Into<PathBuf>) -> Result<()> {
    let directory = directory.into();
    fs::create_dir_all(&directory).map_err(CompileError::KernelCacheIo)?;
    *DIRECTORY
        .write()
        .map_err(|_| Error::poisoned("cuda kernel cache setting"))? = CacheSetting::On(directory);
    Ok(())
}

/// Stops reading and writing the kernel cache directory; compiled kernels stay cached in memory.
pub fn disable_kernel_cache() {
    if let Ok(mut directory) = DIRECTORY.write() {
        *directory = CacheSetting::Off;
    }
}

/// Where a compiled kernel came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CubinOrigin {
    /// NVRTC compiled the source in this call.
    Compiled,
    /// The cache directory held the cubin.
    Cached,
}

/// Returns the cubin of kernel `name` that `source` defines, compiled with `options`, from the cache directory when it holds one.
///
/// # Errors
///
/// Returns an error if NVRTC rejects the source, or the cache directory cannot be read or written.
pub fn cubin(
    name: &str,
    source: &str,
    options: &CompileOptions<'_>,
) -> Result<(Vec<u8>, CubinOrigin)> {
    cached(&nvrtc_key(name, source, options)?, || {
        compile(name, source, options)
    })
}

/// Returns the cubin of Tile IR kernel `name` that `bytecode` defines, compiled for `architecture` by `tileiras`, from the cache directory when it holds one.
///
/// # Errors
///
/// Returns an error if `tileiras` cannot run or rejects the bytecode, or the cache directory cannot be read or written.
pub fn tile_cubin(
    name: &str,
    bytecode: &[u8],
    architecture: &str,
) -> Result<(Vec<u8>, CubinOrigin)> {
    cached(&tile_key(name, bytecode, architecture)?, || {
        tile::compile(name, bytecode, architecture)
    })
}

/// Returns the cubin the cache directory holds under `key`, or compiles it with `compile` and stores it there.
fn cached(key: &str, compile: impl FnOnce() -> Result<Vec<u8>>) -> Result<(Vec<u8>, CubinOrigin)> {
    let Some(directory) = directory()? else {
        return Ok((compile()?, CubinOrigin::Compiled));
    };
    let path = directory.join(format!("{key}.cubin"));
    match fs::read(&path) {
        Ok(bytes) if !bytes.is_empty() => return Ok((bytes, CubinOrigin::Cached)),
        Ok(_) => {}
        Err(error) if error.kind() == ErrorKind::NotFound => {}
        Err(error) => return Err(CompileError::KernelCacheIo(error).into()),
    }

    let bytes = compile()?;
    write(&path, &bytes)?;
    Ok((bytes, CubinOrigin::Compiled))
}

/// Removes the cache entry for a CUDA C++ kernel whose cached cubin failed to load, so the next call compiles it again.
///
/// # Errors
///
/// Returns an error if the entry exists but cannot be removed.
pub fn forget(name: &str, source: &str, options: &CompileOptions<'_>) -> Result<()> {
    forget_key(&nvrtc_key(name, source, options)?)
}

/// Removes the cache entry for a Tile IR kernel whose cached cubin failed to load.
///
/// # Errors
///
/// Returns an error if the entry exists but cannot be removed.
pub fn forget_tile(name: &str, bytecode: &[u8], architecture: &str) -> Result<()> {
    forget_key(&tile_key(name, bytecode, architecture)?)
}

fn forget_key(key: &str) -> Result<()> {
    let Some(directory) = directory()? else {
        return Ok(());
    };
    match fs::remove_file(directory.join(format!("{key}.cubin"))) {
        Err(error) if error.kind() != ErrorKind::NotFound => {
            Err(CompileError::KernelCacheIo(error).into())
        }
        _ => Ok(()),
    }
}

fn compile(name: &str, source: &str, options: &CompileOptions<'_>) -> Result<Vec<u8>> {
    let program = Program::from_source(source).with_name(&format!("{name}.cu"));
    if let Err(source) = program.compile_with_options(options) {
        return Err(CompileError::Nvrtc {
            kernel: name.to_owned(),
            log: program.log().unwrap_or_default(),
            source: Box::new(source),
        }
        .into());
    }
    Ok(program.cubin()?)
}

/// Returns the key of a CUDA C++ kernel: its source, name, and options, and the NVRTC version.
fn nvrtc_key(name: &str, source: &str, options: &CompileOptions<'_>) -> Result<String> {
    let version = nvrtc::version()?.to_string();
    let options = options.as_arguments();
    Ok(key([
        b"nvrtc".as_slice(),
        version.as_bytes(),
        name.as_bytes(),
        source.as_bytes(),
    ]
    .into_iter()
    .chain(options.iter().map(String::as_bytes))))
}

/// Returns the key of a Tile IR kernel: its bytecode, name, and architecture, and the `tileiras` version.
fn tile_key(name: &str, bytecode: &[u8], architecture: &str) -> Result<String> {
    let version = tile::version()?;
    Ok(key([
        b"tileiras".as_slice(),
        version.as_bytes(),
        name.as_bytes(),
        architecture.as_bytes(),
        bytecode,
    ]))
}

/// Returns the hexadecimal SHA-256 of the entry format and `parts`, each terminated by a zero byte.
fn key<'a>(parts: impl IntoIterator<Item = &'a [u8]>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(FORMAT.as_bytes());
    hasher.update([0]);
    for part in parts {
        hasher.update(part);
        hasher.update([0]);
    }
    hex::encode(hasher.finalize())
}

/// Writes an entry through a file private to this process, then renames it, so readers never see a partial cubin.
fn write(path: &Path, bytes: &[u8]) -> Result<()> {
    let partial = path.with_extension(format!("cubin.{}", process::id()));
    fs::write(&partial, bytes)
        .and_then(|()| fs::rename(&partial, path))
        .map_err(|error| {
            // The write already failed; a partial file left behind is never read, since only renamed files are.
            fs::remove_file(&partial).ok();
            CompileError::KernelCacheIo(error).into()
        })
}

#[cfg(test)]
mod tests;
