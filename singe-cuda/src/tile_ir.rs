//! CUDA Tile IR assembly through the toolkit's `tileiras` executable.

use std::{
    env, fs,
    io::ErrorKind,
    path::{Path, PathBuf},
    process,
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

use crate::{
    architecture::GpuArchitecture,
    error::{Error, Result},
    module::ModuleImage,
};

/// NVIDIA CUDA Tile IR optimizing assembler.
#[derive(Debug, Clone)]
pub struct Assembler {
    executable: PathBuf,
    /// Name of the NVIDIA GPU to generate code for.
    pub architecture: GpuArchitecture,
    /// Generate line-number information if present in the input bytecode.
    pub line_info: bool,
}

impl Assembler {
    /// Creates assembler settings for `architecture`, using `tileiras` from the process search path and leaving line information disabled.
    #[must_use]
    pub fn new(architecture: GpuArchitecture) -> Self {
        Self {
            executable: PathBuf::from("tileiras"),
            architecture,
            line_info: false,
        }
    }

    /// Selects the assembler executable path or command name.
    ///
    /// The executable is checked when a command is run.
    #[must_use]
    pub fn with_executable(mut self, executable: impl Into<PathBuf>) -> Self {
        self.executable = executable.into();
        self
    }

    /// Lists all supported bytecode versions.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the assembler process cannot be started or its output cannot be collected. Returns `Error::Compiler` with the process's stderr if the command exits unsuccessfully. Successful stdout is decoded with invalid UTF-8 replaced.
    pub fn supported_versions(&self) -> Result<String> {
        let output = Command::new(&self.executable)
            .arg("--list-versions")
            .output()?;
        if !output.status.success() {
            return Err(Error::Compiler {
                message: String::from_utf8_lossy(&output.stderr).into_owned(),
            });
        }
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }

    /// Assembles a tile bytecode file into a CUDA binary.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidValue` for a virtual architecture. Also propagates errors from creating temporary storage, writing the input, running [`Self::assemble_file`], or reading the output binary. Temporary files are removed on drop when possible.
    pub fn assemble(&self, bytecode: &[u8]) -> Result<ModuleImage<'static>> {
        if self.architecture.is_virtual() {
            return Err(Error::InvalidValue);
        }
        let directory = WorkDirectory::create()?;
        let input = directory.0.join("input.tileir");
        let output = directory.0.join("output.cubin");
        fs::write(&input, bytecode)?;
        self.assemble_file(&input, &output)?;
        Ok(ModuleImage::from_vec(fs::read(output)?))
    }

    /// Assembles a tile bytecode file, writing the resulting binary to `output`.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidValue` for a virtual architecture. Returns an I/O error if the assembler cannot be started or its output collected, or `Error::Compiler` containing stderr if it exits unsuccessfully, including when it rejects the input or cannot write `output`.
    pub fn assemble_file(&self, input: &Path, output: &Path) -> Result<()> {
        if self.architecture.is_virtual() {
            return Err(Error::InvalidValue);
        }
        let mut command = Command::new(&self.executable);
        command
            .arg(format!("--gpu-name={}", self.architecture))
            .arg(input)
            .arg("-o")
            .arg(output);
        if self.line_info {
            command.arg("--lineinfo");
        }
        let result = command.output()?;
        if result.status.success() {
            Ok(())
        } else {
            Err(Error::Compiler {
                message: String::from_utf8_lossy(&result.stderr).into_owned(),
            })
        }
    }
}

struct WorkDirectory(PathBuf);

impl WorkDirectory {
    fn create() -> Result<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let path = env::temp_dir().join(format!(
                "singe-tileir-{}-{}",
                process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self(path)),
                Err(error) if error.kind() == ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error.into()),
            }
        }
    }
}

impl Drop for WorkDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
