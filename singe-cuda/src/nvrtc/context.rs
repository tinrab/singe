use std::sync::Arc;

use super::{CompilationArtifact, OutputKind, Program};
use crate::{
    context::Context,
    error::{Error, Result},
    jit::JitOptions,
    library::Library,
    module::{Module, ModuleImage},
};

impl Context {
    /// Retrieves `output` from a compiled NVRTC program and loads it as a module with `jit_options`.
    ///
    /// Only PTX, cubin, and Tile IR can be loaded directly.
    /// LTO IR and `OptiX` IR are rejected; link LTO IR before loading it.
    ///
    /// # Errors
    ///
    /// Returns an error if `output` is LTO IR or `OptiX` IR, the artifact cannot be retrieved or is empty, the context cannot be bound, the JIT option count exceeds `u32`, CUDA rejects the image or JIT settings, or CUDA returns a null module handle.
    pub fn load_nvrtc_module(
        self: &Arc<Self>,
        program: &Program,
        output: OutputKind,
        jit_options: JitOptions<'_>,
    ) -> Result<Module> {
        let image = loadable_image(program.artifact(output)?)?;
        self.load_module(&image, jit_options)
    }

    /// Retrieves `output` from a compiled NVRTC program and loads it as a CUDA library with `jit_options`.
    ///
    /// Only PTX, cubin, and Tile IR can be loaded directly.
    /// LTO IR and `OptiX` IR are rejected; link LTO IR before loading it.
    ///
    /// See [`Self::load_library`] for eager and lazy loading behavior.
    ///
    /// # Errors
    ///
    /// Returns an error if `output` is LTO IR or `OptiX` IR, the artifact cannot be retrieved or is empty, or [`Self::load_library`] fails.
    pub fn load_nvrtc_library(
        self: &Arc<Self>,
        program: &Program,
        output: OutputKind,
        jit_options: JitOptions<'_>,
    ) -> Result<Library> {
        let image = loadable_image(program.artifact(output)?)?;
        self.load_library(&image, jit_options)
    }
}

fn loadable_image(artifact: CompilationArtifact) -> Result<ModuleImage<'static>> {
    artifact
        .into_image()
        .map_err(|_| Error::out_of_range("nvrtc output kind"))
}
