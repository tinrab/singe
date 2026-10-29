//! CPU emulator for `sm_80` PTX kernels.
//!
//! [`ParsedModule::compile`] validates a kernel without creating an emulator.
//! An [`Emulator`] runs loaded modules on a worker thread. Each operation returns a
//! [`Submission`] that can be waited on or awaited.

mod budget;
mod compiler;
mod constants;
mod engine;
mod error;
mod execution_error;
mod integer;
mod ir;
mod layout;
mod lowering;
mod math;
mod memory;
mod numeric;
mod runtime;
mod snapshot;
mod target;

pub use half::{bf16 as Bf16, f16 as F16};

#[cfg(test)]
mod tests;

use std::{
    collections::BTreeMap,
    fmt::{self, Debug, Formatter},
    sync::{Arc, Mutex, PoisonError},
};

pub use crate::emulator::{
    compiler::LaunchRequirements,
    error::{CompileError, CompileErrorKind},
    execution_error::{ExecutionError, FaultLocation, StalledLane, SubmitError},
    layout::{ModuleLayout, ModuleVariable, ParameterLayout, StorageLayout},
    runtime::{
        Breakpoint, BreakpointId, Buffer, BufferView, Emulator, EmulatorOptions, Event,
        ExecutionReport, Kernel, KernelArg, LaunchConfig, Module, PtxScalar, Shutdown,
        ShutdownMode, Stream, Submission,
    },
    snapshot::{FrameSnapshot, LaneSnapshot, RegisterSnapshot},
};
use crate::{
    ast,
    parser::{self, ParseError},
};

/// Parsed PTX source that can be compiled and loaded on any number of emulators.
///
/// Clones share the source, the syntax tree, and the compiled kernels.
#[derive(Clone)]
pub struct ParsedModule {
    inner: Arc<ParsedModuleInner>,
}

#[derive(Debug)]
struct ParsedModuleInner {
    source: String,
    syntax: ast::Module,
    compiled: Mutex<BTreeMap<String, Arc<CompiledArtifact>>>,
}

impl ParsedModule {
    /// Parses PTX source.
    ///
    /// Parsing checks syntax only; [`compile`](Self::compile) checks whether a kernel
    /// can execute.
    ///
    /// # Errors
    ///
    /// Returns a [`ParseError`] with the byte span of the invalid syntax.
    pub fn parse(source: &str) -> Result<Self, ParseError> {
        let syntax = parser::parse_module(source)?;
        Ok(Self {
            inner: Arc::new(ParsedModuleInner {
                source: source.to_owned(),
                syntax,
                compiled: Mutex::default(),
            }),
        })
    }

    /// Returns the source text that every diagnostic byte span indexes into.
    #[must_use]
    pub fn source(&self) -> &str {
        &self.inner.source
    }

    /// Returns the syntax tree, which may contain instructions that cannot execute.
    #[must_use]
    pub fn syntax(&self) -> &ast::Module {
        &self.inner.syntax
    }

    /// Computes the storage layout of the module's variables without allocating memory.
    ///
    /// # Errors
    ///
    /// Returns an error for an unsupported target, an invalid storage declaration, or an
    /// invalid initializer.
    pub fn layout(&self) -> Result<ModuleLayout, CompileError> {
        ModuleLayout::create(self.syntax())
    }

    /// Validates the `entry` kernel and every function it calls, and lowers them to
    /// executable form.
    ///
    /// Other entry points are not checked and may use unsupported instructions.
    /// Results are cached and shared by clones of this module.
    ///
    /// # Errors
    ///
    /// Returns an error with a source span if `entry` is missing, or if a declaration,
    /// instruction, operand, or symbol reference is invalid or unsupported.
    pub fn compile(&self, entry: &str) -> Result<CompiledKernel, CompileError> {
        let cached = self
            .inner
            .compiled
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(entry)
            .cloned();
        let artifact = if let Some(artifact) = cached {
            artifact
        } else {
            let (executable, launch) = compiler::compile(self.syntax(), entry)?;
            let artifact = Arc::new(CompiledArtifact {
                launch,
                executable: Arc::new(executable),
            });
            Arc::clone(
                self.inner
                    .compiled
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .entry(entry.to_owned())
                    .or_insert(artifact),
            )
        };
        Ok(CompiledKernel {
            source: self.clone(),
            artifact,
        })
    }
}

/// A validated kernel, ready to bind with [`Module::bind`] on any emulator that loaded
/// its source.
///
/// Clones share the compiled code.
#[derive(Clone)]
pub struct CompiledKernel {
    source: ParsedModule,
    artifact: Arc<CompiledArtifact>,
}

#[derive(Debug)]
struct CompiledArtifact {
    launch: LaunchRequirements,
    executable: Arc<ir::Executable>,
}

impl CompiledKernel {
    /// Returns the selected entry point's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.artifact.executable.entry_function().name
    }

    /// Returns the parsed source retained by this compilation.
    #[must_use]
    pub fn source(&self) -> &ParsedModule {
        &self.source
    }

    /// Returns the storage layout of the module's variables.
    #[must_use]
    pub fn module_layout(&self) -> &ModuleLayout {
        &self.artifact.executable.module
    }

    /// Returns the entry point's parameters in argument order.
    #[must_use]
    pub fn parameters(&self) -> &[ParameterLayout] {
        &self.artifact.executable.entry_function().parameters
    }

    /// Returns the block dimensions the kernel accepts.
    #[must_use]
    pub fn launch_requirements(&self) -> &LaunchRequirements {
        &self.artifact.launch
    }

    /// Returns the number of compiled functions: the entry point and each function it calls.
    #[must_use]
    pub fn function_count(&self) -> usize {
        self.artifact.executable.functions.len()
    }

    /// Returns the number of lowered instructions across all compiled functions.
    #[must_use]
    pub fn instruction_count(&self) -> usize {
        self.artifact
            .executable
            .functions
            .iter()
            .map(|function| function.instructions.len())
            .sum()
    }
}

impl Debug for ParsedModule {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParsedModule")
            .field("source_len", &self.inner.source.len())
            .finish_non_exhaustive()
    }
}

impl Debug for CompiledKernel {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompiledKernel")
            .field("name", &self.name())
            .field("function_count", &self.function_count())
            .finish_non_exhaustive()
    }
}
