// Builds exported programs as their users would, under `target/programs`.

use std::{
    path::Path,
    process::{self},
};

use crate::backend::cuda::export::{
    CudaProgram, CudaProgramSet,
    compiled::{CompiledProgram, ProgramError},
};

impl CompiledProgram {
    // Writes `program` under `target/programs/<name>`, and compiles it with its manifest's command.
    //
    // # Errors
    //
    // Returns the compiler's output if `nvcc` rejects the program, or a description of a failure to access its files.
    pub fn create(program: &CudaProgram, name: &str) -> Result<Self, ProgramError> {
        let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../target/programs")
            .join(format!("{name}-{}", process::id()));
        Self::create_in(program, directory)
    }

    // Writes a program of several steps under `target/programs/<name>`, and compiles it with its manifests' commands.
    //
    // # Errors
    //
    // Returns the compiler's output if `nvcc` rejects the program, or a description of a failure to access its files.
    pub fn create_set(program: &CudaProgramSet, name: &str) -> Result<Self, ProgramError> {
        let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../target/programs")
            .join(format!("{name}-{}", process::id()));
        Self::create_set_in(program, directory)
    }
}
