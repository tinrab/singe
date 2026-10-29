//! Discovers NCCL libraries and emits Cargo linker metadata.

use std::{env, path::PathBuf};

use singe_cuda_find::{
    emit_dependency_metadata, emit_link_libraries, emit_rerun_if_env_changed,
    emit_static_link_libraries, find_cuda_runtime, find_nccl,
};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    emit_rerun_if_env_changed();

    link_libraries();
}

fn link_libraries() {
    if let Some(dependency) = emit_dependency_metadata(find_cuda_runtime()) {
        emit_static_link_libraries(&["cudart_static"]);
        let include = env::var_os("CUDA_PATH")
            .map(|root| PathBuf::from(root).join("include"))
            .filter(|path| path.join("cuda_runtime.h").is_file())
            .unwrap_or(dependency.include_path);
        println!(
            "cargo:rustc-env=NCCL_CUDA_INCLUDE_DIR={}",
            include.display()
        );
        let cccl = include.join("cccl");
        if cccl.join("cuda/atomic").is_file() {
            println!("cargo:rustc-env=NCCL_CCCL_INCLUDE_DIR={}", cccl.display());
        }
    }

    if let Some(dependency) = emit_dependency_metadata(find_nccl()) {
        println!(
            "cargo:rustc-env=NCCL_INCLUDE_DIR={}",
            dependency.include_path.display()
        );
        emit_link_libraries(&["nccl"]);
    }
}
