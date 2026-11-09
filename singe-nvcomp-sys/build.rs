//! Compiles the C++ manager bridge, which calls nvCOMP's C++ API and so links nvCOMP at build time.
//! nvCOMP's C API loads at run time like every other NVIDIA library.

use std::env;

use singe_cuda_find::{
    emit_dependency_metadata, emit_link_libraries, emit_rerun_if_env_changed, find_cuda_runtime,
    find_nvcomp,
};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    emit_rerun_if_env_changed();

    if env::var_os("CARGO_FEATURE_NVCOMP_6_0").is_none() || env::var_os("DOCS_RS").is_some() {
        return;
    }

    let nvcomp = find_nvcomp()
        .expect("failed to find nvcomp")
        .expect("nvcomp headers are required for the c++ manager bridge");
    let cuda = find_cuda_runtime()
        .expect("failed to find cuda runtime")
        .expect("cuda runtime headers are required for the c++ manager bridge");
    println!("cargo:rerun-if-changed=cpp/bridge.h");
    println!("cargo:rerun-if-changed=cpp/bridge.cc");
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("cpp/bridge.cc")
        .include(nvcomp.include_path.clone())
        .include(cuda.include_path)
        .compile("singe_nvcomp_bridge");
    if emit_dependency_metadata(Ok(Some(nvcomp))).is_some() {
        emit_link_libraries(&["nvcomp"]);
    }
}
