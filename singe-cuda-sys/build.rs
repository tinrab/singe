use singe_cuda_find::{
    emit_dependency_metadata, emit_library_path_metadata, emit_link_libraries,
    emit_rerun_if_env_changed, emit_static_link_libraries, find_cuda_driver_path,
    find_cuda_runtime, find_nvrtc, find_nvvm,
};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    emit_rerun_if_env_changed();

    link_libraries();
}

fn link_libraries() {
    if let Some(cuda_driver_path) = emit_library_path_metadata(find_cuda_driver_path()) {
        #[cfg(target_os = "linux")]
        println!(
            "cargo:rustc-link-arg=-Wl,-rpath,{}",
            cuda_driver_path.display()
        );
        emit_link_libraries(&["cuda"]);
        // println!("cargo:rustc-link-lib=nvrtc");
    }

    if emit_dependency_metadata(find_cuda_runtime()).is_some() {
        // println!("cargo:rustc-link-lib=cudart");
        emit_static_link_libraries(&["cudart_static"]);
    }

    if emit_dependency_metadata(find_nvrtc()).is_some() {
        emit_link_libraries(&["nvrtc"]);
    }

    if emit_dependency_metadata(find_nvvm()).is_some() {
        emit_link_libraries(&["nvvm"]);
    }
}
