use singe_cuda_find::{
    emit_library_path_metadata, emit_link_libraries, emit_rerun_if_env_changed, find_nvml,
};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    emit_rerun_if_env_changed();

    link_libraries();
}

fn link_libraries() {
    if emit_library_path_metadata(find_nvml()).is_some() {
        emit_link_libraries(&["nvidia-ml"]);
    }
}
