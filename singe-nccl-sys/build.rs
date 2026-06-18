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
    if emit_dependency_metadata(find_cuda_runtime()).is_some() {
        emit_static_link_libraries(&["cudart_static"]);
    }

    if emit_dependency_metadata(find_nccl()).is_some() {
        emit_link_libraries(&["nccl"]);
    }
}
