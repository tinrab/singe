use singe_cuda_find::{
    emit_dependency_metadata, emit_library_path_metadata, emit_link_libraries,
    emit_rerun_if_env_changed, find_cuda_driver_path, find_cufile,
};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    emit_rerun_if_env_changed();

    link_libraries();
}

fn link_libraries() {
    if emit_library_path_metadata(find_cuda_driver_path()).is_some() {
        emit_link_libraries(&["cuda"]);
    }

    if emit_dependency_metadata(find_cufile()).is_some() {
        emit_link_libraries(&["cufile"]);
    }
}
