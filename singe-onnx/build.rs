use std::{error::Error, path::PathBuf};

use bomboni_prost::{ApiConfig, config::CompileConfig};

fn main() -> Result<(), Box<dyn Error>> {
    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR is set by cargo"));
    let descriptor_path = out_dir.join("onnx.bin");
    let proto_root = PathBuf::from("proto");
    let proto_paths = [proto_root.join("onnx.proto")];

    let mut prost_config = prost_build::Config::default();
    prost_config
        .file_descriptor_set_path(&descriptor_path)
        .enable_type_names()
        .protoc_arg("--experimental_allow_proto3_optional");
    prost_config.compile_protos(&proto_paths, &[proto_root.as_path()])?;

    bomboni_prost::compile(CompileConfig {
        file_descriptor_set_path: descriptor_path,
        output_path: out_dir,
        format: true,
        api: ApiConfig {
            helpers_mod: Some("helpers".into()),
            serde: false,
            ..Default::default()
        },
        external_paths: Default::default(),
    })?;

    for proto_path in proto_paths {
        println!("cargo:rerun-if-changed={}", proto_path.display());
    }

    Ok(())
}
