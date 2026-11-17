# singe-onnx

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-onnx.svg)](https://crates.io/crates/singe-onnx)
[![Documentation](https://docs.rs/singe-onnx/badge.svg)](https://docs.rs/singe-onnx)
![License](https://img.shields.io/crates/l/singe-onnx.svg)

Decodes ONNX models from their protobuf encoding into typed Rust values.
Singe's ONNX import reads its models through this crate.

This crate provides:

- **Parsing:** `Model::from_path` and `Model::from_bytes` parse a whole model, including inline tensor data.
- **Structure-only parsing:** `Model::structure_from_bytes` skips inline tensor payloads, so a large model's weights are never allocated.
- **Typed model:** `Model`, `Graph`, `Node`, `Attribute`, `Tensor`, `ValueInfo`, `Type`, and the other ONNX values, with optional fields as `Option` and metadata as a sorted `PropertyMap` that rejects duplicate keys.
- **Raw protobuf:** `proto`, the `prost` types generated from the ONNX schema, and `TryFrom` conversions from them into `Model`.
- **Errors:** `Error` separates read failures (`Io`), invalid protobuf bytes (`Decode`), and decoded messages that break the typed model's rules (`Format`).

## Example

Parse the structure of the model file given as the first argument, and print its operator sets and nodes:

```rust,no_run
use std::{env, fs};

use singe_onnx::Model;

fn main() -> anyhow::Result<()> {
    let path = env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("pass the path of an ONNX model"))?;
    let model = Model::structure_from_bytes(&fs::read(path)?)?;
    println!(
        "{} {}, IR version {}",
        model.producer_name, model.producer_version, model.ir_version
    );
    for import in &model.operator_set_imports {
        let domain = import.domain.as_deref().unwrap_or("ai.onnx");
        println!("operator set {domain} version {}", import.version);
    }
    for node in &model.graph.nodes {
        println!(
            "{} ({}): {:?} -> {:?}",
            node.name, node.operator_type, node.inputs, node.outputs
        );
    }
    Ok(())
}
```

Use `Model::from_path` or `Model::from_bytes` when you need the initializers' tensor data.
