singe-onnx
===

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->
[![Latest version](https://img.shields.io/crates/v/singe-onnx.svg)](https://crates.io/crates/singe-onnx)
[![Documentation](https://docs.rs/singe-onnx/badge.svg)](https://docs.rs/singe-onnx)
![License](https://img.shields.io/crates/l/singe-onnx.svg)

ONNX model utilities.

## Examples

Parse an ONNX file into Singe's typed model representation.

```rust
use singe_onnx::Result;

fn main() -> Result<()> {
    let model = singe_onnx::read_path("model.onnx")?;

    println!("producer: {}", model.producer_name);
    println!("graph: {}", model.graph.name);
    println!("inputs: {}", model.graph.inputs.len());
    println!("outputs: {}", model.graph.outputs.len());
    println!("nodes: {}", model.graph.nodes.len());
    println!("initializers: {}", model.graph.initializers.len());

    for node in &model.graph.nodes {
        println!("{} {:?}", node.operator_type, node.outputs);
    }

    Ok(())
}
```

For large models, you can use the structure-only parser.
This wont materialize inline tensor payloads.

```rust
use singe_onnx::Result;

fn main() -> Result<()> {
    let model = singe_onnx::read_structure_path("model.onnx")?;

    for initializer in &model.graph.initializers {
        println!(
            "{} {:?} {:?}",
            initializer.name,
            initializer.data_type,
            initializer.dimensions
        );
        assert!(initializer.data.is_none());
    }

    Ok(())
}
```
