singe-ptx
===

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->
[![Latest version](https://img.shields.io/crates/v/singe-ptx.svg)](https://crates.io/crates/singe-ptx)
[![Documentation](https://docs.rs/singe-ptx/badge.svg)](https://docs.rs/singe-ptx)
![License](https://img.shields.io/crates/l/singe-ptx.svg)

CUDA PTX parser, AST, and generated instruction metadata utilities.

This crate parses PTX assembly into a Rust AST and exposes generated metadata for known PTX instructions.

## Examples

```rust,ignore
let ptx = r#"
.version 9.1
.target sm_90
.address_size 64

.visible .entry add(.param .u64 data) {
    ret;
}
"#;

let module = singe_ptx::parser::parse_module(ptx)?;
assert_eq!(module.version.major, 9);
assert_eq!(module.items.len(), 1);
```
