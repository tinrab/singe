singe-curand
===

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->
[![Latest version](https://img.shields.io/crates/v/singe-curand.svg)](https://crates.io/crates/singe-curand)
[![Documentation](https://docs.rs/singe-curand/badge.svg)](https://docs.rs/singe-curand)
![License](https://img.shields.io/crates/l/singe-curand.svg)

Safe cuRAND host API wrappers for CUDA random number generation.

This crate wraps generator creation, stream binding, seeding, ordering, and host-side distribution generation over the raw `singe-curand-sys` bindings.

## Examples

```rust,ignore
use singe_curand::{generator::Generator, types::RngType};

fn main() -> singe_curand::error::Result<()> {
    let generator = Generator::create_host(RngType::PseudoDefault)?;
    generator.set_seed(1234)?;

    let mut values = vec![0.0f32; 1024];
    generator.generate_uniform_f32_host(&mut values)?;

    Ok(())
}
```
