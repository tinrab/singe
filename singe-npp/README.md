singe-npp
===

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->
[![Latest version](https://img.shields.io/crates/v/singe-npp.svg)](https://crates.io/crates/singe-npp)
[![Documentation](https://docs.rs/singe-npp/badge.svg)](https://docs.rs/singe-npp)
![License](https://img.shields.io/crates/l/singe-npp.svg)

Safe NVIDIA Performance Primitives wrappers for image and signal APIs.

This crate wraps NPP version queries, stream contexts, image and signal device
memory, image processing pipelines, signal processing pipelines, typed image
views, channel layouts, sizes, rectangles, and temporary workspaces.

## Examples

Runnable examples are available under [`examples/`](examples/).

### Pipeline

```rust,ignore
use singe_cuda::context::Context;
use singe_npp::{
    C3, ImagePipeline, ImageView, StreamContext, Workspace,
    types::{ComparisonOperation, InterpolationMode, Size},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = Context::create()?;
    let stream = ctx.create_stream()?;
    let stream_context = StreamContext::create(&stream)?;
    let mut workspace = Workspace::create();

    let device_memory = todo!();
    let size = Size::new(640, 480);
    let source = ImageView::<u8, C3>::from_memory(&device_memory, size)?;

    let output = ImagePipeline::from_view(&stream_context, &mut workspace, source)
        .resize(Size::new(256, 256), InterpolationMode::Lanczos)?
        .rgb_to_gray()?
        .threshold(128, ComparisonOperation::Less)?
        .filter_sharpen()?
        .finish()?;
    let image = output.copy_to_host_vec()?;

    println!("NPP version: {:?}", singe_npp::version()?);
    Ok(())
}
```
