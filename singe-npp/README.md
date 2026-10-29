# singe-npp

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-npp.svg)](https://crates.io/crates/singe-npp)
[![Documentation](https://docs.rs/singe-npp/badge.svg)](https://docs.rs/singe-npp)
![License](https://img.shields.io/crates/l/singe-npp.svg)

Safe Rust wrappers for NVIDIA Performance Primitives (NPP), for image and signal processing on the GPU.

This crate provides:

- Typed device images, channel layouts, pixel views, and signal buffers.
- Image pipelines for resizing, color conversion, filtering, and segmentation.
- Signal processing, statistics, geometry, and labeling operations.
- Stream contexts and reusable temporary workspaces.

## Example

Create an RGB image and convert it to grayscale:

```rust
use singe_cuda::context::Context;
use singe_npp::{C3, Image, ImagePipeline, StreamContext, Workspace, types::Size};

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let context = Context::create()?;
	let stream = context.create_stream()?;
	let stream_context = StreamContext::create(&stream)?;
	let mut workspace = Workspace::create();
	let image = Image::<u8, C3>::filled(&stream_context, Size::new(16, 16), 128)?;
	let gray = ImagePipeline::from_owned(&stream_context, &mut workspace, image)?
		.rgb_to_gray()?
		.finish()?;
	let pixels = gray.copy_to_host_vec()?;
	assert_eq!(pixels.len(), 16 * 16);
	Ok(())
}
```

`C3` describes three channels per pixel.
Pipeline methods carry the element type and channel layout in the image's type, and `Workspace` reuses scratch memory between calls.
Each method waits for its GPU work before returning.

## More examples

- [Image pipeline](examples/pipeline.rs): crop, resize, convert, filter, and segment an image.
- [Canny edges](examples/npp_canny.rs): detect image edges.
- [Distance transform](examples/distance_transform.rs): compute distances from image features.
- [Contours](examples/find_contour.rs): label regions and extract contour geometry.
- [Watershed](examples/watershed_segmentation.rs): segment an image into regions.

See the [API reference](https://docs.rs/singe-npp).
