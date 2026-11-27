# singe-npp

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-npp.svg)](https://crates.io/crates/singe-npp)
[![Documentation](https://docs.rs/singe-npp/badge.svg)](https://docs.rs/singe-npp)
![License](https://img.shields.io/crates/l/singe-npp.svg)

Safe Rust wrappers for NPP, the NVIDIA Performance Primitives, for image and signal processing on the GPU.

This crate provides:

- **Images and signals:** device images typed by element and channel layout, such as `Image<u8, C3>`, borrowed views of device memory, and signal buffers.
- **Image pipelines:** `ImagePipeline` chains resizing, color conversion, thresholding, filtering, and segmentation, and its type tracks the element type and channel layout after each step.
- **Signal pipelines:** `SignalPipeline` chains arithmetic, conversion, filtering, and statistics over device signals.
- **Execution:** `StreamContext` runs NPP on a CUDA stream, and `Workspace` reuses scratch memory between calls.

## Example

Create an RGB image and convert it to grayscale:

```rust,no_run
use singe_cuda::{context::Context, stream::StreamOptions};
use singe_npp::{C3, Image, ImagePipeline, StreamContext, Workspace, types::Size};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = Context::create()?;
    let stream = context.create_stream(StreamOptions::default())?;
    let stream_context = StreamContext::create(&stream)?;
    let mut workspace = Workspace::create();

    let image = Image::<u8, C3>::filled(&stream_context, Size::new(16, 16), 128)?;
    let gray = ImagePipeline::from_owned(&stream_context, &mut workspace, image)?
        .rgb_to_gray()?
        .finish()?;
    assert_eq!(gray.copy_to_host_vec()?.len(), 16 * 16);
    Ok(())
}
```

`C3` is a layout with three channels per pixel, and `rgb_to_gray` returns a pipeline over a one-channel `C1` image.
Each pipeline step waits for its GPU work before returning.

## More examples

- [Image pipeline](./examples/pipeline.rs): crop, resize, convert, filter, and segment an image.
- [Canny edges](./examples/npp_canny.rs): detect edges in an image.
- [Distance transform](./examples/distance_transform.rs): compute each pixel's distance to the nearest feature.
- [Contours](./examples/find_contour.rs): label regions and extract contour geometry.
- [Watershed](./examples/watershed_segmentation.rs): segment an image into regions.
- [Batched labeling](./examples/batched_label_markers_and_compression.rs): label connected regions in several images in one batch and compress the labels.
