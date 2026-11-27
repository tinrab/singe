# singe-nvcomp

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-nvcomp.svg)](https://crates.io/crates/singe-nvcomp)
[![Documentation](https://docs.rs/singe-nvcomp/badge.svg)](https://docs.rs/singe-nvcomp)
![License](https://img.shields.io/crates/l/singe-nvcomp.svg)

Safe Rust wrappers for NVIDIA nvCOMP, for compressing and decompressing data on the GPU with LZ4, Snappy, Zstd, Deflate, Gzip, GDeflate, ANS, Cascaded, and Bitcomp.

This crate provides:

- **One codec type:** `Codec<C>` holds one codec's options, selected by a marker type such as `Lz4` or `Zstd`, and compresses and decompresses device buffers.
- **Whole buffers:** `compress_one` and `decompress_one` treat a buffer as one chunk, and `compress_contiguous` and `decompress_contiguous` split a large buffer into chunks and process them in one batch.
- **Chunk tables:** `compress_chunks` and `compress_chunks_scoped` compress caller-provided chunks, and the `unsafe` `compress_chunks_async` and `decompress_chunks_async` enqueue batched work without waiting.
- **Compressed data:** `CompressedChunk` and `CompressedBatch` keep compressed device bytes together with the options needed to decompress them.
- **Checksums:** CRC32 computed on the GPU.

## Example

Compress device bytes with LZ4 and check that decompressing them gives the input back:

```rust,no_run
use singe_cuda::{context::Context, memory::DeviceMemory, stream::StreamOptions};
use singe_nvcomp::format::{Codec, Lz4};

fn main() -> anyhow::Result<()> {
    let context = Context::create()?;
    let stream = context.create_stream(StreamOptions::default())?;

    let input = b"Hello, nvCOMP!".repeat(1024);
    let device_input = DeviceMemory::from_slice(&context, &input)?;
    let codec = Codec::<Lz4>::default();
    let compressed = codec.compress_one(&device_input, &stream)?;

    let mut output = DeviceMemory::<u8>::create_uninit(&context, input.len())?;
    let output_bytes = codec.decompress_one(&compressed, &mut output, &stream)?;
    // SAFETY: `decompress_one` returned after writing the first `output_bytes` bytes.
    let output = unsafe { output.assume_init_prefix(output_bytes)? };
    assert_eq!(output.copy_to_host_vec()?, input);

    println!("{} bytes compressed to {}", input.len(), compressed.len());
    Ok(())
}
```

`compress_one` and `decompress_one` wait for the GPU before returning, and `decompress_one` returns the number of bytes it wrote.
Decompressing a raw chunk table stays `unsafe`, because nvCOMP documents corrupt compressed input as undefined behavior.

## Features

- `manager`: adds the `manager` module, for nvCOMP's managers, whose output starts with a header that records the codec and sizes, and the `streaming` module, for gzip between Rust readers and writers.
  It builds a C++ bridge that links nvCOMP at build time, found through `NVCOMP_INCLUDE_PATH` and `NVCOMP_PATH`.

## More examples

- [LZ4 round trip](./examples/lz4_roundtrip.rs): compress and decompress one buffer with LZ4.
- [Standard codecs](./examples/standard_codecs_roundtrip.rs): compress and decompress a chunked buffer with ANS, Bitcomp, Cascaded, Deflate, `GDeflate`, LZ4, Snappy, and Zstd.
- [CRC32](./examples/crc32.rs): compute CRC32 checksums of chunks with the `unsafe` batched API.
