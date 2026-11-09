# singe-nvcomp

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-nvcomp.svg)](https://crates.io/crates/singe-nvcomp)
[![Documentation](https://docs.rs/singe-nvcomp/badge.svg)](https://docs.rs/singe-nvcomp)
![License](https://img.shields.io/crates/l/singe-nvcomp.svg)

Safe Rust wrappers for the NVIDIA nvCOMP compression library.

This crate provides:

- GPU compression and decompression with LZ4, Zstd, Snappy, and other nvCOMP codecs.
- Single-buffer methods that wait for completion, and batched operations over many chunks.
- Compression managers, streaming interfaces, and CRC32 operations.
- Typed chunk tables and explicit workspace management.

## Example

Compress device bytes with LZ4 and verify a round trip:

```rust
use singe_cuda::{context::Context, memory::DeviceMemory};
use singe_nvcomp::format::{Codec, Lz4};

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let context = Context::create()?;
	let stream = context.create_stream()?;
	let input = b"nvCOMP through Singe".repeat(1024);
	let device = DeviceMemory::from_slice(&context, &input)?;
	let codec = Codec::<Lz4>::default();
	let compressed = codec.compress_one(&device, &stream)?;
	let mut output = DeviceMemory::<u8>::create_uninit(&context, input.len())?;
	let written = codec.decompress_one(&compressed, &mut output, &stream)?;
	// SAFETY: the completed helper initialized the reported output prefix.
	let output = unsafe { output.assume_init_prefix(written)? };
	assert_eq!(output.copy_to_host_vec()?, input);
	Ok(())
}
```

`Codec` methods wait for the GPU before returning, and the returned size is the number of output bytes written.
Raw queued operations are `unsafe`: everything they use must stay valid until the work finishes.

## More examples

- [LZ4 round trip](examples/lz4_roundtrip.rs): compress and decompress a buffer.
- [Batched codecs](examples/standard_codecs_roundtrip.rs): exercise multiple compression formats.
- [CRC32](examples/crc32.rs): compute and verify checksums.

See the [API reference](https://docs.rs/singe-nvcomp).
