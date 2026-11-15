//! Fill a pinned vector, copy it through the device, and reuse the pool's memory.

use std::error::Error;

use singe_cuda::{
    context::Context,
    memory::{DeviceMemory, PinnedVec},
};

fn main() -> Result<(), Box<dyn Error>> {
    let context = Context::create()?;
    let stream = context.create_stream()?;
    let pinned = context.pinned_host();

    // A pinned vector grows like any `Vec`, from blocks the context's pool hands out.
    let mut input: PinnedVec<f32> = Vec::new_in(pinned.clone());
    input.extend((0..4096_u16).map(f32::from));

    let mut device = DeviceMemory::<f32>::zeroed(&context, input.len())?;
    stream.sync_scope(|scope| device.copy_from_pinned_async(&input, scope))?;

    // The constructors return CUDA's error where a growing `Vec` would abort.
    let mut output = pinned.zeroed::<f32>(input.len())?;
    device.copy_to_pinned(&mut output)?;
    assert_eq!(output, input);

    // Dropping a pinned vector returns its block to the pool without calling CUDA.
    drop((input, output));
    println!("pinned pool after the copies: {:?}", pinned.stats());
    pinned.release_cached()?;
    println!("pinned pool after release: {:?}", pinned.stats());
    Ok(())
}
