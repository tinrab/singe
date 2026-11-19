use singe::{DataType, Result, Tensor, generation::Decoder};

// Reads a result back, which waits for the work behind it.
pub fn wait(tensor: &Tensor) -> Result<()> {
    tensor.cast(DataType::F32)?.to_vec::<f32>()?;
    Ok(())
}

// One token per sequence, which is what a decoding step takes.
pub fn decode_step<D: Decoder + ?Sized>(decoder: &mut D) -> Result<()> {
    let rows = vec![[1_i64]; decoder.sequences()];
    let tokens = rows.iter().map(<[i64; 1]>::as_slice).collect::<Vec<_>>();
    wait(&decoder.step(&tokens, None)?)
}

// A step past a decoder's capacity fails, so the decoder starts over with `restart` when it is full.
pub fn decode_step_or_restart<D: Decoder + ?Sized>(
    decoder: &mut D,
    held: &mut usize,
    capacity: usize,
    restart: impl FnOnce(&mut D) -> Result<usize>,
) -> Result<()> {
    if *held + 1 >= capacity {
        decoder.reset();
        *held = restart(decoder)?;
    }

    decode_step(decoder)?;
    *held += 1;
    Ok(())
}
