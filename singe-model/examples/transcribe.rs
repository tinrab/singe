// Transcribes a mono 16-bit WAV file at 16 kHz with a Whisper snapshot: `--audio FILE`, and `--language`, English by default.

mod support;

use anyhow::Result;
use singe::{DataType, Shape, Tensor};
use singe_model::{
    audio::automatic_speech_recognition::{
        load_transcriber,
        whisper::{TranscribeOptions, WhisperTask, generation_options, prompt},
    },
    tokenization::TextTokenizer,
};

use crate::support::{Arguments, read_wav};

fn main() -> Result<()> {
    let arguments = Arguments::parse(&[])?;
    let samples = read_wav(arguments.required("audio")?)?;
    let count = i64::try_from(samples.len())?;
    let samples = Tensor::from_vec(arguments.device, Shape::create([count])?, samples)?;
    let tokenizer = TextTokenizer::load(&arguments.snapshot)?;
    let language = arguments.value("language").unwrap_or("en");
    let model = load_transcriber(&arguments.snapshot, arguments.device, DataType::F32)?;
    let options =
        TranscribeOptions::new(generation_options(&arguments.snapshot, 200)?).with_compile(true);
    let tokens = model.transcribe(
        &samples,
        &prompt(&tokenizer, language, WhisperTask::Transcribe)?,
        &options,
    )?;
    println!("{}", tokenizer.decode(&tokens, true)?.trim());
    Ok(())
}
