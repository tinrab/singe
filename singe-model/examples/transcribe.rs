// Transcribes a mono 16-bit WAV file at 16 kHz with a Whisper or Qwen3-ASR snapshot: `--audio FILE`, `--language` to force one, as Whisper's code or Qwen3-ASR's English name, `--max-new-tokens N`, and `--eager`.

mod support;

use anyhow::Result;
use singe::{DataType, Shape, Tensor};
use singe_model::audio::automatic_speech_recognition::load_transcriber;

use crate::support::{Arguments, read_wav};

fn main() -> Result<()> {
    let arguments = Arguments::parse(&["eager"])?;
    let samples = read_wav(arguments.required("audio")?)?;
    let count = i64::try_from(samples.len())?;
    let samples = Tensor::from_vec(arguments.device, Shape::create([count])?, samples)?;
    let max_new_tokens = arguments
        .value("max-new-tokens")
        .map_or(Ok(200), str::parse)?;
    let model = load_transcriber(&arguments.snapshot, arguments.device, DataType::F32)?;
    let transcription = model.transcribe(
        &samples,
        arguments.value("language"),
        max_new_tokens,
        !arguments.flag("eager"),
    )?;
    if let Some(language) = &transcription.language {
        println!("[{language}]");
    }

    println!("{}", transcription.text);
    Ok(())
}
