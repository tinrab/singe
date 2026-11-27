// Silero VAD v5, each 32 ms window's probability of speech over a clip and over a clip with pauses, as `verify.py` runs its ONNX file in onnxruntime, window by window with the state carried.

use anyhow::Result;
use serde::Deserialize;
use singe_model::{
    LoadOptions, Source,
    model::audio::silero_vad::{SAMPLE_RATE, SileroVad, VadState, WINDOW},
    processor::audio::Audio,
};

use crate::support::{Runner, Session};

struct VoiceActivity(SileroVad);

#[derive(Debug, Default, Deserialize)]
struct NoParameters {}

impl Runner for VoiceActivity {
    const TASK: &'static str = "voice-activity-detection";

    type Input = Audio;

    type Parameters = NoParameters;

    type Output = Vec<f32>;

    fn load(source: &Source, _: LoadOptions) -> Result<Self> {
        Ok(Self(SileroVad::load(source)?))
    }

    fn run(&self, inputs: &[Audio], _: &NoParameters) -> Result<Vec<Vec<f32>>> {
        inputs
            .iter()
            .map(|clip| {
                let clip = clip.at_rate(i64::from(SAMPLE_RATE))?;
                let mut state = VadState::create()?;
                clip.samples
                    .chunks_exact(WINDOW)
                    .map(|window| Ok(self.0.probability(window, &mut state)?))
                    .collect()
            })
            .collect()
    }
}

pub fn verify(session: &mut Session) {
    session.check::<VoiceActivity>("speech");
    session.check::<VoiceActivity>("pauses");
}
