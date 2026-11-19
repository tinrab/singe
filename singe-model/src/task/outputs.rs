use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GeneratedText {
    pub generated_text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GeneratedSequence {
    pub generated_text: String,
    pub generated_token_ids: Vec<i64>,
}

// A clip's samples and their rate, as `transformers`' audio pipelines give them.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GeneratedAudio {
    pub audio: Vec<f32>,
    pub sampling_rate: i64,
}
