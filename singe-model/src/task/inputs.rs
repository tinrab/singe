use serde::Deserialize;

use crate::processor::image::Image;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct SeededPrompt {
    pub prompt: String,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct SeededImage {
    pub image: Image,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct SeededImagePrompt {
    pub image: Image,
    pub prompt: String,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ZeroShotImageInput {
    pub image: Image,
    pub candidate_labels: Vec<String>,
}
