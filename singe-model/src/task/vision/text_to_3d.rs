//! `text-to-3d`: a prompt and a seed to a colored mesh, as `diffusers`' `ShapEPipeline` generates one: the prompt's CLIP embedding, normed and scaled, conditions the prior's denoising of a latent under the snapshot's scheduler, Heun's in Shap-E's, with classifier-free guidance against a zero embedding, and the renderer meshes the latent.

use singe::{DataType, Error as SingeError, nn::TokenBatch};

use crate::{
    error::Result,
    head::TextEncoder,
    model::{diffusion::shap_e::Mesh, multimodal::clip::load_text_with_projection},
    processor::text::TextTokenizer,
    source::{LoadOptions, Source},
    task::{
        Pipeline,
        diffusion::{ShapE, ShapEParameters, prompt_tokens},
        inputs::SeededPrompt,
    },
};

#[derive(Debug)]
pub struct TextTo3dPipeline {
    shap_e: ShapE,
    text_encoder: TextEncoder,
    tokenizer: TextTokenizer,
}

impl Pipeline for TextTo3dPipeline {
    type Input = SeededPrompt;

    type Parameters = ShapEParameters;

    type Output = Mesh;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            shap_e: ShapE::load(source, options)?,
            text_encoder: load_text_with_projection(&source.component("text_encoder")?, options)?,
            tokenizer: TextTokenizer::load(&source.subdirectory("tokenizer"))?,
        })
    }

    fn run_one(&self, input: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        // The end-of-text state, which causal attention leaves the padding after it unread, so the tokens go unpadded.
        let tokens = prompt_tokens(&self.tokenizer, &input.prompt)?;
        let batch = TokenBatch::from_sequences(self.shap_e.options().device, &[&tokens])?;
        let embedding = self
            .text_encoder
            .forward_group(&batch)?
            .cast(DataType::F32)?;
        let &[_, width] = embedding.shape().dimensions() else {
            return Err(SingeError::invalid_argument("text embeddings, [1, D]").into());
        };
        let norm = embedding.mul(&embedding)?.sum(&[1], true)?.sqrt()?;
        let scale = (width as f64).sqrt();
        let condition = embedding.div(&norm)?.mul_scalar(scale)?;
        self.shap_e.generate(&condition, input.seed, parameters)
    }
}
