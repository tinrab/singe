// `text-to-3d`: a prompt and a seed to a colored mesh, as `diffusers`' `ShapEPipeline` generates one: the prompt's CLIP embedding, normed and scaled, conditions the prior's denoising of a latent under Heun's scheduler, with classifier-free guidance against a zero embedding, and the renderer meshes the latent.

use serde::Deserialize;
use singe::{
    DataType, Tensor,
    nn::{Model, TokenBatch},
};

use crate::{
    error::Result,
    head::TextEmbeddingLayers,
    model::{
        diffusion::shap_e::{
            Mesh, Prior, Renderer, decode_mesh, load_prior, load_renderer, predict,
        },
        multimodal::clip::load_text_with_projection,
    },
    processing::{
        random::TorchGenerator,
        scheduler::{HeunScheduler, SchedulerConfig},
        text::TextTokenizer,
    },
    source::{LoadOptions, Source},
    task::{
        Pipeline, Task,
        vision::text_to_image::{guide_pair, prompt_tokens},
    },
};

// Shap-E's prior, renderer, and scheduler, which its text and image pipelines share.
#[derive(Debug)]
pub struct ShapE {
    prior: Prior,
    renderer: Renderer,
    scheduler: SchedulerConfig,
    options: LoadOptions,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub struct ShapEParameters {
    pub num_inference_steps: usize,
    pub guidance_scale: f32,
}

impl Default for ShapEParameters {
    fn default() -> Self {
        Self {
            num_inference_steps: 25,
            guidance_scale: 4.0,
        }
    }
}

impl ShapE {
    pub fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            prior: load_prior(&source.component("prior")?, options)?,
            renderer: load_renderer(&source.component("shap_e_renderer")?, options)?,
            scheduler: SchedulerConfig::load(source)?,
            options,
        })
    }

    // A mesh from the latent the prior denoises from `seed`'s noise, conditioned on `condition`, `[1, C]` or `[1, K, C]`, and guided away from a zero condition.
    pub fn generate(
        &self,
        condition: &Tensor,
        seed: u64,
        parameters: &ShapEParameters,
    ) -> Result<Mesh> {
        let mut scheduler = HeunScheduler::create(&self.scheduler, parameters.num_inference_steps)?;
        let config = self.prior.layer().config();
        let (count, width) = (config.embeddings, config.embedding_width);
        let mut latent = TorchGenerator::seeded(seed)
            .normal_tensor(self.options.device, &[1, count, width])?
            .mul_scalar(f64::from(scheduler.initial_scale()))?;
        let guided = parameters.guidance_scale > 1.0;
        let conditions = if guided {
            Tensor::concat(&[condition.zeros_like()?, condition.clone()], 0)?
        } else {
            condition.clone()
        };
        for (index, timestep) in scheduler.timesteps().to_vec().into_iter().enumerate() {
            let input = if guided {
                Tensor::concat(&[latent.clone(), latent.clone()], 0)?
            } else {
                latent.clone()
            };
            let predicted = predict(
                &self.prior,
                &scheduler.scale_input(index, &input)?,
                timestep,
                &conditions,
            )?;
            let predicted = if guided {
                guide_pair(&predicted, parameters.guidance_scale)?
            } else {
                predicted
            };
            latent = scheduler.step(index, &predicted, &latent)?;
        }
        decode_mesh(&self.renderer, &latent.squeeze(0)?)
    }
}

#[derive(Debug)]
pub struct TextTo3dPipeline {
    shap_e: ShapE,
    text_encoder: Model<TextEmbeddingLayers>,
    tokenizer: TextTokenizer,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct TextTo3dInput {
    pub prompt: String,
    pub seed: u64,
}

impl Pipeline for TextTo3dPipeline {
    const TASK: Task = Task::TextTo3d;

    type Input = TextTo3dInput;

    type Parameters = ShapEParameters;

    type Output = Mesh;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            shap_e: ShapE::load(source, options)?,
            text_encoder: load_text_with_projection(&source.component("text_encoder")?, options)?,
            tokenizer: TextTokenizer::load(source.directory().join("tokenizer"))?,
        })
    }

    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>> {
        inputs
            .iter()
            .map(|input| {
                // The end-of-text state, which causal attention leaves the padding after it unread, so the tokens go unpadded.
                let tokens = prompt_tokens(&self.tokenizer, &input.prompt)?;
                let batch = TokenBatch::from_sequences(self.shap_e.options.device, &[&tokens])?;
                let embedding = self
                    .text_encoder
                    .forward_group(&batch)?
                    .cast(DataType::F32)?;
                let norm = embedding.mul(&embedding)?.sum(&[1], true)?.sqrt()?;
                #[expect(clippy::cast_precision_loss, reason = "widths are small")]
                let scale = (embedding.shape().dimensions()[1] as f64).sqrt();
                let condition = embedding.div(&norm)?.mul_scalar(scale)?;
                self.shap_e.generate(&condition, input.seed, parameters)
            })
            .collect()
    }
}
