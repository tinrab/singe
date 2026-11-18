// The encoder block that norms before attention and before the feed-forward, which Whisper, Qwen3-ASR's audio encoder, the vision transformers, and CLIP repeat, under Singe's names `attention_norm`, `attention` (`query`, `key`, `value`, `output`), `feed_forward_norm`, and `feed_forward` (`up`, `down`).

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult, Tensor,
    nn::{
        Activation, DenseAttentionInput, Layer, LayerNorm, Mlp, ModelState, PreNormBlock,
        SelfAttention,
    },
};

use crate::{format::Rules, schema::Schema};

pub type PreNormEncoderBlock = PreNormBlock<SelfAttention, Mlp, LayerNorm>;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PreNormEncoderSpec {
    pub width: i64,
    pub heads: i64,
    pub inner: i64,
    pub activation: Activation,
    pub eps: f32,
    // Whether the keys have a bias, which Whisper's and VideoMAE's lack.
    pub key_bias: bool,
    // Whether each token attends only to those before it, as CLIP's text encoder does.
    pub causal: bool,
}

// Every projection has a bias but, where the spec says so, the keys'.
pub fn pre_norm_encoder_block(
    schema: &mut Schema<'_>,
    spec: &PreNormEncoderSpec,
) -> SingeResult<PreNormEncoderBlock> {
    let width = spec.width;
    let mut attention = schema.scope("attention");
    let mut attention = SelfAttention::create(
        attention.linear("query", width, width, true)?,
        attention.linear("key", width, width, spec.key_bias)?,
        attention.linear("value", width, width, true)?,
        attention.linear("output", width, width, true)?,
        spec.heads,
        width,
    )?;
    if spec.causal {
        attention = attention.causal();
    }

    let mut feed_forward = schema.scope("feed_forward");
    let mlp = Mlp::new(
        feed_forward.linear("up", width, spec.inner, true)?,
        feed_forward.linear("down", spec.inner, width, true)?,
        spec.activation,
    );
    Ok(PreNormBlock::new(
        schema.layer_norm("attention_norm", width, spec.eps)?,
        attention,
        schema.layer_norm("feed_forward_norm", width, spec.eps)?,
        mlp,
    ))
}

// A stack of pre-norm blocks, `blocks.N`, and, where the model has one, a final `norm`, from tokens `[B, T, W]` to their last hidden states.
#[derive(Debug, Clone)]
pub struct PreNormEncoder {
    blocks: Vec<PreNormEncoderBlock>,
    norm: Option<LayerNorm>,
}

impl PreNormEncoder {
    // Registers `layers` blocks under `blocks` and, with `norm`, a final `norm`.
    pub fn create(
        schema: &mut Schema<'_>,
        spec: &PreNormEncoderSpec,
        layers: i64,
        norm: bool,
    ) -> SingeResult<Self> {
        let blocks = (0..layers)
            .map(|layer| pre_norm_encoder_block(&mut schema.scope(format!("blocks.{layer}")), spec))
            .collect::<SingeResult<Vec<_>>>()?;
        let norm = norm
            .then(|| schema.layer_norm("norm", spec.width, spec.eps))
            .transpose()?;
        Ok(Self { blocks, norm })
    }

    // Runs the blocks over `hidden`, each sequence attending to its first `key_lengths` tokens when given.
    pub fn encode(
        &self,
        hidden: &Tensor,
        key_lengths: Option<&Tensor>,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let mut hidden = hidden.clone();
        for block in &self.blocks {
            hidden = block.forward(
                &DenseAttentionInput {
                    hidden: &hidden,
                    angles: None,
                    key_lengths,
                },
                state,
            )?;
        }
        match &self.norm {
            Some(norm) => hidden.apply(norm, state),
            None => Ok(hidden),
        }
    }
}

impl Layer for PreNormEncoder {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        self.encode(input, None, state)
    }
}

// Renames each layer `from.N.<name>` to `to.N.<target>`, for each pair of `names`.
fn layer_rules(from: &str, to: &str, names: &[(&str, &str)]) -> Rules {
    names.iter().fold(Rules::new(), |rules, (name, target)| {
        rules.rename(&format!("{from}.*.{name}"), &format!("{to}.*.{target}"))
    })
}

// Imports `transformers`' Whisper-style layers under `from` (such as `model.encoder.layers`) into encoder blocks under `to` (such as `blocks`).
#[must_use]
pub fn pre_norm_encoder_rules(from: &str, to: &str) -> Rules {
    layer_rules(
        from,
        to,
        &[
            ("self_attn_layer_norm", "attention_norm"),
            ("self_attn.q_proj", "attention.query"),
            ("self_attn.k_proj", "attention.key"),
            ("self_attn.v_proj", "attention.value"),
            ("self_attn.out_proj", "attention.output"),
            ("final_layer_norm", "feed_forward_norm"),
            ("fc1", "feed_forward.up"),
            ("fc2", "feed_forward.down"),
        ],
    )
}

// Imports `transformers`' ViT-style layers under `from` (such as `vit.encoder.layer`), which ViT, YOLOS, and VideoMAE store.
#[must_use]
pub fn vit_layer_rules(from: &str, to: &str) -> Rules {
    layer_rules(
        from,
        to,
        &[
            ("layernorm_before", "attention_norm"),
            ("attention.attention.query", "attention.query"),
            ("attention.attention.key", "attention.key"),
            ("attention.attention.value", "attention.value"),
            ("attention.output.dense", "attention.output"),
            ("layernorm_after", "feed_forward_norm"),
            ("intermediate.dense", "feed_forward.up"),
            ("output.dense", "feed_forward.down"),
        ],
    )
}

// Imports `transformers`' CLIP-style layers under `from` (such as `text_model.encoder.layers`), which CLIP's and OWL-ViT's towers store.
#[must_use]
pub fn clip_layer_rules(from: &str, to: &str) -> Rules {
    layer_rules(
        from,
        to,
        &[
            ("layer_norm1", "attention_norm"),
            ("self_attn.q_proj", "attention.query"),
            ("self_attn.k_proj", "attention.key"),
            ("self_attn.v_proj", "attention.value"),
            ("self_attn.out_proj", "attention.output"),
            ("layer_norm2", "feed_forward_norm"),
            ("mlp.fc1", "feed_forward.up"),
            ("mlp.fc2", "feed_forward.down"),
        ],
    )
}
