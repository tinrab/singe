//! `BitNet` b1.58: Llama's layout whose projections hold ternary weights scaled by one value each and read their input quantized to 8 bits per token, as `AutoBitLinear` runs offline checkpoints.
//! An RMS norm comes before the attention's output projection and before the feed-forward's down projection, and the feed-forward gate is a squared `ReLU`.
//! The output head is the embedding's, unquantized.

mod transformers;

use singe::{
    DataType, Error as SingeError, Result as SingeResult,
    nn::{
        BitLinear, GatedMlp, GroupedQueryAttention, GroupedQueryLayers, HeadProjection, Linear,
        Model,
    },
};

use crate::{
    assembly::{
        CausalLmSpec, DecoderConfig, GroupedDecoder, causal, causal_lm, pre_norm_block_with,
    },
    error::Result,
    format::Loader,
    head::HeadLoader,
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

pub type BitNet = Model<GroupedDecoder>;

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "bitnet",
    formats: &[(Format::Transformers, &["bitnet"])],
    loaders: &[HeadLoader::CausalLm(|loader| Ok(Box::new(load(loader)?)))],
};

/// Registers `{name}.weight`, the checkpoint's packed `U8` `[output / 4, input]`, and `{name}.scale`, and with `norm`, `{name}_norm` over the input.
fn projection(
    schema: &mut Schema<'_>,
    name: &str,
    [input, output]: [i64; 2],
    norm: Option<f32>,
) -> SingeResult<Linear> {
    if output % 4 != 0 {
        return Err(SingeError::invalid_argument(
            "a BitNet projection whose rows do not pack four to a byte",
        ));
    }

    let norm = norm
        .map(|eps| schema.rms_norm(&format!("{name}_norm"), input, eps))
        .transpose()?;
    let scale = schema.parameter(&format!("{name}.scale"), [1])?;
    let weight =
        schema.parameter_typed(&format!("{name}.weight"), [output / 4, input], DataType::U8)?;
    Ok(Linear::new(weight, None).with_bit_linear(BitLinear {
        norm,
        scale,
        packed: true,
    }))
}

fn attention(
    schema: &mut Schema<'_>,
    config: &DecoderConfig,
) -> SingeResult<GroupedQueryAttention> {
    let (width, head) = (config.width, config.head_dim);
    let (heads, kv_heads) = (config.heads, config.kv_heads);
    let mut head_projection = |name: &str, heads: i64| -> SingeResult<HeadProjection> {
        Ok(HeadProjection::new(
            projection(schema, name, [width, heads * head], None)?,
            heads,
            head,
        ))
    };
    let layers = GroupedQueryLayers {
        query: head_projection("query", heads)?,
        key: head_projection("key", kv_heads)?,
        value: head_projection("value", kv_heads)?,
        output: projection(schema, "output", [heads * head, width], Some(config.eps))?,
    };
    GroupedQueryAttention::create(layers, causal())
}

fn feed_forward(schema: &mut Schema<'_>, config: &DecoderConfig) -> SingeResult<GatedMlp> {
    let width = config.width;
    let inner = config
        .inner
        .ok_or_else(|| SingeError::missing_input("a dense feed-forward layer's width"))?;
    Ok(GatedMlp::new(
        projection(schema, "feed_forward.gate", [width, inner], None)?,
        projection(schema, "feed_forward.up", [width, inner], None)?,
        projection(
            schema,
            "feed_forward.down",
            [inner, width],
            Some(config.eps),
        )?,
        config.activation,
    ))
}

pub fn load(loader: &mut Loader<'_>) -> Result<BitNet> {
    let config: DecoderConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        causal_lm(
            schema,
            &config,
            CausalLmSpec::new(Some(config.rotary()), |_| config.kv_cache()),
            |schema, _| {
                pre_norm_block_with(
                    schema,
                    &config,
                    |schema| attention(schema, &config),
                    |schema| feed_forward(schema, &config),
                )
            },
        )
    })?;
    loader.load_whole(&model, &transformers::rules())
}
