// ColModernVBert: ModernVBert's SigLIP tower and ModernBERT encoder, which reads an image's rows in place of its image tokens, and a projection of each final state to a unit embedding a query and a document page compare by late interaction.
// Its images are cut into tiles as SmolVLM's are, through the same tower and connector.

mod modernbert;
mod transformers;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use singe::{
    DataType, Device, Error as SingeError, Result as SingeResult, Shape, Tensor,
    nn::{Layer, Linear, Model, ModelState, StateBuilder},
};

pub use self::modernbert::{ModernBert, ModernBertConfig};
use crate::{
    error::{Error, Result},
    format::Loader,
    head::{DocumentRetriever, HeadLoader},
    model::{
        ModelEntry,
        multimodal::smolvlm::{SmolVlmPreprocessing, SmolVlmVision, SmolVlmVisionConfig},
    },
    processing::{
        chat::{ChatTemplate, Content, Message, Part},
        image::Image,
        text::TextTokenizer,
    },
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColModernVBertConfig {
    pub vision: SmolVlmVisionConfig,
    pub text: ModernBertConfig,
    pub image_token_id: i64,
    // The width of each token's embedding.
    pub embedding: i64,
    pub processor: SmolVlmPreprocessing,
    // The template `sentence-transformers` renders queries and pages through.
    pub template: String,
}

// The encoder and the projection to each token's embedding.
#[derive(Debug, Clone)]
pub struct ColModernVBertText {
    encoder: ModernBert,
    projection: Linear,
}

impl ColModernVBertText {
    // Each token's unit embedding, `[T, D]`, from the sequence's embeddings, `[T, W]`.
    fn embeddings(&self, embedded: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let projected = self
            .encoder
            .encode(embedded, state)?
            .apply(&self.projection, state)?
            .cast(DataType::F32)?;
        let norms = projected
            .mul(&projected)?
            .sum(&[1], true)?
            .sqrt()?
            .clamp(1e-12, f32::MAX)?;
        projected.div(&norms)
    }
}

impl Layer for ColModernVBertText {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        self.embeddings(&self.encoder.embed(input, state)?, state)
    }
}

#[derive(Debug)]
pub struct ColModernVBert {
    config: ColModernVBertConfig,
    vision: Model<SmolVlmVision>,
    text: Model<ColModernVBertText>,
    template: ChatTemplate,
    device: Device,
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "colmodernvbert",
    formats: &[(Format::Transformers, &["modernvbert"])],
    heads: &[HeadLoader::DocumentRetrieval(|loader| {
        Ok(Box::new(load(loader)?))
    })],
};

// Loads the vision tower, as part `vision`, and the encoder with its projection, as part `text`.
pub fn load(loader: &mut Loader<'_>) -> Result<ColModernVBert> {
    let config: ColModernVBertConfig = loader.config(transformers::config)?;
    let data_type = loader.options().data_type;
    let mut builder = StateBuilder::new();
    let vision = {
        let mut schema = Schema::root(&mut builder, data_type);
        SmolVlmVision::create(&mut schema, &config.vision, config.text.width)?
    };
    let vision = loader.load(
        "vision",
        &Model::new(vision, builder.finish()?),
        &transformers::vision_rules(),
    )?;
    let mut builder = StateBuilder::new();
    let text = {
        let mut schema = Schema::root(&mut builder, data_type);
        ColModernVBertText {
            encoder: ModernBert::create(&mut schema, &config.text)?,
            projection: schema.linear("projection", config.text.width, config.embedding, true)?,
        }
    };
    let text = loader.load(
        "text",
        &Model::new(text, builder.finish()?),
        &transformers::text_rules(),
    )?;
    loader.finish()?;
    let template = ChatTemplate::load(loader.source())?
        .ok_or_else(|| Error::unsupported("a retriever without a chat template"))?
        .with_template(&config.template);
    Ok(ColModernVBert {
        config,
        vision,
        text,
        template,
        device: loader.options().device,
    })
}

impl ColModernVBert {
    // Renders one user message holding `content` for `task`, as `sentence-transformers` does.
    fn render(&self, content: Content, task: &str) -> Result<String> {
        let mut arguments = Map::new();
        arguments.insert("task".to_owned(), Value::String(task.to_owned()));
        self.template
            .render(&[Message::new("user", content)], false, &arguments)
    }

    fn tokens(&self, tokenizer: &TextTokenizer, text: &str) -> Result<(Vec<i64>, Tensor)> {
        let tokens = tokenizer.encode(text, true)?;
        let count = i64::try_from(tokens.len()).map_err(|_| SingeError::out_of_range("tokens"))?;
        let tensor = Tensor::from_vec(self.device, Shape::create([count])?, tokens.clone())?;
        Ok((tokens, tensor))
    }
}

impl DocumentRetriever for ColModernVBert {
    // The query, followed by the end-of-utterance tokens the template appends to a query.
    fn query(&self, tokenizer: &TextTokenizer, query: &str) -> Result<Tensor> {
        let text = self.render(Content::Text(query.to_owned()), "query")?;
        let (_, tokens) = self.tokens(tokenizer, &text)?;
        Ok(self.text.forward(&tokens)?)
    }

    // The page as the template's image prompt, its placeholder expanded into its tiles' image tokens, which read the tower's rows.
    fn document(&self, tokenizer: &TextTokenizer, image: &Image) -> Result<Tensor> {
        let processor = &self.config.processor;
        let prompt = self.render(
            Content::Parts(vec![Part::Image {
                image: String::new(),
            }]),
            "document",
        )?;
        let (tiles, grid) = processor.tiles(image);
        let text = processor.expand(
            &prompt,
            &processor.image_token,
            &[processor.image_prompt(grid)],
        )?;
        let (tokens, tensor) = self.tokens(tokenizer, &text)?;
        let pixels = processor.pixels(&tiles, self.device, self.vision.state().floating_type()?)?;
        let rows = self.vision.forward(&pixels)?;
        let dimensions = rows.shape().dimensions();
        let rows = rows.reshape([dimensions[0] * dimensions[1], dimensions[2]])?;
        let state = self.text.state();
        let text = self.text.layer();
        let embedded = text.encoder.embed(&tensor, state)?;
        // Each place reads its token's embedding, or the next of the image's rows, which follow the tokens' in one table.
        let count = i64::try_from(tokens.len()).map_err(|_| SingeError::out_of_range("tokens"))?;
        let mut next = count;
        let places = tokens
            .iter()
            .zip(0..)
            .map(|(&token, place)| {
                if token == self.config.image_token_id {
                    next += 1;
                    next - 1
                } else {
                    place
                }
            })
            .collect::<Vec<i64>>();
        if next - count != rows.shape().dimensions()[0] {
            return Err(SingeError::invalid_argument(format!(
                "a page of {} image tokens for {} image rows",
                next - count,
                rows.shape().dimensions()[0]
            ))
            .into());
        }

        let places = Tensor::from_vec(self.device, Shape::create([count])?, places)?;
        let rows = rows.cast(embedded.data_type())?;
        let embedded = Tensor::concat(&[embedded, rows], 0)?.index_select(0, &places)?;
        Ok(text.embeddings(&embedded, state)?)
    }
}
