//! `image-segmentation`: an image to labeled masks of its regions, as `transformers`' pipeline gives them.
//! A semantic segmenter's pixels each take their best label; a universal segmenter's queries become panoptic segments, as `post_process_panoptic_segmentation` makes them.
//! A concept segmenter, as SAM 3 is, finds every instance of what the `text` parameter names, as `post_process_instance_segmentation` keeps them: each query whose score passes the threshold, in query order, labelled with the text.

use std::{collections::BTreeSet, result::Result as StdResult};

use serde::{Deserialize, Serialize};
use singe::{DataType, Shape, Tensor, tensor::InterpolateMode};

use crate::{
    error::{Error, Result},
    head::{ConceptSegmenter, Head, ImageSegmenter, SegmentQueries, SegmenterLayout},
    model,
    processor::{
        image::{Image, ImageTransforms},
        text::TextTokenizer,
    },
    source::{LoadOptions, Source},
    task::{Pipeline, vision::pixels::mask},
};

/// The size `post_process_panoptic_segmentation` resizes the mask logits to before their sigmoid, whatever the model's input.
const PANOPTIC_SIZE: i64 = 384;

#[derive(Debug)]
enum Segmenter {
    Classes(ImageSegmenter),
    Concepts {
        model: ConceptSegmenter,
        tokenizer: Box<TextTokenizer>,
    },
}

#[derive(Debug)]
pub struct ImageSegmentationPipeline {
    model: Segmenter,
    processor: ImageTransforms,
    options: LoadOptions,
}

/// What a universal segmenter's queries must reach to become segments; a semantic segmenter reads none of them.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub struct ImageSegmentationParameters {
    /// The score a query's best label must exceed, 0.9 by default, where SAM 3's own post-processing takes 0.3.
    pub threshold: f32,
    /// The probability at which a query's mask covers a pixel, 0.5 by default.
    pub mask_threshold: f32,
    /// The share of a query's mask that must remain its own after the queries overlap, 0.5 by default.
    pub overlap_mask_area_threshold: f32,
    /// What a concept segmenter looks for, which it requires and other segmenters ignore.
    pub text: Option<String>,
}

impl Default for ImageSegmentationParameters {
    fn default() -> Self {
        Self {
            threshold: 0.9,
            mask_threshold: 0.5,
            overlap_mask_area_threshold: 0.5,
            text: None,
        }
    }
}

/// A region: its label, a score when the model gives one, and its mask, white where the region is.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Segment {
    pub score: Option<f32>,
    pub label: String,
    pub mask: Image,
}

impl Pipeline for ImageSegmentationPipeline {
    type Input = Image;

    type Parameters = ImageSegmentationParameters;

    type Output = Vec<Segment>;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        let is_concepts = source.function().is_none()
            && model::default_head(source)? == Head::ConceptSegmentation;
        Ok(Self {
            model: if is_concepts {
                Segmenter::Concepts {
                    model: model::load(source, options)?,
                    tokenizer: Box::new(TextTokenizer::load(source)?),
                }
            } else {
                Segmenter::Classes(model::load_served(source, options)?)
            },
            processor: ImageTransforms::load(source)?,
            options,
        })
    }

    fn run_one(&self, image: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        let pixels =
            self.processor
                .preprocess(image, self.options.device, self.options.data_type)?;
        let model = match &self.model {
            Segmenter::Classes(model) => model,
            Segmenter::Concepts { model, tokenizer } => {
                return self.concepts(image, &pixels, model, tokenizer, parameters);
            }
        };
        match model.layout() {
            SegmenterLayout::Semantic => Self::semantic(model, image, &pixels),
            SegmenterLayout::Universal { .. } => {
                Self::panoptic(model, image, &model.queries(&pixels)?, parameters)
            }
        }
    }
}

impl ImageSegmentationPipeline {
    /// Keeps each query whose score passes the threshold, its mask probabilities resized bilinearly to the image and cut at the mask threshold.
    fn concepts(
        &self,
        image: &Image,
        pixels: &Tensor,
        model: &ConceptSegmenter,
        tokenizer: &TextTokenizer,
        parameters: &ImageSegmentationParameters,
    ) -> Result<Vec<Segment>> {
        let text = parameters
            .text
            .as_deref()
            .ok_or_else(|| Error::invalid("a concept segmenter without the text it looks for"))?;
        let tokens = tokenizer.encode_batch(&[text], None, self.options.device)?;
        let queries = model.segment(pixels, &tokens)?;
        let size = (i64::from(image.height), i64::from(image.width));
        let threshold = Tensor::from_vec(
            self.options.device,
            Shape::create([1, 1, 1])?,
            vec![parameters.mask_threshold],
        )?;
        let mut segments = Vec::new();
        for (query, &score) in queries.scores.iter().enumerate() {
            if score <= parameters.threshold {
                continue;
            }

            let row = i64::try_from(query).map_err(|_| Error::invalid("concept queries"))?;
            let inside = queries
                .masks
                .narrow(0, row, 1)?
                .cast(DataType::F32)?
                .sigmoid()?
                .unsqueeze(0)?
                .interpolate(size, InterpolateMode::Bilinear)?
                .squeeze(0)?
                .greater(&threshold)?
                .cast(DataType::F32)?
                .contiguous()?
                .to_vec::<f32>()?;
            segments.push(Segment {
                score: Some(score),
                label: text.to_owned(),
                mask: mask(
                    image.width,
                    image.height,
                    inside.into_iter().map(|value| value > 0.5),
                ),
            });
        }
        Ok(segments)
    }

    fn label(model: &ImageSegmenter, id: usize) -> Result<String> {
        model
            .config()
            .labels()
            .get(id)
            .cloned()
            .ok_or_else(|| Error::unsupported(format!("class {id} beyond the config's labels")))
    }

    /// Gives each pixel its best label, after the scores are resized bilinearly to the image, as `post_process_semantic_segmentation` does, and each label present a mask.
    fn semantic(model: &ImageSegmenter, image: &Image, pixels: &Tensor) -> Result<Vec<Segment>> {
        let (_, best) = model
            .scores(pixels)?
            .cast(DataType::F32)?
            .interpolate(
                (i64::from(image.height), i64::from(image.width)),
                InterpolateMode::Bilinear,
            )?
            .topk(1, 1, true, false)?;
        let best = best.contiguous()?.to_vec::<i64>()?;
        let present = best.iter().copied().collect::<BTreeSet<_>>();
        present
            .into_iter()
            .map(|id| {
                let index = usize::try_from(id)
                    .map_err(|_| Error::unsupported(format!("class {id} of a pixel")))?;
                Ok(Segment {
                    score: None,
                    label: Self::label(model, index)?,
                    mask: mask(
                        image.width,
                        image.height,
                        best.iter().map(|&best| best == id),
                    ),
                })
            })
            .collect()
    }

    /// Makes panoptic segments of the first image's queries, as `post_process_panoptic_segmentation` and `compute_segments` do with no labels fused.
    ///
    /// A query is kept when its best score after a softmax exceeds the threshold and is not no object's. The kept queries' mask probabilities, resized to the image and scaled by their scores, give each pixel to the likeliest query.
    /// A query becomes a segment when it holds pixels and holds more than the overlap threshold's share of the pixels its scaled probability alone reaches the mask threshold on.
    fn panoptic(
        model: &ImageSegmenter,
        image: &Image,
        queries: &SegmentQueries,
        parameters: &ImageSegmentationParameters,
    ) -> Result<Vec<Segment>> {
        let classes = queries.classes.cast(DataType::F32)?;
        let &[_, _, columns] = classes.shape().dimensions() else {
            return Err(Error::invalid("segment classes, [B, Q, labels + 1]"));
        };
        let columns = usize::try_from(columns).map_err(|_| Error::invalid("segment classes"))?;
        let probabilities = classes
            .narrow(0, 0, 1)?
            .softmax(-1)?
            .contiguous()?
            .to_vec::<f32>()?;
        let mut kept = Vec::new();
        for (query, row) in probabilities.chunks(columns).enumerate() {
            let (label, score) = row.iter().copied().enumerate().fold(
                (0, f32::NEG_INFINITY),
                |best, (label, score)| {
                    if score > best.1 { (label, score) } else { best }
                },
            );
            if label + 1 != columns && score > parameters.threshold {
                kept.push((query, label, score));
            }
        }

        if kept.is_empty() {
            return Ok(Vec::new());
        }

        let device = queries.masks.device();
        let indices = kept
            .iter()
            .map(|&(query, ..)| i64::try_from(query))
            .collect::<StdResult<Vec<_>, _>>()
            .map_err(|_| Error::invalid("segment queries"))?;
        let kept_count = i64::try_from(kept.len()).map_err(|_| Error::invalid("segments"))?;
        let indices = Tensor::from_vec(device, Shape::create([kept_count])?, indices)?;
        let scores = Tensor::from_vec(
            device,
            Shape::create([1, kept_count, 1, 1])?,
            kept.iter().map(|&(.., score)| score).collect(),
        )?;
        let (height, width) = (i64::from(image.height), i64::from(image.width));
        let scaled = queries
            .masks
            .narrow(0, 0, 1)?
            .cast(DataType::F32)?
            .interpolate((PANOPTIC_SIZE, PANOPTIC_SIZE), InterpolateMode::Bilinear)?
            .sigmoid()?
            .index_select(1, &indices)?
            .interpolate((height, width), InterpolateMode::Bilinear)?
            .mul(&scores)?;
        let (_, owners) = scaled.topk(1, 1, true, false)?;
        let owners = owners.contiguous()?.to_vec::<i64>()?;
        let mask_threshold =
            Tensor::from_vec(device, Shape::create([1])?, vec![parameters.mask_threshold])?;
        let reached = scaled
            .greater_equal(&mask_threshold)?
            .cast(DataType::F32)?
            .sum(&[2, 3], false)?
            .contiguous()?
            .to_vec::<f32>()?;

        let mut segments = Vec::new();
        for (index, &(_, label, score)) in kept.iter().enumerate() {
            let owner = i64::try_from(index).map_err(|_| Error::invalid("segments"))?;
            let held = owners.iter().filter(|&&best| best == owner).count();
            let alone = reached[index];
            if held == 0
                || alone <= 0.0
                || held as f32 / alone <= parameters.overlap_mask_area_threshold
            {
                continue;
            }

            segments.push(Segment {
                score: Some(((f64::from(score) * 1e6).round() / 1e6) as f32),
                label: Self::label(model, label)?,
                mask: mask(
                    image.width,
                    image.height,
                    owners.iter().map(|&best| best == owner),
                ),
            });
        }
        Ok(segments)
    }
}
