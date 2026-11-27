//! `UperNet` over a `ConvNeXt`: each stage's map layer-normed, a pyramid of the last map's averages at several scales, then a feature pyramid from the coarsest map down, fused at the finest map's size and classified, and resized to the image.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult, Tensor,
    nn::{Activation, Conv2d, Conv2dConfig, Layer, LayerNorm, ModelState},
    tensor::InterpolateMode,
};

use crate::{
    assembly::{ConvNext, ConvNextConfig, ConvNorm, channel_norm},
    error::Result,
    format::Loader,
    head::{Boxed, ClassifierConfig, HeadLoader, ImageSegmenter, SegmenterLayout},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

/// The epsilon of the norm of each stage's map, which `ConvNextBackbone` fixes.
const STAGE_NORM_EPS: f32 = 1e-6;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UperNetConfig {
    pub backbone: ConvNextConfig,
    /// The sides the pyramid averages the last map to.
    pub pool_scales: Vec<i64>,
    /// The channels of every map in the head.
    pub width: i64,
    pub classifier: ClassifierConfig,
}

/// `UperNet` and its backbone: from images to each label's score at each of their pixels, `[B, labels, H, W]`.
#[derive(Debug, Clone)]
pub struct UperNetLayers {
    backbone: ConvNext,
    stage_norms: Vec<LayerNorm>,
    pool_scales: Vec<i64>,
    pyramid: Vec<ConvNorm>,
    bottleneck: ConvNorm,
    laterals: Vec<ConvNorm>,
    outputs: Vec<ConvNorm>,
    fusion: ConvNorm,
    classifier: Conv2d,
}

impl UperNetLayers {
    pub fn create(schema: &mut Schema<'_>, config: &UperNetConfig) -> SingeResult<Self> {
        let width = config.width;
        let widths = &config.backbone.widths;
        let last = widths.last().copied().unwrap_or_default();
        let conv = |schema: &mut Schema<'_>, scope: &str, input: i64, kernel: i64| {
            let names = ("convolution", "norm");
            ConvNorm::create(&mut schema.scope(scope), names, input, width, kernel, 1)
                .map(|layer| layer.with_activation(Activation::Relu))
        };
        let stage_norms = widths
            .iter()
            .enumerate()
            .map(|(index, &channels)| {
                schema.layer_norm(&format!("stage_norms.{index}"), channels, STAGE_NORM_EPS)
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let pyramid = (0..config.pool_scales.len())
            .map(|index| conv(schema, &format!("head.pyramid.{index}"), last, 1))
            .collect::<SingeResult<Vec<_>>>()?;
        let pooled = i64::try_from(config.pool_scales.len()).unwrap_or(i64::MAX);
        let finer = &widths[..widths.len() - 1];
        let laterals = finer
            .iter()
            .enumerate()
            .map(|(index, &channels)| conv(schema, &format!("head.laterals.{index}"), channels, 1))
            .collect::<SingeResult<Vec<_>>>()?;
        let outputs = (0..finer.len())
            .map(|index| conv(schema, &format!("head.outputs.{index}"), width, 3))
            .collect::<SingeResult<Vec<_>>>()?;
        let levels = i64::try_from(widths.len()).unwrap_or(i64::MAX);
        Ok(Self {
            backbone: ConvNext::create(&mut schema.scope("backbone"), &config.backbone)?,
            stage_norms,
            pool_scales: config.pool_scales.clone(),
            pyramid,
            bottleneck: conv(schema, "head.bottleneck", last + pooled * width, 3)?,
            laterals,
            outputs,
            fusion: conv(schema, "head.fusion", levels * width, 3)?,
            classifier: schema.conv2d(
                "head.classifier",
                width,
                config.classifier.label_count()?,
                1,
                Conv2dConfig::default(),
                true,
            )?,
        })
    }
}

/// Returns `map` resized bilinearly to the height and width of `like`, without aligned corners.
fn resized(map: &Tensor, like: &Tensor) -> SingeResult<Tensor> {
    map.interpolate(
        (like.dimension(2)?, like.dimension(3)?),
        InterpolateMode::Bilinear,
    )
}

impl Layer for UperNetLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let maps = self
            .backbone
            .stages(input, state)?
            .iter()
            .zip(&self.stage_norms)
            .map(|(map, norm)| channel_norm(map, norm, state))
            .collect::<SingeResult<Vec<_>>>()?;
        let last = &maps[maps.len() - 1];
        let mut pyramid = vec![last.clone()];
        for (&scale, layer) in self.pool_scales.iter().zip(&self.pyramid) {
            let pooled = last
                .adaptive_avg_pool2d((scale, scale))?
                .apply(layer, state)?;
            pyramid.push(resized(&pooled, last)?);
        }

        let mut laterals = self
            .laterals
            .iter()
            .zip(&maps)
            .map(|(layer, map)| map.apply(layer, state))
            .collect::<SingeResult<Vec<_>>>()?;
        laterals.push(Tensor::concat(&pyramid, 1)?.apply(&self.bottleneck, state)?);
        // From the coarsest map down, each lateral adds the one above it, resized.
        for index in (1..laterals.len()).rev() {
            let above = resized(&laterals[index], &laterals[index - 1])?;
            laterals[index - 1] = laterals[index - 1].add(&above)?;
        }

        let mut outputs = self
            .outputs
            .iter()
            .zip(&laterals)
            .map(|(layer, lateral)| lateral.apply(layer, state))
            .collect::<SingeResult<Vec<_>>>()?;
        outputs.push(laterals[laterals.len() - 1].clone());
        let finest = outputs[0].clone();
        for output in outputs.iter_mut().skip(1) {
            *output = resized(output, &finest)?;
        }

        let logits = Tensor::concat(&outputs, 1)?
            .apply(&self.fusion, state)?
            .apply(&self.classifier, state)?;
        resized(&logits, input)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "upernet",
    formats: &[(Format::Transformers, &["upernet"])],
    loaders: &[HeadLoader::ImageSegmentation(load_segmenter)],
};

pub fn load_segmenter(loader: &mut Loader<'_>) -> Result<ImageSegmenter> {
    let config: UperNetConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        UperNetLayers::create(schema, &config).map(|layers| Boxed::<dyn Layer>(Box::new(layers)))
    })?;
    let model = loader.load_whole(&model, &transformers::rules(config.backbone.depths.len()))?;
    Ok(ImageSegmenter::new(
        model,
        config.classifier,
        SegmenterLayout::Semantic,
    ))
}
