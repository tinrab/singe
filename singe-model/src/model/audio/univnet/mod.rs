//! `UnivNet`: a GAN vocoder that turns noise into a clip through blocks that each upsample it and filter it with location-variable convolutions, whose kernels and biases a predictor computes for each spectrogram frame.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Error as SingeError, Result as SingeResult, Tensor,
    graph::{ComputePrecision, with_convolution_precision},
    nn::{Activation, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, ModelState},
    shape::spatial::Padding,
};

use crate::{
    assembly::same_1d,
    error::{Error, Result},
    format::Loader,
    head::{Boxed, HeadLoader, Vocoder, VocoderLayers},
    model::ModelEntry,
    processor::fbank::AmplitudeMelOptions,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnivNetConfig {
    pub mels: i64,
    pub noise_channels: i64,
    pub hidden: i64,
    pub predictor_hidden: i64,
    pub predictor_blocks: i64,
    pub predictor_kernel: i64,
    pub kernels: Vec<i64>,
    pub strides: Vec<i64>,
    pub dilations: Vec<Vec<i64>>,
    pub leaky_slope: f64,
    pub features: AmplitudeMelOptions,
}

/// The kernel predictor: from the spectrogram, `[B, mels, F]`, each frame's kernels, `[B, layers, in, out, k, F]`, and biases, `[B, layers, out, F]`.
#[derive(Debug, Clone)]
struct KernelPredictor {
    input_conv: Conv1d,
    resblocks: Vec<[Conv1d; 2]>,
    kernel_conv: Conv1d,
    bias_conv: Conv1d,
    shape: [i64; 4],
    slope: f64,
}

impl KernelPredictor {
    fn create(
        schema: &mut Schema<'_>,
        config: &UnivNetConfig,
        kernel: i64,
        layers: i64,
    ) -> SingeResult<Self> {
        let hidden = config.predictor_hidden;
        let size = config.predictor_kernel;
        let (input, output) = (config.hidden, 2 * config.hidden);
        Ok(Self {
            input_conv: schema.conv1d("input_conv", config.mels, hidden, 5, same_1d(5, 1), true)?,
            resblocks: (0..config.predictor_blocks)
                .map(|index| {
                    let mut block = schema.scope(format!("resblocks.{index}"));
                    Ok([
                        block.conv1d("conv1", hidden, hidden, size, same_1d(size, 1), true)?,
                        block.conv1d("conv2", hidden, hidden, size, same_1d(size, 1), true)?,
                    ])
                })
                .collect::<SingeResult<Vec<_>>>()?,
            kernel_conv: schema.conv1d(
                "kernel_conv",
                hidden,
                layers * input * output * kernel,
                size,
                same_1d(size, 1),
                true,
            )?,
            bias_conv: schema.conv1d(
                "bias_conv",
                hidden,
                layers * output,
                size,
                same_1d(size, 1),
                true,
            )?,
            shape: [layers, input, output, kernel],
            slope: config.leaky_slope,
        })
    }

    fn predict(&self, spectrogram: &Tensor, state: &ModelState) -> SingeResult<(Tensor, Tensor)> {
        let &[batch, _, frames] = spectrogram.shape().dimensions() else {
            return Err(SingeError::invalid_argument(
                "a UnivNet spectrogram, [B, mels, F]",
            ));
        };
        let mut hidden = spectrogram
            .apply(&self.input_conv, state)?
            .leaky_relu(self.slope)?;
        for [first, second] in &self.resblocks {
            let inner = hidden
                .apply(first, state)?
                .leaky_relu(self.slope)?
                .apply(second, state)?
                .leaky_relu(self.slope)?;
            hidden = hidden.add(&inner)?;
        }

        let [layers, input, output, kernel] = self.shape;
        Ok((
            hidden
                .apply(&self.kernel_conv, state)?
                .reshape([batch, layers, input, output, kernel, frames])?,
            hidden
                .apply(&self.bias_conv, state)?
                .reshape([batch, layers, output, frames])?,
        ))
    }
}

/// Convolves each `hop` samples of `hidden`, `[B, C, F · hop]`, with its frame's own kernel, `[B, C, O, k, F]`, and bias, `[B, O, F]`, zero-padded only at the outer ends, as `location_variable_convolution` does.
fn location_variable_convolution(
    hidden: &Tensor,
    kernel: &Tensor,
    bias: &Tensor,
    hop: i64,
) -> SingeResult<Tensor> {
    let &[batch, channels, output, size, frames] = kernel.shape().dimensions() else {
        return Err(SingeError::invalid_argument(
            "location-variable kernels, [B, C, O, k, F]",
        ));
    };
    let padding = (size - 1) / 2;
    let windows = hidden
        .pad(
            &[Padding::ZERO, Padding::ZERO, Padding::symmetric(padding)],
            0.0,
        )?
        .unfold(2, hop + 2 * padding, hop)?
        .contiguous()?
        .unfold(3, size, 1)?
        .permute(&[0, 2, 3, 1, 4])?
        .reshape([batch, frames, hop, channels * size])?;
    let kernel =
        kernel
            .permute(&[0, 4, 1, 3, 2])?
            .reshape([batch, frames, channels * size, output])?;
    windows
        .matmul(&kernel)?
        .add(&bias.permute(&[0, 2, 1])?.unsqueeze(2)?)?
        .permute(&[0, 3, 1, 2])?
        .reshape([batch, output, frames * hop])
}

/// A block: a leaky `ReLU` and an upsampling transposed convolution, then gated residual layers, each a dilated convolution and a location-variable convolution.
#[derive(Debug, Clone)]
struct LvcBlock {
    convt_pre: ConvTranspose1d,
    kernel_predictor: KernelPredictor,
    resblocks: Vec<Conv1d>,
    hop: i64,
    hidden: i64,
    slope: f64,
}

impl LvcBlock {
    fn create(
        schema: &mut Schema<'_>,
        config: &UnivNetConfig,
        index: usize,
        hop: i64,
    ) -> SingeResult<Self> {
        let invalid = || SingeError::invalid_argument("a UnivNet block's sizes");
        let stride = *config.strides.get(index).ok_or_else(invalid)?;
        let kernel = *config.kernels.get(index).ok_or_else(invalid)?;
        let dilations = config.dilations.get(index).ok_or_else(invalid)?;
        let hidden = config.hidden;
        let layers = i64::try_from(dilations.len()).map_err(|_| invalid())?;
        Ok(Self {
            convt_pre: schema.conv_transpose1d(
                "convt_pre",
                hidden,
                hidden,
                2 * stride,
                ConvTranspose1dConfig {
                    stride,
                    padding: stride / 2 + stride % 2,
                    output_padding: stride % 2,
                    ..ConvTranspose1dConfig::default()
                },
                true,
            )?,
            kernel_predictor: KernelPredictor::create(
                &mut schema.scope("kernel_predictor"),
                config,
                kernel,
                layers,
            )?,
            resblocks: dilations
                .iter()
                .enumerate()
                .map(|(layer, &dilation)| {
                    schema.conv1d(
                        &format!("resblocks.{layer}.conv"),
                        hidden,
                        hidden,
                        kernel,
                        same_1d(kernel, dilation),
                        true,
                    )
                })
                .collect::<SingeResult<Vec<_>>>()?,
            hop,
            hidden,
            slope: config.leaky_slope,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        spectrogram: &Tensor,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let mut hidden = hidden
            .leaky_relu(self.slope)?
            .apply(&self.convt_pre, state)?;
        let (kernels, biases) = self.kernel_predictor.predict(spectrogram, state)?;
        for (conv, layer) in self.resblocks.iter().zip(0_i64..) {
            let convolved = location_variable_convolution(
                &hidden
                    .leaky_relu(self.slope)?
                    .apply(conv, state)?
                    .leaky_relu(self.slope)?,
                &kernels.select(1, layer)?,
                &biases.select(1, layer)?,
                self.hop,
            )?;
            let gated = convolved
                .narrow(1, 0, self.hidden)?
                .apply(&Activation::Sigmoid, state)?
                .mul(
                    &convolved
                        .narrow(1, self.hidden, self.hidden)?
                        .apply(&Activation::Tanh, state)?,
                )?;
            hidden = hidden.add(&gated)?;
        }
        Ok(hidden)
    }
}

#[derive(Debug, Clone)]
pub struct UnivNetLayers {
    conv_pre: Conv1d,
    resblocks: Vec<LvcBlock>,
    conv_post: Conv1d,
    slope: f64,
}

impl UnivNetLayers {
    pub fn create(schema: &mut Schema<'_>, config: &UnivNetConfig) -> SingeResult<Self> {
        let unpadded = Conv1dConfig::default();
        let mut hop = 1;
        let resblocks = config
            .strides
            .iter()
            .enumerate()
            .map(|(index, &stride)| {
                hop *= stride;
                LvcBlock::create(
                    &mut schema.scope(format!("resblocks.{index}")),
                    config,
                    index,
                    hop,
                )
            })
            .collect::<SingeResult<Vec<_>>>()?;
        Ok(Self {
            conv_pre: schema.conv1d(
                "conv_pre",
                config.noise_channels,
                config.hidden,
                7,
                unpadded,
                true,
            )?,
            resblocks,
            conv_post: schema.conv1d("conv_post", config.hidden, 1, 7, unpadded, true)?,
            slope: config.leaky_slope,
        })
    }
}

impl VocoderLayers for UnivNetLayers {
    fn synthesize(
        &self,
        features: &Tensor,
        noise: &Tensor,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        // A clip is the output, which TF32 convolutions move: full precision.
        with_convolution_precision(ComputePrecision::Full, || {
            let spectrogram = features.transpose(1, 2)?;
            // Both outer convolutions pad by reflection.
            let mut hidden = noise
                .transpose(1, 2)?
                .pad_reflect(2, 3, 3)?
                .apply(&self.conv_pre, state)?;
            for block in &self.resblocks {
                hidden = block.forward(&hidden, &spectrogram, state)?;
            }
            hidden
                .leaky_relu(self.slope)?
                .pad_reflect(2, 3, 3)?
                .apply(&self.conv_post, state)?
                .apply(&Activation::Tanh, state)?
                .squeeze(1)
        })
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "univnet",
    formats: &[(Format::Transformers, &["univnet"])],
    loaders: &[HeadLoader::Vocoder(load_vocoder)],
};

pub fn load_vocoder(loader: &mut Loader<'_>) -> Result<Vocoder> {
    let config: UnivNetConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        UnivNetLayers::create(schema, &config)
            .map(|layers| Boxed::<dyn VocoderLayers>(Box::new(layers)))
    })?;
    Ok(Vocoder::new(
        loader.load_whole(&model, &transformers::rules())?,
        config.features,
        usize::try_from(config.noise_channels)
            .map_err(|_| Error::unsupported("negative noise channels"))?,
    ))
}
