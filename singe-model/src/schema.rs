use std::result::Result as StdResult;

use singe::{
    DataType, Device, Error as SingeError, Result, Shape, Tensor,
    nn::{
        Activation, BatchNorm, Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, Conv3d, Conv3dConfig,
        ConvTranspose1d, ConvTranspose2d, ConvTransposeConfig, GatedMlp, GroupNorm, LayerNorm,
        Linear, Mlp, Model, OffsetRmsNorm, Recurrent, RecurrentConfig, RecurrentDirection,
        RecurrentLayer, RelativeAttention, RelativeAttentionLayers, RmsNorm, StateBuilder,
        StateKey, StateScope,
    },
};

/// A norm a model can register by name, so an assembly can pick one by type.
pub trait SchemaNorm: Sized {
    fn register(schema: &mut Schema<'_>, name: &str, width: i64, eps: f32) -> Result<Self>;
}

impl SchemaNorm for RmsNorm {
    fn register(schema: &mut Schema<'_>, name: &str, width: i64, eps: f32) -> Result<Self> {
        schema.rms_norm(name, width, eps)
    }
}

impl SchemaNorm for OffsetRmsNorm {
    fn register(schema: &mut Schema<'_>, name: &str, width: i64, eps: f32) -> Result<Self> {
        schema.offset_rms_norm(name, width, eps)
    }
}

impl SchemaNorm for LayerNorm {
    fn register(schema: &mut Schema<'_>, name: &str, width: i64, eps: f32) -> Result<Self> {
        schema.layer_norm(name, width, eps)
    }
}

/// A registry of a model's checkpoint tensors by name and shape, as tensors on [`Device::Meta`] in the type the model will hold them in.
#[derive(Debug)]
pub struct Schema<'a> {
    scope: StateScope<'a>,
    data_type: DataType,
}

impl Schema<'_> {
    pub fn model<L, E: From<SingeError>>(
        data_type: DataType,
        build: impl FnOnce(&mut Schema<'_>) -> StdResult<L, E>,
    ) -> StdResult<Model<L>, E> {
        let mut builder = StateBuilder::new();
        let layer = build(&mut Schema {
            scope: builder.unscoped(),
            data_type,
        })?;
        Ok(Model::new(layer, builder.finish()?))
    }

    /// Returns a narrower schema, whose names also start with `name`.
    pub fn scope(&mut self, name: impl AsRef<str>) -> Schema<'_> {
        Schema {
            scope: self.scope.scope(name),
            data_type: self.data_type,
        }
    }

    pub fn parameter<const N: usize>(&mut self, name: &str, shape: [i64; N]) -> Result<StateKey> {
        self.parameter_typed(name, shape, self.data_type)
    }

    /// Registers state of a type of its own, such as a lookup table of integers beside the weights.
    pub fn parameter_typed<const N: usize>(
        &mut self,
        name: &str,
        shape: [i64; N],
        data_type: DataType,
    ) -> Result<StateKey> {
        let tensor = Tensor::zeroed_on(Device::Meta, data_type, Shape::create(shape)?)?;
        self.scope.parameter(name, tensor)
    }

    /// Registers `name` as a second name for the state of `target`, such as an output tied to the token embeddings.
    pub fn tie(&mut self, name: &str, target: &StateKey) -> Result<StateKey> {
        self.scope.tie(name, target)
    }

    /// Registers a linear layer: `{name}.weight`, stored `[out, in]`, and with `bias`, `{name}.bias`.
    pub fn linear(&mut self, name: &str, input: i64, out: i64, bias: bool) -> Result<Linear> {
        let weight = self.parameter(&format!("{name}.weight"), [out, input])?;
        let bias = self.bias(name, out, bias)?;
        Ok(Linear::new(weight, bias))
    }

    pub fn join_rows(&mut self, linears: &[&Linear]) -> Result<()> {
        self.scope.join_rows(linears)
    }

    pub fn attention_projections(&mut self, width: i64, biases: [bool; 4]) -> Result<[Linear; 4]> {
        let [query, key, value, output] = biases;
        let projections = [
            self.linear("query", width, width, query)?,
            self.linear("key", width, width, key)?,
            self.linear("value", width, width, value)?,
            self.linear("output", width, width, output)?,
        ];
        self.join_rows(&[&projections[0], &projections[1], &projections[2]])?;
        Ok(projections)
    }

    pub fn mlp(
        &mut self,
        name: &str,
        [input, inner, output]: [i64; 3],
        activation: Activation,
        bias: bool,
    ) -> Result<Mlp> {
        Ok(Mlp::new(
            self.linear(&format!("{name}.up"), input, inner, bias)?,
            self.linear(&format!("{name}.down"), inner, output, bias)?,
            activation,
        ))
    }

    pub fn gated_mlp(
        &mut self,
        name: &str,
        [input, inner, output]: [i64; 3],
        activation: Activation,
        bias: bool,
    ) -> Result<GatedMlp> {
        let mlp = GatedMlp::new(
            self.linear(&format!("{name}.gate"), input, inner, bias)?,
            self.linear(&format!("{name}.up"), input, inner, bias)?,
            self.linear(&format!("{name}.down"), inner, output, bias)?,
            activation,
        );
        self.join_rows(&mlp.siblings())?;
        Ok(mlp)
    }

    /// Registers a square convolution: `{name}.weight`, stored `[out, in / groups, kernel, kernel]`, and with `bias`, `{name}.bias`.
    pub fn conv2d(
        &mut self,
        name: &str,
        input: i64,
        out: i64,
        kernel: i64,
        config: Conv2dConfig,
        bias: bool,
    ) -> Result<Conv2d> {
        let weight = self.parameter(
            &format!("{name}.weight"),
            [out, input / config.groups, kernel, kernel],
        )?;
        let bias = self.bias(name, out, bias)?;
        Ok(Conv2d::new(weight, bias, config))
    }

    /// Registers a convolution over clips: `{name}.weight`, stored `[out, in / groups, kernel[0], kernel[1], kernel[2]]`, and with `bias`, `{name}.bias`.
    pub fn conv3d(
        &mut self,
        name: &str,
        input: i64,
        out: i64,
        kernel: [i64; 3],
        config: Conv3dConfig,
        bias: bool,
    ) -> Result<Conv3d> {
        let weight = self.parameter(
            &format!("{name}.weight"),
            [out, input / config.groups, kernel[0], kernel[1], kernel[2]],
        )?;
        let bias = self.bias(name, out, bias)?;
        Ok(Conv3d::new(weight, bias, config))
    }

    /// Registers a convolution over sequences: `{name}.weight`, stored `[out, in / groups, kernel]`, and with `bias`, `{name}.bias`.
    pub fn conv1d(
        &mut self,
        name: &str,
        input: i64,
        out: i64,
        kernel: i64,
        config: Conv1dConfig,
        bias: bool,
    ) -> Result<Conv1d> {
        let weight = self.parameter(
            &format!("{name}.weight"),
            [out, input / config.groups, kernel],
        )?;
        let bias = self.bias(name, out, bias)?;
        Ok(Conv1d::new(weight, bias, config))
    }

    /// Registers a group norm over `width` channels in `groups` groups: `{name}.weight` and `{name}.bias`.
    pub fn group_norm(
        &mut self,
        name: &str,
        width: i64,
        groups: i64,
        eps: f32,
    ) -> Result<GroupNorm> {
        Ok(GroupNorm::new(
            Some(self.parameter(&format!("{name}.weight"), [width])?),
            Some(self.parameter(&format!("{name}.bias"), [width])?),
            groups,
            eps,
        ))
    }

    /// Registers a transposed convolution over sequences: `{name}.weight`, stored `[in, out / groups, kernel]`, and with `bias`, `{name}.bias`.
    pub fn conv_transpose1d(
        &mut self,
        name: &str,
        input: i64,
        out: i64,
        kernel: i64,
        config: ConvTransposeConfig<1>,
        bias: bool,
    ) -> Result<ConvTranspose1d> {
        let weight = self.parameter(
            &format!("{name}.weight"),
            [input, out / config.groups, kernel],
        )?;
        let bias = self.bias(name, out, bias)?;
        Ok(ConvTranspose1d::new(weight, bias, config))
    }

    /// Registers a square transposed convolution: `{name}.weight`, stored `[in, out / groups, kernel, kernel]`, and with `bias`, `{name}.bias`.
    pub fn conv_transpose2d(
        &mut self,
        name: &str,
        input: i64,
        out: i64,
        kernel: i64,
        config: ConvTransposeConfig<2>,
        bias: bool,
    ) -> Result<ConvTranspose2d> {
        let weight = self.parameter(
            &format!("{name}.weight"),
            [input, out / config.groups, kernel, kernel],
        )?;
        let bias = self.bias(name, out, bias)?;
        Ok(ConvTranspose2d::new(weight, bias, config))
    }

    /// Registers a batch norm over `width` channels in evaluation: `{name}.weight`, `{name}.bias`, `{name}.running_mean`, and `{name}.running_var`.
    pub fn batch_norm(&mut self, name: &str, width: i64, eps: f32) -> Result<BatchNorm> {
        Ok(BatchNorm::new(
            Some(self.parameter(&format!("{name}.weight"), [width])?),
            Some(self.parameter(&format!("{name}.bias"), [width])?),
            self.parameter(&format!("{name}.running_mean"), [width])?,
            self.parameter(&format!("{name}.running_var"), [width])?,
            eps,
        ))
    }

    /// Registers a recurrent stack under `name`: each layer's linear layers `{name}.N.input`, `[G · H, I]`, `{name}.N.hidden`, `[G · H, O]`, and with a projection, `{name}.N.projection`, `[O, H]`.
    /// `G` is the cell's gate count, `H` the hidden width, `I` the layer's input width, and `O` its output width.
    /// A bidirectional stack's reverse directions are under `{name}.N.reverse`.
    pub fn recurrent(&mut self, name: &str, config: &RecurrentConfig) -> Result<Recurrent> {
        let gates = config.cell.gates() * config.hidden;
        let layers = (0..config.layers)
            .map(|layer| {
                let input = config.layer_input(layer);
                let mut direction = |prefix: String| -> Result<RecurrentDirection> {
                    Ok(RecurrentDirection {
                        input: self.linear(
                            &format!("{prefix}.input"),
                            input,
                            gates,
                            config.biases,
                        )?,
                        hidden: self.linear(
                            &format!("{prefix}.hidden"),
                            config.output(),
                            gates,
                            config.biases,
                        )?,
                        projection: config
                            .projection
                            .map(|projection| {
                                self.linear(
                                    &format!("{prefix}.projection"),
                                    config.hidden,
                                    projection,
                                    false,
                                )
                            })
                            .transpose()?,
                    })
                };
                let forward = direction(format!("{name}.{layer}"))?;
                let reverse = config
                    .bidirectional
                    .then(|| direction(format!("{name}.{layer}.reverse")))
                    .transpose()?;
                Ok(RecurrentLayer { forward, reverse })
            })
            .collect::<Result<Vec<_>>>()?;
        Recurrent::create(*config, layers)
    }

    /// Registers self-attention over relative positions: `query`, `key`, `value`, and `output`, with the biases `biases` asks for, `positions` without one, and `content_bias` and `position_bias`, `[heads, width / heads]`.
    pub fn relative_attention(
        &mut self,
        width: i64,
        heads: i64,
        biases: [bool; 4],
    ) -> Result<RelativeAttention> {
        let [query, key, value, output] = self.attention_projections(width, biases)?;
        let head_dim = width / heads.max(1);
        let layers = RelativeAttentionLayers {
            query,
            key,
            value,
            output,
            positions: self.linear("positions", width, width, false)?,
            content_bias: self.parameter("content_bias", [heads, head_dim])?,
            position_bias: self.parameter("position_bias", [heads, head_dim])?,
        };
        RelativeAttention::create(layers, heads, width)
    }

    /// Registers an `RMSNorm` over its last `width` features: `{name}.weight`.
    pub fn rms_norm(&mut self, name: &str, width: i64, eps: f32) -> Result<RmsNorm> {
        let weight = self.parameter(&format!("{name}.weight"), [width])?;
        Ok(RmsNorm::new(vec![width], Some(weight), eps))
    }

    /// Registers Gemma's `RMSNorm` over its last `width` features: `{name}.weight`.
    pub fn offset_rms_norm(&mut self, name: &str, width: i64, eps: f32) -> Result<OffsetRmsNorm> {
        let weight = self.parameter(&format!("{name}.weight"), [width])?;
        Ok(OffsetRmsNorm::new(vec![width], weight, eps))
    }

    /// Registers a `LayerNorm` over its last `width` features: `{name}.weight` and `{name}.bias`.
    pub fn layer_norm(&mut self, name: &str, width: i64, eps: f32) -> Result<LayerNorm> {
        let weight = self.parameter(&format!("{name}.weight"), [width])?;
        let bias = self.parameter(&format!("{name}.bias"), [width])?;
        Ok(LayerNorm::new(vec![width], Some(weight), Some(bias), eps))
    }

    /// Registers a `LayerNorm` without a bias, as `ModernBERT`'s are: `{name}.weight`.
    pub fn unbiased_layer_norm(&mut self, name: &str, width: i64, eps: f32) -> Result<LayerNorm> {
        let weight = self.parameter(&format!("{name}.weight"), [width])?;
        Ok(LayerNorm::new(vec![width], Some(weight), None, eps))
    }

    fn bias(&mut self, name: &str, width: i64, bias: bool) -> Result<Option<StateKey>> {
        bias.then(|| self.parameter(&format!("{name}.bias"), [width]))
            .transpose()
    }
}
