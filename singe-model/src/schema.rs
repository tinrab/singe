use singe::{
    DataType, Device, Result, Shape, Tensor,
    nn::{
        BatchNorm2d, Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, Conv3d, Conv3dConfig,
        ConvTranspose1d, ConvTranspose2d, ConvTransposeConfig, GroupNorm, LayerNorm, Linear,
        OffsetRmsNorm, RmsNorm, StateBuilder, StateKey, StateScope,
    },
};

// A norm a model can register by name, so an assembly can pick one by type.
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

// Registers a model's checkpoint tensors by shape, on metadata, in the type the model will hold them in.
#[derive(Debug)]
pub struct Schema<'a> {
    scope: StateScope<'a>,
    data_type: DataType,
}

impl<'a> Schema<'a> {
    #[must_use]
    pub const fn new(scope: StateScope<'a>, data_type: DataType) -> Self {
        Self { scope, data_type }
    }

    // Returns a schema with no prefix, for tensors at a checkpoint's top level.
    pub fn root(builder: &'a mut StateBuilder, data_type: DataType) -> Self {
        Self::new(builder.unscoped(), data_type)
    }

    // Returns a narrower schema, whose names also start with `name`.
    pub fn scope(&mut self, name: impl AsRef<str>) -> Schema<'_> {
        Schema {
            scope: self.scope.scope(name),
            data_type: self.data_type,
        }
    }

    pub fn parameter<const N: usize>(&mut self, name: &str, shape: [i64; N]) -> Result<StateKey> {
        self.parameter_typed(name, shape, self.data_type)
    }

    // Registers state of a type of its own, such as a lookup table of integers beside the weights.
    pub fn parameter_typed<const N: usize>(
        &mut self,
        name: &str,
        shape: [i64; N],
        data_type: DataType,
    ) -> Result<StateKey> {
        let tensor = Tensor::zeroed_on(Device::Meta, data_type, Shape::create(shape)?)?;
        self.scope.parameter(name, tensor)
    }

    // Registers `name` as a second name for the state of `target`, such as an output tied to the token embeddings.
    pub fn tie(&mut self, name: &str, target: &StateKey) -> Result<StateKey> {
        self.scope.tie(name, target)
    }

    // Registers `{name}.weight`, stored `[out, in]`, and with `bias`, `{name}.bias`.
    pub fn linear(&mut self, name: &str, input: i64, out: i64, bias: bool) -> Result<Linear> {
        let weight = self.parameter(&format!("{name}.weight"), [out, input])?;
        let bias = bias
            .then(|| self.parameter(&format!("{name}.bias"), [out]))
            .transpose()?;
        Ok(Linear::new(weight, bias))
    }

    // Registers a square convolution: `{name}.weight`, stored `[out, in / groups, kernel, kernel]`, and with `bias`, `{name}.bias`.
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
        let bias = bias
            .then(|| self.parameter(&format!("{name}.bias"), [out]))
            .transpose()?;
        Ok(Conv2d::new(weight, bias, config))
    }

    // Registers a convolution over clips: `{name}.weight`, stored `[out, in / groups, frames, kernel, kernel]`, and with `bias`, `{name}.bias`.
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
        let bias = bias
            .then(|| self.parameter(&format!("{name}.bias"), [out]))
            .transpose()?;
        Ok(Conv3d::new(weight, bias, config))
    }

    // Registers a convolution over sequences: `{name}.weight`, stored `[out, in / groups, kernel]`, and with `bias`, `{name}.bias`.
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
        let bias = bias
            .then(|| self.parameter(&format!("{name}.bias"), [out]))
            .transpose()?;
        Ok(Conv1d::new(weight, bias, config))
    }

    // Registers a group norm over `width` channels in `groups` groups: `{name}.weight` and `{name}.bias`.
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

    // Registers a transposed convolution over sequences: `{name}.weight`, stored `[in, out / groups, kernel]`, and with `bias`, `{name}.bias`.
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
        let bias = bias
            .then(|| self.parameter(&format!("{name}.bias"), [out]))
            .transpose()?;
        Ok(ConvTranspose1d::new(weight, bias, config))
    }

    // Registers a square transposed convolution: `{name}.weight`, stored `[in, out / groups, kernel, kernel]`, and with `bias`, `{name}.bias`.
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
        let bias = bias
            .then(|| self.parameter(&format!("{name}.bias"), [out]))
            .transpose()?;
        Ok(ConvTranspose2d::new(weight, bias, config))
    }

    // Registers a batch norm over `width` channels in evaluation: `{name}.weight`, `{name}.bias`, `{name}.running_mean`, and `{name}.running_var`.
    pub fn batch_norm2d(&mut self, name: &str, width: i64, eps: f32) -> Result<BatchNorm2d> {
        Ok(BatchNorm2d::new(
            Some(self.parameter(&format!("{name}.weight"), [width])?),
            Some(self.parameter(&format!("{name}.bias"), [width])?),
            self.parameter(&format!("{name}.running_mean"), [width])?,
            self.parameter(&format!("{name}.running_var"), [width])?,
            eps,
        ))
    }

    // Registers an RMSNorm over its last `width` features.
    pub fn rms_norm(&mut self, name: &str, width: i64, eps: f32) -> Result<RmsNorm> {
        let weight = self.parameter(&format!("{name}.weight"), [width])?;
        Ok(RmsNorm::new(vec![width], Some(weight), eps))
    }

    // Registers Gemma's RMSNorm over its last `width` features.
    pub fn offset_rms_norm(&mut self, name: &str, width: i64, eps: f32) -> Result<OffsetRmsNorm> {
        let weight = self.parameter(&format!("{name}.weight"), [width])?;
        Ok(OffsetRmsNorm::new(vec![width], weight, eps))
    }

    // Registers a LayerNorm over its last `width` features.
    pub fn layer_norm(&mut self, name: &str, width: i64, eps: f32) -> Result<LayerNorm> {
        let weight = self.parameter(&format!("{name}.weight"), [width])?;
        let bias = self.parameter(&format!("{name}.bias"), [width])?;
        Ok(LayerNorm::new(vec![width], Some(weight), Some(bias), eps))
    }

    // Registers a LayerNorm without a bias, as ModernBERT's are.
    pub fn unbiased_layer_norm(&mut self, name: &str, width: i64, eps: f32) -> Result<LayerNorm> {
        let weight = self.parameter(&format!("{name}.weight"), [width])?;
        Ok(LayerNorm::new(vec![width], Some(weight), None, eps))
    }
}
