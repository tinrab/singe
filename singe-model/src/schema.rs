use singe::{
    DataType, Device, Result, Shape, Tensor,
    nn::{
        Conv2d, Conv2dConfig, LayerNorm, Linear, OffsetRmsNorm, RmsNorm, StateBuilder, StateKey,
        StateScope,
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
        let tensor = Tensor::zeroed_on(Device::Meta, self.data_type, Shape::create(shape)?)?;
        self.scope.parameter(name, tensor)
    }

    // Registers `{name}.weight`, stored `[out, in]`, and with `bias`, `{name}.bias`.
    pub fn linear(&mut self, name: &str, input: i64, out: i64, bias: bool) -> Result<Linear> {
        let weight = self.parameter(&format!("{name}.weight"), [out, input])?;
        let bias = bias
            .then(|| self.parameter(&format!("{name}.bias"), [out]))
            .transpose()?;
        Ok(Linear::new(weight, bias))
    }

    // Registers GPT-2's `Conv1D`: `{name}.weight`, stored `[in, out]`, and `{name}.bias`.
    pub fn conv1d(&mut self, name: &str, input: i64, out: i64) -> Result<Linear> {
        let weight = self.parameter(&format!("{name}.weight"), [input, out])?;
        let bias = self.parameter(&format!("{name}.bias"), [out])?;
        Ok(Linear::new(weight, Some(bias)).with_input_first())
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
}
