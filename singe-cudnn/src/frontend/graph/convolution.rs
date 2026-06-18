use crate::{
    convolution::ConvolutionMode,
    error::{Error, Result},
    frontend::{
        graph::Graph,
        infer::{
            infer_convolution_backward_data_output, infer_convolution_backward_filter_output,
            infer_convolution_forward_output,
        },
        operation::{ConvolutionConfig, Operation},
    },
    tensor::{Shape, TensorId, TensorSpec},
    utility::check_range,
};

impl Graph {
    /// Adds a convolution forward operation with an explicit output tensor.
    ///
    /// This computes the convolution response of input `x` with filter `w`.
    /// This corresponds to the C++ frontend `conv_fprop` operation.
    pub fn convolution_forward(
        &mut self,
        x: TensorId,
        w: TensorId,
        y: TensorId,
        config: ConvolutionConfig,
    ) -> Result<()> {
        let x_tensor = self.tensor_config(x)?.clone();
        let w_tensor = self.tensor_config(w)?.clone();
        self.validate_convolution_config(x, &x_tensor.shape, w, &w_tensor.shape, &config)?;
        let expected = infer_convolution_forward_output(&x_tensor.shape, &w_tensor.shape, &config)?;
        self.validate_tensor_data_type(
            y,
            self.effective_io_data_type(x_tensor.data_type),
            "convolution output",
        )?;
        self.validate_tensor_dimensions(y, expected.dimensions(), "convolution output shape")?;
        self.operations
            .push(Operation::ConvolutionForward { x, w, y, config });
        Ok(())
    }

    /// Adds a convolution forward operation and creates the output tensor.
    pub fn convolution_forward_infer(
        &mut self,
        x: TensorId,
        w: TensorId,
        config: ConvolutionConfig,
    ) -> Result<TensorId> {
        let x_tensor = self.tensor_config(x)?.clone();
        let w_tensor = self.tensor_config(w)?.clone();
        self.validate_convolution_config(x, &x_tensor.shape, w, &w_tensor.shape, &config)?;
        let output_data_type = self.effective_io_data_type(x_tensor.data_type);
        let output_shape =
            infer_convolution_forward_output(&x_tensor.shape, &w_tensor.shape, &config)?;
        let output_shape = Self::default_nhwc_shape(output_shape.dimensions().to_vec())?;
        let output = self.tensor(TensorSpec::new(output_data_type, output_shape));
        self.convolution_forward(x, w, output, config)?;
        Ok(output)
    }

    /// Adds a convolution backward-data operation with an explicit output.
    ///
    /// This computes the data gradient during backpropagation and corresponds
    /// to the C++ frontend `conv_dgrad` operation.
    pub fn convolution_dgrad(
        &mut self,
        w: TensorId,
        dy: TensorId,
        dx: TensorId,
        config: ConvolutionConfig,
    ) -> Result<()> {
        let w_tensor = self.tensor_config(w)?.clone();
        let dy_tensor = self.tensor_config(dy)?.clone();
        validate_convolution_tensor_rank(w, &w_tensor.shape, "convolution filter rank")?;
        validate_convolution_tensor_rank(dy, &dy_tensor.shape, "convolution output-gradient rank")?;
        let expected =
            infer_convolution_backward_data_output(&w_tensor.shape, &dy_tensor.shape, &config)?;
        self.validate_tensor_data_type(
            dx,
            self.effective_io_data_type(dy_tensor.data_type),
            "convolution backward data output",
        )?;
        self.validate_tensor_dimensions(
            dx,
            expected.dimensions(),
            "convolution backward data output shape",
        )?;
        self.operations
            .push(Operation::ConvolutionBackwardData { w, dy, dx, config });
        Ok(())
    }

    /// Adds a convolution backward-data operation and creates the output tensor.
    pub fn convolution_dgrad_infer(
        &mut self,
        w: TensorId,
        dy: TensorId,
        config: ConvolutionConfig,
    ) -> Result<TensorId> {
        let w_tensor = self.tensor_config(w)?.clone();
        let dy_tensor = self.tensor_config(dy)?.clone();
        validate_convolution_tensor_rank(w, &w_tensor.shape, "convolution filter rank")?;
        validate_convolution_tensor_rank(dy, &dy_tensor.shape, "convolution output-gradient rank")?;
        let output_data_type = self.effective_io_data_type(dy_tensor.data_type);
        let output_shape =
            infer_convolution_backward_data_output(&w_tensor.shape, &dy_tensor.shape, &config)?;
        let output_shape = Self::default_nhwc_shape(output_shape.dimensions().to_vec())?;
        let output = self.tensor(TensorSpec::new(output_data_type, output_shape));
        self.convolution_dgrad(w, dy, output, config)?;
        Ok(output)
    }

    /// Adds a convolution backward-filter operation with an explicit output.
    ///
    /// This computes the filter/weight gradient during backpropagation and
    /// corresponds to the C++ frontend `conv_wgrad` operation.
    pub fn convolution_wgrad(
        &mut self,
        x: TensorId,
        dy: TensorId,
        dw: TensorId,
        config: ConvolutionConfig,
    ) -> Result<()> {
        let x_tensor = self.tensor_config(x)?.clone();
        let dy_tensor = self.tensor_config(dy)?.clone();
        validate_convolution_tensor_rank(x, &x_tensor.shape, "convolution input rank")?;
        validate_convolution_tensor_rank(dy, &dy_tensor.shape, "convolution output-gradient rank")?;
        let expected =
            infer_convolution_backward_filter_output(&x_tensor.shape, &dy_tensor.shape, &config)?;
        self.validate_tensor_data_type(
            dw,
            self.effective_io_data_type(x_tensor.data_type),
            "convolution backward filter output",
        )?;
        self.validate_tensor_dimensions(
            dw,
            expected.dimensions(),
            "convolution backward filter output shape",
        )?;
        self.operations
            .push(Operation::ConvolutionBackwardFilter { x, dy, dw, config });
        Ok(())
    }

    /// Adds a convolution backward-filter operation and creates the output tensor.
    pub fn convolution_wgrad_infer(
        &mut self,
        x: TensorId,
        dy: TensorId,
        config: ConvolutionConfig,
    ) -> Result<TensorId> {
        let x_tensor = self.tensor_config(x)?.clone();
        let dy_tensor = self.tensor_config(dy)?.clone();
        validate_convolution_tensor_rank(x, &x_tensor.shape, "convolution input rank")?;
        validate_convolution_tensor_rank(dy, &dy_tensor.shape, "convolution output-gradient rank")?;
        let output_data_type = self.effective_io_data_type(x_tensor.data_type);
        let output_shape =
            infer_convolution_backward_filter_output(&x_tensor.shape, &dy_tensor.shape, &config)?;
        let output_shape = Self::default_nhwc_shape(output_shape.dimensions().to_vec())?;
        let output = self.tensor(TensorSpec::new(output_data_type, output_shape));
        self.convolution_wgrad(x, dy, output, config)?;
        Ok(output)
    }

    fn validate_convolution_config(
        &self,
        x: TensorId,
        x_shape: &Shape,
        w: TensorId,
        w_shape: &Shape,
        config: &ConvolutionConfig,
    ) -> Result<()> {
        validate_convolution_tensor_rank(x, x_shape, "convolution input rank")?;
        let x_rank = x_shape.dimensions().len();
        if w_shape.dimensions().len() != x_rank {
            return Err(Error::FrontendTensorRankMismatch {
                tensor_id: w,
                operation: "convolution filter rank".into(),
                expected: x_rank.to_string(),
                actual: w_shape.dimensions().len(),
            });
        }
        let spatial_dims = x_shape
            .dimensions()
            .len()
            .checked_sub(2)
            .ok_or(Error::InvalidDataShape)?;
        for values in [
            config.pre_paddings(),
            config.post_paddings(),
            config.strides(),
            config.dilations(),
        ] {
            if values.len() != spatial_dims {
                return Err(Error::LengthMismatch {
                    name: "convolution spatial dims".into(),
                    expected: spatial_dims,
                    actual: values.len(),
                });
            }
            if values.iter().any(|value| *value < 0) {
                check_range!("convolution values", false)?;
            }
        }
        if config.mode() != ConvolutionMode::CrossCorrelation
            && config.mode() != ConvolutionMode::Convolution
        {
            check_range!("convolution mode", false)?;
        }
        Ok(())
    }
}

fn validate_convolution_tensor_rank(
    tensor: TensorId,
    shape: &Shape,
    operation: &str,
) -> Result<()> {
    let actual = shape.dimensions().len();
    if actual != 4 {
        return Err(Error::FrontendTensorRankMismatch {
            tensor_id: tensor,
            operation: operation.into(),
            expected: "4".into(),
            actual,
        });
    }
    Ok(())
}
