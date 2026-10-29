use crate::{
    backend::tensor::{Shape, TensorId},
    error::{Error, Result},
    frontend::{
        graph::{Graph, TensorOutputConfig},
        operation::{ConvolutionConfig, ConvolutionOperation, Operation},
    },
    utility::to_usize,
};

impl Graph {
    /// Adds a convolution forward operation with an explicit output tensor.
    ///
    /// Groups follow the input and filter channel dimensions.
    /// The output may have an independently selected data type and layout supported by cuDNN.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensor identities, invalid vector metadata or geometry, incompatible ranks or grouped channels, a response shape mismatch, or a nonzero effective beta with a destination that cannot provide initialized accumulation storage. Also returns an error if a 2D or 3D filter’s scalar element count overflows or exceeds the signed 32-bit indexing limit.
    pub fn convolution_forward_into(
        &mut self,
        x: TensorId,
        w: TensorId,
        y: TensorId,
        config: ConvolutionConfig,
    ) -> Result<()> {
        self.add_convolution(ConvolutionOperation::Forward { x, w, y, config })
    }

    /// Adds a convolution forward operation and creates an output with channels-last strides.
    /// Equal-type vectorized outputs preserve the input's lane count, vector dimension and stride order.
    /// Result placement selects I/O or intermediate precision unless an encoding is explicit.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensor identities, invalid vector metadata or geometry, incompatible ranks or grouped channels, a response shape mismatch, or a nonzero effective beta with a destination that cannot provide initialized accumulation storage. Also returns an error for an unrepresentable output layout, a vectorized output extent incompatible with its lane count, or a forward filter exceeding the signed 32-bit indexing limit.
    pub fn convolution_forward(
        &mut self,
        x: TensorId,
        w: TensorId,
        config: ConvolutionConfig,
        output: TensorOutputConfig,
    ) -> Result<TensorId> {
        output.validate_accumulation(
            config.reads_destination(self.config.data_type_policy),
            "convolution",
        )?;
        let input = self.tensor_spec(x)?;
        let filter = self.tensor_spec(w)?;
        let output_data_type =
            output.resolve_data_type(self.config.data_type_policy, input.data_type);
        let dimensions = config.resolve_geometry()?.infer_forward_dimensions(
            &input.logical_dimensions()?,
            &filter.logical_dimensions()?,
        )?;
        let (shape, vectorization) = match input.vectorization()? {
            Some((count, axis)) if output_data_type == input.data_type => {
                let mut dimensions = dimensions;
                let index = to_usize(axis, "convolution vector dimension")?;
                let dimension = dimensions.get_mut(index).ok_or(Error::InvalidDataShape)?;
                if *dimension % count != 0 {
                    return Err(Error::InvalidDataShape);
                }
                *dimension /= count;
                (
                    input
                        .shape
                        .with_dimensions_preserving_stride_order(dimensions)?,
                    Some((count, axis)),
                )
            }
            _ => (Shape::channels_last(dimensions)?, None),
        };
        let spec = output.into_spec(self.config.data_type_policy, input.data_type, shape);
        let spec = match vectorization {
            Some((count, axis)) => spec.with_vectorization(count, axis)?,
            None => spec,
        };
        let checkpoint = self.mutation_checkpoint();
        let output = self.inferred_tensor(spec, config.name(), "Y");
        if let Err(error) = self.convolution_forward_into(x, w, output, config) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(output)
    }

    /// Adds a data-gradient convolution using the caller's input-gradient dimensions and strides.
    ///
    /// Strided convolution can have several valid inverse shapes.
    /// Validate the response of `dx` and `w` against `dy` rather than choosing one inverse.
    /// A nonzero effective beta requires a bindable `dx` accumulation destination.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensor identities, invalid vector metadata or geometry, incompatible ranks or grouped channels, a response shape mismatch, or a nonzero effective beta with a destination that cannot provide initialized accumulation storage.
    pub fn convolution_backward_data_into(
        &mut self,
        w: TensorId,
        dy: TensorId,
        dx: TensorId,
        config: ConvolutionConfig,
    ) -> Result<()> {
        self.add_convolution(ConvolutionOperation::BackwardData { w, dy, dx, config })
    }

    /// Creates a data-gradient output with the specified dimensions and channels-last strides.
    ///
    /// The input-gradient dimensions are required because the inverse mapping is ambiguous.
    /// Result placement and encoding are independent of convolution math and geometry.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensor identities, invalid vector metadata or geometry, incompatible ranks or grouped channels, a response shape mismatch, or a nonzero effective beta with a destination that cannot provide initialized accumulation storage. Also returns an error if the requested gradient dimensions cannot form a valid channels-last layout.
    pub fn convolution_backward_data(
        &mut self,
        w: TensorId,
        dy: TensorId,
        input_dimensions: impl Into<Vec<i64>>,
        config: ConvolutionConfig,
        output: TensorOutputConfig,
    ) -> Result<TensorId> {
        output.validate_accumulation(
            config.reads_destination(self.config.data_type_policy),
            "convolution data gradient",
        )?;
        let fallback_type = self.tensor_spec(dy)?.data_type;
        let shape = Shape::channels_last(input_dimensions.into())?;
        let output = output.into_spec(self.config.data_type_policy, fallback_type, shape);
        let checkpoint = self.mutation_checkpoint();
        let output = self.inferred_tensor(output, config.name(), "DX");
        if let Err(error) = self.convolution_backward_data_into(w, dy, output, config) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(output)
    }

    /// Adds a weight-gradient convolution using the caller's filter-gradient dimensions and strides.
    /// A nonzero effective beta requires a bindable `dw` accumulation destination.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensor identities, invalid vector metadata or geometry, incompatible ranks or grouped channels, a response shape mismatch, or a nonzero effective beta with a destination that cannot provide initialized accumulation storage.
    pub fn convolution_backward_filter_into(
        &mut self,
        x: TensorId,
        dy: TensorId,
        dw: TensorId,
        config: ConvolutionConfig,
    ) -> Result<()> {
        self.add_convolution(ConvolutionOperation::BackwardFilter { x, dy, dw, config })
    }

    /// Creates a weight-gradient output with the specified dimensions and channels-last strides.
    ///
    /// The filter-gradient dimensions are required because the inverse mapping is ambiguous.
    /// Result placement and encoding are independent of convolution math and geometry.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensor identities, invalid vector metadata or geometry, incompatible ranks or grouped channels, a response shape mismatch, or a nonzero effective beta with a destination that cannot provide initialized accumulation storage. Also returns an error if the requested gradient dimensions cannot form a valid channels-last layout.
    pub fn convolution_backward_filter(
        &mut self,
        x: TensorId,
        dy: TensorId,
        filter_dimensions: impl Into<Vec<i64>>,
        config: ConvolutionConfig,
        output: TensorOutputConfig,
    ) -> Result<TensorId> {
        output.validate_accumulation(
            config.reads_destination(self.config.data_type_policy),
            "convolution weight gradient",
        )?;
        let fallback_type = self.tensor_spec(x)?.data_type;
        let shape = Shape::channels_last(filter_dimensions.into())?;
        let output = output.into_spec(self.config.data_type_policy, fallback_type, shape);
        let checkpoint = self.mutation_checkpoint();
        let output = self.inferred_tensor(output, config.name(), "DW");
        if let Err(error) = self.convolution_backward_filter_into(x, dy, output, config) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(output)
    }

    fn add_convolution(&mut self, operation: ConvolutionOperation) -> Result<()> {
        operation.validate_graph(self)?;
        self.operations.push(Operation::Convolution(operation));
        Ok(())
    }
}
