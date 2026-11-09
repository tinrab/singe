use crate::{
    backend::registry::{
        AliasBehavior, Operation, OperationAttributes, OperationRequest, operation_schema,
    },
    data_type::DataType,
    error::{Error, Result},
    graph::{
        ir::{
            Graph, OperationId, OperationNode, OperationResult, Value, ValueAlias, ValueBinding,
            ValueId,
        },
        operation::{view_output_layout, view_output_shape},
    },
    layout::Layout,
    value::ValueSpec,
};

impl Graph {
    /// Recomputes each operation's result metadata from its operands, in execution order.
    ///
    /// Inputs keep their declared metadata and constants take their tensor's. Other results take the data type and shape their inference rule gives and their first operand's device; a view keeps the layout its operation gives the source layout, and a cast to the operand's own data type keeps a contiguous operand's layout.
    /// A reshape or flatten whose source layout cannot be viewed with the new shape reads a contiguous copy of the source, inserted before it, and an operand of another type than its operation computes on is cast first, as eager dispatch converts it.
    ///
    /// # Errors
    ///
    /// Returns an error if an operation's inference rule rejects its operands or a view is invalid for its source.
    pub(crate) fn infer_value_metadata(&mut self) -> Result<()> {
        let order = self
            .blocks
            .first()
            .map_or_else(Vec::new, |block| block.operations().to_vec());
        for id in order {
            self.convert_operands(id)?;
            self.infer_operation_metadata(id)?;
            self.materialize_reshape_input(id)?;
        }
        Ok(())
    }

    /// Inserts a cast before each operand of `id` whose type differs from the one its operation computes on, as eager dispatch converts operands, so lowerings see the types inference names.
    fn convert_operands(&mut self, id: OperationId) -> Result<()> {
        let operation = self.required_operation(id)?.clone();
        let Some(opcode) = operation.opcode() else {
            return Ok(());
        };
        if matches!(operation.attributes(), OperationAttributes::Input(_))
            || opcode == Operation::Cast
        {
            return Ok(());
        }

        let operands = operation
            .inputs()
            .iter()
            .map(|&input| {
                let value = self.required_value(input)?;
                Ok(ValueSpec::new(value.data_type(), value.shape().clone()))
            })
            .collect::<Result<Vec<_>>>()?;
        let request =
            OperationRequest::new(opcode, operation.attributes().clone(), operands.clone());
        // Inference that fails here fails again when the operation's metadata is inferred, with its own error.
        let Ok(inference) = request.infer() else {
            return Ok(());
        };
        for ((&input, operand), &target) in operation
            .inputs()
            .iter()
            .zip(&operands)
            .zip(inference.kernel_input_data_types())
        {
            if operand.data_type() == target {
                continue;
            }

            let cast = self.insert_operation_before(
                id,
                Operation::Cast,
                OperationAttributes::Cast(target),
                input,
                target,
            )?;
            self.operations[id.index()].replace_input(input, cast);
        }
        Ok(())
    }

    fn infer_operation_metadata(&mut self, id: OperationId) -> Result<()> {
        let operation = self.required_operation(id)?.clone();
        if matches!(operation.attributes(), OperationAttributes::Input(_)) {
            return Ok(());
        }

        let Some(opcode) = operation.opcode() else {
            for value in operation.result_values() {
                if let Some(tensor) = self.constant_tensor(value).cloned() {
                    self.values[value.index()].set_metadata(
                        tensor.data_type(),
                        tensor.layout().clone(),
                        tensor.device(),
                    );
                }
            }
            return Ok(());
        };
        let operands = operation
            .inputs()
            .iter()
            .map(|&input| {
                let value = self.required_value(input)?;
                Ok(ValueSpec::new(value.data_type(), value.shape().clone()))
            })
            .collect::<Result<Vec<_>>>()?;
        let request =
            OperationRequest::new(opcode, operation.attributes().clone(), operands).check()?;
        let outputs = request.inference().outputs();
        let first = self
            .required_value(
                *operation
                    .inputs()
                    .first()
                    .ok_or_else(|| Error::missing_input("infer operation metadata"))?,
            )?
            .clone();
        for (index, value) in operation.result_values().enumerate() {
            let output = outputs
                .get(index)
                .ok_or_else(|| Error::missing_output("infer operation metadata"))?;
            let mut data_type = output.data_type();
            let mut layout = Layout::from_shape(output.shape());
            let mut device = first.device();
            if index == 0 {
                if let Some(alias) = self.required_value(value)?.alias() {
                    let source = self.required_value(alias.source())?;
                    let shape = view_output_shape(opcode, operation.attributes(), source.shape())?;
                    data_type = source.data_type();
                    device = match operation.attributes() {
                        OperationAttributes::CopyTo(target) => *target,
                        _ => source.device(),
                    };
                    layout = if opcode == Operation::Contiguous || device != source.device() {
                        Layout::from_shape(&shape)
                    } else {
                        view_output_layout(opcode, operation.attributes(), source.layout())?
                    };
                } else if matches!(operation.attributes(), OperationAttributes::Cast(target) if *target == first.data_type())
                    && first.layout().is_contiguous()
                {
                    // A cast to the operand's own type may share its contiguous storage; a strided operand is copied, as generated kernels write contiguous results.
                    layout = first.layout().clone();
                }
            }

            self.values[value.index()].set_metadata(data_type, layout, device);
        }
        Ok(())
    }

    fn materialize_reshape_input(&mut self, id: OperationId) -> Result<()> {
        let operation = self.required_operation(id)?;
        if !matches!(
            operation.opcode(),
            Some(Operation::Reshape | Operation::Flatten)
        ) {
            return Ok(());
        }

        let output = self.required_value(
            operation
                .result_values()
                .next()
                .ok_or_else(|| Error::missing_output("materialize reshape input"))?,
        )?;
        let Some(alias) = output.alias() else {
            return Ok(());
        };
        let reshape_alias = alias.clone();
        let source = self.required_value(alias.source())?.clone();
        if source.layout().reshape_view(output.shape())?.is_some() {
            return Ok(());
        }

        let output = output.id();
        let copy_value = self.insert_operation_before(
            id,
            Operation::Contiguous,
            OperationAttributes::None,
            source.id(),
            source.data_type(),
        )?;
        self.operations[id.index()].replace_input(source.id(), copy_value);
        self.values[output.index()].set_alias(Some(reshape_alias.with_source(copy_value)));
        Ok(())
    }

    /// Inserts a one-operand `opcode` of `input` before operation `id`, with a contiguous result of `data_type`, and returns its value.
    fn insert_operation_before(
        &mut self,
        id: OperationId,
        opcode: Operation,
        attributes: OperationAttributes,
        input: ValueId,
        data_type: DataType,
    ) -> Result<ValueId> {
        let source = self.required_value(input)?.clone();
        let source_location = self
            .required_operation(id)?
            .source_location()
            .map(ToOwned::to_owned);
        let inserted_id = OperationId::new(self.operations.len());
        let inserted_value = ValueId::new(self.values.len());
        let schema = operation_schema(opcode);
        let alias = (schema.alias_behavior() != AliasBehavior::None)
            .then(|| ValueAlias::new(source.id(), schema.alias_behavior() == AliasBehavior::View));
        self.values.push(Value::new(
            inserted_value,
            None,
            data_type,
            Layout::from_shape(source.shape()),
            source.device(),
            ValueBinding::produced_by_operation(inserted_id, 0),
            alias,
        ));
        self.constant_tensors.push(None);
        self.value_results.push(Some((inserted_id, 0)));
        let roles = schema.operands().roles_for_count(1);
        let output_role = schema
            .output_role(0)
            .ok_or_else(|| Error::missing_output("insert operation before"))?;
        self.operations.push(
            OperationNode::new(
                inserted_id,
                Some(opcode),
                Some(schema),
                attributes,
                vec![source.id()],
                roles,
                vec![OperationResult::new(0, inserted_value, output_role)],
            )
            .with_source_location(source_location),
        );
        for block in &mut self.blocks {
            block.insert_operation_before(id, inserted_id);
        }

        for partition in &mut self.partitions {
            partition.insert_operation_before(id, inserted_id);
        }
        Ok(inserted_value)
    }
}

impl Graph {
    // Gives a value another physical layout of the same shape, for a backend that lays out values no one else sees.
    #[cfg(feature = "cuda")]
    pub(crate) fn set_value_layout(&mut self, value: ValueId, layout: Layout) -> Result<()> {
        let (data_type, device) = {
            let current = self.required_value(value)?;
            (current.data_type(), current.device())
        };
        self.values[value.index()].set_metadata(data_type, layout, device);
        Ok(())
    }
}
