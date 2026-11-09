use std::collections::{HashMap, HashSet};

use singe_onnx::{DimensionValue, Graph, Tensor as OnnxTensor, TypeValue, ValueInfo};

use crate::{
    onnx::{
        error::{Error, Result},
        importer::{ImportedValue, Importer},
        model::OnnxValueInfo,
        shape::{concrete_shape, value_tensor_type},
        tensor::{data_type, tensor_from_onnx},
    },
    shape::Shape,
};

impl Importer {
    pub(crate) fn import_initializers(&mut self, initializers: &[OnnxTensor]) -> Result<()> {
        for initializer in initializers {
            if initializer.name.is_empty() {
                return Err(Error::invalid_initializer(
                    "<unnamed>",
                    "initializer name is required",
                ));
            }

            let tensor = tensor_from_onnx(initializer, &self.options, self.model_dir.as_deref())?;
            if self
                .initializer_tensors
                .insert(initializer.name.clone(), tensor)
                .is_some()
            {
                return Err(Error::DuplicateValue(initializer.name.clone()));
            }
        }
        Ok(())
    }

    pub(crate) fn import_initializer_constants(&mut self) -> Result<()> {
        let existing = self.values.keys().cloned().collect::<HashSet<_>>();
        let mut initializers = self
            .initializer_tensors
            .iter()
            .map(|(name, tensor)| (name.clone(), tensor.clone()))
            .collect::<Vec<_>>();
        initializers.sort_by(|(left, _), (right, _)| left.cmp(right));
        for (name, tensor) in initializers {
            if existing.contains(&name) {
                continue;
            }

            let node = self.graph.add_input(
                &name,
                tensor.data_type(),
                tensor.shape().clone(),
                tensor.device(),
            );
            self.graph.add_static_initializer(&name, tensor.clone())?;
            self.register_value(
                &name,
                ImportedValue::new(node, tensor.data_type(), tensor.shape().clone()),
            )?;
        }
        Ok(())
    }
}

impl Importer {
    pub(crate) fn import_inputs(&mut self, inputs: &[ValueInfo]) -> Result<()> {
        for input in inputs {
            if input.name.is_empty() {
                return Err(Error::InvalidModel {
                    reason: "graph input name is required".into(),
                });
            }

            let tensor_type = value_tensor_type(input)?;
            let data_type = data_type(&input.name, tensor_type.element_type)?;
            let shape = concrete_shape(&input.name, tensor_type.shape.as_ref(), &self.options)?;
            let node =
                self.graph
                    .add_input(&input.name, data_type, shape.clone(), self.options.device);
            let has_initializer =
                if let Some(initializer) = self.initializer_tensors.get(&input.name) {
                    self.graph
                        .add_initializer(&input.name, initializer.clone())?;
                    true
                } else {
                    false
                };
            self.add_symbolic_guards(input, &shape)?;
            self.register_value(
                &input.name,
                ImportedValue::new(node, data_type, shape.clone()),
            )?;
            self.inputs.push(OnnxValueInfo {
                name: input.name.clone(),
                data_type,
                shape,
                has_initializer,
            });
        }
        Ok(())
    }

    fn add_symbolic_guards(&mut self, input: &ValueInfo, shape: &Shape) -> Result<()> {
        let Some(value_type) = &input.value_type else {
            return Ok(());
        };
        let TypeValue::Tensor(tensor) = &value_type.value else {
            return Ok(());
        };
        let Some(tensor_shape) = &tensor.shape else {
            return Ok(());
        };
        for (axis, dimension) in tensor_shape.dimensions.iter().enumerate() {
            if let Some(DimensionValue::Parameter(symbol)) = &dimension.value {
                let (min, max) = self.options.symbolic_bounds(symbol);
                let value = shape.dimensions()[axis];
                if min.is_some_and(|min| value < min) || max.is_some_and(|max| value > max) {
                    return Err(Error::UnsupportedShape {
                        name: input.name.clone(),
                        reason: format!(
                            "representative dimension `{symbol}` is outside its bounds"
                        ),
                    });
                }

                self.graph
                    .add_input_dimension_guard(&input.name, axis, symbol, min, max)?;
            }
        }
        Ok(())
    }
}

impl Importer {
    pub(crate) fn import_outputs(&self, outputs: &[ValueInfo]) -> Result<Vec<OnnxValueInfo>> {
        let mut seen = HashSet::new();
        for output in outputs {
            if output.name.is_empty() || !seen.insert(output.name.as_str()) {
                return Err(Error::DuplicateValue(output.name.clone()));
            }

            value_tensor_type(output)?;
            self.validate_value_info(output)?;
        }
        outputs
            .iter()
            .map(|output| {
                let value = self
                    .values
                    .get(&output.name)
                    .ok_or_else(|| Error::MissingValue(output.name.clone()))?;
                Ok(OnnxValueInfo {
                    name: output.name.clone(),
                    data_type: value.data_type,
                    shape: value.shape.clone(),
                    has_initializer: false,
                })
            })
            .collect()
    }
}

impl Importer {
    pub(crate) fn validate_value_info(&self, info: &ValueInfo) -> Result<()> {
        let inferred = self.value(&info.name)?;
        if info.value_type.is_none() {
            return Ok(());
        }

        let declared = value_tensor_type(info)?;
        let declared_type = data_type(&info.name, declared.element_type)?;
        if declared_type != inferred.data_type {
            return Err(Error::UnsupportedShape {
                name: info.name.clone(),
                reason: "declared dtype does not match inferred dtype".into(),
            });
        }

        if let Some(shape) = &declared.shape {
            if shape.dimensions.len() != inferred.shape.dimensions().len() {
                return Err(Error::UnsupportedShape {
                    name: info.name.clone(),
                    reason: "declared rank does not match inferred rank".into(),
                });
            }
            for (declared, actual) in shape.dimensions.iter().zip(inferred.shape.dimensions()) {
                if let Some(DimensionValue::Value(value)) = declared.value
                    && (value < 0 || value != *actual)
                {
                    return Err(Error::UnsupportedShape {
                        name: info.name.clone(),
                        reason: "declared dimension does not match inferred dimension".into(),
                    });
                }
            }
        }
        Ok(())
    }
}

impl Importer {
    pub(crate) fn validate_symbolic_metadata(&self, graph: &Graph) -> Result<()> {
        let mut symbols = HashMap::new();
        for info in graph
            .inputs
            .iter()
            .chain(&graph.value_info)
            .chain(&graph.outputs)
        {
            let value = self.value(&info.name)?;
            if info.value_type.is_none() {
                continue;
            }
            if let Some(shape) = &value_tensor_type(info)?.shape {
                for (dimension, actual) in shape.dimensions.iter().zip(value.shape.dimensions()) {
                    if let Some(DimensionValue::Parameter(symbol)) = &dimension.value {
                        let (min, max) = self.options.symbolic_bounds(symbol);
                        if symbol.is_empty()
                            || min.is_some_and(|min| *actual < min)
                            || max.is_some_and(|max| *actual > max)
                            || symbols
                                .insert(symbol, *actual)
                                .is_some_and(|previous| previous != *actual)
                        {
                            return Err(Error::UnsupportedShape {
                                name: info.name.clone(),
                                reason: format!("inconsistent symbolic dimension `{symbol}`"),
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
