use std::{collections::HashMap, sync::Arc};

use singe::{
    DataType, Device, Error as SingeError, Shape, Tensor,
    nn::{Layer, Model, TokenBatch},
};

use crate::{
    error::{Error, Result},
    format::transformers::classifier_config,
    function::{Step, TensorFunction, TensorSpec},
    head::{ClassifierConfig, upload},
    source::Source,
};

#[derive(Debug)]
pub enum Forward<L> {
    Native(Model<L>),
    External(External),
}

impl<L> Clone for Forward<L> {
    fn clone(&self) -> Self {
        match self {
            Self::Native(model) => Self::Native(model.clone()),
            Self::External(external) => Self::External(external.clone()),
        }
    }
}

impl<L> Forward<L> {
    pub fn get_native(&self) -> Result<&Model<L>> {
        match self {
            Self::Native(model) => Ok(model),
            Self::External(_) => Err(Error::unsupported(
                "the layers of a model served from an exported program or an ONNX file",
            )),
        }
    }

    pub fn forward_group(&self, batch: &TokenBatch) -> Result<Tensor>
    where
        L: Layer<TokenBatch>,
    {
        match self {
            Self::Native(model) => Ok(model.forward_group(batch)?),
            Self::External(external) => external.tokens(batch),
        }
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor>
    where
        L: Layer,
    {
        match self {
            Self::Native(model) => Ok(model.forward(input)?),
            Self::External(external) => external.single(input),
        }
    }

    pub fn upload<const N: usize>(&self, dimensions: [i64; N], values: Vec<f32>) -> Result<Tensor> {
        match self {
            Self::Native(model) => Ok(upload(model.state(), dimensions, values)?),
            // An external function moves and casts its inputs itself.
            Self::External(_) => Ok(Tensor::from_vec(
                Device::Cpu,
                Shape::create(dimensions)?,
                values,
            )?),
        }
    }
}

#[derive(Debug, Clone)]
pub struct External {
    function: Arc<dyn TensorFunction>,
    outputs: Vec<String>,
    narrowed: Vec<String>,
}

impl External {
    pub fn create(function: Arc<dyn TensorFunction>, outputs: &[&str]) -> Result<Self> {
        let declared = &function.steps()[0].outputs;
        for output in outputs {
            if !declared.iter().any(|spec| spec.name == *output) {
                let names = declared
                    .iter()
                    .map(|spec| spec.name.as_str())
                    .collect::<Vec<_>>();
                return Err(Error::invalid(format!(
                    "a model without the output {output}, which the task reads; its outputs are {names:?}"
                )));
            }
        }
        Ok(Self {
            function,
            outputs: outputs.iter().map(|&output| output.to_owned()).collect(),
            narrowed: Vec::new(),
        })
    }

    #[must_use]
    pub fn narrowing(mut self, dimension: &str) -> Self {
        self.narrowed.push(dimension.to_owned());
        self
    }

    pub fn tokens(&self, batch: &TokenBatch) -> Result<Tensor> {
        self.run(vec![
            ("input_ids", batch.tokens().clone()),
            ("attention_mask", batch.padding().cast(DataType::I64)?),
            ("token_type_ids", batch.token_types().clone()),
        ])
    }

    pub fn classifier(
        source: &Source,
        function: Arc<dyn TensorFunction>,
        axis: usize,
    ) -> Result<(Self, ClassifierConfig)> {
        let config = classifier_config(source)?;
        let model =
            Self::create(function, &["logits"])?.check_labels(axis, config.labels().len())?;
        Ok((model, config))
    }

    fn check_labels(self, axis: usize, labels: usize) -> Result<Self> {
        let output = &self.outputs[0];
        let width = self
            .output(output)
            .and_then(|spec| spec.shape.get(axis).copied());
        if width.and_then(|width| usize::try_from(width).ok()) != Some(labels) {
            return Err(Error::invalid(format!(
                "output {output} of {width:?} at axis {axis}, where the config names {labels} labels"
            )));
        }
        Ok(self)
    }

    fn interface(&self) -> &Step {
        &self.function.steps()[0]
    }

    #[must_use]
    pub fn output(&self, name: &str) -> Option<&TensorSpec> {
        self.interface()
            .outputs
            .iter()
            .find(|spec| spec.name == name)
    }

    #[must_use]
    pub fn input(&self, name: &str) -> Option<&TensorSpec> {
        self.interface()
            .inputs
            .iter()
            .find(|spec| spec.name == name)
    }

    pub fn single(&self, input: &Tensor) -> Result<Tensor> {
        let declared = &self.interface().inputs;
        let [spec] = declared.as_slice() else {
            return Err(Error::invalid(format!(
                "a model of {} inputs, where the task feeds one",
                declared.len()
            )));
        };
        self.run(vec![(spec.name.as_str(), input.clone())])
    }

    fn run(&self, feeds: Vec<(&str, Tensor)>) -> Result<Tensor> {
        let declared = &self.interface().inputs;
        let feeds = feeds
            .into_iter()
            .filter(|(name, _)| declared.iter().any(|spec| spec.name == *name))
            .map(|(name, tensor)| (name.to_owned(), tensor))
            .collect::<HashMap<_, _>>();
        let mut outputs = self.function.run(&feeds, &self.narrowed)?;
        let mut tensors = self
            .outputs
            .iter()
            .map(|name| {
                outputs
                    .remove(name)
                    .ok_or_else(|| Error::invalid(format!("a run without the output {name}")))
            })
            .collect::<Result<Vec<_>>>()?;
        if tensors.len() == 1 {
            return Ok(tensors.remove(0));
        }

        let axis = i64::try_from(tensors[0].shape().dimensions().len())
            .map_err(|_| SingeError::out_of_range("output rank"))?;
        Ok(Tensor::stack(&tensors, axis)?)
    }
}

pub trait ExternalHead: Sized {
    fn external(source: &Source, function: Arc<dyn TensorFunction>) -> Result<Self>;
}
