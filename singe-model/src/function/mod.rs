//! Models outside Singe's layers, as functions of named tensors: an exported CUDA program, or an imported ONNX file.
//! A task pipeline drives one through [`Forward`](forward::Forward), feeding it the inputs `optimum` names and reading the outputs it names.

pub mod forward;
#[cfg(feature = "onnx")]
pub mod onnx;
#[cfg(feature = "cuda_13_4")]
pub mod program;

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt::{Debug, Display, Formatter, Result as FormatResult},
    fs,
    path::Path,
    result::Result as StdResult,
};

use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};
use singe::{DataType, Device, Error as SingeError, Shape, Tensor};

use crate::error::{Error, Result};

/// The sizes of an ONNX file's named dimensions and inputs that its steps import at.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Shapes {
    /// Each named dimension's sizes, a step per combination of them.
    pub dimensions: BTreeMap<String, Vec<i64>>,
    /// The named dimensions every step follows at each run's size, within their range.
    pub ranges: BTreeMap<String, DimensionRange>,
    /// Inputs whose every axis is fixed.
    pub inputs: BTreeMap<String, Vec<i64>>,
}

/// The sizes a named dimension takes at run time: from `min`, which the file imports at, up to `max`, or without bound when `max` is `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DimensionRange {
    pub min: i64,
    pub max: Option<i64>,
}

impl DimensionRange {
    #[must_use]
    pub fn holds(self, size: i64) -> bool {
        size >= self.min && self.max.is_none_or(|max| size <= max)
    }
}

impl Display for DimensionRange {
    fn fmt(&self, f: &mut Formatter<'_>) -> FormatResult {
        match self.max {
            Some(max) => write!(f, "{}..{max}", self.min),
            None => write!(f, "{}..", self.min),
        }
    }
}

impl Shapes {
    #[must_use]
    pub fn combinations(&self) -> Vec<BTreeMap<String, i64>> {
        let mut combinations = vec![BTreeMap::new()];
        for (name, values) in &self.dimensions {
            combinations = combinations
                .into_iter()
                .flat_map(|combination| {
                    values.iter().map(move |&value| {
                        let mut combination = combination.clone();
                        combination.insert(name.clone(), value);
                        combination
                    })
                })
                .collect();
        }
        combinations
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorSpec {
    pub name: String,
    #[serde(serialize_with = "write_type", deserialize_with = "read_type")]
    pub data_type: DataType,
    pub shape: Vec<i64>,
    pub axes: Vec<Option<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Step {
    pub name: String,
    pub dimensions: BTreeMap<String, i64>,
    /// The named dimensions the step follows at each run's size, which its specs hold at their least.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub ranges: BTreeMap<String, DimensionRange>,
    pub inputs: Vec<TensorSpec>,
    pub outputs: Vec<TensorSpec>,
    pub padded: BTreeSet<String>,
}

impl Display for Step {
    fn fmt(&self, f: &mut Formatter<'_>) -> FormatResult {
        let dimensions = self
            .dimensions
            .iter()
            .map(|(name, size)| format!("{name}={size}"))
            .chain(
                self.ranges
                    .iter()
                    .map(|(name, range)| format!("{name}={range}")),
            )
            .collect::<Vec<_>>();
        write!(f, "{}", dimensions.join(" "))
    }
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde passes the field by reference"
)]
fn write_type<S: Serializer>(data_type: &DataType, serializer: S) -> StdResult<S::Ok, S::Error> {
    serializer.serialize_str(&data_type.to_string())
}

fn read_type<'de, D: Deserializer<'de>>(deserializer: D) -> StdResult<DataType, D::Error> {
    String::deserialize(deserializer)?
        .parse()
        .map_err(D::Error::custom)
}

pub trait TensorFunction: Debug + Send + Sync {
    fn steps(&self) -> &[Step];

    fn run(
        &self,
        inputs: &HashMap<String, Tensor>,
        narrowed_by_caller: &[String],
    ) -> Result<HashMap<String, Tensor>>;
}

pub trait StepRunner: Debug + Send + Sync {
    fn device(&self) -> Device;

    fn run_step(&self, step: usize, inputs: Vec<Tensor>) -> Result<Vec<Tensor>>;
}

#[derive(Debug)]
pub struct Bucketed<R> {
    steps: Vec<Step>,
    runner: R,
}

impl<R: StepRunner> Bucketed<R> {
    pub fn create(steps: Vec<Step>, runner: R) -> Result<Self> {
        let Some(first) = steps.first() else {
            return Err(Error::invalid("a model of no steps"));
        };
        let interface = |specs: &[TensorSpec]| {
            specs
                .iter()
                .map(|spec| (spec.name.clone(), spec.axes.clone()))
                .collect::<Vec<_>>()
        };
        if steps.iter().any(|step| {
            interface(&step.inputs) != interface(&first.inputs)
                || interface(&step.outputs) != interface(&first.outputs)
                || step.padded != first.padded
                || step.ranges != first.ranges
        }) {
            return Err(Error::invalid(
                "steps of one model with different inputs, outputs, named axes, padded dimensions, or ranges",
            ));
        }
        Ok(Self { steps, runner })
    }

    fn sizes(&self, inputs: &HashMap<String, Tensor>) -> Result<BTreeMap<String, i64>> {
        let mut sizes = BTreeMap::new();
        for spec in &self.steps[0].inputs {
            let tensor = inputs
                .get(&spec.name)
                .ok_or_else(|| Error::invalid(format!("missing input {}", spec.name)))?;
            let dimensions = tensor.shape().dimensions();
            if dimensions.len() != spec.axes.len() {
                return Err(Error::invalid(format!(
                    "input {} of {} axes, where the model takes {}",
                    spec.name,
                    dimensions.len(),
                    spec.axes.len()
                )));
            }

            for (axis, (&size, name)) in dimensions.iter().zip(&spec.axes).enumerate() {
                match name {
                    Some(name) => {
                        if let Some(&other) = sizes.get(name)
                            && other != size
                        {
                            return Err(Error::invalid(format!(
                                "inputs whose dimension {name} is both {other} and {size}"
                            )));
                        }

                        sizes.insert(name.clone(), size);
                    }
                    None if size != spec.shape[axis] => {
                        return Err(Error::invalid(format!(
                            "input {} of {size} at axis {axis}, where the model takes {}",
                            spec.name, spec.shape[axis]
                        )));
                    }
                    None => {}
                }
            }
        }
        Ok(sizes)
    }

    fn pads(&self, name: &str, narrowed_by_caller: &[String]) -> bool {
        self.steps[0].padded.contains(name)
            || narrowed_by_caller.iter().any(|caller| caller == name)
    }

    fn choose(
        &self,
        sizes: &BTreeMap<String, i64>,
        narrowed_by_caller: &[String],
    ) -> Result<usize> {
        let cost = |step: &Step| {
            step.inputs
                .iter()
                .map(|spec| spec.shape.iter().map(|&size| size as f64).product::<f64>())
                .sum::<f64>()
        };
        let holds = |step: &Step| {
            sizes.iter().all(|(name, &size)| {
                if let Some(range) = step.ranges.get(name) {
                    return range.holds(size);
                }

                step.dimensions.get(name).is_some_and(|&step| {
                    if self.pads(name, narrowed_by_caller) {
                        size <= step
                    } else {
                        size == step
                    }
                })
            })
        };
        self.steps
            .iter()
            .enumerate()
            .filter(|(_, step)| holds(step))
            .min_by(|(_, first), (_, second)| cost(first).total_cmp(&cost(second)))
            .map(|(index, _)| index)
            .ok_or_else(|| self.no_step(sizes, narrowed_by_caller))
    }

    fn no_step(&self, sizes: &BTreeMap<String, i64>, narrowed_by_caller: &[String]) -> Error {
        let request = sizes
            .iter()
            .map(|(name, size)| format!("{name}={size}"))
            .collect::<Vec<_>>()
            .join(" ");
        let steps = self
            .steps
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let exact = sizes
            .keys()
            .filter(|name| {
                !self.pads(name, narrowed_by_caller) && !self.steps[0].ranges.contains_key(*name)
            })
            .map(String::as_str)
            .collect::<Vec<_>>();
        let rule = if exact.is_empty() {
            String::new()
        } else {
            format!(
                ", and {} must equal a step's, since the model cannot ignore padding there",
                exact.join(" and ")
            )
        };
        Error::invalid(format!(
            "inputs of {request}, which no exported step holds{rule}; the steps are {steps}"
        ))
    }
}

impl<R: StepRunner> TensorFunction for Bucketed<R> {
    fn steps(&self) -> &[Step] {
        &self.steps
    }

    fn run(
        &self,
        inputs: &HashMap<String, Tensor>,
        narrowed_by_caller: &[String],
    ) -> Result<HashMap<String, Tensor>> {
        let sizes = self.sizes(inputs)?;
        let index = self.choose(&sizes, narrowed_by_caller)?;
        let step = &self.steps[index];
        let device = self.runner.device();
        let feeds = step
            .inputs
            .iter()
            .map(|spec| {
                let tensor = inputs[&spec.name].cast(spec.data_type)?;
                // An axis of a ranged dimension takes the run's size rather than the step's.
                let shape = spec
                    .shape
                    .iter()
                    .zip(&spec.axes)
                    .map(|(&size, name)| {
                        name.as_ref()
                            .filter(|name| step.ranges.contains_key(*name))
                            .and_then(|name| sizes.get(name))
                            .copied()
                            .unwrap_or(size)
                    })
                    .collect::<Vec<_>>();
                padded(&tensor, &shape, device)
            })
            .collect::<Result<Vec<_>>>()?;
        let outputs = self.runner.run_step(index, feeds)?;
        if outputs.len() != step.outputs.len() {
            return Err(SingeError::missing_output("exported step").into());
        }

        step.outputs
            .iter()
            .zip(outputs)
            .map(|(spec, mut tensor)| {
                for (axis, name) in spec.axes.iter().enumerate() {
                    let Some(&size) = name.as_ref().and_then(|name| sizes.get(name)) else {
                        continue;
                    };
                    let axis = i64::try_from(axis).map_err(|_| SingeError::out_of_range("axis"))?;
                    tensor = tensor.narrow(axis, 0, size)?;
                }
                Ok((spec.name.clone(), tensor.contiguous()?))
            })
            .collect()
    }
}

fn padded(tensor: &Tensor, shape: &[i64], device: Device) -> Result<Tensor> {
    let tensor = tensor.contiguous()?;
    let dimensions = tensor.shape().dimensions();
    if dimensions == shape {
        return Ok(tensor.to_device(device)?);
    }

    let index = |value: i64| usize::try_from(value).map_err(|_| SingeError::out_of_range("shape"));
    let element = tensor.data_type().size_bytes();
    let bytes = tensor.to_bytes()?;
    let target = shape
        .iter()
        .map(|&size| index(size))
        .collect::<StdResult<Vec<_>, _>>()?;
    let source = dimensions
        .iter()
        .map(|&size| index(size))
        .collect::<StdResult<Vec<_>, _>>()?;
    let mut values = vec![0_u8; target.iter().product::<usize>() * element];
    if let Some((&width, outer)) = source.split_last() {
        let row = width * element;
        let rows = outer.iter().product::<usize>();
        for number in 0..rows {
            let mut rest = number;
            let mut offset = 0;
            let mut stride = target.last().copied().unwrap_or(1);
            for (&size, &length) in outer.iter().zip(&target[..outer.len()]).rev() {
                offset += (rest % size) * stride;
                rest /= size;
                stride *= length;
            }

            values[offset * element..offset * element + row]
                .copy_from_slice(&bytes[number * row..(number + 1) * row]);
        }
    } else {
        values.copy_from_slice(&bytes);
    }
    Ok(Tensor::from_bytes(
        device,
        tensor.data_type(),
        Shape::create(shape)?,
        &values,
    )?)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportManifest {
    pub source: String,
    pub task: Option<String>,
    pub architecture: String,
    pub libraries: Vec<String>,
    pub tuned: bool,
    pub steps: Vec<Step>,
}

impl ExportManifest {
    pub const FILE: &str = "export.json";

    pub fn read(directory: &Path) -> Result<Self> {
        let path = directory.join(Self::FILE);
        let text = fs::read_to_string(&path).map_err(Error::io(&path))?;
        serde_json::from_str(&text).map_err(Error::json(&path))
    }

    pub fn write(&self, directory: &Path) -> Result<()> {
        let path = directory.join(Self::FILE);
        let text = serde_json::to_string_pretty(self).map_err(Error::json(&path))?;
        fs::write(&path, text).map_err(Error::io(&path))
    }
}

pub fn example_inputs(specs: &[TensorSpec], device: Device) -> Result<HashMap<String, Tensor>> {
    specs
        .iter()
        .map(|spec| {
            let shape = Shape::create(spec.shape.clone())?;
            let tensor = if spec.data_type.is_float() {
                Tensor::uniform(device, shape, -1.0, 1.0, spec.data_type, None)?
            } else {
                let size = spec.data_type.size_bytes();
                let mut bytes = vec![0_u8; shape.element_count() * size];
                if spec.data_type == DataType::Bool || spec.name.contains("mask") {
                    let low = if cfg!(target_endian = "little") {
                        0
                    } else {
                        size - 1
                    };
                    for element in bytes.chunks_mut(size) {
                        element[low] = 1;
                    }
                }
                Tensor::from_bytes(device, spec.data_type, shape, &bytes)?
            };
            Ok((spec.name.clone(), tensor))
        })
        .collect()
}

pub fn step_inputs(
    specs: &[TensorSpec],
    inputs: Option<&Path>,
    device: Device,
) -> Result<HashMap<String, Tensor>> {
    let mut generated = example_inputs(specs, device)?;
    let Some(path) = inputs else {
        return Ok(generated);
    };
    for (name, tensor) in read_inputs(path, device)? {
        let fits = specs.iter().any(|spec| {
            spec.name == name
                && spec.data_type == tensor.data_type()
                && spec.shape == tensor.shape().dimensions()
        });
        if fits {
            generated.insert(name, tensor);
        }
    }
    Ok(generated)
}

pub fn read_inputs(path: &Path, device: Device) -> Result<HashMap<String, Tensor>> {
    let bytes = fs::read(path).map_err(Error::io(path))?;
    let file = SafeTensors::deserialize(&bytes)
        .map_err(|error| Error::invalid(format!("safetensors file {}: {error}", path.display())))?;
    file.tensors()
        .into_iter()
        .map(|(name, view)| {
            let data_type = match view.dtype() {
                Dtype::BOOL => DataType::Bool,
                Dtype::U8 => DataType::U8,
                Dtype::I8 => DataType::I8,
                Dtype::U16 => DataType::U16,
                Dtype::I16 => DataType::I16,
                Dtype::U32 => DataType::U32,
                Dtype::I32 => DataType::I32,
                Dtype::U64 => DataType::U64,
                Dtype::I64 => DataType::I64,
                Dtype::F16 => DataType::F16,
                Dtype::BF16 => DataType::Bf16,
                Dtype::F32 => DataType::F32,
                Dtype::F64 => DataType::F64,
                other => {
                    return Err(Error::unsupported(format!(
                        "input {name} of safetensors type {other:?}"
                    )));
                }
            };
            let shape = view
                .shape()
                .iter()
                .map(|&size| i64::try_from(size).map_err(|_| SingeError::out_of_range("shape")))
                .collect::<StdResult<Vec<_>, _>>()?;
            let tensor = Tensor::from_bytes(device, data_type, Shape::create(shape)?, view.data())?;
            Ok((name, tensor))
        })
        .collect()
}
