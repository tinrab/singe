use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, HashMap, hash_map::Entry},
    fmt::{Debug, Display, Formatter, Result as FormatResult},
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use singe::{
    Device, Tensor,
    graph::{GraphExecutionPlan, GraphExecutor, GraphPrepareOptions},
    onnx::{Error as OnnxError, OnnxModel, OnnxOptions, OnnxValueInfo},
};
use singe_onnx::{DimensionValue, Model, TensorDataType, TypeValue, ValueInfo};

#[cfg(feature = "cuda_13_4")]
use std::hash::BuildHasher;

#[cfg(feature = "cuda_13_4")]
use singe::{
    backend::cuda::{
        LoweringPolicy,
        tuning::{GraphTuneOptions, Measurement, TuningRecords, tune_graph},
    },
    graph::{ComputePrecision, GraphCudaOptions},
};

use crate::{
    error::{Error, Result},
    function::{Bucketed, Shapes, Step, StepRunner, TensorFunction, TensorSpec},
};
#[cfg(feature = "cuda_13_4")]
use crate::{
    function::step_inputs,
    tune::{Tuning, read_records},
};

#[derive(Debug)]
pub struct Imported {
    pub step: Step,
    pub model: OnnxModel,
}

type DeclaredAxes = Vec<Option<DimensionValue>>;

fn declared(info: &ValueInfo) -> Option<DeclaredAxes> {
    let TypeValue::Tensor(tensor) = &info.value_type.as_ref()?.value else {
        return None;
    };
    Some(
        tensor
            .shape
            .as_ref()?
            .dimensions
            .iter()
            .map(|dimension| dimension.value.clone())
            .collect(),
    )
}

fn fed_inputs(file: &Model) -> Vec<&ValueInfo> {
    let initializers = file
        .graph
        .initializers
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect::<BTreeSet<_>>();
    file.graph
        .inputs
        .iter()
        .filter(|info| !initializers.contains(info.name.as_str()))
        .collect()
}

fn step_name(dimensions: &BTreeMap<String, i64>) -> String {
    if dimensions.is_empty() {
        return "main".to_owned();
    }

    dimensions
        .iter()
        .map(|(name, size)| format!("{name}_{size}"))
        .collect::<Vec<_>>()
        .join("_")
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
}

fn axes(
    info: &OnnxValueInfo,
    declared: Option<&DeclaredAxes>,
    dimensions: &BTreeMap<String, i64>,
) -> Vec<Option<String>> {
    (0..info.shape.dimensions().len())
        .map(|axis| {
            match declared
                .and_then(|declared| declared.get(axis))
                .cloned()
                .flatten()
            {
                Some(DimensionValue::Parameter(name)) if dimensions.contains_key(&name) => {
                    Some(name)
                }
                _ => None,
            }
        })
        .collect()
}

pub(crate) fn options(
    dimensions: &BTreeMap<String, i64>,
    shapes: &Shapes,
    device: Device,
) -> OnnxOptions {
    let mut options = OnnxOptions::new().with_device(device);
    for (name, &size) in dimensions {
        options = options.with_symbolic_dimension_range(name.clone(), size, Some(size), Some(size));
    }

    for (name, range) in &shapes.ranges {
        options = options.with_symbolic_dimension_range(
            name.clone(),
            range.min,
            Some(range.min),
            range.max,
        );
    }

    for (name, shape) in &shapes.inputs {
        options = options.with_input_shape(name.clone(), shape.clone());
    }
    options
}

pub(crate) fn spec(info: &OnnxValueInfo, axes: Vec<Option<String>>) -> TensorSpec {
    TensorSpec {
        name: info.name.clone(),
        data_type: info.data_type,
        shape: info.shape.dimensions().to_vec(),
        axes,
    }
}

fn output_shapes(model: &OnnxModel) -> Vec<Vec<i64>> {
    model
        .outputs()
        .iter()
        .map(|info| info.shape.dimensions().to_vec())
        .collect()
}

fn check_shapes(inputs: &[(&ValueInfo, Option<DeclaredAxes>)], shapes: &Shapes) -> Result<()> {
    if let Some(name) = shapes
        .dimensions
        .iter()
        .find(|(_, sizes)| sizes.is_empty() || sizes.iter().any(|&size| size < 1))
        .map(|(name, _)| name)
    {
        return Err(Error::invalid(format!(
            "dimension {name} of no sizes or a size below 1"
        )));
    }

    let named = inputs
        .iter()
        .filter_map(|(_, declared)| declared.as_ref())
        .flatten()
        .filter_map(|value| match value {
            Some(DimensionValue::Parameter(name)) => Some(name),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    if let Some((name, range)) = shapes
        .ranges
        .iter()
        .find(|(_, range)| range.min < 1 || range.max.is_some_and(|max| max < range.min))
    {
        return Err(Error::invalid(format!(
            "dimension {name} of the range {range}, which holds no size or a size below 1"
        )));
    }

    if let Some(name) = shapes
        .ranges
        .keys()
        .find(|name| shapes.dimensions.contains_key(*name))
    {
        return Err(Error::invalid(format!(
            "dimension {name} given both sizes and a range"
        )));
    }

    if let Some(name) = shapes
        .dimensions
        .keys()
        .chain(shapes.ranges.keys())
        .find(|name| !named.contains(name))
    {
        return Err(Error::invalid(format!(
            "dimension {name}, which no input has; the inputs' named dimensions are {named:?}"
        )));
    }

    if let Some(name) = shapes
        .inputs
        .keys()
        .find(|name| !inputs.iter().any(|(info, _)| info.name == **name))
    {
        return Err(Error::invalid(format!(
            "input {name}, which the file lacks"
        )));
    }

    let mut unfixed = Vec::new();
    for (info, declared) in inputs {
        if shapes.inputs.contains_key(&info.name) {
            continue;
        }

        let Some(axes) = declared else {
            unfixed.push(format!("{}, whose shape the file leaves out", info.name));
            continue;
        };
        for (axis, value) in axes.iter().enumerate() {
            match value {
                Some(DimensionValue::Value(_)) => {}
                Some(DimensionValue::Parameter(name))
                    if shapes.dimensions.contains_key(name) || shapes.ranges.contains_key(name) => {
                }
                Some(DimensionValue::Parameter(name)) => {
                    unfixed.push(format!(
                        "{} axis {axis}, which --dim {name}=N fixes, or which singe serve's --range {name}=MIN..MAX follows",
                        info.name
                    ));
                }
                _ => unfixed.push(format!(
                    "{} axis {axis}, which --input {}=D0x... fixes",
                    info.name, info.name
                )),
            }
        }
    }

    if !unfixed.is_empty() {
        return Err(Error::invalid(format!(
            "dynamic axes: {}",
            unfixed.join("; ")
        )));
    }
    Ok(())
}

fn padding_candidates(fed: &[(&str, Vec<Option<String>>)]) -> BTreeSet<String> {
    // The batch, the dimension every input has first, and what the attention mask covers are the dimensions a model ignores the padding of.
    let mut candidates = fed
        .iter()
        .filter(|(name, _)| *name == "attention_mask")
        .flat_map(|(_, axes)| axes.iter().flatten().cloned())
        .collect::<BTreeSet<_>>();
    let firsts = fed
        .iter()
        .map(|(_, axes)| axes.first().cloned().flatten())
        .collect::<BTreeSet<_>>();
    if let [Some(batch)] = firsts.into_iter().collect::<Vec<_>>().as_slice() {
        candidates.insert(batch.clone());
    }
    candidates
}

struct Outputs {
    axes: Vec<Vec<Option<String>>>,
    derived: BTreeSet<String>,
}

fn outputs(
    path: &Path,
    shapes: &Shapes,
    model: &OnnxModel,
    dimensions: &BTreeMap<String, i64>,
    candidates: &BTreeSet<String>,
) -> Outputs {
    // The file imports again with each candidate dimension doubled: an output axis that doubles with it follows it, and one that changes any other way is derived from it, so padding that dimension could not be undone.
    let base = output_shapes(model);
    let mut changes = base
        .iter()
        .map(|shape| vec![Vec::new(); shape.len()])
        .collect::<Vec<_>>();
    let mut derived = BTreeSet::new();
    for name in candidates {
        let size = dimensions[name];
        let mut doubled = dimensions.clone();
        doubled.insert(name.clone(), size * 2);
        let probe = OnnxModel::create(path, options(&doubled, shapes, Device::Cpu));
        let Ok(probe) = probe else {
            derived.insert(name.clone());
            continue;
        };
        for (output, (before, after)) in base.iter().zip(output_shapes(&probe)).enumerate() {
            if before.len() != after.len() {
                derived.insert(name.clone());
                continue;
            }

            for (axis, (&before, &after)) in before.iter().zip(&after).enumerate() {
                if before == after {
                    continue;
                }

                if before == size && after == size * 2 {
                    changes[output][axis].push(name.clone());
                } else {
                    derived.insert(name.clone());
                }
            }
        }
    }

    let axes = changes
        .into_iter()
        .map(|axes| {
            axes.into_iter()
                .map(|mut names| match names.len() {
                    0 => None,
                    1 => names.pop(),
                    _ => {
                        derived.extend(names);
                        None
                    }
                })
                .collect()
        })
        .collect();
    Outputs { axes, derived }
}

pub fn import(path: &Path, shapes: &Shapes, device: Device) -> Result<Vec<Imported>> {
    let file = Model::from_path(path).map_err(OnnxError::from)?;
    let inputs = fed_inputs(&file)
        .into_iter()
        .map(|info| (info, declared(info)))
        .collect::<Vec<_>>();
    check_shapes(&inputs, shapes)?;
    let declared_inputs = inputs
        .iter()
        .map(|(info, declared)| (info.name.as_str(), declared.as_ref()))
        .collect::<HashMap<_, _>>();
    // A ranged dimension names its axes as a stepped one does, at its least size.
    let input_axes = |info: &OnnxValueInfo, dimensions: &BTreeMap<String, i64>| {
        let declared = declared_inputs.get(info.name.as_str()).copied().flatten();
        let mut named = dimensions.clone();
        named.extend(
            shapes
                .ranges
                .iter()
                .map(|(name, range)| (name.clone(), range.min)),
        );
        axes(info, declared, &named)
    };

    let models = shapes
        .combinations()
        .into_iter()
        .map(|dimensions| {
            let model = OnnxModel::create(path, options(&dimensions, shapes, device))?;
            Ok((dimensions, model))
        })
        .collect::<Result<Vec<_>>>()?;
    let (first_dimensions, first_model) = &models[0];
    let fed = first_model
        .inputs()
        .iter()
        .filter(|info| !info.has_initializer)
        .map(|info| (info.name.as_str(), input_axes(info, first_dimensions)))
        .collect::<Vec<_>>();
    // A ranged dimension runs at each request's size, so it never pads.
    let mut candidates = padding_candidates(&fed);
    candidates.retain(|name| !shapes.ranges.contains_key(name));
    let Outputs {
        axes: output_axes,
        derived,
    } = outputs(path, shapes, first_model, first_dimensions, &candidates);
    let padded = candidates
        .difference(&derived)
        .cloned()
        .collect::<BTreeSet<_>>();

    let mut names = BTreeSet::new();
    let mut imported = Vec::new();
    for (dimensions, model) in models {
        let step = Step {
            name: step_name(&dimensions),
            inputs: model
                .inputs()
                .iter()
                .filter(|info| !info.has_initializer)
                .map(|info| spec(info, input_axes(info, &dimensions)))
                .collect(),
            outputs: model
                .outputs()
                .iter()
                .zip(&output_axes)
                .map(|(info, axes)| spec(info, axes.clone()))
                .collect(),
            dimensions,
            ranges: shapes.ranges.clone(),
            padded: padded.clone(),
        };
        if !names.insert(step.name.clone()) {
            return Err(Error::invalid(format!(
                "two steps named {}; rename a dimension so their names differ",
                step.name
            )));
        }

        imported.push(Imported { step, model });
    }
    Ok(imported)
}

struct GraphRunner {
    identity: usize,
    plans: Vec<(Vec<String>, GraphExecutionPlan)>,
    device: Device,
}

static RUNNERS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static EXECUTORS: RefCell<HashMap<(usize, usize), GraphExecutor>> = RefCell::new(HashMap::new());
}

impl Debug for GraphRunner {
    fn fmt(&self, f: &mut Formatter<'_>) -> FormatResult {
        f.debug_struct("GraphRunner")
            .field("steps", &self.plans.len())
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

impl StepRunner for GraphRunner {
    fn device(&self) -> Device {
        self.device
    }

    fn run_step(&self, step: usize, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        let (names, plan) = &self.plans[step];
        let inputs = names.iter().cloned().zip(inputs).collect::<HashMap<_, _>>();
        // An executor holds CUDA graph handles, which stay on the thread that made them, so each thread prepares its own from the shared plan.
        EXECUTORS.with_borrow_mut(|executors| {
            let executor = match executors.entry((self.identity, step)) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => entry.insert(plan.create_instance()?),
            };
            Ok(executor.run(&inputs)?)
        })
    }
}

impl Drop for GraphRunner {
    fn drop(&mut self) {
        // While the thread's locals are torn down, they drop the executors themselves, which is the one case `try_with` reports.
        let _ = EXECUTORS.try_with(|executors| {
            executors
                .borrow_mut()
                .retain(|(identity, _), _| *identity != self.identity);
        });
    }
}

pub fn runner(
    imported: Vec<Imported>,
    device: Device,
    options: &GraphPrepareOptions,
) -> Result<Arc<dyn TensorFunction>> {
    let mut steps = Vec::new();
    let mut plans = Vec::new();
    for imported in imported {
        let plan = imported.model.prepare(options.clone())?.execution_plan();
        let names = imported
            .step
            .inputs
            .iter()
            .map(|spec| spec.name.clone())
            .collect();
        plans.push((names, plan));
        steps.push(imported.step);
    }

    let runner = GraphRunner {
        identity: RUNNERS.fetch_add(1, Ordering::Relaxed),
        plans,
        device,
    };
    Ok(Arc::new(Bucketed::create(steps, runner)?))
}

#[cfg(feature = "cuda_13_4")]
pub fn tune_steps<S: BuildHasher>(
    imported: &[Imported],
    inputs: &[HashMap<String, Tensor, S>],
    device: Device,
    options: &GraphTuneOptions,
) -> Result<TuningRecords> {
    let mut merged: Option<TuningRecords> = None;
    for (imported, inputs) in imported.iter().zip(inputs) {
        let records = tune_graph(&imported.model.graph()?, inputs, device, options)?;
        match &mut merged {
            Some(merged) => merged.merge(records)?,
            None => merged = Some(records),
        }
    }
    merged.ok_or_else(|| Error::invalid("tuning a model of no steps"))
}

#[cfg(feature = "cuda_13_4")]
pub fn open(
    path: &Path,
    shapes: &Shapes,
    device: Device,
    tuning: Option<&Tuning>,
) -> Result<Arc<dyn TensorFunction>> {
    let imported = import(path, shapes, device)?;
    // A file does not say which convolutions its model needs at full precision, as DETR's backbone does, so all of them run so, as in an exported program.
    let mut options =
        GraphPrepareOptions::new(device).with_convolution_precision(ComputePrecision::Full);
    if let Some(tuning) = tuning {
        let records = if let Some(path) = &tuning.records {
            read_records(path, None, Measurement::GraphReplay)?
        } else {
            let tune_options = tuning
                .settings
                .options(LoweringPolicy::default(), Measurement::GraphReplay);
            let inputs = imported
                .iter()
                .map(|imported| {
                    step_inputs(&imported.step.inputs, tuning.inputs.as_deref(), device)
                })
                .collect::<Result<Vec<_>>>()?;
            let records = tune_steps(&imported, &inputs, device, &tune_options)?;
            if let Some(save) = &tuning.save {
                records.write(save)?;
            }
            records
        };
        options = options.with_cuda_options(GraphCudaOptions::new().with_tuning_records(records));
    }
    runner(imported, device, &options)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeclaredAxis {
    Size(i64),
    Named(String),
    Unknown,
}

impl Display for DeclaredAxis {
    fn fmt(&self, f: &mut Formatter<'_>) -> FormatResult {
        match self {
            Self::Size(size) => write!(f, "{size}"),
            Self::Named(name) => write!(f, "{name}"),
            Self::Unknown => write!(f, "?"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Declared {
    pub name: String,
    pub data_type: Option<TensorDataType>,
    pub axes: Vec<DeclaredAxis>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeclaredValues {
    pub inputs: Vec<Declared>,
    pub outputs: Vec<Declared>,
}

pub fn declared_values(path: &Path) -> Result<DeclaredValues> {
    let file = Model::from_path(path).map_err(OnnxError::from)?;
    let describe = |info: &ValueInfo| {
        let data_type = match info.value_type.as_ref().map(|value| &value.value) {
            Some(TypeValue::Tensor(tensor)) => Some(tensor.element_type),
            _ => None,
        };
        let axes = declared(info)
            .unwrap_or_default()
            .into_iter()
            .map(|value| match value {
                Some(DimensionValue::Value(size)) => DeclaredAxis::Size(size),
                Some(DimensionValue::Parameter(name)) => DeclaredAxis::Named(name),
                _ => DeclaredAxis::Unknown,
            })
            .collect();
        Declared {
            name: info.name.clone(),
            data_type,
            axes,
        }
    };
    Ok(DeclaredValues {
        inputs: fed_inputs(&file).into_iter().map(describe).collect(),
        outputs: file.graph.outputs.iter().map(describe).collect(),
    })
}
