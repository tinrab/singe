//! Replays the `PyTorch` conformance fixture on CPU eager, CUDA eager, and compiled CUDA graphs.
//!
//! `scripts/references/suites/conformance.py` writes one case per line: a registry operation, its attributes, inputs read from a pool of stored tensors through optional views, and the outputs `PyTorch` returns or the error it raises.
//! Each execution path gives every case an outcome, and the cases that do not pass are listed in a baseline per path under `tests/baselines/conformance/`.
//! The test fails when an outcome changes either way, so a fix shows up as a baseline line removed.
//! `SINGE_UPDATE_CONFORMANCE=1` rewrites the baselines.

use std::{
    collections::{BTreeMap, HashMap},
    env,
    fmt::Write as _,
    fs,
    panic::{AssertUnwindSafe, catch_unwind},
    path::{Path, PathBuf},
    sync::Arc,
};

#[cfg(all(feature = "testing", feature = "cuda"))]
use std::iter::once;

#[cfg(all(feature = "testing", feature = "cuda"))]
use anyhow::anyhow;
use half::{bf16, f16};
use serde::Deserialize;
use serde_json::Value;
use singe::{
    DataType, Device, Shape, Tensor,
    backend::registry::{
        BatchNormalizationAttributes, ClampAttributes, ContractionAttributes, Conv2dAttributes,
        FlattenAttributes, IndexPutAttributes, LayerNormalizationAttributes, Operation,
        OperationAttributes, PadAttributes, Pool2dAttributes, QuantizationAttributes,
        ReductionAttributes, ScaledDotProductAttentionAttributes, SliceAttributes, SortAttributes,
        StridedAttributes, TopKAttributes, TransposeAttributes, registered_operations,
    },
    dispatch::{call, call_one},
    error::Result,
    scalar::Scalar,
    tensor::{InterpolateMode, LossReduction, NormOrder},
};
#[cfg(all(feature = "testing", feature = "cuda"))]
use singe::{
    backend::cuda::export::{CudaExport, CudaProgram, Library, ProgramStep},
    testing::{CompiledProgram, program_target},
};
#[cfg(all(feature = "testing", feature = "cuda"))]
use singe::{
    backend::cuda::{self, KernelLanguage},
    graph::{Graph, GraphCudaOptions},
};
#[cfg(all(feature = "testing", feature = "cuda"))]
use singe_cuda::testing::device_lock;

const FIXTURE: &str = include_str!("../references/conformance/cases.jsonl");

/// Every data type a case can name.
const DATA_TYPES: [DataType; 16] = [
    DataType::Bool,
    DataType::U8,
    DataType::I8,
    DataType::U16,
    DataType::I16,
    DataType::U32,
    DataType::I32,
    DataType::U64,
    DataType::I64,
    DataType::F16,
    DataType::Bf16,
    DataType::F32,
    DataType::F64,
    DataType::F8E4M3,
    DataType::F8E5M2,
    DataType::F4E2M1,
];

/// Registry operations the fixture has no cases for, and why.
const UNCOVERED: [(&str, &str); 11] = [
    (
        "paged_attention_output",
        "is `paged_attention`'s first result as graph capture records it; its cases cover it",
    ),
    (
        "paged_attention_lse",
        "is `paged_attention`'s second result as graph capture records it; its cases cover it",
    ),
    (
        "paged_mla_attention_output",
        "is `paged_mla_attention`'s first result as graph capture records it; its cases cover it",
    ),
    (
        "paged_mla_attention_lse",
        "is `paged_mla_attention`'s second result as graph capture records it; its cases cover it",
    ),
    (
        "assign",
        "writes its destination in place; `tensor_mutation` covers it",
    ),
    (
        "index_write",
        "is an eager `index_put_` writing its destination in place; `cache_writes` checks it against `index_put`, which conformance covers",
    ),
    (
        "copy_to",
        "moves a tensor between devices; the transfer tests cover it",
    ),
    (
        "uniform",
        "draws from Singe's generator, whose streams differ from PyTorch's; `random` checks it",
    ),
    (
        "normal",
        "draws from Singe's generator, whose streams differ from PyTorch's; `random` checks it",
    ),
    (
        "bernoulli",
        "draws from Singe's generator, whose streams differ from PyTorch's; `random` checks it",
    ),
    (
        "multinomial",
        "draws from Singe's generator, whose streams differ from PyTorch's; `random` checks it",
    ),
];

#[derive(Deserialize)]
struct Encoded {
    dtype: String,
    shape: Vec<i64>,
    bytes: String,
}

#[derive(Deserialize)]
struct Operand {
    tensor: String,
    #[serde(default)]
    views: Vec<Value>,
}

#[derive(Deserialize)]
struct Case {
    #[serde(rename = "case")]
    name: String,
    operation: String,
    #[serde(default)]
    attributes: Option<Value>,
    /// The arguments of a composite tensor method, which the case names as its operation.
    #[serde(default)]
    composite: Option<Value>,
    inputs: Vec<Operand>,
    #[serde(default)]
    outputs: Option<Vec<Encoded>>,
    #[serde(default)]
    error: Option<String>,
}

struct Fixture {
    inputs: HashMap<String, Encoded>,
    cases: Vec<Case>,
}

impl Fixture {
    fn load() -> Self {
        let mut inputs = HashMap::new();
        let mut cases = Vec::new();
        for line in FIXTURE.lines().skip(1) {
            let value: Value = serde_json::from_str(line).unwrap();
            if let Some(key) = value.get("input").and_then(Value::as_str) {
                inputs.insert(key.to_owned(), serde_json::from_value(value).unwrap());
            } else {
                cases.push(serde_json::from_value(value).unwrap());
            }
        }
        Self { inputs, cases }
    }
}

/// An execution path the harness replays every case on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExecutionPath {
    Cpu,
    // Meta tensors carry no values, so this path checks the metadata of the CPU results; no baseline lists it.
    Meta,
    #[cfg(all(feature = "testing", feature = "cuda"))]
    Cuda,
    #[cfg(all(feature = "testing", feature = "cuda"))]
    Graph,
    // The graph path with generated kernels emitted as Tile IR where the tile emitter supports them; no baseline lists it.
    #[cfg(all(feature = "testing", feature = "cuda"))]
    TileGraph,
}

impl ExecutionPath {
    fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Meta => "meta",
            #[cfg(all(feature = "testing", feature = "cuda"))]
            Self::Cuda => "cuda",
            #[cfg(all(feature = "testing", feature = "cuda"))]
            Self::Graph => "graph",
            #[cfg(all(feature = "testing", feature = "cuda"))]
            Self::TileGraph => "tile graph",
        }
    }

    fn device(self) -> Device {
        match self {
            Self::Cpu => Device::Cpu,
            Self::Meta => Device::Meta,
            #[cfg(all(feature = "testing", feature = "cuda"))]
            Self::Cuda | Self::Graph | Self::TileGraph => Device::Cuda(0),
        }
    }
}

/// How a path's result compares with `PyTorch`'s.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Outcome {
    /// The results match, or both reject the case.
    Pass,
    /// `PyTorch` has no kernel for the data type and Singe computes a result.
    Extension,
    /// Singe rejects a case `PyTorch` computes.
    Rejected,
    /// A result's data type, shape, or values differ.
    Mismatch,
    /// Singe computes a case `PyTorch` rejects as invalid.
    Accepted,
    /// Singe panics.
    Panicked,
}

impl Outcome {
    const ALL: [Self; 6] = [
        Self::Pass,
        Self::Extension,
        Self::Rejected,
        Self::Mismatch,
        Self::Accepted,
        Self::Panicked,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Extension => "extension",
            Self::Rejected => "rejected",
            Self::Mismatch => "mismatch",
            Self::Accepted => "accepted",
            Self::Panicked => "panicked",
        }
    }

    fn parse(name: &str) -> Self {
        Self::ALL
            .into_iter()
            .find(|outcome| outcome.name() == name)
            .unwrap_or_else(|| panic!("unknown conformance outcome {name}"))
    }
}

fn data_type(name: &str) -> DataType {
    DATA_TYPES
        .into_iter()
        .find(|data_type| format!("{data_type:?}") == name)
        .unwrap_or_else(|| panic!("unknown conformance data type {name}"))
}

fn operation(name: &str) -> Operation {
    registered_operations()
        .iter()
        .copied()
        .find(|operation| operation.name() == name)
        .unwrap_or_else(|| panic!("conformance case names unknown operation {name}"))
}

fn decode(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(&hex[index..index + 2], 16).unwrap())
        .collect()
}

fn integers(value: &Value) -> Vec<i64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item.as_i64().unwrap())
        .collect()
}

fn pair(value: &Value) -> (i64, i64) {
    let values = integers(value);
    (values[0], values[1])
}

fn quad(value: &Value) -> (i64, i64, i64, i64) {
    let values = integers(value);
    (values[0], values[1], values[2], values[3])
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "fixture attributes are written from F32 values"
)]
fn float(value: &Value) -> f32 {
    value.as_f64().unwrap() as f32
}

fn flag(value: &Value) -> bool {
    value.as_bool().unwrap()
}

/// Decodes a case's attributes, written as one key naming the `OperationAttributes` variant.
fn attributes(value: Option<&Value>) -> OperationAttributes {
    use OperationAttributes as Attr;
    let Some(Value::Object(object)) = value else {
        return Attr::None;
    };
    let (kind, value) = object.iter().next().unwrap();
    let field = |name: &str| &value[name];
    match kind.as_str() {
        "Scalar" => {
            let (kind, scalar) = value.as_object().unwrap().iter().next().unwrap();
            Attr::Scalar(match kind.as_str() {
                "float" => Scalar::Float(scalar.as_f64().unwrap()),
                "int" => Scalar::Signed(scalar.as_i64().unwrap()),
                "bool" => Scalar::Bool(flag(scalar)),
                other => panic!("unknown scalar kind {other}"),
            })
        }
        "Dimension" => Attr::Dimension(value.as_i64().unwrap()),
        "Reduction" => Attr::Reduction(ReductionAttributes {
            dimensions: integers(field("dimensions")),
            keepdim: flag(field("keepdim")),
        }),
        "Sort" => Attr::Sort(SortAttributes {
            dim: field("dim").as_i64().unwrap(),
            descending: flag(field("descending")),
        }),
        "TopK" => Attr::TopK(TopKAttributes {
            k: field("k").as_i64().unwrap(),
            dim: field("dim").as_i64().unwrap(),
            largest: flag(field("largest")),
            sorted: flag(field("sorted")),
        }),
        "Clamp" => Attr::Clamp(ClampAttributes {
            min: float(field("min")),
            max: float(field("max")),
        }),
        "Lerp" => Attr::Lerp(float(value)),
        "Transpose" => Attr::Transpose(TransposeAttributes {
            dim0: field("dim0").as_i64().unwrap(),
            dim1: field("dim1").as_i64().unwrap(),
        }),
        "Permute" => Attr::Permute(integers(value)),
        "Shape" => Attr::Shape(Shape::create(integers(value)).unwrap()),
        "Slice" => Attr::Slice(SliceAttributes {
            dim: field("dim").as_i64().unwrap(),
            start: field("start").as_i64().unwrap(),
            length: field("length").as_i64().unwrap(),
            step: field("step").as_i64().unwrap(),
        }),
        "Concat" => Attr::Concat(value.as_i64().unwrap()),
        "IndexSelect" => Attr::IndexSelect(value.as_i64().unwrap()),
        "IndexPut" => Attr::IndexPut(IndexPutAttributes {
            dim: field("dim").as_i64().unwrap(),
            accumulate: flag(field("accumulate")),
        }),
        "RepeatKv" => Attr::RepeatKv(value.as_i64().unwrap()),
        "Flatten" => Attr::Flatten(FlattenAttributes {
            start_dim: field("start_dim").as_i64().unwrap(),
            end_dim: field("end_dim").as_i64().unwrap(),
        }),
        "Strided" => Attr::Strided(StridedAttributes {
            dimensions: integers(field("dimensions")),
            strides: integers(field("strides")),
            offset: usize::try_from(field("offset").as_u64().unwrap()).unwrap(),
        }),
        "Pad" => Attr::Pad(PadAttributes {
            padding: field("padding")
                .as_array()
                .unwrap()
                .iter()
                .map(pair)
                .collect(),
            value: float(field("value")),
        }),
        "Contraction" => Attr::Contraction(ContractionAttributes {
            a_mode: integers(field("a_mode")),
            b_mode: integers(field("b_mode")),
            c_mode: integers(field("c_mode")),
            d_mode: integers(field("d_mode")),
            alpha: float(field("alpha")),
            beta: float(field("beta")),
        }),
        "LayerNormalization" => Attr::LayerNormalization(LayerNormalizationAttributes {
            normalized_shape: integers(field("normalized_shape")),
            has_weight: flag(field("has_weight")),
            has_bias: flag(field("has_bias")),
            eps: float(field("eps")),
        }),
        "BatchNormalization" => Attr::BatchNormalization(BatchNormalizationAttributes {
            has_weight: flag(field("has_weight")),
            has_bias: flag(field("has_bias")),
            has_running_mean: flag(field("has_running_mean")),
            has_running_var: flag(field("has_running_var")),
            training: flag(field("training")),
            momentum: float(field("momentum")),
            eps: float(field("eps")),
        }),
        "Conv2d" => Attr::Conv2d(Conv2dAttributes {
            has_bias: flag(field("has_bias")),
            stride: pair(field("stride")),
            padding: quad(field("padding")),
            dilation: pair(field("dilation")),
            groups: field("groups").as_i64().unwrap(),
        }),
        "Pool2d" => Attr::Pool2d(Pool2dAttributes {
            kernel_size: pair(field("kernel_size")),
            stride: pair(field("stride")),
            padding: quad(field("padding")),
            dilation: pair(field("dilation")),
            ceil_mode: flag(field("ceil_mode")),
            count_include_pad: field("count_include_pad").as_bool(),
        }),
        "ScaledDotProductAttention" => {
            Attr::ScaledDotProductAttention(ScaledDotProductAttentionAttributes {
                scale: field("scale").as_f64().map(|_| float(field("scale"))),
                causal: flag(field("causal")),
                window: field("window").as_i64(),
                softcap: field("softcap").as_f64().map(|_| float(field("softcap"))),
                masked: field("masked").as_bool().unwrap_or(false),
                sinks: field("sinks").as_bool().unwrap_or(false),
                key_lengths: field("key_lengths").as_bool().unwrap_or(false),
            })
        }
        "Quantization" => Attr::Quantization(QuantizationAttributes {
            axis: field("axis").as_i64(),
            block_size: field("block_size").as_i64(),
            data_type: field("data_type").as_str().map(data_type),
            packed: None,
        }),
        "Cast" => Attr::Cast(data_type(value.as_str().unwrap())),
        other => panic!("unknown conformance attributes {other}"),
    }
}

impl Encoded {
    fn tensor(&self, device: Device) -> Result<Tensor> {
        Tensor::from_bytes(
            Device::Cpu,
            data_type(&self.dtype),
            Shape::create(self.shape.clone())?,
            &decode(&self.bytes),
        )?
        .to_device(device)
    }
}

/// Builds an operand on `device`: its pool tensor, then its views in order.
fn operand(fixture: &Fixture, operand: &Operand, device: Device) -> Result<Tensor> {
    let mut tensor = fixture.inputs[&operand.tensor].tensor(device)?;
    for view in &operand.views {
        let view = view.as_array().unwrap();
        tensor = match view[0].as_str().unwrap() {
            "transpose" => {
                tensor.transpose(view[1].as_i64().unwrap(), view[2].as_i64().unwrap())?
            }
            "narrow" => tensor.narrow(
                view[1].as_i64().unwrap(),
                view[2].as_i64().unwrap(),
                view[3].as_i64().unwrap(),
            )?,
            "expand" => tensor.expand(&Shape::create(integers(&view[1]))?)?,
            other => panic!("unknown conformance view {other}"),
        };
    }
    Ok(tensor)
}

/// Runs a case on `path` and returns its outputs.
fn run_case(case: &Case, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
    case.composite.as_ref().map_or_else(
        || {
            call(
                operation(&case.operation),
                attributes(case.attributes.as_ref()),
                &inputs.iter().collect::<Vec<_>>(),
            )
        },
        |arguments| composite(&case.operation, arguments, inputs),
    )
}

fn execute(fixture: &Fixture, case: &Case, path: ExecutionPath) -> Result<Vec<Tensor>> {
    let inputs = case
        .inputs
        .iter()
        .map(|input| operand(fixture, input, path.device()))
        .collect::<Result<Vec<_>>>()?;
    let run = |inputs: &[Tensor]| run_case(case, inputs);
    match path {
        ExecutionPath::Cpu | ExecutionPath::Meta => run(&inputs),
        #[cfg(all(feature = "testing", feature = "cuda"))]
        ExecutionPath::Cuda => run(&inputs),
        #[cfg(all(feature = "testing", feature = "cuda"))]
        ExecutionPath::Graph | ExecutionPath::TileGraph => {
            let named = inputs
                .iter()
                .enumerate()
                .map(|(index, input)| (format!("input{index}"), input.clone()))
                .collect::<Vec<_>>();
            let (graph, _) = Graph::capture(named.clone(), |inputs| run(inputs))?;
            let options = if path == ExecutionPath::TileGraph {
                GraphCudaOptions::default().with_kernel_language(KernelLanguage::TileIr)
            } else {
                GraphCudaOptions::default()
            };
            let mut compiled = cuda::compile_graph(&graph, &options)?;
            // Compilation drops inputs no output reads, such as `addmm`'s with a zero `beta`, and running rejects names a graph does not have.
            let declared = compiled
                .ir()
                .input_values()
                .map(|(_, name)| name.to_owned())
                .collect::<Vec<_>>();
            compiled.run(
                &named
                    .into_iter()
                    .filter(|(name, _)| declared.contains(name))
                    .collect(),
            )
        }
    }
}

/// Calls the composite tensor method `name` with the case's `arguments` on `inputs`.
fn composite(name: &str, arguments: &Value, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
    let integer = |key: &str| arguments[key].as_i64().unwrap();
    let integer_list = |key: &str| integers(&arguments[key]);
    let float = |key: &str| arguments[key].as_f64().unwrap();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the fixture writes these arguments as `F32` values"
    )]
    let single = |key: &str| arguments[key].as_f64().unwrap() as f32;
    let boolean = |key: &str| arguments[key].as_bool().unwrap();
    let several = match name {
        "unbind" => Some(inputs[0].unbind(integer("dim"))?),
        "split" => Some(inputs[0].split(integer("size"), integer("dim"))?),
        "chunk" => Some(inputs[0].chunk(integer("chunks"), integer("dim"))?),
        "grouped_topk_routing" => {
            let (ids, weights) = inputs[0].grouped_topk_routing(
                &inputs[1],
                integer("groups"),
                integer("kept_groups"),
                integer("top_k"),
                float("scale"),
            )?;
            Some(vec![ids, weights])
        }
        _ => None,
    };
    if let Some(outputs) = several {
        return Ok(outputs);
    }
    let result = match name {
        "flip" => inputs[0].flip(&integer_list("dims"))?,
        "gather" => inputs[0].gather(integer("dim"), &inputs[1])?,
        "scatter" => inputs[0].scatter(integer("dim"), &inputs[1], &inputs[2])?,
        "scatter_add" => inputs[0].scatter_add(integer("dim"), &inputs[1], &inputs[2])?,
        "bincount" => inputs[0].bincount(inputs.get(1), integer("bins"))?,
        "dequantize_blocks" => inputs[0].dequantize_blocks(
            &inputs[1],
            (integer("block_rows"), integer("block_columns")),
        )?,
        "any" => inputs[0].any(
            &integer_list("dims"),
            arguments["keepdim"].as_bool().unwrap(),
        )?,
        "all" => inputs[0].all(
            &integer_list("dims"),
            arguments["keepdim"].as_bool().unwrap(),
        )?,
        "roll" => inputs[0].roll(&integer_list("shifts"), &integer_list("dims"))?,
        "unfold" => inputs[0].unfold(integer("dim"), integer("size"), integer("step"))?,
        "diagonal" => inputs[0].diagonal(integer("offset"), integer("dim1"), integer("dim2"))?,
        "select" => inputs[0].select(integer("dim"), integer("index"))?,
        "nonzero" => inputs[0].nonzero()?,
        "masked_select" => inputs[0].masked_select(&inputs[1])?,
        "repeat" => inputs[0].repeat(&integers(&arguments["repeats"]))?,
        "addmm" => inputs[0].addmm(&inputs[1], &inputs[2], integer("beta"), integer("alpha"))?,
        "conv1d" => inputs[0].convolution1d(
            &inputs[1],
            inputs.get(2),
            integer("stride"),
            pair(&arguments["padding"]),
            integer("dilation"),
            integer("groups"),
        )?,
        "leaky_relu" => inputs[0].leaky_relu(float("negative_slope"))?,
        "elu" => inputs[0].elu(float("alpha"))?,
        "selu" => inputs[0].selu()?,
        "celu" => inputs[0].celu(float("alpha"))?,
        "softsign" => inputs[0].softsign()?,
        "hardsigmoid" => inputs[0].hardsigmoid()?,
        "hardswish" => inputs[0].hardswish()?,
        "hardtanh" => inputs[0].hardtanh(single("min"), single("max"))?,
        "softplus" => inputs[0].softplus(float("beta"))?,
        "mish" => inputs[0].mish()?,
        "logsumexp" => inputs[0].logsumexp(&integer_list("dims"), boolean("keepdim"))?,
        "var" => inputs[0].var(
            &integer_list("dims"),
            integer("correction"),
            boolean("keepdim"),
        )?,
        "std" => inputs[0].std(
            &integer_list("dims"),
            integer("correction"),
            boolean("keepdim"),
        )?,
        "vector_norm" => {
            let order = match &arguments["order"] {
                Value::String(order) if order == "inf" => NormOrder::Infinity,
                Value::String(order) if order == "-inf" => NormOrder::NegativeInfinity,
                order => NormOrder::Finite(order.as_f64().unwrap()),
            };
            inputs[0].vector_norm(order, &integer_list("dims"), boolean("keepdim"))?
        }
        "stack" => Tensor::stack(inputs, integer("dim"))?,
        "triu" => inputs[0].triu(integer("diagonal"))?,
        "tril" => inputs[0].tril(integer("diagonal"))?,
        "repeat_interleave" => inputs[0].repeat_interleave(integer("repeats"), integer("dim"))?,
        "movedim" => inputs[0].movedim(integer("source"), integer("destination"))?,
        "outer" => inputs[0].outer(&inputs[1])?,
        "masked_fill" => inputs[0].masked_fill(&inputs[1], integer("value"))?,
        "one_hot" => inputs[0].one_hot(integer("classes"))?,
        "sign" => inputs[0].sign()?,
        "nan_to_num" => inputs[0].nan_to_num(
            float("nan"),
            arguments["posinf"].as_f64(),
            arguments["neginf"].as_f64(),
        )?,
        "einsum" => Tensor::einsum(
            arguments["equation"].as_str().unwrap(),
            &inputs.iter().collect::<Vec<_>>(),
        )?,
        "group_norm" => inputs[0].group_norm(
            integer("groups"),
            inputs.get(1),
            inputs.get(2),
            single("eps"),
        )?,
        "instance_norm" => inputs[0].instance_norm(inputs.get(1), inputs.get(2), single("eps"))?,
        "max_pool1d" | "avg_pool1d" => {
            let padding = integers(&arguments["padding"]);
            if name == "max_pool1d" {
                inputs[0].max_pool1d(
                    integer("kernel"),
                    integer("stride"),
                    (padding[0], padding[1]),
                    1,
                    false,
                )?
            } else {
                inputs[0].avg_pool1d(
                    integer("kernel"),
                    integer("stride"),
                    (padding[0], padding[1]),
                    false,
                    true,
                )?
            }
        }
        "adaptive_avg_pool2d" => {
            let size = integers(&arguments["size"]);
            inputs[0].adaptive_avg_pool2d((size[0], size[1]))?
        }
        "adaptive_max_pool2d" => {
            let size = integers(&arguments["size"]);
            inputs[0].adaptive_max_pool2d((size[0], size[1]))?
        }
        "max_pool3d" | "avg_pool3d" => {
            let triple = |key: &str| {
                let values = integers(&arguments[key]);
                (values[0], values[1], values[2])
            };
            let ceil = boolean("ceil");
            if name == "max_pool3d" {
                inputs[0].max_pool3d(
                    triple("kernel"),
                    triple("stride"),
                    triple("padding"),
                    (1, 1, 1),
                    ceil,
                )?
            } else {
                inputs[0].avg_pool3d(
                    triple("kernel"),
                    triple("stride"),
                    triple("padding"),
                    ceil,
                    boolean("include_pad"),
                )?
            }
        }
        "conv3d" => {
            let triple = |key: &str| {
                let values = integers(&arguments[key]);
                (values[0], values[1], values[2])
            };
            inputs[0].convolution3d(
                &inputs[1],
                inputs.get(2),
                triple("stride"),
                triple("padding"),
                triple("dilation"),
                integer("groups"),
            )?
        }
        "conv_transpose2d" => inputs[0].convolution_transpose2d(
            &inputs[1],
            inputs.get(2),
            pair(&arguments["stride"]),
            pair(&arguments["padding"]),
            pair(&arguments["output_padding"]),
            pair(&arguments["dilation"]),
            integer("groups"),
        )?,
        "conv_transpose1d" => inputs[0].convolution_transpose1d(
            &inputs[1],
            inputs.get(2),
            integer("stride"),
            integer("padding"),
            integer("output_padding"),
            integer("dilation"),
            integer("groups"),
        )?,
        "rfft" => inputs[0].rfft()?,
        "irfft" => inputs[0].irfft(integer("length"))?,
        "interpolate" => {
            let size = integers(&arguments["size"]);
            let mode = match arguments["mode"].as_str().unwrap() {
                "nearest" => InterpolateMode::Nearest,
                _ => InterpolateMode::Bilinear,
            };
            inputs[0].interpolate((size[0], size[1]), mode)?
        }
        "cross_entropy" => {
            let reduction = match arguments["reduction"].as_str().unwrap() {
                "none" => LossReduction::None,
                "sum" => LossReduction::Sum,
                _ => LossReduction::Mean,
            };
            inputs[0].cross_entropy(&inputs[1], reduction)?
        }
        "cosine_similarity" => {
            inputs[0].cosine_similarity(&inputs[1], integer("dim"), single("eps"))?
        }
        other => panic!("unknown composite operation {other}"),
    };
    Ok(vec![result])
}

/// Returns `bytes` of `data_type` as F64 values, or `None` for data types compared bit for bit.
fn floats(data_type: DataType, bytes: &[u8]) -> Option<Vec<f64>> {
    let values = match data_type {
        DataType::F16 => bytes
            .as_chunks()
            .0
            .iter()
            .map(|&chunk| f16::from_le_bytes(chunk).to_f64())
            .collect(),
        DataType::Bf16 => bytes
            .as_chunks()
            .0
            .iter()
            .map(|&chunk| bf16::from_le_bytes(chunk).to_f64())
            .collect(),
        DataType::F32 => bytes
            .as_chunks()
            .0
            .iter()
            .map(|&chunk| f64::from(f32::from_le_bytes(chunk)))
            .collect(),
        DataType::F64 => bytes
            .as_chunks()
            .0
            .iter()
            .map(|&chunk| f64::from_le_bytes(chunk))
            .collect(),
        _ => return None,
    };
    Some(values)
}

// The relative and absolute tolerances of `data_type`, those of `torch.testing.assert_close`; as PyTorch's `OpInfo` overrides them, half-precision softmaxes, attention, and rotary embeddings, whose kernels round their intermediate values differently, allow a few more units in the last place.
fn tolerance(data_type: DataType, operation: &str) -> (f64, f64) {
    let rounds_intermediates = matches!(
        operation,
        "softmax"
            | "log_softmax"
            | "scaled_dot_product_attention"
            | "paged_attention"
            | "paged_mla_attention"
            | "rotary_embedding"
    );
    // Composites that sum long products in another order than PyTorch's kernels do.
    let reassociates = matches!(
        operation,
        "conv3d"
            | "conv_transpose1d"
            | "conv_transpose2d"
            | "rfft"
            | "irfft"
            | "grouped_matmul"
            | "paged_mla_attention"
    );
    match data_type {
        DataType::F32 if reassociates => (1e-5, 1e-4),
        DataType::F16 if rounds_intermediates => (1e-2, 1e-3),
        DataType::Bf16 if rounds_intermediates => (5e-2, 1e-2),
        DataType::F16 => (1e-3, 1e-5),
        DataType::Bf16 => (1.6e-2, 1e-5),
        DataType::F32 => (1.3e-6, 1e-5),
        _ => (1e-7, 1e-7),
    }
}

/// Compares Singe's outputs with `PyTorch`'s and describes the first difference, if any.
fn difference(actual: &[Tensor], expected: &[Encoded], operation: &str) -> Option<String> {
    if actual.len() != expected.len() {
        return Some(format!(
            "{} outputs, expected {}",
            actual.len(),
            expected.len()
        ));
    }
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let data_type = data_type(&expected.dtype);
        if actual.data_type() != data_type {
            return Some(format!(
                "output {index} is {:?}, expected {data_type:?}",
                actual.data_type()
            ));
        }
        if actual.shape().dimensions() != expected.shape {
            return Some(format!(
                "output {index} has shape {:?}, expected {:?}",
                actual.shape().dimensions(),
                expected.shape
            ));
        }
        let bytes = actual
            .to_device(Device::Cpu)
            .and_then(|tensor| tensor.contiguous())
            .and_then(|tensor| tensor.to_bytes());
        let bytes = match bytes {
            Ok(bytes) => bytes,
            Err(error) => return Some(format!("output {index} cannot be read: {error}")),
        };
        let expected_bytes = decode(&expected.bytes);
        let Some(values) = floats(data_type, &bytes) else {
            let same = bytes.len() == expected_bytes.len()
                && bytes
                    .iter()
                    .zip(&expected_bytes)
                    .all(|(&actual, &reference)| {
                        actual == reference
                            || (is_nan(data_type, actual) && is_nan(data_type, reference))
                    });
            if !same {
                return Some(format!(
                    "output {index} bytes {} differ from {}",
                    hex(&bytes),
                    expected.bytes
                ));
            }
            continue;
        };
        let expected_values = floats(data_type, &expected_bytes).unwrap();
        let (relative, absolute) = tolerance(data_type, operation);
        for (position, (&value, &reference)) in values.iter().zip(&expected_values).enumerate() {
            let close = value == reference
                || (value.is_nan() && reference.is_nan())
                || (value - reference).abs() <= relative.mul_add(reference.abs(), absolute);
            if !close {
                return Some(format!(
                    "output {index} element {position} is {value}, expected {reference}"
                ));
            }
        }
    }
    None
}

/// Returns whether `byte` is a NaN of the 8-bit floating `data_type`, whose sign GPU arithmetic does not keep.
fn is_nan(data_type: DataType, byte: u8) -> bool {
    match data_type {
        DataType::F8E4M3 => byte & 0x7f == 0x7f,
        DataType::F8E5M2 => byte & 0x7c == 0x7c && byte & 0x03 != 0,
        _ => false,
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut text, byte| {
        write!(text, "{byte:02x}").unwrap();
        text
    })
}

/// Runs a case on `path` and returns its outcome, with a description of anything but a pass.
///
/// A case both reject passes whatever the reasons, since Singe may decline a data type before it reaches the check `PyTorch` failed.
fn check(fixture: &Fixture, case: &Case, path: ExecutionPath) -> (Outcome, String) {
    let result = catch_unwind(AssertUnwindSafe(|| execute(fixture, case, path)));
    let Ok(result) = result else {
        return (Outcome::Panicked, "panicked".to_owned());
    };
    match (result, &case.outputs, &case.error) {
        (Ok(actual), Some(expected), _) => difference(&actual, expected, &case.operation)
            .map_or_else(
                || (Outcome::Pass, String::new()),
                |difference| (Outcome::Mismatch, difference),
            ),
        (Err(error), Some(_), _) => (Outcome::Rejected, error.to_string()),
        (Ok(_), None, Some(reason)) if reason.starts_with("NotImplementedError") => {
            (Outcome::Extension, reason.clone())
        }
        (Ok(_), None, Some(reason)) => (Outcome::Accepted, reason.clone()),
        (Err(_), None, _) => (Outcome::Pass, String::new()),
        (Ok(_), None, None) => panic!("{} has neither outputs nor an error", case.name),
    }
}

fn baseline_path(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/baselines/conformance")
        .join(format!("{path}.txt"))
}

/// Reads a path's baseline: each case that does not pass, by name.
fn read_baseline(path: &str) -> BTreeMap<String, Outcome> {
    fs::read_to_string(baseline_path(path))
        .unwrap_or_default()
        .lines()
        .map(|line| {
            let (outcome, case) = line.split_once(' ').unwrap();
            (case.to_owned(), Outcome::parse(outcome))
        })
        .collect()
}

/// Runs every case on `path`, checks the outcomes against its baseline, and returns them.
fn replay(fixture: &Fixture, path: ExecutionPath, update: bool) -> Vec<Outcome> {
    let mut outcomes = Vec::with_capacity(fixture.cases.len());
    let mut report = String::new();
    for case in &fixture.cases {
        let (outcome, detail) = check(fixture, case, path);
        if outcome != Outcome::Pass {
            writeln!(report, "{} {}: {detail}", outcome.name(), case.name).unwrap();
        }
        outcomes.push(outcome);
    }
    let report_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../target")
        .join(format!("conformance-{}.txt", path.name()));
    fs::write(&report_path, &report).unwrap();
    let lines = fixture
        .cases
        .iter()
        .zip(&outcomes)
        .filter(|(_, outcome)| **outcome != Outcome::Pass)
        .fold(String::new(), |mut lines, (case, outcome)| {
            writeln!(lines, "{} {}", outcome.name(), case.name).unwrap();
            lines
        });
    if update {
        fs::write(baseline_path(path.name()), lines).unwrap();
        return outcomes;
    }
    let baseline = read_baseline(path.name());
    let changes = fixture
        .cases
        .iter()
        .zip(&outcomes)
        .filter_map(|(case, &outcome)| {
            let expected = baseline.get(&case.name).copied().unwrap_or(Outcome::Pass);
            (outcome != expected).then(|| {
                format!(
                    "{}: {} (baseline {})",
                    case.name,
                    outcome.name(),
                    expected.name()
                )
            })
        })
        .collect::<Vec<_>>();
    assert!(
        changes.is_empty(),
        "{} {} outcomes changed; see {} and rerun with SINGE_UPDATE_CONFORMANCE=1 once they are intended:\n{}",
        changes.len(),
        path.name(),
        report_path.display(),
        changes.join("\n")
    );
    outcomes
}

#[test]
fn test_every_registry_operation_has_conformance_cases_or_a_reason() {
    let fixture = Fixture::load();
    let missing = registered_operations()
        .iter()
        .map(|operation| operation.name())
        .filter(|name| {
            !UNCOVERED.iter().any(|(operation, _)| operation == name)
                && !fixture.cases.iter().any(|case| case.operation == *name)
        })
        .collect::<Vec<_>>();
    assert!(
        missing.is_empty(),
        "operations without conformance cases: {missing:?}"
    );
}

#[test]
fn test_operations_match_pytorch_on_every_path() {
    let fixture = Fixture::load();
    let update = env::var_os("SINGE_UPDATE_CONFORMANCE").is_some();
    #[cfg_attr(
        not(all(feature = "testing", feature = "cuda")),
        expect(unused_mut, reason = "only CUDA builds replay more paths")
    )]
    let mut summaries = vec![(
        ExecutionPath::Cpu,
        replay(&fixture, ExecutionPath::Cpu, update),
    )];
    #[cfg(all(feature = "testing", feature = "cuda"))]
    {
        let _lock = device_lock(0).unwrap();
        for path in [ExecutionPath::Cuda, ExecutionPath::Graph] {
            summaries.push((path, replay(&fixture, path, update)));
        }
    }
    for (path, outcomes) in summaries {
        let counts = Outcome::ALL
            .iter()
            .map(|outcome| {
                let count = outcomes.iter().filter(|other| *other == outcome).count();
                format!("{} {count}", outcome.name())
            })
            .collect::<Vec<_>>();
        eprintln!("conformance {}: {}", path.name(), counts.join(", "));
    }
}

#[test]
fn test_meta_results_describe_the_cpu_results() {
    let fixture = Fixture::load();
    let mut failures = Vec::new();
    // A composite may read its operands' values, which meta tensors do not have.
    for case in fixture.cases.iter().filter(|case| case.composite.is_none()) {
        let Ok(expected) = execute(&fixture, case, ExecutionPath::Cpu) else {
            continue;
        };
        let actual = match execute(&fixture, case, ExecutionPath::Meta) {
            Ok(actual) => actual,
            Err(error) => {
                failures.push(format!("{}: {error}", case.name));
                continue;
            }
        };
        let described = actual.len() == expected.len()
            && actual.iter().zip(&expected).all(|(actual, expected)| {
                actual.data_type() == expected.data_type() && actual.shape() == expected.shape()
            });
        if !described {
            failures.push(format!(
                "{}: metadata differs from the cpu result",
                case.name
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} meta results differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn test_meta_views_match_cpu_layouts_and_share_their_input_storage() {
    type View = fn(&Tensor) -> Result<Tensor>;
    let views: [(&str, View); 10] = [
        ("transpose", |tensor| tensor.transpose(0, 2)),
        ("permute", |tensor| tensor.permute(&[2, 0, 1])),
        ("narrow", |tensor| tensor.narrow(2, 1, 2)),
        ("reshape", |tensor| tensor.reshape([6, 4])),
        ("view", |tensor| tensor.view(&Shape::create([2, 12])?)),
        ("flatten", |tensor| tensor.flatten(0, 1)),
        ("expand", |tensor| {
            tensor.narrow(0, 0, 1)?.expand(&Shape::create([5, 3, 4])?)
        }),
        ("unsqueeze", |tensor| {
            call_one(
                Operation::Unsqueeze,
                OperationAttributes::Dimension(1),
                &[tensor],
            )
        }),
        ("squeeze", |tensor| {
            call_one(
                Operation::Squeeze,
                OperationAttributes::Dimension(0),
                &[&tensor.narrow(0, 0, 1)?],
            )
        }),
        ("slice", |tensor| {
            call_one(
                Operation::Slice,
                OperationAttributes::Slice(SliceAttributes {
                    dim: 1,
                    start: 1,
                    length: 2,
                    step: 1,
                }),
                &[tensor],
            )
        }),
    ];
    let values = (0..24_u8).map(f32::from).collect::<Vec<_>>();
    let input = Tensor::from_vec(Device::Cpu, Shape::create([2, 3, 4]).unwrap(), values).unwrap();
    let strided = |tensor: &Tensor| {
        tensor
            .transpose(0, 1)
            .unwrap()
            .contiguous()
            .unwrap()
            .transpose(0, 1)
            .unwrap()
    };
    for (name, view) in views {
        for transposed in [false, true] {
            let meta = input.to_device(Device::Meta).unwrap();
            let (cpu, meta) = if transposed {
                (strided(&input), strided(&meta))
            } else {
                (input.clone(), meta)
            };
            let context = format!("{name} of a strided={transposed} input");
            let (expected, actual) = match (view(&cpu), view(&meta)) {
                (Ok(expected), Ok(actual)) => (expected, actual),
                (Err(_), Err(_)) => continue,
                (expected, actual) => panic!("{context}: cpu {expected:?} but meta {actual:?}"),
            };
            assert_eq!(actual.layout(), expected.layout(), "{context}: layout");
            assert_eq!(
                Arc::ptr_eq(actual.storage(), meta.storage()),
                Arc::ptr_eq(expected.storage(), cpu.storage()),
                "{context}: shares its input's storage"
            );
        }
    }
}

// Tile IR compiles slowly, so one sample of each operation and data type runs through it in a compiled graph, and each must end as the graph baseline records it.
#[cfg(all(feature = "testing", feature = "cuda"))]
#[test]
fn test_cuda_tile_kernels_match_the_cuda_baseline() {
    let fixture = Fixture::load();
    let baseline = read_baseline(ExecutionPath::Graph.name());
    let _lock = device_lock(0).unwrap();
    let changes = fixture
        .cases
        .iter()
        .filter(|case| case.name.ends_with(" matrix"))
        .filter_map(|case| {
            let (outcome, detail) = check(&fixture, case, ExecutionPath::TileGraph);
            let expected = baseline.get(&case.name).copied().unwrap_or(Outcome::Pass);
            (outcome != expected).then(|| format!("{}: {} {detail}", case.name, outcome.name()))
        })
        .collect::<Vec<_>>();
    assert!(
        changes.is_empty(),
        "{} tile outcomes differ from the graph baseline:\n{}",
        changes.len(),
        changes.join("\n")
    );
}

// Runs `case` as a program, and returns whether it ran; only a program that must call a library can decline.
#[cfg(all(feature = "testing", feature = "cuda"))]
fn check_program(
    fixture: &Fixture,
    case: &Case,
    export: &CudaExport,
    label: &str,
    with_library_only: bool,
) -> anyhow::Result<bool> {
    let device = ExecutionPath::Graph.device();
    let named = case
        .inputs
        .iter()
        .enumerate()
        .map(|(index, input)| {
            operand(fixture, input, device).map(|tensor| (format!("input{index}"), tensor))
        })
        .collect::<Result<Vec<_>>>()?;
    let (graph, _) = Graph::capture(named.clone(), |inputs| run_case(case, inputs))?;
    let program: CudaProgram = export.export(&graph)?;
    if with_library_only
        && !program
            .steps
            .iter()
            .any(|step| matches!(step, ProgramStep::Library { .. }))
    {
        return Ok(false);
    }
    let name = case
        .name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect::<String>();
    let compiled = CompiledProgram::create(&program, &format!("{label}-{name}"))?;
    let inputs = program
        .inputs
        .iter()
        .map(|input| {
            named
                .iter()
                .find(|(name, _)| *name == input.name)
                .map(|(_, tensor)| tensor.clone())
                .ok_or_else(|| anyhow!("no input named {}", input.name))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    // A second run reuses everything the first set up.
    let actual = compiled.run(&inputs, 2)?;
    let expected = case
        .outputs
        .as_ref()
        .ok_or_else(|| anyhow!("the case has no outputs"))?;
    difference(&actual, expected, &case.operation)
        .map_or(Ok(true), |difference| Err(anyhow!(difference)))
}

// One case per operation, since compiling every case would take too long.
#[cfg(all(feature = "testing", feature = "cuda"))]
#[test]
fn test_exported_programs_match_pytorch_on_a_sample_of_cases() {
    let fixture = Fixture::load();
    let _lock = device_lock(0).unwrap();
    let Some(architecture) = program_target(ExecutionPath::Graph.device()) else {
        return;
    };
    let baseline = read_baseline(ExecutionPath::Graph.name());
    let mut passing = BTreeMap::<&str, Vec<&Case>>::new();
    for case in &fixture.cases {
        let expected = baseline.get(&case.name).copied().unwrap_or(Outcome::Pass);
        if case.outputs.is_some() && expected == Outcome::Pass {
            passing.entry(&case.operation).or_default().push(case);
        }
    }
    let generated = CudaExport::new(architecture).without_libraries();
    // Each library alone takes everything it can, and all together choose among themselves.
    let mut library_exports = vec![("libraries".to_owned(), CudaExport::new(architecture))];
    library_exports.extend(Library::ALL.map(|library| {
        (
            library.name().to_owned(),
            CudaExport::new(architecture).with_libraries([library]),
        )
    }));
    let (mut ran, mut failures) = (0, Vec::new());
    for cases in passing.values() {
        let case = cases[cases.len() / 2];
        let exports = library_exports
            .iter()
            .map(|(label, export)| (label.as_str(), export, true));
        for (label, export, library_only) in once(("generated", &generated, false)).chain(exports) {
            let result = catch_unwind(AssertUnwindSafe(|| {
                check_program(&fixture, case, export, label, library_only)
            }));
            match result {
                Ok(Ok(true)) => ran += 1,
                Ok(Ok(false)) => {}
                Ok(Err(reason)) => {
                    failures.push(format!("{label} {}: {reason:?}", case.name));
                }
                Err(_) => failures.push(format!("{label} {}: panicked", case.name)),
            }
        }
    }
    println!("{ran} programs ran");
    assert!(
        failures.is_empty(),
        "{} of the programs failed:\n{}",
        failures.len(),
        failures.join("\n\n")
    );
    assert!(ran >= 100, "only {ran} programs ran");
}
