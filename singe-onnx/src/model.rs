use std::{collections::BTreeMap, fs, mem::take, path::Path};

use bomboni_request::error::{CommonError, PathErrorStep, RequestError, RequestResult};
use num_enum::{IntoPrimitive, TryFromPrimitive};
use prost::Message;

use crate::{
    error::{Error, FormatError, Result},
    proto::{onnx, onnx_structure},
};

/// Metadata key-value pairs from ONNX `StringStringEntryProto` lists.
///
/// ONNX represents maps as repeated key-value entries for protobuf compatibility.
/// This parser normalizes those entries into a sorted map and rejects duplicate keys.
pub type PropertyMap = BTreeMap<String, String>;

/// Top-level ONNX model container.
///
/// A model bundles one parameterized computation graph with metadata, imported operator sets, optional training information, and model-local functions.
#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    /// ONNX IR version targeted by this model.
    pub ir_version: i64,
    /// Operator sets this model relies on.
    ///
    /// Nodes bind to the same-domain, same-operator entry with the highest imported version.
    pub operator_set_imports: Vec<OperatorSetId>,
    /// Framework or tool that generated the model.
    pub producer_name: String,
    /// Version of the producer framework or tool.
    pub producer_version: String,
    /// Reverse-DNS model domain, when specified.
    ///
    /// Together with `model_version` and `graph.name`, this identifies the graph.
    pub domain: Option<String>,
    /// Version of the graph encoded in the model.
    pub model_version: i64,
    /// Human-readable model documentation.
    /// Markdown is allowed by ONNX.
    pub doc_string: Option<String>,
    /// Parameterized graph evaluated to execute the model.
    pub graph: Graph,
    /// Named model metadata values.
    pub metadata_props: PropertyMap,
    /// Training-specific initialization and update steps.
    pub training_info: Vec<TrainingInfo>,
    /// Function definitions local to this model.
    pub functions: Vec<Function>,
}

/// Identifier for an imported ONNX operator set.
///
/// Operator sets are uniquely identified by `(domain, version)`.
/// The empty domain denotes the standard ONNX operator set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperatorSetId {
    /// Operator set domain, or `None` for the standard ONNX operator set.
    pub domain: Option<String>,
    /// Operator set version.
    pub version: i64,
}

/// ONNX computation graph.
///
/// A graph contains a topologically sorted DAG of nodes, constant initializers, graph inputs and outputs, value type information, and graph-level metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct Graph {
    /// Nodes in topological order.
    pub nodes: Vec<Node>,
    /// Graph name.
    pub name: String,
    /// Named tensor constants used as graph inputs.
    ///
    /// ONNX requires every initializer to have a unique name across dense and sparse initializers.
    /// The name may also appear in the graph input list.
    pub initializers: Vec<Tensor>,
    /// Sparse-format tensor constants used as graph inputs.
    pub sparse_initializers: Vec<SparseTensor>,
    /// Human-readable graph documentation.
    /// Markdown is allowed by ONNX.
    pub doc_string: Option<String>,
    /// Graph inputs.
    pub inputs: Vec<ValueInfo>,
    /// Graph outputs.
    pub outputs: Vec<ValueInfo>,
    /// Optional type and shape information for intermediate graph values.
    pub value_info: Vec<ValueInfo>,
    /// Quantization parameter annotations for graph tensors.
    pub quantization_annotations: Vec<TensorAnnotation>,
    /// Named graph metadata values.
    pub metadata_props: PropertyMap,
}

/// ONNX graph node.
///
/// Nodes represent computation stages, such as a `Conv` operator that consumes image, filter, and bias tensors and produces a convolved output.
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    /// Input value names.
    pub inputs: Vec<String>,
    /// Output value names.
    pub outputs: Vec<String>,
    /// Optional graph-local node identifier.
    pub name: String,
    /// Symbolic operator identifier to execute.
    pub operator_type: String,
    /// Domain of the operator set containing `operator_type`.
    pub domain: Option<String>,
    /// Function overload identifier for model-local function resolution.
    pub overload: Option<String>,
    /// Additional named attributes for the node.
    pub attributes: Vec<Attribute>,
    /// Human-readable node documentation.
    /// Markdown is allowed by ONNX.
    pub doc_string: Option<String>,
    /// Named node metadata values.
    pub metadata_props: PropertyMap,
}

/// Named ONNX attribute.
///
/// Attributes are typed values attached to nodes or functions.
/// ONNX requires exactly one content field matching the declared attribute type.
#[derive(Debug, Clone, PartialEq)]
pub struct Attribute {
    /// Attribute name.
    pub name: String,
    /// Discriminator describing which attribute value is present.
    pub attribute_type: AttributeType,
    /// Human-readable attribute documentation.
    /// Markdown is allowed by ONNX.
    pub doc_string: Option<String>,
    /// Parsed attribute value.
    pub value: AttributeValue,
}

/// Value carried by an [`Attribute`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum AttributeValue {
    /// Singular floating-point value.
    Float(f32),
    /// Singular integer value.
    Int(i64),
    /// Singular UTF-8 string value.
    String(String),
    /// Singular dense tensor value.
    Tensor(Tensor),
    /// Singular graph value.
    Graph(Graph),
    /// Singular sparse tensor value.
    SparseTensor(SparseTensor),
    /// Singular type descriptor value.
    Type(Type),
    /// Repeated floating-point values.
    Floats(Vec<f32>),
    /// Repeated integer values.
    Ints(Vec<i64>),
    /// Repeated UTF-8 string values.
    Strings(Vec<String>),
    /// Repeated dense tensor values.
    Tensors(Vec<Tensor>),
    /// Repeated graph values.
    Graphs(Vec<Graph>),
    /// Repeated sparse tensor values.
    SparseTensors(Vec<SparseTensor>),
    /// Repeated type descriptor values.
    Types(Vec<Type>),
    /// Reference to the named attribute of the enclosing function, which supplies the value at each call site.
    ///
    /// ONNX allows references only inside function bodies; the referenced attribute has the type in [`Attribute::attribute_type`].
    Reference(String),
}

/// Serialized ONNX tensor value.
///
/// Tensor elements are stored in row-major order.
/// Data may be embedded in the protobuf message or stored externally through [`Tensor::external_data`].
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    /// Tensor shape.
    pub dimensions: Vec<i64>,
    /// Element data type.
    pub data_type: TensorDataType,
    /// Segment represented by this tensor when a large tensor is split into chunks.
    pub segment: Option<TensorSegment>,
    /// Optional tensor name in the ONNX value namespace.
    pub name: String,
    /// Human-readable tensor documentation.
    /// Markdown is allowed by ONNX.
    pub doc_string: Option<String>,
    /// Whether tensor content is embedded in the protobuf or stored externally.
    pub data_location: TensorDataLocation,
    /// Parsed inline tensor data, if present and not external.
    pub data: Option<TensorElements>,
    /// External data location metadata.
    ///
    /// Recognized ONNX keys include `location`, `offset`, `length`, and `checksum`.
    /// `location` is relative to the model file directory.
    pub external_data: PropertyMap,
    /// Named tensor metadata values.
    pub metadata_props: PropertyMap,
}

/// Inline tensor storage.
///
/// ONNX may store tensor elements either in type-specific repeated fields or in a raw little-endian byte buffer.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum TensorElements {
    /// `float_data`: `FLOAT` values or interleaved real/imaginary `COMPLEX64` values.
    Float(Vec<f32>),
    /// `int32_data`: integer, boolean, half, bfloat16, float8, float4, 4-bit, and 2-bit values.
    Int32(Vec<i32>),
    /// `string_data`: UTF-8 strings.
    String(Vec<String>),
    /// `int64_data`: `INT64` values.
    Int64(Vec<i64>),
    /// `double_data`: `DOUBLE` values or interleaved real/imaginary `COMPLEX128` values.
    Double(Vec<f64>),
    /// `uint64_data`: `UINT32` and `UINT64` values.
    Uint64(Vec<u64>),
    /// `raw_data`: fixed-width little-endian tensor bytes.
    ///
    /// ONNX does not allow raw storage for `STRING` or `UNDEFINED` tensors.
    Raw(Vec<u8>),
}

/// Chunk range for a segmented tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorSegment {
    /// First stored element in the original tensor.
    pub begin: i64,
    /// One-past-last stored element in the original tensor.
    pub end: i64,
}

/// Serialized ONNX sparse tensor value.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseTensor {
    /// Non-default values, encoded as a tensor of shape `[NNZ]`.
    ///
    /// When used as a graph sparse initializer, ONNX requires this tensor to have a non-empty name serving as the sparse tensor name.
    pub values: Option<Tensor>,
    /// Indices of non-default values.
    ///
    /// Indices may be `[NNZ, rank]` index tuples or `[NNZ]` linearized indices, and must be sorted without duplicates.
    pub indices: Option<Tensor>,
    /// Dense shape represented by the sparse tensor.
    pub dimensions: Vec<i64>,
}

/// Name, type, shape, documentation, and metadata for an ONNX value.
#[derive(Debug, Clone, PartialEq)]
pub struct ValueInfo {
    /// Value name.
    pub name: String,
    /// Optional value type and shape descriptor.
    pub value_type: Option<Type>,
    /// Human-readable value documentation.
    /// Markdown is allowed by ONNX.
    pub doc_string: Option<String>,
    /// Named value metadata values.
    pub metadata_props: PropertyMap,
}

/// ONNX type descriptor.
#[derive(Debug, Clone, PartialEq)]
pub struct Type {
    /// Optional semantic denotation for the whole type.
    ///
    /// ONNX defines standard type denotations in its `TypeDenotation` document.
    pub denotation: Option<String>,
    /// Concrete kind of type.
    pub value: TypeValue,
}

/// Concrete ONNX type kind.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum TypeValue {
    /// Tensor type.
    Tensor(TensorType),
    /// Sequence type.
    Sequence(Box<TypeSequence>),
    /// Map type.
    Map(Box<TypeMap>),
    /// Optional wrapper type.
    Optional(Box<TypeOptional>),
    /// Sparse tensor type.
    SparseTensor(SparseTensorType),
    /// Opaque type named by a domain and a name.
    Opaque(OpaqueType),
}

/// Dense tensor type and optional shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorType {
    /// Tensor element type.
    /// ONNX requires this to be a valid non-undefined tensor data type.
    pub element_type: TensorDataType,
    /// Optional tensor shape.
    pub shape: Option<TensorShape>,
}

/// Sequence type.
#[derive(Debug, Clone, PartialEq)]
pub struct TypeSequence {
    /// Type and optional shape of each sequence element.
    pub element_type: Type,
}

/// Map type.
#[derive(Debug, Clone, PartialEq)]
pub struct TypeMap {
    /// Map key type.
    ///
    /// ONNX requires this to be an integral tensor data type or `STRING`.
    pub key_type: TensorDataType,
    /// Map value type.
    pub value_type: Type,
}

/// Optional wrapper type.
#[derive(Debug, Clone, PartialEq)]
pub struct TypeOptional {
    /// Type and optional shape of the wrapped element.
    pub element_type: Type,
}

/// Sparse tensor type and optional shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseTensorType {
    /// Sparse tensor element type.
    /// ONNX requires this to be a valid non-undefined tensor data type.
    pub element_type: TensorDataType,
    /// Optional dense shape represented by the sparse tensor.
    pub shape: Option<TensorShape>,
}

/// Opaque type, which ONNX identifies only by name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpaqueType {
    /// Domain of the type's name; empty means the model's domain.
    pub domain: String,
    /// Name of the type within its domain.
    pub name: String,
}

/// ONNX tensor shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorShape {
    /// Shape dimensions.
    pub dimensions: Vec<Dimension>,
}

/// One tensor dimension.
///
/// A dimension may be a concrete integer value, a symbolic parameter, or absent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dimension {
    /// Optional standard semantic denotation for this axis.
    pub denotation: Option<String>,
    /// Concrete or symbolic dimension value.
    pub value: Option<DimensionValue>,
}

/// Concrete or symbolic dimension value.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DimensionValue {
    /// Integer dimension size.
    Value(i64),
    /// Symbolic dimension parameter name.
    Parameter(String),
}

/// Quantization annotation for a tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorAnnotation {
    /// Annotated tensor name.
    pub tensor_name: String,
    /// Mapping from predefined quantization parameter keys to tensor names.
    ///
    /// For 8-bit linear quantization, ONNX predefines keys such as `SCALE_TENSOR` and `ZERO_POINT_TENSOR`.
    pub quant_parameter_tensor_names: PropertyMap,
}

/// ONNX training metadata.
///
/// A training info entry describes an optional initialization graph and one training-algorithm step.
/// Executing the algorithm and applying `update_binding` performs one training update step.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingInfo {
    /// Graph used to compute initial tensors before training starts.
    pub initialization: Option<Graph>,
    /// Graph representing a training algorithm step.
    pub algorithm: Option<Graph>,
    /// Bindings from initialization graph outputs to mutable initializers.
    pub initialization_binding: PropertyMap,
    /// Bindings from algorithm outputs to mutable initializers.
    pub update_binding: PropertyMap,
}

/// Model-local ONNX function definition.
///
/// A function is uniquely identified inside a model by `(domain, name, overload)`.
/// Functions may reference other local functions, but recursive references are not allowed.
#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    /// Function name, analogous to a node `operator_type`.
    pub name: String,
    /// Function input names.
    pub inputs: Vec<String>,
    /// Function output names.
    pub outputs: Vec<String>,
    /// Attribute parameter names without default values.
    pub attributes: Vec<String>,
    /// Attribute parameters with default values.
    pub attribute_protos: Vec<Attribute>,
    /// Nodes in the function body.
    pub nodes: Vec<Node>,
    /// Human-readable function documentation.
    /// Markdown is allowed by ONNX.
    pub doc_string: Option<String>,
    /// Operator sets relied on by the function body.
    pub operator_set_imports: Vec<OperatorSetId>,
    /// Function domain.
    pub domain: String,
    /// Function overload identifier.
    pub overload: String,
    /// Optional type information for values used by the function.
    pub value_info: Vec<ValueInfo>,
    /// Named function metadata values.
    pub metadata_props: PropertyMap,
}

/// ONNX attribute type discriminator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, TryFromPrimitive, IntoPrimitive)]
#[repr(i32)]
#[non_exhaustive]
pub enum AttributeType {
    /// Undefined attribute type.
    Undefined = 0,
    /// Singular `f32`.
    Float = 1,
    /// Singular `i64`.
    Int = 2,
    /// Singular UTF-8 string.
    String = 3,
    /// Singular dense tensor.
    Tensor = 4,
    /// Singular graph.
    Graph = 5,
    /// Repeated `f32` values.
    Floats = 6,
    /// Repeated `i64` values.
    Ints = 7,
    /// Repeated UTF-8 strings.
    Strings = 8,
    /// Repeated dense tensors.
    Tensors = 9,
    /// Repeated graphs.
    Graphs = 10,
    /// Singular sparse tensor.
    SparseTensor = 11,
    /// Repeated sparse tensors.
    SparseTensors = 12,
    /// Singular type descriptor.
    Type = 13,
    /// Repeated type descriptors.
    Types = 14,
}

/// ONNX tensor element data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, TryFromPrimitive, IntoPrimitive)]
#[repr(i32)]
#[non_exhaustive]
pub enum TensorDataType {
    /// Undefined tensor data type.
    Undefined = 0,
    /// 32-bit IEEE floating point.
    Float = 1,
    /// 8-bit unsigned integer.
    Uint8 = 2,
    /// 8-bit signed integer.
    Int8 = 3,
    /// 16-bit unsigned integer.
    Uint16 = 4,
    /// 16-bit signed integer.
    Int16 = 5,
    /// 32-bit signed integer.
    Int32 = 6,
    /// 64-bit signed integer.
    Int64 = 7,
    /// UTF-8 string.
    String = 8,
    /// Boolean.
    Bool = 9,
    /// IEEE 754 half-precision floating point.
    Float16 = 10,
    /// 64-bit IEEE floating point.
    Double = 11,
    /// 32-bit unsigned integer.
    Uint32 = 12,
    /// 64-bit unsigned integer.
    Uint64 = 13,
    /// Complex number with `f32` real and imaginary components.
    Complex64 = 14,
    /// Complex number with `f64` real and imaginary components.
    Complex128 = 15,
    /// 16-bit bfloat floating point.
    Bfloat16 = 16,
    /// E4M3 8-bit floating point, finite values plus NaN, no infinity.
    Float8e4m3fn = 17,
    /// E4M3 8-bit floating point, finite values plus NaN, no infinity, no negative zero.
    Float8e4m3fnuz = 18,
    /// E5M2 8-bit floating point, supporting NaN and infinity.
    Float8e5m2 = 19,
    /// E5M2 8-bit floating point, supporting NaN, no infinity, no negative zero.
    Float8e5m2fnuz = 20,
    /// 4-bit unsigned integer in range `[0, 15]`.
    Uint4 = 21,
    /// 4-bit signed integer in range `[-8, 7]` using two's-complement representation.
    Int4 = 22,
    /// 4-bit E2M1 floating point.
    Float4e2m1 = 23,
    /// E8M0 8-bit scale type used by OCP microscaling formats.
    Float8e8m0 = 24,
    /// 2-bit unsigned integer in range `[0, 3]`.
    Uint2 = 25,
    /// 2-bit signed integer in range `[-2, 1]` using two's-complement representation.
    Int2 = 26,
    /// 6-bit E2M3 floating point, finite values only.
    Float6e2m3 = 27,
    /// 6-bit E3M2 floating point, finite values only.
    Float6e3m2 = 28,
}

/// ONNX tensor data storage location.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, TryFromPrimitive, IntoPrimitive)]
#[repr(i32)]
#[non_exhaustive]
pub enum TensorDataLocation {
    /// Data is embedded in `raw_data` or a type-specific repeated field.
    #[default]
    Default = 0,
    /// Data is stored externally and described by `external_data`.
    External = 1,
}

// The conversions use a private trait, so the public API offers only `TryFrom` and keeps `bomboni_request` out of it.
trait FromProto<T>: Sized {
    fn from_proto(value: T) -> RequestResult<Self>;
}

trait ProtoInto<U> {
    fn parse_into(self) -> RequestResult<U>;
}

impl<T, U> ProtoInto<U> for T
where
    U: FromProto<T>,
{
    fn parse_into(self) -> RequestResult<U> {
        U::from_proto(self)
    }
}

impl Model {
    /// Decodes and parses an ONNX model from protobuf bytes, including inline tensor data.
    ///
    /// # Errors
    ///
    /// Returns an error if the bytes are not a valid ONNX protobuf message, or the decoded model violates the typed ONNX format requirements.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::try_from(onnx::ModelProto::decode(bytes)?)
    }

    /// Reads and parses an ONNX model file, including inline tensor data.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read, the bytes are not a valid ONNX protobuf message, or the decoded model violates the typed ONNX format requirements.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_bytes(&fs::read(path)?)
    }

    /// Decodes and parses an ONNX model from protobuf bytes, skipping inline tensor payloads.
    ///
    /// The structure-only schema leaves out the payload fields, so `prost` skips them as unknown fields instead of allocating weight buffers, and each initializer's `data` is `None`.
    ///
    /// # Errors
    ///
    /// Returns an error if the bytes are not a valid ONNX protobuf message, or the decoded model violates the typed ONNX format requirements.
    pub fn structure_from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::try_from(onnx_structure::ModelProto::decode(bytes)?)
    }
}

impl TryFrom<onnx::ModelProto> for Model {
    type Error = Error;

    fn try_from(value: onnx::ModelProto) -> Result<Self> {
        Self::from_proto(value).map_err(|error| Error::Format(FormatError::new(error)))
    }
}

impl TryFrom<onnx_structure::ModelProto> for Model {
    type Error = Error;

    fn try_from(value: onnx_structure::ModelProto) -> Result<Self> {
        Self::from_proto(value).map_err(|error| Error::Format(FormatError::new(error)))
    }
}

impl FromProto<onnx::ModelProto> for Model {
    fn from_proto(value: onnx::ModelProto) -> RequestResult<Self> {
        Ok(Self {
            ir_version: value.ir_version,
            operator_set_imports: parse_repeated(
                value.opset_import,
                onnx::ModelProto::OPSET_IMPORT_FIELD_NAME,
            )?,
            producer_name: value.producer_name,
            producer_version: value.producer_version,
            domain: non_empty(value.domain),
            model_version: value.model_version,
            doc_string: non_empty(value.doc_string),
            graph: parse_required(value.graph, onnx::ModelProto::GRAPH_FIELD_NAME)?,
            metadata_props: parse_property_map(value.metadata_props)
                .map_err(|error| error.wrap_field(onnx::ModelProto::METADATA_PROPS_FIELD_NAME))?,
            training_info: parse_repeated(
                value.training_info,
                onnx::ModelProto::TRAINING_INFO_FIELD_NAME,
            )?,
            functions: parse_repeated(value.functions, onnx::ModelProto::FUNCTIONS_FIELD_NAME)?,
        })
    }
}

impl FromProto<onnx_structure::ModelProto> for Model {
    fn from_proto(value: onnx_structure::ModelProto) -> RequestResult<Self> {
        onnx::ModelProto::from(value).parse_into()
    }
}

impl FromProto<onnx::OperatorSetIdProto> for OperatorSetId {
    fn from_proto(value: onnx::OperatorSetIdProto) -> RequestResult<Self> {
        Ok(Self {
            domain: non_empty(value.domain),
            version: value.version,
        })
    }
}

impl FromProto<onnx::GraphProto> for Graph {
    fn from_proto(value: onnx::GraphProto) -> RequestResult<Self> {
        Ok(Self {
            nodes: parse_repeated(value.node, onnx::GraphProto::NODE_FIELD_NAME)?,
            name: value.name,
            initializers: parse_repeated(
                value.initializer,
                onnx::GraphProto::INITIALIZER_FIELD_NAME,
            )?,
            sparse_initializers: parse_repeated(
                value.sparse_initializer,
                onnx::GraphProto::SPARSE_INITIALIZER_FIELD_NAME,
            )?,
            doc_string: non_empty(value.doc_string),
            inputs: parse_repeated(value.input, onnx::GraphProto::INPUT_FIELD_NAME)?,
            outputs: parse_repeated(value.output, onnx::GraphProto::OUTPUT_FIELD_NAME)?,
            value_info: parse_repeated(value.value_info, onnx::GraphProto::VALUE_INFO_FIELD_NAME)?,
            quantization_annotations: parse_repeated(
                value.quantization_annotation,
                onnx::GraphProto::QUANTIZATION_ANNOTATION_FIELD_NAME,
            )?,
            metadata_props: parse_property_map(value.metadata_props)
                .map_err(|error| error.wrap_field(onnx::GraphProto::METADATA_PROPS_FIELD_NAME))?,
        })
    }
}

impl FromProto<onnx::NodeProto> for Node {
    fn from_proto(value: onnx::NodeProto) -> RequestResult<Self> {
        Ok(Self {
            inputs: value.input,
            outputs: value.output,
            name: value.name,
            operator_type: value.op_type,
            domain: non_empty(value.domain),
            overload: non_empty(value.overload),
            attributes: parse_repeated(value.attribute, onnx::NodeProto::ATTRIBUTE_FIELD_NAME)?,
            doc_string: non_empty(value.doc_string),
            metadata_props: parse_property_map(value.metadata_props)
                .map_err(|error| error.wrap_field(onnx::NodeProto::METADATA_PROPS_FIELD_NAME))?,
        })
    }
}

impl FromProto<onnx::AttributeProto> for Attribute {
    fn from_proto(value: onnx::AttributeProto) -> RequestResult<Self> {
        if value.name.is_empty() {
            return Err(RequestError::field(
                onnx::AttributeProto::NAME_FIELD_NAME,
                CommonError::RequiredFieldMissing,
            ));
        }

        let attribute_type = parse_enum(value.r#type, onnx::AttributeProto::TYPE_FIELD_NAME)?;
        let parsed_value = match attribute_type {
            AttributeType::Undefined => {
                return Err(RequestError::field(
                    onnx::AttributeProto::TYPE_FIELD_NAME,
                    CommonError::InvalidEnumValue,
                ));
            }
            _ if !value.ref_attr_name.is_empty() => AttributeValue::Reference(value.ref_attr_name),
            AttributeType::Float => AttributeValue::Float(value.f),
            AttributeType::Int => AttributeValue::Int(value.i),
            AttributeType::String => {
                AttributeValue::String(parse_string(value.s, onnx::AttributeProto::S_FIELD_NAME)?)
            }
            AttributeType::Tensor => {
                AttributeValue::Tensor(parse_required(value.t, onnx::AttributeProto::T_FIELD_NAME)?)
            }
            AttributeType::Graph => {
                AttributeValue::Graph(parse_required(value.g, onnx::AttributeProto::G_FIELD_NAME)?)
            }
            AttributeType::SparseTensor => AttributeValue::SparseTensor(parse_required(
                value.sparse_tensor,
                onnx::AttributeProto::SPARSE_TENSOR_FIELD_NAME,
            )?),
            AttributeType::Type => AttributeValue::Type(parse_required(
                value.tp,
                onnx::AttributeProto::TP_FIELD_NAME,
            )?),
            AttributeType::Floats => AttributeValue::Floats(value.floats),
            AttributeType::Ints => AttributeValue::Ints(value.ints),
            AttributeType::Strings => AttributeValue::Strings(parse_strings(
                value.strings,
                onnx::AttributeProto::STRINGS_FIELD_NAME,
            )?),
            AttributeType::Tensors => AttributeValue::Tensors(parse_repeated(
                value.tensors,
                onnx::AttributeProto::TENSORS_FIELD_NAME,
            )?),
            AttributeType::Graphs => AttributeValue::Graphs(parse_repeated(
                value.graphs,
                onnx::AttributeProto::GRAPHS_FIELD_NAME,
            )?),
            AttributeType::SparseTensors => AttributeValue::SparseTensors(parse_repeated(
                value.sparse_tensors,
                onnx::AttributeProto::SPARSE_TENSORS_FIELD_NAME,
            )?),
            AttributeType::Types => AttributeValue::Types(parse_repeated(
                value.type_protos,
                onnx::AttributeProto::TYPE_PROTOS_FIELD_NAME,
            )?),
        };

        Ok(Self {
            name: value.name,
            attribute_type,
            doc_string: non_empty(value.doc_string),
            value: parsed_value,
        })
    }
}

impl FromProto<onnx::TensorProto> for Tensor {
    fn from_proto(mut value: onnx::TensorProto) -> RequestResult<Self> {
        let data_type = parse_enum(value.data_type, onnx::TensorProto::DATA_TYPE_FIELD_NAME)?;
        if data_type == TensorDataType::Undefined {
            return Err(RequestError::field(
                onnx::TensorProto::DATA_TYPE_FIELD_NAME,
                CommonError::RequiredFieldMissing,
            ));
        }

        let data_location = parse_enum(
            value.data_location,
            onnx::TensorProto::DATA_LOCATION_FIELD_NAME,
        )?;
        let populated = [
            !value.raw_data.is_empty(),
            !value.float_data.is_empty(),
            !value.int32_data.is_empty(),
            !value.int64_data.is_empty(),
            !value.double_data.is_empty(),
            !value.uint64_data.is_empty(),
            !value.string_data.is_empty(),
        ]
        .into_iter()
        .filter(|present| *present)
        .count();
        if populated > 1 || data_location == TensorDataLocation::External && populated != 0 {
            return Err(RequestError::field(
                onnx::TensorProto::DATA_LOCATION_FIELD_NAME,
                CommonError::TypeMismatch,
            ));
        }

        let data = match data_location {
            TensorDataLocation::External => None,
            TensorDataLocation::Default => parse_tensor_data(&mut value, data_type)?,
        };
        if data.is_none() && populated != 0 {
            return Err(RequestError::field(
                onnx::TensorProto::DATA_TYPE_FIELD_NAME,
                CommonError::TypeMismatch,
            ));
        }

        Ok(Self {
            dimensions: value.dims,
            data_type,
            segment: parse_optional(value.segment, onnx::TensorProto::SEGMENT_FIELD_NAME)?,
            name: value.name,
            doc_string: non_empty(value.doc_string),
            data_location,
            data,
            external_data: parse_property_map(value.external_data)
                .map_err(|error| error.wrap_field(onnx::TensorProto::EXTERNAL_DATA_FIELD_NAME))?,
            metadata_props: parse_property_map(value.metadata_props)
                .map_err(|error| error.wrap_field(onnx::TensorProto::METADATA_PROPS_FIELD_NAME))?,
        })
    }
}

impl FromProto<onnx::tensor_proto::Segment> for TensorSegment {
    fn from_proto(value: onnx::tensor_proto::Segment) -> RequestResult<Self> {
        Ok(Self {
            begin: value.begin,
            end: value.end,
        })
    }
}

impl FromProto<onnx::SparseTensorProto> for SparseTensor {
    fn from_proto(value: onnx::SparseTensorProto) -> RequestResult<Self> {
        Ok(Self {
            values: parse_optional(value.values, onnx::SparseTensorProto::VALUES_FIELD_NAME)?,
            indices: parse_optional(value.indices, onnx::SparseTensorProto::INDICES_FIELD_NAME)?,
            dimensions: value.dims,
        })
    }
}

impl FromProto<onnx::ValueInfoProto> for ValueInfo {
    fn from_proto(value: onnx::ValueInfoProto) -> RequestResult<Self> {
        Ok(Self {
            name: value.name,
            value_type: parse_optional(value.r#type, onnx::ValueInfoProto::TYPE_FIELD_NAME)?,
            doc_string: non_empty(value.doc_string),
            metadata_props: parse_property_map(value.metadata_props).map_err(|error| {
                error.wrap_field(onnx::ValueInfoProto::METADATA_PROPS_FIELD_NAME)
            })?,
        })
    }
}

impl FromProto<onnx::TypeProto> for Type {
    fn from_proto(value: onnx::TypeProto) -> RequestResult<Self> {
        Ok(Self {
            denotation: non_empty(value.denotation),
            value: parse_required(value.value, onnx::TypeProto::VALUE_ONEOF_NAME)?,
        })
    }
}

impl FromProto<onnx::type_proto::Value> for TypeValue {
    fn from_proto(value: onnx::type_proto::Value) -> RequestResult<Self> {
        Ok(match value {
            onnx::type_proto::Value::TensorType(value) => Self::Tensor(value.parse_into()?),
            onnx::type_proto::Value::SequenceType(value) => {
                Self::Sequence(Box::new((*value).parse_into()?))
            }
            onnx::type_proto::Value::MapType(value) => Self::Map(Box::new((*value).parse_into()?)),
            onnx::type_proto::Value::OptionalType(value) => {
                Self::Optional(Box::new((*value).parse_into()?))
            }
            onnx::type_proto::Value::SparseTensorType(value) => {
                Self::SparseTensor(value.parse_into()?)
            }
            onnx::type_proto::Value::OpaqueType(value) => Self::Opaque(OpaqueType {
                domain: value.domain,
                name: value.name,
            }),
        })
    }
}

impl FromProto<onnx::type_proto::Tensor> for TensorType {
    fn from_proto(value: onnx::type_proto::Tensor) -> RequestResult<Self> {
        Ok(Self {
            element_type: parse_enum(value.elem_type, "elem_type")?,
            shape: parse_optional(value.shape, "shape")?,
        })
    }
}

impl FromProto<onnx::type_proto::Sequence> for TypeSequence {
    fn from_proto(value: onnx::type_proto::Sequence) -> RequestResult<Self> {
        Ok(Self {
            element_type: parse_required(value.elem_type.map(|value| *value), "elem_type")?,
        })
    }
}

impl FromProto<onnx::type_proto::Map> for TypeMap {
    fn from_proto(value: onnx::type_proto::Map) -> RequestResult<Self> {
        Ok(Self {
            key_type: parse_enum(value.key_type, "key_type")?,
            value_type: parse_required(value.value_type.map(|value| *value), "value_type")?,
        })
    }
}

impl FromProto<onnx::type_proto::Optional> for TypeOptional {
    fn from_proto(value: onnx::type_proto::Optional) -> RequestResult<Self> {
        Ok(Self {
            element_type: parse_required(value.elem_type.map(|value| *value), "elem_type")?,
        })
    }
}

impl FromProto<onnx::type_proto::SparseTensor> for SparseTensorType {
    fn from_proto(value: onnx::type_proto::SparseTensor) -> RequestResult<Self> {
        Ok(Self {
            element_type: parse_enum(value.elem_type, "elem_type")?,
            shape: parse_optional(value.shape, "shape")?,
        })
    }
}

impl FromProto<onnx::TensorShapeProto> for TensorShape {
    fn from_proto(value: onnx::TensorShapeProto) -> RequestResult<Self> {
        Ok(Self {
            dimensions: parse_repeated(value.dim, onnx::TensorShapeProto::DIM_FIELD_NAME)?,
        })
    }
}

impl FromProto<onnx::tensor_shape_proto::Dimension> for Dimension {
    fn from_proto(value: onnx::tensor_shape_proto::Dimension) -> RequestResult<Self> {
        Ok(Self {
            denotation: non_empty(value.denotation),
            value: parse_optional(
                value.value,
                onnx::tensor_shape_proto::Dimension::VALUE_ONEOF_NAME,
            )?,
        })
    }
}

impl FromProto<onnx::tensor_shape_proto::dimension::Value> for DimensionValue {
    fn from_proto(value: onnx::tensor_shape_proto::dimension::Value) -> RequestResult<Self> {
        Ok(match value {
            onnx::tensor_shape_proto::dimension::Value::DimValue(value) => Self::Value(value),
            onnx::tensor_shape_proto::dimension::Value::DimParam(value) => Self::Parameter(value),
        })
    }
}

impl FromProto<onnx::TensorAnnotation> for TensorAnnotation {
    fn from_proto(value: onnx::TensorAnnotation) -> RequestResult<Self> {
        Ok(Self {
            tensor_name: value.tensor_name,
            quant_parameter_tensor_names: parse_property_map(value.quant_parameter_tensor_names)
                .map_err(|error| {
                    error
                        .wrap_field(onnx::TensorAnnotation::QUANT_PARAMETER_TENSOR_NAMES_FIELD_NAME)
                })?,
        })
    }
}

impl FromProto<onnx::TrainingInfoProto> for TrainingInfo {
    fn from_proto(value: onnx::TrainingInfoProto) -> RequestResult<Self> {
        Ok(Self {
            initialization: parse_optional(
                value.initialization,
                onnx::TrainingInfoProto::INITIALIZATION_FIELD_NAME,
            )?,
            algorithm: parse_optional(
                value.algorithm,
                onnx::TrainingInfoProto::ALGORITHM_FIELD_NAME,
            )?,
            initialization_binding: parse_property_map(value.initialization_binding).map_err(
                |error| {
                    error.wrap_field(onnx::TrainingInfoProto::INITIALIZATION_BINDING_FIELD_NAME)
                },
            )?,
            update_binding: parse_property_map(value.update_binding).map_err(|error| {
                error.wrap_field(onnx::TrainingInfoProto::UPDATE_BINDING_FIELD_NAME)
            })?,
        })
    }
}

impl FromProto<onnx::FunctionProto> for Function {
    fn from_proto(value: onnx::FunctionProto) -> RequestResult<Self> {
        Ok(Self {
            name: value.name,
            inputs: value.input,
            outputs: value.output,
            attributes: value.attribute,
            attribute_protos: parse_repeated(
                value.attribute_proto,
                onnx::FunctionProto::ATTRIBUTE_PROTO_FIELD_NAME,
            )?,
            nodes: parse_repeated(value.node, onnx::FunctionProto::NODE_FIELD_NAME)?,
            doc_string: non_empty(value.doc_string),
            operator_set_imports: parse_repeated(
                value.opset_import,
                onnx::FunctionProto::OPSET_IMPORT_FIELD_NAME,
            )?,
            domain: value.domain,
            overload: value.overload,
            value_info: parse_repeated(
                value.value_info,
                onnx::FunctionProto::VALUE_INFO_FIELD_NAME,
            )?,
            metadata_props: parse_property_map(value.metadata_props).map_err(|error| {
                error.wrap_field(onnx::FunctionProto::METADATA_PROPS_FIELD_NAME)
            })?,
        })
    }
}

impl From<onnx_structure::ModelProto> for onnx::ModelProto {
    fn from(value: onnx_structure::ModelProto) -> Self {
        Self {
            ir_version: value.ir_version,
            opset_import: value.opset_import.into_iter().map(Into::into).collect(),
            producer_name: value.producer_name,
            producer_version: value.producer_version,
            domain: value.domain,
            model_version: value.model_version,
            doc_string: value.doc_string,
            graph: value.graph.map(Into::into),
            metadata_props: value.metadata_props.into_iter().map(Into::into).collect(),
            training_info: value.training_info.into_iter().map(Into::into).collect(),
            functions: value.functions.into_iter().map(Into::into).collect(),
            configuration: Vec::new(),
        }
    }
}

impl From<onnx_structure::OperatorSetIdProto> for onnx::OperatorSetIdProto {
    fn from(value: onnx_structure::OperatorSetIdProto) -> Self {
        Self {
            domain: value.domain,
            version: value.version,
        }
    }
}

impl From<onnx_structure::GraphProto> for onnx::GraphProto {
    fn from(value: onnx_structure::GraphProto) -> Self {
        Self {
            node: value.node.into_iter().map(Into::into).collect(),
            name: value.name,
            initializer: value.initializer.into_iter().map(Into::into).collect(),
            sparse_initializer: value
                .sparse_initializer
                .into_iter()
                .map(Into::into)
                .collect(),
            doc_string: value.doc_string,
            input: value.input.into_iter().map(Into::into).collect(),
            output: value.output.into_iter().map(Into::into).collect(),
            value_info: value.value_info.into_iter().map(Into::into).collect(),
            quantization_annotation: value
                .quantization_annotation
                .into_iter()
                .map(Into::into)
                .collect(),
            metadata_props: value.metadata_props.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<onnx_structure::NodeProto> for onnx::NodeProto {
    fn from(value: onnx_structure::NodeProto) -> Self {
        Self {
            input: value.input,
            output: value.output,
            name: value.name,
            op_type: value.op_type,
            attribute: value.attribute.into_iter().map(Into::into).collect(),
            doc_string: value.doc_string,
            domain: value.domain,
            overload: value.overload,
            metadata_props: value.metadata_props.into_iter().map(Into::into).collect(),
            device_configurations: Vec::new(),
        }
    }
}

impl From<onnx_structure::AttributeProto> for onnx::AttributeProto {
    fn from(value: onnx_structure::AttributeProto) -> Self {
        Self {
            name: value.name,
            ref_attr_name: value.ref_attr_name,
            doc_string: value.doc_string,
            r#type: value.r#type,
            f: value.f,
            i: value.i,
            s: value.s,
            t: value.t.map(Into::into),
            g: value.g.map(Into::into),
            sparse_tensor: value.sparse_tensor.map(Into::into),
            tp: value.tp.map(Into::into),
            floats: value.floats,
            ints: value.ints,
            strings: value.strings,
            tensors: value.tensors.into_iter().map(Into::into).collect(),
            graphs: value.graphs.into_iter().map(Into::into).collect(),
            sparse_tensors: value.sparse_tensors.into_iter().map(Into::into).collect(),
            type_protos: value.type_protos.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<onnx_structure::TensorProto> for onnx::TensorProto {
    fn from(value: onnx_structure::TensorProto) -> Self {
        Self {
            dims: value.dims,
            data_type: value.data_type,
            segment: value.segment.map(Into::into),
            float_data: Vec::new(),
            int32_data: Vec::new(),
            string_data: Vec::new(),
            int64_data: Vec::new(),
            name: value.name,
            doc_string: value.doc_string,
            raw_data: Vec::new(),
            external_data: value.external_data.into_iter().map(Into::into).collect(),
            data_location: value.data_location,
            double_data: Vec::new(),
            uint64_data: Vec::new(),
            metadata_props: value.metadata_props.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<onnx_structure::tensor_proto::Segment> for onnx::tensor_proto::Segment {
    fn from(value: onnx_structure::tensor_proto::Segment) -> Self {
        Self {
            begin: value.begin,
            end: value.end,
        }
    }
}

impl From<onnx_structure::SparseTensorProto> for onnx::SparseTensorProto {
    fn from(value: onnx_structure::SparseTensorProto) -> Self {
        Self {
            values: value.values.map(Into::into),
            indices: value.indices.map(Into::into),
            dims: value.dims,
        }
    }
}

impl From<onnx_structure::ValueInfoProto> for onnx::ValueInfoProto {
    fn from(value: onnx_structure::ValueInfoProto) -> Self {
        Self {
            name: value.name,
            r#type: value.r#type.map(Into::into),
            doc_string: value.doc_string,
            metadata_props: value.metadata_props.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<onnx_structure::TypeProto> for onnx::TypeProto {
    fn from(value: onnx_structure::TypeProto) -> Self {
        Self {
            denotation: value.denotation,
            value: value.value.map(Into::into),
        }
    }
}

impl From<onnx_structure::type_proto::Value> for onnx::type_proto::Value {
    fn from(value: onnx_structure::type_proto::Value) -> Self {
        match value {
            onnx_structure::type_proto::Value::TensorType(value) => Self::TensorType(value.into()),
            onnx_structure::type_proto::Value::SequenceType(value) => {
                Self::SequenceType(Box::new((*value).into()))
            }
            onnx_structure::type_proto::Value::MapType(value) => {
                Self::MapType(Box::new((*value).into()))
            }
            onnx_structure::type_proto::Value::SparseTensorType(value) => {
                Self::SparseTensorType(value.into())
            }
            onnx_structure::type_proto::Value::OptionalType(value) => {
                Self::OptionalType(Box::new((*value).into()))
            }
            onnx_structure::type_proto::Value::OpaqueType(value) => {
                Self::OpaqueType(onnx::type_proto::Opaque {
                    domain: value.domain,
                    name: value.name,
                })
            }
        }
    }
}

impl From<onnx_structure::type_proto::Tensor> for onnx::type_proto::Tensor {
    fn from(value: onnx_structure::type_proto::Tensor) -> Self {
        Self {
            elem_type: value.elem_type,
            shape: value.shape.map(Into::into),
        }
    }
}

impl From<onnx_structure::type_proto::Sequence> for onnx::type_proto::Sequence {
    fn from(value: onnx_structure::type_proto::Sequence) -> Self {
        Self {
            elem_type: value.elem_type.map(|value| Box::new((*value).into())),
        }
    }
}

impl From<onnx_structure::type_proto::Map> for onnx::type_proto::Map {
    fn from(value: onnx_structure::type_proto::Map) -> Self {
        Self {
            key_type: value.key_type,
            value_type: value.value_type.map(|value| Box::new((*value).into())),
        }
    }
}

impl From<onnx_structure::type_proto::Optional> for onnx::type_proto::Optional {
    fn from(value: onnx_structure::type_proto::Optional) -> Self {
        Self {
            elem_type: value.elem_type.map(|value| Box::new((*value).into())),
        }
    }
}

impl From<onnx_structure::type_proto::SparseTensor> for onnx::type_proto::SparseTensor {
    fn from(value: onnx_structure::type_proto::SparseTensor) -> Self {
        Self {
            elem_type: value.elem_type,
            shape: value.shape.map(Into::into),
        }
    }
}

impl From<onnx_structure::TensorShapeProto> for onnx::TensorShapeProto {
    fn from(value: onnx_structure::TensorShapeProto) -> Self {
        Self {
            dim: value.dim.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<onnx_structure::tensor_shape_proto::Dimension> for onnx::tensor_shape_proto::Dimension {
    fn from(value: onnx_structure::tensor_shape_proto::Dimension) -> Self {
        Self {
            denotation: value.denotation,
            value: value.value.map(Into::into),
        }
    }
}

impl From<onnx_structure::tensor_shape_proto::dimension::Value>
    for onnx::tensor_shape_proto::dimension::Value
{
    fn from(value: onnx_structure::tensor_shape_proto::dimension::Value) -> Self {
        match value {
            onnx_structure::tensor_shape_proto::dimension::Value::DimValue(value) => {
                Self::DimValue(value)
            }
            onnx_structure::tensor_shape_proto::dimension::Value::DimParam(value) => {
                Self::DimParam(value)
            }
        }
    }
}

impl From<onnx_structure::TensorAnnotation> for onnx::TensorAnnotation {
    fn from(value: onnx_structure::TensorAnnotation) -> Self {
        Self {
            tensor_name: value.tensor_name,
            quant_parameter_tensor_names: value
                .quant_parameter_tensor_names
                .into_iter()
                .map(Into::into)
                .collect(),
        }
    }
}

impl From<onnx_structure::TrainingInfoProto> for onnx::TrainingInfoProto {
    fn from(value: onnx_structure::TrainingInfoProto) -> Self {
        Self {
            initialization: value.initialization.map(Into::into),
            algorithm: value.algorithm.map(Into::into),
            initialization_binding: value
                .initialization_binding
                .into_iter()
                .map(Into::into)
                .collect(),
            update_binding: value.update_binding.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<onnx_structure::FunctionProto> for onnx::FunctionProto {
    fn from(value: onnx_structure::FunctionProto) -> Self {
        Self {
            name: value.name,
            input: value.input,
            output: value.output,
            attribute: value.attribute,
            node: value.node.into_iter().map(Into::into).collect(),
            doc_string: value.doc_string,
            opset_import: value.opset_import.into_iter().map(Into::into).collect(),
            domain: value.domain,
            attribute_proto: value.attribute_proto.into_iter().map(Into::into).collect(),
            value_info: value.value_info.into_iter().map(Into::into).collect(),
            overload: value.overload,
            metadata_props: value.metadata_props.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<onnx_structure::StringStringEntryProto> for onnx::StringStringEntryProto {
    fn from(value: onnx_structure::StringStringEntryProto) -> Self {
        Self {
            key: value.key,
            value: value.value,
        }
    }
}

fn non_empty(value: String) -> Option<String> {
    (!value.is_empty()).then_some(value)
}

fn parse_enum<T>(value: i32, field: &'static str) -> RequestResult<T>
where
    T: TryFrom<i32>,
{
    value
        .try_into()
        .map_err(|_| RequestError::field(field, CommonError::InvalidEnumValue))
}

fn parse_required<T, U>(value: Option<T>, field: &'static str) -> RequestResult<U>
where
    U: FromProto<T>,
{
    value
        .ok_or_else(|| RequestError::field(field, CommonError::RequiredFieldMissing))?
        .parse_into()
        .map_err(|error| error.wrap_field(field))
}

fn parse_optional<T, U>(value: Option<T>, field: &'static str) -> RequestResult<Option<U>>
where
    U: FromProto<T>,
{
    value
        .map(|value| value.parse_into().map_err(|error| error.wrap_field(field)))
        .transpose()
}

fn parse_repeated<T, U>(values: Vec<T>, field: &'static str) -> RequestResult<Vec<U>>
where
    U: FromProto<T>,
{
    values
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            value.parse_into().map_err(|error| {
                error.wrap_path([
                    PathErrorStep::Field(field.into()),
                    PathErrorStep::Index(index),
                ])
            })
        })
        .collect()
}

fn parse_property_map(values: Vec<onnx::StringStringEntryProto>) -> RequestResult<PropertyMap> {
    let mut map = PropertyMap::new();
    for onnx::StringStringEntryProto { key, value } in values {
        if map.insert(key, value).is_some() {
            return Err(CommonError::DuplicateValue.into());
        }
    }
    Ok(map)
}

fn parse_string(value: Vec<u8>, field: &'static str) -> RequestResult<String> {
    String::from_utf8(value).map_err(|_| {
        RequestError::field(
            field,
            CommonError::InvalidStringFormat {
                expected: "utf-8".into(),
            },
        )
    })
}

fn parse_strings(values: Vec<Vec<u8>>, field: &'static str) -> RequestResult<Vec<String>> {
    values
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            String::from_utf8(value).map_err(|_| {
                RequestError::path(
                    [
                        PathErrorStep::Field(field.into()),
                        PathErrorStep::Index(index),
                    ],
                    CommonError::InvalidStringFormat {
                        expected: "utf-8".into(),
                    },
                )
            })
        })
        .collect()
}

fn parse_tensor_data(
    value: &mut onnx::TensorProto,
    data_type: TensorDataType,
) -> RequestResult<Option<TensorElements>> {
    if !value.raw_data.is_empty() {
        if data_type == TensorDataType::String {
            return Err(RequestError::field(
                onnx::TensorProto::RAW_DATA_FIELD_NAME,
                CommonError::TypeMismatch,
            ));
        }
        return Ok(Some(TensorElements::Raw(take(&mut value.raw_data))));
    }

    match data_type {
        TensorDataType::Undefined => unreachable!("undefined tensor data type is rejected earlier"),
        TensorDataType::Float | TensorDataType::Complex64 => Ok((!value.float_data.is_empty())
            .then(|| TensorElements::Float(take(&mut value.float_data)))),
        TensorDataType::Int32
        | TensorDataType::Uint8
        | TensorDataType::Int8
        | TensorDataType::Uint16
        | TensorDataType::Int16
        | TensorDataType::Bool
        | TensorDataType::Float16
        | TensorDataType::Bfloat16
        | TensorDataType::Float8e4m3fn
        | TensorDataType::Float8e4m3fnuz
        | TensorDataType::Float8e5m2
        | TensorDataType::Float8e5m2fnuz
        | TensorDataType::Uint4
        | TensorDataType::Int4
        | TensorDataType::Float4e2m1
        | TensorDataType::Float8e8m0
        | TensorDataType::Uint2
        | TensorDataType::Int2
        | TensorDataType::Float6e2m3
        | TensorDataType::Float6e3m2 => Ok((!value.int32_data.is_empty())
            .then(|| TensorElements::Int32(take(&mut value.int32_data)))),
        TensorDataType::String => {
            if value.string_data.is_empty() {
                Ok(None)
            } else {
                Ok(Some(TensorElements::String(parse_strings(
                    take(&mut value.string_data),
                    onnx::TensorProto::STRING_DATA_FIELD_NAME,
                )?)))
            }
        }
        TensorDataType::Int64 => Ok((!value.int64_data.is_empty())
            .then(|| TensorElements::Int64(take(&mut value.int64_data)))),
        TensorDataType::Double | TensorDataType::Complex128 => Ok((!value.double_data.is_empty())
            .then(|| TensorElements::Double(take(&mut value.double_data)))),
        TensorDataType::Uint32 | TensorDataType::Uint64 => Ok((!value.uint64_data.is_empty())
            .then(|| TensorElements::Uint64(take(&mut value.uint64_data)))),
    }
}
