//! Output inference rules, one function per operation family, selected by the operation table.

mod elementwise;
mod neural;
mod random;
mod reduction;
mod shape;

pub use self::{
    elementwise::{
        bitwise_not, cast, clamp, copy_to, custom, float_only, floating_unary, is_nan, lerp,
        logical_not, promoted_binary, quantization, reverse_divide_scalar, rounding, same_as_input,
        scalar, signed_unary, square, where_select,
    },
    neural::{
        attention, avg_pool2d, batch_norm, contraction, conv2d, conv3d, embedding, grouped_matmul,
        irfft, layer_norm, matmul, max_pool2d, paged_attention, paged_attention_lse,
        paged_attention_output, paged_mla_attention, paged_mla_attention_lse,
        paged_mla_attention_output, quantized_matmul, rfft, rotary_embedding, scaled_matmul,
        softmax,
    },
    random::{bernoulli, multinomial, normal, uniform},
    reduction::{arg_reduction, argsort, cumulative, reduction, topk},
    shape::{
        as_strided, assign, concat, expand, flatten, index, index_put, index_select, index_write,
        pad, permute, repeat_kv, reshape, slice, squeeze, strided_scatter, take, transpose,
        unsqueeze,
    },
};
use crate::{
    backend::registry::{
        AliasBehavior, Operation, OperationAttributes, OperationInference, OperationRequest,
        OperationSchema, ResultAlias, table::operation_inference_rule,
    },
    data_type::DataType,
    error::{Error, Result},
    shape::Shape,
    value::ValueSpec,
};

/// Infers the results of `request` with the rule its operation declares.
pub fn infer_operation(
    request: &OperationRequest,
    schema: &OperationSchema,
) -> Result<OperationInference> {
    if request.inputs().is_empty() {
        return Err(Error::missing_input("infer operation"));
    }

    let context = InferenceContext {
        operation: request.operation(),
        attributes: request.attributes(),
        inputs: request.inputs(),
        schema,
    };
    operation_inference_rule(request.operation())(&context)
}

/// Infers the results of one operation request.
pub type InferenceRule = fn(&InferenceContext<'_>) -> Result<OperationInference>;

/// The operation, attributes, input specifications, and schema an inference rule reads.
#[derive(Debug, Clone, Copy)]
pub struct InferenceContext<'a> {
    operation: Operation,
    attributes: &'a OperationAttributes,
    inputs: &'a [ValueSpec],
    schema: &'a OperationSchema,
}

impl<'a> InferenceContext<'a> {
    fn input(&self, index: usize) -> Result<&'a ValueSpec> {
        self.inputs
            .get(index)
            .ok_or_else(|| Error::missing_input("input"))
    }

    fn first(&self) -> &'a ValueSpec {
        &self.inputs[0]
    }

    fn shape(&self) -> &'a Shape {
        self.first().shape()
    }

    /// Returns the attributes selected by `accessor`, or an error for another attribute variant.
    fn attributes<T: ?Sized>(
        &self,
        accessor: impl FnOnce(&'a OperationAttributes) -> Option<&'a T>,
    ) -> Result<&'a T> {
        accessor(self.attributes).ok_or_else(|| Error::invalid_argument("operation attribute kind"))
    }

    /// Returns the value selected by `accessor`, or an error for another attribute variant.
    fn value<T>(&self, accessor: impl FnOnce(&'a OperationAttributes) -> Option<T>) -> Result<T> {
        accessor(self.attributes).ok_or_else(|| Error::invalid_argument("operation attribute kind"))
    }

    fn require_same_types(&self) -> Result<()> {
        let first = self.first();
        for input in &self.inputs[1..] {
            if input.data_type() != first.data_type() {
                return Err(Error::data_type_mismatch(
                    "operation operands",
                    first.data_type(),
                    input.data_type(),
                ));
            }
        }
        Ok(())
    }

    fn optional_shape(&self, index: usize, present: bool) -> Result<Option<&'a Shape>> {
        present
            .then(|| self.input(index).map(ValueSpec::shape))
            .transpose()
    }

    /// Returns one result per output role with `shape`, keeping the first input's data type.
    fn same_type(&self, shape: &Shape) -> OperationInference {
        self.finish(shape, self.first().data_type())
    }

    /// Returns one result per output role with `shape` and `data_type`, computed from every input converted to `data_type`.
    fn converted(&self, shape: &Shape, data_type: DataType) -> OperationInference {
        let mut inference = self.finish(shape, data_type);
        inference.kernel_input_data_types = vec![data_type; self.inputs.len()];
        inference
    }

    /// Returns one result per output role with `shape` and `data_type` mapped through the role.
    fn finish(&self, shape: &Shape, data_type: DataType) -> OperationInference {
        let outputs = self
            .schema
            .output_roles()
            .iter()
            .map(|role| ValueSpec::new(role.data_type(data_type), shape.clone()))
            .collect::<Vec<_>>();
        let alias = match self.schema.alias_behavior() {
            AliasBehavior::None | AliasBehavior::WritesInput => ResultAlias::Independent,
            AliasBehavior::View
                if matches!(
                    self.operation,
                    Operation::Reshape | Operation::Flatten | Operation::AsStrided
                ) =>
            {
                ResultAlias::MayAlias(0)
            }
            AliasBehavior::View => ResultAlias::ViewOf(0),
            AliasBehavior::MayAliasInput => ResultAlias::MayAlias(0),
        };
        OperationInference {
            aliases: vec![alias; outputs.len()],
            outputs,
            computation_data_type: data_type,
            kernel_input_data_types: self.inputs.iter().map(ValueSpec::data_type).collect(),
        }
    }
}

fn require_indices(input: &ValueSpec) -> Result<()> {
    if input.data_type() == DataType::I64 {
        Ok(())
    } else {
        Err(Error::unsupported_data_type("indices must be i64"))
    }
}
