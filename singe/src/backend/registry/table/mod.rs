use crate::{
    backend::registry::{
        ASSIGN_OPERANDS, ATTENTION_LSE_OUTPUTS, AliasBehavior, BATCH_NORM_OPERANDS,
        BINARY_OPERANDS, CONTRACTION_OPERANDS, CONV2D_OPERANDS, ClampAttributes,
        EMBEDDING_OPERANDS, GROUPED_MATMUL_OPERANDS, INDEX_LEADING_OPERANDS, INDEX_OUTPUT,
        INDEX_PUT_LEADING_OPERANDS, INDEX_SELECT_OPERANDS, INDEX_WRITE_LEADING_OPERANDS,
        LAYER_NORM_OPERANDS, LERP_OPERANDS, LSE_OUTPUT, MASK_OUTPUT, MATMUL_OPERANDS, NO_OUTPUTS,
        OperandRole, OperandSchema, OperationAttributeKind, OperationAttributes,
        OperationExecutorFamily, OutputRole, PAGED_ATTENTION_OPERANDS,
        PAGED_MLA_ATTENTION_OPERANDS, QUANTIZE_OPERANDS, QUANTIZED_MATMUL_OPERANDS,
        RANDOM_OPERANDS, ROTARY_EMBEDDING_OPERANDS, ReductionAttributes, SAMPLING_OPERANDS,
        SCALED_DOT_PRODUCT_ATTENTION_OPERANDS, SCALED_MATMUL_OPERANDS, SCATTER_OPERANDS,
        SortAttributes, TERNARY_WHERE_OPERANDS, TOPK_OUTPUTS, UNARY_OPERANDS, VALUE_OUTPUT,
        infer::{self, InferenceRule},
        support::OperationSupportProfile,
    },
    data_type::DataType,
    dispatch::{CustomOperation, call_one},
    error::Result,
    tensor::Tensor,
};

/// Generates the tensor method an operation table row declares.
macro_rules! tensor_method {
    (none; $variant:ident; $attribute_kind:ident; $($doc:literal),+) => {};
    (unary($method:ident); $variant:ident; $attribute_kind:ident; $($doc:literal),+) => {
        $(#[doc = $doc])+
        ///
        /// # Errors
        ///
        /// Returns an error if the backend does not support the operation for this tensor.
        pub fn $method(&self) -> Result<Tensor> {
            call_one(Operation::$variant, OperationAttributes::None, &[self])
        }
    };
    (binary($method:ident); $variant:ident; $attribute_kind:ident; $($doc:literal),+) => {
        $(#[doc = $doc])+
        ///
        /// Operands of different data types are promoted to a common type first.
        ///
        /// # Errors
        ///
        /// Returns an error if shapes, devices, data types, or backend support are invalid.
        pub fn $method(&self, other: &Tensor) -> Result<Tensor> {
            let (lhs, rhs) = self.binary_operands(other, Operation::$variant)?;
            call_one(Operation::$variant, OperationAttributes::None, &[&lhs, &rhs])
        }
    };
    (reduction($method:ident); $variant:ident; $attribute_kind:ident; $($doc:literal),+) => {
        $(#[doc = $doc])+
        ///
        /// An empty `dimensions` slice selects all axes, and negative dimensions count backward from the last axis.
        /// `keepdim` retains reduced axes with size one; otherwise those axes are removed.
        ///
        /// # Errors
        ///
        /// Returns an error if an axis is out of range or repeated, or the backend does not support the tensor type or layout.
        pub fn $method(&self, dimensions: &[i64], keepdim: bool) -> Result<Tensor> {
            call_one(
                Operation::$variant,
                OperationAttributes::$attribute_kind(ReductionAttributes {
                    dimensions: dimensions.to_vec(),
                    keepdim,
                }),
                &[self],
            )
        }
    };
    (value($method:ident, $parameter:ident: $parameter_type:ty); $variant:ident; $attribute_kind:ident; $($doc:literal),+) => {
        $(#[doc = $doc])+
        ///
        /// # Errors
        ///
        /// Returns an error if the argument is invalid for this tensor, or the backend does not support the operation.
        pub fn $method(&self, $parameter: $parameter_type) -> Result<Tensor> {
            call_one(Operation::$variant, OperationAttributes::$attribute_kind($parameter), &[self])
        }
    };
    (fields($method:ident, $attributes:ident { $($field:ident: $field_type:ty),+ }); $variant:ident; $attribute_kind:ident; $($doc:literal),+) => {
        $(#[doc = $doc])+
        ///
        /// # Errors
        ///
        /// Returns an error if the parameters are invalid for this tensor, or the backend does not support the operation.
        pub fn $method(&self, $($field: $field_type),+) -> Result<Tensor> {
            call_one(
                Operation::$variant,
                OperationAttributes::$attribute_kind($attributes { $($field),+ }),
                &[self],
            )
        }
    };
    (indexed($method:ident); $variant:ident; $attribute_kind:ident; $($doc:literal),+) => {
        $(#[doc = $doc])+
        ///
        /// Negative `dim` values count backward from the last axis.
        ///
        /// # Errors
        ///
        /// Returns an error if `dim` or an index is out of range, the operands are on different devices, or the backend does not support the index data type.
        pub fn $method(&self, dim: i64, indices: &Tensor) -> Result<Tensor> {
            call_one(Operation::$variant, OperationAttributes::$attribute_kind(dim), &[self, indices])
        }
    };
    (paired($method:ident, $parameter:ident: $parameter_type:ty); $variant:ident; $attribute_kind:ident; $($doc:literal),+) => {
        $(#[doc = $doc])+
        ///
        /// # Errors
        ///
        /// Returns an error if shapes, devices, data types, or backend support are invalid.
        pub fn $method(&self, other: &Tensor, $parameter: $parameter_type) -> Result<Tensor> {
            call_one(Operation::$variant, OperationAttributes::$attribute_kind($parameter), &[self, other])
        }
    };
}

macro_rules! operation_table {
    ($(
        $(#[doc = $doc:literal])+
        $variant:ident {
            name: $name:literal,
            support: $profile:ident,
            family: $executor_family:ident,
            elementwise: $elementwise_kind:ident,
            attributes: $attribute_kind:ident,
            operands: $operand_schema:expr,
            infer: $infer:ident,
            outputs: $output_schema:ident,
            alias: $alias_behavior:ident,
            method: $method_kind:ident $(($($method:tt)+))?,
            onnx: $onnx:expr $(,)?
        }
    ),+ $(,)?) => {
        /// Canonical operation codes used by logical inference and backend dispatch.
        /// Backend availability, operand roles, attributes, and output rules come from the registry schema.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum Operation {
            $( $(#[doc = $doc])+ $variant,)+
            /// An operation registered at run time with [`CustomOperation::register`].
            Custom(CustomOperation),
        }

        /// All built-in operations.
        ///
        /// The list includes CPU-only operations and operations disabled for CUDA so callers can enumerate the full support matrix.
        /// Custom operations are not listed.
        pub const REGISTERED_OPERATIONS: &[Operation] = &[
            $(Operation::$variant,)+
        ];

        /// Returns the registry name used in operation diagnostics.
        pub fn operation_name(operation: Operation) -> &'static str {
            match operation {
                $(Operation::$variant => $name,)+
                Operation::Custom(custom) => custom.name(),
            }
        }

        pub fn operation_support_profile(operation: Operation) -> OperationSupportProfile {
            match operation {
                $(Operation::$variant => OperationSupportProfile::$profile,)+
                Operation::Custom(_) => OperationSupportProfile::Custom,
            }
        }

        pub fn operation_executor_family(operation: Operation) -> OperationExecutorFamily {
            match operation {
                $(Operation::$variant => OperationExecutorFamily::$executor_family,)+
                Operation::Custom(_) => OperationExecutorFamily::Custom,
            }
        }

        fn operation_elementwise_kind(operation: Operation) -> OperationElementwiseKind {
            match operation {
                $(Operation::$variant => OperationElementwiseKind::$elementwise_kind,)+
                Operation::Custom(_) => OperationElementwiseKind::None,
            }
        }

        pub fn operation_attribute_kind(operation: Operation) -> OperationAttributeKind {
            match operation {
                $(Operation::$variant => OperationAttributeKind::$attribute_kind,)+
                Operation::Custom(_) => OperationAttributeKind::Custom,
            }
        }

        pub fn operation_operand_schema(operation: Operation) -> OperandSchema {
            match operation {
                $(Operation::$variant => $operand_schema,)+
                Operation::Custom(custom) => OperandSchema::Fixed(custom.operands()),
            }
        }

        /// Returns the function that infers the results of `operation`.
        pub fn operation_inference_rule(operation: Operation) -> InferenceRule {
            match operation {
                $(Operation::$variant => infer::$infer,)+
                Operation::Custom(_) => infer::custom,
            }
        }

        pub fn operation_output_schema(operation: Operation) -> &'static [OutputRole] {
            match operation {
                $(Operation::$variant => $output_schema,)+
                Operation::Custom(custom) => custom.outputs(),
            }
        }

        pub fn operation_alias_behavior(operation: Operation) -> AliasBehavior {
            match operation {
                $(Operation::$variant => AliasBehavior::$alias_behavior,)+
                Operation::Custom(_) => AliasBehavior::None,
            }
        }

        impl Operation {
            /// Returns the ONNX operator this operation maps onto one to one, with the same operands and no attributes.
            #[must_use]
            pub fn onnx_operator(self) -> Option<&'static str> {
                match self {
                    $(Operation::$variant => $onnx,)+
                    Operation::Custom(_) => None,
                }
            }

            /// Returns the operation that maps one to one onto the ONNX operator `name`.
            #[must_use]
            pub fn from_onnx_operator(name: &str) -> Option<Self> {
                REGISTERED_OPERATIONS
                    .iter()
                    .copied()
                    .find(|operation| operation.onnx_operator() == Some(name))
            }
        }

        impl Tensor {
            $(tensor_method!($method_kind $(($($method)+))?; $variant; $attribute_kind; $($doc),+);)+
        }
    };
}

operation_table! {
    /// Adds two operands elementwise with broadcasting.
    Add {
        name: "add",
        support: ElementWise,
        family: Binary,
        elementwise: CommutativeBinary,
        attributes: None,
        operands: OperandSchema::Fixed(BINARY_OPERANDS),
        infer: promoted_binary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: Some("Add"),
    },
    /// Subtracts the right operand from the left with broadcasting.
    Sub {
        name: "sub",
        support: ElementWise,
        family: Binary,
        elementwise: Binary,
        attributes: None,
        operands: OperandSchema::Fixed(BINARY_OPERANDS),
        infer: promoted_binary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: Some("Sub"),
    },
    /// Multiplies two operands elementwise with broadcasting.
    Mul {
        name: "mul",
        support: ElementWise,
        family: Binary,
        elementwise: CommutativeBinary,
        attributes: None,
        operands: OperandSchema::Fixed(BINARY_OPERANDS),
        infer: promoted_binary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: Some("Mul"),
    },
    /// Divides the left operand by the right with broadcasting.
    Div {
        name: "div",
        support: ElementWise,
        family: Binary,
        elementwise: Binary,
        attributes: None,
        operands: OperandSchema::Fixed(BINARY_OPERANDS),
        infer: promoted_binary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: Some("Div"),
    },
    /// Multiplies vectors, matrices, or broadcast batches of matrices.
    MatMul {
        name: "matmul",
        support: MatMul,
        family: Binary,
        elementwise: None,
        attributes: None,
        operands: OperandSchema::Fixed(MATMUL_OPERANDS),
        infer: matmul,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: Some("MatMul"),
    },
    /// Negates every input element.
    Neg {
        name: "neg",
        support: ElementWise,
        family: Unary,
        elementwise: Unary,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: signed_unary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: Some("Neg"),
    },
    /// Computes the absolute value of every input element.
    Abs {
        name: "abs",
        support: ElementWise,
        family: Unary,
        elementwise: IdempotentUnary,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: signed_unary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: unary(abs),
        onnx: Some("Abs"),
    },
    /// Computes the elementwise square root.
    Sqrt {
        name: "sqrt",
        support: ElementWise,
        family: Unary,
        elementwise: Unary,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: floating_unary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: unary(sqrt),
        onnx: Some("Sqrt"),
    },
    /// Computes the elementwise natural exponential.
    Exp {
        name: "exp",
        support: ElementWise,
        family: Unary,
        elementwise: Unary,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: floating_unary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: unary(exp),
        onnx: Some("Exp"),
    },
    /// Computes the elementwise natural logarithm.
    Log {
        name: "log",
        support: ElementWise,
        family: Unary,
        elementwise: Unary,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: floating_unary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: unary(log),
        onnx: Some("Log"),
    },
    /// Computes the elementwise sine in radians.
    Sin {
        name: "sin",
        support: ElementWise,
        family: Unary,
        elementwise: Unary,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: floating_unary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: unary(sin),
        onnx: Some("Sin"),
    },
    /// Computes the elementwise cosine in radians.
    Cos {
        name: "cos",
        support: ElementWise,
        family: Unary,
        elementwise: Unary,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: floating_unary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: unary(cos),
        onnx: Some("Cos"),
    },
    /// Computes the elementwise tangent in radians.
    Tan {
        name: "tan",
        support: ElementWise,
        family: Unary,
        elementwise: Unary,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: floating_unary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: unary(tan),
        onnx: Some("Tan"),
    },
    /// Computes the elementwise hyperbolic sine.
    Sinh {
        name: "sinh",
        support: ElementWise,
        family: Unary,
        elementwise: Unary,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: floating_unary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: unary(sinh),
        onnx: Some("Sinh"),
    },
    /// Computes the elementwise hyperbolic cosine.
    Cosh {
        name: "cosh",
        support: ElementWise,
        family: Unary,
        elementwise: Unary,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: floating_unary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: unary(cosh),
        onnx: Some("Cosh"),
    },
    /// Computes the elementwise hyperbolic tangent.
    Tanh {
        name: "tanh",
        support: ElementWise,
        family: Unary,
        elementwise: Unary,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: floating_unary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: unary(tanh),
        onnx: Some("Tanh"),
    },
    /// Computes the elementwise inverse sine in radians.
    Asin {
        name: "asin",
        support: ElementWise,
        family: Unary,
        elementwise: Unary,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: floating_unary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: unary(asin),
        onnx: Some("Asin"),
    },
    /// Computes the elementwise inverse cosine in radians.
    Acos {
        name: "acos",
        support: ElementWise,
        family: Unary,
        elementwise: Unary,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: floating_unary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: unary(acos),
        onnx: Some("Acos"),
    },
    /// Computes the elementwise inverse tangent in radians.
    Atan {
        name: "atan",
        support: ElementWise,
        family: Unary,
        elementwise: Unary,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: floating_unary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: unary(atan),
        onnx: Some("Atan"),
    },
    /// Rounds each element toward positive infinity.
    Ceil {
        name: "ceil",
        support: ElementWise,
        family: Unary,
        elementwise: IdempotentUnary,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: rounding,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: unary(ceil),
        onnx: Some("Ceil"),
    },
    /// Rounds each element toward negative infinity.
    Floor {
        name: "floor",
        support: ElementWise,
        family: Unary,
        elementwise: IdempotentUnary,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: rounding,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: unary(floor),
        onnx: Some("Floor"),
    },
    /// Rounds each element toward zero.
    Truncate {
        name: "trunc",
        support: ElementWise,
        family: Unary,
        elementwise: IdempotentUnary,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: rounding,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: unary(truncate),
        onnx: None,
    },
    /// Computes one divided by each input element.
    Reciprocal {
        name: "reciprocal",
        support: ElementWise,
        family: Unary,
        elementwise: Unary,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: floating_unary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: unary(reciprocal),
        onnx: Some("Reciprocal"),
    },
    /// Multiplies each input element by itself.
    Square {
        name: "square",
        support: ElementWise,
        family: Unary,
        elementwise: Unary,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: square,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: unary(square),
        onnx: None,
    },
    /// Computes the reciprocal square root of each element.
    Rsqrt {
        name: "rsqrt",
        support: ElementWise,
        family: Unary,
        elementwise: Unary,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: floating_unary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: unary(rsqrt),
        onnx: None,
    },
    /// Computes the elementwise inverse hyperbolic sine.
    Asinh {
        name: "asinh",
        support: ElementWise,
        family: Unary,
        elementwise: Unary,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: floating_unary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: unary(asinh),
        onnx: Some("Asinh"),
    },
    /// Computes the elementwise inverse hyperbolic cosine.
    Acosh {
        name: "acosh",
        support: ElementWise,
        family: Unary,
        elementwise: Unary,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: floating_unary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: unary(acosh),
        onnx: Some("Acosh"),
    },
    /// Computes the elementwise inverse hyperbolic tangent.
    Atanh {
        name: "atanh",
        support: ElementWise,
        family: Unary,
        elementwise: Unary,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: floating_unary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: unary(atanh),
        onnx: Some("Atanh"),
    },
    /// Computes the elementwise Gauss error function.
    Erf {
        name: "erf",
        support: ElementWise,
        family: Unary,
        elementwise: Unary,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: floating_unary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: unary(erf),
        onnx: Some("Erf"),
    },
    /// Rounds each element to the nearest integer, ties to even; integers stay as they are.
    Round {
        name: "round",
        support: ElementWise,
        family: Unary,
        elementwise: IdempotentUnary,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: rounding,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: unary(round),
        onnx: Some("Round"),
    },
    /// Tests each element for NaN; integers and Booleans never are.
    IsNan {
        name: "isnan",
        support: ElementWise,
        family: Unary,
        elementwise: Unary,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: is_nan,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: unary(isnan),
        onnx: Some("IsNaN"),
    },
    /// Computes `log(1 + x)` elementwise, exact for small `x`.
    Log1p {
        name: "log1p",
        support: ElementWise,
        family: Unary,
        elementwise: Unary,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: floating_unary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: unary(log1p),
        onnx: None,
    },
    /// Computes `exp(x) − 1` elementwise, exact for small `x`.
    Expm1 {
        name: "expm1",
        support: ElementWise,
        family: Unary,
        elementwise: Unary,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: floating_unary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: unary(expm1),
        onnx: None,
    },
    /// Applies the rectified linear activation elementwise.
    Relu {
        name: "relu",
        support: ElementWise,
        family: Activation,
        elementwise: IdempotentUnary,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: signed_unary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: unary(relu),
        onnx: Some("Relu"),
    },
    /// Applies the tanh approximation of the Gaussian error linear activation.
    Gelu {
        name: "gelu",
        support: ElementWise,
        family: Activation,
        elementwise: Unary,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: float_only,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: unary(gelu),
        onnx: None,
    },
    /// Applies the error-function formula for the Gaussian error linear activation.
    GeluExact {
        name: "gelu_exact",
        support: ElementWise,
        family: Activation,
        elementwise: Unary,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: float_only,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: unary(gelu_exact),
        onnx: None,
    },
    /// Multiplies each input element by its sigmoid.
    Silu {
        name: "silu",
        support: ElementWise,
        family: Activation,
        elementwise: Unary,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: float_only,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: unary(silu),
        onnx: None,
    },
    /// Multiplies the first operand by the `SiLU` activation of the second, with broadcasting.
    SwiGlu {
        name: "swiglu",
        support: ElementWise,
        family: Activation,
        elementwise: Binary,
        attributes: None,
        operands: OperandSchema::Fixed(BINARY_OPERANDS),
        infer: float_only,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: binary(swiglu),
        onnx: None,
    },
    /// Applies the logistic sigmoid elementwise.
    Sigmoid {
        name: "sigmoid",
        support: ElementWise,
        family: Activation,
        elementwise: Unary,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: floating_unary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: unary(sigmoid),
        onnx: Some("Sigmoid"),
    },
    /// Normalizes exponentials along a selected dimension.
    Softmax {
        name: "softmax",
        support: Softmax,
        family: Activation,
        elementwise: None,
        attributes: Dimension,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: softmax,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: value(softmax, dim: i64),
        onnx: None,
    },
    /// Computes logarithms of softmax probabilities along a selected dimension.
    LogSoftmax {
        name: "log_softmax",
        support: Softmax,
        family: Activation,
        elementwise: None,
        attributes: Dimension,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: softmax,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: value(log_softmax, dim: i64),
        onnx: None,
    },
    /// Computes elementwise logical conjunction with broadcasting.
    LogicalAnd {
        name: "logical_and",
        support: Comparison,
        family: Comparison,
        elementwise: CommutativeBinary,
        attributes: None,
        operands: OperandSchema::Fixed(BINARY_OPERANDS),
        infer: promoted_binary,
        outputs: MASK_OUTPUT,
        alias: None,
        method: binary(logical_and),
        onnx: None,
    },
    /// Computes elementwise logical disjunction with broadcasting.
    LogicalOr {
        name: "logical_or",
        support: Comparison,
        family: Comparison,
        elementwise: CommutativeBinary,
        attributes: None,
        operands: OperandSchema::Fixed(BINARY_OPERANDS),
        infer: promoted_binary,
        outputs: MASK_OUTPUT,
        alias: None,
        method: binary(logical_or),
        onnx: None,
    },
    /// Computes elementwise logical negation.
    LogicalNot {
        name: "logical_not",
        support: Comparison,
        family: Unary,
        elementwise: Unary,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: logical_not,
        outputs: MASK_OUTPUT,
        alias: None,
        method: unary(logical_not),
        onnx: None,
    },
    /// Produces a boolean equality mask with broadcasting.
    Equal {
        name: "equal",
        support: Comparison,
        family: Comparison,
        elementwise: CommutativeBinary,
        attributes: None,
        operands: OperandSchema::Fixed(BINARY_OPERANDS),
        infer: promoted_binary,
        outputs: MASK_OUTPUT,
        alias: None,
        method: binary(equal),
        onnx: None,
    },
    /// Produces a boolean inequality mask with broadcasting.
    NotEqual {
        name: "not_equal",
        support: Comparison,
        family: Comparison,
        elementwise: CommutativeBinary,
        attributes: None,
        operands: OperandSchema::Fixed(BINARY_OPERANDS),
        infer: promoted_binary,
        outputs: MASK_OUTPUT,
        alias: None,
        method: binary(not_equal),
        onnx: None,
    },
    /// Tests whether each left element is less than its right element.
    Less {
        name: "less",
        support: Comparison,
        family: Comparison,
        elementwise: Binary,
        attributes: None,
        operands: OperandSchema::Fixed(BINARY_OPERANDS),
        infer: promoted_binary,
        outputs: MASK_OUTPUT,
        alias: None,
        method: binary(less),
        onnx: None,
    },
    /// Tests whether each left element is less than or equal to its right element.
    LessEqual {
        name: "less_equal",
        support: Comparison,
        family: Comparison,
        elementwise: Binary,
        attributes: None,
        operands: OperandSchema::Fixed(BINARY_OPERANDS),
        infer: promoted_binary,
        outputs: MASK_OUTPUT,
        alias: None,
        method: binary(less_equal),
        onnx: None,
    },
    /// Tests whether each left element is greater than its right element.
    Greater {
        name: "greater",
        support: Comparison,
        family: Comparison,
        elementwise: Binary,
        attributes: None,
        operands: OperandSchema::Fixed(BINARY_OPERANDS),
        infer: promoted_binary,
        outputs: MASK_OUTPUT,
        alias: None,
        method: binary(greater),
        onnx: None,
    },
    /// Tests whether each left element is greater than or equal to its right element.
    GreaterEqual {
        name: "greater_equal",
        support: Comparison,
        family: Comparison,
        elementwise: Binary,
        attributes: None,
        operands: OperandSchema::Fixed(BINARY_OPERANDS),
        infer: promoted_binary,
        outputs: MASK_OUTPUT,
        alias: None,
        method: binary(greater_equal),
        onnx: None,
    },
    /// Computes elementwise modulo with broadcasting.
    Modulo {
        name: "modulo",
        support: ElementWise,
        family: NumericBinary,
        elementwise: Binary,
        attributes: None,
        operands: OperandSchema::Fixed(BINARY_OPERANDS),
        infer: promoted_binary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: binary(modulo),
        onnx: None,
    },
    /// Computes elementwise remainders with broadcasting.
    Remainder {
        name: "remainder",
        support: ElementWise,
        family: NumericBinary,
        elementwise: Binary,
        attributes: None,
        operands: OperandSchema::Fixed(BINARY_OPERANDS),
        infer: promoted_binary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: binary(remainder),
        onnx: None,
    },
    /// Computes the elementwise two-argument inverse tangent with broadcasting.
    Atan2 {
        name: "atan2",
        support: ElementWise,
        family: NumericBinary,
        elementwise: Binary,
        attributes: None,
        operands: OperandSchema::Fixed(BINARY_OPERANDS),
        infer: promoted_binary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: binary(atan2),
        onnx: None,
    },
    /// Selects elementwise maximum values with broadcasting.
    Maximum {
        name: "maximum",
        support: ElementWise,
        family: NumericBinary,
        elementwise: CommutativeBinary,
        attributes: None,
        operands: OperandSchema::Fixed(BINARY_OPERANDS),
        infer: promoted_binary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: binary(maximum),
        onnx: None,
    },
    /// Selects elementwise minimum values with broadcasting.
    Minimum {
        name: "minimum",
        support: ElementWise,
        family: NumericBinary,
        elementwise: CommutativeBinary,
        attributes: None,
        operands: OperandSchema::Fixed(BINARY_OPERANDS),
        infer: promoted_binary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: binary(minimum),
        onnx: None,
    },
    /// Sums elements over selected dimensions.
    Sum {
        name: "sum",
        support: Reduction,
        family: Reduction,
        elementwise: None,
        attributes: Reduction,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: reduction,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: reduction(sum),
        onnx: None,
    },
    /// Averages elements over selected dimensions.
    Mean {
        name: "mean",
        support: Reduction,
        family: Reduction,
        elementwise: None,
        attributes: Reduction,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: reduction,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: reduction(mean),
        onnx: None,
    },
    /// Reduces selected dimensions to their maximum values.
    Max {
        name: "max",
        support: Reduction,
        family: Reduction,
        elementwise: None,
        attributes: Reduction,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: reduction,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: reduction(max_reduce),
        onnx: None,
    },
    /// Reduces selected dimensions to their minimum values.
    Min {
        name: "min",
        support: Reduction,
        family: Reduction,
        elementwise: None,
        attributes: Reduction,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: reduction,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: reduction(min_reduce),
        onnx: None,
    },
    /// Multiplies elements over selected dimensions.
    Product {
        name: "product",
        support: Reduction,
        family: Reduction,
        elementwise: None,
        attributes: Reduction,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: reduction,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: reduction(product),
        onnx: None,
    },
    /// Computes the logarithm of the sum of the exponentials over dimensions.
    LogSumExp {
        name: "logsumexp",
        support: Reduction,
        family: Reduction,
        elementwise: None,
        attributes: Reduction,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: reduction,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: reduction(logsumexp),
        onnx: None,
    },
    /// Computes the conjunction of the operands' bits, or of their truth values for Booleans.
    BitwiseAnd {
        name: "bitwise_and",
        support: ElementWise,
        family: NumericBinary,
        elementwise: CommutativeBinary,
        attributes: None,
        operands: OperandSchema::Fixed(BINARY_OPERANDS),
        infer: promoted_binary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: binary(bitwise_and),
        onnx: None,
    },
    /// Computes the disjunction of the operands' bits, or of their truth values for Booleans.
    BitwiseOr {
        name: "bitwise_or",
        support: ElementWise,
        family: NumericBinary,
        elementwise: CommutativeBinary,
        attributes: None,
        operands: OperandSchema::Fixed(BINARY_OPERANDS),
        infer: promoted_binary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: binary(bitwise_or),
        onnx: None,
    },
    /// Computes the exclusive disjunction of the operands' bits, or of their truth values for Booleans.
    BitwiseXor {
        name: "bitwise_xor",
        support: ElementWise,
        family: NumericBinary,
        elementwise: CommutativeBinary,
        attributes: None,
        operands: OperandSchema::Fixed(BINARY_OPERANDS),
        infer: promoted_binary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: binary(bitwise_xor),
        onnx: None,
    },
    /// Complements the operand's bits, or negates its truth values for Booleans.
    BitwiseNot {
        name: "bitwise_not",
        support: ElementWise,
        family: NumericBinary,
        elementwise: Unary,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: bitwise_not,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: unary(bitwise_not),
        onnx: None,
    },
    /// Shifts integers left by the other operand's bits, giving zero for a shift outside the type's width.
    BitwiseLeftShift {
        name: "bitwise_left_shift",
        support: ElementWise,
        family: NumericBinary,
        elementwise: Binary,
        attributes: None,
        operands: OperandSchema::Fixed(BINARY_OPERANDS),
        infer: promoted_binary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: binary(bitwise_left_shift),
        onnx: None,
    },
    /// Shifts integers right by the other operand's bits, arithmetically for signed types, giving the sign's fill for a shift outside the type's width.
    BitwiseRightShift {
        name: "bitwise_right_shift",
        support: ElementWise,
        family: NumericBinary,
        elementwise: Binary,
        attributes: None,
        operands: OperandSchema::Fixed(BINARY_OPERANDS),
        infer: promoted_binary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: binary(bitwise_right_shift),
        onnx: None,
    },
    /// Raises the operand to the other operand's power; integer powers wrap, and negative integer exponents give the integer part of the reciprocal.
    Pow {
        name: "pow",
        support: ElementWise,
        family: NumericBinary,
        elementwise: Binary,
        attributes: None,
        operands: OperandSchema::Fixed(BINARY_OPERANDS),
        infer: promoted_binary,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: binary(pow),
        onnx: None,
    },
    /// Raises the operand to a scalar power, promoting the operand as the scalar's category requires.
    PowScalar {
        name: "pow_scalar",
        support: ElementWise,
        family: NumericBinary,
        elementwise: Scalar,
        attributes: Scalar,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: scalar,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Adds a retained scalar to every input element.
    AddScalar {
        name: "add_scalar",
        support: ElementWise,
        family: NumericOther,
        elementwise: Scalar,
        attributes: Scalar,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: scalar,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Subtracts a retained scalar from every input element.
    SubScalar {
        name: "sub_scalar",
        support: ElementWise,
        family: NumericOther,
        elementwise: Scalar,
        attributes: Scalar,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: scalar,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Subtracts every input element from a retained scalar.
    RSubScalar {
        name: "rsub_scalar",
        support: ElementWise,
        family: NumericOther,
        elementwise: Scalar,
        attributes: Scalar,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: scalar,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Multiplies every input element by a retained scalar.
    Scale {
        name: "scale",
        support: ElementWise,
        family: NumericOther,
        elementwise: Scalar,
        attributes: Scalar,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: scalar,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Divides a retained scalar by every input element.
    RDivScalar {
        name: "rdiv_scalar",
        support: ElementWise,
        family: NumericOther,
        elementwise: Scalar,
        attributes: Scalar,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: reverse_divide_scalar,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Pads tensor axes according to retained padding widths and fill parameters.
    Pad {
        name: "pad",
        support: DecomposedInGraphs,
        family: Shape,
        elementwise: None,
        attributes: Pad,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: pad,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Swaps two tensor axes as a logical view.
    Transpose {
        name: "transpose",
        support: AliasAnyStrided,
        family: Shape,
        elementwise: None,
        attributes: Transpose,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: transpose,
        outputs: VALUE_OUTPUT,
        alias: View,
        method: none,
        onnx: None,
    },
    /// Reorders tensor axes as a logical view.
    Permute {
        name: "permute",
        support: AliasAnyStrided,
        family: Shape,
        elementwise: None,
        attributes: Permute,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: permute,
        outputs: VALUE_OUTPUT,
        alias: View,
        method: none,
        onnx: None,
    },
    /// Changes logical dimensions while preserving the element count.
    Reshape {
        name: "reshape",
        support: AliasAnyStrided,
        family: Shape,
        elementwise: None,
        attributes: Shape,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: reshape,
        outputs: VALUE_OUTPUT,
        alias: View,
        method: none,
        onnx: None,
    },
    /// Removes a selected unit-sized axis.
    Squeeze {
        name: "squeeze",
        support: AliasAnyStrided,
        family: Shape,
        elementwise: None,
        attributes: Dimension,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: squeeze,
        outputs: VALUE_OUTPUT,
        alias: View,
        method: value(squeeze, dim: i64),
        onnx: None,
    },
    /// Inserts a unit-sized axis.
    Unsqueeze {
        name: "unsqueeze",
        support: AliasAnyStrided,
        family: Shape,
        elementwise: None,
        attributes: Dimension,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: unsqueeze,
        outputs: VALUE_OUTPUT,
        alias: View,
        method: value(unsqueeze, dim: i64),
        onnx: None,
    },
    /// Selects a range along a tensor axis as a logical view.
    Slice {
        name: "slice",
        support: AliasAnyStrided,
        family: Shape,
        elementwise: None,
        attributes: Slice,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: slice,
        outputs: VALUE_OUTPUT,
        alias: View,
        method: none,
        onnx: None,
    },
    /// Copies logical tensor values into independent storage.
    Copy {
        name: "copy",
        support: ContiguousAlias,
        family: Shape,
        elementwise: Unary,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: same_as_input,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Produces contiguous tensor storage, allowing an existing compatible allocation to be reused.
    Contiguous {
        name: "contiguous",
        support: ContiguousAlias,
        family: Shape,
        elementwise: None,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: same_as_input,
        outputs: VALUE_OUTPUT,
        alias: MayAliasInput,
        method: none,
        onnx: None,
    },
    /// Joins an ordered list of tensors along one dimension.
    Concat {
        name: "concat",
        support: Materialization,
        family: Shape,
        elementwise: None,
        attributes: Concat,
        operands: OperandSchema::Variadic { leading: &[], role: OperandRole::Input, min: 1 },
        infer: concat,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Selects values along one axis using an index tensor.
    IndexSelect {
        name: "index_select",
        support: Indexing,
        family: Shape,
        elementwise: None,
        attributes: IndexSelect,
        operands: OperandSchema::Fixed(INDEX_SELECT_OPERANDS),
        infer: index_select,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: indexed(index_select),
        onnx: None,
    },
    /// Selects elements of consecutive axes with broadcast index tensors, one per indexed axis.
    Index {
        name: "index",
        support: AllDataTypes,
        family: Shape,
        elementwise: None,
        attributes: Dimension,
        operands: OperandSchema::Variadic { leading: INDEX_LEADING_OPERANDS, role: OperandRole::Indices, min: 1 },
        infer: index,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Returns a copy of the input with values written into, or added to, the elements `index` selects.
    IndexPut {
        name: "index_put",
        support: AllDataTypes,
        family: Shape,
        elementwise: None,
        attributes: IndexPut,
        operands: OperandSchema::Variadic { leading: INDEX_PUT_LEADING_OPERANDS, role: OperandRole::Indices, min: 1 },
        infer: index_put,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Writes values into, or adds them to, the elements `index` selects of the first operand, where they are, as an eager `index_put_` does.
    IndexWrite {
        name: "index_write",
        support: AllDataTypesEagerOnly,
        family: Shape,
        elementwise: None,
        attributes: IndexPut,
        operands: OperandSchema::Variadic { leading: INDEX_WRITE_LEADING_OPERANDS, role: OperandRole::Indices, min: 1 },
        infer: index_write,
        outputs: NO_OUTPUTS,
        alias: WritesInput,
        method: none,
        onnx: None,
    },
    /// Gathers values along a selected axis using an index tensor.
    Take {
        name: "take",
        support: Indexing,
        family: Shape,
        elementwise: None,
        attributes: IndexSelect,
        operands: OperandSchema::Fixed(INDEX_SELECT_OPERANDS),
        infer: take,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: indexed(take),
        onnx: None,
    },
    /// Repeats adjacent key/value attention heads.
    RepeatKv {
        name: "repeat_kv",
        support: Materialization,
        family: Shape,
        elementwise: None,
        attributes: RepeatKv,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: repeat_kv,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: value(repeat_kv, repeats: i64),
        onnx: None,
    },
    /// Contracts operands by mode labels and combines the result with a scaled addend.
    Contraction {
        name: "contraction",
        support: Contraction,
        family: TensorAlgebra,
        elementwise: None,
        attributes: Contraction,
        operands: OperandSchema::Fixed(CONTRACTION_OPERANDS),
        infer: contraction,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Looks up embedding rows using an index tensor.
    Embedding {
        name: "embedding",
        support: Indexing,
        family: Neural,
        elementwise: None,
        attributes: None,
        operands: OperandSchema::Fixed(EMBEDDING_OPERANDS),
        infer: embedding,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Performs grouped two-dimensional convolution with an optional bias.
    Conv2d {
        name: "conv2d",
        support: ConvolutionCudnn,
        family: Neural,
        elementwise: None,
        attributes: Conv2d,
        operands: OperandSchema::TrailingOptional { required: CONV2D_OPERANDS, optional: OperandRole::Bias },
        infer: conv2d,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Convolves three spatial axes, with an optional bias.
    Conv3d {
        name: "conv3d",
        support: Convolution3d,
        family: Neural,
        elementwise: None,
        attributes: Conv3d,
        operands: OperandSchema::TrailingOptional { required: CONV2D_OPERANDS, optional: OperandRole::Bias },
        infer: conv3d,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Reduces two-dimensional spatial windows to maximum values.
    MaxPool2d {
        name: "max_pool2d",
        support: Pooling,
        family: Neural,
        elementwise: None,
        attributes: Pool2d,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: max_pool2d,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Averages two-dimensional spatial windows with a selected padding divisor policy.
    AvgPool2d {
        name: "avg_pool2d",
        support: Pooling,
        family: Neural,
        elementwise: None,
        attributes: Pool2d,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: avg_pool2d,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Normalizes channels using current-batch or retained running statistics and optional affine operands.
    BatchNorm {
        name: "batch_norm",
        support: Normalization,
        family: Neural,
        elementwise: None,
        attributes: BatchNormalization,
        operands: OperandSchema::Fixed(BATCH_NORM_OPERANDS),
        infer: batch_norm,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Normalizes trailing dimensions using their mean and variance, with optional affine operands.
    LayerNorm {
        name: "layer_norm",
        support: Normalization,
        family: Neural,
        elementwise: None,
        attributes: LayerNormalization,
        operands: OperandSchema::Fixed(LAYER_NORM_OPERANDS),
        infer: layer_norm,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Normalizes trailing dimensions by their root mean square, with optional affine operands.
    RmsNorm {
        name: "rms_norm",
        support: Normalization,
        family: Neural,
        elementwise: None,
        attributes: LayerNormalization,
        operands: OperandSchema::Fixed(LAYER_NORM_OPERANDS),
        infer: layer_norm,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Transforms the last axis's real signal into its `n / 2 + 1` frequencies, as `[real, imaginary]` pairs along a trailing axis of two.
    Rfft {
        name: "rfft",
        support: Fourier,
        family: Neural,
        elementwise: None,
        attributes: None,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: rfft,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Transforms `[..., k, 2]` frequencies into a real signal of the attribute's length.
    Irfft {
        name: "irfft",
        support: Fourier,
        family: Neural,
        elementwise: None,
        attributes: Length,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: irfft,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Rotates embedding components using supplied cosine and sine tensors.
    RotaryEmbedding {
        name: "rotary_embedding",
        support: DecomposedInGraphs,
        family: Neural,
        elementwise: None,
        attributes: None,
        operands: OperandSchema::Fixed(ROTARY_EMBEDDING_OPERANDS),
        infer: rotary_embedding,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Computes scaled query-key attention, optional causal masking, and weighted values.
    ScaledDotProductAttention {
        name: "scaled_dot_product_attention",
        support: Attention,
        family: Neural,
        elementwise: None,
        attributes: ScaledDotProductAttention,
        operands: OperandSchema::Variadic { leading: SCALED_DOT_PRODUCT_ATTENTION_OPERANDS, role: OperandRole::Bias, min: 0 },
        infer: attention,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Attends ragged queries over a paged key and value cache, as `FlashInfer` batches decode and prefill.
    PagedAttention {
        name: "paged_attention",
        support: PagedAttention,
        family: Neural,
        elementwise: None,
        attributes: ScaledDotProductAttention,
        operands: OperandSchema::Fixed(PAGED_ATTENTION_OPERANDS),
        infer: paged_attention,
        outputs: ATTENTION_LSE_OUTPUTS,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Returns paged attention's output alone, as graph capture records the operation's first result.
    PagedAttentionOutput {
        name: "paged_attention_output",
        support: PagedAttention,
        family: Neural,
        elementwise: None,
        attributes: ScaledDotProductAttention,
        operands: OperandSchema::Fixed(PAGED_ATTENTION_OPERANDS),
        infer: paged_attention_output,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Returns paged attention's log-sum-exp alone, as graph capture records the operation's second result.
    PagedAttentionLse {
        name: "paged_attention_lse",
        support: PagedAttention,
        family: Neural,
        elementwise: None,
        attributes: ScaledDotProductAttention,
        operands: OperandSchema::Fixed(PAGED_ATTENTION_OPERANDS),
        infer: paged_attention_lse,
        outputs: LSE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Multiplies rows sorted by group, as mixture-of-experts layers sort tokens by expert, each by its group's weight.
    GroupedMatMul {
        name: "grouped_matmul",
        support: GroupedMatMul,
        family: Neural,
        elementwise: None,
        attributes: None,
        operands: OperandSchema::Fixed(GROUPED_MATMUL_OPERANDS),
        infer: grouped_matmul,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Attends `DeepSeek`'s absorbed latent queries over a paged latent cache, returning the output and each row's base-two log-sum-exp.
    PagedMlaAttention {
        name: "paged_mla_attention",
        support: PagedMlaAttention,
        family: Neural,
        elementwise: None,
        attributes: ScaledDotProductAttention,
        operands: OperandSchema::Fixed(PAGED_MLA_ATTENTION_OPERANDS),
        infer: paged_mla_attention,
        outputs: ATTENTION_LSE_OUTPUTS,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Returns latent attention's output alone, as graph capture records the operation's first result.
    PagedMlaAttentionOutput {
        name: "paged_mla_attention_output",
        support: PagedMlaAttention,
        family: Neural,
        elementwise: None,
        attributes: ScaledDotProductAttention,
        operands: OperandSchema::Fixed(PAGED_MLA_ATTENTION_OPERANDS),
        infer: paged_mla_attention_output,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Returns latent attention's log-sum-exp alone, as graph capture records the operation's second result.
    PagedMlaAttentionLse {
        name: "paged_mla_attention_lse",
        support: PagedMlaAttention,
        family: Neural,
        elementwise: None,
        attributes: ScaledDotProductAttention,
        operands: OperandSchema::Fixed(PAGED_MLA_ATTENTION_OPERANDS),
        infer: paged_mla_attention_lse,
        outputs: LSE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Combines a selected range of axes into one logical dimension.
    Flatten {
        name: "flatten",
        support: AliasAnyStrided,
        family: Shape,
        elementwise: None,
        attributes: Flatten,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: flatten,
        outputs: VALUE_OUTPUT,
        alias: View,
        method: none,
        onnx: None,
    },
    /// Converts tensor values to a selected data type.
    Cast {
        name: "cast",
        support: Cast,
        family: Shape,
        elementwise: None,
        attributes: Cast,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: cast,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: value(cast, data_type: DataType),
        onnx: None,
    },
    /// Converts floating values to a quantized representation using a scale and zero point.
    Quantize {
        name: "quantize",
        support: Quantize,
        family: Shape,
        elementwise: None,
        attributes: Quantization,
        operands: OperandSchema::Fixed(QUANTIZE_OPERANDS),
        infer: quantization,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Converts quantized values using their scale and zero point.
    Dequantize {
        name: "dequantize",
        support: Dequantize,
        family: Shape,
        elementwise: None,
        attributes: Quantization,
        operands: OperandSchema::Fixed(QUANTIZE_OPERANDS),
        infer: quantization,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Multiplies FP8 matrices scaled by `F32` factors into a wider type, as `torch._scaled_mm` does.
    ScaledMatMul {
        name: "scaled_mm",
        support: ScaledMatMul,
        family: Binary,
        elementwise: None,
        attributes: Cast,
        operands: OperandSchema::Fixed(SCALED_MATMUL_OPERANDS),
        infer: scaled_matmul,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Multiplies quantized matrix operands using their respective scales and zero points.
    QuantizedMatMul {
        name: "quantized_matmul",
        support: QuantizedMatMul,
        family: Binary,
        elementwise: None,
        attributes: None,
        operands: OperandSchema::Fixed(QUANTIZED_MATMUL_OPERANDS),
        infer: quantized_matmul,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Clips every element to retained lower and upper bounds.
    Clamp {
        name: "clamp",
        support: ElementWise,
        family: NumericOther,
        elementwise: None,
        attributes: Clamp,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: clamp,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: fields(clamp, ClampAttributes { min: f32, max: f32 }),
        onnx: None,
    },
    /// Selects between two broadcast operands using a boolean condition.
    Where {
        name: "where",
        support: ElementWise,
        family: NumericOther,
        elementwise: None,
        attributes: None,
        operands: OperandSchema::Fixed(TERNARY_WHERE_OPERANDS),
        infer: where_select,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Returns indices of maximum values along a selected dimension.
    ArgMax {
        name: "argmax",
        support: ArgReduceMaterialized,
        family: ArgReduction,
        elementwise: None,
        attributes: Dimension,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: arg_reduction,
        outputs: INDEX_OUTPUT,
        alias: None,
        method: value(argmax, dim: i64),
        onnx: None,
    },
    /// Returns indices of minimum values along a selected dimension.
    ArgMin {
        name: "argmin",
        support: ArgReduceMaterialized,
        family: ArgReduction,
        elementwise: None,
        attributes: Dimension,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: arg_reduction,
        outputs: INDEX_OUTPUT,
        alias: None,
        method: value(argmin, dim: i64),
        onnx: None,
    },
    /// Returns the positions that stably sort each slice along a dimension, ascending or descending, with NaN largest.
    Argsort {
        name: "argsort",
        support: Sort,
        family: ArgReduction,
        elementwise: None,
        attributes: Sort,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: argsort,
        outputs: INDEX_OUTPUT,
        alias: None,
        method: fields(argsort, SortAttributes { dim: i64, descending: bool }),
        onnx: None,
    },
    /// Accumulates running sums along a dimension.
    Cumsum {
        name: "cumsum",
        support: Scan,
        family: Reduction,
        elementwise: None,
        attributes: Dimension,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: cumulative,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: value(cumsum, dim: i64),
        onnx: None,
    },
    /// Accumulates running products along a dimension.
    Cumprod {
        name: "cumprod",
        support: Scan,
        family: Reduction,
        elementwise: None,
        attributes: Dimension,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: cumulative,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: value(cumprod, dim: i64),
        onnx: None,
    },
    /// Projects the selected values from a top-k operation.
    TopKValues {
        name: "topk_values",
        support: TopKValues,
        family: ArgReduction,
        elementwise: None,
        attributes: TopK,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: topk,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Selects top-k values and their indices along a dimension.
    TopK {
        name: "topk",
        support: TopKValues,
        family: ArgReduction,
        elementwise: None,
        attributes: TopK,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: topk,
        outputs: TOPK_OUTPUTS,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Projects the selected indices from a top-k operation.
    TopKIndices {
        name: "topk_indices",
        support: ArgReduceMaterialized,
        family: ArgReduction,
        elementwise: None,
        attributes: TopK,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: topk,
        outputs: INDEX_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Broadcasts immutable tensor storage through logical zero-stride axes.
    Expand {
        name: "expand",
        support: AliasAnyStrided,
        family: Shape,
        elementwise: None,
        attributes: Shape,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: expand,
        outputs: VALUE_OUTPUT,
        alias: View,
        method: none,
        onnx: None,
    },
    /// Computes the first operand plus a retained weight times the difference between the second and first operands.
    Lerp {
        name: "lerp",
        support: ElementWise,
        family: NumericOther,
        elementwise: None,
        attributes: Lerp,
        operands: OperandSchema::Fixed(LERP_OPERANDS),
        infer: lerp,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: paired(lerp, t: f32),
        onnx: None,
    },
    /// Writes the second operand into the elements the first operand addresses.
    Assign {
        name: "assign",
        support: AllDataTypesEagerOnly,
        family: Shape,
        elementwise: None,
        attributes: None,
        operands: OperandSchema::Fixed(ASSIGN_OPERANDS),
        infer: assign,
        outputs: NO_OUTPUTS,
        alias: WritesInput,
        method: none,
        onnx: None,
    },
    /// Views elements of a flat operand through explicit dimensions, strides, and offset.
    AsStrided {
        name: "as_strided",
        support: AliasAnyStrided,
        family: Shape,
        elementwise: None,
        attributes: Strided,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: as_strided,
        outputs: VALUE_OUTPUT,
        alias: View,
        method: none,
        onnx: None,
    },
    /// Returns a copy of a flat operand with the source written at strided positions.
    StridedScatter {
        name: "strided_scatter",
        support: AllDataTypes,
        family: Shape,
        elementwise: None,
        attributes: Strided,
        operands: OperandSchema::Fixed(SCATTER_OPERANDS),
        infer: strided_scatter,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Samples uniform values in `[low, high)` from a generator.
    Uniform {
        name: "uniform",
        support: Random,
        family: NumericOther,
        elementwise: None,
        attributes: Random,
        operands: OperandSchema::Fixed(RANDOM_OPERANDS),
        infer: uniform,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Samples normal values with a mean and standard deviation from a generator.
    Normal {
        name: "normal",
        support: Random,
        family: NumericOther,
        elementwise: None,
        attributes: Random,
        operands: OperandSchema::Fixed(RANDOM_OPERANDS),
        infer: normal,
        outputs: VALUE_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Samples `true` with each element's probability.
    Bernoulli {
        name: "bernoulli",
        support: Random,
        family: NumericOther,
        elementwise: None,
        attributes: Random,
        operands: OperandSchema::Fixed(SAMPLING_OPERANDS),
        infer: bernoulli,
        outputs: MASK_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Samples category indices per row, with replacement, in proportion to nonnegative weights.
    Multinomial {
        name: "multinomial",
        support: Random,
        family: NumericOther,
        elementwise: None,
        attributes: Random,
        operands: OperandSchema::Fixed(SAMPLING_OPERANDS),
        infer: multinomial,
        outputs: INDEX_OUTPUT,
        alias: None,
        method: none,
        onnx: None,
    },
    /// Transfers values to a selected device, allowing aliases when no transfer is needed.
    CopyTo {
        name: "copy_to",
        support: AliasAnyStrided,
        family: CopyTo,
        elementwise: None,
        attributes: CopyTo,
        operands: OperandSchema::Fixed(UNARY_OPERANDS),
        infer: copy_to,
        outputs: VALUE_OUTPUT,
        alias: MayAliasInput,
        method: none,
        onnx: None,
    },
}

impl Operation {
    /// Returns the operation name used in diagnostics and support tables.
    #[must_use]
    pub fn name(self) -> &'static str {
        operation_name(self)
    }

    /// Returns the attribute category required by this registry operation.
    #[must_use]
    pub fn attribute_kind(self) -> OperationAttributeKind {
        operation_attribute_kind(self)
    }

    /// Returns whether this operation requires the supplied attribute category.
    #[must_use]
    pub fn has_attribute_kind(self, kind: OperationAttributeKind) -> bool {
        self.attribute_kind() == kind
    }

    /// Returns whether the registry classifies this operation as unary elementwise, including idempotent operations.
    #[must_use]
    pub fn is_unary_elementwise(self) -> bool {
        operation_elementwise_kind(self).is_unary()
    }

    /// Returns whether the registry classifies this operation as binary elementwise, including commutative operations.
    #[must_use]
    pub fn is_binary_elementwise(self) -> bool {
        operation_elementwise_kind(self).is_binary()
    }

    /// Returns whether the registry classifies this operation as elementwise with a scalar argument.
    #[must_use]
    pub fn is_scalar_elementwise(self) -> bool {
        operation_elementwise_kind(self).is_scalar()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationElementwiseKind {
    None,
    Unary,
    IdempotentUnary,
    Binary,
    CommutativeBinary,
    Scalar,
}

impl OperationElementwiseKind {
    fn is_unary(self) -> bool {
        matches!(self, Self::Unary | Self::IdempotentUnary)
    }

    fn is_binary(self) -> bool {
        matches!(self, Self::Binary | Self::CommutativeBinary)
    }

    fn is_scalar(self) -> bool {
        matches!(self, Self::Scalar)
    }
}
