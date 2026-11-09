//! The kernel IR: what one kernel computes, independently of how a target runs it.
//!
//! A [`Kernel`] has operands, sizes, scalars, axes, an expression graph of [`Node`]s, and the [`Store`]s that write its outputs.
//!
//! - A size is an extent, stride, or offset, either written into the kernel's code ([`Size::Static`]) or passed at launch ([`Size::Argument`]), so one compiled kernel serves every value of an argument size.
//! - An axis iterates `0..extent`.
//!   Parallel axes are independent of each other and are never reduced; a target spreads them across programs.
//!   Reduction axes are walked inside one program, and only they may be reduced.
//! - Every node has a domain: the axes its value depends on.
//!   A load depends on the axes its access uses, an operation on its operands' axes, and a reduction on its operand's axes except the reduced ones.
//! - A store runs once for each point of the axes its access uses, and the value it stores may depend only on those axes.
//! - An access may add integer node values times strides to its offset, which gathers and scatters through index tensors; it then uses the axes those nodes depend on too.
//!   [`Expr::CheckIndex`] bounds such a value first, reporting a status code for an index out of range, so no access leaves its operand.
//!
//! Loads of `F16`, `BF16`, and the 8-bit floating types produce `F32` values and stores round back, so arithmetic always happens in a node's stated [`ElementType`].

mod text;
mod verify;

pub use self::verify::VerifyError;

/// The type of an operand's elements or of a computed value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ElementType {
    Bool,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F16,
    Bf16,
    F32,
    F64,
    /// `PyTorch`'s `float8_e4m3fn`: four exponent bits, three mantissa bits, and no infinities.
    F8E4M3,
    /// `PyTorch`'s `float8_e5m2`: five exponent bits, two mantissa bits, and IEEE 754 infinities.
    F8E5M2,
}

impl ElementType {
    /// Returns the size of one element in bytes.
    #[must_use]
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 | Self::F8E4M3 | Self::F8E5M2 => 1,
            Self::U16 | Self::I16 | Self::F16 | Self::Bf16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }

    /// Returns whether the type is a floating-point type.
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(
            self,
            Self::F16 | Self::Bf16 | Self::F32 | Self::F64 | Self::F8E4M3 | Self::F8E5M2
        )
    }

    /// Returns whether the type is a signed or unsigned integer type.
    #[must_use]
    pub const fn is_integer(self) -> bool {
        !self.is_float() && !matches!(self, Self::Bool)
    }

    /// Returns the type values of this storage type are computed in: `F32` for `F16`, `BF16`, and the 8-bit floating types, and the type itself otherwise.
    #[must_use]
    pub const fn compute(self) -> Self {
        match self {
            Self::F16 | Self::Bf16 | Self::F8E4M3 | Self::F8E5M2 => Self::F32,
            other => other,
        }
    }

    /// Returns the lowercase name the canonical text uses.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::U8 => "u8",
            Self::I8 => "i8",
            Self::U16 => "u16",
            Self::I16 => "i16",
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::U64 => "u64",
            Self::I64 => "i64",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::F8E4M3 => "f8e4m3",
            Self::F8E5M2 => "f8e5m2",
        }
    }
}

/// Identifies an operand of a kernel, in declaration order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OperandId(pub usize);

/// Identifies a size of a kernel, in declaration order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SizeId(pub usize);

/// Identifies a scalar of a kernel, in declaration order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ScalarId(pub usize);

/// Identifies an axis of a kernel, in declaration order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AxisId(pub usize);

/// Identifies a node of a kernel's expression graph, in creation order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub usize);

/// Whether a kernel reads or writes an operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperandRole {
    Input,
    Output,
}

/// A buffer the kernel reads or writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Operand {
    pub element: ElementType,
    pub role: OperandRole,
}

/// An extent, stride, or offset, in elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Size {
    /// A value written into the kernel's code.
    Static(u64),
    /// A value passed at launch, so one compiled kernel serves every value.
    Argument,
}

/// Whether an axis is spread across programs or walked inside one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AxisKind {
    /// Independent iterations that are never reduced.
    Parallel,
    /// Iterations walked inside one program, which reductions may combine.
    Reduction,
}

/// An axis iterating `0..extent`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Axis {
    pub extent: SizeId,
    pub kind: AxisKind,
}

/// The element of an operand at offset `base + Σ axis × stride + Σ index × stride`, in elements, where each index is the value of an unsigned integer node.
///
/// Strided views, broadcasts (a stride of zero, or an axis left out), transposes, slices, and concatenation offsets are all accesses, and indexed terms gather or scatter through index tensors.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Access {
    pub operand: OperandId,
    pub base: Option<SizeId>,
    pub strides: Vec<(AxisId, SizeId)>,
    /// Nodes whose values, times their strides, add to the offset.
    pub indirect: Vec<(NodeId, SizeId)>,
}

impl Access {
    /// Returns the axes whose indices the access's strides multiply; [`Kernel::access_domain`] adds the axes its indexed terms depend on.
    pub fn axes(&self) -> impl Iterator<Item = AxisId> + '_ {
        self.strides.iter().map(|&(axis, _)| axis)
    }
}

/// A constant value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Literal {
    Bool(bool),
    Int(i64),
    Float(f64),
}

/// An operation on one value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Neg,
    Abs,
    Exp,
    Log,
    Sqrt,
    Rsqrt,
    Reciprocal,
    Sin,
    Cos,
    Tan,
    Tanh,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Asinh,
    Acosh,
    Atanh,
    Erf,
    // `log(1 + x)`, exact for small `x`.
    Log1p,
    // `exp(x) − 1`, exact for small `x`.
    Expm1,
    Floor,
    Ceil,
    Trunc,
    /// The nearest integer, ties to even.
    Round,
    /// Logical negation of a `bool`.
    Not,
    /// Whether a floating-point value is NaN, as a `bool`.
    IsNan,
    /// The complement of an integer's bits.
    BitNot,
}

impl UnaryOp {
    /// Returns the lowercase name the canonical text uses.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Neg => "neg",
            Self::Abs => "abs",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Sqrt => "sqrt",
            Self::Rsqrt => "rsqrt",
            Self::Reciprocal => "reciprocal",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tan => "tan",
            Self::Tanh => "tanh",
            Self::Asin => "asin",
            Self::Acos => "acos",
            Self::Atan => "atan",
            Self::Sinh => "sinh",
            Self::Cosh => "cosh",
            Self::Asinh => "asinh",
            Self::Acosh => "acosh",
            Self::Atanh => "atanh",
            Self::Erf => "erf",
            Self::Log1p => "log1p",
            Self::Expm1 => "expm1",
            Self::Floor => "floor",
            Self::Ceil => "ceil",
            Self::Trunc => "trunc",
            Self::Round => "round",
            Self::Not => "not",
            Self::BitNot => "bitnot",
            Self::IsNan => "is_nan",
        }
    }

    /// Returns whether the operation produces a `bool` regardless of its operand's type.
    #[must_use]
    pub const fn is_predicate(self) -> bool {
        matches!(self, Self::IsNan)
    }
}

/// An operation on two values of one type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    /// Division, truncating toward zero for integers.
    Div,
    /// The remainder with the sign of the dividend, as C's `fmod` and `%`.
    Rem,
    Pow,
    /// The larger value, NaN if either is NaN.
    Max,
    /// The smaller value, NaN if either is NaN.
    Min,
    Atan2,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    /// Logical conjunction of `bool` values.
    And,
    /// Logical disjunction of `bool` values.
    Or,
    /// The conjunction of two integers' bits.
    BitAnd,
    /// The disjunction of two integers' bits.
    BitOr,
    /// The exclusive disjunction of two integers' bits.
    BitXor,
    /// An integer shifted left by the right operand's bits, as `PyTorch` shifts: zero for a shift outside `0..bits`.
    ShiftLeft,
    /// An integer shifted right by the right operand's bits, arithmetically for signed types, as `PyTorch` shifts: the sign's fill for a shift outside `0..bits`.
    ShiftRight,
}

impl BinaryOp {
    /// Returns the lowercase name the canonical text uses.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::Rem => "rem",
            Self::Pow => "pow",
            Self::Max => "max",
            Self::Min => "min",
            Self::Atan2 => "atan2",
            Self::Equal => "eq",
            Self::NotEqual => "ne",
            Self::Less => "lt",
            Self::LessEqual => "le",
            Self::Greater => "gt",
            Self::GreaterEqual => "ge",
            Self::And => "and",
            Self::Or => "or",
            Self::BitAnd => "bitand",
            Self::BitOr => "bitor",
            Self::BitXor => "bitxor",
            Self::ShiftLeft => "shl",
            Self::ShiftRight => "shr",
        }
    }

    /// Returns whether the operation compares its operands and produces a `bool`.
    #[must_use]
    pub const fn is_comparison(self) -> bool {
        matches!(
            self,
            Self::Equal
                | Self::NotEqual
                | Self::Less
                | Self::LessEqual
                | Self::Greater
                | Self::GreaterEqual
        )
    }
}

/// How a reduction combines values; every combiner is associative.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Combiner {
    Sum,
    Product,
    /// The largest value, NaN if any is NaN.
    Max,
    /// The smallest value, NaN if any is NaN.
    Min,
    /// The logarithm of the sum of the values' exponentials, of floating-point values, combined as `max + log(1 + exp(min - max))` so no exponential overflows.
    LogSumExp,
}

impl Combiner {
    /// Returns the lowercase name the canonical text uses.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Product => "product",
            Self::Max => "max",
            Self::Min => "min",
            Self::LogSumExp => "log_sum_exp",
        }
    }
}

/// What a node computes.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// The element an access addresses, widened to its compute type.
    Load(Access),
    Literal(Literal),
    Scalar(ScalarId),
    /// The current index along an axis.
    AxisIndex(AxisId),
    Unary(UnaryOp, NodeId),
    Binary(BinaryOp, NodeId, NodeId),
    /// `a × b + c` with one rounding, for floating-point values of one type.
    MulAdd(NodeId, NodeId, NodeId),
    Select {
        condition: NodeId,
        if_true: NodeId,
        if_false: NodeId,
    },
    /// The operand converted to the node's type, rounding to nearest even and saturating integers.
    Cast(NodeId),
    /// The operand combined over `axes`, which must all be reduction axes.
    Reduce {
        combiner: Combiner,
        axes: Vec<AxisId>,
        value: NodeId,
    },
    /// `value` combined over the points of reduction axis `axis` up to and including the current one, an inclusive scan, typed as `value`; only stores read it, in a kernel with no other reduction axis.
    Scan {
        combiner: Combiner,
        axis: AxisId,
        value: NodeId,
    },
    /// The `I64` position of `value`'s largest point over `axes`, or its smallest without `maximum`, counting the points in row-major order over `axes`; the first of equal values wins, and NaN ranks above every number.
    ArgReduce {
        maximum: bool,
        axes: Vec<AxisId>,
        value: NodeId,
    },
    /// The sum over `axes` of `value` weighted by the softmax of `logit` over them, for floating-point operands.
    /// Targets compute it in one pass, keeping a running maximum of the logits and the sums of the exponentials and of the weighted values, both rescaled whenever the maximum grows.
    SoftmaxSum {
        axes: Vec<AxisId>,
        logit: NodeId,
        value: NodeId,
    },
    /// The signed integer `value` as an index into `0..extent`, as a `U64`.
    /// A negative value counts from the end when `wrap` holds; a value still out of range reports `status` to the kernel's status word and becomes zero.
    CheckIndex {
        value: NodeId,
        extent: SizeId,
        wrap: bool,
        status: u32,
    },
    /// `value`, reporting `status` to the kernel's status word where the `bool` `condition` is false.
    Require {
        value: NodeId,
        condition: NodeId,
        status: u32,
    },
}

impl Expr {
    // The nodes this expression reads, including the indexed terms of a load.
    #[must_use]
    pub fn inputs(&self) -> Vec<NodeId> {
        match self {
            Self::Load(access) => access.indirect.iter().map(|&(node, _)| node).collect(),
            Self::Literal(_) | Self::Scalar(_) | Self::AxisIndex(_) => Vec::new(),
            Self::Unary(_, value)
            | Self::Cast(value)
            | Self::Reduce { value, .. }
            | Self::Scan { value, .. }
            | Self::ArgReduce { value, .. }
            | Self::CheckIndex { value, .. } => vec![*value],
            Self::Binary(_, lhs, rhs) => vec![*lhs, *rhs],
            Self::MulAdd(a, b, c) => vec![*a, *b, *c],
            Self::Select {
                condition,
                if_true,
                if_false,
            } => vec![*condition, *if_true, *if_false],
            Self::SoftmaxSum { logit, value, .. } => vec![*logit, *value],
            Self::Require {
                value, condition, ..
            } => vec![*value, *condition],
        }
    }
}

/// A value in the expression graph and its type.
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub element: ElementType,
    pub expr: Expr,
}

/// A write of `value` to the element `access` addresses, at every point of the access's domain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Store {
    pub access: Access,
    pub value: NodeId,
    /// Combines `value` into the element already there instead of replacing it, atomically, so stores that address one element all count; only [`Combiner::Sum`] is supported.
    pub combiner: Option<Combiner>,
}

/// One kernel: its operands, sizes, scalars, axes, expression graph, and stores.
///
/// Build a kernel with the `add_*` methods and [`Self::store`], then check it with [`Self::verify`].
#[derive(Debug, Clone, PartialEq)]
pub struct Kernel {
    /// The function name targets give the kernel, as profilers report it.
    pub name: String,
    pub operands: Vec<Operand>,
    pub sizes: Vec<Size>,
    pub scalars: Vec<ElementType>,
    pub axes: Vec<Axis>,
    pub nodes: Vec<Node>,
    pub stores: Vec<Store>,
}

impl Kernel {
    /// Starts an empty kernel named `name`.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            operands: Vec::new(),
            sizes: Vec::new(),
            scalars: Vec::new(),
            axes: Vec::new(),
            nodes: Vec::new(),
            stores: Vec::new(),
        }
    }

    pub fn add_operand(&mut self, element: ElementType, role: OperandRole) -> OperandId {
        self.operands.push(Operand { element, role });
        OperandId(self.operands.len() - 1)
    }

    pub fn add_size(&mut self, size: Size) -> SizeId {
        self.sizes.push(size);
        SizeId(self.sizes.len() - 1)
    }

    pub fn add_scalar(&mut self, element: ElementType) -> ScalarId {
        self.scalars.push(element);
        ScalarId(self.scalars.len() - 1)
    }

    pub fn add_axis(&mut self, extent: SizeId, kind: AxisKind) -> AxisId {
        self.axes.push(Axis { extent, kind });
        AxisId(self.axes.len() - 1)
    }

    /// Adds a node computing `expr` as `element`.
    pub fn add_node(&mut self, element: ElementType, expr: Expr) -> NodeId {
        self.nodes.push(Node { element, expr });
        NodeId(self.nodes.len() - 1)
    }

    /// Adds a load of the element `access` addresses, typed as its operand's compute type.
    ///
    /// # Panics
    ///
    /// Panics if `access` names an operand the kernel does not have.
    pub fn load(&mut self, access: Access) -> NodeId {
        // Loads of one input element share a node, so emitters read it once and row caches keep it between loops.
        if self.operands[access.operand.0].role == OperandRole::Input
            && let Some(index) = self
                .nodes
                .iter()
                .position(|node| matches!(&node.expr, Expr::Load(known) if *known == access))
        {
            return NodeId(index);
        }

        let element = self.operands[access.operand.0].element.compute();
        self.add_node(element, Expr::Load(access))
    }

    /// Adds a constant of type `element`.
    pub fn literal(&mut self, element: ElementType, literal: Literal) -> NodeId {
        self.add_node(element, Expr::Literal(literal))
    }

    /// Adds `op` applied to `value`, typed as `value` or as `bool` for a predicate.
    ///
    /// # Panics
    ///
    /// Panics if `value` is not a node of the kernel.
    pub fn unary(&mut self, op: UnaryOp, value: NodeId) -> NodeId {
        let element = if op.is_predicate() {
            ElementType::Bool
        } else {
            self.nodes[value.0].element
        };
        self.add_node(element, Expr::Unary(op, value))
    }

    /// Adds `op` applied to `lhs` and `rhs`, typed as `lhs` or as `bool` for a comparison.
    ///
    /// # Panics
    ///
    /// Panics if `lhs` is not a node of the kernel.
    pub fn binary(&mut self, op: BinaryOp, lhs: NodeId, rhs: NodeId) -> NodeId {
        let element = if op.is_comparison() {
            ElementType::Bool
        } else {
            self.nodes[lhs.0].element
        };
        self.add_node(element, Expr::Binary(op, lhs, rhs))
    }

    /// Adds `a × b + c` with one rounding, typed as `a`.
    ///
    /// # Panics
    ///
    /// Panics if `a` is not a node of the kernel.
    pub fn mul_add(&mut self, a: NodeId, b: NodeId, c: NodeId) -> NodeId {
        let element = self.nodes[a.0].element;
        self.add_node(element, Expr::MulAdd(a, b, c))
    }

    /// Adds `if_true` where `condition` holds and `if_false` elsewhere, typed as `if_true`.
    ///
    /// # Panics
    ///
    /// Panics if `if_true` is not a node of the kernel.
    pub fn select(&mut self, condition: NodeId, if_true: NodeId, if_false: NodeId) -> NodeId {
        let element = self.nodes[if_true.0].element;
        self.add_node(
            element,
            Expr::Select {
                condition,
                if_true,
                if_false,
            },
        )
    }

    /// Adds `value` converted to `element`.
    pub fn cast(&mut self, element: ElementType, value: NodeId) -> NodeId {
        self.add_node(element, Expr::Cast(value))
    }

    /// Adds `value` combined over `axes` with `combiner`, typed as `value`.
    ///
    /// # Panics
    ///
    /// Panics if `value` is not a node of the kernel.
    pub fn reduce(&mut self, combiner: Combiner, axes: Vec<AxisId>, value: NodeId) -> NodeId {
        let element = self.nodes[value.0].element;
        self.add_node(
            element,
            Expr::Reduce {
                combiner,
                axes,
                value,
            },
        )
    }

    /// Adds the sum over `axes` of `value` weighted by the softmax of `logit`, typed as `value`.
    ///
    /// # Panics
    ///
    /// Panics if `value` is not a node of the kernel.
    pub fn softmax_sum(&mut self, axes: Vec<AxisId>, logit: NodeId, value: NodeId) -> NodeId {
        let element = self.nodes[value.0].element;
        self.add_node(element, Expr::SoftmaxSum { axes, logit, value })
    }

    /// Adds the inclusive scan of `value` with `combiner` along `axis`.
    pub fn scan(&mut self, combiner: Combiner, axis: AxisId, value: NodeId) -> NodeId {
        let element = self.nodes[value.0].element;
        self.add_node(
            element,
            Expr::Scan {
                combiner,
                axis,
                value,
            },
        )
    }

    /// Adds the `I64` position of `value`'s largest point over `axes`, or its smallest without `maximum`.
    pub fn arg_reduce(&mut self, maximum: bool, axes: Vec<AxisId>, value: NodeId) -> NodeId {
        self.add_node(
            ElementType::I64,
            Expr::ArgReduce {
                maximum,
                axes,
                value,
            },
        )
    }

    /// Adds a store of `value` to the element `access` addresses.
    pub fn store(&mut self, access: Access, value: NodeId) {
        self.stores.push(Store {
            access,
            value,
            combiner: None,
        });
    }

    /// Adds a store that adds `value` to the element `access` addresses, atomically.
    pub fn store_sum(&mut self, access: Access, value: NodeId) {
        self.stores.push(Store {
            access,
            value,
            combiner: Some(Combiner::Sum),
        });
    }

    /// Adds `value`, a signed integer, checked as an index into `0..extent`; see [`Expr::CheckIndex`].
    pub fn check_index(
        &mut self,
        value: NodeId,
        extent: SizeId,
        wrap: bool,
        status: u32,
    ) -> NodeId {
        self.add_node(
            ElementType::U64,
            Expr::CheckIndex {
                value,
                extent,
                wrap,
                status,
            },
        )
    }

    /// Adds `value`, reporting `status` where `condition` is false; see [`Expr::Require`].
    ///
    /// # Panics
    ///
    /// Panics if `value` is not a node of the kernel.
    pub fn require(&mut self, value: NodeId, condition: NodeId, status: u32) -> NodeId {
        let element = self.nodes[value.0].element;
        self.add_node(
            element,
            Expr::Require {
                value,
                condition,
                status,
            },
        )
    }

    /// Returns whether the kernel reports status codes, so it takes a status word.
    #[must_use]
    pub fn reports_status(&self) -> bool {
        self.nodes
            .iter()
            .any(|node| matches!(node.expr, Expr::CheckIndex { .. } | Expr::Require { .. }))
    }

    /// Returns the axes `access` uses: those its strides multiply and those its indexed terms depend on, in axis order.
    ///
    /// # Panics
    ///
    /// Panics if an indexed term names a node the kernel does not have.
    #[must_use]
    pub fn access_domain(&self, access: &Access) -> Vec<AxisId> {
        let mut axes = vec![false; self.axes.len()];
        self.mark_access(access, &mut axes);
        axes.iter()
            .enumerate()
            .filter(|(_, used)| **used)
            .map(|(index, _)| AxisId(index))
            .collect()
    }

    fn mark_access(&self, access: &Access, used: &mut [bool]) {
        for axis in access.axes() {
            used[axis.0] = true;
        }
        for &(node, _) in &access.indirect {
            self.mark_domain(node, used);
        }
    }

    /// Returns the sizes passed at launch, in the order a target's arguments list them.
    pub fn argument_sizes(&self) -> impl Iterator<Item = SizeId> + '_ {
        self.sizes
            .iter()
            .enumerate()
            .filter(|(_, size)| matches!(size, Size::Argument))
            .map(|(index, _)| SizeId(index))
    }

    /// Returns the axes `node`'s value depends on, in axis order.
    ///
    /// # Panics
    ///
    /// Panics if the graph refers to a node it does not have; [`Self::verify`] rejects such kernels.
    #[must_use]
    pub fn domain(&self, node: NodeId) -> Vec<AxisId> {
        let mut axes = vec![false; self.axes.len()];
        self.mark_domain(node, &mut axes);
        axes.iter()
            .enumerate()
            .filter(|(_, used)| **used)
            .map(|(index, _)| AxisId(index))
            .collect()
    }

    fn mark_domain(&self, node: NodeId, used: &mut [bool]) {
        match &self.nodes[node.0].expr {
            Expr::Load(access) => self.mark_access(access, used),
            Expr::Literal(_) | Expr::Scalar(_) => {}
            Expr::AxisIndex(axis) => used[axis.0] = true,
            Expr::Unary(_, value) | Expr::Cast(value) | Expr::CheckIndex { value, .. } => {
                self.mark_domain(*value, used);
            }
            // A scan's value at each point of its axis depends on that point.
            Expr::Scan { axis, value, .. } => {
                self.mark_domain(*value, used);
                used[axis.0] = true;
            }
            Expr::Binary(_, lhs, rhs)
            | Expr::Require {
                value: lhs,
                condition: rhs,
                ..
            } => {
                self.mark_domain(*lhs, used);
                self.mark_domain(*rhs, used);
            }
            Expr::MulAdd(a, b, c) => {
                self.mark_domain(*a, used);
                self.mark_domain(*b, used);
                self.mark_domain(*c, used);
            }
            Expr::Select {
                condition,
                if_true,
                if_false,
            } => {
                self.mark_domain(*condition, used);
                self.mark_domain(*if_true, used);
                self.mark_domain(*if_false, used);
            }
            Expr::Reduce {
                axes: reduced,
                value,
                ..
            }
            | Expr::ArgReduce {
                axes: reduced,
                value,
                ..
            } => {
                let mut inner = vec![false; used.len()];
                self.mark_domain(*value, &mut inner);
                for axis in reduced {
                    inner[axis.0] = false;
                }
                for (outer, inner) in used.iter_mut().zip(inner) {
                    *outer |= inner;
                }
            }
            Expr::SoftmaxSum {
                axes: reduced,
                logit,
                value,
            } => {
                let mut inner = vec![false; used.len()];
                self.mark_domain(*logit, &mut inner);
                self.mark_domain(*value, &mut inner);
                for axis in reduced {
                    inner[axis.0] = false;
                }
                for (outer, inner) in used.iter_mut().zip(inner) {
                    *outer |= inner;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
