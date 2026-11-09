mod resolve;

pub use self::resolve::concrete_shape_bucket_from_ir;

use crate::graph::{InputDimensionGuard, ir::Graph};

pub fn symbolic_dimension_key_material_from_ir(ir: &Graph) -> Vec<SymbolicDimensionKey> {
    ir.ordered_input_dimension_guards()
        .into_iter()
        .map(symbolic_dimension_key)
        .collect()
}

fn symbolic_dimension_key(guard: &InputDimensionGuard) -> SymbolicDimensionKey {
    SymbolicDimensionKey::new(
        guard.input(),
        guard.axis(),
        guard.name(),
        guard.min(),
        guard.max(),
    )
}

/// Concrete dynamic-shape bucket values included in CUDA graph cache keys.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConcreteShapeBucket {
    /// Concrete bucket dimensions in stable input/axis order.
    dimensions: Vec<ConcreteShapeBucketDimension>,
}

/// Concrete bucket value for one guarded input axis.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConcreteShapeBucketDimension {
    /// Input name that owns the guarded axis.
    input: String,
    /// Input axis guarded by this symbolic dimension.
    axis: usize,
    /// Symbolic dimension name.
    symbol: String,
    /// Concrete bucket value used for this key.
    value: i64,
}

impl ConcreteShapeBucket {
    pub(crate) fn new(dimensions: Vec<ConcreteShapeBucketDimension>) -> Self {
        Self { dimensions }
    }

    /// Returns concrete bucket dimensions in stable input/axis order.
    #[must_use]
    pub fn dimensions(&self) -> &[ConcreteShapeBucketDimension] {
        &self.dimensions
    }
}

impl ConcreteShapeBucketDimension {
    pub(crate) fn new(
        input: impl Into<String>,
        axis: usize,
        symbol: impl Into<String>,
        value: i64,
    ) -> Self {
        Self {
            input: input.into(),
            axis,
            symbol: symbol.into(),
            value,
        }
    }

    /// Returns the input name that owns the guarded axis.
    #[must_use]
    pub fn input(&self) -> &str {
        &self.input
    }

    /// Returns the input axis guarded by this symbolic dimension.
    #[must_use]
    pub fn axis(&self) -> usize {
        self.axis
    }

    /// Returns the symbolic dimension name.
    #[must_use]
    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    /// Returns the concrete bucket value used for this key.
    #[must_use]
    pub fn value(&self) -> i64 {
        self.value
    }
}

/// Symbolic dimension metadata included in CUDA graph cache keys.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SymbolicDimensionKey {
    /// Input name that owns the guarded axis.
    input: String,
    /// Input axis guarded by this symbolic dimension.
    axis: usize,
    /// Symbolic dimension name.
    symbol: String,
    /// Inclusive minimum accepted runtime value.
    min: Option<i64>,
    /// Inclusive maximum accepted runtime value.
    max: Option<i64>,
}

impl SymbolicDimensionKey {
    pub(crate) fn new(
        input: impl Into<String>,
        axis: usize,
        symbol: impl Into<String>,
        min: Option<i64>,
        max: Option<i64>,
    ) -> Self {
        Self {
            input: input.into(),
            axis,
            symbol: symbol.into(),
            min,
            max,
        }
    }

    /// Returns the input name that owns the guarded axis.
    #[must_use]
    pub fn input(&self) -> &str {
        &self.input
    }

    /// Returns the input axis guarded by this symbolic dimension.
    #[must_use]
    pub fn axis(&self) -> usize {
        self.axis
    }

    /// Returns the symbolic dimension name.
    #[must_use]
    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    /// Returns the inclusive minimum accepted runtime value.
    #[must_use]
    pub fn min(&self) -> Option<i64> {
        self.min
    }

    /// Returns the inclusive maximum accepted runtime value.
    #[must_use]
    pub fn max(&self) -> Option<i64> {
        self.max
    }
}

/// CUDA graph dynamic-shape cache policy.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DynamicShapePolicy {
    /// Dynamic-shape mode.
    mode: DynamicShapeMode,
    /// Optional warning/error threshold for recordings made from one symbolic graph.
    max_recordings_per_graph: Option<usize>,
}

/// How CUDA graph cache keys treat symbolic dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DynamicShapeMode {
    /// Reject symbolic dimensions before CUDA graph compilation.
    Reject,
    /// Record one CUDA graph per exact concrete symbolic-dimension value.
    ExactShape,
}

impl Default for DynamicShapePolicy {
    fn default() -> Self {
        Self::reject()
    }
}

impl DynamicShapePolicy {
    /// Rejects dynamic dimensions before CUDA graph compilation.
    #[must_use]
    pub fn reject() -> Self {
        Self {
            mode: DynamicShapeMode::Reject,
            max_recordings_per_graph: None,
        }
    }

    /// Records one CUDA graph per exact concrete symbolic dimension value.
    #[must_use]
    pub fn exact_shape() -> Self {
        Self {
            mode: DynamicShapeMode::ExactShape,
            max_recordings_per_graph: None,
        }
    }

    /// Sets the maximum expected recordings for one symbolic graph.
    #[must_use]
    pub fn with_max_recordings_per_graph(mut self, max: usize) -> Self {
        self.max_recordings_per_graph = Some(max);
        self
    }

    /// Returns the dynamic-shape mode.
    #[must_use]
    pub fn mode(&self) -> DynamicShapeMode {
        self.mode
    }

    /// Returns the optional recording threshold for one symbolic graph.
    #[must_use]
    pub fn max_recordings_per_graph(&self) -> Option<usize> {
        self.max_recordings_per_graph
    }
}
