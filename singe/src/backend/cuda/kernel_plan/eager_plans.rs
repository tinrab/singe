// The plans eager operations run, kept by what they compute and read, so a repeated call plans and loads its kernels once.
//
// Planning renders and hashes a kernel's code, which costs more than launching it, and a decode step repeats the same operations on the same layouts every time.

use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
    sync::{Arc, LazyLock, Mutex},
};

use crate::{
    backend::{
        cuda::kernel_plan::source::{KernelInput, KernelPlan, KernelPlanner},
        registry::{BitHash, Operation, OperationAttributes},
    },
    data_type::DataType,
    error::Result,
    graph::NumericalPolicy,
    layout::Layout,
    tensor::Tensor,
    utility::Locked,
};

// The most entries kept; past it the cache starts over, which a workload of this many shapes barely notices.
const LIMIT: usize = 4096;

static PLANS: LazyLock<Mutex<HashMap<PlanKey, Arc<[KernelPlan]>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

// Which planner method makes an entry's plans.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlanKind {
    Operation(Operation),
    Index,
    IndexPut,
    Concat,
    RepeatKv,
    FusedAttention,
    PagedAttention,
    PagedMlaAttention,
}

// What an eager plan depends on: the default planner plans from these alone, and the numerical policy reaches the parameter keys it reads.
struct PlanKey {
    kind: PlanKind,
    attributes: OperationAttributes,
    numbers: Vec<i64>,
    inputs: Vec<(DataType, Layout)>,
    policy: NumericalPolicy,
}

impl PartialEq for PlanKey {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
            && self.attributes.bit_eq(&other.attributes)
            && self.numbers == other.numbers
            && self.inputs == other.inputs
            && self.policy == other.policy
    }
}

impl Eq for PlanKey {}

impl Hash for PlanKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.kind.hash(state);
        self.attributes.bit_hash(state);
        self.numbers.hash(state);
        self.inputs.hash(state);
        self.policy.hash(state);
    }
}

// Returns the plans `build` makes with the default planner from `inputs`' kernel inputs, made once for each kind, attributes, numbers, and input layouts.
//
// The kind, attributes, and numbers must be everything `build` reads besides the inputs.
pub fn plans(
    kind: PlanKind,
    attributes: &OperationAttributes,
    numbers: &[i64],
    inputs: &[&Tensor],
    build: impl FnOnce(&KernelPlanner, &[KernelInput]) -> Result<Vec<KernelPlan>>,
) -> Result<Arc<[KernelPlan]>> {
    let key = PlanKey {
        kind,
        attributes: attributes.clone(),
        numbers: numbers.to_vec(),
        inputs: inputs
            .iter()
            .map(|input| (input.data_type(), input.layout().clone()))
            .collect(),
        policy: NumericalPolicy::current(),
    };
    if let Some(plans) = PLANS.locked("eager kernel plans")?.get(&key) {
        return Ok(Arc::clone(plans));
    }

    let metadata = inputs
        .iter()
        .map(|input| KernelInput::from_source(*input))
        .collect::<Vec<_>>();
    let plans = Arc::<[KernelPlan]>::from(build(&KernelPlanner::default(), &metadata)?);
    let mut cached = PLANS.locked("eager kernel plans")?;
    if cached.len() >= LIMIT {
        cached.clear();
    }

    cached.insert(key, Arc::clone(&plans));
    drop(cached);
    Ok(plans)
}

// Returns the one plan the default planner makes for `operation` on `inputs`.
pub fn plan(
    operation: Operation,
    attributes: &OperationAttributes,
    inputs: &[&Tensor],
) -> Result<Arc<[KernelPlan]>> {
    plans(
        PlanKind::Operation(operation),
        attributes,
        &[],
        inputs,
        |planner, metadata| Ok(vec![planner.create(operation, attributes, metadata)?]),
    )
}
