mod shard;

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fmt::{Debug, Formatter, Result as FmtResult},
    fs, iter, mem,
    path::Path,
    sync::Arc,
};

use safetensors::{
    Dtype,
    tensor::{self, TensorInfo, TensorView},
};

use self::shard::{AlignedBytes, Shard, shape_of, stored_type};
#[cfg(feature = "cuda")]
use crate::backend::cuda::upload::CudaUpload;
use crate::{
    data_type::DataType,
    device::Device,
    error::{CheckpointError, Error, Result},
    nn::ModelState,
    shape::Shape,
    state::StateBindingSpec,
    tensor::Tensor,
    utility::{to_i64, to_usize},
};

impl ModelState {
    /// Serializes all named parameters and persistent buffers.
    /// Tied aliases are included under each schema name; loading also accepts one name per tie.
    ///
    /// # Errors
    ///
    /// Returns an error if a binding uses an unsupported checkpoint data type, a dimension cannot be represented, tensor readback fails, or safetensors view construction or serialization fails.
    pub fn safetensors_bytes(&self) -> Result<Vec<u8>> {
        let mut owned = BTreeMap::new();
        for (binding, tensor) in self.bindings() {
            let dtype = checkpoint_dtype(tensor.data_type())?;
            let dimensions = tensor
                .shape()
                .dimensions()
                .iter()
                .map(|&dimension| to_usize(dimension, "checkpoint dimension"))
                .collect::<Result<Vec<_>>>()?;
            owned.insert(
                binding.name().to_owned(),
                (dtype, dimensions, tensor.to_bytes()?),
            );
        }

        let views = owned
            .iter()
            .map(|(name, (dtype, shape, bytes))| {
                Ok((name, TensorView::new(*dtype, shape.clone(), bytes)?))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(tensor::serialize(views, None)?)
    }

    /// Serializes parameters and persistent buffers to a safetensors file.
    /// An existing file at `path` is replaced.
    ///
    /// # Errors
    ///
    /// Returns an error if state serialization fails or the file cannot be written.
    pub fn save_safetensors(&self, path: impl AsRef<Path>) -> Result<()> {
        fs::write(path, self.safetensors_bytes()?)?;
        Ok(())
    }

    /// Reads a safetensors file and returns a validated replacement state on the existing bindings' devices.
    /// The current state remains unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or checkpoint parsing, schema validation, device transfer, or state replacement fails.
    pub fn load_safetensors(&self, path: impl AsRef<Path>) -> Result<Self> {
        let mut checkpoint = Checkpoint::open([path])?;
        let plan = LoadPlan::identity(&checkpoint);
        self.load_exactly(&mut checkpoint, &plan)
    }

    /// Checks keys, dtype, shape, and all tied entries before publication.
    /// A failed load leaves the current model and all prepared executors intact.
    ///
    /// # Errors
    ///
    /// Returns an error if the checkpoint is malformed, keys are missing or unexpected, data types or shapes differ from the state schema, tied values disagree, dimensions cannot be represented, tensor construction or device transfer fails, or state replacement validation fails.
    pub fn load_safetensors_bytes(&self, bytes: &[u8]) -> Result<Self> {
        let mut checkpoint = Checkpoint::from_bytes(bytes)?;
        let plan = LoadPlan::identity(&checkpoint);
        self.load_exactly(&mut checkpoint, &plan)
    }

    fn load_exactly(&self, checkpoint: &mut Checkpoint<'_>, plan: &LoadPlan) -> Result<Self> {
        self.load_checkpoint(checkpoint, plan, CheckpointOptions::default())
    }

    // Loads the state `plan` gives each binding from `checkpoint`, and publishes it only once every binding has its tensor.
    // A stored tensor that serves a binding whole streams from its file straight to the device, so no file is held whole on the host.
    // Stored tensors `plan` does not read are left alone; whether any may be left is the caller's to decide.
    pub fn load_checkpoint(
        &self,
        checkpoint: &mut Checkpoint<'_>,
        plan: &LoadPlan,
        options: CheckpointOptions,
    ) -> Result<Self> {
        self.check_plan(checkpoint, plan)?;
        let mut loading = Loading::default();
        self.load_stored(checkpoint, plan, options, &mut loading)?;
        self.load_derived(checkpoint, plan, options, &mut loading)?;
        self.publish(&loading)
    }

    // Refuses a plan that names a target outside this state's schema, leaves a binding without a source, or reads a tensor `checkpoint` lacks.
    fn check_plan(&self, checkpoint: &Checkpoint<'_>, plan: &LoadPlan) -> Result<()> {
        let bindings = &self.snapshot.schema.bindings;
        let provided = plan
            .targets
            .keys()
            .filter_map(|name| bindings.get(name).map(StateBindingSpec::id))
            .collect::<HashSet<_>>();
        let mut missing = self
            .bindings()
            .filter(|(binding, _)| !provided.contains(&binding.id()))
            .map(|(binding, _)| binding.name().to_owned())
            .collect::<Vec<_>>();
        let mut unexpected = plan
            .targets
            .keys()
            .filter(|name| !bindings.contains_key(*name))
            .cloned()
            .collect::<Vec<_>>();
        missing.sort();
        unexpected.sort();
        if !missing.is_empty() || !unexpected.is_empty() {
            return Err(CheckpointError::Keys {
                missing,
                unexpected,
            }
            .into());
        }

        let absent = plan
            .stored_names()
            .into_iter()
            .filter(|name| !checkpoint.contains(name))
            .map(str::to_owned)
            .collect::<Vec<_>>();
        if !absent.is_empty() {
            return Err(CheckpointError::Absent { names: absent }.into());
        }
        Ok(())
    }

    // Reads each stored tensor the plan takes whole or as a piece, shard by shard in the order their bytes lie, then assembles the pieces.
    fn load_stored(
        &self,
        checkpoint: &mut Checkpoint<'_>,
        plan: &LoadPlan,
        options: CheckpointOptions,
        loading: &mut Loading,
    ) -> Result<()> {
        let mut uses = HashMap::<&str, Vec<(&str, Option<&[i64]>)>>::new();
        for (target, source) in &plan.targets {
            match source {
                TensorSource::Stored(name) => {
                    uses.entry(name).or_default().push((target, None));
                }
                TensorSource::Pieces(pieces) => {
                    for piece in pieces {
                        uses.entry(&piece.name)
                            .or_default()
                            .push((target, Some(&piece.origin)));
                    }
                }
                TensorSource::Derived(_) => {}
            }
        }

        for shard in 0..checkpoint.shards.len() {
            for stored_name in checkpoint.shards[shard].names() {
                let Some(targets) = uses.get(stored_name.as_str()) else {
                    continue;
                };
                let info = checkpoint.shards[shard].info(&stored_name)?;
                for &(target, origin) in targets {
                    if let Some(origin) = origin {
                        self.load_piece(
                            &mut checkpoint.shards[shard],
                            &info,
                            (target, origin),
                            options,
                            loading,
                        )?;
                    } else {
                        self.load_whole(
                            &mut checkpoint.shards,
                            (shard, &info),
                            target,
                            options,
                            loading,
                        )?;
                    }
                }
            }
        }

        for (id, assembly) in mem::take(&mut loading.pieces) {
            let device = self.target_device(&assembly.name, options)?;
            let name = assembly.name.clone();
            let tensor = assembly.finish()?.to_device(device)?;
            loading.insert(id, &name, tensor)?;
        }
        Ok(())
    }

    // Reads a stored tensor that is the whole of `target`.
    fn load_whole(
        &self,
        shards: &mut [Shard<'_>],
        (shard, info): (usize, &TensorInfo),
        target: &str,
        options: CheckpointOptions,
        loading: &mut Loading,
    ) -> Result<()> {
        let binding = &self.snapshot.schema.bindings[target];
        let data_type = binding.spec().data_type();
        accept_type(target, stored_type(info.dtype)?, data_type, options.convert)?;
        let shape = shape_of(info)?;
        if &shape != binding.spec().shape() {
            return Err(Error::shape_mismatch(
                target.to_owned(),
                binding.spec().shape().dimensions().to_vec(),
                shape.dimensions().to_vec(),
            ));
        }

        // Ties compare as the schema's type, a chunk at a time from their files, before the second uploads.
        if let Some(tied) = loading.tied.get(&binding.id()) {
            if !same_values(shards, tied, (shard, info), data_type)? {
                return Err(Error::invalid_argument(format!(
                    "conflicting tied state bindings at `{target}`"
                )));
            }

            return Ok(());
        }

        let device = self.target_device(target, options)?;
        if self.has_tie(binding.id()) {
            loading.tied.insert(binding.id(), (shard, info.clone()));
        }

        let tensor = loading.read(&mut shards[shard], info, data_type, &shape, device)?;
        loading.insert(binding.id(), target, tensor)
    }

    // Places a stored tensor into the host copy of `target` being assembled from pieces.
    fn load_piece(
        &self,
        shard: &mut Shard<'_>,
        info: &TensorInfo,
        (target, origin): (&str, &[i64]),
        options: CheckpointOptions,
        loading: &mut Loading,
    ) -> Result<()> {
        let binding = &self.snapshot.schema.bindings[target];
        let data_type = binding.spec().data_type();
        accept_type(target, stored_type(info.dtype)?, data_type, options.convert)?;
        let host = shard.tensor_as(info, data_type)?;
        let shape = shape_of(info)?;
        loading
            .pieces
            .entry(binding.id())
            .or_insert_with(|| PieceAssembly {
                name: target.to_owned(),
                data_type,
                shape: binding.spec().shape().clone(),
                bytes: vec![0; binding.spec().shape().element_count() * data_type.size_bytes()],
                placed: Vec::new(),
            })
            .place(origin, shape.dimensions(), &host.to_bytes()?)
    }

    // Reads the inputs of each derived target onto its device as they are stored, and keeps what its derivation makes of them.
    fn load_derived(
        &self,
        checkpoint: &mut Checkpoint<'_>,
        plan: &LoadPlan,
        options: CheckpointOptions,
        loading: &mut Loading,
    ) -> Result<()> {
        // A stored tensor several derivations read, such as a fused projection split into parts, is read once and kept until its last reader.
        let derivations = plan
            .targets
            .iter()
            .filter_map(|(target, source)| match source {
                TensorSource::Derived(derivation) => Some((target, derivation)),
                TensorSource::Stored(_) | TensorSource::Pieces(_) => None,
            })
            .map(|(target, derivation)| {
                Ok((target, derivation, self.target_device(target, options)?))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut readers = HashMap::<(&str, Device), usize>::new();
        for (_, derivation, device) in &derivations {
            for name in &derivation.inputs {
                *readers.entry((name.as_str(), *device)).or_default() += 1;
            }
        }

        let mut kept = HashMap::<(&str, Device), Tensor>::new();
        for (target, derivation, device) in derivations {
            let binding = &self.snapshot.schema.bindings[target];
            let mut inputs = Vec::with_capacity(derivation.inputs.len());
            for name in &derivation.inputs {
                let key = (name.as_str(), device);
                let input = if let Some(input) = kept.remove(&key) {
                    input
                } else {
                    let (shard, info) = checkpoint.info(name)?;
                    let stored = stored_type(info.dtype)?;
                    let shape = shape_of(&info)?;
                    loading.read(&mut checkpoint.shards[shard], &info, stored, &shape, device)?
                };
                let left = readers.entry(key).or_default();
                *left = left.saturating_sub(1);
                if *left > 0 {
                    kept.insert(key, input.clone());
                }

                inputs.push(input);
            }

            let derived = (derivation.derive)(&inputs)?;
            let data_type = binding.spec().data_type();
            accept_type(target, derived.data_type(), data_type, options.convert)?;
            if derived.shape() != binding.spec().shape() {
                return Err(Error::shape_mismatch(
                    target.clone(),
                    binding.spec().shape().dimensions().to_vec(),
                    derived.shape().dimensions().to_vec(),
                ));
            }

            loading.insert(binding.id(), target, derived.cast(data_type)?)?;
        }
        Ok(())
    }

    // Where `target`'s tensor goes: the options' device, or its binding's current one.
    fn target_device(&self, target: &str, options: CheckpointOptions) -> Result<Device> {
        options
            .device
            .map_or_else(|| self.named_tensor(target).map(Tensor::device), Some)
            .ok_or_else(|| Error::missing_input("checkpoint tensor's binding"))
    }

    fn has_tie(&self, id: usize) -> bool {
        self.snapshot
            .schema
            .bindings
            .values()
            .filter(|binding| binding.id() == id)
            .count()
            > 1
    }

    fn publish(&self, loading: &Loading) -> Result<Self> {
        let replacements = self
            .snapshot
            .schema
            .bindings
            .values()
            .map(|binding| {
                let tensor = loading.tensors.get(&binding.id()).ok_or_else(|| {
                    Error::missing_input(format!("checkpoint `{}`", binding.name()))
                })?;
                Ok((binding.name().to_owned(), Tensor::clone(tensor)))
            })
            .collect::<Result<HashMap<_, _>>>()?;
        self.replace(&replacements)
    }
}

// Refuses a tensor of `source` type for a binding of `target` type, unless both are floating and `convert` allows the cast.
fn accept_type(name: &str, source: DataType, target: DataType, convert: bool) -> Result<()> {
    if source != target && !(convert && source.is_float() && target.is_float()) {
        return Err(Error::unsupported_operation_with_details(
            "checkpoint dtype",
            format!("`{name}` expected {target}, got {source}"),
        ));
    }
    Ok(())
}

// Safetensors files read as one checkpoint, whose tensors stay where they lie until a load reads them.
pub struct Checkpoint<'a> {
    shards: Vec<Shard<'a>>,
    // The shard each stored name is in.
    locations: HashMap<String, usize>,
}

impl Checkpoint<'static> {
    // Reads the headers of `files`, which together hold each tensor once.
    pub fn open<P: AsRef<Path>>(files: impl IntoIterator<Item = P>) -> Result<Self> {
        let shards = files
            .into_iter()
            .map(Shard::open)
            .collect::<Result<Vec<_>>>()?;
        Self::from_shards(shards)
    }
}

impl<'a> Checkpoint<'a> {
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self> {
        Self::from_shards(vec![Shard::from_bytes(bytes)?])
    }

    fn from_shards(shards: Vec<Shard<'a>>) -> Result<Self> {
        let mut locations = HashMap::new();
        for (index, shard) in shards.iter().enumerate() {
            for name in shard.names() {
                if locations.insert(name.clone(), index).is_some() {
                    return Err(Error::invalid_argument(format!(
                        "checkpoint tensor `{name}` stored twice"
                    )));
                }
            }
        }
        Ok(Self { shards, locations })
    }

    // Returns the stored names, sorted.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        let mut names = self
            .locations
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>();
        names.sort_unstable();
        names
    }

    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.locations.contains_key(name)
    }

    fn info(&self, name: &str) -> Result<(usize, TensorInfo)> {
        let shard = *self
            .locations
            .get(name)
            .ok_or_else(|| Error::missing_input(format!("checkpoint tensor `{name}`")))?;
        Ok((shard, self.shards[shard].info(name)?))
    }
}

impl Debug for Checkpoint<'_> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter
            .debug_struct("Checkpoint")
            .field("shards", &self.shards.len())
            .field("tensors", &self.locations.len())
            .finish()
    }
}

// Where each state tensor comes from in a checkpoint, by the state's names.
#[derive(Debug, Clone, Default)]
pub struct LoadPlan {
    targets: BTreeMap<String, TensorSource>,
}

impl LoadPlan {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    // Reads every stored tensor into the binding of its own name, as Singe stores state.
    #[must_use]
    pub fn identity(checkpoint: &Checkpoint<'_>) -> Self {
        Self {
            targets: checkpoint
                .names()
                .into_iter()
                .map(|name| (name.to_owned(), TensorSource::Stored(name.to_owned())))
                .collect(),
        }
    }

    // Gives `target` its source, refusing a second one; pieces of one target gather into one source.
    pub fn insert(&mut self, target: impl Into<String>, source: TensorSource) -> Result<()> {
        let target = target.into();
        match (self.targets.get_mut(&target), source) {
            (None, source) => {
                self.targets.insert(target, source);
                Ok(())
            }
            (Some(TensorSource::Pieces(pieces)), TensorSource::Pieces(more)) => {
                pieces.extend(more);
                Ok(())
            }
            (Some(_), _) => Err(Error::invalid_argument(format!(
                "load plan gives `{target}` two sources"
            ))),
        }
    }

    // Returns every stored name the plan reads.
    #[must_use]
    pub fn stored_names(&self) -> BTreeSet<&str> {
        self.targets
            .values()
            .flat_map(|source| -> Vec<&str> {
                match source {
                    TensorSource::Stored(name) => vec![name],
                    TensorSource::Pieces(pieces) => {
                        pieces.iter().map(|piece| piece.name.as_str()).collect()
                    }
                    TensorSource::Derived(derivation) => {
                        derivation.inputs.iter().map(String::as_str).collect()
                    }
                }
            })
            .collect()
    }
}

// Where one state tensor comes from.
#[derive(Debug, Clone)]
pub enum TensorSource {
    // One stored tensor, as it is stored.
    Stored(String),
    // Stored tensors placed into it, together covering it once, as checkpoints hold each expert of what state packs together.
    Pieces(Vec<StatePiece>),
    // A tensor computed on the device from stored ones.
    Derived(Derivation),
}

// A stored tensor that holds part of a state tensor, starting at `origin`; it may leave out leading axes of extent one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatePiece {
    pub name: String,
    pub origin: Vec<i64>,
}

type Derive = dyn Fn(&[Tensor]) -> Result<Tensor> + Send + Sync;

// A state tensor computed from stored `inputs`, which arrive on its device as they are stored, such as a transposed weight, one part of a fused one, or a weight folded from a weight-normed pair.
// What it returns is cast to the binding's type when the load converts.
#[derive(Clone)]
pub struct Derivation {
    inputs: Vec<String>,
    derive: Arc<Derive>,
}

impl Derivation {
    pub fn new(
        inputs: Vec<String>,
        derive: impl Fn(&[Tensor]) -> Result<Tensor> + Send + Sync + 'static,
    ) -> Self {
        Self {
            inputs,
            derive: Arc::new(derive),
        }
    }
}

impl Debug for Derivation {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter
            .debug_struct("Derivation")
            .field("inputs", &self.inputs)
            .finish_non_exhaustive()
    }
}

// How a load places and types what it reads.
#[derive(Debug, Clone, Copy, Default)]
pub struct CheckpointOptions {
    // Where tensors go, or each binding's current device for `None`.
    pub device: Option<Device>,
    // Casts floating tensors to the schema's type rather than refusing a different one.
    pub convert: bool,
}

// What a load has gathered: each logical binding's tensor, and where the first of each tie lies, across shards.
#[derive(Default)]
struct Loading {
    tensors: HashMap<usize, Tensor>,
    tied: HashMap<usize, (usize, TensorInfo)>,
    pieces: HashMap<usize, PieceAssembly>,
    // The page-locked buffers each CUDA device's tensors upload through.
    #[cfg(feature = "cuda")]
    uploads: Vec<(Device, CudaUpload)>,
}

impl Loading {
    // Keeps `tensor` as a binding's, or checks it against the tensor another name of the same tied binding already gave.
    fn insert(&mut self, id: usize, name: &str, tensor: Tensor) -> Result<()> {
        if let Some(existing) = self.tensors.get(&id) {
            if existing.to_bytes()? != tensor.to_bytes()? {
                return Err(Error::invalid_argument(format!(
                    "conflicting tied state bindings at `{name}`"
                )));
            }

            return Ok(());
        }

        self.tensors.insert(id, tensor);
        Ok(())
    }

    // Reads `info`'s tensor as `target` onto `device`: a chunk at a time on CUDA, and whole on the host otherwise.
    fn read(
        &mut self,
        shard: &mut Shard<'_>,
        info: &TensorInfo,
        target: DataType,
        #[cfg_attr(
            not(feature = "cuda"),
            expect(unused_variables, reason = "only CUDA uploads by shape")
        )]
        shape: &Shape,
        device: Device,
    ) -> Result<Tensor> {
        // Booleans go through the host, whose tensors check each value.
        #[cfg(feature = "cuda")]
        if device.is_cuda() && target != DataType::Bool {
            let at = if let Some(at) = self.uploads.iter().position(|(other, _)| *other == device) {
                at
            } else {
                self.uploads.push((device, CudaUpload::create(device)?));
                self.uploads.len() - 1
            };
            // A tensor stored as another type uploads as it is stored and casts on the device.
            let source = stored_type(info.dtype)?;
            let uploaded = self.uploads[at].1.tensor(source, shape, |offset, chunk| {
                shard.read_as(info, source, offset, chunk)
            })?;
            return uploaded.cast(target);
        }

        shard.tensor_as(info, target)?.to_device(device)
    }
}

// How many bytes of two tied tensors compare at once.
const COMPARED_BYTES: usize = 1 << 20;

// Whether the tensor `tied` locates and `info` in shard `shard` hold the same values as `target`.
fn same_values(
    shards: &mut [Shard<'_>],
    (tied_shard, tied): &(usize, TensorInfo),
    (shard, info): (usize, &TensorInfo),
    target: DataType,
) -> Result<bool> {
    if tied.shape != info.shape {
        return Ok(false);
    }

    let total = shape_of(info)?.element_count() * target.size_bytes();
    let (mut first, mut second) = (
        AlignedBytes::zeroed(COMPARED_BYTES),
        AlignedBytes::zeroed(COMPARED_BYTES),
    );
    let mut offset = 0;
    while offset < total {
        let length = COMPARED_BYTES.min(total - offset);
        shards[*tied_shard].read_as(tied, target, offset, &mut first.as_mut()[..length])?;
        shards[shard].read_as(info, target, offset, &mut second.as_mut()[..length])?;
        if first.as_ref()[..length] != second.as_ref()[..length] {
            return Ok(false);
        }

        offset += length;
    }
    Ok(true)
}

// A state tensor being assembled on the host from pieces, which may arrive across shards.
struct PieceAssembly {
    name: String,
    data_type: DataType,
    shape: Shape,
    bytes: Vec<u8>,
    // Each piece's origin and shape, whose volumes say whether the tensor is covered once none overlap.
    placed: Vec<(Vec<i64>, Vec<i64>)>,
}

impl PieceAssembly {
    // Places a piece whose missing leading axes have extent one, as one expert's matrix is of all experts'.
    fn place(&mut self, origin: &[i64], shape: &[i64], bytes: &[u8]) -> Result<()> {
        let whole = self.shape.dimensions();
        let shape = iter::repeat_n(1, whole.len().saturating_sub(shape.len()))
            .chain(shape.iter().copied())
            .collect::<Vec<_>>();
        let shape = shape.as_slice();
        let invalid = |what: &str| {
            Error::invalid_argument(format!("checkpoint piece of `{}` {what}", self.name))
        };
        if origin.len() != whole.len()
            || shape.len() != whole.len()
            || origin
                .iter()
                .zip(shape)
                .zip(whole)
                .any(|((&origin, &extent), &size)| origin < 0 || origin + extent > size)
        {
            return Err(invalid("falls outside its tensor"));
        }

        let overlaps = |(other_origin, other_shape): &(Vec<i64>, Vec<i64>)| {
            (0..whole.len()).all(|axis| {
                origin[axis] < other_origin[axis] + other_shape[axis]
                    && other_origin[axis] < origin[axis] + shape[axis]
            })
        };
        if self.placed.iter().any(overlaps) {
            return Err(invalid("overlaps another"));
        }

        // Row-major strides of the whole, in bytes, and the runs of the last axis a piece copies.
        let element = self.data_type.size_bytes();
        let mut strides = vec![element; whole.len()];
        for axis in (0..whole.len().saturating_sub(1)).rev() {
            strides[axis] = strides[axis + 1] * to_usize(whole[axis + 1], "checkpoint dimension")?;
        }

        let last = whole.len().saturating_sub(1);
        let run = to_usize(
            shape.get(last).copied().unwrap_or(1),
            "checkpoint dimension",
        )? * element;
        if run > 0 {
            let mut index = vec![0_i64; last];
            for chunk in bytes.chunks_exact(run) {
                let mut offset = 0;
                for axis in 0..whole.len() {
                    let position = origin[axis] + index.get(axis).copied().unwrap_or(0);
                    offset += to_usize(position, "checkpoint dimension")? * strides[axis];
                }

                self.bytes[offset..offset + run].copy_from_slice(chunk);
                for axis in (0..last).rev() {
                    index[axis] += 1;
                    if index[axis] < shape[axis] {
                        break;
                    }

                    index[axis] = 0;
                }
            }
        }

        self.placed.push((origin.to_vec(), shape.to_vec()));
        Ok(())
    }

    fn finish(self) -> Result<Tensor> {
        let covered = self
            .placed
            .iter()
            .map(|(_, shape)| shape.iter().product::<i64>())
            .sum::<i64>();
        if covered != to_i64(self.shape.element_count(), "checkpoint element count")? {
            return Err(Error::invalid_argument(format!(
                "checkpoint pieces leave part of `{}` unset",
                self.name
            )));
        }
        Tensor::from_bytes(Device::Cpu, self.data_type, &self.shape, &self.bytes)
    }
}

fn checkpoint_dtype(dtype: DataType) -> Result<Dtype> {
    Ok(match dtype {
        DataType::Bool => Dtype::BOOL,
        DataType::U8 => Dtype::U8,
        DataType::I8 => Dtype::I8,
        DataType::U16 => Dtype::U16,
        DataType::I16 => Dtype::I16,
        DataType::U32 => Dtype::U32,
        DataType::I32 => Dtype::I32,
        DataType::U64 => Dtype::U64,
        DataType::I64 => Dtype::I64,
        DataType::F16 => Dtype::F16,
        DataType::Bf16 => Dtype::BF16,
        DataType::F32 => Dtype::F32,
        DataType::F64 => Dtype::F64,
        DataType::F8E4M3 | DataType::F8E5M2 | DataType::F4E2M1 => {
            return Err(Error::unsupported_operation(
                "checkpoint low precision dtype",
            ));
        }
    })
}
