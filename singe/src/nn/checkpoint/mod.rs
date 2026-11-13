mod shard;

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt::{Debug, Formatter, Result as FmtResult},
    fs, iter, mem,
    path::Path,
};

use safetensors::{
    Dtype,
    tensor::{self, TensorInfo, TensorView},
};
use serde::Deserialize;

use self::shard::{Shard, shape_of, stored_type};
#[cfg(feature = "cuda")]
use crate::backend::cuda::upload::CudaUpload;
use crate::{
    data_type::DataType,
    device::Device,
    error::{CheckpointError, Error, Result},
    nn::ModelState,
    shape::Shape,
    tensor::Tensor,
    utility::{to_i64, to_usize},
};

// The names a snapshot's checkpoint goes by: `transformers`' first, then `diffusers`'.
const CHECKPOINT_STEMS: [&str; 2] = ["model", "diffusion_pytorch_model"];

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
        self.load_shards(vec![Shard::open(path)?], &CheckpointOptions::default())
    }

    /// Checks keys, dtype, shape, and all tied entries before publication.
    /// A failed load leaves the current model and all prepared executors intact.
    ///
    /// # Errors
    ///
    /// Returns an error if the checkpoint is malformed, keys are missing or unexpected, data types or shapes differ from the state schema, tied values disagree, dimensions cannot be represented, tensor construction or device transfer fails, or state replacement validation fails.
    pub fn load_safetensors_bytes(&self, bytes: &[u8]) -> Result<Self> {
        self.load_shards(
            vec![Shard::from_bytes(bytes)?],
            &CheckpointOptions::default(),
        )
    }

    // Loads the checkpoint of a Hugging Face snapshot directory: `model.safetensors`, or the shards `model.safetensors.index.json` names.
    // A `diffusers` snapshot names the same files `diffusion_pytorch_model`.
    // Each tensor is read from its file straight to its device, so no shard is held whole on the host.
    pub fn load_checkpoint(
        &self,
        directory: impl AsRef<Path>,
        options: &CheckpointOptions<'_>,
    ) -> Result<Self> {
        let directory = directory.as_ref();
        let stem = CHECKPOINT_STEMS
            .into_iter()
            .find(|stem| {
                directory.join(format!("{stem}.safetensors")).exists()
                    || directory
                        .join(format!("{stem}.safetensors.index.json"))
                        .exists()
            })
            .unwrap_or(CHECKPOINT_STEMS[0]);
        let index = directory.join(format!("{stem}.safetensors.index.json"));
        if !index.exists() {
            let shard = Shard::open(directory.join(format!("{stem}.safetensors")))?;
            return self.load_shards(vec![shard], options);
        }

        let index: ShardIndex =
            serde_json::from_slice(&fs::read(index)?).map_err(CheckpointError::Index)?;
        let mut files = index.weight_map.values().collect::<Vec<_>>();
        files.sort();
        files.dedup();
        let shards = files
            .into_iter()
            .map(|file| Shard::open(directory.join(file)))
            .collect::<Result<Vec<_>>>()?;
        self.load_shards(shards, options)
    }

    // Checks the names `shards` hold, reads each tensor, and publishes the state they make.
    fn load_shards(
        &self,
        mut shards: Vec<Shard<'_>>,
        options: &CheckpointOptions<'_>,
    ) -> Result<Self> {
        let names = shards.iter().flat_map(Shard::names).collect::<Vec<_>>();
        self.check_keys(names.iter().map(String::as_str), options)?;
        let mut loading = Loading::default();
        for shard in 0..shards.len() {
            self.load_shard(&mut shards, shard, options, &mut loading)?;
        }

        self.finish_pieces(options, &mut loading)?;
        self.publish(&loading)
    }

    // Refuses a checkpoint whose names, under `options`, do not cover this state's or name something else.
    fn check_keys<'a>(
        &self,
        names: impl Iterator<Item = &'a str>,
        options: &CheckpointOptions<'_>,
    ) -> Result<()> {
        let names = names
            .filter_map(|name| {
                options
                    .piece(name)
                    .map(|piece| piece.state)
                    .or_else(|| options.state_name(name))
            })
            .collect::<HashSet<_>>();
        let provided = self
            .bindings()
            .filter(|(binding, _)| names.contains(binding.name()))
            .map(|(binding, _)| binding.id())
            .collect::<HashSet<_>>();
        let mut missing = self
            .bindings()
            .filter(|(binding, _)| !provided.contains(&binding.id()))
            .map(|(binding, _)| binding.name().to_owned())
            .collect::<Vec<_>>();
        let mut unexpected = names
            .iter()
            .filter(|name| !self.snapshot.schema.bindings.contains_key(*name))
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
        Ok(())
    }

    // Reads, casts, and places each tensor of one shard, in the order its bytes lie.
    fn load_shard(
        &self,
        shards: &mut [Shard<'_>],
        shard: usize,
        options: &CheckpointOptions<'_>,
        loading: &mut Loading,
    ) -> Result<()> {
        for stored_name in shards[shard].names() {
            let piece = options.piece(&stored_name);
            let Some(name) = piece
                .as_ref()
                .map(|piece| piece.state.clone())
                .or_else(|| options.state_name(&stored_name))
            else {
                continue;
            };
            let info = shards[shard].info(&stored_name)?;
            let binding = &self.snapshot.schema.bindings[&name];
            let target = binding.spec().data_type();
            let source = stored_type(info.dtype)?;
            if source != target && !(options.convert && source.is_float() && target.is_float()) {
                return Err(Error::unsupported_operation_with_details(
                    "checkpoint dtype",
                    format!("`{name}` expected {target}, got {source}"),
                ));
            }

            let shape = shape_of(&info)?;
            if let Some(piece) = piece {
                let host = shards[shard].tensor_as(&info, target)?;
                loading
                    .pieces
                    .entry(binding.id())
                    .or_insert_with(|| PieceAssembly {
                        name: name.clone(),
                        data_type: target,
                        shape: binding.spec().shape().clone(),
                        bytes: vec![
                            0;
                            binding.spec().shape().element_count() * target.size_bytes()
                        ],
                        placed: Vec::new(),
                    })
                    .place(&piece.origin, shape.dimensions(), &host.to_bytes()?)?;
                continue;
            }

            if &shape != binding.spec().shape() {
                return Err(Error::shape_mismatch(
                    name,
                    binding.spec().shape().dimensions().to_vec(),
                    shape.dimensions().to_vec(),
                ));
            }

            // Ties compare as the schema's type, a chunk at a time from their files, before the second uploads.
            if let Some(tied) = loading.tied.get(&binding.id()) {
                if !same_values(shards, tied, (shard, &info), target)? {
                    return Err(Error::invalid_argument(format!(
                        "conflicting tied state bindings at `{name}`"
                    )));
                }

                continue;
            }

            let device = options
                .device
                .map_or_else(|| self.named_tensor(&name).map(Tensor::device), Some)
                .ok_or_else(|| Error::missing_input("checkpoint tensor's binding"))?;
            if self.has_tie(binding.id()) {
                loading.tied.insert(binding.id(), (shard, info.clone()));
            }

            let tensor = loading.read(&mut shards[shard], &info, target, &shape, device)?;
            loading.tensors.insert(binding.id(), tensor);
        }
        Ok(())
    }

    // Turns the pieces gathered across shards into their tensors, refusing one left partly unset.
    fn finish_pieces(&self, options: &CheckpointOptions<'_>, loading: &mut Loading) -> Result<()> {
        for (id, assembly) in mem::take(&mut loading.pieces) {
            let device = options
                .device
                .map_or_else(
                    || self.named_tensor(&assembly.name).map(Tensor::device),
                    Some,
                )
                .ok_or_else(|| Error::missing_input("checkpoint tensor's binding"))?;
            loading
                .tensors
                .insert(id, assembly.finish()?.to_device(device)?);
        }
        Ok(())
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

// Maps a checkpoint name to its state name, or to `None` to ignore it.
pub type StateNameMap = dyn Fn(&str) -> Option<String>;

// A checkpoint tensor that holds part of a state tensor, starting at `origin`, as checkpoints hold each expert of what state packs together.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatePiece {
    pub state: String,
    pub origin: Vec<i64>,
}

// Maps a checkpoint name to the piece it holds, or to `None` for a whole tensor.
pub type StatePieceMap = dyn Fn(&str) -> Option<StatePiece>;

// How a checkpoint becomes state.
#[derive(Clone, Copy, Default)]
pub struct CheckpointOptions<'a> {
    // Where tensors go, or each binding's current device for `None`.
    pub device: Option<Device>,
    // Casts floating tensors to the schema's type, on the host, rather than refusing a different one.
    pub convert: bool,
    // How checkpoint names become state names; they map to themselves without this.
    pub names: Option<&'a StateNameMap>,
    // Which checkpoint tensors are pieces of a state tensor, which `names` then does not see.
    pub pieces: Option<&'a StatePieceMap>,
}

impl CheckpointOptions<'_> {
    fn state_name(&self, name: &str) -> Option<String> {
        self.names
            .map_or_else(|| Some(name.to_owned()), |names| names(name))
    }

    fn piece(&self, name: &str) -> Option<StatePiece> {
        self.pieces.and_then(|pieces| pieces(name))
    }
}

impl Debug for CheckpointOptions<'_> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter
            .debug_struct("CheckpointOptions")
            .field("device", &self.device)
            .field("convert", &self.convert)
            .field("names", &self.names.is_some())
            .field("pieces", &self.pieces.is_some())
            .finish()
    }
}

#[derive(Deserialize)]
struct ShardIndex {
    weight_map: BTreeMap<String, String>,
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
    // Reads `info`'s tensor as `target` onto `device`: a chunk at a time on CUDA, and whole on the host otherwise.
    fn read(
        &mut self,
        shard: &mut Shard<'_>,
        info: &TensorInfo,
        target: DataType,
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
            return self.uploads[at].1.tensor(target, shape, |offset, chunk| {
                shard.read_as(info, target, offset, chunk)
            });
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
    let (mut first, mut second) = (vec![0; COMPARED_BYTES], vec![0; COMPARED_BYTES]);
    let mut offset = 0;
    while offset < total {
        let length = COMPARED_BYTES.min(total - offset);
        shards[*tied_shard].read_as(tied, target, offset, &mut first[..length])?;
        shards[shard].read_as(info, target, offset, &mut second[..length])?;
        if first[..length] != second[..length] {
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
