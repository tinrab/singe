use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt::{Debug, Formatter, Result as FmtResult},
    fs, iter, mem,
    path::Path,
};

use safetensors::{
    Dtype, SafeTensors,
    tensor::{self, TensorView},
};
use serde::Deserialize;

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
        self.load_safetensors_bytes(&read(path)?)
    }

    /// Checks keys, dtype, shape, and all tied entries before publication.
    /// A failed load leaves the current model and all prepared executors intact.
    ///
    /// # Errors
    ///
    /// Returns an error if the checkpoint is malformed, keys are missing or unexpected, data types or shapes differ from the state schema, tied values disagree, dimensions cannot be represented, tensor construction or device transfer fails, or state replacement validation fails.
    pub fn load_safetensors_bytes(&self, bytes: &[u8]) -> Result<Self> {
        let checkpoint = SafeTensors::deserialize(bytes)?;
        let options = CheckpointOptions::default();
        self.check_keys(checkpoint.names().into_iter(), &options)?;
        let mut loading = Loading::default();
        self.load_shard(&checkpoint, &options, &mut loading)?;
        self.publish(&loading)
    }

    // Loads the checkpoint of a Hugging Face snapshot directory: `model.safetensors`, or the shards `model.safetensors.index.json` names, read one at a time.
    // A `diffusers` snapshot names the same files `diffusion_pytorch_model`.
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
        let mut loading = Loading::default();
        if !index.exists() {
            let bytes = read(directory.join(format!("{stem}.safetensors")))?;
            let checkpoint = SafeTensors::deserialize(&bytes)?;
            self.check_keys(checkpoint.names().into_iter(), options)?;
            self.load_shard(&checkpoint, options, &mut loading)?;
            self.finish_pieces(options, &mut loading)?;
            return self.publish(&loading);
        }

        let index: ShardIndex =
            serde_json::from_slice(&read(index)?).map_err(CheckpointError::Index)?;
        self.check_keys(index.weight_map.keys().map(String::as_str), options)?;
        let mut shards = index.weight_map.values().collect::<Vec<_>>();
        shards.sort();
        shards.dedup();
        for shard in shards {
            let bytes = read(directory.join(shard))?;
            let checkpoint = SafeTensors::deserialize(&bytes)?;
            self.load_shard(&checkpoint, options, &mut loading)?;
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

    // Casts and places each tensor of one shard.
    fn load_shard(
        &self,
        checkpoint: &SafeTensors<'_>,
        options: &CheckpointOptions<'_>,
        loading: &mut Loading,
    ) -> Result<()> {
        for (name, view) in checkpoint.tensors() {
            let piece = options.piece(&name);
            let Some(name) = piece
                .as_ref()
                .map(|piece| piece.state.clone())
                .or_else(|| options.state_name(&name))
            else {
                continue;
            };
            let binding = &self.snapshot.schema.bindings[&name];
            let target = binding.spec().data_type();
            let source = stored_type(view.dtype())?;
            if source != target && !(options.convert && source.is_float() && target.is_float()) {
                return Err(Error::unsupported_operation_with_details(
                    "checkpoint dtype",
                    format!("`{name}` expected {target}, got {source}"),
                ));
            }

            let shape = Shape::create(
                view.shape()
                    .iter()
                    .map(|&dimension| to_i64(dimension, "checkpoint dimension"))
                    .collect::<Result<Vec<_>>>()?,
            )?;
            if let Some(piece) = piece {
                let host =
                    Tensor::from_bytes(Device::Cpu, source, &shape, view.data())?.cast(target)?;
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

            // Casting on the host keeps one copy of a weight on the device, and ties compare before anything uploads.
            let host =
                Tensor::from_bytes(Device::Cpu, source, &shape, view.data())?.cast(target)?;
            if let Some(tied) = loading.tied.get(&binding.id()) {
                if *tied != host.to_bytes()? {
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
                loading.tied.insert(binding.id(), host.to_bytes()?);
            }

            loading
                .tensors
                .insert(binding.id(), host.to_device(device)?);
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

// What a load has gathered: each logical binding's tensor, and the host bytes of those with ties, across shards.
#[derive(Default)]
struct Loading {
    tensors: HashMap<usize, Tensor>,
    tied: HashMap<usize, Vec<u8>>,
    pieces: HashMap<usize, PieceAssembly>,
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

fn read(path: impl AsRef<Path>) -> Result<Vec<u8>> {
    Ok(fs::read(path)?)
}

fn stored_type(dtype: Dtype) -> Result<DataType> {
    Ok(match dtype {
        Dtype::BOOL => DataType::Bool,
        Dtype::U8 => DataType::U8,
        Dtype::I8 => DataType::I8,
        Dtype::U16 => DataType::U16,
        Dtype::I16 => DataType::I16,
        Dtype::U32 => DataType::U32,
        Dtype::I32 => DataType::I32,
        Dtype::U64 => DataType::U64,
        Dtype::I64 => DataType::I64,
        Dtype::F16 => DataType::F16,
        Dtype::BF16 => DataType::Bf16,
        Dtype::F32 => DataType::F32,
        Dtype::F64 => DataType::F64,
        other => {
            return Err(Error::unsupported_operation_with_details(
                "checkpoint dtype",
                format!("{other:?}"),
            ));
        }
    })
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
