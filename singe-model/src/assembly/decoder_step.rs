// What an encoder-decoder's decoder reads in a cached step: each block's keys and values per sequence, written at each row's position, its cross-attention's keys and values over each sequence's encoder states, and the masks that keep a row from reading past its own position.
// Every sequence holds its rows densely, so one step's shapes serve every position, and beams take their parents' rows by reordering them.

use singe::{
    DataType, Device, Error as SingeError, Result as SingeResult, Shape, Tensor,
    graph::capture,
    index::TensorIndex,
    nn::{CacheSize, DecodeCache, DecodeStep, TensorGroup},
};

// What sizes a `Seq2SeqCache` besides its sequences and capacity: its attention's heads and their width, and the most encoder states a sequence's cross-attention reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Seq2SeqCacheSpec {
    pub heads: i64,
    pub head_dim: i64,
    pub sources: i64,
}

// One decoder block's caches: its own keys and values, `[S, capacity + 1, H, D]`, whose last slot takes the writes of rows that hold or pad, and its cross-attention's over each sequence's encoder states, `[S, sources, H, D]`, with how many each sequence has, `[S]` of `I32`.
#[derive(Debug, Clone)]
pub struct Seq2SeqCache {
    keys: Tensor,
    values: Tensor,
    source_keys: Tensor,
    source_values: Tensor,
    source_lengths: Tensor,
}

impl Seq2SeqCache {
    // Writes the keys and values of a step's rows, `[T, H, D]` each, at their slots.
    pub fn write(&self, rows: &StepRows, keys: &Tensor, values: &Tensor) -> SingeResult<()> {
        let at = [
            TensorIndex::Indices(rows.sequences.clone()),
            TensorIndex::Indices(rows.slots.clone()),
        ];
        self.keys.index_put_(&at, keys)?;
        self.values.index_put_(&at, values)
    }

    // Each row's sequence's keys and values at every position, `[T, H, capacity, D]` each.
    pub fn read(&self, rows: &StepRows) -> SingeResult<(Tensor, Tensor)> {
        let &[_, slots, _, _] = self.keys.shape().dimensions() else {
            return Err(SingeError::invalid_argument("a decoder cache's keys"));
        };
        let read = |cache: &Tensor| {
            cache
                .index_select(0, &rows.sequences)?
                .narrow(1, 0, slots - 1)?
                .transpose(1, 2)
        };
        Ok((read(&self.keys)?, read(&self.values)?))
    }

    // Writes sequence `sequence`'s cross-attention keys and values, `[F, H, D]` each, and its count of encoder states.
    pub fn write_sources(
        &self,
        sequence: usize,
        keys: &Tensor,
        values: &Tensor,
    ) -> SingeResult<()> {
        let sequence = i64::try_from(sequence).map_err(|_| SingeError::out_of_range("sequence"))?;
        let &[count, _, _] = keys.shape().dimensions() else {
            return Err(SingeError::invalid_argument(
                "cross-attention keys and values are [sources, heads, dim]",
            ));
        };
        let at = [
            TensorIndex::Single(sequence),
            TensorIndex::Slice {
                start: Some(0),
                end: Some(count),
                step: 1,
            },
        ];
        self.source_keys.index_put_(&at, keys)?;
        self.source_values.index_put_(&at, values)?;
        let length = i32::try_from(count).map_err(|_| SingeError::out_of_range("sources"))?;
        self.source_lengths.index_put_(
            &[TensorIndex::Slice {
                start: Some(sequence),
                end: Some(sequence + 1),
                step: 1,
            }],
            &Tensor::from_vec(
                self.source_lengths.device(),
                Shape::create([1])?,
                vec![length],
            )?,
        )
    }

    // Each row's sequence's cross-attention keys and values, `[T, H, sources, D]` each, and how many of them it reads, `[T]` of `I32`.
    pub fn read_sources(&self, rows: &StepRows) -> SingeResult<(Tensor, Tensor, Tensor)> {
        let read = |cache: &Tensor| cache.index_select(0, &rows.sequences)?.transpose(1, 2);
        Ok((
            read(&self.source_keys)?,
            read(&self.source_values)?,
            self.source_lengths.index_select(0, &rows.sequences)?,
        ))
    }

    // Gives each sequence the keys and values sequence `parents[s]` held, `[S]` of `I64`, as beams continue their parents.
    pub fn reorder(&self, parents: &Tensor) -> SingeResult<()> {
        self.keys.assign(&self.keys.index_select(0, parents)?)?;
        self.values.assign(&self.values.index_select(0, parents)?)
    }
}

impl TensorGroup for Seq2SeqCache {
    fn named_tensors(&self) -> Vec<(String, Tensor)> {
        [
            ("keys", &self.keys),
            ("values", &self.values),
            ("source_keys", &self.source_keys),
            ("source_values", &self.source_values),
            ("source_lengths", &self.source_lengths),
        ]
        .into_iter()
        .map(|(name, tensor)| (name.to_owned(), tensor.clone()))
        .collect()
    }
}

impl DecodeCache for Seq2SeqCache {
    type Spec = Seq2SeqCacheSpec;

    fn create(
        device: Device,
        data_type: DataType,
        size: CacheSize,
        spec: Seq2SeqCacheSpec,
    ) -> SingeResult<Self> {
        let zeroed =
            |dimensions: [i64; 4]| Tensor::zeroed_on(device, data_type, Shape::create(dimensions)?);
        let own = [size.sequences, size.capacity + 1, spec.heads, spec.head_dim];
        let sources = [size.sequences, spec.sources, spec.heads, spec.head_dim];
        Ok(Self {
            keys: zeroed(own)?,
            values: zeroed(own)?,
            source_keys: zeroed(sources)?,
            source_values: zeroed(sources)?,
            source_lengths: Tensor::zeroed_on(
                device,
                DataType::I32,
                Shape::create([size.sequences])?,
            )?,
        })
    }

    fn rebuilt(&self, tensors: &[Tensor]) -> SingeResult<Self> {
        let [keys, values, source_keys, source_values, source_lengths] = tensors else {
            return Err(SingeError::missing_input(
                "an encoder-decoder cache's tensors",
            ));
        };
        Ok(Self {
            keys: keys.clone(),
            values: values.clone(),
            source_keys: source_keys.clone(),
            source_values: source_values.clone(),
            source_lengths: source_lengths.clone(),
        })
    }
}

// What a decoder looks up by position for a capacity: the mask of each query position over every key position, `[C, C]`, negative infinity where the key comes after the query, and, for T5, each pair's relative position bucket, `[C, C]` of `I64`.
#[derive(Debug, Clone)]
pub struct Seq2SeqTables {
    causal: Tensor,
    buckets: Option<Tensor>,
}

impl Seq2SeqTables {
    // The tables for `capacity` positions, with the bucket `bucket` gives each key's position less its query's, when given.
    pub fn create(
        data_type: DataType,
        device: Device,
        capacity: i64,
        bucket: Option<&dyn Fn(i64) -> i64>,
    ) -> SingeResult<Self> {
        let shape = Shape::create([capacity, capacity])?;
        if device.is_meta() {
            return Ok(Self {
                causal: Tensor::zeroed_on(device, data_type, shape.clone())?,
                buckets: bucket
                    .map(|_| Tensor::zeroed_on(device, DataType::I64, shape.clone()))
                    .transpose()?,
            });
        }

        let pairs = || (0..capacity).flat_map(|query| (0..capacity).map(move |key| key - query));
        let causal = pairs()
            .map(|relative| if relative > 0 { f32::NEG_INFINITY } else { 0.0 })
            .collect::<Vec<_>>();
        Ok(Self {
            causal: Tensor::from_vec(device, shape.clone(), causal)?.cast(data_type)?,
            buckets: bucket
                .map(|bucket| {
                    Tensor::from_vec(device, shape.clone(), pairs().map(bucket).collect())
                })
                .transpose()?,
        })
    }

    pub fn buckets(&self) -> SingeResult<&Tensor> {
        self.buckets
            .as_ref()
            .ok_or_else(|| SingeError::missing_input("relative position buckets"))
    }
}

impl TensorGroup for Seq2SeqTables {
    fn named_tensors(&self) -> Vec<(String, Tensor)> {
        let mut tensors = vec![("causal".to_owned(), self.causal.clone())];
        if let Some(buckets) = &self.buckets {
            tensors.push(("buckets".to_owned(), buckets.clone()));
        }
        tensors
    }
}

// Where a decode step's rows read and write their sequences' caches: the sequence of each row, its position, the slot it writes, the capacity for a row that holds or pads, and its mask over its sequence's keys, `[T, 1, 1, C]`.
#[derive(Debug, Clone)]
pub struct StepRows {
    sequences: Tensor,
    positions: Tensor,
    slots: Tensor,
    mask: Tensor,
}

impl StepRows {
    pub fn create(step: &DecodeStep, tables: &Seq2SeqTables) -> SingeResult<Self> {
        let (sequences, positions) = (step.sequence_ids(), step.positions());
        let (&[rows], &[capacity, _]) = (
            positions.shape().dimensions(),
            tables.causal.shape().dimensions(),
        ) else {
            return Err(SingeError::invalid_argument("a decode step's positions"));
        };
        let device = positions.device();
        let order = capture::literal(device, |device| Tensor::arange(device, 0, rows, 1))?;
        let scratch = capture::literal(device, |device| {
            Tensor::from_vec(
                device,
                Shape::create([rows])?,
                vec![
                    capacity;
                    usize::try_from(rows).map_err(|_| SingeError::out_of_range("rows"))?
                ],
            )
        })?;
        // A sequence's own rows come first among its rows, and padding follows them.
        let ends = step
            .paging()
            .query_offsets
            .index_select(0, sequences)?
            .add(&step.real_tokens().index_select(0, sequences)?)?;
        Ok(Self {
            sequences: sequences.clone(),
            positions: positions.clone(),
            slots: Tensor::where_select(&order.less(&ends)?, positions, &scratch)?,
            mask: tables
                .causal
                .index_select(0, positions)?
                .reshape([rows, 1, 1, capacity])?,
        })
    }

    #[must_use]
    pub const fn positions(&self) -> &Tensor {
        &self.positions
    }

    #[must_use]
    pub const fn mask(&self) -> &Tensor {
        &self.mask
    }
}
