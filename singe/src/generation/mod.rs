// Token generation over any decoder that turns tokens into logits, and the decoder machinery a cached model shares.
// Tokens stay on the host between steps, since stop checks and tokenizers read them there.

use std::{
    collections::{BTreeMap, HashMap},
    iter,
};

#[cfg(feature = "cuda")]
use crate::graph::GraphCudaOptions;
use crate::{
    device::Device,
    error::{Error, Result},
    graph::{Graph, GraphExecutor, GraphPrepareOptions, SharedBuffers},
    nn::{
        DecodeStep, DecoderModel, InputGroup, Model, PageLayout, PageTable, TensorGroup,
        caches_from_inputs, named_cache_inputs,
    },
    random::Generator,
    shape::Shape,
    tensor::{SamplingOptions, Tensor},
};

// A decoder that keeps its own cache between steps.
pub trait Decoder {
    // Returns how many sequences each step takes.
    fn sequences(&self) -> usize;

    // Extends each sequence with its `tokens`, and returns the logits of each sequence's last token, `[S, V]`.
    //
    // An empty slice holds a finished sequence, which then costs no capacity, and whose row of logits means nothing.
    // At least one sequence must have tokens, and a step that fails changes nothing.
    fn step(&mut self, tokens: &[&[i64]]) -> Result<Tensor>;

    // Forgets every sequence, so the next step starts at position zero.
    fn reset(&mut self);

    // Forgets one sequence and frees what it holds, so its row can start another while the others run.
    fn release(&mut self, sequence: usize) -> Result<()>;

    // The most extra tokens a step may read, past one token per row, or `None` when a step takes whatever it is given, and a whole prompt reads in one step.
    fn largest_chunk(&self) -> Option<usize> {
        None
    }

    // Summarizes what a compiled decoder prepared, if it did.
    fn report(&self) -> Result<Option<String>> {
        Ok(None)
    }
}

// What to generate and when to stop.
#[derive(Debug, Clone, Default)]
pub struct GenerationOptions {
    // The most tokens to generate for each sequence.
    pub max_new_tokens: usize,
    // Tokens that end a sequence, and are returned as its last token.
    pub stop_tokens: Vec<i64>,
    // How to choose each token, or the most likely one for `None`.
    pub sampling: Option<SamplingOptions>,
    // Tokens never chosen, as Whisper's generation config lists in `suppress_tokens`.
    pub suppressed_tokens: Vec<i64>,
    // Tokens not chosen as a sequence's first, as Whisper's `begin_suppress_tokens`.
    pub first_suppressed_tokens: Vec<i64>,
}

// Generates tokens after `prompt`, calling `on_token` with each as it is chosen, and returns them.
pub fn generate(
    decoder: &mut (impl Decoder + ?Sized),
    prompt: &[i64],
    options: &GenerationOptions,
    generator: Option<&Generator>,
    mut on_token: impl FnMut(i64) -> Result<()>,
) -> Result<Vec<i64>> {
    let mut sequences = generate_batch(decoder, &[prompt], options, generator, |_, token| {
        on_token(token)
    })?;
    sequences
        .pop()
        .ok_or_else(|| Error::missing_output("generated sequence"))
}

// One token a running batch chose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedToken {
    // Which sequence, by the order they joined.
    pub sequence: usize,
    pub token: i64,
    // Whether this is the sequence's last, so its row is free again.
    pub finished: bool,
}

struct RunningSequence {
    id: usize,
    // What its next step reads: its prompt, and then the token last chosen.
    pending: Vec<i64>,
    generated: usize,
}

// A batch whose sequences join and leave while the others run, each in a row of its decoder.
// A finished sequence frees its row and its pages at once, and an idle row costs nothing but its place in a step.
pub struct RunningBatch<'a, D: Decoder + ?Sized> {
    decoder: &'a mut D,
    options: GenerationOptions,
    rows: Vec<Option<RunningSequence>>,
    joined: usize,
    // What each suppression adds to the logits, `[2, V]`: the first row for later tokens, the second for a sequence's first, built at the first step.
    suppression: Option<Tensor>,
}

impl<'a, D: Decoder + ?Sized> RunningBatch<'a, D> {
    // Starts an empty batch, forgetting whatever `decoder` held.
    pub fn create(decoder: &'a mut D, options: GenerationOptions) -> Result<Self> {
        if options.max_new_tokens == 0 {
            return Err(Error::invalid_argument(
                "a running batch generates at least one token per sequence",
            ));
        }

        decoder.reset();
        let rows = (0..decoder.sequences()).map(|_| None).collect();
        Ok(Self {
            decoder,
            options,
            rows,
            joined: 0,
            suppression: None,
        })
    }

    // Starts a sequence after `prompt` in a free row, and returns its id, or `None` when every row is taken.
    pub fn join(&mut self, prompt: &[i64]) -> Result<Option<usize>> {
        self.join_with(prompt, |_, _| Ok(()))
    }

    // Starts a sequence as `join` does, after `prepare` has run with the decoder and the row it takes, as an encoder-decoder writes the row's sources there before its first step.
    // A sequence whose preparation fails does not join, and its row stays free.
    pub fn join_with(
        &mut self,
        prompt: &[i64],
        prepare: impl FnOnce(&D, usize) -> Result<()>,
    ) -> Result<Option<usize>> {
        if prompt.is_empty() {
            return Err(Error::invalid_argument("generation prompt is empty"));
        }

        let Some(index) = self.rows.iter().position(Option::is_none) else {
            return Ok(None);
        };
        prepare(self.decoder, index)?;
        let row = &mut self.rows[index];
        let id = self.joined;
        *row = Some(RunningSequence {
            id,
            pending: prompt.to_vec(),
            generated: 0,
        });
        self.joined += 1;
        Ok(Some(id))
    }

    // Returns whether no sequence is running.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.iter().all(Option::is_none)
    }

    // Runs one step and returns the token each running sequence chose, in row order, which is none for a sequence still reading its prompt.
    // A step that fails, as one that overruns a shared cache does, changes nothing, so the caller can wait for a sequence to finish, or stop.
    pub fn advance(&mut self, generator: Option<&Generator>) -> Result<Vec<GeneratedToken>> {
        if self.is_empty() {
            return Ok(Vec::new());
        }

        let pending = self
            .rows
            .iter()
            .map(|row| row.as_ref().map_or(0, |row| row.pending.len()))
            .collect::<Vec<_>>();
        let portions = step_portions(&pending, self.decoder.largest_chunk());
        let tokens = self
            .rows
            .iter()
            .zip(&portions)
            .map(|(row, &portion)| row.as_ref().map_or(&[][..], |row| &row.pending[..portion]))
            .collect::<Vec<_>>();
        let logits = self.decoder.step(&tokens)?;
        let first = self
            .rows
            .iter()
            .zip(&portions)
            .map(|(row, &portion)| {
                row.as_ref()
                    .is_some_and(|row| row.generated == 0 && row.pending.len() == portion)
            })
            .collect::<Vec<_>>();
        let logits = self.suppress(logits, &first)?;
        let chosen = next_tokens(&logits, &self.options, generator)?;
        if chosen.len() != self.rows.len() {
            return Err(Error::missing_output("generated tokens"));
        }

        let mut generated = Vec::new();
        for (index, ((row, &token), &portion)) in
            self.rows.iter_mut().zip(&chosen).zip(&portions).enumerate()
        {
            let Some(running) = row else {
                continue;
            };
            running.pending.drain(..portion);
            if !running.pending.is_empty() {
                continue;
            }

            running.generated += 1;
            let finished = running.generated == self.options.max_new_tokens
                || self.options.stop_tokens.contains(&token);
            generated.push(GeneratedToken {
                sequence: running.id,
                token,
                finished,
            });
            if finished {
                *row = None;
                self.decoder.release(index)?;
            } else {
                running.pending.push(token);
            }
        }
        Ok(generated)
    }
}

impl<D: Decoder + ?Sized> RunningBatch<'_, D> {
    // Adds negative infinity to the logits of each row's suppressed tokens, with the first-token suppression on the rows in `first`.
    fn suppress(&mut self, logits: Tensor, first: &[bool]) -> Result<Tensor> {
        if self.options.suppressed_tokens.is_empty()
            && self.options.first_suppressed_tokens.is_empty()
        {
            return Ok(logits);
        }

        let vocabulary = logits.shape().dimensions()[1];
        if self.suppression.is_none() {
            let width = usize::try_from(vocabulary)
                .map_err(|_| Error::out_of_range("generation vocabulary"))?;
            let mut table = vec![0.0_f32; 2 * width];
            let mut mask = |token: i64, rows: &[usize]| {
                let index = usize::try_from(token)
                    .ok()
                    .filter(|&index| index < width)
                    .ok_or_else(|| {
                        Error::invalid_argument(format!(
                            "suppressed token {token} is outside a vocabulary of {width}"
                        ))
                    })?;
                for row in rows {
                    table[row * width + index] = f32::NEG_INFINITY;
                }
                Ok::<_, Error>(())
            };
            for &token in &self.options.suppressed_tokens {
                mask(token, &[0, 1])?;
            }

            for &token in &self.options.first_suppressed_tokens {
                mask(token, &[1])?;
            }

            self.suppression = Some(
                Tensor::from_vec(logits.device(), Shape::create([2, vocabulary])?, table)?
                    .cast(logits.data_type())?,
            );
        }

        let rows = first
            .iter()
            .map(|&first| i64::from(first))
            .collect::<Vec<_>>();
        let count =
            i64::try_from(rows.len()).map_err(|_| Error::out_of_range("generation rows"))?;
        let rows = Tensor::from_vec(logits.device(), Shape::create([count])?, rows)?;
        let suppression = self
            .suppression
            .as_ref()
            .ok_or_else(|| Error::missing_output("suppression"))?;
        logits.add(&suppression.index_select(0, &rows)?)
    }
}

// How many of its `pending` tokens each row reads in a step, so the step fits what the decoder compiled: a token per running row, and up to `largest` more, given to the first rows that have them.
fn step_portions(pending: &[usize], largest: Option<usize>) -> Vec<usize> {
    let Some(largest) = largest else {
        return pending.to_vec();
    };

    let spare: usize = pending.iter().map(|tokens| tokens.saturating_sub(1)).sum();
    let mut extra = spare.min(largest);
    pending
        .iter()
        .map(|&tokens| {
            let taken = tokens.saturating_sub(1).min(extra);
            extra -= taken;
            tokens.min(1) + taken
        })
        .collect()
}

// Generates tokens after each of `prompts`, which may differ in length, calling `on_token` with a sequence's index and each token as it is chosen, and returns each sequence's tokens.
//
// A finished sequence frees what it holds, and its row stays idle for the rest of the batch's steps.
pub fn generate_batch(
    decoder: &mut (impl Decoder + ?Sized),
    prompts: &[&[i64]],
    options: &GenerationOptions,
    generator: Option<&Generator>,
    mut on_token: impl FnMut(usize, i64) -> Result<()>,
) -> Result<Vec<Vec<i64>>> {
    if prompts.is_empty() || prompts.iter().any(|prompt| prompt.is_empty()) {
        return Err(Error::invalid_argument("generation prompt is empty"));
    }

    if prompts.len() != decoder.sequences() {
        return Err(Error::length_mismatch(
            "generation prompts",
            decoder.sequences(),
            prompts.len(),
        ));
    }

    let mut tokens = vec![Vec::new(); prompts.len()];
    if options.max_new_tokens == 0 {
        decoder.reset();
        return Ok(tokens);
    }

    let mut batch = RunningBatch::create(decoder, options.clone())?;
    for prompt in prompts {
        batch.join(prompt)?;
    }

    while !batch.is_empty() {
        for generated in batch.advance(generator)? {
            tokens[generated.sequence].push(generated.token);
            on_token(generated.sequence, generated.token)?;
        }
    }
    Ok(tokens)
}

// Chooses a token for each row of `logits`, `[S, V]`.
fn next_tokens(
    logits: &Tensor,
    options: &GenerationOptions,
    generator: Option<&Generator>,
) -> Result<Vec<i64>> {
    options
        .sampling
        .as_ref()
        .map_or_else(
            || logits.argmax(-1),
            |sampling| logits.sample(sampling, generator),
        )?
        .to_vec::<i64>()
}

// The steps past one token per row that a compiled decoder compiles, by the extra tokens they read: `smallest`, then four times as many each time, up to `largest` and to one less than a sequence's capacity, which is always the last.
// A step pads to the smallest compiled one that holds it, so a prompt up to the largest reads in one step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromptChunks {
    pub smallest: usize,
    pub largest: usize,
}

impl PromptChunks {
    // Compiles only the step of one token per row, so a prompt reads a token per step.
    pub const NONE: Self = Self {
        smallest: 0,
        largest: 0,
    };

    // Prompts read in steps of at most 1024 tokens, the larger compiled only when the capacity reaches them.
    pub const STANDARD: Self = Self {
        smallest: 16,
        largest: 1024,
    };

    // Returns the extra tokens of each step to compile for sequences of `capacity` tokens, smallest first.
    #[must_use]
    pub fn sizes(self, capacity: i64) -> Vec<usize> {
        let limit = self
            .largest
            .min(usize::try_from(capacity.saturating_sub(1)).unwrap_or(0));
        if limit == 0 {
            return Vec::new();
        }

        let mut sizes = Vec::new();
        let mut size = self.smallest.max(1);
        while size < limit {
            sizes.push(size);
            size = size.saturating_mul(4);
        }

        sizes.push(limit);
        sizes
    }
}

impl Default for PromptChunks {
    fn default() -> Self {
        Self::STANDARD
    }
}

// How much a [`CachedDecoder`] holds, and which steps it compiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecoderOptions {
    // How many sequences each step takes.
    pub sequences: usize,
    // The most tokens each sequence holds, prompt included.
    pub capacity: i64,
    // The slots per cache page.
    pub page_size: i64,
    // The tokens a cache holds across its sequences, or enough for every sequence's capacity at once.
    pub cache_tokens: Option<i64>,
    // The prompt steps a compiled decoder compiles beside the step of one token per row.
    pub prompt_chunks: PromptChunks,
}

impl DecoderOptions {
    #[must_use]
    pub const fn new(sequences: usize, capacity: i64) -> Self {
        Self {
            sequences,
            capacity,
            page_size: 32,
            cache_tokens: None,
            prompt_chunks: PromptChunks::STANDARD,
        }
    }

    #[must_use]
    pub const fn with_page_size(mut self, page_size: i64) -> Self {
        self.page_size = page_size;
        self
    }

    // Shares a cache of `cache_tokens` tokens among the sequences, which a step overrunning it fails on and changes nothing.
    #[must_use]
    pub const fn with_cache_tokens(mut self, cache_tokens: i64) -> Self {
        self.cache_tokens = Some(cache_tokens);
        self
    }

    #[must_use]
    pub const fn with_prompt_chunks(mut self, prompt_chunks: PromptChunks) -> Self {
        self.prompt_chunks = prompt_chunks;
        self
    }

    // Returns the extra tokens of each prompt step a compiled decoder compiles, smallest first.
    #[must_use]
    pub fn chunk_sizes(&self) -> Vec<usize> {
        self.prompt_chunks.sizes(self.capacity)
    }

    // Plans the pages of a decoder with these options, with room in each step's page table to pad it up to the largest prompt step.
    pub fn page_table(&self) -> Result<PageTable> {
        let table = match self.cache_tokens {
            Some(cache_tokens) => PageTable::create_shared(
                self.sequences,
                self.capacity,
                self.page_size,
                cache_tokens,
            )?,
            None => PageTable::create(self.sequences, self.capacity, self.page_size)?,
        };
        table.with_padding(self.chunk_sizes().last().copied().unwrap_or(0))
    }
}

// One decode step as a graph, the inputs it reads, and the logits it computed on them.
#[derive(Debug)]
pub struct DecoderStepGraph {
    pub graph: Graph,
    pub inputs: HashMap<String, Tensor>,
    pub logits: Tensor,
    // Where the step's tokens fall in its paged cache, which `CudaExport::with_decoding` writes a generation loop from.
    pub layout: PageLayout,
}

// A model's generation state: its pages and caches, and optionally compiled steps, by the tokens they read.
//
// Once compiled, a decoder runs only steps it compiled: a step pads to the smallest compiled one that holds it, and a larger step fails.
#[derive(Debug)]
pub struct CachedDecoder<L: DecoderModel> {
    model: Model<L>,
    device: Device,
    pages: PageTable,
    tables: L::Tables,
    caches: Vec<L::Cache>,
    chunks: Vec<usize>,
    compiled: BTreeMap<usize, GraphExecutor>,
}

impl<L: DecoderModel> CachedDecoder<L> {
    // Creates a decoder on the device `model`'s state is on, in the type of its weights.
    pub fn create(model: &Model<L>, options: DecoderOptions) -> Result<Self> {
        let device = model.state().device()?;
        let data_type = model.state().floating_type()?;
        let pages = options.page_table()?;
        Ok(Self {
            model: model.clone(),
            device,
            tables: model.layer().tables(data_type, device, options.capacity)?,
            caches: model
                .layer()
                .caches(data_type, device, pages.cache_size()?)?,
            pages,
            chunks: options.chunk_sizes(),
            compiled: BTreeMap::new(),
        })
    }

    // Compiles the step of one token per sequence and a step for each prompt chunk the options asked for, which every later step then runs.
    //
    // Preparing freezes bound state, and the cache changes on every step, so the cache is an input the step reads and writes in place.
    pub fn compile(&mut self) -> Result<()> {
        self.compile_under(GraphPrepareOptions::new(self.device))
    }

    // Compiles the steps as `compile` does, under `cuda`, whose records hold for a step only if they were tuned on `chunk_step_graph` of its size.
    #[cfg(feature = "cuda")]
    pub fn compile_with(&mut self, cuda: GraphCudaOptions) -> Result<()> {
        self.compile_under(GraphPrepareOptions::new(self.device).with_cuda_options(cuda))
    }

    fn compile_under(&mut self, options: GraphPrepareOptions) -> Result<()> {
        // Only one step runs at a time, so the steps share their buffers.
        let options = options.sharing_buffers(&SharedBuffers::default());
        let mut compiled = BTreeMap::new();
        for &chunk in iter::once(&0).chain(&self.chunks) {
            let tokens = self.pages.sequences() + chunk;
            compiled.insert(tokens, self.compile_step(options.clone(), tokens)?);
        }

        self.compiled = compiled;
        Ok(())
    }

    fn compile_step(&self, options: GraphPrepareOptions, tokens: usize) -> Result<GraphExecutor> {
        let (layer, real) = (self.model.layer(), self.model.state());
        let pages = &self.pages;
        let data_type = real.floating_type()?;
        // Capturing on metadata runs no kernels and leaves the real cache untouched.
        let tables = layer.tables(data_type, Device::Meta, pages.capacity())?;
        let state = real
            .to(Device::Meta, None)?
            .with_buffers(tables.named_tensors())?;
        let sequences =
            i64::try_from(pages.sequences()).map_err(|_| Error::out_of_range("sequences"))?;
        let tokens = i64::try_from(tokens).map_err(|_| Error::out_of_range("step tokens"))?;
        let caches = layer.caches(data_type, Device::Meta, pages.cache_size()?)?;
        let inputs = DecodeStep::meta(tokens, sequences, pages.table_length()?)?
            .named_tensors()
            .into_iter()
            .chain(named_cache_inputs(&caches));
        let (graph, _) =
            Graph::capture_with_bindings(inputs, state.capture_bindings()?, |inputs| {
                let (step, captured) = inputs.split_at(DecodeStep::COUNT);
                let step = DecodeStep::from_tensors(step)?;
                let caches = caches_from_inputs(&caches, captured)?;
                Ok(vec![layer.forward(&step, &tables, &caches, &state)?])
            })?;
        let bound = real.with_buffers(self.tables.named_tensors())?;
        let options = options.with_inputs_in_place(named_cache_inputs(&self.caches));
        bound.program(&graph)?.prepare(bound.snapshot(), options)
    }

    // The step of one token per sequence over this model's weights, with inputs of its own, which tuning and exporting take.
    // Its keys and layouts are those of the step this decoder compiles, so records tuned on it hold there.
    pub fn step_graph(&self) -> Result<DecoderStepGraph> {
        self.chunk_step_graph(0)
    }

    // The step that reads `chunk` extra tokens, as `step_graph` gives the step that reads none.
    pub fn chunk_step_graph(&self, chunk: usize) -> Result<DecoderStepGraph> {
        let layer = self.model.layer();
        let data_type = self.model.state().floating_type()?;
        let caches = layer.caches(data_type, self.device, self.pages.cache_size()?)?;
        let mut pages = self.pages.clone();
        pages.reset();
        let first = vec![0_i64; chunk + 1];
        let tokens = iter::once(first.as_slice())
            .chain(iter::repeat_n(
                &[0_i64][..],
                pages.sequences().saturating_sub(1),
            ))
            .collect::<Vec<_>>();
        let step = pages.step(self.device, &tokens)?;
        let inputs = step
            .named_tensors()
            .into_iter()
            .chain(named_cache_inputs(&caches))
            .collect::<Vec<_>>();
        let state = self
            .model
            .state()
            .with_buffers(self.tables.named_tensors())?;
        let (graph, outputs) =
            Graph::capture_with_bindings(inputs, state.capture_bindings()?, |inputs| {
                let (step, captured) = inputs.split_at(DecodeStep::COUNT);
                let step = DecodeStep::from_tensors(step)?;
                let caches = caches_from_inputs(&caches, captured)?;
                Ok(vec![layer.forward(&step, &self.tables, &caches, &state)?])
            })?;
        let logits = outputs
            .into_iter()
            .next()
            .ok_or_else(|| Error::missing_output("decoder step graph"))?;
        // Capturing ran the step, and tuning must compare a candidate's writes against caches no step has written.
        for (_, cache) in named_cache_inputs(&caches) {
            cache.fill_(0.0)?;
        }

        Ok(DecoderStepGraph {
            layout: pages.layout()?,
            graph,
            inputs: step
                .named_tensors()
                .into_iter()
                .chain(named_cache_inputs(&caches))
                .collect(),
            logits,
        })
    }

    // Takes the step `other` compiled, for this decoder's own caches, so a new sequence does not prepare again.
    pub fn share_compiled(&mut self, other: &Self) -> Result<()> {
        if other.compiled.is_empty() {
            return Err(Error::invalid_argument(
                "decoder has no compiled step to share",
            ));
        }

        if self.pages.sequences() != other.pages.sequences()
            || self.pages.capacity() != other.pages.capacity()
            || self.pages.page_size() != other.pages.page_size()
            || self.pages.table_length()? != other.pages.table_length()?
            || self.chunks != other.chunks
        {
            return Err(Error::invalid_argument(
                "decoders share a compiled step only at one size",
            ));
        }

        if self.device != other.device
            || self.model.state().snapshot().revision() != other.model.state().snapshot().revision()
        {
            return Err(Error::invalid_argument(
                "decoders share a compiled step only over the same weights on one device",
            ));
        }

        // This decoder's steps run one at a time too, though not with the other decoder's.
        let buffers = SharedBuffers::default();
        self.compiled = other
            .compiled
            .iter()
            .map(|(&tokens, executor)| {
                let instance =
                    executor.create_instance_sharing(named_cache_inputs(&self.caches), &buffers)?;
                Ok((tokens, instance))
            })
            .collect::<Result<_>>()?;
        Ok(())
    }

    // The executor of the compiled step of one token per sequence, whose report says how many specializations it needed.
    #[must_use]
    pub fn executor(&self) -> Option<&GraphExecutor> {
        self.compiled.get(&self.pages.sequences())
    }

    // Every compiled step's executor.
    pub fn executors(&self) -> impl Iterator<Item = &GraphExecutor> {
        self.compiled.values()
    }

    // Returns each block's cache, which a model may write outside a step, such as an encoder-decoder's cross-attention keys and values.
    #[must_use]
    pub fn caches(&self) -> &[L::Cache] {
        &self.caches
    }

    // Returns the model this decoder runs.
    #[must_use]
    pub const fn model(&self) -> &Model<L> {
        &self.model
    }

    // Returns the slots per cache page.
    #[must_use]
    pub const fn page_size(&self) -> i64 {
        self.pages.page_size()
    }
}

impl<L: DecoderModel> Decoder for CachedDecoder<L> {
    fn sequences(&self) -> usize {
        self.pages.sequences()
    }

    fn step(&mut self, tokens: &[&[i64]]) -> Result<Tensor> {
        // A failed step leaves the table where the cache's contents are.
        let mut pages = self.pages.clone();
        let logits = if self.compiled.is_empty() {
            let step = pages.step(self.device, tokens)?;
            self.model
                .layer()
                .forward(&step, &self.tables, &self.caches, self.model.state())?
        } else {
            let total = PageTable::step_tokens(tokens);
            let largest = self.compiled.keys().last().copied().unwrap_or(0);
            let (&size, executor) = self.compiled.range_mut(total..).next().ok_or_else(|| {
                Error::invalid_argument(format!(
                    "a decode step of {total} tokens is larger than the largest compiled step, of {largest}"
                ))
            })?;
            let step = pages.step_padded(self.device, tokens, size)?;
            let inputs = step
                .named_tensors()
                .into_iter()
                .chain(named_cache_inputs(&self.caches))
                .collect::<HashMap<_, _>>();
            executor
                .run(&inputs)?
                .pop()
                .ok_or_else(|| Error::missing_output("compiled decode step"))?
        };
        self.pages = pages;
        Ok(logits)
    }

    fn reset(&mut self) {
        self.pages.reset();
    }

    fn release(&mut self, sequence: usize) -> Result<()> {
        self.pages.release(sequence)
    }

    fn largest_chunk(&self) -> Option<usize> {
        let sequences = self.pages.sequences();
        self.compiled.keys().last().map(|tokens| tokens - sequences)
    }

    fn report(&self) -> Result<Option<String>> {
        if self.compiled.is_empty() {
            return Ok(None);
        }

        let reports = self
            .compiled
            .values()
            .map(GraphExecutor::report)
            .collect::<Result<Vec<_>>>()?;
        Ok(Some(reports.join("\n")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::Shape;

    // Answers each step with logits favouring a token one above each sequence's last.
    struct Counting {
        sequences: usize,
        steps: Vec<Vec<Vec<i64>>>,
        resets: usize,
        released: Vec<usize>,
        largest: Option<usize>,
    }

    impl Counting {
        fn new(sequences: usize) -> Self {
            Self {
                sequences,
                steps: Vec::new(),
                resets: 0,
                released: Vec::new(),
                largest: None,
            }
        }
    }

    impl Decoder for Counting {
        fn sequences(&self) -> usize {
            self.sequences
        }

        fn step(&mut self, tokens: &[&[i64]]) -> Result<Tensor> {
            self.steps
                .push(tokens.iter().map(|tokens| tokens.to_vec()).collect());
            let mut logits = vec![0.0_f32; tokens.len() * 8];
            for (row, tokens) in tokens.iter().enumerate() {
                let best = tokens.last().copied().unwrap_or_default() + 1;
                logits[row * 8 + usize::try_from(best).unwrap() % 8] = 1.0;
            }
            Tensor::from_vec(
                Device::Cpu,
                Shape::create([i64::try_from(tokens.len()).unwrap(), 8]).unwrap(),
                logits,
            )
        }

        fn largest_chunk(&self) -> Option<usize> {
            self.largest
        }

        fn reset(&mut self) {
            self.resets += 1;
        }

        fn release(&mut self, sequence: usize) -> Result<()> {
            self.released.push(sequence);
            Ok(())
        }
    }

    fn run(prompt: &[i64], max_new_tokens: usize, stop_tokens: Vec<i64>) -> (Vec<i64>, Counting) {
        let mut decoder = Counting::new(1);
        let options = GenerationOptions {
            max_new_tokens,
            stop_tokens,
            ..GenerationOptions::default()
        };
        let mut seen = Vec::new();
        let tokens = generate(&mut decoder, prompt, &options, None, |token| {
            seen.push(token);
            Ok(())
        })
        .unwrap();
        assert_eq!(seen, tokens);
        (tokens, decoder)
    }

    #[test]
    fn test_a_step_reads_up_to_the_largest_chunk() {
        assert_eq!(step_portions(&[5, 1], None), [5, 1]);
        assert_eq!(step_portions(&[5, 1], Some(3)), [4, 1]);
        assert_eq!(step_portions(&[5, 1], Some(8)), [5, 1]);
        assert_eq!(step_portions(&[2, 1], Some(0)), [1, 1]);
        // A chunk spreads over the rows that have tokens to spare, and an idle row reads nothing.
        assert_eq!(step_portions(&[3, 0, 3], Some(3)), [3, 0, 2]);
        assert_eq!(step_portions(&[1, 1], Some(1)), [1, 1]);
    }

    #[test]
    fn test_prompt_chunks_grow_by_four_up_to_the_capacity() {
        let chunks = PromptChunks::default();
        assert_eq!(chunks.sizes(512), [16, 64, 256, 511]);
        assert_eq!(chunks.sizes(5000), [16, 64, 256, 1024]);
        assert_eq!(chunks.sizes(10), [9]);
        assert!(chunks.sizes(1).is_empty());
        assert!(PromptChunks::NONE.sizes(512).is_empty());
        let exact = PromptChunks {
            smallest: 16,
            largest: 3,
        };
        assert_eq!(exact.sizes(448), [3]);
        let small = PromptChunks {
            smallest: 1,
            largest: 3,
        };
        assert_eq!(small.sizes(32), [1, 3]);
    }

    // Whatever a prompt's length, no step reads more than the largest chunk past a token per row, and the next token follows the prompt's last.
    #[test]
    fn test_a_prompt_of_any_length_reads_in_steps_up_to_the_largest_chunk() {
        for length in 1..=20 {
            let prompt = (0..length).collect::<Vec<i64>>();
            let mut decoder = Counting::new(2);
            decoder.largest = Some(5);
            let options = GenerationOptions {
                max_new_tokens: 2,
                ..GenerationOptions::default()
            };
            let mut batch = RunningBatch::create(&mut decoder, options).unwrap();
            batch.join(&prompt).unwrap();
            let mut generated = Vec::new();
            while !batch.is_empty() {
                generated.extend(batch.advance(None).unwrap());
            }
            assert_eq!(generated.len(), 2);
            assert_eq!(generated[0].token, length % 8);
            let mut read = Vec::new();
            for step in &decoder.steps {
                let total = step.iter().map(|tokens| tokens.len().max(1)).sum::<usize>();
                assert!(total <= 7, "{step:?}");
                read.extend_from_slice(&step[0]);
            }
            assert_eq!(read[..prompt.len()], prompt);
            // A prompt reads in as few steps as the largest chunk allows.
            assert_eq!(
                decoder.steps.len(),
                usize::try_from(length).unwrap().div_ceil(6) + 1
            );
        }
    }

    #[test]
    fn test_generation_runs_the_prompt_then_one_token_at_a_time() {
        let (tokens, decoder) = run(&[0, 1], 3, Vec::new());
        assert_eq!(tokens, [2, 3, 4]);
        assert_eq!(
            decoder.steps,
            [vec![vec![0, 1]], vec![vec![2]], vec![vec![3]]]
        );
        assert_eq!(decoder.resets, 1);
    }

    // A suppressed token is never chosen, and a first-suppressed one only after the first token, as Whisper's generation config asks.
    #[test]
    fn test_suppressed_tokens_are_not_chosen() {
        let mut decoder = Counting::new(1);
        let options = GenerationOptions {
            max_new_tokens: 3,
            suppressed_tokens: vec![2],
            first_suppressed_tokens: vec![1],
            ..GenerationOptions::default()
        };
        let tokens = generate(&mut decoder, &[0], &options, None, |_| Ok(())).unwrap();
        // The favoured token is one above the last; with it suppressed, the first token left wins.
        assert_eq!(tokens, [0, 1, 0]);
        let options = GenerationOptions {
            max_new_tokens: 1,
            suppressed_tokens: vec![8],
            ..GenerationOptions::default()
        };
        let error = generate(&mut decoder, &[0], &options, None, |_| Ok(())).unwrap_err();
        assert!(
            error.to_string().contains("outside a vocabulary"),
            "{error}"
        );
    }

    #[test]
    fn test_generation_returns_the_stop_token_and_runs_no_further() {
        let (tokens, decoder) = run(&[0], 5, vec![2]);
        assert_eq!(tokens, [1, 2]);
        assert_eq!(decoder.steps.len(), 2);
    }

    // `usize::MAX` means until a stop token, so it must not reserve anything up front.
    #[test]
    fn test_generation_reserves_nothing_for_its_limit() {
        let (tokens, _) = run(&[0], usize::MAX, vec![2]);
        assert_eq!(tokens, [1, 2]);
    }

    #[test]
    fn test_generating_nothing_runs_nothing() {
        let (tokens, decoder) = run(&[0], 0, Vec::new());
        assert!(tokens.is_empty());
        assert!(decoder.steps.is_empty());
    }

    #[test]
    fn test_generation_refuses_an_empty_prompt_and_the_wrong_batch() {
        let mut decoder = Counting::new(2);
        let options = GenerationOptions {
            max_new_tokens: 1,
            ..GenerationOptions::default()
        };
        assert!(generate_batch(&mut decoder, &[&[1], &[]], &options, None, |_, _| Ok(())).is_err());
        assert!(generate_batch(&mut decoder, &[&[1]], &options, None, |_, _| Ok(())).is_err());
    }

    // A finished sequence holds for the rest of the batch's steps and keeps nothing more.
    #[test]
    fn test_batches_finish_each_sequence_on_its_own_and_keep_stepping() {
        let mut decoder = Counting::new(2);
        let options = GenerationOptions {
            max_new_tokens: 3,
            stop_tokens: vec![2],
            sampling: None,
            ..GenerationOptions::default()
        };
        let mut seen = Vec::new();
        let tokens = generate_batch(
            &mut decoder,
            &[&[1], &[3, 4]],
            &options,
            None,
            |sequence, token| {
                seen.push((sequence, token));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(tokens, [vec![2], vec![5, 6, 7]]);
        assert_eq!(seen[0], (0, 2));
        assert_eq!(
            seen.iter().filter(|(sequence, _)| *sequence == 0).count(),
            1
        );
        assert_eq!(decoder.steps[1], [vec![], vec![5]]);
        assert_eq!(decoder.steps[2], [vec![], vec![6]]);
    }

    #[test]
    fn test_a_sequence_joins_the_row_another_left_while_the_rest_run() {
        let mut decoder = Counting::new(2);
        let options = GenerationOptions {
            max_new_tokens: 2,
            ..GenerationOptions::default()
        };
        let mut batch = RunningBatch::create(&mut decoder, options).unwrap();
        assert_eq!(batch.join(&[0]).unwrap(), Some(0));
        batch.advance(None).unwrap();
        assert_eq!(batch.join(&[4, 5]).unwrap(), Some(1));
        assert_eq!(batch.join(&[6]).unwrap(), None);
        // The first takes its last token, and leaves its row to a third.
        let tokens = batch.advance(None).unwrap();
        assert_eq!(tokens[0].sequence, 0);
        assert!(tokens[0].finished);
        assert_eq!(batch.join(&[6]).unwrap(), Some(2));
        let tokens = batch.advance(None).unwrap();
        assert_eq!(
            tokens
                .iter()
                .map(|token| token.sequence)
                .collect::<Vec<_>>(),
            [2, 1]
        );
        assert!(tokens[1].finished);
        let tokens = batch.advance(None).unwrap();
        assert_eq!(tokens[0].sequence, 2);
        assert!(batch.is_empty());
        assert!(batch.advance(None).unwrap().is_empty());
        assert_eq!(decoder.released, [0, 1, 0]);
        assert_eq!(
            decoder.steps,
            [
                vec![vec![0], vec![]],
                vec![vec![1], vec![4, 5]],
                vec![vec![6], vec![6]],
                vec![vec![7], vec![]],
            ]
        );
    }
}
