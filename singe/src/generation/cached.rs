use std::{collections::HashMap, iter};

#[cfg(feature = "cuda")]
use crate::graph::GraphCudaOptions;
use crate::{
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    generation::{
        Decoder, GenerationOptions, choose,
        compiled::{CompiledSteps, capture_step},
    },
    graph::{Graph, GraphExecutor, GraphPrepareOptions, SharedBuffers},
    nn::{
        DecodeCache, DecodeStep, DecoderModel, Model, ModelState, PageLayout, PageTable,
        TensorGroup, caches_from_inputs, named_cache_inputs,
    },
    shape::Shape,
    tensor::{Draws, Tensor},
};

// The steps past one token per row that a compiled decoder compiles, by the extra tokens they read: `smallest`, then twice as many each time, up to `largest` and to one less than a sequence's capacity, which is always the last.
// A step pads to the smallest compiled one that holds it, so a prompt up to the largest reads in one step, padded to less than twice its rows.
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
            size = size.saturating_mul(2);
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

// One decode step as a graph, the inputs it reads, and what it computed on them: the logits of each sequence's last token, or with a choice, the token each sequence chose.
#[derive(Debug)]
pub struct DecoderStepGraph {
    pub graph: Graph,
    pub inputs: HashMap<String, Tensor>,
    pub output: Tensor,
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
    compiled: Option<(CompiledSteps, GraphPrepareOptions)>,
    unfollowed: Tensor,
}

const PREVIOUS: &str = "previous";

const FIRST_TOKENS: &str = "first_tokens";
const STREAMS: &str = "streams";

fn previous_tokens(device: Device, sequences: usize) -> Result<Tensor> {
    let sequences = i64::try_from(sequences).map_err(|_| Error::out_of_range("sequences"))?;
    Tensor::zeroed_on(device, DataType::I64, Shape::create([sequences])?)
}

fn step_forward<L: DecoderModel>(
    layer: &L,
    choice: Option<&GenerationOptions>,
    step: &DecodeStep,
    tables: &L::Tables,
    caches: &[L::Cache],
    extra: &[Tensor],
    state: &ModelState,
) -> Result<Vec<Tensor>> {
    // A step reads the tokens the step before chose, then `first_tokens` when it chooses, and `streams` when it samples.
    let [previous, rest @ ..] = extra else {
        return Err(Error::missing_input("the tokens the step before chose"));
    };
    let logits = layer.forward(&step.follow(previous)?, tables, caches, state)?;
    let Some(options) = choice else {
        return Ok(vec![logits]);
    };
    let (first, draws) = match rest {
        [first] => (first, Draws::Generator(None)),
        [first, streams] => (first, Draws::Streams(streams)),
        _ => return Err(Error::missing_input("the rows choosing their first token")),
    };
    Ok(vec![choose(
        &logits,
        first,
        options,
        &[options.sampling],
        draws,
    )?])
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
            unfollowed: previous_tokens(device, pages.sequences())?,
            pages,
            chunks: options.chunk_sizes(),
            compiled: None,
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
        self.compiled = Some((CompiledSteps::default(), options));
        let compiled = self.compile_chunk(0);
        if compiled.is_err() {
            self.compiled = None;
        }
        compiled
    }

    // Compiles the step that reads `chunk` extra tokens per sequence, under the options `compile` took.
    fn compile_chunk(&mut self, chunk: usize) -> Result<()> {
        let (layer, data_type) = (self.model.layer(), self.model.state().floating_type()?);
        let meta = layer.caches(data_type, Device::Meta, self.pages.cache_size()?)?;
        let (steps, options) = self
            .compiled
            .as_mut()
            .ok_or_else(|| Error::invalid_argument("a decoder that did not compile"))?;
        steps.compile(
            &self.model,
            &self.tables,
            &self.pages,
            self.pages.sequences() + chunk,
            &meta,
            &self.caches,
            vec![(
                PREVIOUS.to_owned(),
                previous_tokens(Device::Meta, self.pages.sequences())?,
            )],
            options.clone(),
            |step, tables, caches, extra, state| {
                step_forward(layer, None, step, tables, caches, extra, state)
            },
        )
    }

    // Compiles the step a step of `tokens` runs, the smallest of the planned sizes that holds it, if it is not yet compiled.
    fn compile_for(&mut self, tokens: &[&[i64]]) -> Result<()> {
        let (total, sequences) = (PageTable::step_tokens(tokens), self.pages.sequences());
        let needed = iter::once(0)
            .chain(self.chunks.iter().copied())
            .find(|&chunk| chunk + sequences >= total);
        match needed {
            Some(chunk)
                if self
                    .compiled
                    .as_ref()
                    .is_some_and(|(steps, _)| steps.get(chunk + sequences).is_none()) =>
            {
                self.compile_chunk(chunk)
            }
            _ => Ok(()),
        }
    }

    // The step of one token per sequence over this model's weights, with inputs of its own, which tuning and exporting take.
    // Its keys and layouts are those of the step this decoder compiles, so records tuned on it hold there.
    pub fn step_graph(&self) -> Result<DecoderStepGraph> {
        self.chunk_step_graph(0)
    }

    // The step that reads `chunk` extra tokens, as `step_graph` gives the step that reads none.
    pub fn chunk_step_graph(&self, chunk: usize) -> Result<DecoderStepGraph> {
        self.capture_step(chunk, None)
    }

    // The step that reads `chunk` extra tokens and chooses each sequence's next token as `options` asks.
    // Besides the step's own inputs it reads `first_tokens`, `[S]` of `I64`, one where a sequence's next token is its first, and when it samples, `streams`, `[S, 2]` of `I64`, each sequence's random stream as a running batch keeps it; it returns the chosen tokens, `[S]` of `I64`, so a program of it chooses as a running batch does.
    pub fn choosing_step_graph(
        &self,
        chunk: usize,
        options: &GenerationOptions,
    ) -> Result<DecoderStepGraph> {
        self.capture_step(chunk, Some(options))
    }

    fn capture_step(
        &self,
        chunk: usize,
        choice: Option<&GenerationOptions>,
    ) -> Result<DecoderStepGraph> {
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
        let mut extra = vec![(
            PREVIOUS.to_owned(),
            previous_tokens(self.device, pages.sequences())?,
        )];
        if let Some(options) = choice {
            extra.push((
                FIRST_TOKENS.to_owned(),
                previous_tokens(self.device, pages.sequences())?,
            ));
            if options.sampling.is_some() {
                let sequences = i64::try_from(pages.sequences())
                    .map_err(|_| Error::out_of_range("sequences"))?;
                extra.push((
                    STREAMS.to_owned(),
                    Tensor::zeroed_on(self.device, DataType::I64, Shape::create([sequences, 2])?)?,
                ));
            }
        }

        let (graph, outputs) = capture_step(
            self.model.state(),
            &self.tables,
            &step,
            &caches,
            extra.clone(),
            |step, tables, caches, extra, state| {
                step_forward(layer, choice, step, tables, caches, extra, state)
            },
        )?;
        let output = outputs
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
                .chain(extra)
                .collect(),
            output,
        })
    }

    pub fn encode_step_graph(
        &self,
        encoder: &ModelState,
        inputs: Vec<(String, Tensor)>,
        write: impl Fn(&[Tensor], &[L::Cache]) -> Result<Vec<Tensor>>,
    ) -> Result<(Graph, HashMap<String, Tensor>)> {
        // An encode step reads its own inputs, then this decoder's caches, which it writes the encoding to.
        let own = inputs.len();
        let inputs = inputs
            .into_iter()
            .chain(named_cache_inputs(&self.caches))
            .collect::<Vec<_>>();
        let bindings = encoder
            .capture_bindings()?
            .merged(self.model.state().capture_bindings()?);
        let (graph, _) = Graph::capture_with_bindings(inputs.clone(), bindings, |tensors| {
            let (own, captured) = tensors.split_at(own);
            write(own, &caches_from_inputs(&self.caches, captured)?)
        })?;
        Ok((graph, inputs.into_iter().collect()))
    }

    // Takes the step `other` compiled, for this decoder's own caches, so a new sequence does not prepare again.
    pub fn share_compiled(&mut self, other: &Self) -> Result<()> {
        let Some((steps, options)) = &other.compiled else {
            return Err(Error::invalid_argument(
                "decoder has no compiled step to share",
            ));
        };

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

        self.compiled = Some((steps.shared_with(&self.caches)?, options.clone()));
        Ok(())
    }

    // The executor of the compiled step of one token per sequence, whose report says how many specializations it needed.
    #[must_use]
    pub fn executor(&self) -> Option<&GraphExecutor> {
        self.compiled
            .as_ref()
            .and_then(|(steps, _)| steps.get(self.pages.sequences()))
    }

    // Every compiled step's executor.
    pub fn executors(&self) -> impl Iterator<Item = &GraphExecutor> {
        self.compiled
            .iter()
            .flat_map(|(steps, _)| steps.executors())
    }

    // Gives each sequence the cached keys of `parents[sequence]`, as beam search continues each row from its parent's; a sequence must be as long as its parent.
    pub fn reorder(&self, parents: &[usize]) -> Result<()> {
        let (from, to) = self.pages.parent_pages(parents)?;
        if from.is_empty() {
            return Ok(());
        }

        let count = i64::try_from(from.len()).map_err(|_| Error::out_of_range("pages"))?;
        let indices =
            |pages: Vec<i64>| Tensor::from_vec(self.device, Shape::create([count])?, pages);
        let (from, to) = (indices(from)?, indices(to)?);
        for cache in &self.caches {
            cache.copy_pages(&from, &to)?;
        }
        Ok(())
    }

    // Returns each block's cache, which a model may write outside a step, such as an encoder-decoder's cross-attention keys and values.
    #[must_use]
    pub fn caches(&self) -> &[L::Cache] {
        &self.caches
    }

    // Returns the most tokens each sequence holds.
    #[must_use]
    pub const fn capacity(&self) -> i64 {
        self.pages.capacity()
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

    fn step(&mut self, tokens: &[&[i64]], previous: Option<&Tensor>) -> Result<Tensor> {
        self.compile_for(tokens)?;
        if let Some((steps, _)) = &mut self.compiled {
            let previous = previous.unwrap_or(&self.unfollowed).clone();
            return steps
                .run(
                    &mut self.pages,
                    self.device,
                    tokens,
                    &self.caches,
                    |_| Ok(vec![(PREVIOUS.to_owned(), previous)]),
                    |_| Ok(()),
                )?
                .pop()
                .ok_or_else(|| Error::missing_output("compiled decode step"));
        }

        // A failed step leaves the table where the cache's contents are.
        let mut pages = self.pages.clone();
        let mut step = pages.step(self.device, tokens)?;
        if let Some(previous) = previous {
            step = step.follow(previous)?;
        }

        let logits =
            self.model
                .layer()
                .forward(&step, &self.tables, &self.caches, self.model.state())?;
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
        let (steps, _) = self.compiled.as_ref()?;
        let sequences = self.pages.sequences();
        let compiled = steps.largest().map(|tokens| tokens - sequences);
        compiled.max(self.chunks.last().copied())
    }

    fn report(&self) -> Result<Option<String>> {
        let Some((steps, _)) = &self.compiled else {
            return Ok(None);
        };

        let reports = steps
            .executors()
            .map(GraphExecutor::report)
            .collect::<Result<Vec<_>>>()?;
        Ok(Some(reports.join("\n")))
    }
}
