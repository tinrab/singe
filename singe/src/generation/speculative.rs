use std::iter;

use crate::{
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    generation::{DecoderOptions, GenerationOptions, compiled::CompiledStepCache},
    graph::{GraphPrepareOptions, SharedArena},
    nn::{CacheSize, DecodeCache, DecodeStep, DecoderModel, Model, ModelState, PageTable},
    shape::Shape,
    tensor::Tensor,
};

pub trait DraftingModel: DecoderModel {
    fn hidden_width(&self) -> i64;

    fn forward_rows(
        &self,
        step: &DecodeStep,
        tables: &Self::Tables,
        caches: &[Self::Cache],
        state: &ModelState,
    ) -> Result<(Tensor, Tensor)>;

    fn draft_cache(
        &self,
        data_type: DataType,
        device: Device,
        size: CacheSize,
    ) -> Result<Self::Cache>;

    fn draft(
        &self,
        step: &DecodeStep,
        hidden: &Tensor,
        tables: &Self::Tables,
        cache: &Self::Cache,
        state: &ModelState,
    ) -> Result<(Tensor, Tensor)>;

    fn draft_buffers(&self, _state: &ModelState) -> Result<Vec<(String, Tensor)>> {
        Ok(Vec::new())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Speculation {
    pub tokens: Vec<i64>,
    pub steps: usize,
    pub accepted: usize,
}

const DRAFT_HIDDEN: &str = "draft_hidden";

#[derive(Debug)]
pub struct SpeculativeDecoder<L: DraftingModel> {
    model: Model<L>,
    device: Device,
    drafts: usize,
    pages: PageTable,
    draft_pages: PageTable,
    tables: L::Tables,
    caches: Vec<L::Cache>,
    draft_caches: Vec<L::Cache>,
    sizes: Vec<usize>,
    size: CacheSize,
    decoder_steps: CompiledStepCache,
    draft_steps: CompiledStepCache,
}

fn pair(outputs: Vec<Tensor>) -> Result<(Tensor, Tensor)> {
    let mut outputs = outputs.into_iter();
    match (outputs.next(), outputs.next()) {
        (Some(logits), Some(hidden)) => Ok((logits, hidden)),
        _ => Err(Error::missing_output("speculative step")),
    }
}

fn leading((logits, hidden): (Tensor, Tensor), rows: usize) -> Result<(Tensor, Tensor)> {
    let rows = i64::try_from(rows).map_err(|_| Error::out_of_range("step rows"))?;
    Ok((logits.narrow(0, 0, rows)?, hidden.narrow(0, 0, rows)?))
}

fn most_likely(logits: &Tensor) -> Result<Vec<i64>> {
    logits.argmax(-1)?.to_vec::<i64>()
}

impl<L: DraftingModel> SpeculativeDecoder<L> {
    pub fn create(model: &Model<L>, options: DecoderOptions, drafts: usize) -> Result<Self> {
        if options.sequences != 1 || drafts == 0 {
            return Err(Error::invalid_argument(
                "speculative decoding runs one sequence, with one draft or more",
            ));
        }

        let device = model.state().device()?;
        let data_type = model.state().floating_type()?;
        let mut sizes = options
            .chunk_sizes()
            .into_iter()
            .map(|chunk| chunk + 1)
            .chain([1, drafts + 1])
            .collect::<Vec<_>>();
        sizes.sort_unstable();
        sizes.dedup();
        let largest = sizes.last().copied().unwrap_or(1);
        let pages =
            PageTable::create(1, options.capacity, options.page_size)?.with_padding(largest)?;
        let layer = model.layer();
        let size = CacheSize {
            trail: i64::try_from(drafts).map_err(|_| Error::out_of_range("drafts"))?,
            ..pages.cache_size()?
        };
        Ok(Self {
            device,
            drafts,
            tables: layer.tables(data_type, device, options.capacity)?,
            caches: layer.caches(data_type, device, size)?,
            draft_caches: vec![layer.draft_cache(data_type, device, size)?],
            size,
            draft_pages: pages.clone(),
            pages,
            sizes,
            model: model.with_state(
                model
                    .state()
                    .with_buffers(layer.draft_buffers(model.state())?)?,
            )?,
            decoder_steps: CompiledStepCache::default(),
            draft_steps: CompiledStepCache::default(),
        })
    }

    pub fn compile(&mut self) -> Result<()> {
        let options =
            GraphPrepareOptions::new(self.device).sharing_buffers(&SharedArena::default());
        let layer = self.model.layer();
        let data_type = self.model.state().floating_type()?;
        let size = self.size;
        let meta = layer.caches(data_type, Device::Meta, size)?;
        let draft_meta = vec![layer.draft_cache(data_type, Device::Meta, size)?];
        let (mut decoder, mut draft) = (CompiledStepCache::default(), CompiledStepCache::default());
        for &rows in &self.sizes {
            decoder.compile(
                &self.model,
                &self.tables,
                &self.pages,
                rows,
                &meta,
                &self.caches,
                Vec::new(),
                options.clone(),
                |step, tables, caches, _, state| {
                    let (logits, hidden) = layer.forward_rows(step, tables, caches, state)?;
                    Ok(vec![logits, hidden])
                },
            )?;
            let tokens = i64::try_from(rows).map_err(|_| Error::out_of_range("step rows"))?;
            let hidden = Tensor::zeroed_on(
                Device::Meta,
                data_type,
                Shape::create([tokens, layer.hidden_width()])?,
            )?;
            draft.compile(
                &self.model,
                &self.tables,
                &self.draft_pages,
                rows,
                &draft_meta,
                &self.draft_caches,
                vec![(DRAFT_HIDDEN.to_owned(), hidden)],
                options.clone(),
                |step, tables, caches, extra, state| {
                    let (logits, hidden) =
                        layer.draft(step, &extra[0], tables, &caches[0], state)?;
                    Ok(vec![logits, hidden])
                },
            )?;
        }

        self.decoder_steps = decoder;
        self.draft_steps = draft;
        Ok(())
    }

    fn run(
        &mut self,
        tokens: &[i64],
        written: Option<(i64, &Tensor)>,
        hidden: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let rows = tokens.len();
        let write = |step: &DecodeStep| -> Result<()> {
            if let Some((offset, values)) = written {
                let &[length, ..] = values.shape().dimensions() else {
                    return Err(Error::invalid_argument("written token values"));
                };
                step.tokens().narrow(0, offset, length)?.assign(values)?;
            }
            Ok(())
        };
        let (layer, state, device) = (self.model.layer(), self.model.state(), self.device);
        let (steps, pages, caches) = if hidden.is_some() {
            (
                &mut self.draft_steps,
                &mut self.draft_pages,
                &self.draft_caches,
            )
        } else {
            (&mut self.decoder_steps, &mut self.pages, &self.caches)
        };
        let outputs = if steps.is_empty() {
            let mut stepped = pages.clone();
            let step = stepped.step(device, &[tokens])?;
            write(&step)?;
            let outputs = match hidden {
                Some(hidden) => layer.draft(&step, hidden, &self.tables, &caches[0], state)?,
                None => layer.forward_rows(&step, &self.tables, caches, state)?,
            };
            *pages = stepped;
            outputs
        } else {
            // Padding rows read zeros, and their outputs are dropped; a compiled step reads its inputs whole, from their start.
            let padded = |size: usize| -> Result<Vec<(String, Tensor)>> {
                let Some(hidden) = hidden else {
                    return Ok(Vec::new());
                };
                let padding =
                    i64::try_from(size - rows).map_err(|_| Error::out_of_range("step rows"))?;
                let padded = if padding == 0 {
                    hidden.copy()?
                } else {
                    let &[_, width, ..] = hidden.shape().dimensions() else {
                        return Err(Error::invalid_argument("draft hidden states"));
                    };
                    let zeros = Tensor::zeroed_on(
                        device,
                        hidden.data_type(),
                        Shape::create([padding, width])?,
                    )?;
                    Tensor::concat(&[hidden.clone(), zeros], 0)?
                };
                Ok(vec![(DRAFT_HIDDEN.to_owned(), padded)])
            };
            pair(steps.run(pages, device, &[tokens], caches, padded, write)?)?
        };
        leading(outputs, rows)
    }

    fn run_all(&mut self, tokens: &[i64], hidden: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        let largest = self
            .decoder_steps
            .largest()
            .unwrap_or_else(|| tokens.len().max(1));
        let mut logits = Vec::new();
        let mut hiddens = Vec::new();
        for (index, chunk) in tokens.chunks(largest).enumerate() {
            let rows = hidden
                .map(|hidden| {
                    let start = i64::try_from(index * largest)
                        .map_err(|_| Error::out_of_range("step rows"))?;
                    let length =
                        i64::try_from(chunk.len()).map_err(|_| Error::out_of_range("step rows"))?;
                    hidden.narrow(0, start, length)
                })
                .transpose()?;
            let (chunk_logits, chunk_hidden) = self.run(chunk, None, rows.as_ref())?;
            logits.push(chunk_logits);
            hiddens.push(chunk_hidden);
        }
        Ok((Tensor::concat(&logits, 0)?, Tensor::concat(&hiddens, 0)?))
    }

    fn draft_after(
        &mut self,
        real: i64,
        followers: &[i64],
        hidden: &Tensor,
        count: usize,
    ) -> Result<Tensor> {
        // Drafts past the last real row wrote places this run writes again.
        self.draft_pages.truncate(0, real)?;
        let (logits, hidden) = self.run_all(followers, Some(hidden))?;
        let last =
            i64::try_from(followers.len()).map_err(|_| Error::out_of_range("step rows"))? - 1;
        // Each draft stays on the device and feeds the next draft step there.
        let mut drafted = vec![logits.narrow(0, last, 1)?.argmax(-1)?];
        let mut hidden = hidden.narrow(0, last, 1)?;
        while drafted.len() < count {
            let previous = drafted[drafted.len() - 1].clone();
            let (logits, next) = self.run(&[0], Some((0, &previous)), Some(&hidden))?;
            drafted.push(logits.argmax(-1)?);
            hidden = next;
        }

        Tensor::concat(&drafted, 0)
    }

    pub fn generate(&mut self, prompt: &[i64], options: &GenerationOptions) -> Result<Speculation> {
        if options.sampling.is_some()
            || !options.suppressed_tokens.is_empty()
            || !options.first_suppressed_tokens.is_empty()
        {
            return Err(Error::invalid_argument(
                "speculative decoding chooses the most likely token and suppresses none",
            ));
        }

        if prompt.is_empty() {
            return Err(Error::invalid_argument("generation needs a prompt"));
        }

        let mut speculation = Speculation {
            tokens: Vec::new(),
            steps: 0,
            accepted: 0,
        };
        if options.max_new_tokens == 0 {
            return Ok(speculation);
        }

        self.pages.reset();
        self.draft_pages.reset();
        let capacity = self.pages.capacity();
        let length =
            |tokens: usize| i64::try_from(tokens).map_err(|_| Error::out_of_range("tokens"));
        let (logits, mut hidden) = self.run_all(prompt, None)?;
        let last = length(prompt.len())? - 1;
        let mut next = most_likely(&logits.narrow(0, last, 1)?)?[0];
        let mut base = length(prompt.len())?;
        let mut followers = prompt[1..]
            .iter()
            .copied()
            .chain(iter::once(next))
            .collect::<Vec<_>>();
        let mut real = 0;
        loop {
            speculation.tokens.push(next);
            if options.stop_tokens.contains(&next)
                || speculation.tokens.len() >= options.max_new_tokens
            {
                return Ok(speculation);
            }

            // No more drafts than the tokens still wanted after the next one, and than the cache holds past it.
            let left = options.max_new_tokens - speculation.tokens.len();
            let room = usize::try_from(capacity - base - 1).unwrap_or(0);
            let count = self.drafts.min(left).min(room);
            let drafted = (count > 0)
                .then(|| self.draft_after(real, &followers, &hidden, count))
                .transpose()?;
            real = base;
            if base >= capacity {
                return Ok(speculation);
            }

            // The check reads the drafts where they are, and the host reads them with its choices, once a round.
            let mut checked = vec![0; count + 1];
            checked[0] = next;
            let (logits, checked_hidden) =
                self.run(&checked, drafted.as_ref().map(|drafted| (1, drafted)), None)?;
            let chosen = logits.argmax(-1)?;
            let ids = match &drafted {
                Some(drafted) => Tensor::concat(&[drafted.clone(), chosen], 0)?,
                None => chosen,
            }
            .to_vec::<i64>()?;
            let (drafts, choices) = ids.split_at(count);
            let accepted = drafts
                .iter()
                .zip(choices)
                .take_while(|(draft, choice)| draft == choice)
                .count();
            speculation.steps += 1;
            hidden = checked_hidden.narrow(0, 0, length(accepted + 1)?)?;
            if accepted < drafts.len() {
                // The caches keep the state the kept rows left, from the trail the check wrote.
                for cache in &self.caches {
                    cache.keep_trail(accepted)?;
                }

                self.pages.truncate(0, base + length(accepted + 1)?)?;
            }

            for &token in &choices[..accepted] {
                speculation.tokens.push(token);
                speculation.accepted += 1;
                if options.stop_tokens.contains(&token)
                    || speculation.tokens.len() >= options.max_new_tokens
                {
                    return Ok(speculation);
                }
            }

            next = choices[accepted];
            base += length(accepted + 1)?;
            followers = choices[..=accepted].to_vec();
        }
    }
}
