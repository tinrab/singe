use std::collections::{BTreeMap, HashMap};

#[cfg(feature = "cuda")]
use crate::backend::cuda::nvtx::CudaNvtxRange;
use crate::{
    device::Device,
    error::{Error, Result},
    graph::{Graph, GraphExecutor, GraphPrepareOptions, SharedArena},
    nn::{
        DecodeCache, DecodeStep, DecoderModel, Model, ModelState, PageTable, TensorGroup,
        caches_from_inputs, named_cache_inputs,
    },
    tensor::Tensor,
};

#[derive(Debug, Default)]
pub struct CompiledStepCache {
    executors: BTreeMap<usize, GraphExecutor>,
}

impl CompiledStepCache {
    pub fn is_empty(&self) -> bool {
        self.executors.is_empty()
    }

    pub fn largest(&self) -> Option<usize> {
        self.executors.keys().last().copied()
    }

    pub fn get(&self, rows: usize) -> Option<&GraphExecutor> {
        self.executors.get(&rows)
    }

    pub fn executors(&self) -> impl Iterator<Item = &GraphExecutor> {
        self.executors.values()
    }

    pub fn compile<L, C>(
        &mut self,
        model: &Model<L>,
        tables: &L::Tables,
        pages: &PageTable,
        rows: usize,
        meta: &[C],
        caches: &[C],
        extra: Vec<(String, Tensor)>,
        options: GraphPrepareOptions,
        forward: impl Fn(&DecodeStep, &L::Tables, &[C], &[Tensor], &ModelState) -> Result<Vec<Tensor>>,
    ) -> Result<()>
    where
        L: DecoderModel,
        C: DecodeCache,
    {
        let (layer, real) = (model.layer(), model.state());
        let data_type = real.floating_type()?;
        let sequences =
            i64::try_from(pages.sequences()).map_err(|_| Error::out_of_range("sequences"))?;
        let tokens = i64::try_from(rows).map_err(|_| Error::out_of_range("step rows"))?;
        let (graph, _) = {
            #[cfg(feature = "cuda")]
            let _range = CudaNvtxRange::new(|| "singe decoder step capture");
            capture_step(
                &real.to(Device::Meta, None)?,
                &layer.tables(data_type, Device::Meta, pages.capacity())?,
                &DecodeStep::meta(tokens, sequences, pages.table_length()?)?,
                meta,
                extra,
                forward,
            )?
        };
        #[cfg(feature = "cuda")]
        let _range = CudaNvtxRange::new(|| "singe decoder step prepare");
        let bound = real.with_buffers(tables.named_tensors())?;
        let options = options.with_inputs_in_place(named_cache_inputs(caches));
        let executor = bound.program(&graph)?.prepare(bound.snapshot(), options)?;
        self.executors.insert(rows, executor);
        Ok(())
    }

    pub fn run<C: DecodeCache>(
        &mut self,
        pages: &mut PageTable,
        device: Device,
        tokens: &[&[i64]],
        caches: &[C],
        extra: impl FnOnce(usize) -> Result<Vec<(String, Tensor)>>,
        written: impl FnOnce(&DecodeStep) -> Result<()>,
    ) -> Result<Vec<Tensor>> {
        let total = PageTable::step_tokens(tokens);
        let largest = self.largest().unwrap_or(0);
        let (&size, executor) = self.executors.range_mut(total..).next().ok_or_else(|| {
            Error::invalid_argument(format!(
                "a decode step of {total} tokens is larger than the largest compiled step, of {largest}"
            ))
        })?;
        let mut stepped = pages.clone();
        let step = stepped.step_padded(device, tokens, size)?;
        written(&step)?;
        let inputs = step
            .named_tensors()
            .into_iter()
            .chain(named_cache_inputs(caches))
            .chain(extra(size)?)
            .collect::<HashMap<_, _>>();
        let outputs = executor.enqueue(&inputs)?;
        *pages = stepped;
        Ok(outputs)
    }

    pub fn shared_with<C: DecodeCache>(&self, caches: &[C]) -> Result<Self> {
        // The new decoder's steps run one at a time too, though not with these.
        let buffers = SharedArena::default();
        Ok(Self {
            executors: self
                .executors
                .iter()
                .map(|(&rows, executor)| {
                    Ok((
                        rows,
                        executor.create_instance_sharing(named_cache_inputs(caches), &buffers)?,
                    ))
                })
                .collect::<Result<_>>()?,
        })
    }
}

pub fn capture_step<T, C>(
    state: &ModelState,
    tables: &T,
    step: &DecodeStep,
    caches: &[C],
    extra: Vec<(String, Tensor)>,
    forward: impl Fn(&DecodeStep, &T, &[C], &[Tensor], &ModelState) -> Result<Vec<Tensor>>,
) -> Result<(Graph, Vec<Tensor>)>
where
    T: TensorGroup,
    C: DecodeCache,
{
    let state = state.with_buffers(tables.named_tensors())?;
    let extras = extra.len();
    let inputs = step
        .named_tensors()
        .into_iter()
        .chain(named_cache_inputs(caches))
        .chain(extra);
    Graph::capture_with_bindings(inputs, state.capture_bindings()?, |inputs| {
        let (packed, captured) = inputs
            .split_first()
            .ok_or_else(|| Error::missing_input("a decode step"))?;
        let (captured, extra) = captured.split_at(captured.len() - extras);
        forward(
            &step.repacked(packed)?,
            tables,
            &caches_from_inputs(caches, captured)?,
            extra,
            &state,
        )
    })
}
