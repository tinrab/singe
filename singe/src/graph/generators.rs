use std::{borrow::Cow, collections::HashMap};

use crate::{device::Device, error::Result, random::Generator, tensor::Tensor};

/// A generator a graph reads through the input `name`, advancing it by `blocks` Philox blocks per run.
#[derive(Debug, Clone)]
pub struct GraphGenerator {
    pub(crate) name: String,
    pub(crate) generator: Generator,
    pub(crate) device: Device,
    pub(crate) blocks: u64,
}

/// Adds the current state of each generator whose input the caller did not supply, advancing it past this run.
pub fn with_generator_states<'a>(
    generators: &[GraphGenerator],
    inputs: &'a HashMap<String, Tensor>,
) -> Result<Cow<'a, HashMap<String, Tensor>>> {
    if generators
        .iter()
        .all(|generator| inputs.contains_key(&generator.name))
    {
        return Ok(Cow::Borrowed(inputs));
    }

    let mut inputs = inputs.clone();
    for generator in generators {
        if !inputs.contains_key(&generator.name) {
            let state = generator.generator.reserve(generator.blocks)?;
            inputs.insert(generator.name.clone(), state.tensor(generator.device)?);
        }
    }
    Ok(Cow::Owned(inputs))
}
