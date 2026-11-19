// The Decision Transformer: GPT-2's causal blocks over a trajectory's steps, each step three tokens, its return to go, its state, and its action, embedded linearly with its timestep's embedding added; a step's state token predicts its action, and its action token its next state and return.
//
// Its state is `embeddings` (`return`, `state`, `action`, `timestep`, `norm`, and `positions`, of which every token reads the first), `blocks`, `norm`, and `predict` (`state`, `action`, `return`).

mod transformers;

use std::iter;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Error as SingeError, Result as SingeResult, Shape, Tensor,
    nn::{Embedding, Layer, LayerNorm, Linear, Model, ModelState, StateKey},
};

use crate::{
    assembly::{FinalNorm, PreNormEncoder, PreNormEncoderConfig},
    error::Result,
    format::Loader,
    head::{HeadLoader, Trajectory, TrajectoryModel, TrajectoryPredictions},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionTransformerConfig {
    pub encoder: PreNormEncoderConfig,
    pub layers: i64,
    pub state_dim: i64,
    pub action_dim: i64,
    // The timesteps an episode's steps embed.
    pub episode: i64,
    pub max_positions: i64,
    // Whether actions are squashed into `(-1, 1)`.
    pub action_tanh: bool,
}

#[derive(Debug, Clone)]
pub struct DecisionTransformerLayers {
    config: DecisionTransformerConfig,
    returns: Linear,
    states: Linear,
    actions: Linear,
    timesteps: Embedding,
    embedding_norm: LayerNorm,
    positions: StateKey,
    encoder: PreNormEncoder,
    predict_state: Linear,
    predict_action: Linear,
    predict_return: Linear,
}

impl DecisionTransformerLayers {
    pub fn create(
        schema: &mut Schema<'_>,
        config: &DecisionTransformerConfig,
    ) -> SingeResult<Self> {
        let width = config.encoder.width;
        let (returns, states, actions, timesteps, embedding_norm, positions) = {
            let mut embeddings = schema.scope("embeddings");
            (
                embeddings.linear("return", 1, width, true)?,
                embeddings.linear("state", config.state_dim, width, true)?,
                embeddings.linear("action", config.action_dim, width, true)?,
                Embedding::new(embeddings.parameter("timestep.weight", [config.episode, width])?),
                embeddings.layer_norm("norm", width, config.encoder.eps)?,
                embeddings.parameter("positions.weight", [config.max_positions, width])?,
            )
        };
        let encoder =
            PreNormEncoder::create(schema, &config.encoder, config.layers, FinalNorm::Included)?;
        let mut predict = schema.scope("predict");
        Ok(Self {
            config: config.clone(),
            returns,
            states,
            actions,
            timesteps,
            embedding_norm,
            positions,
            encoder,
            predict_state: predict.linear("state", width, config.state_dim, true)?,
            predict_action: predict.linear("action", width, config.action_dim, true)?,
            predict_return: predict.linear("return", width, 1, true)?,
        })
    }

    // Each step's predicted states, `[T, S]`, actions, `[T, A]`, and returns, `[T, 1]`, from its returns to go, `[T, 1]`, states, `[T, S]`, actions, `[T, A]`, and timesteps, `[T]` of `I64`.
    fn predict(
        &self,
        [returns, states, actions, timesteps]: [&Tensor; 4],
        state: &ModelState,
    ) -> SingeResult<[Tensor; 3]> {
        let width = self.config.encoder.width;
        let &[steps, _] = states.shape().dimensions() else {
            return Err(SingeError::invalid_argument("a trajectory's states"));
        };
        let time = timesteps.apply(&self.timesteps, state)?;
        let tokens = [
            returns.apply(&self.returns, state)?,
            states.apply(&self.states, state)?,
            actions.apply(&self.actions, state)?,
        ]
        .iter()
        .map(|embedded| embedded.add(&time))
        .collect::<SingeResult<Vec<_>>>()?;
        // Every token takes the first learned position, as the model passes position zero for each.
        let position = state.tensor(&self.positions)?.narrow(0, 0, 1)?;
        let hidden = Tensor::stack(&tokens, 1)?
            .reshape([1, 3 * steps, width])?
            .apply(&self.embedding_norm, state)?
            .add(&position)?;
        let hidden = self
            .encoder
            .encode(&hidden, None, state)?
            .reshape([steps, 3, width])?;
        let (after_state, after_action) = (hidden.select(1, 1)?, hidden.select(1, 2)?);
        let mut action = after_state.apply(&self.predict_action, state)?;
        if self.config.action_tanh {
            action = action.tanh()?;
        }
        Ok([
            after_action.apply(&self.predict_state, state)?,
            action,
            after_action.apply(&self.predict_return, state)?,
        ])
    }
}

impl Layer for DecisionTransformerLayers {
    // Reads a trajectory packed as `[T, 1 + S + A + 1]` in `F32`, which holds its timesteps exactly: each step's return to go, state, action, and timestep.
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let (states, actions) = (self.config.state_dim, self.config.action_dim);
        let data_type = state.floating_type()?;
        let returns = input.narrow(1, 0, 1)?.cast(data_type)?;
        let step_states = input.narrow(1, 1, states)?.cast(data_type)?;
        let step_actions = input.narrow(1, 1 + states, actions)?.cast(data_type)?;
        let timesteps = input
            .narrow(1, 1 + states + actions, 1)?
            .squeeze(1)?
            .cast(DataType::I64)?;
        let [predicted_states, predicted_actions, predicted_returns] =
            self.predict([&returns, &step_states, &step_actions, &timesteps], state)?;
        Tensor::concat(&[predicted_states, predicted_actions, predicted_returns], 1)
    }
}

pub type DecisionTransformer = Model<DecisionTransformerLayers>;

impl TrajectoryModel for DecisionTransformer {
    fn predict(&self, trajectory: &Trajectory) -> Result<TrajectoryPredictions> {
        let config = &self.layer().config;
        let (states, actions) = (
            usize::try_from(config.state_dim)
                .map_err(|_| SingeError::out_of_range("state width"))?,
            usize::try_from(config.action_dim)
                .map_err(|_| SingeError::out_of_range("action width"))?,
        );
        let steps = trajectory.states.len();
        for (name, length) in [
            ("actions", trajectory.actions.len()),
            ("returns_to_go", trajectory.returns_to_go.len()),
            ("timesteps", trajectory.timesteps.len()),
        ] {
            if length != steps {
                return Err(SingeError::length_mismatch(name, steps, length).into());
            }
        }

        for (name, rows, width) in [
            ("states", &trajectory.states, states),
            ("actions", &trajectory.actions, actions),
        ] {
            if let Some((step, row)) = rows.iter().enumerate().find(|(_, row)| row.len() != width) {
                return Err(SingeError::length_mismatch(
                    format!("{name}[{step}]"),
                    width,
                    row.len(),
                )
                .into());
            }
        }

        let packed = (0..steps)
            .flat_map(|step| {
                iter::once(trajectory.returns_to_go[step])
                    .chain(trajectory.states[step].iter().copied())
                    .chain(trajectory.actions[step].iter().copied())
                    .chain(iter::once(trajectory.timesteps[step] as f32))
            })
            .collect::<Vec<f32>>();
        let width = 2 + states + actions;
        let shape = Shape::create([
            i64::try_from(steps).map_err(|_| SingeError::out_of_range("states"))?,
            i64::try_from(width).map_err(|_| SingeError::out_of_range("step width"))?,
        ])?;
        let input = Tensor::from_vec(self.state().device()?, shape, packed)?;
        let output = self.forward(&input)?.cast(DataType::F32)?.to_vec::<f32>()?;
        let row = states + actions + 1;
        let rows = output.chunks(row);
        Ok(TrajectoryPredictions {
            state_preds: rows.clone().map(|row| row[..states].to_vec()).collect(),
            action_preds: rows
                .clone()
                .map(|row| row[states..states + actions].to_vec())
                .collect(),
            return_preds: rows.map(|row| row[states + actions]).collect(),
        })
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "decision_transformer",
    formats: &[(Format::Transformers, &["decision_transformer"])],
    loaders: &[HeadLoader::Trajectory(|loader| Ok(Box::new(load(loader)?)))],
};

pub fn load(loader: &mut Loader<'_>) -> Result<DecisionTransformer> {
    let config: DecisionTransformerConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        DecisionTransformerLayers::create(schema, &config)
    })?;
    loader.load_whole(&model, &transformers::rules())
}
