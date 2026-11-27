//! A model of trajectories, as `transformers`' Decision Transformer is: each step's state, action, and return to go, and its timestep, to each step's predicted action, next state, and next return.

use std::fmt::Debug;

use serde::{Deserialize, Serialize};

use crate::error::Result;

/// A trajectory's steps, in order: each one's state, action, return to go, and timestep in its episode.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Trajectory {
    pub states: Vec<Vec<f32>>,
    pub actions: Vec<Vec<f32>>,
    pub returns_to_go: Vec<f32>,
    pub timesteps: Vec<i64>,
}

/// Each step's predictions, named as `transformers`' `DecisionTransformerOutput` names them.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TrajectoryOutput {
    /// Each step's predicted next state, after its state and action.
    pub state_preds: Vec<Vec<f32>>,
    /// Each step's predicted action, after its state.
    pub action_preds: Vec<Vec<f32>>,
    /// Each step's predicted next return, after its state and action.
    pub return_preds: Vec<f32>,
}

pub trait TrajectoryModel: Debug {
    fn predict(&self, trajectory: &Trajectory) -> Result<TrajectoryOutput>;
}
