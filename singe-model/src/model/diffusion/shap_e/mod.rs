// Shap-E: a prior transformer that denoises a 3-D asset's latent, 1024 vectors, conditioned on a CLIP embedding of a text or an image, and a renderer that projects the latent into a NeRF-STF MLP's weights, queries its signed distance on a grid, and meshes it by marching cubes, coloring each vertex by the MLP's texture.
//
// The prior's state is `time` (`up`, `down`), `proj_in`, `embedding_norm` where it norms its condition, `embedding_proj`, `positions`, `norm_in`, `blocks.N` (`attention_norm`, `attention` with `query`, `key`, `value`, and `output`, `feed_forward_norm`, and `feed_forward` with `up` and `down`), `norm_out`, and `proj_to_clip`.
// The renderer's is `params_proj.N` (`proj`, `norm`) for each projected weight, `mlp.N`, and the marching cubes tables `cases` and `masks`.

mod diffusers;

use std::{collections::BTreeMap, f64::consts::FRAC_PI_2};

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Error as SingeError, Result as SingeResult, Shape, Tensor,
    nn::{Activation, LayerNorm, Linear, Mlp, Model, ModelState, StateKey},
};

use crate::{
    assembly::{
        FinalNorm, PYTORCH_NORM_EPS, PreNormEncoder, PreNormEncoderConfig, SinusoidOrder,
        timestep_input,
    },
    error::Result,
    format::Loader,
    schema::Schema,
    source::{LoadOptions, Source},
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PriorConfig {
    pub width: i64,
    pub heads: i64,
    pub layers: i64,
    // The latent's vectors and their width.
    pub embeddings: i64,
    pub embedding_width: i64,
    // The width of the condition, and whether it is layer-normed first.
    pub condition_width: i64,
    pub condition_norm: bool,
    pub clip_width: i64,
    pub time_width: i64,
}

// `PriorTransformer` as Shap-E uses it: the condition's tokens, the timestep's, and the latent's vectors, attending to each other with no mask, and the latent's predicted again.
#[derive(Debug, Clone)]
pub struct PriorLayers {
    config: PriorConfig,
    time: Mlp,
    proj_in: Linear,
    embedding_norm: Option<LayerNorm>,
    embedding_proj: Linear,
    positions: StateKey,
    norm_in: LayerNorm,
    encoder: PreNormEncoder,
    norm_out: LayerNorm,
    proj_to_clip: Linear,
}

impl PriorLayers {
    pub fn create(schema: &mut Schema<'_>, config: &PriorConfig) -> SingeResult<Self> {
        let width = config.width;
        let encoder = PreNormEncoderConfig {
            width,
            heads: config.heads,
            inner: 4 * width,
            activation: Activation::GeluExact,
            eps: PYTORCH_NORM_EPS,
            projection_biases: [true; 4],
            causal: false,
            layer_scale: false,
        };
        Ok(Self {
            config: config.clone(),
            time: schema.mlp(
                "time",
                [width, config.time_width, width],
                Activation::GeluExact,
                true,
            )?,
            proj_in: schema.linear("proj_in", config.embedding_width, width, true)?,
            embedding_norm: config
                .condition_norm
                .then(|| {
                    schema.layer_norm("embedding_norm", config.condition_width, PYTORCH_NORM_EPS)
                })
                .transpose()?,
            embedding_proj: schema.linear("embedding_proj", config.condition_width, width, true)?,
            positions: schema.parameter("positions", [1, config.embeddings, width])?,
            norm_in: schema.layer_norm("norm_in", width, PYTORCH_NORM_EPS)?,
            encoder: PreNormEncoder::create(schema, &encoder, config.layers, FinalNorm::Omitted)?,
            norm_out: schema.layer_norm("norm_out", width, PYTORCH_NORM_EPS)?,
            proj_to_clip: schema.linear("proj_to_clip", width, config.clip_width, true)?,
        })
    }

    // The predicted latent, `[B, N, D]`, of `latent`, `[B, N, D]`, at the timestep whose features `time` holds, `[1, W]`, conditioned on `condition`, `[B, C]` or `[B, K, C]`.
    fn predict(
        &self,
        latent: &Tensor,
        time: &Tensor,
        condition: &Tensor,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let &[batch, _, _] = latent.shape().dimensions() else {
            return Err(SingeError::invalid_argument("prior latent"));
        };
        let time = time
            .apply(&self.time, state)?
            .expand(Shape::create([batch, self.config.width])?)?
            .unsqueeze(1)?;
        let condition = match &self.embedding_norm {
            Some(norm) => condition.apply(norm, state)?,
            None => condition.clone(),
        };
        let mut condition = condition.apply(&self.embedding_proj, state)?;
        if condition.shape().dimensions().len() == 2 {
            condition = condition.unsqueeze(1)?;
        }

        let &[_, tokens, _] = condition.shape().dimensions() else {
            return Err(SingeError::invalid_argument("prior condition"));
        };
        let prefix = tokens + 1;
        let latent = latent.apply(&self.proj_in, state)?;
        // The condition's and the timestep's tokens take no learned positions.
        let positions = state
            .tensor(&self.positions)?
            .pad(&[(0, 0), (prefix, 0), (0, 0)], 0.0)?;
        let hidden = Tensor::concat(&[condition, time, latent], 1)?
            .add(&positions)?
            .apply(&self.norm_in, state)?;
        let hidden = self.encoder.encode(&hidden, None, state)?;
        let &[_, length, _] = hidden.shape().dimensions() else {
            return Err(SingeError::invalid_argument("prior hidden states"));
        };
        hidden
            .narrow(1, prefix, length - prefix)?
            .apply(&self.norm_out, state)?
            .apply(&self.proj_to_clip, state)?
            .narrow(2, 0, self.config.embedding_width)
    }
}

#[derive(Debug)]
pub struct Prior {
    model: Model<PriorLayers>,
}

impl Prior {
    #[must_use]
    pub fn config(&self) -> &PriorConfig {
        &self.model.layer().config
    }

    // The prediction for `latent` at `timestep`, in `F32`.
    pub fn predict(&self, latent: &Tensor, timestep: f32, condition: &Tensor) -> Result<Tensor> {
        let (layers, state) = (self.model.layer(), self.model.state());
        let data_type = state.floating_type()?;
        let time = timestep_input(
            timestep,
            layers.config.width,
            SinusoidOrder::CosinesFirst,
            0.0,
            state,
        )?;
        Ok(layers
            .predict(
                &latent.cast(data_type)?,
                &time,
                &condition.cast(data_type)?,
                state,
            )?
            .cast(DataType::F32)?)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RendererConfig {
    // Each projected weight's rows and columns, which consecutive runs of the latent's vectors project to.
    pub projected: Vec<[i64; 2]>,
    pub latent_width: i64,
    // The MLP's layers' input and output widths.
    pub layers: Vec<[i64; 2]>,
    pub insert_direction_at: usize,
}

// `ChannelsProj`: each of a run of the latent's vectors to a row of a weight, normed, plus a bias.
#[derive(Debug, Clone)]
struct ProjectedWeight {
    weight: StateKey,
    bias: StateKey,
    norm: LayerNorm,
    rows: i64,
    columns: i64,
}

#[derive(Debug, Clone)]
pub struct RendererLayers {
    config: RendererConfig,
    projected: Vec<ProjectedWeight>,
    // The MLP's weights and biases; the first projected weights replace its first weights.
    mlp: Vec<(StateKey, StateKey)>,
    cases: StateKey,
    masks: StateKey,
}

impl RendererLayers {
    pub fn create(schema: &mut Schema<'_>, config: &RendererConfig) -> SingeResult<Self> {
        let projected = config
            .projected
            .iter()
            .enumerate()
            .map(|(index, &[rows, columns])| {
                let mut projection = schema.scope(format!("params_proj.{index}"));
                Ok(ProjectedWeight {
                    weight: projection
                        .parameter("proj.weight", [rows * columns, config.latent_width])?,
                    bias: projection.parameter("proj.bias", [rows * columns])?,
                    norm: projection.layer_norm("norm", columns, PYTORCH_NORM_EPS)?,
                    rows,
                    columns,
                })
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let mlp = config
            .layers
            .iter()
            .enumerate()
            .map(|(index, &[input, output])| {
                Ok((
                    schema.parameter(&format!("mlp.{index}.weight"), [output, input])?,
                    schema.parameter(&format!("mlp.{index}.bias"), [output])?,
                ))
            })
            .collect::<SingeResult<Vec<_>>>()?;
        Ok(Self {
            config: config.clone(),
            projected,
            mlp,
            cases: schema.parameter_typed("cases", [256, 5, 3], DataType::I64)?,
            masks: schema.parameter_typed("masks", [256, 5], DataType::Bool)?,
        })
    }

    // The MLP's weights for `latent`, `[N, D]`: the projected ones first, then its own.
    fn weights(&self, latent: &Tensor, state: &ModelState) -> SingeResult<Vec<Tensor>> {
        let mut start = 0;
        let mut weights = Vec::new();
        for projected in &self.projected {
            let vectors = latent.narrow(0, start, projected.rows)?;
            start += projected.rows;
            let weight = state.tensor(&projected.weight)?.reshape([
                projected.rows,
                projected.columns,
                self.config.latent_width,
            ])?;
            // `einsum("bvd,vcd->bvc")`: each vector against its row's block of the weight.
            let row = vectors
                .unsqueeze(1)?
                .matmul(&weight.transpose(1, 2)?)?
                .squeeze(1)?
                .apply(&projected.norm, state)?
                .add(
                    &state
                        .tensor(&projected.bias)?
                        .reshape([projected.rows, projected.columns])?,
                )?;
            weights.push(row);
        }

        for (weight, _) in self.mlp.iter().skip(weights.len()) {
            weights.push(state.tensor(weight)?.clone());
        }
        Ok(weights)
    }

    // The MLP's outputs at `positions`, `[P, 3]`, with `weights`: `[P, 12]`, of which the first is the signed distance before its `tanh` and 3 to 5 the texture before its sigmoid.
    fn query(
        &self,
        positions: &Tensor,
        weights: &[Tensor],
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let &[count, _] = positions.shape().dimensions() else {
            return Err(SingeError::invalid_argument("renderer query positions"));
        };
        // NeRF's encoding: the position, then the sines of it scaled by 2^0 to 2^14, and of those shifted by π/2.
        let encode = |degrees: i64| -> SingeResult<Tensor> {
            let factors = Tensor::from_vec(
                positions.device(),
                Shape::create([1, degrees, 1])?,
                (0..degrees)
                    .map(|degree| {
                        let degree = i32::try_from(degree)
                            .map_err(|_| SingeError::out_of_range("encoding degree"))?;
                        Ok(2.0_f32.powi(degree))
                    })
                    .collect::<SingeResult<Vec<_>>>()?,
            )?;
            let scaled = positions
                .unsqueeze(1)?
                .mul(&factors)?
                .reshape([count, 3 * degrees])?;
            let shifted = scaled.add_scalar(FRAC_PI_2)?;
            Tensor::concat(
                &[
                    positions.clone(),
                    Tensor::concat(&[scaled, shifted], 1)?.sin()?,
                ],
                1,
            )
        };
        let mut hidden = encode(15)?;
        let last = weights.len() - 1;
        for (index, weight) in weights.iter().enumerate() {
            if index == self.config.insert_direction_at {
                // No direction: its encoding's 51 features are zeros.
                let zeros = Tensor::zeroed_on(
                    positions.device(),
                    hidden.data_type(),
                    Shape::create([count, 51])?,
                )?;
                hidden = Tensor::concat(&[hidden, zeros], 1)?;
            }

            hidden = hidden
                .matmul(&weight.transpose(0, 1)?)?
                .add(state.tensor(&self.mlp[index].1)?)?;
            if index < last {
                hidden = hidden.apply(&Activation::Silu, state)?;
            }
        }
        Ok(hidden)
    }
}

// A mesh with a color for each vertex, in [0, 1] and linear.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Mesh {
    pub vertices: Vec<[f32; 3]>,
    pub faces: Vec<[i64; 3]>,
    pub colors: Vec<[f32; 3]>,
}

// The grid's side and the points queried at once, as `decode_to_mesh` queries them.
const GRID: usize = 128;
const BATCH: usize = 4096 * 16;

#[derive(Debug)]
pub struct Renderer {
    model: Model<RendererLayers>,
}

impl Renderer {
    // `decode_to_mesh`: the latent's MLP's signed distance on a 128³ grid over [-1, 1]³, bordered by -1, meshed by marching cubes, and each vertex colored by the MLP's texture, converted from sRGB to linear.
    pub fn decode_mesh(&self, latent: &Tensor) -> Result<Mesh> {
        let (layers, state) = (self.model.layer(), self.model.state());
        let device = state.device()?;
        let data_type = state.floating_type()?;
        let weights = layers.weights(&latent.cast(data_type)?, state)?;
        // `volume_query_points`, computed on the CPU: x slowest and z fastest, each at `index / 127 · 2 − 1`.
        let coordinate = |index: usize| (index as f32 / (GRID - 1) as f32) * 2.0 + -1.0;
        let points = (0..GRID * GRID * GRID)
            .flat_map(|index| {
                [
                    coordinate(index / (GRID * GRID)),
                    coordinate(index / GRID % GRID),
                    coordinate(index % GRID),
                ]
            })
            .collect::<Vec<f32>>();
        let mut field = Vec::with_capacity(GRID * GRID * GRID);
        for chunk in points.chunks(BATCH * 3) {
            let count = i64::try_from(chunk.len() / 3)
                .map_err(|_| SingeError::out_of_range("query batch"))?;
            let positions = Tensor::from_vec(device, Shape::create([count, 3])?, chunk.to_vec())?
                .cast(data_type)?;
            let outputs = layers
                .query(&positions, &weights, state)?
                .narrow(1, 0, 1)?
                .tanh()?;
            field.extend(outputs.cast(DataType::F32)?.to_vec::<f32>()?);
        }

        let cases = state
            .tensor(&layers.cases)?
            .cast(DataType::I64)?
            .to_vec::<i64>()?;
        let masks = state
            .tensor(&layers.masks)?
            .cast(DataType::I64)?
            .to_vec::<i64>()?;
        let (vertices, faces) = marching_cubes(&field, &cases, &masks);
        if vertices.is_empty() {
            return Ok(Mesh {
                vertices,
                faces,
                colors: Vec::new(),
            });
        }

        let mut colors = Vec::with_capacity(vertices.len());
        for chunk in vertices.chunks(BATCH) {
            let count =
                i64::try_from(chunk.len()).map_err(|_| SingeError::out_of_range("query batch"))?;
            let flat = chunk.iter().flatten().copied().collect::<Vec<f32>>();
            let positions =
                Tensor::from_vec(device, Shape::create([count, 3])?, flat)?.cast(data_type)?;
            let texture = layers
                .query(&positions, &weights, state)?
                .narrow(1, 3, 3)?
                .sigmoid()?
                .cast(DataType::F32)?
                .to_vec::<f32>()?;
            colors.extend(texture.chunks(3).map(|color| {
                let linear = |value: f32| {
                    if value <= 0.040_45 {
                        value / 12.92
                    } else {
                        ((value + 0.055) / 1.055).powf(2.4)
                    }
                };
                [linear(color[0]), linear(color[1]), linear(color[2])]
            }));
        }
        Ok(Mesh {
            vertices,
            faces,
            colors,
        })
    }
}

// `MeshDecoder`'s grid: `GRID³` values x slowest, bordered by -1 to `SIDE³`, whose edges are numbered x-edges first, then y-edges, then z-edges, each in its grid's order.
const SIDE: usize = GRID + 2;
const CELLS: usize = SIDE - 1;
const X_EDGES: usize = CELLS * SIDE * SIDE;
const Y_EDGES: usize = SIDE * CELLS * SIDE;

const fn grid_value(field: &[f32], [x, y, z]: [usize; 3]) -> f32 {
    if x == 0 || y == 0 || z == 0 || x == SIDE - 1 || y == SIDE - 1 || z == SIDE - 1 {
        -1.0
    } else {
        field[((x - 1) * GRID + (y - 1)) * GRID + (z - 1)]
    }
}

const fn edge(axis: usize, x: usize, y: usize, z: usize) -> usize {
    match axis {
        0 => (x * SIDE + y) * SIDE + z,
        1 => X_EDGES + (x * CELLS + y) * SIDE + z,
        _ => X_EDGES + Y_EDGES + (x * SIDE + y) * CELLS + z,
    }
}

// The triangles of each cube of the grid, as edge numbers, from Shap-E's tables of each cube's triangles, as its own edges, and which of its five are used.
fn triangles(field: &[f32], cases: &[i64], masks: &[i64]) -> Vec<[usize; 3]> {
    let mut triangles = Vec::new();
    for x in 0..CELLS {
        for y in 0..CELLS {
            for z in 0..CELLS {
                let mut bits = 0usize;
                for (bit, (dx, dy, dz)) in [
                    (0, 0, 0),
                    (1, 0, 0),
                    (0, 1, 0),
                    (1, 1, 0),
                    (0, 0, 1),
                    (1, 0, 1),
                    (0, 1, 1),
                    (1, 1, 1),
                ]
                .iter()
                .enumerate()
                {
                    if grid_value(field, [x + dx, y + dy, z + dz]) > 0.0 {
                        bits |= 1 << bit;
                    }
                }

                // `_create_flat_edge_indices`: the cube's twelve edges.
                let edges = [
                    edge(0, x, y, z),
                    edge(0, x, y + 1, z),
                    edge(0, x, y, z + 1),
                    edge(0, x, y + 1, z + 1),
                    edge(1, x, y, z),
                    edge(1, x + 1, y, z),
                    edge(1, x, y, z + 1),
                    edge(1, x + 1, y, z + 1),
                    edge(2, x, y, z),
                    edge(2, x + 1, y, z),
                    edge(2, x, y + 1, z),
                    edge(2, x + 1, y + 1, z),
                ];
                for triangle in 0..5 {
                    if masks[bits * 5 + triangle] != 0 {
                        let local = &cases[(bits * 5 + triangle) * 3..][..3];
                        triangles.push([
                            edges[local[0] as usize],
                            edges[local[1] as usize],
                            edges[local[2] as usize],
                        ]);
                    }
                }
            }
        }
    }
    triangles
}

// Where the field crosses zero on edge `number`, between its two corners, over [-1, 1]³.
fn crossing(field: &[f32], number: usize) -> [f32; 3] {
    let (axis, rest) = if number < X_EDGES {
        (0, number)
    } else if number < X_EDGES + Y_EDGES {
        (1, number - X_EDGES)
    } else {
        (2, number - X_EDGES - Y_EDGES)
    };
    let (spans_y, spans_z) = match axis {
        0 => (SIDE, SIDE),
        1 => (CELLS, SIDE),
        _ => (SIDE, CELLS),
    };
    let first = [
        rest / (spans_y * spans_z),
        rest / spans_z % spans_y,
        rest % spans_z,
    ];
    let mut second = first;
    second[axis] += 1;
    let (s1, s2) = (grid_value(field, first), grid_value(field, second));
    let t = s1 / (s1 - s2);
    let place = |index: usize| (index as f32 / (SIDE - 1) as f32) * 2.0 + -1.0;
    [0, 1, 2].map(|index| t * place(second[index]) + (1.0 - t) * place(first[index]))
}

// `MeshDecoder`: marching cubes over `field`, `GRID³` x slowest, with Shap-E's tables; vertices are the used edges' crossings, in edge order.
fn marching_cubes(field: &[f32], cases: &[i64], masks: &[i64]) -> (Vec<[f32; 3]>, Vec<[i64; 3]>) {
    let triangles = triangles(field, cases, masks);
    let mut numbers = triangles
        .iter()
        .flatten()
        .map(|&edge| (edge, 0))
        .collect::<BTreeMap<usize, i64>>();
    for (number, renumbered) in numbers.values_mut().zip(0_i64..) {
        *number = renumbered;
    }

    let faces = triangles
        .iter()
        .map(|triangle| triangle.map(|edge| numbers[&edge]))
        .collect();
    let vertices = numbers
        .keys()
        .map(|&number| crossing(field, number))
        .collect();
    (vertices, faces)
}

pub fn load_prior(source: &Source, options: LoadOptions) -> Result<Prior> {
    let mut loader = Loader::new(source, options);
    let config: PriorConfig = loader.config(diffusers::prior_config)?;
    let model = Schema::model(options.data_type, |schema| {
        PriorLayers::create(schema, &config)
    })?;
    Ok(Prior {
        model: loader.load_whole(&model, &diffusers::prior_rules())?,
    })
}

pub fn load_renderer(source: &Source, options: LoadOptions) -> Result<Renderer> {
    let mut loader = Loader::new(source, options);
    let config: RendererConfig = loader.config(diffusers::renderer_config)?;
    let model = Schema::model(options.data_type, |schema| {
        RendererLayers::create(schema, &config)
    })?;
    Ok(Renderer {
        model: loader.load_whole(&model, &diffusers::renderer_rules(&config))?,
    })
}
