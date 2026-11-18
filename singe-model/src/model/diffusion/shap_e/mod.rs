// Shap-E: a prior transformer that denoises a 3-D asset's latent, 1024 vectors, conditioned on a CLIP embedding of a text or an image, and a renderer that projects the latent into a NeRF-STF MLP's weights, queries its signed distance on a grid, and meshes it by marching cubes, coloring each vertex by the MLP's texture.
//
// The prior's state is `time` (`input`, `output`), `proj_in`, `embedding_norm` where it norms its condition, `embedding_proj`, `positions`, `norm_in`, `blocks.N` (`norm1`, `attention` with `query`, `key`, `value`, and `output`, `norm3`, and `feed_forward` with `up` and `down`), `norm_out`, and `proj_to_clip`.
// The renderer's is `params_proj.N` (`proj`, `norm`) for each projected weight, `mlp.N`, and the marching cubes tables `cases` and `masks`.

mod diffusers;

use std::f64::consts::FRAC_PI_2;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Result as SingeResult, Shape, Tensor,
    nn::{Activation, LayerNorm, Linear, Model, ModelState, StateBuilder, StateKey},
    tensor::AttentionOptions,
};

use crate::{
    assembly::timestep_input,
    error::{Error, Result},
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

#[derive(Debug, Clone)]
struct PriorBlock {
    norm1: LayerNorm,
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    norm3: LayerNorm,
    up: Linear,
    down: Linear,
    heads: i64,
}

impl PriorBlock {
    fn forward(&self, hidden: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let dimensions = hidden.shape().dimensions();
        let (batch, length, width) = (dimensions[0], dimensions[1], dimensions[2]);
        let normed = hidden.apply(&self.norm1, state)?;
        let heads = |projection: &Linear| -> SingeResult<Tensor> {
            normed
                .apply(projection, state)?
                .reshape([batch, length, self.heads, width / self.heads])?
                .permute(&[0, 2, 1, 3])
        };
        let attended = heads(&self.query)?
            .attend(
                &heads(&self.key)?,
                &heads(&self.value)?,
                None,
                None,
                AttentionOptions::default(),
            )?
            .permute(&[0, 2, 1, 3])?
            .contiguous()?
            .reshape([batch, length, width])?
            .apply(&self.output, state)?;
        let hidden = hidden.add(&attended)?;
        let fed = hidden
            .apply(&self.norm3, state)?
            .apply(&self.up, state)?
            .apply(&Activation::GeluExact, state)?
            .apply(&self.down, state)?;
        hidden.add(&fed)
    }
}

// `PriorTransformer` as Shap-E uses it: the condition's tokens, the timestep's, and the latent's vectors, attending to each other with no mask, and the latent's predicted again.
#[derive(Debug, Clone)]
pub struct PriorLayers {
    config: PriorConfig,
    time_input: Linear,
    time_output: Linear,
    proj_in: Linear,
    embedding_norm: Option<LayerNorm>,
    embedding_proj: Linear,
    positions: StateKey,
    norm_in: LayerNorm,
    blocks: Vec<PriorBlock>,
    norm_out: LayerNorm,
    proj_to_clip: Linear,
}

impl PriorLayers {
    pub fn create(schema: &mut Schema<'_>, config: &PriorConfig) -> SingeResult<Self> {
        let width = config.width;
        let blocks = (0..config.layers)
            .map(|index| {
                let mut block = schema.scope(format!("blocks.{index}"));
                let (query, key, value, output) = {
                    let mut attention = block.scope("attention");
                    (
                        attention.linear("query", width, width, true)?,
                        attention.linear("key", width, width, true)?,
                        attention.linear("value", width, width, true)?,
                        attention.linear("output", width, width, true)?,
                    )
                };
                let mut feed_forward = block.scope("feed_forward");
                let (up, down) = (
                    feed_forward.linear("up", width, 4 * width, true)?,
                    feed_forward.linear("down", 4 * width, width, true)?,
                );
                Ok(PriorBlock {
                    norm1: block.layer_norm("norm1", width, 1e-5)?,
                    query,
                    key,
                    value,
                    output,
                    norm3: block.layer_norm("norm3", width, 1e-5)?,
                    up,
                    down,
                    heads: config.heads,
                })
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let mut time = schema.scope("time");
        let (time_input, time_output) = (
            time.linear("input", width, config.time_width, true)?,
            time.linear("output", config.time_width, width, true)?,
        );
        Ok(Self {
            config: config.clone(),
            time_input,
            time_output,
            proj_in: schema.linear("proj_in", config.embedding_width, width, true)?,
            embedding_norm: config
                .condition_norm
                .then(|| schema.layer_norm("embedding_norm", config.condition_width, 1e-5))
                .transpose()?,
            embedding_proj: schema.linear("embedding_proj", config.condition_width, width, true)?,
            positions: schema.parameter("positions", [1, config.embeddings, width])?,
            norm_in: schema.layer_norm("norm_in", width, 1e-5)?,
            blocks,
            norm_out: schema.layer_norm("norm_out", width, 1e-5)?,
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
        let batch = latent.shape().dimensions()[0];
        let width = self.config.width;
        let time = time
            .apply(&self.time_input, state)?
            .apply(&Activation::GeluExact, state)?
            .apply(&self.time_output, state)?
            .expand(Shape::create([batch, width])?)?
            .unsqueeze(1)?;
        let condition = match &self.embedding_norm {
            Some(norm) => condition.apply(norm, state)?,
            None => condition.clone(),
        };
        let mut condition = condition.apply(&self.embedding_proj, state)?;
        if condition.shape().dimensions().len() == 2 {
            condition = condition.unsqueeze(1)?;
        }

        let prefix = condition.shape().dimensions()[1] + 1;
        let latent = latent.apply(&self.proj_in, state)?;
        // The condition's and the timestep's tokens take no learned positions.
        let positions = state
            .tensor(&self.positions)?
            .pad(&[(0, 0), (prefix, 0), (0, 0)], 0.0)?;
        let mut hidden = Tensor::concat(&[condition, time, latent], 1)?
            .add(&positions)?
            .apply(&self.norm_in, state)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden, state)?;
        }

        let length = hidden.shape().dimensions()[1];
        hidden
            .narrow(1, prefix, length - prefix)?
            .apply(&self.norm_out, state)?
            .apply(&self.proj_to_clip, state)?
            .narrow(2, 0, self.config.embedding_width)
    }
}

pub type Prior = Model<PriorLayers>;

impl PriorLayers {
    #[must_use]
    pub const fn config(&self) -> &PriorConfig {
        &self.config
    }
}

pub fn load_prior(source: &Source, options: LoadOptions) -> Result<Prior> {
    let mut loader = Loader::new(source, options);
    let config: PriorConfig = loader.config(diffusers::prior_config)?;
    let mut builder = StateBuilder::new();
    let layers = {
        let mut schema = Schema::root(&mut builder, options.data_type);
        PriorLayers::create(&mut schema, &config)?
    };
    let model = loader.load(
        "prior",
        &Model::new(layers, builder.finish()?),
        &diffusers::prior_rules(),
    )?;
    loader.finish()?;
    Ok(model)
}

// The prior's prediction for `latent` at `timestep`, in `F32`.
pub fn predict(
    prior: &Prior,
    latent: &Tensor,
    timestep: f32,
    condition: &Tensor,
) -> Result<Tensor> {
    let data_type = prior.state().floating_type()?;
    let time = timestep_input(
        timestep,
        prior.layer().config().width,
        true,
        0.0,
        prior.state(),
    )?;
    Ok(prior
        .layer()
        .predict(
            &latent.cast(data_type)?,
            &time,
            &condition.cast(data_type)?,
            prior.state(),
        )?
        .cast(DataType::F32)?)
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
                    norm: projection.layer_norm("norm", columns, 1e-5)?,
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
        let count = positions.shape().dimensions()[0];
        // NeRF's encoding: the position, then the sines of it scaled by 2^0 to 2^14, and of those shifted by π/2.
        let encode = |degrees: i64| -> SingeResult<Tensor> {
            let factors = Tensor::from_vec(
                positions.device(),
                Shape::create([1, degrees, 1])?,
                (0..degrees)
                    .map(|degree| 2.0_f32.powi(i32::try_from(degree).unwrap_or(0)))
                    .collect::<Vec<_>>(),
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

pub type Renderer = Model<RendererLayers>;

pub fn load_renderer(source: &Source, options: LoadOptions) -> Result<Renderer> {
    let mut loader = Loader::new(source, options);
    let config: RendererConfig = loader.config(diffusers::renderer_config)?;
    let mut builder = StateBuilder::new();
    let layers = {
        let mut schema = Schema::root(&mut builder, options.data_type);
        RendererLayers::create(&mut schema, &config)?
    };
    let model = loader.load(
        "renderer",
        &Model::new(layers, builder.finish()?),
        &diffusers::renderer_rules(&config),
    )?;
    loader.finish()?;
    Ok(model)
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

// `decode_to_mesh`: the latent's MLP's signed distance on a 128³ grid over [-1, 1]³, bordered by -1, meshed by marching cubes, and each vertex colored by the MLP's texture, converted from sRGB to linear.
#[expect(clippy::cast_precision_loss, reason = "grid indices are small")]
#[expect(
    clippy::suboptimal_flops,
    reason = "rounds each product as the reference does, which the verifications compare with"
)]
pub fn decode_mesh(renderer: &Renderer, latent: &Tensor) -> Result<Mesh> {
    let layers = renderer.layer();
    let state = renderer.state();
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
        let count = i64::try_from(chunk.len() / 3).map_err(|_| Error::unsupported("a batch"))?;
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
        let count = i64::try_from(chunk.len()).map_err(|_| Error::unsupported("a batch"))?;
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

// `MeshDecoder`: marching cubes over `field`, `GRID³` x slowest, bordered by -1 to `(GRID + 2)³`, with Shap-E's tables of each cube's triangles, as edge numbers, and which of its five are used; vertices are the used edges' crossings, in edge order, over [-1, 1]³.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "grid indices are small"
)]
#[expect(
    clippy::suboptimal_flops,
    reason = "rounds each product as the reference does, which the verifications compare with"
)]
fn marching_cubes(field: &[f32], cases: &[i64], masks: &[i64]) -> (Vec<[f32; 3]>, Vec<[i64; 3]>) {
    let side = GRID + 2;
    let value = |x: usize, y: usize, z: usize| {
        if x == 0 || y == 0 || z == 0 || x == side - 1 || y == side - 1 || z == side - 1 {
            -1.0
        } else {
            field[((x - 1) * GRID + (y - 1)) * GRID + (z - 1)]
        }
    };
    let cells = side - 1;
    // Each axis's edges, numbered x-edges first, then y-edges, then z-edges, each in its grid's order.
    let x_edges = cells * side * side;
    let y_edges = side * cells * side;
    let edge = |axis: usize, x: usize, y: usize, z: usize| match axis {
        0 => (x * side + y) * side + z,
        1 => x_edges + (x * cells + y) * side + z,
        _ => x_edges + y_edges + (x * side + y) * cells + z,
    };
    let mut triangles = Vec::new();
    for x in 0..cells {
        for y in 0..cells {
            for z in 0..cells {
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
                    if value(x + dx, y + dy, z + dz) > 0.0 {
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

    let mut used = triangles.iter().flatten().copied().collect::<Vec<usize>>();
    used.sort_unstable();
    used.dedup();
    let renumbered = |edge: usize| used.binary_search(&edge).unwrap_or(0) as i64;
    let faces = triangles
        .iter()
        .map(|triangle| {
            [
                renumbered(triangle[0]),
                renumbered(triangle[1]),
                renumbered(triangle[2]),
            ]
        })
        .collect();
    // Each used edge's two corners, the field's values there, and the crossing between them.
    let place = |index: usize| (index as f32 / (side - 1) as f32) * 2.0 + -1.0;
    let vertices = used
        .iter()
        .map(|&number| {
            let (axis, rest) = if number < x_edges {
                (0, number)
            } else if number < x_edges + y_edges {
                (1, number - x_edges)
            } else {
                (2, number - x_edges - y_edges)
            };
            let (spans_y, spans_z) = match axis {
                0 => (side, side),
                1 => (cells, side),
                _ => (side, cells),
            };
            let (x, y, z) = (
                rest / (spans_y * spans_z),
                rest / spans_z % spans_y,
                rest % spans_z,
            );
            let (x2, y2, z2) = match axis {
                0 => (x + 1, y, z),
                1 => (x, y + 1, z),
                _ => (x, y, z + 1),
            };
            let (s1, s2) = (value(x, y, z), value(x2, y2, z2));
            let t = s1 / (s1 - s2);
            let (p1, p2) = (
                [place(x), place(y), place(z)],
                [place(x2), place(y2), place(z2)],
            );
            [
                t * p2[0] + (1.0 - t) * p1[0],
                t * p2[1] + (1.0 - t) * p1[1],
                t * p2[2] + (1.0 - t) * p1[2],
            ]
        })
        .collect();
    (vertices, faces)
}
