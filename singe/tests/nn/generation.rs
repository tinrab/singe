//! An exported generation loop generates what Singe's own running batch generates: the same rows, draws, and stop tokens, with nothing of Singe.
#![cfg(all(feature = "testing", feature = "cuda"))]

use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    process::{self, Command},
};

use singe::{
    Device, Shape, Tensor,
    backend::cuda::export::{CudaExport, Decoding},
    generation::{
        CachedDecoder, DecoderOptions, GenerationOptions, PromptChunking, generate_batch,
    },
    nn::{
        Activation, CausalLm, CausalLmParts, Embedding, GatedMlp, GroupedQueryAttention,
        GroupedQueryLayers, HeadProjection, KvCacheSpec, Linear, Model, PreNormBlock, RmsNorm,
        RotaryFrequencies, RotaryLayout, RotarySpec, StateBuilder, StateKey,
    },
    random::Generator,
    tensor::{AttentionOptions, SamplingOptions},
    testing::{build_set_generation_loop, maybe_lock_device, program_target},
};

const DEVICE: Device = Device::Cuda(0);
const VOCABULARY: i64 = 32;
const WIDTH: i64 = 32;
const HEADS: i64 = 2;
const HEAD_DIM: i64 = 16;
const INNER: i64 = 64;
const EPS: f32 = 1e-6;

type TinyDecoder = CausalLm<PreNormBlock<GroupedQueryAttention, GatedMlp>>;

// Values spread over [-scale, scale], different for each `seed`, so every weight differs from its neighbours.
fn weights(shape: &[i64], seed: f32, scale: f32) -> Tensor {
    let count = shape.iter().product::<i64>();
    let values = (0..count)
        .map(|index| seed.mul_add(1.7, index as f32 * 0.618).sin() * scale)
        .collect::<Vec<_>>();
    Tensor::from_vec(DEVICE, Shape::create(shape.to_vec()).unwrap(), values).unwrap()
}

// Registers a model's parameters, each with weights of its own.
struct WeightBuilder {
    builder: StateBuilder,
    seed: f32,
}

impl WeightBuilder {
    fn tensor(&mut self, name: &str, shape: &[i64]) -> StateKey {
        self.seed += 1.0;
        self.builder
            .parameter(name, weights(shape, self.seed, 0.3))
            .unwrap()
    }

    fn linear(&mut self, name: &str, input: i64, output: i64) -> Linear {
        Linear::new(self.tensor(name, &[output, input]), None)
    }

    fn norm(&mut self, name: &str) -> RmsNorm {
        RmsNorm::new(vec![WIDTH], Some(self.tensor(name, &[WIDTH])), EPS)
    }

    fn heads(&mut self, name: &str, heads: i64) -> HeadProjection {
        HeadProjection::new(self.linear(name, WIDTH, heads * HEAD_DIM), heads, HEAD_DIM)
    }
}

// A causal model of two blocks of grouped-query attention and a gated feed-forward layer, on CUDA.
fn tiny_model() -> Model<TinyDecoder> {
    let mut weights = WeightBuilder {
        builder: StateBuilder::new(),
        seed: 0.0,
    };
    let embedding = weights.tensor("embedding", &[VOCABULARY, WIDTH]);
    let blocks = (0..2)
        .map(|layer| {
            let name = |part: &str| format!("layers.{layer}.{part}");
            let attention = GroupedQueryAttention::create(
                GroupedQueryLayers {
                    query: weights.heads(&name("query"), HEADS),
                    key: weights.heads(&name("key"), 1),
                    value: weights.heads(&name("value"), 1),
                    output: weights.linear(&name("output"), HEADS * HEAD_DIM, WIDTH),
                },
                AttentionOptions {
                    causal: true,
                    ..AttentionOptions::default()
                },
            )
            .unwrap();
            PreNormBlock::new(
                weights.norm(&name("attention_norm")),
                attention,
                weights.norm(&name("feed_forward_norm")),
                GatedMlp::new(
                    weights.linear(&name("gate"), WIDTH, INNER),
                    weights.linear(&name("up"), WIDTH, INNER),
                    weights.linear(&name("down"), INNER, WIDTH),
                    Activation::Silu,
                ),
            )
        })
        .collect::<Vec<_>>();
    let norm = weights.norm("norm");
    let mut builder = weights.builder;
    let head = builder.tie("head", &embedding).unwrap();
    let decoder = CausalLm::new(CausalLmParts {
        embedding: Embedding::new(embedding),
        embedding_scale: None,
        logit_scale: None,
        logit_cap: None,
        rotary: Some(RotarySpec {
            head_dim: HEAD_DIM,
            frequencies: RotaryFrequencies::new(10_000.0),
            layout: RotaryLayout::HalfSplit,
        }),
        caches: vec![
            KvCacheSpec {
                kv_heads: 1,
                head_dim: HEAD_DIM,
            };
            blocks.len()
        ],
        blocks,
        norm,
        head: Linear::new(head, None),
    });
    Model::new(decoder, builder.finish().unwrap())
}

// Two rows, a prompt step of three tokens past one per row, and a cache of 32 tokens each.
fn options() -> DecoderOptions {
    DecoderOptions::new(2, 32)
        .with_page_size(4)
        .with_prompt_chunks(PromptChunking {
            smallest: 3,
            largest: 3,
        })
}

const PROMPTS: &[&[i64]] = &[&[1, 5, 9, 13, 2, 7, 4], &[3], &[8, 17, 11]];
const SEED: u64 = 7;

// Three prompts in two rows, the third joining as the first row frees, drawn from one seed, through Singe's compiled decoder.
fn generate_compiled(model: &Model<TinyDecoder>, generation: &GenerationOptions) -> Vec<Vec<i64>> {
    let mut decoder = CachedDecoder::create(model, options()).unwrap();
    decoder.compile().unwrap();
    generate_batch(
        &mut decoder,
        PROMPTS,
        generation,
        Some(&Generator::new(SEED)),
        |_, _, _| Ok(()),
        |_, _| Ok(()),
    )
    .unwrap()
}

#[test]
fn test_an_exported_loop_generates_what_a_running_batch_generates() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let Some(architecture) = program_target(DEVICE) else {
        return;
    };
    let model = tiny_model();
    let sampling = SamplingOptions::new()
        .with_temperature(0.8)
        .with_top_k(Some(8));
    let unstopped = generate_compiled(
        &model,
        &GenerationOptions {
            max_new_tokens: 8,
            sampling: Some(sampling),
            ..GenerationOptions::default()
        },
    );
    // A token the first prompt draws partway through stops it there, and stops any other prompt that draws it.
    let generation = GenerationOptions {
        max_new_tokens: 8,
        stop_tokens: vec![unstopped[0][3]],
        sampling: Some(sampling),
        ..GenerationOptions::default()
    };
    let expected = generate_compiled(&model, &generation);
    assert_eq!(expected[0].len(), 4, "{expected:?}");

    let decoder = CachedDecoder::create(&model, options()).unwrap();
    let step = decoder.choosing_step_graph(0, &generation).unwrap();
    let prompt = decoder.choosing_step_graph(3, &generation).unwrap();
    let program = CudaExport::new(architecture)
        .with_decoding(Decoding {
            layout: step.layout,
            stop_tokens: generation.stop_tokens,
            tokenizer: None,
        })
        .export_steps(&[("decode", &step.graph), ("prompt", &prompt.graph)])
        .unwrap();
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!(
        "../target/programs/generation-loop-{}",
        process::id()
    ));
    let _ = fs::remove_dir_all(&directory);
    let executable = build_set_generation_loop(&program, &directory).unwrap();
    let prompts = PROMPTS
        .iter()
        .map(|prompt| {
            prompt
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(directory.join("prompts.txt"), prompts).unwrap();
    let output = Command::new(&executable)
        .args(["--seed", &SEED.to_string(), "8", "--prompts", "prompts.txt"])
        .current_dir(&directory)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let mut generated = BTreeMap::<usize, Vec<i64>>::new();
    for line in String::from_utf8(output.stdout).unwrap().lines() {
        let (prompt, token) = line.split_once(' ').unwrap();
        generated
            .entry(prompt.parse().unwrap())
            .or_default()
            .push(token.parse().unwrap());
    }
    assert_eq!(generated.into_values().collect::<Vec<_>>(), expected);
    let _ = fs::remove_dir_all(&directory);
}
