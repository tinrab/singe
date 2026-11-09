//! Attention with each batch's count of keys against attention with the padding mask those counts make.

use std::collections::HashMap;

#[cfg(all(feature = "testing", feature = "cuda"))]
use singe::testing::maybe_lock_device;
use singe::{DataType, Device, Shape, Tensor, graph::Graph, tensor::AttentionOptions};
#[cfg(all(feature = "testing", feature = "cuda"))]
use singe::{
    backend::cuda::{
        LoweringPolicy,
        export::{CudaExport, Library, ProgramStep},
    },
    graph::{GraphCudaOptions, GraphPrepareOptions},
    testing::{CompiledProgram, program_target},
};
use singe_testing::{Tolerance, assert_matches, host};

const EXACT_ENOUGH: Tolerance = Tolerance::absolute(1e-6, "both run the same CPU kernels");

const BATCH: i64 = 3;
const HEADS: i64 = 2;
const KEYS: i64 = 40;
const DIM: i64 = 64;

// Each batch's count of keys: all of them, one, and some.
const LENGTHS: [i32; 3] = [40, 1, 23];

#[expect(clippy::cast_precision_loss, reason = "test values have few elements")]
fn values(shape: &[i64], seed: f32, device: Device, data_type: DataType) -> Tensor {
    let count = shape.iter().product::<i64>();
    let values = (0..count)
        .map(|index| (index as f32).mul_add(0.37, seed).sin() * 0.5)
        .collect::<Vec<_>>();
    Tensor::from_vec(device, Shape::create(shape.to_vec()).unwrap(), values)
        .unwrap()
        .cast(data_type)
        .unwrap()
}

struct Case {
    query: Tensor,
    key: Tensor,
    value: Tensor,
    lengths: Tensor,
    options: AttentionOptions,
}

impl Case {
    fn create(queries: i64, causal: bool, device: Device, data_type: DataType) -> Self {
        Self {
            query: values(&[BATCH, HEADS, queries, DIM], 0.1, device, data_type),
            key: values(&[BATCH, HEADS, KEYS, DIM], 0.2, device, data_type),
            value: values(&[BATCH, HEADS, KEYS, DIM], 0.3, device, data_type),
            lengths: Tensor::from_vec(device, Shape::create([BATCH]).unwrap(), LENGTHS.to_vec())
                .unwrap(),
            options: AttentionOptions {
                causal,
                ..AttentionOptions::default()
            },
        }
    }

    // The same attention with the Boolean mask the counts make, which reference values come from.
    fn masked(&self) -> Vec<f32> {
        let device = self.query.device();
        let mask = (0..BATCH)
            .flat_map(|batch| {
                let length = i64::from(LENGTHS[usize::try_from(batch).unwrap()]);
                (0..KEYS).map(move |key| key < length)
            })
            .collect::<Vec<_>>();
        let mask =
            Tensor::from_vec(device, Shape::create([BATCH, 1, 1, KEYS]).unwrap(), mask).unwrap();
        let wide = |tensor: &Tensor| {
            tensor
                .cast(DataType::F32)
                .unwrap()
                .to_device(Device::Cpu)
                .unwrap()
        };
        host(
            &wide(&self.query)
                .attend(
                    &wide(&self.key),
                    &wide(&self.value),
                    Some(&mask.to_device(Device::Cpu).unwrap()),
                    None,
                    self.options,
                )
                .unwrap(),
        )
    }

    fn padded(&self) -> Tensor {
        self.query
            .attend_padded(&self.key, &self.value, &self.lengths, self.options)
            .unwrap()
    }

    fn graph(&self) -> (Graph, HashMap<String, Tensor>) {
        let inputs = [
            ("query", &self.query),
            ("key", &self.key),
            ("value", &self.value),
            ("lengths", &self.lengths),
        ]
        .map(|(name, tensor)| (name.to_owned(), tensor.clone()));
        let options = self.options;
        let (graph, _) = Graph::capture(inputs.clone(), |tensors| {
            Ok(vec![tensors[0].attend_padded(
                &tensors[1],
                &tensors[2],
                &tensors[3],
                options,
            )?])
        })
        .unwrap();
        (graph, inputs.into_iter().collect())
    }
}

fn shapes() -> [(i64, bool); 3] {
    // One query, as decoding has, a block of queries, and a causal block.
    [(1, false), (17, false), (17, true)]
}

#[test]
fn test_key_lengths_hide_what_the_padding_mask_hides_on_cpu() {
    for (queries, causal) in shapes() {
        let case = Case::create(queries, causal, Device::Cpu, DataType::F32);
        assert_matches(
            &host(&case.padded()),
            &case.masked(),
            EXACT_ENOUGH,
            &format!("{queries} queries, causal {causal}"),
        );
        let (graph, inputs) = case.graph();
        assert_matches(
            &host(&graph.run(&inputs).unwrap()[0]),
            &case.masked(),
            EXACT_ENOUGH,
            &format!("{queries} queries, causal {causal}, as a graph"),
        );
    }
}

#[cfg(all(feature = "testing", feature = "cuda"))]
#[test]
fn test_key_lengths_hide_what_the_padding_mask_hides_on_cuda() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    for (data_type, tolerance) in [
        (
            DataType::F32,
            Tolerance::absolute(1e-5, "the GPU's F32 kernels sum in another order"),
        ),
        (
            DataType::Bf16,
            Tolerance::absolute(0.02, "BF16 keeps eight mantissa bits"),
        ),
    ] {
        for (queries, causal) in shapes() {
            let case = Case::create(queries, causal, device, data_type);
            let expected = case.masked();
            let what = format!("{data_type:?}, {queries} queries, causal {causal}");
            assert_matches(
                &host(&case.padded()),
                &expected,
                tolerance,
                &format!("{what}, eagerly"),
            );
            let (graph, inputs) = case.graph();
            for (how, options) in [
                ("compiled", GraphCudaOptions::new()),
                (
                    "through generated kernels",
                    GraphCudaOptions::new()
                        .with_lowering_policy(LoweringPolicy::generated_kernels()),
                ),
            ] {
                let mut executor = graph
                    .prepare(GraphPrepareOptions::new(device).with_cuda_options(options))
                    .unwrap();
                assert_matches(
                    &host(&executor.run(&inputs).unwrap()[0]),
                    &expected,
                    tolerance,
                    &format!("{what}, {how}"),
                );
                // Half precision is what cuDNN's attention takes, and the counts reach it as its sequence lengths.
                if data_type == DataType::Bf16 && how == "compiled" {
                    let report = executor.report().unwrap();
                    assert!(
                        report
                            .contains("scaled_dot_product_attention: provider Some(CudnnBackend)"),
                        "{what}: {report}"
                    );
                }
            }

            // A program passes the counts to cuDNN's attention as a compiled graph does.
            if let Some(architecture) = program_target(device) {
                let program = CudaExport::new(architecture).export(&graph).unwrap();
                let calls_cudnn = program.steps.iter().any(|step| {
                    matches!(step, ProgramStep::Library { call, .. } if call.library() == Library::Cudnn)
                });
                assert_eq!(calls_cudnn, data_type == DataType::Bf16, "{what}");
                let compiled = CompiledProgram::create(
                    &program,
                    &format!("padded_{data_type:?}_{queries}_{causal}"),
                )
                .unwrap();
                let ordered = ["query", "key", "value", "lengths"].map(|name| inputs[name].clone());
                assert_matches(
                    &host(&compiled.run(&ordered, 1).unwrap()[0]),
                    &expected,
                    tolerance,
                    &format!("{what}, as a program"),
                );
            }
        }
    }
}
