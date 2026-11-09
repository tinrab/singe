#[cfg(feature = "cuda_13_4")]
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use singe::{DataType, Device, Shape, Tensor};
#[cfg(feature = "cuda_13_4")]
use singe::{
    backend::cuda::LoweringPolicy,
    graph::{Graph, GraphCudaOptions, GraphPrepareOptions},
};
use singe_model::vision::image_feature_extraction::{ImageEncoder, load_image_encoder};

#[cfg(feature = "cuda_13_4")]
use singe_testing::{EVERY_PATH, assert_within_half_gap, check_paths, lock_device, pattern_image};
use singe_testing::{Reference, Tolerance, assert_matches, host, reference};

const MODELS: [&str; 2] = ["base", "no_qkv_bias"];

fn directory(name: &str) -> PathBuf {
    Path::new(singe_testing::references!())
        .join("image_feature_extraction/dinov2")
        .join(name)
}

fn tiny(name: &str) -> Reference {
    reference!(format!("image_feature_extraction/dinov2/{name}"))
}

fn type_key(data_type: DataType) -> &'static str {
    match data_type {
        DataType::F32 => "f32",
        _ => "bf16",
    }
}

fn tolerance(data_type: DataType) -> Tolerance {
    match data_type {
        DataType::F32 => Tolerance::absolute(1e-4, "F32 leaves room for arithmetic alone"),
        _ => Tolerance::absolute(
            0.05,
            "BF16 keeps eight mantissa bits, and Singe accumulates attention and norms in F32 where PyTorch rounds some steps",
        ),
    }
}

fn images(reference: &Reference, device: Device, data_type: DataType) -> Tensor {
    reference.tensor("inputs.images", device, data_type)
}

fn load(name: &str, device: Device, data_type: DataType) -> ImageEncoder {
    load_image_encoder(directory(name), device, data_type).unwrap()
}

// Each tiny model's final hidden states follow `transformers`, in both types, and its pooled output is the class token's.
fn check_references(device: Device) {
    for name in MODELS {
        let reference = tiny(name);
        for data_type in [DataType::F32, DataType::Bf16] {
            let what = format!("{name} {data_type:?} on {device:?}");
            let model = load(name, device, data_type);
            let hidden = model
                .forward(&images(&reference, device, data_type))
                .unwrap();
            let expected = reference.values(&format!("{}.hidden", type_key(data_type)));
            assert_matches(
                &host(&hidden),
                &expected,
                tolerance(data_type),
                &format!("{what} hidden states"),
            );
            let dimensions = hidden.shape().dimensions();
            let (tokens, width) = (
                usize::try_from(dimensions[1]).unwrap(),
                usize::try_from(dimensions[2]).unwrap(),
            );
            let classes = expected
                .chunks_exact(tokens * width)
                .flat_map(|image| image[..width].to_vec())
                .collect::<Vec<_>>();
            assert_matches(
                &host(&model.layer().pooled(&hidden).unwrap()),
                &classes,
                tolerance(data_type),
                &format!("{what} class embeddings"),
            );
        }
    }
}

#[test]
fn test_tiny_dinov2_matches_transformers_on_cpu() {
    check_references(Device::Cpu);
}

#[test]
fn test_an_image_of_another_size_is_refused() {
    let model = load("base", Device::Cpu, DataType::F32);
    let image = Tensor::from_vec(
        Device::Cpu,
        Shape::create([1, 3, 20, 20]).unwrap(),
        vec![0.0_f32; 1200],
    )
    .unwrap();
    let error = model.forward(&image).unwrap_err();
    assert!(error.to_string().contains("patch count"), "{error}");
}

#[cfg(feature = "cuda_13_4")]
fn real_snapshot() -> PathBuf {
    singe_testing::snapshot("DINOV2")
}

#[cfg(feature = "cuda_13_4")]
fn real_reference() -> Reference {
    reference!("image_feature_extraction/dinov2/real")
}

// The tokens of DINOv2-small's hidden states the reference keeps, every `token_stride`th from the class token.
#[cfg(feature = "cuda_13_4")]
fn sampled(hidden: &Tensor, reference: &Reference) -> Vec<f32> {
    let stride = reference.metadata()["token_stride"].as_i64().unwrap();
    let tokens = hidden.shape().dimensions()[1];
    let rows = (0..tokens)
        .step_by(usize::try_from(stride).unwrap())
        .map(|token| hidden.narrow(1, token, 1).unwrap())
        .collect::<Vec<_>>();
    host(&Tensor::concat(&rows, 1).unwrap())
}

#[cfg(feature = "cuda_13_4")]
#[test]
#[ignore = "needs real snapshots, through SINGE_SNAPSHOT_<MODEL>"]
fn test_dinov2_f32_matches_transformers_on_cpu() {
    let reference = real_reference();
    let model = load_image_encoder(real_snapshot(), Device::Cpu, DataType::F32).unwrap();
    let image = pattern_image(reference.metadata()["size"].as_i64().unwrap(), Device::Cpu);
    let hidden = model.forward(&image).unwrap();
    assert_matches(
        &sampled(&hidden, &reference),
        &reference.values("f32.tokens"),
        Tolerance::absolute(
            1e-3,
            "F32 sums twelve layers over 1370 tokens in another order",
        ),
        "DINOv2-small F32 on CPU",
    );
}

#[cfg(feature = "cuda_13_4")]
mod cuda {
    use super::*;

    const DEVICE: Device = Device::Cuda(0);

    fn run(graph: &Graph, inputs: &HashMap<String, Tensor>, options: GraphCudaOptions) -> Tensor {
        let mut executor = graph
            .prepare(GraphPrepareOptions::new(DEVICE).with_cuda_options(options))
            .unwrap();
        executor.run(inputs).unwrap().remove(0)
    }

    fn generated_kernels() -> GraphCudaOptions {
        GraphCudaOptions::new().with_lowering_policy(LoweringPolicy::generated_kernels())
    }

    #[test]
    fn test_tiny_dinov2_matches_transformers_on_cuda() {
        let Some(_lock) = lock_device() else {
            return;
        };
        check_references(DEVICE);
    }

    // A captured tiny model computes what it computes eagerly on every path.
    #[test]
    fn test_a_tuned_tiny_dinov2_compiles_and_exports_from_its_records() {
        let Some(_lock) = lock_device() else {
            return;
        };
        let reference = tiny("base");
        for data_type in [DataType::F32, DataType::Bf16] {
            let model = load("base", DEVICE, data_type);
            let input = images(&reference, DEVICE, data_type);
            let (graph, _) = model.capture_graph("images", input.clone()).unwrap();
            check_paths(
                &graph,
                &[("images", input)],
                &reference.values(&format!("{}.hidden", type_key(data_type))),
                tolerance(data_type),
                &format!("dinov2 {data_type:?}"),
                &EVERY_PATH,
                DEVICE,
            );
        }
    }

    // DINOv2-small at its native 518 by 518 attends over 1370 tokens without a mask, eagerly, compiled with cuDNN's attention, and through generated kernels.
    // F32 follows PyTorch's F32, and BF16, whose rounding over twelve layers moves PyTorch's own results by up to 0.4, stays within twice PyTorch's BF16 distance from F32, worst and on average.
    #[test]
    #[ignore = "needs real snapshots, through SINGE_SNAPSHOT_<MODEL>"]
    fn test_dinov2_matches_transformers_on_cuda() {
        let Some(_lock) = lock_device() else {
            return;
        };
        let reference = real_reference();
        let size = reference.metadata()["size"].as_i64().unwrap();
        let wide = reference.values("f32.tokens");
        for data_type in [DataType::F32, DataType::Bf16] {
            let model = load_image_encoder(real_snapshot(), DEVICE, data_type).unwrap();
            let image = pattern_image(size, DEVICE).cast(data_type).unwrap();
            let check = |hidden: &Tensor, how: &str| {
                let actual = sampled(hidden, &reference);
                let what = format!("DINOv2-small {data_type:?} {how}");
                if data_type == DataType::F32 {
                    assert_matches(
                        &actual,
                        &wide,
                        Tolerance::absolute(
                            1e-3,
                            "F32 sums twelve layers over 1370 tokens in another order",
                        ),
                        &what,
                    );
                    return;
                }
                assert_within_half_gap(&actual, &wide, reference.half_gap(), 2.0, &what);
            };
            check(&model.forward(&image).unwrap(), "eagerly");
            let (graph, _) = model.capture_graph("images", image.clone()).unwrap();
            let inputs = HashMap::from([("images".to_owned(), image)]);
            check(&run(&graph, &inputs, GraphCudaOptions::new()), "compiled");
            check(
                &run(&graph, &inputs, generated_kernels()),
                "through generated kernels",
            );
        }
    }
}
