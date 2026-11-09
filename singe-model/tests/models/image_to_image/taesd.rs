#[cfg(feature = "cuda_13_4")]
use std::fs;
use std::path::{Path, PathBuf};

use singe::{DataType, Device};
#[cfg(feature = "cuda_13_4")]
use singe::{
    Tensor,
    nn::{Model, Sequential},
};
use singe_model::vision::image_to_image::{Autoencoder, load_autoencoder};

#[cfg(feature = "cuda_13_4")]
use singe_testing::{EVERY_PATH, check_paths, lock_device, pattern_image, target_directory};
use singe_testing::{Reference, Tolerance, assert_matches, host, reference};

const MODELS: [&str; 2] = ["base", "bilinear"];

fn directory(name: &str) -> PathBuf {
    Path::new(singe_testing::references!())
        .join("image_to_image/taesd")
        .join(name)
}

fn tiny(name: &str) -> Reference {
    reference!(format!("image_to_image/taesd/{name}"))
}

fn type_key(data_type: DataType) -> &'static str {
    match data_type {
        DataType::F32 => "f32",
        _ => "f16",
    }
}

fn tolerance(data_type: DataType) -> Tolerance {
    match data_type {
        DataType::F32 => Tolerance::absolute(1e-4, "F32 leaves room for arithmetic alone"),
        _ => Tolerance::absolute(
            0.02,
            "F16 keeps eleven mantissa bits, and a chain of convolutions rounds at each",
        ),
    }
}

fn load(name: &str, device: Device, data_type: DataType) -> Autoencoder {
    load_autoencoder(directory(name), device, data_type).unwrap()
}

// Each tiny autoencoder encodes fixed images and decodes fixed latents as `diffusers` does, in both types.
fn check_references(device: Device) {
    for name in MODELS {
        let reference = tiny(name);
        for data_type in [DataType::F32, DataType::F16] {
            let key = type_key(data_type);
            let what = format!("{name} {data_type:?} on {device:?}");
            let model = load(name, device, data_type);
            let images = reference.tensor("inputs.images", device, data_type);
            let latents = reference.tensor("inputs.latents", device, data_type);
            assert_matches(
                &host(&model.encode(&images).unwrap()),
                &reference.values(&format!("{key}.latents")),
                tolerance(data_type),
                &format!("{what} latents"),
            );
            assert_matches(
                &host(&model.decode(&latents).unwrap()),
                &reference.values(&format!("{key}.images")),
                tolerance(data_type),
                &format!("{what} images"),
            );
        }
    }
}

#[test]
fn test_tiny_taesd_matches_diffusers_on_cpu() {
    check_references(Device::Cpu);
}

#[test]
fn test_a_snapshot_of_another_diffusers_class_is_refused() {
    let error = load_autoencoder(
        Path::new(singe_testing::references!()).join("text_generation/gpt2/base"),
        Device::Cpu,
        DataType::F32,
    )
    .unwrap_err();
    assert!(error.to_string().contains("model config"), "{error}");
}

#[cfg(feature = "cuda_13_4")]
fn real_snapshot() -> PathBuf {
    singe_testing::snapshot("TAESD")
}

// Writes a decoded image in [-1, 1] as a binary PPM a person can look at.
#[cfg(feature = "cuda_13_4")]
fn write_image(image: &Tensor, name: &str) {
    let dimensions = image.shape().dimensions();
    let (height, width) = (
        usize::try_from(dimensions[2]).unwrap(),
        usize::try_from(dimensions[3]).unwrap(),
    );
    let values = host(image);
    let plane = height * width;
    let mut bytes = format!("P6\n{width} {height}\n255\n").into_bytes();
    for pixel in 0..plane {
        for channel in 0..3 {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "the value is clamped to a byte's range"
            )]
            bytes.push(((values[channel * plane + pixel] + 1.0) * 127.5).clamp(0.0, 255.0) as u8);
        }
    }
    let path = target_directory(name);
    fs::write(&path, bytes).unwrap();
    eprintln!("decoded image: {}", path.display());
}

// TAESD encodes a fixed 256 by 256 image and decodes its latents as `diffusers` does.
#[cfg(feature = "cuda_13_4")]
fn check_real(device: Device, data_type: DataType, tolerance: Tolerance) {
    let reference: Reference = reference!("image_to_image/taesd/real");
    let key = type_key(data_type);
    let size = reference.metadata()["size"].as_i64().unwrap();
    let stride = usize::try_from(reference.metadata()["stride"].as_i64().unwrap()).unwrap();
    let model = load_autoencoder(real_snapshot(), device, data_type).unwrap();
    let what = format!("TAESD {data_type:?} on {device:?}");
    let image = pattern_image(size, device).cast(data_type).unwrap();
    assert_matches(
        &host(&model.encode(&image).unwrap()),
        &reference.values(&format!("{key}.latents")),
        tolerance,
        &format!("{what} latents"),
    );
    let latents = reference.tensor("f32.latents", device, data_type);
    let decoded = model.decode(&latents).unwrap();
    write_image(&decoded, &format!("taesd-{}.ppm", type_key(data_type)));
    let values = host(&decoded);
    let pixels = usize::try_from(size).unwrap();
    let samples = (0..3)
        .flat_map(|channel| {
            (0..pixels).step_by(stride).flat_map(move |y| {
                (0..pixels)
                    .step_by(stride)
                    .map(move |x| (channel * pixels + y) * pixels + x)
            })
        })
        .map(|index| values[index])
        .collect::<Vec<_>>();
    assert_matches(
        &samples,
        &reference.values(&format!("{key}.image_samples")),
        tolerance,
        &format!("{what} decoded image"),
    );
}

#[cfg(feature = "cuda_13_4")]
#[test]
#[ignore = "needs real snapshots, through SINGE_SNAPSHOT_<MODEL>"]
fn test_taesd_f32_matches_diffusers_on_cpu() {
    check_real(
        Device::Cpu,
        DataType::F32,
        Tolerance::absolute(
            1e-3,
            "F32 sums a 256 by 256 image's convolutions in another order",
        ),
    );
}

#[cfg(feature = "cuda_13_4")]
mod cuda {
    use super::*;

    const DEVICE: Device = Device::Cuda(0);

    // A captured half computes what it computes eagerly on every path.
    fn check_half(
        half: &Model<Sequential>,
        input: &Tensor,
        expected: &[f32],
        label: &str,
        data_type: DataType,
    ) {
        let (graph, _) = half.capture_graph("input", input.clone()).unwrap();
        check_paths(
            &graph,
            &[("input", input.clone())],
            expected,
            tolerance(data_type),
            &format!("taesd {label} {data_type:?}"),
            &EVERY_PATH,
            DEVICE,
        );
    }

    #[test]
    fn test_tiny_taesd_matches_diffusers_on_cuda() {
        let Some(_lock) = lock_device() else {
            return;
        };
        check_references(DEVICE);
    }

    #[test]
    fn test_a_tuned_tiny_taesd_compiles_and_exports_from_its_records() {
        let Some(_lock) = lock_device() else {
            return;
        };
        let reference = tiny("base");
        for data_type in [DataType::F32, DataType::F16] {
            let key = type_key(data_type);
            let model = load("base", DEVICE, data_type);
            let images = reference.tensor("inputs.images", DEVICE, data_type);
            let latents = reference.tensor("inputs.latents", DEVICE, data_type);
            check_half(
                model.encoder(),
                &images,
                &reference.values(&format!("{key}.latents")),
                "encoder",
                data_type,
            );
            check_half(
                model.decoder(),
                &latents,
                &reference.values(&format!("{key}.images")),
                "decoder",
                data_type,
            );
        }
    }

    #[test]
    #[ignore = "needs real snapshots, through SINGE_SNAPSHOT_<MODEL>"]
    fn test_taesd_matches_diffusers_on_cuda() {
        let Some(_lock) = lock_device() else {
            return;
        };
        check_real(
            DEVICE,
            DataType::F32,
            Tolerance::absolute(
                1e-3,
                "F32 sums a 256 by 256 image's convolutions in another order",
            ),
        );
        check_real(
            DEVICE,
            DataType::F16,
            Tolerance::absolute(0.05, "F16 rounds each of a deep stack of convolutions"),
        );
    }
}
