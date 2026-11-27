// How an output is compared with the reference's, by task, as `support.py` compares them: generated tokens exactly, features by cosine, and scores within `TOLERANCE` with their labels, spans, or order exactly.

use std::collections::HashMap;

use serde_json::Value;

const TOLERANCE: f64 = 1e-3;

// Whether `got` matches `want` for `task`, and what the comparison found.
pub fn outputs(task: &str, want: &Value, got: &Value) -> (bool, String) {
    match task {
        "image-to-image"
        | "unconditional-image-generation"
        | "text-to-image"
        | "image-text-to-image" => images(want, got),
        "text-to-video" | "image-to-video" | "video-to-video" | "image-text-to-video" => {
            frames(want, got)
        }
        "text-to-3d" | "image-to-3d" => meshes(want, got),
        "depth-estimation" => images(&want["depth"], &got["depth"]),
        "image-segmentation" => masks(want, got),
        "keypoint-detection" => keypoints(want, got),
        "keypoint-matching" => matches(want, got),
        "mask-generation" | "mask-tracking" => masks(&regions(want), &regions(got)),
        "audio-to-audio" | "text-to-audio" => waveforms(want, got),
        "text-generation" | "image-text-to-text" | "video-text-to-text" => tokens(want, got),
        "feature-extraction" | "image-feature-extraction" | "video-feature-extraction" => {
            features(want, got)
        }
        "automatic-speech-recognition" => text(want, got),
        "summarization"
        | "translation"
        | "image-to-text"
        | "document-question-answering"
        | "table-question-answering"
        | "any-to-any"
        | "audio-text-to-text"
        | "visual-question-answering" => recorded_keys(want, got),
        "decision"
        | "visual-document-retrieval"
        | "time-series-forecasting"
        | "reinforcement-learning"
        | "sentence-embeddings"
        | "tabular-classification"
        | "tabular-regression" => match largest_difference(want, got) {
            Some(error) => (error < TOLERANCE, format!("largest difference {error:.1e}")),
            None => (false, format!("{task} differs: {got}, expected {want}")),
        },
        _ => scored(task, want, got),
    }
}

fn numbers(value: &Value) -> Vec<f64> {
    match value {
        Value::Number(number) => number.as_f64().into_iter().collect(),
        Value::Array(items) => items.iter().flat_map(numbers).collect(),
        _ => Vec::new(),
    }
}

// The same tokens, or a non-empty start of the reference's that stops where the reference generated one of the end tokens it records, which `generate` keeps and Singe leaves out.
fn tokens(want: &Value, got: &Value) -> (bool, String) {
    let ends = numbers(&want["end_token_ids"]);
    let (want, got) = (
        numbers(&want["generated_token_ids"]),
        numbers(&got["generated_token_ids"]),
    );
    if got.is_empty() {
        return (false, format!("no tokens, expected {}", want.len()));
    }

    let same = want
        .iter()
        .zip(&got)
        .take_while(|(want, got)| want == got)
        .count();
    if same == got.len() && got.len() < want.len() && ends.contains(&want[same]) {
        return (
            true,
            format!("{same} of {} tokens, stopping at the end token", want.len()),
        );
    }
    (
        same == want.len() && same == got.len(),
        format!("{same} of {} tokens", want.len()),
    )
}

fn cosine(left: &[f64], right: &[f64]) -> f64 {
    let dot = left
        .iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum::<f64>();
    let norm = |values: &[f64]| values.iter().map(|value| value * value).sum::<f64>().sqrt();
    dot / (norm(left) * norm(right)).max(1e-12)
}

// Each row's cosine with the reference's: a mean above 0.999 and none below 0.99.
fn features(want: &Value, got: &Value) -> (bool, String) {
    let rows = |value: &Value| match value {
        Value::Array(items) if items.iter().all(Value::is_array) => {
            items.iter().map(numbers).collect::<Vec<_>>()
        }
        other => vec![numbers(other)],
    };
    let (want, got) = (rows(want), rows(got));
    if want.len() != got.len()
        || want
            .iter()
            .zip(&got)
            .any(|(want, got)| want.len() != got.len())
    {
        return (false, "shapes differ".to_owned());
    }

    let similarities = want
        .iter()
        .zip(&got)
        .map(|(want, got)| cosine(want, got))
        .collect::<Vec<_>>();
    let lowest = similarities.iter().copied().fold(f64::INFINITY, f64::min);
    let mean = similarities.iter().sum::<f64>() / similarities.len() as f64;
    (
        mean > 0.999 && lowest > 0.99,
        format!("mean cosine {mean:.6}, lowest {lowest:.6}"),
    )
}

// The least peak signal-to-noise ratio of an image against the reference's, in decibels.
const LEAST_PSNR: f64 = 40.0;

// The pixels of an image's PNG data URL, as pipelines return images.
fn decoded(value: &Value) -> Option<image::RgbImage> {
    let data = value.as_str()?.split_once(',')?.1;
    Some(image::load_from_memory(&base64(data)?).ok()?.to_rgb8())
}

// Standard base64, with or without padding.
fn base64(text: &str) -> Option<Vec<u8>> {
    let sextet = |character: u8| match character {
        b'A'..=b'Z' => Some(character - b'A'),
        b'a'..=b'z' => Some(character - b'a' + 26),
        b'0'..=b'9' => Some(character - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    };
    let mut bytes = Vec::with_capacity(text.len() / 4 * 3);
    let (mut word, mut bits) = (0_u32, 0);
    for character in text.bytes().filter(|&character| character != b'=') {
        word = word << 6 | u32::from(sextet(character)?);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            bytes.push(u8::try_from((word >> bits) & 0xff).ok()?);
        }
    }
    Some(bytes)
}

// Vertex and face counts within 0.1%, and each vertex the reference sampled at a vertex of the mesh within `TOLERANCE`, with that vertex's color within `TOLERANCE`.
// Marching cubes flips a cell where the field is near zero on either side of a rounding, which shifts the indices after it, so vertices are matched by place rather than by index.
fn meshes(want: &Value, got: &Value) -> (bool, String) {
    let point = |value: &Value| -> Option<[f64; 3]> {
        let items = value.as_array()?;
        Some([
            items.first()?.as_f64()?,
            items.get(1)?.as_f64()?,
            items.get(2)?.as_f64()?,
        ])
    };
    let points = |value: &Value| -> Vec<[f64; 3]> {
        value
            .as_array()
            .map_or_else(Vec::new, |items| items.iter().filter_map(point).collect())
    };
    let (vertices, colors) = (points(&got["vertices"]), points(&got["colors"]));
    let faces = got["faces"].as_array().map_or(0, Vec::len);
    let close = |got: usize, want: &Value| {
        want.as_f64().is_some_and(|want| {
            let got = got as f64;
            (got - want).abs() <= want * 1e-3
        })
    };
    if !close(vertices.len(), &want["vertex_count"]) || !close(faces, &want["face_count"]) {
        return (
            false,
            format!(
                "{} vertices and {faces} faces, expected {} and {}",
                vertices.len(),
                want["vertex_count"],
                want["face_count"]
            ),
        );
    }
    // Singe's vertices by cell of a grid as fine as the field's, to find each sampled vertex's nearest.
    let cell = |point: &[f64; 3]| {
        let index = |value: f64| ((value + 1.0) * 64.0).floor() as i64;
        (index(point[0]), index(point[1]), index(point[2]))
    };
    let mut cells = HashMap::<(i64, i64, i64), Vec<usize>>::new();
    for (index, vertex) in vertices.iter().enumerate() {
        cells.entry(cell(vertex)).or_default().push(index);
    }
    let (mut farthest, mut color_error) = (0.0_f64, 0.0_f64);
    for (vertex, color) in points(&want["vertices"])
        .iter()
        .zip(points(&want["colors"]))
    {
        let (x, y, z) = cell(vertex);
        let nearest = (-1..=1)
            .flat_map(|dx| {
                (-1..=1).flat_map(move |dy| (-1..=1).map(move |dz| (x + dx, y + dy, z + dz)))
            })
            .filter_map(|key| cells.get(&key))
            .flatten()
            .map(|&index| {
                let distance = (0..3)
                    .map(|axis| (vertices[index][axis] - vertex[axis]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                (distance, index)
            })
            .min_by(|a, b| a.0.total_cmp(&b.0));
        let Some((distance, index)) = nearest else {
            return (
                false,
                format!("no vertex near the sampled one at {vertex:?}"),
            );
        };
        farthest = farthest.max(distance);
        let got_color = colors.get(index).copied().unwrap_or([f64::NAN; 3]);
        color_error = (0..3)
            .map(|axis| (got_color[axis] - color[axis]).abs())
            .fold(color_error, f64::max);
    }
    (
        farthest < TOLERANCE && color_error < TOLERANCE,
        format!(
            "{} vertices; sampled vertices within {farthest:.1e} of the mesh's, colors within {color_error:.1e}",
            vertices.len()
        ),
    )
}

// A clip's frames, each compared as an image, with the worst frame's finding: the first that could not be compared, or else the one of the lowest PSNR.
fn frames(want: &Value, got: &Value) -> (bool, String) {
    let (Some(want), Some(got)) = (want.as_array(), got.as_array()) else {
        return (false, "a clip that is not an array of frames".to_owned());
    };
    if want.len() != got.len() {
        return (
            false,
            format!("{} frames, expected {}", got.len(), want.len()),
        );
    }

    let results = want
        .iter()
        .zip(got)
        .map(|(want, got)| psnr(want, got))
        .collect::<Vec<_>>();
    let rank = |psnr: Option<f64>| psnr.unwrap_or(f64::NEG_INFINITY);
    let Some((index, (_, detail))) = results
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| rank(left.0).total_cmp(&rank(right.0)))
    else {
        return (false, "a clip of no frames".to_owned());
    };
    (
        results
            .iter()
            .all(|(psnr, _)| psnr.is_some_and(|psnr| psnr >= LEAST_PSNR)),
        format!("{} frames; worst, frame {index}: {detail}", want.len()),
    )
}

// The same size, and a peak signal-to-noise ratio of at least `LEAST_PSNR`.
fn images(want: &Value, got: &Value) -> (bool, String) {
    let (psnr, detail) = psnr(want, got);
    (psnr.is_some_and(|psnr| psnr >= LEAST_PSNR), detail)
}

// The peak signal-to-noise ratio of an image against the reference's, in decibels, or `None` when they cannot be compared, and what the comparison found.
fn psnr(want: &Value, got: &Value) -> (Option<f64>, String) {
    let (Some(want), Some(got)) = (decoded(want), decoded(got)) else {
        return (None, "an output that is not a PNG data URL".to_owned());
    };
    if want.dimensions() != got.dimensions() {
        return (
            None,
            format!(
                "size {:?}, expected {:?}",
                got.dimensions(),
                want.dimensions()
            ),
        );
    }

    let squares = want
        .as_raw()
        .iter()
        .zip(got.as_raw())
        .map(|(&want, &got)| (f64::from(want) - f64::from(got)).powi(2))
        .sum::<f64>();
    let error = squares / want.as_raw().len() as f64;
    let psnr = if error == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0_f64.powi(2) / error).log10()
    };
    (Some(psnr), format!("PSNR {psnr:.1} dB"))
}

// The least signal-to-noise ratio of a waveform against the reference's, in decibels.
const LEAST_SNR: f64 = 30.0;

// The same rate and length, and a signal-to-noise ratio of at least `LEAST_SNR`.
fn waveforms(want: &Value, got: &Value) -> (bool, String) {
    let (want_samples, got_samples) = (numbers(&want["audio"]), numbers(&got["audio"]));
    if want["sampling_rate"] != got["sampling_rate"] || want_samples.len() != got_samples.len() {
        return (
            false,
            format!(
                "{} samples at {} Hz, expected {} at {}",
                got_samples.len(),
                got["sampling_rate"],
                want_samples.len(),
                want["sampling_rate"]
            ),
        );
    }

    let noise = want_samples
        .iter()
        .zip(&got_samples)
        .map(|(want, got)| (want - got).powi(2))
        .sum::<f64>();
    let signal = want_samples
        .iter()
        .map(|sample| sample.powi(2))
        .sum::<f64>();
    let snr = if noise == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (signal / noise).log10()
    };
    (snr >= LEAST_SNR, format!("SNR {snr:.1} dB"))
}

// Generated masks as regions of a mask and a score each, as `masks` compares them.
fn regions(output: &Value) -> Value {
    let masks = output["masks"].as_array().cloned().unwrap_or_default();
    let scores = output["scores"].as_array().cloned().unwrap_or_default();
    Value::Array(
        masks
            .into_iter()
            .zip(scores)
            .map(|(mask, score)| serde_json::json!({"score": score, "mask": mask}))
            .collect(),
    )
}

// The most of a mask's pixels that may differ from the reference's, where scores near a tie at a region's edge pick the other label.
const MOST_DIFFERING: f64 = 2e-3;

// The same regions in order, with the same labels and scores, and masks differing in at most `MOST_DIFFERING` of their pixels.
fn masks(want: &Value, got: &Value) -> (bool, String) {
    let (Some(want), Some(got)) = (want.as_array(), got.as_array()) else {
        return (false, "an output that is not a list of regions".to_owned());
    };
    if want.len() != got.len() {
        return (
            false,
            format!("{} regions, expected {}", got.len(), want.len()),
        );
    }

    let strip = |segments: &[Value]| {
        Value::Array(
            segments
                .iter()
                .map(|segment| {
                    let mut segment = segment.clone();
                    if let Some(fields) = segment.as_object_mut() {
                        fields.remove("mask");
                    }
                    segment
                })
                .collect(),
        )
    };
    let (matched, detail) = scored("image-segmentation", &strip(want), &strip(got));
    if !matched {
        return (matched, detail);
    }

    let mut worst = 0.0_f64;
    for (want, got) in want.iter().zip(got) {
        let (Some(want), Some(got)) = (decoded(&want["mask"]), decoded(&got["mask"])) else {
            return (false, "a mask that is not a PNG data URL".to_owned());
        };
        if want.dimensions() != got.dimensions() {
            return (false, "masks of different sizes".to_owned());
        }
        let differing = want
            .pixels()
            .zip(got.pixels())
            .filter(|(want, got)| want != got)
            .count();
        let share = differing as f64 / f64::from(want.width() * want.height());
        worst = worst.max(share);
    }
    (
        worst <= MOST_DIFFERING,
        format!(
            "{} regions, masks differing in at most {:.2}% of pixels",
            want.len(),
            worst * 100.0
        ),
    )
}

// The same keypoints within a hundredth of a pixel, scores within `TOLERANCE`, and the descriptors the reference recorded, its first keypoints', by cosine.
fn keypoints(want: &Value, got: &Value) -> (bool, String) {
    let count = |value: &Value, key: &str| value[key].as_array().map_or(0, Vec::len);
    if count(want, "keypoints") != count(got, "keypoints") {
        return (
            false,
            format!(
                "{} keypoints, expected {}",
                count(got, "keypoints"),
                count(want, "keypoints")
            ),
        );
    }

    let largest = |key: &str| {
        numbers(&want[key])
            .iter()
            .zip(numbers(&got[key]))
            .map(|(want, got)| (want - got).abs())
            .fold(0.0, f64::max)
    };
    let (points, scores) = (largest("keypoints"), largest("scores"));
    let described = count(want, "descriptors");
    let got_descriptors = Value::Array(
        got["descriptors"]
            .as_array()
            .map(|rows| rows.iter().take(described).cloned().collect())
            .unwrap_or_default(),
    );
    let (matched, detail) = features(&want["descriptors"], &got_descriptors);
    (
        matched && points < 1e-2 && scores < TOLERANCE,
        format!(
            "{} keypoints, largest offset {points:.1e}, largest score difference {scores:.1e}, {detail}",
            count(want, "keypoints")
        ),
    )
}

// The same pairs of keypoints, in any order, since near-equal scores may sort either way, each with its score within `TOLERANCE`.
fn matches(want: &Value, got: &Value) -> (bool, String) {
    let pairs = |value: &Value| {
        value
            .as_array()
            .map(|matches| {
                matches
                    .iter()
                    .map(|found| {
                        let point = |key: &str| {
                            numbers(&Value::Array(
                                found[key]
                                    .as_object()
                                    .map(|point| point.values().cloned().collect())
                                    .unwrap_or_default(),
                            ))
                        };
                        (
                            format!(
                                "{:?} {:?}",
                                point("keypoint_image_0"),
                                point("keypoint_image_1")
                            ),
                            found["score"].as_f64().unwrap_or(f64::NAN),
                        )
                    })
                    .collect::<std::collections::BTreeMap<_, _>>()
            })
            .unwrap_or_default()
    };
    let (want, got) = (pairs(want), pairs(got));
    if !want.keys().eq(got.keys()) {
        let missing = want.keys().filter(|key| !got.contains_key(*key)).count();
        let extra = got.keys().filter(|key| !want.contains_key(*key)).count();
        return (
            false,
            format!(
                "{} matches, expected {}: {missing} missing, {extra} extra",
                got.len(),
                want.len()
            ),
        );
    }

    let error = want
        .iter()
        .map(|(key, want)| (want - got[key]).abs())
        .fold(0.0, f64::max);
    (
        error < TOLERANCE,
        format!(
            "{} matches, largest score difference {error:.1e}",
            want.len()
        ),
    )
}

// The keys the reference records, the same, which Singe's output may extend.
fn recorded_keys(want: &Value, got: &Value) -> (bool, String) {
    let Some(fields) = want.as_object() else {
        return (want == got, got.to_string());
    };
    let shown = fields
        .keys()
        .map(|key| (key.clone(), got.get(key).cloned().unwrap_or(Value::Null)))
        .collect::<serde_json::Map<_, _>>();
    (
        fields
            .iter()
            .all(|(key, value)| got.get(key) == Some(value)),
        Value::Object(shown).to_string(),
    )
}

// The largest difference of two outputs' numbers, relative above one, if everything else in them is the same.
fn largest_difference(want: &Value, got: &Value) -> Option<f64> {
    match (want, got) {
        (Value::Object(want), Value::Object(got)) => {
            if want.len() != got.len() {
                return None;
            }
            want.iter().try_fold(0.0, |most, (key, value)| {
                Some(f64::max(most, largest_difference(value, got.get(key)?)?))
            })
        }
        (Value::Array(want), Value::Array(got)) => {
            if want.len() != got.len() {
                return None;
            }
            want.iter().zip(got).try_fold(0.0, |most, (want, got)| {
                Some(f64::max(most, largest_difference(want, got)?))
            })
        }
        // Relative to the reference's value above one, which records six significant digits.
        (Value::Number(want), Value::Number(got)) => {
            let want = want.as_f64()?;
            Some((want - got.as_f64()?).abs() / want.abs().max(1.0))
        }
        _ => (want == got).then_some(0.0),
    }
}

// The same text, but for the whitespace around it.
fn text(want: &Value, got: &Value) -> (bool, String) {
    let text = |value: &Value| value["text"].as_str().unwrap_or_default().trim().to_owned();
    (text(want) == text(got), format!("{:?}", text(got)))
}

// Everything but the scores exactly, and every score within `TOLERANCE`.
fn scored(task: &str, want: &Value, got: &Value) -> (bool, String) {
    let (want_scores, got_scores) = (scores(want), scores(got));
    let shape = |value: &Value| without_scores(value);
    if shape(want) != shape(got) || want_scores.len() != got_scores.len() {
        return (false, format!("{task} differs: {got}, expected {want}"));
    }

    let error = want_scores
        .iter()
        .zip(&got_scores)
        .map(|(want, got)| (want - got).abs())
        .fold(0.0, f64::max);
    (
        error < TOLERANCE,
        format!("largest score difference {error:.1e}"),
    )
}

// The values under every `score` or `scores` key, in order.
fn scores(value: &Value) -> Vec<f64> {
    match value {
        Value::Array(items) => items.iter().flat_map(scores).collect(),
        Value::Object(fields) => fields
            .iter()
            .flat_map(|(key, value)| {
                if key == "score" || key == "scores" {
                    numbers(value)
                } else {
                    scores(value)
                }
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn without_scores(value: &Value) -> Value {
    match value {
        Value::Array(items) => Value::Array(items.iter().map(without_scores).collect()),
        Value::Object(fields) => Value::Object(
            fields
                .iter()
                .filter(|(key, _)| *key != "score" && *key != "scores")
                .map(|(key, value)| (key.clone(), without_scores(value)))
                .collect(),
        ),
        other => other.clone(),
    }
}

// A half-precision case's looser match: the top label's or answer's score within this, and the first `LOOSE_TOKENS` tokens of a generation.
const LOOSE_SCORE: f64 = 0.05;

const LOOSE_TOKENS: usize = 8;

// A half-precision case's match, as `support.py` makes it: the same top label or answer with its score within `LOOSE_SCORE`, or the same first `LOOSE_TOKENS` generated tokens, and for other tasks the match `outputs` makes.
pub fn loose(task: &str, want: &Value, got: &Value) -> (bool, String) {
    fn top(mut value: &Value) -> &Value {
        while let Value::Array(items) = value {
            match items.first() {
                Some(first) => value = first,
                None => return &Value::Null,
            }
        }
        value
    }

    match task {
        "text-generation" => {
            let (want, got) = (
                numbers(&want["generated_token_ids"]),
                numbers(&got["generated_token_ids"]),
            );
            let count = LOOSE_TOKENS.min(want.len());
            let same = want
                .iter()
                .zip(&got)
                .take_while(|(want, got)| want == got)
                .count();
            (
                same >= count,
                format!("{same} of {} tokens, {count} needed", want.len()),
            )
        }
        "text-classification" | "question-answering" => {
            let key = if task == "text-classification" {
                "label"
            } else {
                "answer"
            };
            let (want, got) = (top(want), top(got));
            let error = match (want["score"].as_f64(), got["score"].as_f64()) {
                (Some(want), Some(got)) => (want - got).abs(),
                _ => f64::INFINITY,
            };
            (
                want[key] == got[key] && error < LOOSE_SCORE,
                format!("top {key} {}, score difference {error:.1e}", got[key]),
            )
        }
        _ => outputs(task, want, got),
    }
}
