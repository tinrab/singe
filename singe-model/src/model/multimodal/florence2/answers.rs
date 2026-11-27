//! Florence-2's answers parsed as `Florence2Processor.post_process_generation` parses them: location tokens dequantized to the image's pixels, truncated, and grouped by the phrases before them.
//! Its patterns' lookaheads become searches for the first of the markers they look ahead for, since Rust's regular expressions have none.

use std::sync::LazyLock;

use regex::Regex;
use serde::Serialize;

/// A task's answer as structure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum ParsedAnswer {
    Boxes {
        bboxes: Vec<Vec<i32>>,
        labels: Vec<String>,
    },
    QuadBoxes {
        quad_boxes: Vec<Vec<i32>>,
        labels: Vec<String>,
    },
    Polygons {
        polygons: Vec<Vec<Vec<i32>>>,
        labels: Vec<String>,
    },
    BoxesOrPolygons {
        bboxes: Vec<Vec<i32>>,
        bboxes_labels: Vec<String>,
        polygons: Vec<Vec<Vec<i32>>>,
        polygons_labels: Vec<String>,
    },
}

/// The bins each axis's locations fall in.
const BINS: f32 = 1000.0;

/// The markers a phrase ends at, as the patterns' lookaheads read them; polygons' phrases also end at `<poly>`.
const PHRASE_ENDS: &[&str] = &[
    "<od>", "</od>", "<box>", "</box>", "<bbox>", "</bbox>", "<loc_",
];

fn pattern(expression: &str) -> Regex {
    Regex::new(expression).unwrap_or_else(|error| unreachable!("{error}"))
}

static PHRASES: LazyLock<Regex> = LazyLock::new(|| pattern(r"([^<]+(?:<loc_\d+>){4,})"));
static BARE_BOXES: LazyLock<Regex> = LazyLock::new(|| pattern(r"(?:(?:<loc_\d+>){4,})"));

static BOX: LazyLock<Regex> =
    LazyLock::new(|| pattern(r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"));

static LOCATION: LazyLock<Regex> = LazyLock::new(|| pattern(r"<loc_(\d+)>"));

static QUAD: LazyLock<Regex> = LazyLock::new(|| {
    pattern(
        r"(.+?)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>",
    )
});

static POLYGON_PHRASES: LazyLock<Regex> =
    LazyLock::new(|| pattern(r"([^<]+(?:<loc_\d+>|<sep>|<poly>|</poly>){4,})"));

static BARE_POLYGONS: LazyLock<Regex> =
    LazyLock::new(|| pattern(r"(?:(?:<loc_\d+>|<sep>|<poly>|</poly>){4,})"));

static LEADING_LOCATION: LazyLock<Regex> = LazyLock::new(|| pattern(r"^<loc_\d+>"));
static POLYGON: LazyLock<Regex> = LazyLock::new(|| pattern(r"<poly>(.*?)</poly>"));
static POLYGON_POINTS: LazyLock<Regex> = LazyLock::new(|| pattern(r"((?:<loc_\d+>)+)(?:<sep>|$)"));

/// Returns `(bin + 0.5) · size / bins` truncated, as `dequantize` computes it in `F32`.
fn dequantized(bin: i32, size: u32) -> i32 {
    ((bin as f32 + 0.5) * (size as f32 / BINS)) as i32
}

/// Returns the phrase before the first of `ends` in `text`, after leading whitespace and stripped, or `None` when no end follows on the phrase's line, as `^\s*(.*?)(?=…)` matches.
fn phrase(text: &str, ends: &[&str]) -> Option<String> {
    let text = text.trim_start();
    let end = ends.iter().filter_map(|end| text.find(end)).min()?;
    let phrase = &text[..end];
    (!phrase.contains('\n')).then(|| phrase.trim().to_owned())
}

/// Returns `text` without the characters outside ASCII, as `.encode("ascii", "ignore")` drops them.
fn ascii(text: &str) -> String {
    text.chars().filter(char::is_ascii).collect()
}

fn bins(text: &str) -> Vec<i32> {
    LOCATION
        .captures_iter(text)
        .filter_map(|capture| capture[1].parse().ok())
        .collect()
}

/// Returns `bins` as points dequantized to an image of `width × height`, x first.
fn points(bins: &[i32], (width, height): (u32, u32)) -> Vec<i32> {
    bins.as_chunks::<2>()
        .0
        .iter()
        .flat_map(|&[x, y]| [dequantized(x, width), dequantized(y, height)])
        .collect()
}

fn boxes(text: &str, size: (u32, u32)) -> Vec<Vec<i32>> {
    BOX.captures_iter(text)
        .map(|capture| {
            let bins = (1..=4)
                .map(|group| capture[group].parse().unwrap_or(0))
                .collect::<Vec<i32>>();
            points(&bins, size)
        })
        .collect()
}

/// `parse_description_with_bboxes_from_text_and_spans`: each phrase's boxes, each box an instance with the phrase as its label, leaving out the phrases in `banned_phrases`.
fn described_boxes(
    text: &str,
    size: (u32, u32),
    empty_phrases: bool,
    banned_phrases: &[String],
) -> Vec<(Vec<i32>, String)> {
    let phrases = if empty_phrases {
        &*BARE_BOXES
    } else {
        &*PHRASES
    };
    let mut instances = Vec::new();
    for found in phrases.find_iter(text) {
        let phrase_text = found
            .as_str()
            .replacen("<ground>", "", 1)
            .replacen("<obj>", "", 1);
        let Some(label) = phrase(&phrase_text, PHRASE_ENDS) else {
            continue;
        };
        if banned_phrases.contains(&label) {
            continue;
        }

        let label = ascii(&label);
        for found_box in boxes(&phrase_text, size) {
            instances.push((found_box, label.clone()));
        }
    }
    instances
}

/// `parse_description_with_polygons_from_text_and_spans` without a box at the start: each phrase's polygons, each `<poly>` group an instance.
fn described_polygons(
    text: &str,
    size: (u32, u32),
    empty_phrases: bool,
) -> Vec<(Vec<Vec<i32>>, String)> {
    let phrases = if empty_phrases {
        &*BARE_POLYGONS
    } else {
        &*POLYGON_PHRASES
    };
    let mut ends = PHRASE_ENDS.to_vec();
    ends.push("<poly>");
    let mut instances = Vec::new();
    for found in phrases.find_iter(text) {
        let phrase_text = found.as_str();
        let stripped = LEADING_LOCATION.replace(phrase_text, "");
        let Some(label) = phrase(&stripped, &ends) else {
            continue;
        };
        let groups = if phrase_text.contains("<poly>") && phrase_text.contains("</poly>") {
            POLYGON
                .captures_iter(phrase_text)
                .map(|capture| capture[1].to_owned())
                .collect()
        } else {
            vec![phrase_text.to_owned()]
        };
        for group in groups {
            let polygons = POLYGON_POINTS
                .captures_iter(&group)
                .map(|capture| {
                    let mut bins = bins(&capture[1]);
                    if bins.len() % 2 == 1 {
                        bins.pop();
                    }
                    points(&bins, size)
                })
                .collect::<Vec<_>>();
            if !polygons.is_empty() {
                instances.push((polygons, label.clone()));
            }
        }
    }
    instances
}

/// Parses `answer`, cleaned of its start, end, and padding tokens, for the task `post_process_generation` gives `task`, on an image of `width × height`; `None` for a task whose answer is text.
#[must_use]
pub fn parsed(
    task: &str,
    answer: &str,
    size: (u32, u32),
    banned_phrases: &[String],
) -> Option<ParsedAnswer> {
    Some(match task {
        "<OD>" | "<DENSE_REGION_CAPTION>" => {
            let (bboxes, labels) = described_boxes(answer, size, false, &[])
                .into_iter()
                .unzip();
            ParsedAnswer::Boxes { bboxes, labels }
        }
        "<REGION_PROPOSAL>" => {
            let (bboxes, labels) = described_boxes(answer, size, true, &[]).into_iter().unzip();
            ParsedAnswer::Boxes { bboxes, labels }
        }
        "<CAPTION_TO_PHRASE_GROUNDING>" => {
            let (bboxes, labels) = described_boxes(answer, size, false, banned_phrases)
                .into_iter()
                .unzip();
            ParsedAnswer::Boxes { bboxes, labels }
        }
        "<OCR_WITH_REGION>" => {
            let (quad_boxes, labels) = QUAD
                .captures_iter(answer)
                .map(|capture| {
                    let bins = (2..=9)
                        .map(|group| capture[group].parse().unwrap_or(0))
                        .collect::<Vec<i32>>();
                    (points(&bins, size), capture[1].trim().to_owned())
                })
                .unzip();
            ParsedAnswer::QuadBoxes { quad_boxes, labels }
        }
        "<REFERRING_EXPRESSION_SEGMENTATION>" | "<REGION_TO_SEGMENTATION>" => {
            let (polygons, labels) = described_polygons(answer, size, true).into_iter().unzip();
            ParsedAnswer::Polygons { polygons, labels }
        }
        "<OPEN_VOCABULARY_DETECTION>" => {
            let ((bboxes, bboxes_labels), (polygons, polygons_labels)) =
                if answer.contains("<poly>") {
                    let polygons: (Vec<_>, Vec<_>) =
                        described_polygons(answer, size, false).into_iter().unzip();
                    ((Vec::new(), Vec::new()), polygons)
                } else {
                    let boxes: (Vec<_>, Vec<_>) = described_boxes(answer, size, false, &[])
                        .into_iter()
                        .unzip();
                    (boxes, (Vec::new(), Vec::new()))
                };
            ParsedAnswer::BoxesOrPolygons {
                bboxes,
                bboxes_labels,
                polygons,
                polygons_labels,
            }
        }
        _ => return None,
    })
}
