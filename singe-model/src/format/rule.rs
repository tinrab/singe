//! How another format's tensor names become Singe's: ordered rules, the first that matches a stored name deciding where it goes.
//!
//! A pattern is dot-separated segments, where `*` stands for any one segment.
//! It matches a stored name that starts with those segments, and the segments after them carry over to each target.
//! Each `*` of a target takes the next segment a `*` of its pattern matched.

use std::{
    borrow::Cow,
    fmt::{Debug, Formatter, Result as FmtResult},
    sync::Arc,
};

use singe::{
    Error as SingeError, Result as SingeResult, Tensor,
    nn::{Derivation, LoadPlan, TensorSource},
};

use crate::error::Result;

/// A stored name a pattern matched: what each `*` matched, and the segments past the pattern.
#[derive(Debug, Clone)]
pub struct Matched<'a> {
    pub name: &'a str,
    pub captures: Vec<&'a str>,
    /// The segments past the pattern, each with a leading dot, or empty.
    pub rest: String,
}

impl Matched<'_> {
    /// Returns `pattern` with its `*`s filled by the captures in order, and the rest appended.
    #[must_use]
    pub fn target(&self, pattern: &str) -> String {
        let mut captures = self.captures.iter();
        let mut target = pattern
            .split('.')
            .map(|segment| {
                if segment == "*" {
                    captures.next().copied().unwrap_or(segment)
                } else {
                    segment
                }
            })
            .collect::<Vec<_>>()
            .join(".");
        target.push_str(&self.rest);
        target
    }

    /// Reads capture `index` as a number, such as a layer's or an expert's.
    ///
    /// # Errors
    ///
    /// Returns an error if there is no capture `index`, or it is not an integer.
    pub fn number(&self, index: usize) -> SingeResult<i64> {
        self.captures
            .get(index)
            .and_then(|capture| capture.parse().ok())
            .ok_or_else(|| SingeError::invalid_argument(format!("`{}` has no number", self.name)))
    }
}

type Custom = dyn Fn(&Matched<'_>) -> SingeResult<Vec<(String, TensorSource)>> + Send + Sync;

#[derive(Clone)]
enum Action {
    Rename(String),
    Skip,
    Transpose(String),
    /// An even split of a tensor along `axis` into one part per target, after a transpose if `transpose` is set.
    Split {
        axis: usize,
        targets: Vec<String>,
        transpose: bool,
    },
    Custom(Arc<Custom>),
}

#[derive(Clone)]
struct Rule {
    pattern: Vec<String>,
    action: Action,
}

impl Debug for Rule {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter
            .debug_struct("Rule")
            .field("pattern", &self.pattern.join("."))
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, Default)]
pub struct RuleSet {
    rules: Vec<Rule>,
}

impl RuleSet {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn with(mut self, pattern: &str, action: Action) -> Self {
        self.rules.push(Rule {
            pattern: pattern.split('.').map(str::to_owned).collect(),
            action,
        });
        self
    }

    #[must_use]
    pub fn rename(self, from: &str, to: &str) -> Self {
        self.with(from, Action::Rename(to.to_owned()))
    }

    #[must_use]
    pub fn rename_each(self, from: &str, to: &str, pairs: &[(&str, &str)]) -> Self {
        pairs.iter().fold(self, |rules, (name, target)| {
            rules.rename(&join(from, name), &join(to, target))
        })
    }

    /// Leaves what `from` matches unread, without [`Loader::finish`](crate::format::Loader::finish) refusing it.
    #[must_use]
    pub fn skip(self, from: &str) -> Self {
        self.with(from, Action::Skip)
    }

    /// Renames what `from` matches to `to` with a matrix's two axes swapped, as weights stored `[in, out]` need.
    #[must_use]
    pub fn transpose(self, from: &str, to: &str) -> Self {
        self.with(from, Action::Transpose(to.to_owned()))
    }

    /// Splits one stored tensor evenly along `axis` into `to`, in order, as fused projections are stored.
    #[must_use]
    pub fn split(self, from: &str, axis: usize, to: &[&str]) -> Self {
        self.with(
            from,
            Action::Split {
                axis,
                targets: to.iter().map(|&target| target.to_owned()).collect(),
                transpose: false,
            },
        )
    }

    /// Transposes a fused matrix stored `[in, parts · out]`, then splits it evenly into `to`, in order.
    #[must_use]
    pub fn transpose_split(self, from: &str, to: &[&str]) -> Self {
        self.with(
            from,
            Action::Split {
                axis: 0,
                targets: to.iter().map(|&target| target.to_owned()).collect(),
                transpose: true,
            },
        )
    }

    /// Gives each name `from` matches the targets and sources `map` makes, such as pieces of a packed tensor.
    #[must_use]
    pub fn custom(
        self,
        from: &str,
        map: impl Fn(&Matched<'_>) -> SingeResult<Vec<(String, TensorSource)>> + Send + Sync + 'static,
    ) -> Self {
        self.with(from, Action::Custom(Arc::new(map)))
    }

    /// Imports what `from` matches to `to` through `transform`, such as a reshape to the layout Singe stores it in.
    #[must_use]
    pub fn derive(
        self,
        from: &str,
        to: &str,
        transform: impl Fn(&Tensor) -> SingeResult<Tensor> + Send + Sync + 'static,
    ) -> Self {
        let to = to.to_owned();
        let transform = Arc::new(transform);
        self.custom(from, move |found| {
            let transform = Arc::clone(&transform);
            Ok(vec![(
                found.target(&to),
                TensorSource::Derived(Derivation::new(
                    vec![found.name.to_owned()],
                    move |inputs| transform(&inputs[0]),
                )),
            )])
        })
    }

    /// Folds a weight stored by `torch.nn.utils.weight_norm` under `from` into one weight at `to`: its direction, `weight_v`, scaled to its magnitude, `weight_g`, over every axis but `dim`.
    /// It reads either the names `weight_norm` gives or those its parametrization does.
    #[must_use]
    pub fn weight_norm(self, from: &str, to: &str, dim: usize) -> Self {
        let mut rules = self;
        for (magnitude, direction) in [
            ("weight_g", "weight_v"),
            (
                "parametrizations.weight.original0",
                "parametrizations.weight.original1",
            ),
        ] {
            let to = to.to_owned();
            let direction_name = direction.to_owned();
            rules = rules
                .custom(&format!("{from}.{magnitude}"), move |found| {
                    let prefix = &found.name[..found.name.len() - magnitude.len()];
                    let direction = format!("{prefix}{direction_name}");
                    Ok(vec![(
                        found.target(&to),
                        TensorSource::Derived(Derivation::new(
                            vec![found.name.to_owned(), direction],
                            move |inputs| {
                                let (magnitude, direction) = (&inputs[0], &inputs[1]);
                                let axes = (0..direction.shape().rank())
                                    .filter(|&axis| usize::try_from(axis).ok() != Some(dim))
                                    .collect::<Vec<i64>>();
                                let norm = direction.mul(direction)?.sum(&axes, true)?.sqrt()?;
                                direction.mul(&magnitude.div(&norm)?)
                            },
                        )),
                    )])
                })
                // Read through the magnitude's rule.
                .skip(&format!("{from}.{direction}"));
        }
        rules
    }

    /// Appends `other`'s rules, which match only names these leave.
    #[must_use]
    pub fn then(mut self, other: Self) -> Self {
        self.rules.extend(other.rules);
        self
    }

    /// Plans the names any rule matches, returning the plan and those names; names no rule matches are left for another part.
    pub fn plan<'n>(
        &self,
        names: impl IntoIterator<Item = &'n str>,
    ) -> Result<(LoadPlan, Vec<&'n str>)> {
        let mut plan = LoadPlan::new();
        let mut matched = Vec::new();
        for name in names {
            let loaded = loaded_name(name);
            let Some((rule, found)) = self.rules.iter().find_map(|rule| {
                matches(&rule.pattern, &loaded).map(|found| (rule, Matched { name, ..found }))
            }) else {
                continue;
            };
            matched.push(name);
            for (target, source) in apply(&rule.action, &found)? {
                plan.insert(target, source)?;
            }
        }
        Ok((plan, matched))
    }
}

#[must_use]
pub fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_owned()
    } else {
        format!("{prefix}.{name}")
    }
}

/// Returns `name` as `transformers` loads it in every model, whose old checkpoints call a layer norm's weight and bias `gamma` and `beta`.
fn loaded_name(name: &str) -> Cow<'_, str> {
    if name.contains("LayerNorm.gamma") || name.contains("LayerNorm.beta") {
        Cow::Owned(
            name.replace("LayerNorm.gamma", "LayerNorm.weight")
                .replace("LayerNorm.beta", "LayerNorm.bias"),
        )
    } else {
        Cow::Borrowed(name)
    }
}

fn matches<'a>(pattern: &[String], name: &'a str) -> Option<Matched<'a>> {
    let segments = name.split('.').collect::<Vec<_>>();
    if segments.len() < pattern.len() {
        return None;
    }

    let mut captures = Vec::new();
    for (expected, segment) in pattern.iter().zip(&segments) {
        if expected == "*" {
            captures.push(*segment);
        } else if expected != segment {
            return None;
        }
    }

    let rest = segments[pattern.len()..]
        .iter()
        .fold(String::new(), |mut rest, segment| {
            rest.push('.');
            rest.push_str(segment);
            rest
        });
    Some(Matched {
        name,
        captures,
        rest,
    })
}

fn apply(action: &Action, found: &Matched<'_>) -> SingeResult<Vec<(String, TensorSource)>> {
    let stored = || TensorSource::Stored(found.name.to_owned());
    Ok(match action {
        Action::Rename(to) => vec![(found.target(to), stored())],
        Action::Skip => Vec::new(),
        Action::Transpose(to) => vec![(
            found.target(to),
            TensorSource::Derived(Derivation::new(vec![found.name.to_owned()], |inputs| {
                inputs[0].transpose(0, 1)?.contiguous()
            })),
        )],
        Action::Split {
            axis,
            targets,
            transpose,
        } => {
            let (axis, count, transpose) = (*axis, targets.len(), *transpose);
            targets
                .iter()
                .enumerate()
                .map(|(index, to)| {
                    let derive = move |inputs: &[Tensor]| {
                        let tensor = if transpose {
                            inputs[0].transpose(0, 1)?
                        } else {
                            inputs[0].clone()
                        };
                        part(&tensor, axis, index, count)
                    };
                    (
                        found.target(to),
                        TensorSource::Derived(Derivation::new(vec![found.name.to_owned()], derive)),
                    )
                })
                .collect()
        }
        Action::Custom(map) => map(found)?,
    })
}

/// Returns part `index` of `count` equal parts of `tensor` along `axis`.
fn part(tensor: &Tensor, axis: usize, index: usize, count: usize) -> SingeResult<Tensor> {
    let extent = tensor
        .shape()
        .dimensions()
        .get(axis)
        .copied()
        .ok_or_else(|| SingeError::out_of_range("split axis"))?;
    let (index, count) = (
        i64::try_from(index).map_err(|_| SingeError::out_of_range("split part"))?,
        i64::try_from(count).map_err(|_| SingeError::out_of_range("split parts"))?,
    );
    if extent % count != 0 {
        return Err(SingeError::invalid_argument(format!(
            "an axis of {extent} does not split into {count} parts"
        )));
    }

    let size = extent / count;
    let axis = i64::try_from(axis).map_err(|_| SingeError::out_of_range("split axis"))?;
    tensor.narrow(axis, index * size, size)?.contiguous()
}
