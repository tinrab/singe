//! C++ frontend errata rules for excluding known problematic engine choices.

use std::{env, fs, path::Path};

use serde::{Deserialize, Serialize};
use serde_json::Number;

use crate::{
    backend::{engine::EngineIndex, execution::KnobChoiceInfo, knob::BackendKnobType},
    error::{Error, Result},
};

/// Dimension order used by an errata input-shape pattern.
/// Convolution filter patterns always use NCHW order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrataShapeFormat {
    /// Batch, channel, height, width dimension order.
    #[serde(rename = "NCHW")]
    Nchw,
    /// Batch, height, width, channel dimension order.
    #[serde(rename = "NHWC")]
    Nhwc,
}

/// Four-dimensional errata condition, with `None` entries matching any dimension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrataShapePattern {
    format: ErrataShapeFormat,
    dimensions: [Option<i64>; 4],
}

impl ErrataShapePattern {
    /// Creates a pattern with nonnegative explicit dimensions and `None` wildcards.
    /// The C++ JSON wildcard `-1` is decoded before typed construction.
    ///
    /// # Errors
    /// Returns an error for any negative explicit dimension, including `Some(-1)`.
    pub fn create(format: ErrataShapeFormat, dimensions: [Option<i64>; 4]) -> Result<Self> {
        for (axis, dimension) in dimensions.iter().enumerate() {
            if let Some(dimension) = dimension
                && *dimension < 0
            {
                return Err(Error::ErrataShapeDimensionInvalid {
                    axis,
                    dimension: *dimension,
                });
            }
        }
        Ok(Self { format, dimensions })
    }

    /// Dimension order used by the pattern, independently of tensor strides.
    #[must_use]
    pub fn format(&self) -> ErrataShapeFormat {
        self.format
    }

    /// Pattern dimensions in the declared format, with `None` wildcards.
    #[must_use]
    pub fn dimensions(&self) -> &[Option<i64>; 4] {
        &self.dimensions
    }

    /// Matches logical NCHW dimensions against this pattern.
    /// NHWC patterns are reordered for comparison, and other ranks do not match.
    #[must_use]
    pub fn matches(&self, actual: &[i64]) -> bool {
        let order = match self.format {
            ErrataShapeFormat::Nchw => [0, 1, 2, 3],
            ErrataShapeFormat::Nhwc => [0, 2, 3, 1],
        };
        actual.len() == 4
            && self
                .dimensions
                .iter()
                .zip(order)
                .all(|(expected, axis)| expected.is_none_or(|expected| actual[axis] == expected))
    }
}

/// One engine exclusion, optionally limited by knobs, cuDNN versions and convolution shapes.
#[derive(Debug, Clone)]
pub struct ErrataRule {
    id: Option<String>,
    operation: String,
    engine: EngineIndex,
    knobs: Vec<KnobChoiceInfo>,
    version_start: u64,
    version_end: Option<u64>,
    input_shape: Option<ErrataShapePattern>,
    filter_shape: Option<ErrataShapePattern>,
}

impl ErrataRule {
    /// Creates a rule for a nonempty C++ operation name and a nonnegative engine index.
    ///
    /// # Errors
    /// Returns an error for an empty operation.
    /// [`EngineIndex`] already guarantees a nonnegative index.
    pub fn create(operation: impl Into<String>, engine: EngineIndex) -> Result<Self> {
        let operation = operation.into();
        if operation.is_empty() {
            return Err(Error::ErrataOperationEmpty);
        }
        Ok(Self {
            id: None,
            operation,
            engine,
            knobs: Vec::new(),
            version_start: 0,
            version_end: None,
            input_shape: None,
            filter_shape: None,
        })
    }

    /// Sets the optional diagnostic identifier and returns the updated rule.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Adds a knob value that must be present for this rule to match.
    #[must_use]
    pub fn with_knob(mut self, knob: BackendKnobType, value: i64) -> Self {
        self.knobs.push(KnobChoiceInfo {
            knob_type: knob,
            value,
        });
        self
    }

    /// Replaces all required knob choices; an empty list removes knob conditions.
    #[must_use]
    pub fn with_knobs(mut self, knobs: impl Into<Vec<KnobChoiceInfo>>) -> Self {
        self.knobs = knobs.into();
        self
    }

    /// Limits matching to `[start, end)`. An absent end has no upper bound.
    ///
    /// # Errors
    /// Returns an error when a supplied end is no greater than the start.
    pub fn with_versions(mut self, start: u64, end: Option<u64>) -> Result<Self> {
        if let Some(end) = end
            && end <= start
        {
            return Err(Error::ErrataVersionRangeInvalid { start, end });
        }
        self.version_start = start;
        self.version_end = end;
        Ok(self)
    }

    /// Sets a four-dimensional input pattern. `None` entries are wildcards.
    /// Passing `None` for the complete pattern removes the input-shape condition.
    /// Removing the condition does not require input dimensions when matching plans.
    ///
    /// # Errors
    /// Returns an error for negative explicit dimensions.
    pub fn with_input_shape(
        mut self,
        format: ErrataShapeFormat,
        dimensions: impl Into<Option<[Option<i64>; 4]>>,
    ) -> Result<Self> {
        self.input_shape = dimensions
            .into()
            .map(|dimensions| ErrataShapePattern::create(format, dimensions))
            .transpose()?;
        Ok(self)
    }

    /// Sets a four-dimensional convolution filter pattern in NCHW order.
    /// Passing `None` removes the condition, independently of the input pattern.
    ///
    /// # Errors
    /// Returns an error for negative explicit dimensions.
    pub fn with_filter_shape(
        mut self,
        dimensions: impl Into<Option<[Option<i64>; 4]>>,
    ) -> Result<Self> {
        self.filter_shape = dimensions
            .into()
            .map(|dimensions| ErrataShapePattern::create(ErrataShapeFormat::Nchw, dimensions))
            .transpose()?;
        Ok(self)
    }

    /// Returns the optional diagnostic rule identifier.
    #[must_use]
    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    /// Returns the C++ operation-name prefix required by this rule.
    #[must_use]
    pub fn operation(&self) -> &str {
        &self.operation
    }

    /// Returns the native engine index excluded by this rule.
    #[must_use]
    pub fn engine(&self) -> EngineIndex {
        self.engine
    }

    /// Knob choices that must all be present for the rule to match.
    #[must_use]
    pub fn knobs(&self) -> &[KnobChoiceInfo] {
        &self.knobs
    }

    /// Inclusive cuDNN runtime version lower bound.
    #[must_use]
    pub fn version_start(&self) -> u64 {
        self.version_start
    }

    /// Exclusive cuDNN runtime version upper bound, or `None` when unbounded.
    #[must_use]
    pub fn version_end(&self) -> Option<u64> {
        self.version_end
    }

    /// Optional input-shape condition in its declared NCHW or NHWC format.
    #[must_use]
    pub fn input_shape(&self) -> Option<&ErrataShapePattern> {
        self.input_shape.as_ref()
    }

    /// Optional convolution filter condition, always in logical KCHW order.
    #[must_use]
    pub fn filter_shape(&self) -> Option<&ErrataShapePattern> {
        self.filter_shape.as_ref()
    }

    fn matches(&self, plan: ErrataPlan<'_>) -> bool {
        plan.cudnn_version >= self.version_start
            && self.version_end.is_none_or(|end| plan.cudnn_version < end)
            && self.matches_source(plan.source)
            && self.input_shape.as_ref().is_none_or(|pattern| {
                plan.input_shape
                    .is_some_and(|actual| pattern.matches(actual))
            })
            && self.filter_shape.as_ref().is_none_or(|pattern| {
                plan.filter_shape
                    .is_some_and(|actual| pattern.matches(actual))
            })
    }

    fn matches_source(&self, source: ErrataPlanSource<'_>) -> bool {
        match source {
            ErrataPlanSource::Engine {
                operation,
                engine,
                knobs,
            } => {
                self.operation == operation
                    && self.engine == engine
                    && self.knobs.iter().all(|required| knobs.contains(required))
            }
            ErrataPlanSource::Tag(tag) => {
                let Some(engine_and_knobs) = tag
                    .strip_prefix(self.operation.as_str())
                    .and_then(|suffix| suffix.strip_prefix("_eng"))
                else {
                    return false;
                };
                let (engine, knobs) = engine_and_knobs
                    .split_once('_')
                    .unwrap_or((engine_and_knobs, ""));
                engine
                    .parse::<i64>()
                    .is_ok_and(|engine| engine == self.engine.as_i64())
                    && self.knobs.iter().all(|required| {
                        knobs.split('_').any(|token| {
                            let Some((kind, value)) = token
                                .strip_prefix('k')
                                .and_then(|choice| choice.split_once('='))
                            else {
                                return false;
                            };
                            kind.parse::<u32>()
                                .is_ok_and(|kind| kind == required.knob_type.raw())
                                && value
                                    .parse::<i64>()
                                    .is_ok_and(|value| value == required.value)
                        })
                    })
            }
        }
    }
}

/// Plan properties used to match errata without requiring native descriptor queries.
#[derive(Debug, Clone, Copy)]
pub struct ErrataPlan<'a> {
    source: ErrataPlanSource<'a>,
    /// Version number reported by the loaded cuDNN runtime.
    pub cudnn_version: u64,
    /// Logical `[N, C, H, W]` input dimensions, independently of physical strides.
    /// NHWC rule patterns are converted to this order before comparison.
    pub input_shape: Option<&'a [i64]>,
    /// Logical `[K, C, H, W]` convolution filter dimensions.
    pub filter_shape: Option<&'a [i64]>,
}

#[derive(Debug, Clone, Copy)]
enum ErrataPlanSource<'a> {
    Engine {
        operation: &'a str,
        engine: EngineIndex,
        knobs: &'a [KnobChoiceInfo],
    },
    Tag(&'a str),
}

impl<'a> ErrataPlan<'a> {
    /// Uses cached engine metadata and a C++ operation prefix such as `ConvFwd` or `Matmul`.
    #[must_use]
    pub fn from_engine(
        operation: &'a str,
        engine: EngineIndex,
        knobs: &'a [KnobChoiceInfo],
        cudnn_version: u64,
    ) -> Self {
        Self {
            source: ErrataPlanSource::Engine {
                operation,
                engine,
                knobs,
            },
            cudnn_version,
            input_shape: None,
            filter_shape: None,
        }
    }

    /// Matches a C++ execution-plan tag without querying live descriptors.
    /// Tags use `operation_eng<number>_k<kind>=<value>` with additional knob tokens as needed.
    /// Engines and knobs match whole numeric tokens, including signed knob values.
    /// Missing or malformed tag fields do not match rules that require those fields.
    #[must_use]
    pub fn from_tag(tag: &'a str, cudnn_version: u64) -> Self {
        Self {
            source: ErrataPlanSource::Tag(tag),
            cudnn_version,
            input_shape: None,
            filter_shape: None,
        }
    }

    /// Supplies logical NCHW input dimensions for shape-conditioned rules.
    /// A rule requiring input dimensions does not match when these are absent.
    #[must_use]
    pub fn with_input_shape(mut self, dimensions: &'a [i64]) -> Self {
        self.input_shape = Some(dimensions);
        self
    }

    /// Supplies logical KCHW filter dimensions for shape-conditioned rules.
    #[must_use]
    pub fn with_filter_shape(mut self, dimensions: &'a [i64]) -> Self {
        self.filter_shape = Some(dimensions);
        self
    }
}

/// Parsed engine exclusions from C++ frontend errata JSON or typed Rust rules.
#[derive(Debug, Clone, Default)]
pub struct Errata {
    rules: Vec<ErrataRule>,
}

impl Errata {
    /// Creates an errata set from typed rules, preserving their order.
    pub fn new(rules: impl Into<Vec<ErrataRule>>) -> Self {
        Self {
            rules: rules.into(),
        }
    }

    /// Reads C++ `rules`, `knob`, version-range and convolution-shape fields.
    /// An omitted version end or the C++ `-1` sentinel leaves the range unbounded.
    /// Input and filter shape entries of `-1` are independent wildcards.
    ///
    /// # Errors
    ///
    /// Returns an error if the JSON cannot be decoded or a rule has an empty operation, negative engine index, invalid version range, malformed or unrepresentable knob value, or invalid shape dimensions or format. Rule-validation errors include the zero-based rule index.
    pub fn from_json(json: &str) -> Result<Self> {
        let document: JsonErrata = serde_json::from_str(json)?;
        let rules = document
            .rules
            .into_iter()
            .enumerate()
            .map(|(index, rule)| {
                parse_rule(rule).map_err(|source| Error::ErrataInvalidRule {
                    index,
                    source: Box::new(source),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { rules })
    }

    /// Exports typed or loaded rules in the C++ frontend JSON format.
    /// Preserves rule order, IDs, numeric knobs, version bounds and shape conditions.
    /// Wildcards use `-1`; absent optional conditions and unbounded version ends are omitted.
    ///
    /// # Errors
    /// Returns an error if JSON serialization fails.
    pub fn to_json(&self) -> Result<String> {
        let document = JsonErrata {
            rules: self.rules.iter().map(JsonRule::from).collect(),
        };
        Ok(serde_json::to_string_pretty(&document)?)
    }

    /// Loads `CUDNN_ERRATA_JSON_FILE`, falling back to `path` when the variable is absent.
    /// Returns `None` when neither source is configured.
    /// Configured files that cannot be read or parsed return an error.
    ///
    /// # Errors
    ///
    /// Returns an error if the configured file cannot be read as UTF-8, the JSON cannot be decoded, or a rule fails validation.
    pub fn load_from_config(path: Option<&Path>) -> Result<Option<Self>> {
        let path = env::var_os("CUDNN_ERRATA_JSON_FILE")
            .map(Into::into)
            .or_else(|| path.map(Path::to_path_buf));
        let Some(path) = path else {
            return Ok(None);
        };
        let json =
            fs::read_to_string(&path).map_err(|source| Error::ErrataFileRead { path, source })?;
        Self::from_json(&json).map(Some)
    }

    /// Borrows the ordered engine-exclusion rules.
    #[must_use]
    pub fn rules(&self) -> &[ErrataRule] {
        &self.rules
    }

    /// Returns all matching rules so callers can apply additional runtime conditions.
    /// Predicates on this iterator replace the C++ helper's custom callback.
    /// The iterator borrows the plan metadata, but yielded rules borrow only this errata set.
    pub fn matching_rules<'rules, 'plan>(
        &'rules self,
        plan: ErrataPlan<'plan>,
    ) -> impl Iterator<Item = &'rules ErrataRule> + use<'rules, 'plan> {
        self.rules.iter().filter(move |rule| rule.matches(plan))
    }

    /// Returns whether at least one rule matches the supplied plan metadata.
    #[must_use]
    pub fn is_blocked(&self, plan: ErrataPlan<'_>) -> bool {
        self.matching_rules(plan).next().is_some()
    }
}

#[derive(Serialize, Deserialize)]
struct JsonErrata {
    rules: Vec<JsonRule>,
}

#[derive(Serialize, Deserialize)]
struct JsonRule {
    #[serde(skip_serializing_if = "Option::is_none")]
    rule_id: Option<String>,
    operation: String,
    engine: i64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    knob: Vec<String>,
    #[serde(default)]
    cudnn_version_start: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    cudnn_version_end: Option<Number>,
    #[serde(skip_serializing_if = "Option::is_none")]
    shape_format: Option<ErrataShapeFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_shape: Option<Vec<i64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    filter_shape: Option<Vec<i64>>,
}

impl From<&ErrataRule> for JsonRule {
    fn from(rule: &ErrataRule) -> Self {
        let shape = |pattern: &ErrataShapePattern| {
            pattern
                .dimensions
                .map(|dimension| dimension.unwrap_or(-1))
                .to_vec()
        };
        Self {
            rule_id: rule.id.clone(),
            operation: rule.operation.clone(),
            engine: rule.engine.as_i64(),
            knob: rule
                .knobs
                .iter()
                .map(|choice| format!("k{}={}", choice.knob_type.raw(), choice.value))
                .collect(),
            cudnn_version_start: rule.version_start,
            cudnn_version_end: rule.version_end.map(Into::into),
            shape_format: rule.input_shape.as_ref().map(|pattern| pattern.format),
            input_shape: rule.input_shape.as_ref().map(shape),
            filter_shape: rule.filter_shape.as_ref().map(shape),
        }
    }
}

fn parse_rule(json: JsonRule) -> Result<ErrataRule> {
    let version_end = match json.cudnn_version_end {
        None => None,
        Some(end) if end.as_i64() == Some(-1) => None,
        Some(end) => Some(end.as_u64().ok_or(Error::ErrataRuleSyntax {
            reason: "version end must be nonnegative or -1",
        })?),
    };
    let mut rule = EngineIndex::try_from(json.engine)
        .and_then(|engine| ErrataRule::create(json.operation, engine))
        .and_then(|rule| rule.with_versions(json.cudnn_version_start, version_end))?;
    rule.id = json.rule_id;
    for choice in json.knob {
        let (kind, value) = choice
            .strip_prefix('k')
            .and_then(|text| text.split_once('='))
            .ok_or_else(|| Error::ErrataKnobSyntax {
                choice: choice.clone(),
            })?;
        let kind = kind
            .parse::<u32>()
            .map(BackendKnobType::from)
            .map_err(|source| Error::ErrataKnobNumber {
                choice: choice.clone(),
                source,
            })?;
        let value = value
            .parse::<i64>()
            .map_err(|source| Error::ErrataKnobNumber {
                choice: choice.clone(),
                source,
            })?;
        rule = rule.with_knob(kind, value);
    }
    if let Some(shape) = json.input_shape {
        let format = json.shape_format.ok_or(Error::ErrataRuleSyntax {
            reason: "input_shape requires shape_format",
        })?;
        rule = rule.with_input_shape(format, parse_shape(shape)?)?;
    }
    if let Some(shape) = json.filter_shape {
        rule = rule.with_filter_shape(parse_shape(shape)?)?;
    }
    Ok(rule)
}

fn parse_shape(shape: Vec<i64>) -> Result<[Option<i64>; 4]> {
    let shape: [i64; 4] = shape.try_into().map_err(|_| Error::ErrataRuleSyntax {
        reason: "shape patterns require four dimensions",
    })?;
    Ok(shape.map(|dimension| (dimension != -1).then_some(dimension)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_errata_rule_preserves_validation_error_source() {
        let error =
            Errata::from_json(r#"{"rules":[{"operation":"ConvFwd","engine":-1}]}"#).unwrap_err();
        let Error::ErrataInvalidRule { index, source } = error else {
            panic!("expected an indexed rule error");
        };
        assert_eq!(index, 0);
        assert!(matches!(*source, Error::EngineIndexInvalid { value: -1 }));
    }

    #[test]
    fn test_invalid_errata_knob_preserves_numeric_error_source() {
        use std::error::Error as StdError;

        let error = Errata::from_json(
            r#"{"rules":[{"operation":"ConvFwd","engine":1,"knob":["kbad=2"]}]}"#,
        )
        .unwrap_err();
        let Error::ErrataInvalidRule { source, .. } = error else {
            panic!("expected an indexed rule error");
        };
        assert!(matches!(*source, Error::ErrataKnobNumber { .. }));
        assert!(StdError::source(source.as_ref()).is_some());
    }
}
