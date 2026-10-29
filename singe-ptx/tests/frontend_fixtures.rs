//! Compiler acceptance, rejection, and source diagnostics.

use std::collections::BTreeSet;

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use singe_ptx::emulator::ParsedModule;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixtures {
    prelude: String,
    declarations: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    name: String,
    body: String,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    error_source: Option<String>,
}

impl Fixtures {
    fn source(&self, case: &Case) -> String {
        format!(
            "{}\n.entry kernel() {{\n{}\n{}\nret;\n}}\n",
            self.prelude, self.declarations, case.body
        )
    }
}

#[test]
fn test_frontend_fixture_expectations() -> Result<()> {
    let fixtures: Fixtures = serde_json::from_str(include_str!("fixtures/frontend/cases.json"))?;
    let mut names = BTreeSet::new();
    for case in &fixtures.cases {
        ensure!(names.insert(&case.name), "duplicate fixture {}", case.name);
        let source = fixtures.source(case);
        let parsed = ParsedModule::parse(&source)
            .with_context(|| format!("fixture {} must parse", case.name))?;
        match (parsed.compile("kernel"), &case.error) {
            (Ok(_), None) => {}
            (Err(error), Some(expected)) => {
                ensure!(
                    error.to_string().contains(expected),
                    "fixture {}: expected {expected:?}, got {error}",
                    case.name
                );
                let span = error.span;
                ensure!(
                    span.start < span.end && source.get(span.start..span.end).is_some(),
                    "fixture {}: invalid diagnostic span {span:?}",
                    case.name
                );
                if let Some(expected) = &case.error_source {
                    ensure!(
                        source[span.start..span.end].contains(expected),
                        "fixture {}: error points at {:?}, expected {expected:?}",
                        case.name,
                        &source[span.start..span.end]
                    );
                }
            }
            (Ok(_), Some(expected)) => {
                anyhow::bail!("fixture {}: expected rejection {expected:?}", case.name)
            }
            (Err(error), None) => {
                anyhow::bail!("fixture {}: expected compilation, got {error}", case.name)
            }
        }
    }
    Ok(())
}

#[cfg(feature = "cuda_tests")]
#[test]
fn test_frontend_expectations_match_nvidia_assembler() -> Result<()> {
    use std::{
        env, fs,
        process::{self, Command},
    };
    let fixtures: Fixtures = serde_json::from_str(include_str!("fixtures/frontend/cases.json"))?;
    let directory = env::temp_dir().join(format!("singe-ptx-assembler-{}", process::id()));
    fs::create_dir_all(&directory)?;
    let source = directory.join("kernel.ptx");
    for case in &fixtures.cases {
        fs::write(&source, fixtures.source(case))?;
        let result = Command::new("ptxas")
            .arg("-arch=sm_80")
            .arg(&source)
            .arg("-o")
            .arg(directory.join("kernel.cubin"))
            .output()
            .context("cuda-tests requires ptxas on PATH")?;
        assert_eq!(
            result.status.success(),
            case.error.is_none(),
            "{}: {}",
            case.name,
            String::from_utf8_lossy(&result.stderr)
        );
    }
    fs::remove_dir_all(directory)?;
    Ok(())
}
