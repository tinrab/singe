//! Public error rendering, including Unicode columns and end-of-input spans.

use anyhow::Result;
use singe_ptx::{
    ast::Span,
    emulator::{
        Emulator, EmulatorOptions, ExecutionError, LaunchConfig, ParsedModule, ShutdownMode,
    },
    parser::{ParseError, ParseErrorKind},
};

#[test]
fn test_parse_diagnostics_render_unicode_tabs_eof_and_multiline_spans() {
    for (source, span, excerpt) in [
        (
            "\tλ bad",
            Span::new(4, 7),
            "\n --> 1:4\n1 | \tλ bad\n  | \t  ^^^",
        ),
        ("λ\r\nbad", Span::new(4, 7), "\n --> 2:1\n2 | bad\n  | ^^^"),
        ("λ\n", Span::new(3, 3), "\n --> 2:1\n2 | \n  | ^"),
        (
            "first\nsecond",
            Span::new(2, 10),
            "\n --> 1:3\n1 | first\n  |   ^^^",
        ),
    ] {
        let error = ParseError {
            kind: ParseErrorKind::UnexpectedEof,
            span,
        };
        assert_eq!(
            error.with_source(source).to_string(),
            format!("{error}{excerpt}")
        );
    }
    let error = ParsedModule::parse(".version 9.4 .target sm_80 .entry k() {").unwrap_err();
    let source = ".version 9.4 .target sm_80 .entry k() {";
    assert_eq!(error.span, Span::new(source.len(), source.len()));
    assert!(error.with_source(source).to_string().ends_with('^'));
}

#[test]
fn test_compiler_and_runtime_errors_render_their_real_source_spans() -> Result<()> {
    let source = ".version 9.4\n.target sm_80\n.entry k() { /* λ */ unsupported; }";
    let parsed = ParsedModule::parse(source)?;
    let error = parsed.compile("k").unwrap_err();
    let diagnostic = error.with_source(parsed.source()).to_string();
    assert!(
        diagnostic.contains("\n --> 3:22\n3 | .entry k() { /* λ */ unsupported; }"),
        "{diagnostic}"
    );
    assert!(diagnostic.ends_with("^^^^^^^^^^^^"), "{diagnostic}");

    let source = ".version 9.4\n.target sm_80\n.entry k() { /* λ */ trap; }";
    let device = Emulator::create(EmulatorOptions::default())?;
    let parsed = ParsedModule::parse(source)?;
    let module = device.load(parsed)?.wait()?;
    let error = module
        .kernel("k")?
        .launch(LaunchConfig::create([1; 3], [1; 3], 0)?, [])?
        .wait()
        .unwrap_err();
    let diagnostic = error.with_source(source).to_string();
    assert!(
        diagnostic.contains("\n --> 3:22\n3 | .entry k() { /* λ */ trap; }"),
        "{diagnostic}"
    );
    assert!(diagnostic.ends_with("^^^^^"), "{diagnostic}");
    assert!(device.shutdown(ShutdownMode::Drain).wait().is_err());
    Ok(())
}

#[test]
fn test_missing_context_and_invalid_utf8_boundaries_keep_the_original_error() {
    let error = ParseError {
        kind: ParseErrorKind::UnexpectedEof,
        span: Span::new(1, 2),
    };
    assert_eq!(error.with_source("λ").to_string(), error.to_string());
    assert_eq!(error.with_source("").to_string(), error.to_string());
    assert_eq!(
        ExecutionError::Cancelled
            .with_source("any source")
            .to_string(),
        "command cancelled"
    );
}
