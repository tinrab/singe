//! Source context for parser, compiler, and execution errors.

use std::fmt::{self, Debug, Display, Formatter};

use crate::ast::Span;

/// Displays an error followed by the first source line its span covers, with a marker
/// under the span.
///
/// Columns count Unicode characters, while source spans retain byte offsets.
/// Tabs are preserved in the marker's indentation.
/// Errors without a source span, or spans invalid for the supplied source, retain
/// their ordinary display without a source excerpt.
#[derive(Clone, Copy)]
pub struct SourceDiagnostic<'a> {
    error: &'a (dyn Display + Send + Sync),
    source: &'a str,
    span: Option<Span>,
}

impl<'a> SourceDiagnostic<'a> {
    pub(crate) fn new(
        error: &'a (dyn Display + Send + Sync),
        source: &'a str,
        span: Option<Span>,
    ) -> Self {
        Self {
            error,
            source,
            span,
        }
    }
}

impl Debug for SourceDiagnostic<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("SourceDiagnostic")
            .field("span", &self.span)
            .finish_non_exhaustive()
    }
}

impl Display for SourceDiagnostic<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self.error, f)?;
        let Some(span) = self.span else {
            return Ok(());
        };
        if self.source.get(span.start..span.end).is_none() {
            return Ok(());
        }
        let prefix = &self.source[..span.start];
        let line = prefix.bytes().filter(|byte| *byte == b'\n').count() + 1;
        let start = prefix.rfind('\n').map_or(0, |index| index + 1);
        let end = self.source[span.start..]
            .find('\n')
            .map_or(self.source.len(), |index| span.start + index);
        let column = self.source[start..span.start].chars().count() + 1;
        let text = self.source[start..end].trim_end_matches('\r');
        let marker_len = self.source[span.start..span.end.min(end)]
            .chars()
            .count()
            .max(1);
        let width = line.to_string().len();
        write!(
            f,
            "\n --> {line}:{column}\n{line:>width$} | {text}\n{:width$} | ",
            ""
        )?;
        for character in self.source[start..span.start].chars() {
            f.write_str(if character == '\t' { "\t" } else { " " })?;
        }
        for _ in 0..marker_len {
            f.write_str("^")?;
        }
        Ok(())
    }
}
