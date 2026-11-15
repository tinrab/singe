use std::fmt::Write as _;

use crate::{
    backend::cuda::export::program::{LoopEncoder, LoopTokenizer},
    utility::emit,
};

fn hex(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        return "-".to_owned();
    }

    bytes.iter().fold(String::new(), |mut text, byte| {
        let _ = write!(text, "{byte:02x}");
        text
    })
}

fn ranges(text: &mut String, name: &str, ranges: &[[u32; 2]]) {
    emit!(text, "{name} {}", ranges.len());
    for [start, end] in ranges {
        emit!(text, "{start} {end}");
    }
}

fn encoder(text: &mut String, encoder: &LoopEncoder) {
    let [before, after] = &encoder.template;
    emit!(
        text,
        "template 2\n{}\n{}",
        hex(before.as_bytes()),
        hex(after.as_bytes())
    );
    emit!(text, "placeholder 1\n{}", encoder.placeholder.unwrap_or(-1));
    emit!(text, "added {}", encoder.added.len());
    for id in &encoder.added {
        emit!(text, "{id}");
    }

    emit!(text, "merges {}", encoder.merges.len());
    for [left, right, merged] in &encoder.merges {
        emit!(text, "{left} {right} {merged}");
    }

    let classes = &encoder.classes;
    ranges(text, "letters", &classes.letters);
    ranges(text, "marks", &classes.marks);
    ranges(text, "numbers", &classes.numbers);
    ranges(text, "whitespace", &classes.whitespace);
    emit!(
        text,
        "marks_are_letters 1\n{}",
        u8::from(encoder.marks_are_letters)
    );
    let Some(composition) = &encoder.composition else {
        emit!(text, "composition 0");
        return;
    };

    emit!(text, "composition 1");
    emit!(text, "decompositions {}", composition.decompositions.len());
    for (point, parts) in &composition.decompositions {
        let written = parts
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(" ");
        emit!(text, "{point} {} {written}", parts.len());
    }

    emit!(
        text,
        "combining_classes {}",
        composition.combining_classes.len()
    );
    for [start, end, class] in &composition.combining_classes {
        emit!(text, "{start} {end} {class}");
    }

    emit!(text, "pairs {}", composition.pairs.len());
    for [first, second, composed] in &composition.pairs {
        emit!(text, "{first} {second} {composed}");
    }
}

pub fn tokenizer_file(tokenizer: &LoopTokenizer) -> String {
    let mut text = String::new();
    emit!(text, "tokens {}", tokenizer.tokens.len());
    for token in &tokenizer.tokens {
        emit!(text, "{} {}", u8::from(token.special), hex(&token.bytes));
    }

    if let Some(encoding) = &tokenizer.encoder {
        emit!(text, "encoder 1");
        encoder(&mut text, encoding);
    } else {
        emit!(text, "encoder 0");
    }
    text
}
