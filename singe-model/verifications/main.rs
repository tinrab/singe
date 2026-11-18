// Runs each model's verification: its cases through Singe's pipelines, compared with the outputs the reference libraries gave on them, each timed.
// `cargo test -p singe-model --features cuda_13_4 --test verifications -- [FILTER]` runs the cases whose `model/case` labels contain `FILTER`, such as `text/bert` or `bert/ranking`; `just verify FILTER` also runs their Python side.
// It fails when a case fails, or when no case ran.

mod audio;
mod diffusion;
mod multimodal;
mod reinforcement_learning;
mod support;
mod tabular;
mod text;
mod time_series;
mod video;
mod vision;

use std::process::ExitCode;

use clap::Parser;

use crate::support::{Session, Verification};

const CATEGORIES: &[&[Verification]] = &[
    text::VERIFICATIONS,
    vision::VERIFICATIONS,
    multimodal::VERIFICATIONS,
    video::VERIFICATIONS,
    audio::VERIFICATIONS,
    time_series::VERIFICATIONS,
    reinforcement_learning::VERIFICATIONS,
    tabular::VERIFICATIONS,
    diffusion::VERIFICATIONS,
];

#[derive(Debug, Parser)]
#[command(name = "verifications", bin_name = "verifications")]
struct Arguments {
    #[arg(
        default_value = "",
        hide_default_value = true,
        help = "Runs only the cases whose `model/case` labels contain this, such as `text/bert` or `bert/ranking`"
    )]
    filter: String,
}

fn main() -> ExitCode {
    let arguments = Arguments::parse();
    let (mut passed, mut failed, mut skipped) = (0, 0, 0);
    for verification in CATEGORIES.iter().copied().flatten() {
        let mut session = match Session::open(verification.name, &arguments.filter) {
            Ok(session) => session,
            Err(error) => {
                println!("{}: FAILED, {error:#}", verification.name);
                failed += 1;
                continue;
            }
        };

        (verification.verify)(&mut session);
        passed += session.passed;
        failed += session.failed;
        skipped += session.skipped;
    }

    println!("{passed} passed, {failed} failed, {skipped} skipped");
    if passed + failed == 0 {
        println!("no case ran");
        return ExitCode::FAILURE;
    }

    if failed == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
