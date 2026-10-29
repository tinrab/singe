//! Run stack-overflow regressions out of process so aborts and hangs fail the test.
use singe_ptx::{emulator::ParsedModule, parser::ParseErrorKind};
use std::{
    env,
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};
const CHILD_CASE: &str = "SINGE_PTX_DEPTH_CASE";
const LIMIT: usize = 128;
fn module(body: &str) -> String {
    format!(".version 9.4\n.target sm_80\n.address_size 64\n.visible .entry k() {{ {body} }}")
}
fn initializer(expression: &str, dimensions: usize) -> String {
    format!(
        ".version 9.4\n.target sm_80\n.address_size 64\n.global .u32 value{} = {expression};\n.visible .entry k() {{ ret; }}",
        "[1]".repeat(dimensions)
    )
}
fn wrapped(left: &str, right: &str, count: usize) -> String {
    format!("{}1{}", left.repeat(count), right.repeat(count))
}
fn check(source: &str, accepted: bool, executable: bool) {
    let result = ParsedModule::parse(source);
    if accepted {
        let parsed = result.expect("input within the limit must parse");
        let cloned = parsed.clone();
        let syntax = parsed.syntax().clone();
        drop(syntax);
        if executable {
            cloned.compile("k").expect("ordinary input must compile");
        }
        drop((parsed, cloned));
    } else {
        let error = result.expect_err("excessive depth must be rejected");
        assert!(
            matches!(error.kind, ParseErrorKind::DepthLimitExceeded),
            "{error:?}"
        );
        assert!(error.span.start < error.span.end);
        assert!(error.span.end <= source.len());
    }
}
fn check_expressions() {
    for (left, right, executable) in [
        ("(", ")", true),
        ("-", "", true),
        ("~", "", true),
        ("!", "", true),
        ("+", "", true),
        ("(.u64)", "", true),
        ("generic(", ")", false),
        ("0xff(", ")", false),
    ] {
        for count in [64, LIMIT, LIMIT + 1, 2000] {
            check(
                &initializer(&wrapped(left, right, count), 0),
                count <= LIMIT,
                executable,
            );
        }
    }
}

fn check_operands() {
    for (left, right) in [("(", ")"), ("{", "}")] {
        for count in [64, LIMIT, LIMIT + 1, 2000] {
            check(
                &module(&format!(
                    "mov.u32 %r, {}; ret;",
                    wrapped(left, right, count)
                )),
                count <= LIMIT,
                false,
            );
        }
    }
    for count in [LIMIT - 1, LIMIT] {
        let tuple = wrapped("(", ")", count);
        for operand in [format!("!{tuple}"), format!("{tuple}|1")] {
            check(
                &module(&format!("mov.u32 %r, {operand}; ret;")),
                count < LIMIT,
                false,
            );
        }
    }
    check(
        &module(&format!(
            "mov.u32 %r, {}{}; ret;",
            "(".repeat(129),
            ")".repeat(129)
        )),
        true,
        false,
    );
}

fn run_case(case: &str) {
    match case {
        "blocks" => {
            for count in [0, 64, LIMIT, LIMIT + 1, 2000] {
                check(
                    &module(&format!("{}ret;{}", "{".repeat(count), "}".repeat(count))),
                    count <= LIMIT,
                    true,
                );
            }
        }
        "expressions" => check_expressions(),
        "lists" => {
            for count in [64, LIMIT, LIMIT + 1, 2000] {
                check(
                    &initializer(&wrapped("{", "}", count), count),
                    count <= LIMIT,
                    true,
                );
            }
            check(
                &initializer(&format!("{}{}", "{".repeat(129), "}".repeat(129)), 0),
                true,
                false,
            );
        }
        "operands" => check_operands(),
        "binary" => {
            for operators in [64, LIMIT, LIMIT + 1, 3999] {
                check(
                    &initializer(&vec!["1"; operators + 1].join("+"), 0),
                    operators <= LIMIT,
                    true,
                );
            }
            let deep = vec!["1"; LIMIT + 1].join("+");
            check(&initializer(&format!("1+({deep})"), 0), false, true);
        }
        "conditional" => {
            for count in [64, LIMIT, LIMIT + 1, 2000] {
                check(
                    &initializer(&format!("{}1", "1?1:".repeat(count)), 0),
                    count <= LIMIT,
                    true,
                );
            }
        }
        "mixed" => {
            for count in [64, 65] {
                let expression = wrapped("(", ")", count);
                let prefix = "{".repeat(64);
                let suffix = "}".repeat(64);
                check(
                    &module(&format!("{prefix}mov.u32 %r, {expression}; ret;{suffix}")),
                    count == 64,
                    false,
                );
                check(
                    &module(&format!(
                        "{prefix}.local .u32 v = {expression}; ret;{suffix}"
                    )),
                    count == 64,
                    false,
                );
            }
        }
        _ => panic!("unknown parser depth case"),
    }
}
fn subprocess(case: &str) {
    let mut child = Command::new(env::current_exe().expect("test executable"))
        .args(["--exact", "test_parser_depth_child", "--nocapture"])
        .env(CHILD_CASE, case)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn regression subprocess");
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if child.try_wait().expect("poll subprocess").is_some() {
            let output = child.wait_with_output().expect("collect output");
            assert!(
                output.status.success(),
                "{case}: {}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        if Instant::now() >= deadline {
            child.kill().expect("stop timed-out subprocess");
            child.wait().expect("reap subprocess");
            panic!("{case}: subprocess exceeded 30 seconds");
        }
        thread::sleep(Duration::from_millis(10));
    }
}
#[test]
fn test_parser_depth_child() {
    let Ok(case) = env::var(CHILD_CASE) else {
        return;
    };
    thread::Builder::new()
        .stack_size(2 * 1024 * 1024)
        .spawn(move || run_case(&case))
        .expect("spawn bounded-stack thread")
        .join()
        .expect("parser regression thread panicked");
}
#[test]
fn test_rejects_excessive_depth_without_aborting() {
    for case in [
        "blocks",
        "expressions",
        "lists",
        "operands",
        "binary",
        "conditional",
        "mixed",
    ] {
        subprocess(case);
    }
}
