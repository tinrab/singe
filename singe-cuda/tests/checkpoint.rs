//! End-to-end CUDA checkpoint and CRIU restore test.
//!
//! This test is ignored by default because it needs Linux, CRIU, a checkpoint-capable
//! NVIDIA driver, and permission to dump and restore a child process.
//! CRIU usually needs either `CAP_SYS_ADMIN` or `CAP_CHECKPOINT_RESTORE`. The
//! most direct way to run this test is to run the cargo command under `sudo` or
//! as root in the same development environment.
//!
//! Compile the test as the current user first:
//!
//! ```bash
//! cargo test -p singe-cuda --features testing --test checkpoint --no-run
//! ```
//!
//! Then run the compiled test binary with sudo.
//! This gives CRIU the privileges it needs:
//!
//! ```bash
//! CHECKPOINT_TEST="$(find target/debug/deps -maxdepth 1 -type f -executable -name 'checkpoint-*' | head -n1)"
//! sudo env \
//!   PATH="$PATH" \
//!   CUDA_PATH="$CUDA_PATH" \
//!   LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
//!   RUST_TEST_THREADS=1 \
//!   "$CHECKPOINT_TEST" checkpoint_process_e2e --ignored --exact --nocapture
//! ```
//!
//! On systems where `criu` is a wrapper, `SINGE_CUDA_CRIU=/path/to/criu` can point the test at the real CRIU binary.

#![cfg(target_os = "linux")]

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use singe_cuda::checkpoint::{CheckpointProcess, LockOptions, ProcessState};

#[test]
#[ignore = "requires Linux, CRIU, a checkpoint-capable CUDA driver, and permission to checkpoint and restore a child process"]
fn checkpoint_process_e2e() {
    run(criu().arg("check"));

    let test_dir =
        env::temp_dir().join(format!("singe-cuda-checkpoint-e2e-{}", std::process::id()));
    let images_dir = test_dir.join("images");
    fs::create_dir_all(&images_dir).unwrap();

    let child_bin = build_child(&test_dir);
    let child = Command::new(&child_bin)
        .arg(&test_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut child = ChildGuard::new(child);
    let pid = child.pid();

    wait_for_file(&test_dir.join("ready"));
    assert_eq!(request_counter(&test_dir, 1), 101);

    let process = CheckpointProcess::from_pid(pid);
    assert_eq!(process.state().unwrap(), ProcessState::Running);

    process
        .lock(LockOptions::new().with_timeout(Duration::from_secs(10)))
        .unwrap();
    assert_eq!(process.state().unwrap(), ProcessState::Locked);

    process.checkpoint().unwrap();
    assert_eq!(process.state().unwrap(), ProcessState::Checkpointed);

    run(criu()
        .arg("dump")
        .arg("--shell-job")
        .arg("--images-dir")
        .arg(&images_dir)
        .arg("--tree")
        .arg(pid.to_string()));
    let status = child.wait().unwrap();
    assert!(
        !status.success(),
        "criu dump should terminate the original child, got {status}"
    );

    run(criu()
        .arg("restore")
        .arg("--shell-job")
        .arg("--restore-detached")
        .arg("--images-dir")
        .arg(&images_dir));
    wait_for_checkpoint_state(process, ProcessState::Checkpointed);

    process.restore(&[]).unwrap();
    assert_eq!(process.state().unwrap(), ProcessState::Locked);

    process.unlock().unwrap();
    assert_eq!(process.state().unwrap(), ProcessState::Running);

    assert_eq!(request_counter(&test_dir, 2), 102);
    send_release(&test_dir);
    wait_for_process_exit(pid);
    child.disarm();

    let _ = fs::remove_dir_all(test_dir);
}

struct ChildGuard {
    pid: i32,
    child: Option<std::process::Child>,
    disarmed: bool,
}

impl ChildGuard {
    fn new(child: std::process::Child) -> Self {
        Self {
            pid: child.id() as i32,
            child: Some(child),
            disarmed: false,
        }
    }

    fn pid(&self) -> i32 {
        self.pid
    }

    fn wait(&mut self) -> std::io::Result<std::process::ExitStatus> {
        self.child.as_mut().unwrap().wait()
    }

    fn disarm(&mut self) {
        self.disarmed = true;
        self.child = None;
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if self.disarmed {
            return;
        }

        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }

        let _ = Command::new("kill")
            .arg("-TERM")
            .arg(self.pid.to_string())
            .status();
        thread::sleep(Duration::from_millis(100));
        let _ = Command::new("kill")
            .arg("-KILL")
            .arg(self.pid.to_string())
            .status();
    }
}

fn build_child(test_dir: &Path) -> PathBuf {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/checkpoint_child.c");
    let binary = test_dir.join("checkpoint_child");
    let cuda_path = env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".to_string());

    run(
        Command::new(env::var_os("CC").unwrap_or_else(|| "cc".into()))
            .arg(&source)
            .arg("-o")
            .arg(&binary)
            .arg(format!("-I{cuda_path}/include"))
            .arg(cuda_driver_library_search_arg())
            .arg("-lcuda"),
    );

    binary
}

fn cuda_driver_library_search_arg() -> String {
    let library_path = singe_cuda_find::find_cuda_driver_path()
        .unwrap()
        .expect("CUDA driver library was not found");
    format!("-L{}", library_path.display())
}

fn criu() -> Command {
    Command::new(env::var_os("SINGE_CUDA_CRIU").unwrap_or_else(|| "criu".into()))
}

fn run(command: &mut Command) {
    let output = command.output().unwrap_or_else(|error| {
        panic!("failed to run {command:?}: {error}");
    });

    assert!(
        output.status.success(),
        "command failed: {command:?}\nstatus: {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

fn request_counter(test_dir: &Path, sequence: usize) -> i32 {
    fs::write(test_dir.join(format!("request-{sequence}")), b"next").unwrap();
    let response = test_dir.join(format!("response-{sequence}"));
    wait_for_file(&response);
    fs::read_to_string(response)
        .unwrap()
        .trim()
        .parse()
        .unwrap()
}

fn send_release(test_dir: &Path) {
    fs::write(test_dir.join("release"), b"").unwrap();
}

fn wait_for_checkpoint_state(process: CheckpointProcess, expected: ProcessState) {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match process.state() {
            Ok(state) if state == expected => return,
            _ if Instant::now() < deadline => thread::sleep(Duration::from_millis(25)),
            result => panic!("timed out waiting for {expected}; last state result: {result:?}"),
        }
    }
}

fn wait_for_file(path: &Path) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while !path.exists() {
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {}",
            path.display()
        );
        thread::sleep(Duration::from_millis(25));
    }
}

fn wait_for_process_exit(pid: i32) {
    let proc_path = PathBuf::from(format!("/proc/{pid}"));
    let deadline = Instant::now() + Duration::from_secs(30);
    while proc_path.exists() {
        assert!(
            Instant::now() < deadline,
            "timed out waiting for restored process {pid} to exit"
        );
        thread::sleep(Duration::from_millis(25));
    }
}
