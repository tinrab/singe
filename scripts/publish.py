import argparse
import graphlib
import json
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

HEADER = "dist/katex_header.html"
# Files cargo writes into every package rather than taking from the work tree.
GENERATED = {"Cargo.lock", "Cargo.toml", "Cargo.toml.orig", ".cargo_vcs_info.json"}
INDEX = "https://index.crates.io"
USER_AGENT = "singe-publish (https://github.com/tinrab/singe)"
VERSION = re.compile(
    r"^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?$"
)


def precedence(version):
    """A sort key that orders versions by Semantic Versioning precedence, where a pre-release comes before its release."""
    match = VERSION.match(version)
    if match is None:
        raise ValueError(f"{version} is not a semantic version")
    major, minor, patch, prerelease = match.groups()
    release = (int(major), int(minor), int(patch))
    if prerelease is None:
        return (release, 1, ())
    # Numeric identifiers sort below alphanumeric ones, and numerically among themselves.
    identifiers = tuple(
        (0, int(part), "") if part.isdigit() else (1, 0, part)
        for part in prerelease.split(".")
    )
    return (release, 0, identifiers)


def index_path(name):
    """The crate's path in the sparse index, which crates.io serves as static files."""
    name = name.lower()
    if len(name) <= 2:
        return f"{len(name)}/{name}"
    if len(name) == 3:
        return f"3/{name[0]}/{name}"
    return f"{name[:2]}/{name[2:4]}/{name}"


def published_versions(name):
    """Every version of the crate on crates.io, yanked ones included, since crates.io never takes a version twice."""
    request = urllib.request.Request(
        f"{INDEX}/{index_path(name)}", headers={"User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return []
        raise
    return [json.loads(line)["vers"] for line in body.splitlines() if line.strip()]


def workspace():
    """The workspace root, and every workspace package from `cargo metadata`, by name."""
    metadata = json.loads(
        subprocess.run(
            ["cargo", "metadata", "--format-version", "1", "--no-deps"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    return Path(metadata["workspace_root"]), {
        package["name"]: package for package in metadata["packages"]
    }


def crate_directory(package):
    return Path(package["manifest_path"]).parent


def copy_headers(root, packages):
    """Copies the repository's KaTeX header into each crate whose manifest names it, as `just prepare` does."""
    for package in packages.values():
        directory = crate_directory(package)
        if HEADER in (directory / "Cargo.toml").read_text():
            (directory / HEADER).parent.mkdir(exist_ok=True)
            shutil.copyfile(root / HEADER, directory / HEADER)


def unpublishable_files(root, packages, names):
    """Why the work tree cannot be published as it is: changes git has not committed, or files a crate packages that git does not track."""
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        return [f"uncommitted: {line}" for line in status.splitlines()]
    tracked = set(
        subprocess.run(
            ["git", "ls-files"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.splitlines()
    )
    problems = []
    for name in names:
        package = packages[name]
        directory = crate_directory(package).relative_to(root)
        # A workspace `license-file` is copied into the package under its own name.
        allowed = (
            GENERATED
            | {HEADER}
            | (
                {Path(package["license_file"]).name}
                if package.get("license_file")
                else set()
            )
        )
        listing = subprocess.run(
            ["cargo", "package", "--list", "--allow-dirty", "--package", name],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        problems += [
            f"untracked: {name} packages {directory / path}"
            for path in listing.splitlines()
            if path not in allowed and str(directory / path) not in tracked
        ]
    return problems


def publish_order(packages):
    """Every workspace crate, each after the workspace crates its published manifest names.

    That is its normal and build dependencies, and the dev-dependencies with a version, which crates.io must already have.
    A path-only dev-dependency is stripped when a crate is packaged, so it does not order anything, which also breaks the cycle a crate and its test helpers form.
    """
    graph = {
        name: {
            dependency["name"]
            for dependency in package["dependencies"]
            if dependency["name"] in packages
            and (dependency["kind"] != "dev" or dependency["req"] != "*")
        }
        for name, package in packages.items()
    }
    return list(graphlib.TopologicalSorter(graph).static_order())


def publish(names, dry_run):
    """Publishes `names` in order, one `cargo publish` each, so each waits for crates.io to have the crates it needs.

    First, one dry run packages and builds them all, resolving the unpublished ones from the workspace.
    It stops on any crate that would not build from its package before a single crate is uploaded, and the uploads then skip the build it made of the same files.
    """
    command = [
        "cargo",
        "publish",
        "--locked",
        "--allow-dirty",
        "--dry-run",
        *(argument for name in names for argument in ("--package", name)),
    ]
    print(" ".join(command), flush=True)
    code = subprocess.run(command, check=False).returncode
    if code != 0:
        print("the dry run failed, so nothing was uploaded", file=sys.stderr)
        return code
    if dry_run:
        return 0
    for name in names:
        command = [
            "cargo",
            "publish",
            "--locked",
            "--allow-dirty",
            "--no-verify",
            "--package",
            name,
        ]
        print(" ".join(command), flush=True)
        code = subprocess.run(command, check=False).returncode
        if code != 0:
            print(f"stopped at {name}; run again to resume", file=sys.stderr)
            return code
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="NAME",
        help="a crate never to publish; repeat for several",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="package and verify without uploading"
    )
    arguments = parser.parse_args()

    root, packages = workspace()
    # `publish = false` is an empty list of registries; no `publish` key is null.
    publishable = {
        name for name, package in packages.items() if package["publish"] != []
    }
    unknown = sorted(set(arguments.exclude) - publishable)
    if unknown:
        parser.error(f"no publishable workspace crate is named {', '.join(unknown)}")

    pending, behind = [], []
    for name in publish_order(packages):
        if name not in publishable:
            continue
        if name in arguments.exclude:
            print(f"exclude {name}")
            continue
        version = packages[name]["version"]
        versions = published_versions(name)
        newest = max(versions, key=precedence, default=None)
        if newest is None or precedence(version) > precedence(newest):
            print(f"publish {name} {version} (crates.io has {newest or 'none'})")
            pending.append(name)
        elif version in versions or precedence(version) == precedence(newest):
            print(f"skip {name} {version} (already published)")
        else:
            print(f"behind {name} {version} (crates.io has {newest})")
            behind.append(name)

    if behind:
        sys.exit(
            f"raise the version of {', '.join(behind)} past crates.io's newest before publishing"
        )
    if not pending:
        print("nothing to publish")
        return

    copy_headers(root, packages)
    problems = unpublishable_files(root, packages, pending)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        sys.exit("commit or remove these first; nothing was uploaded")
    sys.exit(publish(pending, arguments.dry_run))


if __name__ == "__main__":
    main()
