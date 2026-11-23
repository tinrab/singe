"""Lists the workspace crates whose version crates.io does not have, and publishes them with --publish."""

import argparse
import json
import ssl
import subprocess
import sys
import urllib.error
import urllib.request

# The index only lists public versions, and `cargo publish` uploads over its own verified connection, so certificates are not checked.
UNVERIFIED = ssl._create_unverified_context()


def published_versions(name):
    name = name.lower()
    if len(name) <= 2:
        path = f"{len(name)}/{name}"
    elif len(name) == 3:
        path = f"3/{name[0]}/{name}"
    else:
        path = f"{name[:2]}/{name[2:4]}/{name}"
    request = urllib.request.Request(
        f"https://index.crates.io/{path}", headers={"User-Agent": "singe-publish"}
    )
    try:
        with urllib.request.urlopen(
            request, timeout=30, context=UNVERIFIED
        ) as response:
            lines = response.read().decode().splitlines()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return set()
        raise
    return {json.loads(line)["vers"] for line in lines if line.strip()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true", help="run cargo publish")
    arguments = parser.parse_args()

    metadata = json.loads(
        subprocess.run(
            ["cargo", "metadata", "--format-version", "1", "--no-deps"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    pending = []
    # `publish = false` is an empty list of registries.
    for package in metadata["packages"]:
        if package["publish"] == []:
            continue
        name, version = package["name"], package["version"]
        if version in published_versions(name):
            print(f"skip {name} {version}")
        else:
            print(f"publish {name} {version}")
            pending.append(name)

    if not pending or not arguments.publish:
        return
    # Cargo publishes the packages in dependency order.
    command = [
        "cargo",
        "publish",
        "--allow-dirty",
        *(f"--package={name}" for name in pending),
    ]
    sys.exit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()
