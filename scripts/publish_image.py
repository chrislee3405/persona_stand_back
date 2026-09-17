"""Publish each commit once. Reruns reuse it; registry errors fail closed."""
import json
import os
from pathlib import Path
import subprocess


def run(*args):
    return subprocess.check_output(args, text=True).strip()


def publish():
    repository = os.environ["GITHUB_REPOSITORY"]
    revision = os.environ["GITHUB_SHA"]
    image = "ghcr.io/" + repository.lower()
    tag = image + ":sha-" + revision
    # Only the registry's explicit missing-manifest/name response permits a build.
    probe = subprocess.run(["skopeo", "inspect", "docker://" + tag], text=True, capture_output=True)
    if probe.returncode:
        if not any(code in probe.stderr.lower() for code in ["manifest unknown", "name unknown"]):
            raise RuntimeError("Cannot safely determine whether the commit image exists: " + probe.stderr)
        if repository.endswith("_front"):
            cdn = os.environ.get("VITE_CDN_BASE", "")
            if not cdn.startswith("https://") or "\n" in cdn or "\r" in cdn:
                raise RuntimeError("Set VITE_CDN_BASE to the production HTTPS CDN URL")
            Path(".env").write_text("VITE_CDN_BASE=" + cdn + "\n")
        subprocess.run(["docker", "build", "--platform", "linux/amd64",
            "--label", "org.opencontainers.image.revision=" + revision,
            "--label", "org.opencontainers.image.source=https://github.com/" + repository,
            "-t", tag, "."], check=True)
        subprocess.run(["docker", "push", tag], check=True)
    info = json.loads(run("skopeo", "inspect", "docker://" + tag))
    labels = info.get("Labels") or {}
    if labels.get("org.opencontainers.image.revision") != revision or labels.get("org.opencontainers.image.source") != "https://github.com/" + repository:
        raise RuntimeError("Existing commit image does not match its source; refusing to overwrite")
    receipt = {"image": image + "@" + info["Digest"], "revision": revision,
               "source": "https://github.com/" + repository,
               "publicationRun": os.environ["GITHUB_SERVER_URL"] + "/" + repository + "/actions/runs/" + os.environ["GITHUB_RUN_ID"]}
    Path("image.json").write_text(json.dumps(receipt, indent=2) + "\n")
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
        summary.write("### Published commit image\n```json\n" + json.dumps(receipt, indent=2) + "\n```\n")


if __name__ == "__main__":
    publish()
