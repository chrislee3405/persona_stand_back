"""Publish each commit once. Reruns reuse it; registry errors fail closed."""
import json
import os
import re
from pathlib import Path
import subprocess


def run(*args):
    return subprocess.check_output(args, text=True).strip()


def inspect_optional(tag):
    probe = subprocess.run(["skopeo", "inspect", "docker://" + tag], text=True, capture_output=True)
    if probe.returncode == 0:
        return json.loads(probe.stdout)
    if any(code in probe.stderr.lower() for code in ["manifest unknown", "name unknown"]):
        return None
    raise RuntimeError("Cannot safely determine whether the release image exists: " + probe.stderr)


def validate_labels(info, repository, revision, version):
    labels = info.get("Labels") or {}
    if (labels.get("org.opencontainers.image.revision") != revision
            or labels.get("org.opencontainers.image.source") != "https://github.com/" + repository
            or labels.get("org.opencontainers.image.version") != version
            or not re.fullmatch(r"sha256:[a-f0-9]{64}", info.get("Digest", ""))):
        raise RuntimeError("Image source/revision/version mismatch; never reuse a release marker for another build")


def publish():
    version = Path("RELEASE_VERSION").read_text().strip()
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:-rc\.[1-9][0-9]*)?", version):
        raise RuntimeError("RELEASE_VERSION must be X.Y.Z or X.Y.Z-rc.N")
    repository = os.environ["GITHUB_REPOSITORY"]
    revision = os.environ["GITHUB_SHA"]
    image = "ghcr.io/" + repository.lower()
    tag = image + ":sha-" + revision
    release_tag = image + ":release-" + version
    existing_release = inspect_optional(release_tag)
    if existing_release:
        validate_labels(existing_release, repository, revision, version)
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
            "--label", "org.opencontainers.image.version=" + version,
            "--label", "org.opencontainers.image.revision=" + revision,
            "--label", "org.opencontainers.image.source=https://github.com/" + repository,
            "-t", tag, "."], check=True)
        subprocess.run(["docker", "push", tag], check=True)
    info = json.loads(run("skopeo", "inspect", "docker://" + tag))
    validate_labels(info, repository, revision, version)
    if existing_release and existing_release["Digest"] != info["Digest"]:
        raise RuntimeError("Release marker already belongs to another digest; increment RELEASE_VERSION in all three repositories")
    if not existing_release:
        subprocess.run(["skopeo", "copy", "--all", "--preserve-digests",
                        "docker://" + image + "@" + info["Digest"], "docker://" + release_tag], check=True)
    published = json.loads(run("skopeo", "inspect", "docker://" + release_tag))
    validate_labels(published, repository, revision, version)
    if published["Digest"] != info["Digest"]:
        raise RuntimeError("Release tag does not match the commit image")
    receipt = {"releaseVersion": version, "releaseTag": release_tag, "image": image + "@" + info["Digest"], "revision": revision,
               "source": "https://github.com/" + repository,
               "publicationRun": os.environ["GITHUB_SERVER_URL"] + "/" + repository + "/actions/runs/" + os.environ["GITHUB_RUN_ID"]}
    Path("image.json").write_text(json.dumps(receipt, indent=2) + "\n")
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
        summary.write("### Published commit image\n```json\n" + json.dumps(receipt, indent=2) + "\n```\n")


if __name__ == "__main__":
    publish()
