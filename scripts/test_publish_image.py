import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch, mock_open
import publish_image


def info(revision="a" * 40, version="1.0.3-rc.1", digest="b" * 64):
    return {"Digest": "sha256:" + digest, "Labels": {
        "org.opencontainers.image.revision": revision,
        "org.opencontainers.image.source": "https://github.com/example/persona_stand_back",
        "org.opencontainers.image.version": version}}


class PublishOnce(unittest.TestCase):
    def exercise(self, marker=None, commit=None, error=None, published=None, version="1.0.3-rc.1"):
        calls = []
        def execute(args, **kwargs):
            calls.append(args)
            if args[:2] == ["skopeo", "inspect"]:
                if error:
                    return subprocess.CompletedProcess(args, 1, "", error)
                value = marker if ":release-" in args[-1] else commit
                return subprocess.CompletedProcess(args, 0, json.dumps(value), "") if value else subprocess.CompletedProcess(args, 1, "", "manifest unknown")
            return subprocess.CompletedProcess(args, 0)
        env = {"GITHUB_REPOSITORY": "example/persona_stand_back", "GITHUB_SHA": "a" * 40,
               "GITHUB_SERVER_URL": "https://github.com", "GITHUB_RUN_ID": "123", "GITHUB_STEP_SUMMARY": "summary"}
        with patch("builtins.open", mock_open()), patch.dict(publish_image.os.environ, env), patch.object(Path, "read_text", return_value=version), patch.object(Path, "write_text") as write, patch.object(publish_image.subprocess, "run", side_effect=execute), patch.object(publish_image, "run", side_effect=[json.dumps(commit or info()), json.dumps(published or info())]):
            publish_image.publish()
        receipt = json.loads(write.call_args.args[0])
        self.assertEqual(receipt["releaseVersion"], version)
        return calls

    def test_rerun_reuses_same_commit_and_marker_without_mutation(self):
        self.assertTrue(all(c[:2] == ["skopeo", "inspect"] for c in self.exercise(info(), info())))

    def test_new_candidate_builds_and_copies_exact_digest(self):
        calls = self.exercise()
        self.assertEqual([c[:2] for c in calls], [["skopeo", "inspect"], ["skopeo", "inspect"], ["docker", "build"], ["docker", "push"], ["skopeo", "copy"]])
        self.assertIn("org.opencontainers.image.version=1.0.3-rc.1", calls[2])
        self.assertIn("--preserve-digests", calls[-1])
        self.assertIn("@sha256:" + "b" * 64, calls[-1][-2])

    def test_existing_commit_can_complete_missing_marker(self):
        calls = self.exercise(commit=info())
        self.assertEqual(calls[-1][:2], ["skopeo", "copy"])
        self.assertFalse(any(c[0] == "docker" for c in calls))

    def test_marker_cannot_be_reused_for_another_commit(self):
        with self.assertRaisesRegex(RuntimeError, "mismatch"):
            self.exercise(marker=info(revision="c" * 40))

    def test_marker_cannot_be_reused_for_another_digest(self):
        with self.assertRaisesRegex(RuntimeError, "another digest"):
            self.exercise(marker=info(digest="c" * 64), commit=info())

    def test_auth_and_network_errors_fail_closed(self):
        for message in ["unauthorized", "connection timed out", "permission denied"]:
            with self.assertRaisesRegex(RuntimeError, "Cannot safely determine"):
                self.exercise(error=message)

    def test_incorrect_published_version_or_digest_is_rejected(self):
        for value in [info(version="1.0.2"), info(digest="c" * 64)]:
            with self.assertRaises(RuntimeError):
                self.exercise(published=value)

    def test_invalid_version_is_rejected(self):
        for value in ["latest", "v1.0.3", "1.0.3-rc.0", "1.0.3\nEVIL=1"]:
            with self.assertRaisesRegex(RuntimeError, "RELEASE_VERSION"):
                self.exercise(version=value)

if __name__ == "__main__":
    unittest.main()
