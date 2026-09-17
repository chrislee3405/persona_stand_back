import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch, mock_open
import publish_image


class PublishOnce(unittest.TestCase):
    def exercise(self, probe):
        with patch("builtins.open", mock_open()):
            env = {"GITHUB_REPOSITORY": "example/persona_stand_back", "GITHUB_SHA": "a" * 40,
                   "GITHUB_SERVER_URL": "https://github.com", "GITHUB_RUN_ID": "123",
                   "GITHUB_STEP_SUMMARY": "summary"}
            info = {"Digest": "sha256:" + "b" * 64,
                    "Labels": {"org.opencontainers.image.revision": "a" * 40,
                               "org.opencontainers.image.source": "https://github.com/example/persona_stand_back"}}
            with patch.dict(publish_image.os.environ, env), patch.object(publish_image.subprocess, "run", return_value=probe) as execute, patch.object(publish_image, "run", return_value=json.dumps(info)), patch.object(Path, "write_text"):
                publish_image.publish()
                return [call.args[0] for call in execute.call_args_list]

    def test_existing_commit_is_reused_without_build_or_push(self):
        calls = self.exercise(subprocess.CompletedProcess([], 0, "", ""))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][:2], ["skopeo", "inspect"])

    def test_missing_commit_is_built_and_published(self):
        calls = self.exercise(subprocess.CompletedProcess([], 1, "", "manifest unknown"))
        self.assertEqual([call[:2] for call in calls], [["skopeo", "inspect"], ["docker", "build"], ["docker", "push"]])

    def test_auth_and_network_errors_never_allow_build(self):
        for message in ["unauthorized", "connection timed out", "permission denied"]:
            with self.assertRaisesRegex(RuntimeError, "Cannot safely determine"):
                self.exercise(subprocess.CompletedProcess([], 1, "", message))


if __name__ == "__main__":
    unittest.main()
