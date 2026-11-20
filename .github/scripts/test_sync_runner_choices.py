# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for runner dropdown generation; run with unittest discovery."""

import configparser
import json
import tempfile
import unittest
from pathlib import Path

from sync_runner_choices import BEGIN, END, runner_labels, update_options


class RunnerChoicesTests(unittest.TestCase):
    """Cover enabled runner selection and isolated workflow updates."""

    def labels(self, content: str) -> list[str]:
        """Read a temporary testplan."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "testplan.ini"
            path.write_text(content, encoding="utf-8")
            return runner_labels(path)

    def test_selects_enabled_unique_labels_in_order(self):
        labels = self.labels(
            "[first]\ntarget_available=true\nruns_on=runner-b\n"
            "[disabled]\ntarget_available=false\nruns_on=runner-disabled\n"
            "[second]\ntarget_available=true\nruns_on=runner-a\n"
            "[duplicate]\ntarget_available=true\nruns_on=runner-b\n"
            "[unspecified]\nruns_on=runner-unspecified\n"
        )
        self.assertEqual(labels, ["runner-b", "runner-a"])

    def test_rejects_empty_enabled_label(self):
        with self.assertRaisesRegex(ValueError, "empty runs_on"):
            self.labels("[target]\ntarget_available=true\nruns_on= \n")

    def test_rejects_missing_enabled_label(self):
        with self.assertRaises(KeyError):
            self.labels("[target]\ntarget_available=true\n")

    def test_rejects_invalid_availability(self):
        with self.assertRaises(ValueError):
            self.labels("[target]\ntarget_available=typo\nruns_on=runner\n")

    def test_rejects_empty_enabled_list(self):
        with self.assertRaisesRegex(ValueError, "no enabled"):
            self.labels("[target]\ntarget_available=false\nruns_on=runner\n")

    def test_rejects_missing_testplan(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                runner_labels(Path(directory) / "missing.ini")

    def test_rejects_malformed_testplan(self):
        with self.assertRaises(configparser.Error):
            self.labels("not an ini file\n")

    def test_changes_only_marked_block_and_is_idempotent(self):
        before = "on:\n  workflow_dispatch:\n    inputs:\n      runs_on:\n        type: choice\n"
        after = "\n  workflow_call:\n    inputs:\n      runs_on:\n        type: string\n"
        original = f"{before}{BEGIN}\n        options:\n          - old\n{END}{after}"
        labels = ["runner-b", 'runner-"quoted"']
        updated = update_options(original, labels)
        expected = "\n".join(f"          - {json.dumps(label)}" for label in labels)
        self.assertEqual(updated, f"{before}{BEGIN}\n        options:\n{expected}\n{END}{after}")
        self.assertEqual(update_options(updated, labels), updated)

    def test_rejects_invalid_markers(self):
        for content in ("", BEGIN, END, f"{END}\n{BEGIN}", f"{BEGIN}\n{BEGIN}\n{END}"):
            with self.subTest(content=content), self.assertRaises(ValueError):
                update_options(content, ["runner"])


if __name__ == "__main__":
    unittest.main()
