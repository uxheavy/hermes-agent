"""Focused tests for deterministic bounded Plane model guidance."""

from __future__ import annotations

import copy
import unittest

from plane_runtime.g1_contract import G1RunSnapshot
from plane_runtime.presentation import PresentationBoundsError, build_model_guidance
from tests.plane_runtime.test_g1_runtime_process import _digest, make_snapshot


class PresentationTests(unittest.TestCase):
    def test_guidance_is_deterministic_and_contains_exact_assignment_context(self) -> None:
        snapshot = G1RunSnapshot.from_dict(make_snapshot())

        first = build_model_guidance(snapshot)
        second = build_model_guidance(snapshot)

        self.assertEqual(first, second)
        self.assertIn('"workspaceRef":"workspace:test"', first)
        self.assertIn('"targetRef":"target:test"', first)
        self.assertIn('"objective":"Return a deterministic runtime outcome."', first)
        self.assertIn('"acceptanceCriteria":["The outcome is bounded."]', first)
        self.assertIn('"contextRefs":[]', first)
        self.assertIn('"operationRef":"operation:work_item.read"', first)
        self.assertIn('"required":["project_id","issue_id"]', first)
        self.assertIn("Call plane_operation with the listed exact input shape", first)
        self.assertIn("operation:catalog.describe", first)
        self.assertIn("Never guess input field names.", first)
        self.assertIn("Disclosure is not authorization.", first)
        self.assertIn("Ordinary final text is not publication.", first)

    def test_guidance_rejects_oversize_without_truncating_behavioral_prompt(self) -> None:
        raw = copy.deepcopy(make_snapshot())
        raw["profile"]["behavioralPrompt"] = "x" * 32768  # type: ignore[index]
        raw["contentDigest"] = _digest(
            "snapshot",
            {key: value for key, value in raw.items() if key != "contentDigest"},
        )
        snapshot = G1RunSnapshot.from_dict(raw)

        with self.assertRaisesRegex(PresentationBoundsError, "32768 UTF-8 bytes"):
            build_model_guidance(snapshot)


if __name__ == "__main__":
    unittest.main()
