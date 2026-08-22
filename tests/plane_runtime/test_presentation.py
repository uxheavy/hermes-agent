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
        self.assertIn('"required":["preparedCallRef"]', first)
        self.assertNotIn('"project_id"', first)
        self.assertNotIn('"issue_id"', first)
        self.assertIn("Call plane_operation directly with the listed exact input shape", first)
        self.assertIn("eager operations are already disclosed and must not be rediscovered", first)
        self.assertIn("operation:catalog.describe once", first)
        self.assertIn("operation:search_workspace the first Plane call", first)
        self.assertIn("copy its opaque workItemReadCall string", first)
        self.assertIn("do not construct an action/operationRef/input envelope", first)
        self.assertIn("Never guess input field names.", first)
        self.assertIn("Disclosure is not authorization.", first)
        self.assertIn("Ordinary final text is not publication.", first)

    def test_manager_guidance_prioritizes_prepared_search_over_catalog_rediscovery(self) -> None:
        raw = copy.deepcopy(make_snapshot())
        raw["assignment"] = {
            "assignmentRef": "assignment:manager",
            "revision": "revision:one",
            "targetRef": "target:assigned-work-item",
            "objective": "Coordinate a bounded Manager objective for the assigned work item.",
            "acceptanceCriteria": ["Publish one reviewable result."],
        }
        raw["toolCatalog"]["eagerOperations"] = [  # type: ignore[index]
            {
                "operationRef": operation_ref,
                "schemaDigest": "content:" + "e" * 64,
                "inputSchema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["query"],
                    "properties": {"query": {"type": "string"}},
                },
                "disclosure": "eager",
            }
            for operation_ref in (
                "operation:catalog.search",
                "operation:catalog.describe",
                "operation:search_workspace",
                "operation:agent.assignment.delegate",
            )
        ]
        raw["contentDigest"] = _digest(
            "snapshot",
            {key: value for key, value in raw.items() if key != "contentDigest"},
        )

        guidance = build_model_guidance(G1RunSnapshot.from_dict(raw))

        self.assertIn('"operation:search_workspace"', guidance)
        self.assertIn("operation:search_workspace the first Plane call", guidance)
        self.assertIn("do not begin with catalog.search or catalog.describe", guidance)
        self.assertNotIn("catalog.describe before invocation", guidance)

    def test_guidance_renders_frozen_standard_route_and_preserves_no_route_output(self) -> None:
        no_route = build_model_guidance(G1RunSnapshot.from_dict(make_snapshot()))
        self.assertIn(
            'Plane invocation guidance:\n{"acceptanceCriteria":["The outcome is bounded."],"contextRefs":[],"eagerOperations":',
            no_route,
        )
        self.assertNotIn('"standardRoute":', no_route)

        raw = copy.deepcopy(make_snapshot())
        raw["toolCatalog"]["standardRoute"] = {  # type: ignore[index]
            "schemaVersion": "plane.standard-route/v1",
            "steps": [{"operationRef": "operation:search_workspace"}],
        }
        raw["contentDigest"] = _digest(
            "snapshot",
            {key: value for key, value in raw.items() if key != "contentDigest"},
        )

        guidance = build_model_guidance(G1RunSnapshot.from_dict(raw))

        self.assertIn(
            '"standardRoute":{"schemaVersion":"plane.standard-route/v1","steps":[{"operationRef":"operation:search_workspace"}]}',
            guidance,
        )

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
