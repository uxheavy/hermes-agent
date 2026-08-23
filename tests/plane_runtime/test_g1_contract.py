"""Focused tests for the accepted eager-operation runtime contract."""

from __future__ import annotations

import copy
import unittest
from collections.abc import Mapping

from plane_runtime.g1_contract import (
    G1ContractError,
    G1RunSnapshot,
    G1_CONTRACT_DIGESTS,
)
from tests.plane_runtime.test_g1_runtime_process import _digest, make_snapshot


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


class G1ContractTests(unittest.TestCase):
    def test_eager_schema_is_preserved_and_digests_match_plane_artifacts(self) -> None:
        raw = make_snapshot()
        snapshot = G1RunSnapshot.from_dict(raw)

        self.assertEqual(
            _thaw(snapshot.raw["toolCatalog"]["eagerOperations"][0]["inputSchema"]),  # type: ignore[index]
            raw["toolCatalog"]["eagerOperations"][0]["inputSchema"],  # type: ignore[index]
        )
        self.assertEqual(
            G1_CONTRACT_DIGESTS["runSnapshot"],
            "ca2944e248210658a6c0514c29e23d9dc002d2f9d397e6b5c7aef50d36202dc1",
        )
        self.assertEqual(
            G1_CONTRACT_DIGESTS["runtimeEvent"],
            "d0fb1c67a7424f5359f9c09ff7206ef7d3d0d6e90e62b724c4a5e4e4bc13412d",
        )
        self.assertEqual(
            G1_CONTRACT_DIGESTS["runtimeExit"],
            "f596e131d3d1bf94c52352fa2156d6dedf4c793f1b31d3fbd6b7a478f4401df9",
        )

    def test_eager_presentation_fields_are_strict(self) -> None:
        raw = make_snapshot()
        operation = raw["toolCatalog"]["eagerOperations"][0]  # type: ignore[index]

        extra = copy.deepcopy(raw)
        extra["toolCatalog"]["eagerOperations"][0]["unexpected"] = True  # type: ignore[index]
        with self.assertRaisesRegex(G1ContractError, "unknown field"):
            G1RunSnapshot.from_dict(extra)

        missing_schema = copy.deepcopy(raw)
        del missing_schema["toolCatalog"]["eagerOperations"][0]["inputSchema"]  # type: ignore[index]
        with self.assertRaisesRegex(G1ContractError, "missing field"):
            G1RunSnapshot.from_dict(missing_schema)

        progressive = copy.deepcopy(raw)
        progressive["toolCatalog"]["eagerOperations"][0]["disclosure"] = "progressive"  # type: ignore[index]
        with self.assertRaisesRegex(G1ContractError, "must be eager"):
            G1RunSnapshot.from_dict(progressive)

        self.assertIsNotNone(operation)

    def test_model_toolset_is_required_at_the_g1_boundary(self) -> None:
        missing = make_snapshot()
        del missing["toolCatalog"]["modelToolset"]  # type: ignore[index]
        with self.assertRaisesRegex(G1ContractError, "modelToolset"):
            G1RunSnapshot.from_dict(missing)

    def test_snapshot_content_digest_authenticates_input_schema(self) -> None:
        raw = make_snapshot()
        raw["toolCatalog"]["eagerOperations"][0]["inputSchema"]["required"] = ["other"]  # type: ignore[index]
        with self.assertRaisesRegex(G1ContractError, "immutable content"):
            G1RunSnapshot.from_dict(raw)

    def test_code_mode_phase_is_bounded_and_typed(self) -> None:
        raw = make_snapshot()
        raw["runtimePolicy"]["codeModePhase"] = "post_search"  # type: ignore[index]
        raw["contentDigest"] = _digest(
            "snapshot", {key: value for key, value in raw.items() if key != "contentDigest"}
        )
        snapshot = G1RunSnapshot.from_dict(raw)
        self.assertEqual(snapshot.code_mode_phase, "post_search")

        invalid = copy.deepcopy(raw)
        invalid["runtimePolicy"]["codeModePhase"] = "before_search"  # type: ignore[index]
        invalid["contentDigest"] = _digest(
            "snapshot", {key: value for key, value in invalid.items() if key != "contentDigest"}
        )
        with self.assertRaisesRegex(G1ContractError, "codeModePhase"):
            G1RunSnapshot.from_dict(invalid)

    def test_input_schema_and_aggregate_bounds_reject_without_truncation(self) -> None:
        per_schema = make_snapshot()
        per_schema["toolCatalog"]["eagerOperations"][0]["inputSchema"] = {  # type: ignore[index]
            "description": "x" * (16 * 1024)
        }
        per_schema["contentDigest"] = _digest(
            "snapshot",
            {key: value for key, value in per_schema.items() if key != "contentDigest"},
        )
        with self.assertRaisesRegex(G1ContractError, "bounded canonical JSON Schema"):
            G1RunSnapshot.from_dict(per_schema)

        over_count = make_snapshot()
        over_count["toolCatalog"] = {
            "catalogDigest": "content:" + "c" * 64,
            "modelToolset": "standard",
            "eagerOperations": [
                {
                    "operationRef": f"operation:operation-{index}",
                    "schemaDigest": "content:" + "d" * 64,
                    "inputSchema": {"type": "object"},
                    "disclosure": "eager",
                }
                for index in range(65)
            ],
        }
        over_count["contentDigest"] = _digest(
            "snapshot",
            {key: value for key, value in over_count.items() if key != "contentDigest"},
        )
        with self.assertRaisesRegex(G1ContractError, "at most 64"):
            G1RunSnapshot.from_dict(over_count)

        aggregate = make_snapshot()
        aggregate["toolCatalog"] = {
            "catalogDigest": "content:" + "c" * 64,
            "modelToolset": "standard",
            "eagerOperations": [
                {
                    "operationRef": f"operation:operation-{index}",
                    "schemaDigest": "content:" + "d" * 64,
                    "inputSchema": {"description": "x" * 8_200},
                    "disclosure": "eager",
                }
                for index in range(64)
            ],
        }
        aggregate["contentDigest"] = _digest(
            "snapshot",
            {key: value for key, value in aggregate.items() if key != "contentDigest"},
        )
        with self.assertRaisesRegex(G1ContractError, "canonical JSON bytes"):
            G1RunSnapshot.from_dict(aggregate)


if __name__ == "__main__":
    unittest.main()
