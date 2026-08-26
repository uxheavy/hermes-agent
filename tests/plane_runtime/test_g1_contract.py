"""Focused tests for the accepted eager-operation runtime contract."""

from __future__ import annotations

import copy
import unittest
from collections.abc import Mapping

from plane_runtime.g1_contract import (
    G1ContractError,
    G1RunSnapshot,
    G1_CONTRACT_DIGESTS,
    G1_MANIFEST_DIGEST,
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
            G1_CONTRACT_DIGESTS,
            {
                "runSnapshot": "e84f7b5b2a92c98d1fd1bcdbd2bfc6079692cba69e7027b3f40f700b7cc0d673",
                "invocationEnvelope": "b7a15d74406f1624cdb7cd95b42edfd1ffee596abe57e4f00ed60e2e23ded995",
                "runtimeEvent": "a4c91dee656fdff4b1afbe8fd7fba0b4d7be9fbeecb88da748e3f3e200841444",
                "runtimeExit": "4b3bf10674ebadc8e51e9aad26b856509cf792e238ac6af424fbfc2cc87ac181",
                "runtimeDurableState": "444c944ec8a5054f33c8662470529a1f4565d42ff06138438beceeef7967a0da",
            },
        )
        self.assertEqual(
            G1_MANIFEST_DIGEST,
            "f897bd22adcd91b713cb82480a1a196d9145fb434e8e155fe7c377554eb1f6be",
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

    def test_snapshot_content_digest_authenticates_input_schema(self) -> None:
        raw = make_snapshot()
        raw["toolCatalog"]["eagerOperations"][0]["inputSchema"]["required"] = ["other"]  # type: ignore[index]
        with self.assertRaisesRegex(G1ContractError, "immutable content"):
            G1RunSnapshot.from_dict(raw)

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
