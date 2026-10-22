import json
import unittest
from pathlib import Path

CASES = json.loads(Path(__file__).with_name("cases.json").read_text())["cases"]
BY_ID = {case["id"]: case for case in CASES}


class FixtureIntegrity(unittest.TestCase):
    def test_unique_cases_and_closed_records(self):
        self.assertEqual(len(CASES), 24)
        self.assertEqual(len(BY_ID), len(CASES))
        for case in CASES:
            self.assertEqual(set(case), {"id", "input", "expected"})
            self.assertIn(case["expected"]["static"], {"accepted", "missing", "duplicate", "contradictory"})

    def test_negative_single_field_mutations(self):
        base = BY_ID["scalar_context"]["input"]
        fields = {
            "missing_initial_declaration": "initial",
            "duplicate_initial_declaration": "initial",
            "wrong_initial_type": "initial",
            "missing_initial_port": "initial",
            "foreign_initial_target": "initial",
            "foreign_initial_node": "initial",
            "missing_same_owner_initial_node": "initial",
            "missing_context_interface_port": "bindings",
            "unmarked_destination": "nodes",
            "context_destination_type_mismatch": "nodes",
            "missing_second_target_declaration": "targets",
        }
        for name, field in fields.items():
            actual = BY_ID[name]["input"]
            self.assertEqual({key for key in base if actual[key] != base[key]}, {field}, name)
            self.assertEqual(BY_ID[name]["expected"]["provider_calls"], 0)

    def test_collision_order_is_only_permutation(self):
        left = BY_ID["mixed_source_kinds"]["input"]
        right = BY_ID["mixed_source_kinds_reversed"]["input"]
        self.assertEqual(left["bindings"], list(reversed(right["bindings"])))
        self.assertEqual({k:v for k,v in left.items() if k != "bindings"}, {k:v for k,v in right.items() if k != "bindings"})

    def test_collection_materialization_accounting(self):
        case = BY_ID["collection_context"]
        sizes = [len(item["text"].encode("utf-8")) for item in case["input"]["source_items"]]
        expected = case["expected"]
        self.assertEqual(expected["materialization_artifacts"], len(sizes) + 1)
        self.assertEqual(expected["materialization_bytes"], 2 * sum(sizes))
        self.assertEqual(expected["materialization_edges"], len(sizes))
        self.assertEqual(len(expected["lineage"]["source_parents"]), len(sizes))


if __name__ == "__main__":
    unittest.main()
