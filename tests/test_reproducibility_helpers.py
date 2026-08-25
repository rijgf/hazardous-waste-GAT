from __future__ import annotations

import unittest

from src.reproducibility import (
    derive_seed,
    deserialize_plan,
    plan_sha256,
    serialize_plan,
)


class ReproducibilityHelperTests(unittest.TestCase):
    def test_same_inputs_derive_same_seed(self) -> None:
        first = derive_seed(42, "test", "small", 3, "0.25_0.75", 1)
        second = derive_seed(42, "test", "small", 3, "0.25_0.75", 1)
        self.assertEqual(first, second)

    def test_different_namespace_derives_different_seed(self) -> None:
        train_seed = derive_seed(42, "train", "small", 3)
        test_seed = derive_seed(42, "test", "small", 3)
        self.assertNotEqual(train_seed, test_seed)

    def test_plan_round_trip_and_hash_are_stable(self) -> None:
        plan = {
            ("K2", 2): ["D2", "n::G2::S1", "D2"],
            ("K1", 1): ["D1", "n::G1::S2", "n::G3::S2", "D1"],
        }
        same_plan_different_insertion_order = {
            ("K1", 1): ["D1", "n::G1::S2", "n::G3::S2", "D1"],
            ("K2", 2): ["D2", "n::G2::S1", "D2"],
        }

        encoded = serialize_plan(plan)
        decoded = deserialize_plan(encoded)

        self.assertEqual(decoded, plan)
        self.assertEqual(encoded, serialize_plan(decoded))
        self.assertEqual(plan_sha256(plan), plan_sha256(decoded))
        self.assertEqual(plan_sha256(plan), plan_sha256(same_plan_different_insertion_order))


if __name__ == "__main__":
    unittest.main()
