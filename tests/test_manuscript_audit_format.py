"""Presentation-only checks: a format change must not weaken numeric checks."""
import unittest

import audit_pareto_delivery as audit


class ManuscriptAuditFormatTests(unittest.TestCase):
    def test_accepts_verified_coverage_cells_in_gfm_table(self):
        manuscript = """| 模型 | 被覆盖 ↓ | 覆盖基线 ↑ |
| --- | --- | --- |
| Train-L | 0.5148 ± 0.1520 | 0.1707 ± 0.1946 |
"""
        audit.assert_manuscript_row(
            manuscript, ["Train-L", "0.5148 ± 0.1520", "0.1707 ± 0.1946"], "coverage")

    def test_preserves_legacy_html_coverage_validation(self):
        manuscript = ("<table><tbody><tr><td>Train-L</td><td>0.5148 ± 0.1520</td>"
                      "<td>0.1707 ± 0.1946</td></tr></tbody></table>")
        audit.assert_manuscript_row(
            manuscript, ["Train-L", "0.5148 ± 0.1520", "0.1707 ± 0.1946"], "legacy coverage")

    def test_does_not_accept_changed_value_missing_column_or_swapped_directions(self):
        expected = ["Train-L", "0.5148 ± 0.1520", "0.1707 ± 0.1946"]
        corruptions = [
            "| Train-L | 0.5149 ± 0.1520 | 0.1707 ± 0.1946 |",
            "| Train-L | 0.5148 ± 0.1520 |",
            "| Train-L | 0.1707 ± 0.1946 | 0.5148 ± 0.1520 |",
        ]
        for manuscript in corruptions:
            with self.subTest(manuscript=manuscript), self.assertRaises(AssertionError):
                audit.assert_manuscript_row(manuscript, expected, "changed coverage")

    def test_accepts_only_explicit_method_alias_without_loosening_numeric_cells(self):
        manuscript = "| PPO | 1138.513 | 9.9346 | 0.822036 |"
        audit.assert_manuscript_row(
            manuscript, ["PPO-Transformer", "1138.513", "9.9346", "0.822036"], "small PPO",
            first_cell_aliases=("PPO",))

    def test_checks_relocated_checkpoint_hash_for_its_own_model(self):
        manuscript = """| 方法 | 训练实例数 | 训练seed | 训练时间（s） |
| --- | --- | --- | --- |
| PPO（Train-S） | 24 | 3853184734 | 71.153 |

Train-S SHA256：25b48dadc776f21a7e0aacfeea12f2cfe92632f1ba4fb3e0c856ac25b26ea565
"""
        audit.assert_checkpoint_row(
            manuscript, ["PPO（Train-S）", 24, 3853184734, "71.153"], "Train-S",
            "25b48dadc776f21a7e0aacfeea12f2cfe92632f1ba4fb3e0c856ac25b26ea565")

    def test_rejects_missing_checkpoint_hash_or_hash_assigned_to_wrong_model(self):
        row = "| PPO（Train-S） | 24 | 3853184734 | 71.153 |"
        checkpoint = "25b48dadc776f21a7e0aacfeea12f2cfe92632f1ba4fb3e0c856ac25b26ea565"
        for suffix in ("", "\n\nTrain-L SHA256：" + checkpoint):
            with self.subTest(suffix=suffix), self.assertRaises(AssertionError):
                audit.assert_checkpoint_row(row + suffix,
                    ["PPO（Train-S）", 24, 3853184734, "71.153"], "Train-S", checkpoint)

    def test_does_not_match_another_models_hash_in_the_same_paragraph(self):
        checkpoint = "25b48dadc776f21a7e0aacfeea12f2cfe92632f1ba4fb3e0c856ac25b26ea565"
        manuscript = ("| PPO（Train-S） | 24 | 3853184734 | 71.153 |\n\n"
                      "Train-S尚未记录。Train-L SHA256：" + checkpoint)
        with self.assertRaises(AssertionError):
            audit.assert_checkpoint_row(manuscript,
                ["PPO（Train-S）", 24, 3853184734, "71.153"], "Train-S", checkpoint)

    def test_preserves_full_checkpoint_hash_in_legacy_table(self):
        checkpoint = "25b48dadc776f21a7e0aacfeea12f2cfe92632f1ba4fb3e0c856ac25b26ea565"
        manuscript = "| PPO（Train-S） | 24 | 3853184734 | 71.153 | " + checkpoint + " |"
        audit.assert_checkpoint_row(manuscript,
            ["PPO（Train-S）", 24, 3853184734, "71.153"], "Train-S", checkpoint)

    def test_checks_short_checkpoint_alias_and_full_relocated_identity(self):
        checkpoint = "25b48dadc776f21a7e0aacfeea12f2cfe92632f1ba4fb3e0c856ac25b26ea565"
        manuscript = ("| PPO（Train-S） | 24 | 3853184734 | 71.153 | New-S |\n\n"
                      "Train-S（New-S）SHA256：" + checkpoint)
        audit.assert_checkpoint_row(manuscript,
            ["PPO（Train-S）", 24, 3853184734, "71.153"], "Train-S", checkpoint,
            checkpoint_alias="New-S")


if __name__ == "__main__":
    unittest.main()
