import io
import unittest

import pandas as pd

from analyzer import (
    group_fairness_metrics,
    load_dataset,
    selection_rate_parity,
    train_and_compare,
)


class DatasetTests(unittest.TestCase):
    def test_load_dataset_converts_numbers_and_removes_exact_duplicates(self):
        source = io.StringIO(
            "outcome,age,group\n"
            "yes,21.0,A\n"
            "yes,21.0,A\n"
            "no,35.0,B\n"
        )

        result = load_dataset(source)

        self.assertEqual(result.original_rows, 3)
        self.assertEqual(result.duplicate_rows, 1)
        self.assertEqual(len(result.data), 2)
        self.assertTrue(pd.api.types.is_numeric_dtype(result.data["age"]))

    def test_small_groups_are_visible_but_excluded_from_gaps(self):
        actual = pd.Series(
            ["yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes", "no"]
        )
        predicted = pd.Series(
            ["yes", "no", "no", "no", "yes", "yes", "yes", "no", "yes", "yes"]
        )
        groups = pd.Series(
            ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C"]
        )

        result = group_fairness_metrics(
            y_test=actual,
            predictions=predicted,
            groups=groups,
            positive_label="yes",
            minimum_group_size=3,
        )

        self.assertEqual(len(result["by_group"]), 3)
        self.assertTrue(
            result["by_group"].loc[
                result["by_group"]["Group"] == "C", "Support"
            ].iloc[0].startswith("Below minimum")
        )
        self.assertAlmostEqual(result["gaps"]["Selection rate"], 0.5)
        self.assertAlmostEqual(result["gaps"]["Selection-rate parity"], 1 / 3)

    def test_selection_rate_parity_can_be_recovered_from_group_table(self):
        old_results_table = pd.DataFrame(
            {
                "Group": ["FEMALE", "MALE", "SMALL"],
                "Selection rate": [0.234, 0.229, 1.0],
                "Support": ["Included", "Included", "Below minimum (30)"],
            }
        )

        self.assertAlmostEqual(
            selection_rate_parity(old_results_table), 0.229 / 0.234
        )

    def test_training_returns_comparison_and_uses_sensitive_field_for_evaluation(self):
        data = pd.DataFrame(
            {
                "outcome": ["yes", "no"] * 40,
                "signal": [1, 0] * 40,
                "gender": ["A", "B", "B", "A"] * 20,
            }
        )

        result = train_and_compare(
            data=data,
            target="outcome",
            sensitive_columns=["gender"],
            model_names=["Logistic regression"],
            positive_label="yes",
            minimum_group_size=2,
        )

        self.assertEqual(len(result["comparison"]), 1)
        self.assertIn("gender", result["fairness"])
        self.assertEqual(
            len(result["fairness"]["gender"]["Logistic regression"]["by_group"]), 2
        )
        self.assertNotIn(
            "gender",
            result["models"]["Logistic regression"]["importance"]["Feature"].tolist(),
        )


if __name__ == "__main__":
    unittest.main()
