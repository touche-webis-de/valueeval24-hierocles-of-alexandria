import csv
from pathlib import Path
import unittest
from valueeval24_hierocles_of_alexandria import combine_attained_and_constrained, combine_detailed_values

examples_path = Path("data") / "examples"


class TestProcessing(unittest.TestCase):

    def test_combine_attained_and_constrained(self):
        file_name = examples_path / "simple-output.tsv"
        with open(file_name, newline="") as file:
            input = csv.DictReader(file, delimiter="\t")
            for predictions in combine_attained_and_constrained(input):
                self.assertEqual(len(predictions.keys()), 4 + 19)
                self.assertGreaterEqual(predictions["Universalism: nature"], 0.5)

    def test_combine_detailed_values(self):
        file_name = examples_path / "simple-output.tsv"
        with open(file_name, newline="") as file:
            input = csv.DictReader(file, delimiter="\t")
            predictions = list(combine_detailed_values(input))
            self.assertEqual(len(predictions[0].keys()), 4 + 2 * 12)
            self.assertGreaterEqual(predictions[0]["Universalism attained"], 0.5)
            self.assertEqual(len(predictions[1].keys()), 4 + 2 * 12)
            self.assertGreaterEqual(predictions[1]["Universalism constrained"], 0.5)

    def test_combine_attained_and_constrained_and_then_detailed_values(self):
        file_name = examples_path / "simple-output.tsv"
        with open(file_name, newline="") as file:
            input = csv.DictReader(file, delimiter="\t")
            for predictions in combine_detailed_values(combine_attained_and_constrained(input)):
                self.assertEqual(len(predictions.keys()), 4 + 12)
                self.assertGreaterEqual(predictions["Universalism"], 0.5)

    def test_combine_detailed_values_and_then_attained_and_constrained(self):
        file_name = examples_path / "simple-output.tsv"
        with open(file_name, newline="") as file:
            input = csv.DictReader(file, delimiter="\t")
            with self.assertRaises(KeyError):
                for _ in combine_attained_and_constrained(combine_detailed_values(input)):
                    print("Never reached!", flush=True)


if __name__ == '__main__':
    unittest.main()
