import csv
from pathlib import Path
import unittest

from valueeval24_hierocles_of_alexandria import ValueClassifier

examples_path = Path("data") / "examples"


class TestClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._classifier = ValueClassifier(use_cpu=True)

    def test_simple_tsv(self):
        file_name = "simple.tsv"
        with open(examples_path / file_name, newline="") as file:
            predictions = list(self._classifier.predict(csv.DictReader(file, delimiter="\t")))
            self.assertGreaterEqual(predictions[0]["Universalism: nature attained"], 0.5)
            self.assertGreaterEqual(predictions[1]["Universalism: nature constrained"], 0.5)

    def test_simple_txt(self):
        file_name = "simple.txt"
        with open(examples_path / file_name) as file:
            predictions = list(self._classifier.predict(file))
            self.assertGreaterEqual(predictions[0]["Universalism: nature attained"], 0.5)
            self.assertGreaterEqual(predictions[1]["Universalism: nature constrained"], 0.5)


if __name__ == '__main__':
    unittest.main()
