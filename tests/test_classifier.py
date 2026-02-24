from pathlib import Path
import unittest
import pyvalues

from valueeval24_hierocles_of_alexandria import ValueClassifier

examples_path = Path("data") / "examples"


class TestClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._classifier = ValueClassifier(use_cpu=True)

    def test_simple_tsv(self):
        file_name = "simple.tsv"
        documents = list(pyvalues.values.Values.read_tsv(
            examples_path / file_name,
            id_field="Text-ID",
            read_values=False)
        )
        self.assertEqual(len(documents), 1)

        document = documents[0]
        if document.segments is None:
            raise ValueError()
        predictions = list(self._classifier.classify_document_for_refined_values_with_attainment(
            segments=document.segments,
            language=document.language
        ))
        self.assertEqual(len(predictions), 2)
        self.assertGreaterEqual(predictions[0][0].universalism_nature.attained, 0.5)
        self.assertGreaterEqual(predictions[1][0].universalism_nature.constrained, 0.5)

    def test_simple_txt(self):
        file_name = "simple.txt"
        with open(examples_path / file_name) as file:
            predictions = list(self._classifier.classify_document_for_refined_values_with_attainment(
                segments=file
            ))
            self.assertGreaterEqual(predictions[0][0].universalism_nature.attained, 0.5)
            self.assertGreaterEqual(predictions[1][0].universalism_nature.constrained, 0.5)


if __name__ == '__main__':
    unittest.main()
