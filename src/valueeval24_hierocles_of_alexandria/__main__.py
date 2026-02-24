import argparse
import pyvalues
from transformers import BitsAndBytesConfig

from .value_classifier import ValueClassifier

parser = argparse.ArgumentParser(
    prog="valueeval24-hierocles-of-alexandria",
    description="Detect human values in text (winning model of ValueEval'24)"
)
parser.add_argument(
    "--input", type=str, required=True,
    help="Input file, either a '.txt' with one English sentence per line or a '.tsv' with the column 'Text' (the sentence) " +
    "and optional columns 'Language' (one of 'BG', 'DE', 'EL', 'EN' (default), 'FR', 'HE', 'IT', 'NL', or 'TR'), 'Text-ID' " +
    "(change of values indicates new text), and 'Sentence-ID' (arbitrary integer to identify the sentence)."
)
parser.add_argument(
    "--output", type=str, required=True,
    help="Output '.tsv' file."
)
parser.add_argument(
    "--cpu", action=argparse.BooleanOptionalAction,
    help="Force usage of CPU (and not GPU)."
)
parser.add_argument(
    "--quantization", choices=["none", "8bit", "4bit"],
    help="Use a quantized model. The full, 8bit, and 4bit models require about 20GB, 10GB, and 5GB GPU RAM, respectively."
)

opts = parser.parse_args()


def get_classifier():
    kwargs = {}
    if opts.quantization == "none":
        kwargs["quantization_config"] = None
    elif opts.quantization == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif opts.quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    return ValueClassifier(use_cpu=opts.cpu, **kwargs)

with open(opts.output, "w") as output_file:
    writer = pyvalues.RefinedValuesWithAttainment.writer_tsv_with_text(output_file)
    classifier = get_classifier()

    if opts.input.endswith(".txt"):
        with open(opts.input) as file:
            predictions = classifier.classify_document_for_refined_values_with_attainment(
                segments=file
            )
            writer.write_all(predictions, language="en")  # type: ignore
    elif opts.input.endswith(".tsv"):
        for document in pyvalues.values.Values.read_tsv(
            opts.input,
            id_field="Text-ID",
            read_values=False
        ):
            if document.segments is not None:
                predictions = classifier.classify_document_for_refined_values_with_attainment(
                    segments=document.segments,
                    language=document.language
                )
                writer.write_all(
                    predictions,
                    language=document.language,
                    document_id=document.id
                )
    else:
        raise ValueError(f"Input file '{opts.input}' ends neither with '.txt' nor '.tsv'")
