import argparse
import csv

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
    "--combine-attained-and-constrained", action=argparse.BooleanOptionalAction,
    help="Combine attained and constrained scores to one (taking the sum)."
)
parser.add_argument(
    "--combine-detailed-values", action=argparse.BooleanOptionalAction,
    help="Combine detailed values (e.g., 'Universalism: concern', 'Universalism: nature', and 'Universalism: tolerance') " +
    "to one (taking the maximum)."
)
parser.add_argument(
    "--quantization"
)

opts = parser.parse_args()


def predict(input):
    quantization_config = None
    classifier = ValueClassifier(use_cpu=opts.cpu, quantization_config=quantization_config)
    classifier.predict_to_tsv(
        input,
        output_file=opts.output,
        attained_and_constrained=not opts.combine_attained_and_constrained,
        detailed_values=not opts.combine_detailed_values)


if opts.input.endswith(".txt"):
    with open(opts.input) as file:
        predict(file)
elif opts.input.endswith(".tsv"):
    with open(opts.input, newline="") as file:
        predict(csv.DictReader(file, delimiter="\t"))
else:
    raise ValueError(f"Input file '{opts.input}' ends neither with '.txt' nor '.tsv'")
