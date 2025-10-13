import csv
import datasets
import numpy
import transformers
from transformers.training_args import TrainingArguments
from typing import Generator
import pandas
from pathlib import Path
import sys

from .multi_head_model import MultiHead_MultiLabel_XL, id2lang_dict, lang_dict, lang_default

datasets.utils.logging.disable_progress_bar()

values = ["Self-direction: thought", "Self-direction: action", "Stimulation", "Hedonism", "Achievement",
          "Power: dominance", "Power: resources", "Face", "Security: personal", "Security: societal", "Tradition",
          "Conformity: rules", "Conformity: interpersonal", "Humility", "Benevolence: caring",
          "Benevolence: dependability", "Universalism: concern", "Universalism: nature", "Universalism: tolerance"]
labels = sum([[value + " attained", value + " constrained"] for value in values], [])
id2label = {idx: label for idx, label in enumerate(labels)}
label_thresholds = [
    0.10, 0.30, 0.25, 0.25, 0.25, 0.25, 0.35, 0.30, 0.35, 0.25,
    0.35, 0.15, 0.20, 0.25, 0.10, 0.20, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.25, 0.30, 0.15, 0.10, 0.15, 0.10, 0.00, 0.30, 0.10,
    0.15, 0.40, 0.15, 0.20, 0.10, 0.10, 0.25, 0.20
]

model_name = "SotirisLegkas/multi-head-xlm-xl-tokens-38"
look_back = 2


def predictions_to_tsv(predictions, output_file=sys.stdout):
    if isinstance(output_file, str):
        with open(output_file, "w", newline="") as output_file_handle:
            predictions_to_tsv(predictions, output_file=output_file_handle)
    else:
        fieldnames = ["Text-ID", "Sentence-ID", "Text", "Language"] + labels
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for prediction in predictions:
            writer.writerow(prediction)


class ValueClassifier(object):

    def __init__(self, use_cpu=False):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model = MultiHead_MultiLabel_XL.from_pretrained(model_name, problem_type="multi_label_classification")
        self.trainer = transformers.Trainer(model=self.model, args=TrainingArguments(model_name, use_cpu=use_cpu))
        unused_train_dir = Path(model_name)
        unused_train_dir.rmdir()
        unused_train_dir.parent.rmdir()

    def _validate_sentence_id(self, validated, entry, i):
        if "Sentence-ID" in entry.keys():
            if isinstance(entry["Sentence-ID"], int):
                validated["Sentence-ID"] = entry["Sentence-ID"]
            elif isinstance(entry["Sentence-ID"], str):
                if entry["Sentence-ID"].isdigit():
                    validated["Sentence-ID"] = int(entry["Sentence-ID"])
                else:
                    raise ValueError(f"'Sentence-ID' of entry {i} can not be cast to an int as it is '{entry['Sentence-ID']}'")
            else:
                raise ValueError(f"'Sentence-ID' of entry {i} is not an int but {type(entry['Sentence-ID'])}")

    def _validate_text_id(self, validated, entry, i):
        if "Text-ID" in entry.keys():
            if isinstance(entry["Text-ID"], str):
                if entry["Text-ID"] != validated["Text-ID"] and "Sentence-ID" not in entry.keys():
                    validated["Sentence-ID"] = 1
                validated["Text-ID"] = entry["Text-ID"]
            else:
                raise ValueError(f"'Text-ID' of entry {i} is not a str but {type(entry['Text-ID'])}")

    def _validate_language(self, validated, entry, i):
        if "Language" in entry.keys():
            if isinstance(entry["Language"], str):
                if entry["Language"] in lang_dict.keys():
                    validated["Language"] = lang_dict[entry["Language"]]
                else:
                    raise ValueError(f"Unknown language of entry {i}: {entry['Language']}")
            else:
                raise ValueError(f"'Language' of entry {i} is not a str but {type(entry['Language'])}")

    def _validate_input_data(self, data: Generator[str | dict, None, None]) -> Generator[dict, None, None]:
        last_text_id = "0"
        last_sentence_id = 0

        for i, entry in enumerate(data):
            validated = None
            if isinstance(entry, str):
                validated = {
                    "Text-ID": last_text_id,
                    "Sentence-ID": last_sentence_id + 1,
                    "Text": entry,
                    "Language": lang_default
                }
            elif isinstance(entry, dict):
                if "Text" in entry.keys():
                    if isinstance(entry["Text"], str):
                        validated = {
                            "Text-ID": last_text_id,
                            "Sentence-ID": last_sentence_id + 1,
                            "Text": entry["Text"],
                            "Language": lang_default
                        }
                        self._validate_sentence_id(validated, entry, i)
                        self._validate_text_id(validated, entry, i)
                        self._validate_language(validated, entry, i)
                    else:
                        raise ValueError(f"'Text' of entry {i} is not a str but {type(entry['Text'])}")
                else:
                    raise ValueError(f"No 'Text' in entry {i}, only {', '.join(entry.keys())}")
            else:
                raise ValueError(f"Entry {i} is neither a str nor dict, but {type(entry)}")

            last_sentence_id = validated["Sentence-ID"]
            last_text_id = validated["Text-ID"]
            yield validated

    def _sigmoid(self, predictions):
        return 1/(1 + numpy.exp(-predictions))

    def _map_to_confidence(self, x, threshold):
        if -0.00000001 <= x <= 1.00000001:
            if x >= threshold:
                return (x - threshold) / (threshold - 1) * (-0.5) + 0.5
            else:
                return x / threshold * 0.5
        else:
            raise ValueError(f"{x} outside of interval [0,1].")

    def _thresholds(self, predictions):
        predictions_sigmoid = self._sigmoid(predictions)
        confidence = [
            self._map_to_confidence(x, threshold) for x, threshold in zip(predictions_sigmoid.tolist(), label_thresholds)
        ]
        return confidence

    def _predict_for_text(self, text_data: Generator[dict, None, None]) -> Generator[dict, None, None]:
        previous_sentences = None
        last_text_id = None
        for entry in text_data:
            if last_text_id != entry["Text-ID"]:
                previous_sentences = []
            text = entry["Text"]
            text_in_context = "".join(previous_sentences[-look_back:] + [text])
            input = {"Text": text_in_context, "language": entry["Language"]}
            input.update(self.tokenizer(text_in_context, padding="max_length", max_length=512, truncation=True))
            input = datasets.Dataset.from_dict({key: [value] for key, value in input.items()})
            predictions, _, _ = self.trainer.predict(input, metric_key_prefix="predict")
            confidence_values = self._thresholds(predictions[0])
            output = entry.copy()
            output["Language"] = id2lang_dict[output["Language"]]
            for j, label in enumerate(labels):
                output[label] = confidence_values[j]
            yield output

            # prepare for next sentence
            last_text_id = entry["Text-ID"]
            text_labels = [f"<{id2label[j]}>" for j, value in enumerate(confidence_values) if value >= 0.5]
            if len(text_labels) > 0:
                previous_sentences.append(f"{text} <{' '.join(text_labels)}> </s> ")
            else:
                previous_sentences.append(f"{text} <NONE> </s> ")

    def predict(self, data) -> Generator[dict, None, None]:
        if isinstance(data, pandas.DataFrame):
            return self.predict(data.to_dict("records"))
        else:
            return self._predict_for_text(self._validate_input_data(data))

    def predict_to_tsv(self, data, output_file=sys.stdout) -> None:
        predictions_to_tsv(self.predict(data), output_file)
