import csv
import numpy
import transformers
from typing import Generator
import torch
import pandas
import sys

from .multi_head_model import MultiHead_MultiLabel_XL, id2lang_dict, lang_dict, lang_default

values = [
    "Self-direction: thought", "Self-direction: action", "Stimulation", "Hedonism", "Achievement",
    "Power: dominance", "Power: resources", "Face", "Security: personal", "Security: societal", "Tradition",
    "Conformity: rules", "Conformity: interpersonal", "Humility", "Benevolence: caring",
    "Benevolence: dependability", "Universalism: concern", "Universalism: nature", "Universalism: tolerance"
]
coarse_values = [
    "Self-direction", "Stimulation", "Hedonism", "Achievement", "Power", "Face", "Security", "Tradition", "Conformity",
    "Humility", "Benevolence", "Universalism"
]
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


def get_gpu_memory():
    """
    Get the available memory for each GPU in MB.

    Based on https://stackoverflow.com/a/59571639 .

    Returns:
    - list: The memory per GPU in MB (empty list if none are available)
    """
    import subprocess
    try:
        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        command_output = subprocess.check_output(command.split()).decode('ascii')
        memory_free_info = command_output.split('\n')[:-1][1:]
        memory_free_values = [
            int(x.split()[0]) for i, x in enumerate(memory_free_info)
        ]
        return memory_free_values
    except FileNotFoundError:
        return []


def combine_attained_and_constrained(predictions, mode=sum):
    """
    Combines the attained and constrained scores of each value.

    Parameters:
    - predictions (list): The predictions of the ValueClassifier
    - mode: Function to combine the two scores (sum by default)

    Returns:
    - list: The predictions with scores combined

    Raises:
        KeyError: If the input does not contain scores for all detailed values, e.g., as combine_detailed_values has been
        called beforehand
    """
    for prediction in predictions:
        converted_prediction = {
            "Text-ID": prediction["Text-ID"],
            "Sentence-ID": prediction["Sentence-ID"],
            "Text": prediction["Text"],
            "Language": prediction["Language"]
        }
        target_labels = values
        for target_label in target_labels:
            input_prediction = [
                float(prediction[target_label + " attained"]),
                float(prediction[target_label + " constrained"])
            ]
            converted_prediction[target_label] = mode(input_prediction)
        yield converted_prediction


def combine_detailed_values(predictions, mode=max):
    """
    Combines the detailed values (e.g., 'Universalism: concern', 'Universalism: nature', and 'Universalism: tolerance')
    to one per "coarse" value.

    Parameters:
    - predictions (list): The predictions of the ValueClassifier (split in attained and constrained or not)
    - mode: Function to combine the scores (max by default)

    Returns:
    - list: The predictions with scores combined
    """
    for prediction in predictions:
        converted_prediction = {
            "Text-ID": prediction["Text-ID"],
            "Sentence-ID": prediction["Sentence-ID"],
            "Text": prediction["Text"],
            "Language": prediction["Language"]
        }
        target_labels = coarse_values
        if "Achievement attained" in prediction.keys():
            for target_label in target_labels:
                input_prediction = [
                    float(prediction[label]) for label in prediction.keys()
                    if label.startswith(target_label) and label.endswith(" attained")
                ]
                converted_prediction[target_label + " attained"] = mode(input_prediction)
                input_prediction = [
                    float(prediction[label]) for label in prediction.keys()
                    if label.startswith(target_label) and label.endswith(" constrained")
                ]
                converted_prediction[target_label + " constrained"] = mode(input_prediction)
            yield converted_prediction
        else:
            for target_label in target_labels:
                input_prediction = [
                    float(prediction[label]) for label in prediction.keys()
                    if label.startswith(target_label)
                ]
                converted_prediction[target_label] = mode(input_prediction)
            yield converted_prediction


def write_predictions(predictions, output_file=sys.stdout):
    """
    Writes the predictions to a TSV file (tab-separated-values).

    Parameters:
    - predictions (list): The predictions of the ValueClassifier (split in attained and constrained or not, detailed or not)
    - output_file: File name or file handle to write to
    """
    if isinstance(output_file, str):
        with open(output_file, "w", newline="") as output_file_handle:
            write_predictions(predictions, output_file=output_file_handle)
    else:
        writer = None
        for prediction in predictions:
            if writer is None:
                fieldnames = ["Text-ID", "Sentence-ID", "Text", "Language"]
                if "Self-direction: thought attained" in prediction.keys():
                    fieldnames = fieldnames + labels
                elif "Self-direction: thought" in prediction.keys():
                    fieldnames = fieldnames + values
                elif "Self-direction attained" in prediction.keys():
                    fieldnames = fieldnames + sum(
                        [[value + " attained", value + " constrained"] for value in coarse_values],
                        []
                    )
                else:
                    fieldnames = fieldnames + coarse_values
                writer = csv.DictWriter(output_file, fieldnames=fieldnames, delimiter="\t")
                writer.writeheader()
            writer.writerow(prediction)


class ValueClassifier(object):
    """
    The classifier of team Hierocles of Alexandria, winning the ValueEval'24 shared task.
    """

    def __init__(self, use_cpu=False, **kwargs):
        """
        Creates the classifier.

        Parameters:
        - use_cpu (bool): Whether to force using the CPU even if a GPU is available
        - kwargs: Arguments passed on to the model; if "quantization_config" is not set, try to autodetect which quantization
          to use based on available GPU memory; set "quantization_config" to "None" to force use the full model
        """
        cpu = use_cpu or not torch.cuda.is_available()
        if not cpu and "quantization_config" not in kwargs:
            gpu_memory_gigabyte = get_gpu_memory()[0] / 1024
            if gpu_memory_gigabyte >= 20:
                print(f"Detected GPU with at least 20GB memory (namely {gpu_memory_gigabyte}GB): taking full model")
            elif gpu_memory_gigabyte >= 10:
                print(f"Detected GPU with between 10GB and 20GB of memory (namely {gpu_memory_gigabyte}GB): taking 8bit model")
                kwargs["quantization_config"] = transformers.BitsAndBytesConfig(load_in_8bit=True)
            elif gpu_memory_gigabyte >= 5:
                print(f"Detected GPU with between 5GB and 10GB of memory (namely {gpu_memory_gigabyte}GB): taking 4bit model")
                kwargs["quantization_config"] = transformers.BitsAndBytesConfig(load_in_4bit=True)
            else:
                print(f"Detected GPU with less than 5GB of memory (namely {gpu_memory_gigabyte}GB): trying CPU")
                cpu = True

        self._device = torch.device("cpu" if cpu else "cuda")
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self._model = MultiHead_MultiLabel_XL.from_pretrained(
            model_name, problem_type="multi_label_classification", **kwargs
        )
        if self._model.device.type != self._device.type:
            self._model.to(self._device)  # type: ignore

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
            elif isinstance(entry["Text-ID"], int):
                if str(entry["Text-ID"]) != validated["Text-ID"] and "Sentence-ID" not in entry.keys():
                    validated["Sentence-ID"] = 1
                validated["Text-ID"] = str(entry["Text-ID"])
            else:
                raise ValueError(f"'Text-ID' of entry {i} is not a str or int but {type(entry['Text-ID'])}")

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
        previous_sentences = []
        last_text_id = None
        for entry in text_data:
            if last_text_id != entry["Text-ID"]:
                previous_sentences = []
            text = entry["Text"]
            text_in_context = "".join(previous_sentences[-look_back:] + [text])
            input = {"language": [entry["Language"]]}
            input.update(self._tokenizer(
                text_in_context, padding="max_length", max_length=512, truncation=True, return_tensors="pt"
            ))
            input["input_ids"] = input["input_ids"].to(self._device)  # type: ignore
            input["attention_mask"] = input["attention_mask"].to(self._device)  # type: ignore
            output = self._model(**input)
            predictions = output.logits.detach().cpu().numpy()
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

    def predict(
            self,
            data,
            attained_and_constrained=True,
            detailed_values=True
            ) -> Generator[dict, None, None] | dict | pandas.DataFrame:
        """
        Applies this classifier to the input data.

        Parameters:
        - data: The data to predict for, either a single sentence or dictionary, a list or generator of these, or a Pandas
          data frame with corresponging columns:
          Text-ID (consecutive sentences with the same ID are treated as belonging to the same text; optional: assumed to be
          the same as the previous sentence if it does not exist),
          Sentence-ID (a number, ignored; optional: assumed to be one plus the previous one in case the Text-ID is the same
          and 1 if not),
          Text (the actual text of the sentence; if only a str is provided, the str is taken as Text), and
          Language (two-letter language code, one of 'EN' (default), 'EL', 'DE', 'TR', 'FR', 'BG', 'HE', 'IT', or 'NL')
        - attained_and_constrained (bool): Whether to combinee the attained and constrained scores of each value.
        - detailed_values (bool): Whether to combines the detailed values (e.g., 'Universalism: concern', 'Universalism:
          nature', and 'Universalism: tolerance') to one per "coarse" value.

        Returns:
        - The predictions, either a single text/dictionary, generator, or data frame, matching the input data; the output
          matches the input (with defaults put in), but additionally contains the predictions with the scores as values
        """
        if isinstance(data, str):
            return next(self.predict(
                [data],
                attained_and_constrained=attained_and_constrained,
                detailed_values=detailed_values))  # type: ignore
        if isinstance(data, pandas.DataFrame):
            return pandas.DataFrame.from_records(
                self.predict(
                    data.to_dict("records"),
                    attained_and_constrained=attained_and_constrained,
                    detailed_values=detailed_values
                )
            )
        else:
            predictions = self._predict_for_text(self._validate_input_data(data))
            if not attained_and_constrained:
                predictions = combine_attained_and_constrained(predictions)
            if not detailed_values:
                predictions = combine_detailed_values(predictions)

            return predictions

    def predict_to_tsv(self, data, output_file=sys.stdout, **kwargs) -> None:
        """
        Applies this classifier to the input data and writes the predictions to a file.

        Parameters:
        - data: The data to predict for, either a single sentence or dictionary, a list or generator of these, or a Pandas
          data frame with corresponging columns:
          Text-ID (consecutive sentences with the same ID are treated as belonging to the same text; optional: assumed to be
          the same as the previous sentence if it does not exist),
          Sentence-ID (a number, ignored; optional: assumed to be one plus the previous one in case the Text-ID is the same
          and 1 if not),
          Text (the actual text of the sentence; if only a str is provided, the str is taken as Text), and
          Language (two-letter language code, one of 'EN' (default), 'EL', 'DE', 'TR', 'FR', 'BG', 'HE', 'IT', or 'NL')
        - output_file: File name or file handle to write to
        - attained_and_constrained (bool): Whether to combinee the attained and constrained scores of each value.
        - detailed_values (bool): Whether to combines the detailed values (e.g., 'Universalism: concern', 'Universalism:
          nature', and 'Universalism: tolerance') to one per "coarse" value.
        """
        write_predictions(self.predict(data, **kwargs), output_file)
