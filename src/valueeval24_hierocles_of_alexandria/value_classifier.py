import numpy
import pyvalues
import transformers
from typing import Generator, Iterable, Tuple
import torch
from pydantic_extra_types.language_code import LanguageAlpha2

from .multi_head_model import MultiHead_MultiLabel_XL, lang_dict

labels = pyvalues.RefinedValuesWithAttainment.names()
id2label = {idx: label for idx, label in enumerate(labels)}
label_thresholds = [
    0.10, 0.30, 0.25, 0.25, 0.25, 0.25, 0.35, 0.30, 0.35, 0.25,
    0.35, 0.15, 0.20, 0.25, 0.10, 0.20, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.25, 0.30, 0.15, 0.10, 0.15, 0.10, 0.01, 0.30, 0.10,
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


class ValueEval24Classifier(pyvalues.RefinedValuesWithAttainmentClassifier):
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

    def classify_segments_for_refined_values_with_attainment(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 | str = LanguageAlpha2("en")
    ) -> Generator[pyvalues.RefinedValuesWithAttainment, None, None]:
        if language.upper() not in lang_dict.keys():
            self._raise_unsupported_language(language)

        previous_segments = []
        for segment in segments:
            segment_in_context = "".join(previous_segments[-look_back:] + [segment])
            input = {"language": [lang_dict[language.upper()]]}
            input.update(self._tokenizer(
                segment_in_context,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt"
            ))
            input["input_ids"] = input["input_ids"].to(self._device)  # type: ignore
            input["attention_mask"] = input["attention_mask"].to(self._device)  # type: ignore
            output = self._model(**input)
            predictions = output.logits.detach().cpu().numpy()
            confidence_values = self._thresholds(predictions[0])
            yield pyvalues.RefinedValuesWithAttainment.from_list(confidence_values, cap_at_one=True)

            # prepare for next segment
            text_labels = [f"<{id2label[j]}>" for j, value in enumerate(confidence_values) if value >= 0.5]
            if len(text_labels) > 0:
                previous_segments.append(f"{segment} <{' '.join(text_labels)}> </s> ")
            else:
                previous_segments.append(f"{segment} <NONE> </s> ")
