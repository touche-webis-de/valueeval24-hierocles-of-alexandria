import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, Tuple, Union
from transformers.models.xlm_roberta_xl.modeling_xlm_roberta_xl import (
    XLMRobertaXLPreTrainedModel, XLMRobertaXLModel, XLMRobertaXLLayer
)


lang_dict = {'EN': 0,
             'EL': 1,
             'DE': 2,
             'TR': 3,
             'FR': 4,
             'BG': 5,
             'HE': 6,
             'IT': 7,
             'NL': 8}

lang_default = lang_dict["EN"]
id2lang_dict = {v: k for k, v in lang_dict.items()}


class XLMRobertaXLClassificationHeadCustom(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()

        self.extra_transformer_layer_1 = XLMRobertaXLLayer(config)
        # self.extra_transformer_layer_2 = XLMRobertaXLLayer(config)
        # self.extra_transformer_layer_3 = XLMRobertaXLLayer(config)
        # self.extra_transformer_layer_4 = RobertaLayer(config)
        # self.extra_transformer_layer_5 = RobertaLayer(config)

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features    # [:, 0, :]  # take <s> token (equiv. to [CLS])

        x = self.extra_transformer_layer_1(x)
        # x = self.extra_transformer_layer_2(x[0])
        # x = self.extra_transformer_layer_3(x[0])
        # x = self.extra_transformer_layer_4(x[0])
        # x = self.extra_transformer_layer_5(x[0])
        x = x[0][:, 0, :]

        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class MultiHead_MultiLabel_XL(XLMRobertaXLPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = XLMRobertaXLModel(config, add_pooling_layer=True)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )

        self.dropout = nn.Dropout(classifier_dropout)
        self.language_heads = nn.ModuleDict({
            'EN': XLMRobertaXLClassificationHeadCustom(config),
            'EL': XLMRobertaXLClassificationHeadCustom(config),
            'DE': XLMRobertaXLClassificationHeadCustom(config),
            'TR': XLMRobertaXLClassificationHeadCustom(config),
            'FR': XLMRobertaXLClassificationHeadCustom(config),
            'BG': XLMRobertaXLClassificationHeadCustom(config),
            'HE': XLMRobertaXLClassificationHeadCustom(config),
            'IT': XLMRobertaXLClassificationHeadCustom(config),
            'NL': XLMRobertaXLClassificationHeadCustom(config),
        })

        # self.classifier = RobertaClassificationHead(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        language: list[int] = []
    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        batch_sz = len(language)
        heads = [self.language_heads[id2lang_dict[int(language[i])]] for i in range(batch_sz)]

        split_tensors = torch.split(outputs.last_hidden_state, split_size_or_sections=1, dim=0)

        if batch_sz == 1:
            logits = heads[0](split_tensors[0])
        else:
            logits = torch.cat([heads[i](split_tensors[i]) for i in range(batch_sz)], dim=0)

        loss = None
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,  # type: ignore
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
