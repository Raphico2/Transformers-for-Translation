import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from HyperParameters import hyper_parameters, model_repository


class GermanEnglishTranslatorWithRootModifers(nn.Module):
    def __init__(self, device, model_base_name=model_repository["MODEL_T5"]):
        super().__init__()
        self.Device = device
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_base_name).to(self.Device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_base_name)


    def forward(self, input_batch, input_padding_mask, target_batch, attention_tgt_mask):

        labels = torch.where(target_batch == 0, torch.tensor(-100).to(self.Device), target_batch)
        labels = labels.to(self.Device)

        output = self.model.forward(
                    input_ids=input_batch,
                    attention_mask=input_padding_mask,
                    decoder_attention_mask=attention_tgt_mask,
                    labels=labels,
        )

        translated_tokens_id = torch.argmax(output.logits, dim=-1)
        loss = output.loss

        return translated_tokens_id, loss

    def translate(self, german_batch, source_attention_mask, max_length=400, num_beams=4):

        translated_tokens = self.model.generate(
                   input_ids=german_batch,
                   attention_mask=source_attention_mask,
                   max_length=max_length,
                   num_beams=num_beams,
                   eos_token_id=self.tokenizer.eos_token_id,
                   early_stopping=True,
                   )
        return translated_tokens
