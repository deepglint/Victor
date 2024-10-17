#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..croc_arch import CrocMetaModel, CrocMetaForCausalLM
import torch.nn.init as init
import math
import random
from transformers import BertConfig, BertLayer


class CrocConfig(LlamaConfig):
    model_type = "croc_llama"


class CrocLlamaModel(CrocMetaModel, LlamaModel):
    config_class = CrocConfig

    def __init__(self, config: LlamaConfig):
        super(CrocLlamaModel, self).__init__(config)

class CustomTransformer(nn.Module):
    def __init__(self, num_layers=1, input_dim=4096, output_dim=4096, max_position_embeddings=576):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, input_dim)
        self.layers = nn.ModuleList([
            BertLayer(BertConfig(hidden_size=input_dim, intermediate_size=input_dim*4, num_attention_heads=input_dim//64))
            for _ in range(num_layers)
        ])
        self.resize = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        seq_length = x.size(1)
        
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        position_embeddings = self.position_embeddings(position_ids)
        x = x + position_embeddings.unsqueeze(0).expand_as(x)
        for layer in self.layers:
            x = layer(x)[0]  
        return self.resize(x)


class CrocLlamaForCausalLM(LlamaForCausalLM, CrocMetaForCausalLM):
    config_class = CrocConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = CrocLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.transform_block = CustomTransformer(1,config.hidden_size,config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.hidden_size = config.hidden_size
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                img_labels,
                select_patch,
                image_pos
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        is_croc_stage = not torch.all(select_patch == 0)
        
        if is_croc_stage:
            target_lenght = attention_mask.shape[-1]
            sequence_length = target_lenght
            dtype, device = inputs_embeds.dtype, inputs_embeds.device
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length,target_lenght),fill_value=1.0,dtype=dtype,device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask = causal_mask[None, None, :, :].expand(inputs_embeds.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, 1.0
                )
            if labels is not None:
                for i in range(labels.shape[0]):
                    ignore_index = labels.shape[1]
                    for j in range(labels.shape[1]):
                        if labels[i, j] != -100:
                            ignore_index = j
                            break
                    causal_mask[i, :, :, :ignore_index] = 0
            attention_mask = 1.0 - causal_mask
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
            if is_croc_stage:
                pre_img = self.transform_block(hidden_states[image_pos].view(hidden_states.shape[0],-1,hidden_states.shape[-1]))
            else:
                pre_img = self.transform_block(hidden_states.repeat(1, 576//hidden_states.shape[1]+1, 1)[:,:576,:])
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        im_loss = 0
        if is_croc_stage:
            if pre_img.shape[1] != 0:
                img_labels_flat = img_labels[image_pos].float()
                pre_img_flat = pre_img.view(-1,self.hidden_size).float()
                select_patch_flat = select_patch[image_pos]
                
                im_loss = nn.SmoothL1Loss(beta=0.01)(img_labels_flat,pre_img_flat)*10
        else:
            img_labels_flat = pre_img.view(-1,self.hidden_size)[1].float().detach()
            im_loss += nn.SmoothL1Loss(beta=0.01)(img_labels_flat,pre_img.view(-1,self.hidden_size)[1].float())
        loss += im_loss
        if self.device.index == 0 and is_croc_stage:
            print(f'im_loss: {im_loss}, all_loss: {loss}')

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                img_labels,
                select_patch,
                image_pos
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            input_ids = input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels = labels,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("croc_llama", CrocConfig)
AutoModelForCausalLM.register(CrocConfig, CrocLlamaForCausalLM)
