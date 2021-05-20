import math

import torch
import torch.utils.checkpoint as cp
import torch.nn as nn
from .bert_utils import find_pruneable_heads_and_indices,prune_linear_layer,gelu,swish,gelu_new
import logging
from ..builder import LANGUAGE

from commons.runner import load_checkpoint


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new, "mish": mish}
BertLayerNorm = torch.nn.LayerNorm
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.get('vocab_size'), config.get('hidden_size'), padding_idx=config.get('pad_token_id'))
        self.position_embeddings = nn.Embedding(config.get('max_position_embeddings'), config.get('hidden_size'))
        self.token_type_embeddings = nn.Embedding(config.get('type_vocab_size'), config.get('hidden_size'))

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.get('hidden_size'), eps=config.get('layer_norm_eps'))
        self.dropout = nn.Dropout(config.get('hidden_dropout_prob'))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        if config.get('hidden_size') % config.get('num_attention_heads') != 0 and config.get("embedding_size") is None:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.get('hidden_size'), config.get('num_attention_heads'))
            )

        self.num_attention_heads = config.get('num_attention_heads')
        self.attention_head_size = int(config.get('hidden_size')/config.get('num_attention_heads'))
        self.all_head_size = self.num_attention_heads*self.attention_head_size

        self.query = nn.Linear(config.get('hidden_size'), self.all_head_size)
        self.key = nn.Linear(config.get('hidden_size'), self.all_head_size)
        self.value = nn.Linear(config.get('hidden_size'), self.all_head_size)

        self.dropout = nn.Dropout(config.get('attention_probs_dropout_prob'))

    def transpose_for_scores(self,x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0,2,1,3)

    def forward(self,
                hidden_states,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=False,
                ):
        mixed_query_layer = self.query(hidden_states)
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.get('hidden_size'), config.get('hidden_size'))
        self.LayerNorm = BertLayerNorm(config.get('hidden_size'), eps=config.get('layer_norm_eps'))
        self.dropout = nn.Dropout(config.get('hidden_dropout_prob'))

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.get('hidden_size'), config.get('intermediate_size'))
        if isinstance(config.get('hidden_act'), str):
            self.intermediate_act_fn = ACT2FN[config.get('hidden_act')]
        else:
            self.intermediate_act_fn = config.get('hidden_act')

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.get('intermediate_size'), config.get('hidden_size'))
        self.LayerNorm = BertLayerNorm(config.get('hidden_size'), eps=config.get('layer_norm_eps'))
        self.dropout = nn.Dropout(config.get('hidden_dropout_prob'))

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self,config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.get('is_decoder',False)
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self,
                hidden_states,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=False):
        self_attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask, output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs

class BertEncoder(nn.Module):
    def __init__(self,config):
        super(BertEncoder, self).__init__()
        self.config = config
        self.layer = nn.ModuleList(BertLayer(config) for _ in range(config.get('num_hidden_layers')))

    def forward(self,
                hidden_states,
                attention_mask=None,
                head_mask = None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_tuple=False,
                ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i,layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.config.get("with_cp",False):
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = cp.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if return_tuple:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return hidden_states

class Bert(nn.Module):
    def __init__(self,config):
        super(Bert, self).__init__()
        self.config = config
        self.embeddings = BertEmbeddings(self.config)
        self.encoder = BertEncoder(self.config)

    def get_extended_attention_mask(self, attention_mask, input_shape, device):
        """Makes broadcastable attention mask and causal mask so that future and maked tokens are ignored.
        Arguments:
            attention_mask: torch.Tensor with 1 indicating tokens to ATTEND to
            input_shape: tuple, shape of input_ids
            device: torch.Device, usually self.device
        Returns:
            torch.Tensor with dtype of attention_mask.dtype
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.get("is_decoder",False):
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=extended_attention_mask.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked = False):
        """
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        attention_probs has shape bsz x n_heads x N x N
        Arguments:
            head_mask: torch.Tensor or None: has shape [num_heads] or [num_hidden_layers x num_heads]
            num_hidden_layers: int
        Returns:
             Tensor of shape shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
             or list with [None] for each layer
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)  # switch to fload if need + fp16 compatibility
        return head_mask

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=None,
                output_hidden_states=None,
                return_tuple=None,
                ):
        output_attentions = output_attentions if output_attentions is not None else self.config.get('output_attentions')
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.get('output_hidden_states')
        )
        return_tuple = return_tuple if return_tuple is not None else self.config.get('use_return_tuple')

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape,
                                                                                 device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.get('is_decoder', False) and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.get('num_hidden_layers'))

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_tuple=return_tuple,
        )
        # print("encoder_outputs.type", type(encoder_outputs))
        # print("encoder_outputs",encoder_outputs)
        # first_token_tensor = encoder_outputs[:, 0]
        return encoder_outputs

class VLBert(Bert):
    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                visual_tokens=None,
                visual_attention_mask=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=None,
                output_hidden_states=None,
                return_tuple=None,
                ):
        output_attentions = output_attentions if output_attentions is not None else self.config.get('output_attentions')
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.get('output_hidden_states')
        )
        return_tuple = return_tuple if return_tuple is not None else self.config.get('use_return_tuple')

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # support visual
        if visual_attention_mask is None:
            visual_attention_mask = torch.one((visual_tokens.size(0),visual_tokens.size(1)),device=device)

        fusion_attention_mask = torch.cat([attention_mask,visual_attention_mask],dim=1)
        fusion_shape =(input_shape[0],input_shape[1]+visual_tokens.size(1))


        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(fusion_attention_mask, fusion_shape,
                                                                                 device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.get('is_decoder', False) and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.get('num_hidden_layers'))

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds
        )

        vl_feature = torch.cat([embedding_output,visual_tokens],dim=1)
        encoder_outputs = self.encoder(
            vl_feature,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_tuple=return_tuple,
        )
        # print("encoder_outputs.type", type(encoder_outputs))
        # print("encoder_outputs",encoder_outputs)
        # first_token_tensor = encoder_outputs[:, 0]
        return encoder_outputs


@LANGUAGE.register_module
class BertModel(nn.Module):
    def __init__(self,
                 frozen_stages=-1,
                 return_all=False,
                 vocab_size=30522,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 pad_token_id=0,
                 with_cp=False,):
        super(BertModel, self).__init__()
        self.config={}
        self.config['vocab_size']=vocab_size
        self.config['hidden_size']=hidden_size
        self.config['num_hidden_layers']=num_hidden_layers
        self.config['num_attention_heads']=num_attention_heads
        self.config['intermediate_size']=intermediate_size
        self.config['hidden_act']=hidden_act
        self.config['hidden_dropout_prob']=hidden_dropout_prob
        self.config['attention_probs_dropout_prob']=attention_probs_dropout_prob
        self.config['max_position_embeddings']=max_position_embeddings
        self.config['type_vocab_size']=type_vocab_size
        self.config['initializer_range']=initializer_range
        self.config['layer_norm_eps']=layer_norm_eps
        self.config['pad_token_id']=pad_token_id
        self.config['with_cp']=with_cp

        self.frozen_stages = frozen_stages
        self.return_all = return_all

        self.bert = Bert(self.config)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages>=0:
            self.bert.embeddings.eval()
            for param in self.bert.embeddings.parameters():
                param.requires_grad=False

            for i in range(1, self.frozen_stages + 1):
                m = getattr(self.bert.encoder.layer, f'{i-1}')
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False


    def init_weights(self,pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger(__name__)
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m,(nn.Linear,nn.Embedding)):
                    m.weight.data.normal_(mean=0.0,std=self.config.get('initializer_range'))
                elif isinstance(m,BertLayerNorm):
                    m.bias.data.zero_()
                    m.weight.data.fill_(1.0)
                if isinstance(m,nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
        else:
            raise TypeError('pretrained must be a str or None')


    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=None,
                output_hidden_states=None,
                return_tuple=None,
                ):
        out = self.bert(input_ids,attention_mask,token_type_ids,position_ids,head_mask,
                        inputs_embeds,encoder_hidden_states,encoder_attention_mask,
                        output_attentions,output_hidden_states,return_tuple)
        if self.return_all:
            return out
        first_token_tensor = out[:, 0]
        return first_token_tensor


@LANGUAGE.register_module
class VLBertModel(nn.Module):
    def __init__(self,
                 frozen_stages=-1,
                 return_all=True,
                 vocab_size=30522,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 pad_token_id=0,
                 with_cp=False,):
        super(VLBertModel, self).__init__()
        self.config={}
        self.config['vocab_size']=vocab_size
        self.config['hidden_size']=hidden_size
        self.config['num_hidden_layers']=num_hidden_layers
        self.config['num_attention_heads']=num_attention_heads
        self.config['intermediate_size']=intermediate_size
        self.config['hidden_act']=hidden_act
        self.config['hidden_dropout_prob']=hidden_dropout_prob
        self.config['attention_probs_dropout_prob']=attention_probs_dropout_prob
        self.config['max_position_embeddings']=max_position_embeddings
        self.config['type_vocab_size']=type_vocab_size
        self.config['initializer_range']=initializer_range
        self.config['layer_norm_eps']=layer_norm_eps
        self.config['pad_token_id']=pad_token_id
        self.config['with_cp']=with_cp

        self.frozen_stages = frozen_stages
        self.return_all = return_all

        self.bert = VLBert(self.config)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages>=0:
            self.bert.embeddings.eval()
            for param in self.bert.embeddings.parameters():
                param.requires_grad=False

            for i in range(1, self.frozen_stages + 1):
                m = getattr(self.bert.encoder.layer, f'{i-1}')
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False


    def init_weights(self,pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger(__name__)
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m,(nn.Linear,nn.Embedding)):
                    m.weight.data.normal_(mean=0.0,std=self.config.get('initializer_range'))
                elif isinstance(m,BertLayerNorm):
                    m.bias.data.zero_()
                    m.weight.data.fill_(1.0)
                if isinstance(m,nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
        else:
            raise TypeError('pretrained must be a str or None')


    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                visual_tokens=None,
                visual_attention_mask=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=None,
                output_hidden_states=None,
                return_tuple=None,
                ):
        out = self.bert(input_ids,attention_mask,token_type_ids,position_ids,visual_tokens,visual_attention_mask,head_mask,
                        inputs_embeds,encoder_hidden_states,encoder_attention_mask,
                        output_attentions,output_hidden_states,return_tuple)
        if self.return_all:
            return out
        first_token_tensor = out[:, 0]
        return first_token_tensor



class BertEncoderFunction(nn.Module):
    def __init__(self,
                 frozen_stages=-1,
                 return_all=True,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 initializer_range=0.02,
                 layer_norm_eps=1e-5,
                 pad_token_id=0,
                 with_cp=False, ):
        super(BertEncoderFunction, self).__init__()
        self.config={}
        self.config['hidden_size']=hidden_size
        self.config['num_hidden_layers']=num_hidden_layers
        self.config['num_attention_heads']=num_attention_heads
        self.config['intermediate_size']=intermediate_size
        self.config['hidden_act']=hidden_act
        self.config['hidden_dropout_prob']=hidden_dropout_prob
        self.config['attention_probs_dropout_prob']=attention_probs_dropout_prob
        self.config['initializer_range']=initializer_range
        self.config['layer_norm_eps']=layer_norm_eps
        self.config['pad_token_id']=pad_token_id
        self.config['with_cp']=with_cp

        self.frozen_stages = frozen_stages
        self.return_all = return_all

        self.encoder =  BertEncoder(self.config)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages>=0:
            for i in range(0, self.frozen_stages ):
                m = getattr(self.encoder.encoder.layer, f'{i}')
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False


    def init_weights(self,pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger(__name__)
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, (nn.Linear, nn.Embedding)):
                    m.weight.data.normal_(mean=0.0, std=self.config.get('initializer_range'))
                elif isinstance(m, BertLayerNorm):
                    m.bias.data.zero_()
                    m.weight.data.fill_(1.0)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
        else:
            raise TypeError('pretrained must be a str or None')

    def get_extended_attention_mask(self, attention_mask, input_shape, device):
        """Makes broadcastable attention mask and causal mask so that future and maked tokens are ignored.
        Arguments:
            attention_mask: torch.Tensor with 1 indicating tokens to ATTEND to
            input_shape: tuple, shape of input_ids
            device: torch.Device, usually self.device
        Returns:
            torch.Tensor with dtype of attention_mask.dtype
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.get("is_decoder", False):
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=extended_attention_mask.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        """
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        attention_probs has shape bsz x n_heads x N x N
        Arguments:
            head_mask: torch.Tensor or None: has shape [num_heads] or [num_hidden_layers x num_heads]
            num_hidden_layers: int
        Returns:
             Tensor of shape shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
             or list with [None] for each layer
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask
    def forward(self,
                hidden_states,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                ):

        head_mask = self.get_head_mask(head_mask, self.config.get('num_hidden_layers'))
        input_shape = hidden_states.size()[:-1]
        device = hidden_states.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)
        out = self.encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask)
        if not self.return_all:
            out = out[:, 0]

        return out