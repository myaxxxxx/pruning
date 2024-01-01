from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.distributed import fsdp_wrap
from fairseq.models import BaseFairseqModel, register_model
from fairseq.distributed.fully_sharded_data_parallel import FullyShardedDataParallel
from fairseq.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    GradMultiply,
    GumbelVectorQuantizer,
    LayerNorm,
    MultiheadAttention,
    RelPositionalEncoding,
    SamePad,
    TransposeLast,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.conformer_layer import ConformerWav2Vec2EncoderLayer
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import buffered_arange, index_put, is_xla_tensor
import numpy as np




from fairseq.models.hubert import HubertModel
import torch 

from fairseq.models.wav2vec.wav2vec2 import (
    EXTRACTOR_MODE_CHOICES,
    MASKING_DISTRIBUTION_CHOICES,
    LAYER_TYPE_CHOICES,
    ConvFeatureExtractionModel,
    TransformerEncoder,
    TransformerSentenceEncoderWithAdapterLayer,
    TransformerSentenceEncoderLayer
)

from fairseq.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    GradMultiply,
    GumbelVectorQuantizer,
    LayerNorm,
    MultiheadAttention,
)

import torch.nn.functional as F




import math
import torch.nn.functional as F


def pad_to_multiple(x, multiple, dim=-1, value=0):
    # Inspired from https://github.com/lucidrains/local-attention/blob/master/local_attention/local_attention.py#L41
    if x is None:
        return None, 0
    tsz = x.size(dim)
    m = tsz / multiple
    remainder = math.ceil(m) * multiple - tsz
    if m.is_integer():
        return x, 0
    pad_offset = (0,) * (-1 - dim) * 2

    return F.pad(x, (*pad_offset, 0, remainder), value=value), remainder

embedding_dim = C = 512
num_attention_heads = 8


# batch: 48

# batch_first
# T x B x C
inputs = torch.rand(32, 48, 512)
self_attn = MultiheadAttention(
    embedding_dim,
    num_attention_heads,
    # dropout=attention_dropout,
    self_attention=True,

)

outputs, atten = self_attn(inputs, inputs, inputs,
                           )

print(outputs.size())
print(atten.size())

pruning_scores = atten.view(48, -1, 32).sum(dim=1)

_, idx = torch.topk(pruning_scores, 12, dim=1, largest=True, sorted=True)  # [B, left_tokens]

index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]
outputs = outputs.transpose(0, 1)

x_others = torch.gather(outputs, dim=1, index=index)  # [B, left_tokens, C

# print(pruning_scores)
print(pruning_scores.size())

# print(idx)
print(idx.size())
print(index.size())
print(outputs.size())
print(x_others.size())
            #  left_tokens = math.ceil(keep_rate * (N - 1)) N = token_num
            


class TomeTransformerSentenceEncoderLayer(TransformerSentenceEncoderLayer):
    # T x B x C
    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x
        
        N, b, c = x.size()

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
                need_weights=False,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)

            layer_result = x

            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
            )
            
            pruning_scores = atten.view(b, -1, c).sum(dim=1)
            left_tokens = math.ceil(0.9 * (N - 1)) #  N = token_num
            index = idx.unsqueeze(-1).expand(-1, -1, c)  # [B, left_tokens, C]
            x = x.transpose(0, 1)
            _, idx = torch.topk(pruning_scores, left_tokens, dim=1, largest=True, sorted=True)  # [B, left_tokens]
            x = torch.gather(x, dim=1, index=index)  # [B, left_tokens, C]
            
            ### apply tome    x_others = torch.gather(non_cls, dim=1, index=index)  # [B, left_tokens, C]    
            # pruning_scores = attention_probs.view(batch_size, -1, sz).sum(dim=1)
            #  left_tokens = math.ceil(keep_rate * (N - 1))
            
            # index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)

            layer_result = x

            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, (attn, layer_result)
    




class TomeTransformerEncoder(TransformerEncoder):
    

    def extract_features(
        self,
        x,
        padding_mask=None,
        tgt_layer=None,
        min_layer=0,
        corpus_key=None,
    ):

        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        # pad to the sequence length dimension
        x, pad_length = pad_to_multiple(
            x, self.required_seq_len_multiple, dim=-2, value=0
        )
        if pad_length > 0 and padding_mask is None:
            padding_mask = x.new_zeros((x.size(0), x.size(1)), dtype=torch.bool)
            padding_mask[:, -pad_length:] = True
        else:
            padding_mask, _ = pad_to_multiple(
                padding_mask, self.required_seq_len_multiple, dim=-1, value=True
            )
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        r = None

        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random() if self.layerdrop > 0 else 1
            if not self.training or (dropout_probability > self.layerdrop):
                layer_check = layer
                if isinstance(layer, FullyShardedDataParallel):
                    layer_check = layer.unwrapped_module
                if (corpus_key is None) or (
                    not isinstance(layer_check, (
                        TransformerSentenceEncoderWithAdapterLayer,
                        )
                    )
                ):
                    x, (z, lr) = layer(
                        x, self_attn_padding_mask=padding_mask, need_weights=False
                    )
                else:
                    x, (z, lr) = layer(
                        x,
                        self_attn_padding_mask=padding_mask,
                        need_weights=False,
                        corpus_key=corpus_key,
                    )
                if i >= min_layer:
                    layer_results.append((x, z, lr))
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # undo paddding
        if pad_length > 0:
            x = x[:, :-pad_length]

            def undo_pad(a, b, c):
                return (
                    a[:-pad_length],
                    b[:-pad_length] if b is not None else b,
                    c[:-pad_length],
                )

            layer_results = [undo_pad(*u) for u in layer_results]

        return x, layer_results




class TomeHubertModel(HubertModel):
    pass
    


    # def forward()