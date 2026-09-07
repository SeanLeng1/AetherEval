"""Missing GPT-2 classification architecture; native generation is untouched."""

import torch
from sglang.srt.layers.pooler import Pooler, PoolingType, score_and_pool
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.gpt2 import GPT2LMHeadModel, GPT2Model
from torch import nn

from .gpt2_context import install_context_patch

install_context_patch()


class GPT2ForSequenceClassification(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.transformer = GPT2Model(config, quant_config, prefix=f"{prefix}.transformer".lstrip("."))
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=False)

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, get_embedding=True, **kwargs):
        hidden = self.transformer(input_ids, positions, forward_batch)
        return score_and_pool(self.score, self.pooler, hidden, forward_batch, input_ids)

    def load_weights(self, weights):
        def backbone():
            for name, value in weights:
                if name == "score.weight":
                    default_weight_loader(self.score.weight, value)
                else:
                    yield name, value

        GPT2LMHeadModel.load_weights(self, backbone())


EntryClass = GPT2ForSequenceClassification
