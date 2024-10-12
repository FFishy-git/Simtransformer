from .model_base import TransformerEncoder, ReadIn, ReadOut, nnModule
from .utils import EasyDict
from torch import Tensor
from typing import Optional

class GPT2Standard(nnModule):
    def __init__(self, config: EasyDict):
        super().__init__()
        self.readin = ReadIn(config.vocab_size, config.hidden_size)
        self.encoder = TransformerEncoder(config)
        self.readout = ReadOut(config.hidden_size, config.vocab_size)

    def forward(self, input_token_ids: Tensor, mask: Optional[Tensor] = None):
        x = self.readin(input_token_ids)
        x = self.encoder(x, mask)
        output = self.readout(x)
        return output