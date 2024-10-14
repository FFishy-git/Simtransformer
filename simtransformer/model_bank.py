from .model_base import TransformerEncoder, ReadIn, ReadOut, nnModule
from .utils import EasyDict
from torch import Tensor
from typing import Optional
import torch.nn as nn

class GPT2Standard(nnModule):
    def __init__(self, config: EasyDict, input_size: int, output_size: int):
        super().__init__()
        self.readin = ReadIn(input_size, config.hidden_size)
        self.encoder = TransformerEncoder(config)
        self.readout = ReadOut(config.hidden_size, output_size)

    def forward(self, input_token_ids: Tensor, mask: Optional[Tensor] = None):
        x = self.readin(input_token_ids)
        x = self.encoder(x, mask)
        output = self.readout(x)
        return output

class GPT2LinearReg(nnModule):
    def __init__(self, config: EasyDict, input_size: int, output_size: int):
        super().__init__()
        self.readin = nn.Linear(input_size, config.hidden_size)
        self.encoder = TransformerEncoder(config)
        self.readout = nn.Linear(config.hidden_size, output_size)

    def forward(self, input_tokens: Tensor, mask: Optional[Tensor] = None):
        x = self.readin(input_tokens)
        x = self.encoder(x, mask)
        output = self.readout(x)
        return output