# Start with some standard imports.
import torch.nn as nn
from torch.nn.functional import relu


class Adapter(nn.Module):
    def __init__(self):
        super(Adapter, self).__init__()
        self.layer = nn.Sequential(nn.Linear(128, 128))

    def forward(self, x):
        x = self.layer(x)
        return x
