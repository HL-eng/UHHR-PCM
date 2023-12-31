import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import random
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, embedding_dims):
        super(Attention, self).__init__()
        self.embed_dim = embedding_dims

        self.att1 = nn.Linear(2 * self.embed_dim, 256)
        self.att2 = nn.Linear(2 * self.embed_dim, 1)
        self.softmax = nn.Softmax(0)

    def forward(self, node1, u_rep, num_neighs):
        uv_reps = u_rep.repeat(num_neighs, 1)
        x = torch.cat((node1, uv_reps), 1)
        # print(node1.size())
        # print(uv_reps.size())
        # print(x.size())
        x = F.relu(self.att1(x))
        x = F.dropout(x, training=self.training)
        x = self.att2(x)
        att = F.softmax(x, dim=0)
        return att
