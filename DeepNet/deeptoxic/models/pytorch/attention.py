import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class DotAttention(nn.Module):

    def __init__(self, hidden_size, require_mask=False):
        super(DotAttention, self).__init__()

        self.require_mask = require_mask
        self.hidden_size = hidden_size
        # attn_vector is unique for sequence
        self.attn_vector = nn.Parameter(
            torch.Tensor(1, hidden_size), requires_grad=True)
        init.xavier_uniform(self.attn_vector.data)

    def get_mask(self):
        pass

    def forward(self, inputs, lengths=None):
        batch_size, max_len = inputs.size()[:2]

        # apply attention layer
        weights = torch.bmm(inputs,
                            self.attn_vector  # (1, hidden_size)
                            .unsqueeze(0)  # (1, 1, hidden_size)
                            .transpose(2, 1)  # (1, hidden_size, 1)
                            .repeat(batch_size, 1, 1))  # (batch_size, hidden_size, 1))  , same as broadcast in numpy

        # weights size : batch_size * sequence_length * 1
        attn_energies = F.softmax(F.relu(weights.squeeze()), dim=1)

        # create mask based on the sentence lengths
        if self.require_mask and lengths is not None:
            idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0).cuda()
            mask = torch.tensor((idxes < lengths.data.unsqueeze(1)).float())
            # apply mask and re-normalize attention scores (weights)
            masked = attn_energies * mask
            attn_energies = masked

        _sums = attn_energies.sum(dim=1).unsqueeze(1).expand_as(attn_energies)  # sums per row
        attn_weights = attn_energies / _sums
        weighted = torch.mul(inputs, attn_weights.unsqueeze(-1).expand_as(inputs))
        representations = weighted.sum(1).squeeze()

        return representations, attn_weights
