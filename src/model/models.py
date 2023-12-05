import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h):
        super().__init__()
        self.d_model, self.h = d_model, h
        self.d_head = d_model // h

        self.weight_q = nn.Linear(d_model, d_model)
        self.weight_k = nn.Linear(d_model, d_model)
        self.weight_v = nn.Linear(d_model, d_model)

        self.weight_o = nn.Linear(d_model, d_model)

    def _get_multihead_vectors(self, x, weight):
        vectors = weight(x)
        seq_len = vectors.shape[1]

        vectors = vectors.view(-1, self.h, seq_len, self.d_head)
        return vectors

    def forward(self, query, key, value, mask):
        q = self._get_multihead_vectors(query, self.weight_q)
        k = self._get_multihead_vectors(key, self.weight_k)
        v = self._get_multihead_vectors(value, self.weight_v)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (
            self.d_head**0.5
        )  # (batch_size, h, seq_len, seq_len)
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_dists = F.softmax(attention_scores, dim=-1)
        attention_values = torch.matmul(attention_dists, v)

        attention_values = attention_values.view(-1, q.shape[-2], self.d_model)
        output = self.weight_o(attention_values)

        return output


if __name__ == "__main__":
    batch_size = 8
    seq_len = 256
    d_model = 512
    h = 8
    d_q = d_k = d_v = d_model // h

    x = torch.randn(batch_size, seq_len, d_model)

    model = MultiHeadAttention(d_model=d_model, h=h)
    print(model(x).shape)
