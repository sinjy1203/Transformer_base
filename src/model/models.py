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

        vectors = vectors.view(-1, seq_len, self.h, self.d_head)
        vectors = vectors.transpose(1, 2)
        return vectors

    def forward(self, query, key, value, mask=None):
        q = self._get_multihead_vectors(query, self.weight_q)
        k = self._get_multihead_vectors(key, self.weight_k)
        v = self._get_multihead_vectors(value, self.weight_v)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (
            self.d_head**0.5
        )  # (batch_size, h, seq_len, seq_len)
        if mask:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_dists = F.softmax(attention_scores, dim=-1)
        attention_values = torch.matmul(attention_dists, v)

        attention_values = attention_values.transpose(1, 2)
        attention_values = attention_values.contiguous().view(
            -1, query.shape[1], self.d_model
        )
        output = self.weight_o(attention_values)

        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.fn1 = nn.Linear(d_model, d_ff)
        self.fn2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = F.relu(self.fn1(x))
        x = self.fn2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, h=8, d_ff=2048, p_drop=0.1):
        super().__init__()

        self.multihead_attention = MultiHeadAttention(d_model, h)
        self.dropout1 = nn.Dropout(p_drop)
        self.layer_norm1 = nn.LayerNorm(d_model)

        self.positionwise_feedforward = PositionWiseFeedForward(d_model, d_ff)
        self.dropout2 = nn.Dropout(p_drop)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = x + self.dropout1(self.multihead_attention(x, x, x, mask))
        x = self.layer_norm1(x)

        x = x + self.dropout2(self.positionwise_feedforward(x))
        x = self.layer_norm2(x)
        return x


if __name__ == "__main__":
    batch_size = 8
    seq_len = 256
    d_model = 512
    h = 8
    d_q = d_k = d_v = d_model // h

    x = torch.randn(batch_size, seq_len, d_model)

    model = EncoderLayer()
    print(model(x, None).shape)
    # model = MultiHeadAttention(d_model=d_model, h=h)
    # print(model(x, x, x, None).shape)
