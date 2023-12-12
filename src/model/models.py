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
        if mask is not None:
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

    def forward(self, x, attention_mask=None):
        x = x + self.dropout1(self.multihead_attention(x, x, x, attention_mask))
        x = self.layer_norm1(x)

        x = x + self.dropout2(self.positionwise_feedforward(x))
        x = self.layer_norm2(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, h=8, d_ff=2048, p_drop=0.1):
        super().__init__()

        self.masked_multihead_attention = MultiHeadAttention(d_model, h)
        self.dropout1 = nn.Dropout(p_drop)
        self.layer_norm1 = nn.LayerNorm(d_model)

        self.multihead_attention = MultiHeadAttention(d_model, h)
        self.dropout2 = nn.Dropout(p_drop)
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.positionwise_feedforward = PositionWiseFeedForward(d_model, d_ff)
        self.dropout3 = nn.Dropout(p_drop)
        self.layer_norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, attention1_mask=None, attention2_mask=None):
        x = x + self.dropout1(self.masked_multihead_attention(x, x, x, attention1_mask))
        x = self.layer_norm1(x)

        x = x + self.dropout2(
            self.multihead_attention(x, encoder_output, encoder_output, attention2_mask)
        )
        x = self.layer_norm2(x)

        x = x + self.dropout3(self.positionwise_feedforward(x))
        x = self.layer_norm3(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_seq_len=256, device=torch.device("cuda")):
        super().__init__()

        self.encoding = torch.zeros(max_seq_len, d_model)
        self.encoding.requires_grad = False

        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        position.shape

        _2i = torch.arange(0, d_model, step=2, dtype=torch.float)

        self.encoding[:, 0::2] = torch.sin(position / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(position / (10000 ** (_2i / d_model)))
        self.encoding = self.encoding.unsqueeze(0).to(device)

    def forward(self, x):
        return x + self.encoding[:, : x.shape[1], :]


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model=512):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        x = self.embedding(x) * (self.d_model**0.5)
        return x


class Embedding(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        max_seq_len=256,
        p_drop=0.1,
        device=torch.device("cuda"),
    ):
        super().__init__()

        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(
            d_model, max_seq_len, device=device
        )
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        x = self.token_embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model=512, h=8, d_ff=2048, p_drop=0.1, N=6):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, h, d_ff, p_drop) for _ in range(N)]
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, attention_mask=None):
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.layer_norm(x)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model=512, h=8, d_ff=2048, p_drop=0.1, N=6):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, h, d_ff, p_drop) for _ in range(N)]
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, attention1_mask=None, attention2_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, attention1_mask, attention2_mask)
        x = self.layer_norm(x)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_de_size,
        vocab_en_size,
        d_model=512,
        max_seq_len=256,
        h=8,
        d_ff=2048,
        p_drop=0.1,
        N=6,
        device=torch.device("cuda"),
    ):
        super().__init__()
        self.embedding_src = Embedding(
            vocab_de_size, d_model, max_seq_len, p_drop, device=device
        )
        self.embedding_tgt = Embedding(
            vocab_en_size, d_model, max_seq_len, p_drop, device=device
        )
        self.encoder = Encoder(d_model, h, d_ff, p_drop, N)
        self.decoder = Decoder(d_model, h, d_ff, p_drop, N)
        self.linear = nn.Linear(d_model, vocab_en_size)

    def forward(self, src, tgt):
        encoder_attention_mask = self.pad_mask(src, src)
        decoder_attention1_mask = self.pad_mask(tgt, tgt) & self.subsequent_mask(
            tgt, tgt
        )
        decoder_attention2_mask = self.pad_mask(tgt, src)

        encoder_output = self.encoder(self.embedding_src(src), encoder_attention_mask)
        decoder_output = self.decoder(
            self.embedding_tgt(tgt),
            encoder_output,
            decoder_attention1_mask,
            decoder_attention2_mask,
        )

        output = self.linear(decoder_output)
        return output

    def pad_mask(self, q, k, pad_idx=1):
        q_seq_len, k_seq_len = q.shape[1], k.shape[1]

        q_mask = q.ne(pad_idx)[:, None, :, None]
        q_mask = q_mask.repeat(1, 1, 1, k_seq_len)

        k_mask = k.ne(pad_idx)[:, None, None, :]
        k_mask = k_mask.repeat(1, 1, q_seq_len, 1)

        mask = q_mask & k_mask
        mask.requires_grad = False
        return mask

    def subsequent_mask(self, q, k):
        q_seq_len, k_seq_len = q.shape[1], k.shape[1]

        mask = (
            torch.tril(torch.ones(q_seq_len, k_seq_len)).type(torch.bool).to(q.device)
        )
        mask.requires_grad = False
        return mask


if __name__ == "__main__":
    batch_size = 8
    seq_len = 256
    d_model = 512
    h = 8
    d_q = d_k = d_v = d_model // h

    x = torch.randn(batch_size, seq_len, d_model)

    # model = EncoderLayer()
    # print(model(x, None).shape)
    # model = DecoderLayer()
    # print(model(x, x, None, None).shape)
    # model = MultiHeadAttention(d_model=d_model, h=h)
    # print(model(x, x, x, None).shape)
    model = Transformer(vocab_size=36550)
    print(model(x, x).shape)
