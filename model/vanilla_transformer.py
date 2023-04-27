import torch
import torch.nn as nn
import math

VOCAB_SIZE = 1000


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len, :]
        x = x + pe
        return x


class VanillaTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        num_heads,
        dropout_prob,
        max_seq_len,
        vocab_size,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        self.max_length = max_seq_len
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_dim, num_heads, hidden_dim, dropout_prob
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(1)
        x = self.classifer(x)
        return x


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    print(len(tokenizer.get_vocab()))
    texts = [
        "This is a short text.",
        "This is a much longer text that will need to be truncated.",
        "This text is longer than the maximum sequence length and will need to be split into multiple sequences.",
        "This is another short text.",
    ]
    encoded_texts = tokenizer.batch_encode_plus(
        texts,
        padding="max_length",
        truncation=True,
        max_length=32,
        return_attention_mask=True,
        return_tensors="pt",
    )
    model = VanillaTransformer(
        input_dim=384,
        hidden_dim=768,
        num_layers=1,
        num_heads=1,
        max_seq_len=32,
        dropout_prob=0.1,
        vocab_size=len(tokenizer.get_vocab()),
    ).cuda()
    out = model(encoded_texts["input_ids"].cuda())
    print(out.shape)
