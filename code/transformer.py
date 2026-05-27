import torch


class CATransformer(torch.nn.Module):
    """A transformer model for learning cellular automaton rules."""
    
    def __init__(self,
        seq_len: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()

        self.token_emb = torch.nn.Embedding(2, d_model)
        self.pos_emb = torch.nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        self.out = torch.nn.Linear(d_model, 2)
    
    def forward(self, x):
        h = self.token_emb(x) + self.pos_emb[:, :x.size(1), :]
        h = self.encoder(h)
        logits = self.out(h)
        return logits