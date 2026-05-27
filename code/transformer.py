import torch
from torch.utils.data import DataLoader



class CATransformer(torch.nn.Module):
    """A transformer model for learning cellular automaton rules."""
    
    def __init__(self,
        seq_len: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
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


def train(
    model: CATransformer,
    data_loader: DataLoader,
    device: str,
    n_epochs: int = 10,
    lr: float = 1e-3,
) -> None:
    """Trains the transformer model on the cellular automaton dataset."""
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0

        for x, y in data_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = loss_fn(logits.view(-1, 2), y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")


def evaluate(
    model: CATransformer,
    data_loader: DataLoader,
    device: str,
) -> dict[str, float]:
    """Evaluates the transformer model on the cellular automaton dataset."""
    
    model.eval()
    
    total_cells = 0
    correct_cells = 0
    total_sequences = 0
    correct_sequences = 0

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)

            predictions = model(x).argmax(dim=-1)

            correct = predictions == y
            correct_cells += correct.sum().item()
            total_cells += correct.numel()
            correct_sequences += (correct.all(dim=1)).sum().item()
            total_sequences += x.size(0)
    
    return {
        "cell_accuracy": correct_cells / total_cells,
        "sequence_accuracy": correct_sequences / total_sequences,
    }