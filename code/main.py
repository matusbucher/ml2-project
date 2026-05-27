from data_generation import *
from transformer import *


def first_experiment():
    """Runs the first experiment with rule number 110 and 1 step."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    rule_number = 110
    seq_len = 32
    steps = 1

    train_size = 20000
    test_size = 2000

    n_epochs = 10
    lr = 1e-3

    train_dataset = CADataset(
        n_samples=train_size,
        seq_len=seq_len,
        rule_number=rule_number,
        steps=steps
    )

    test_dataset = CADataset(
        n_samples=test_size,
        seq_len=seq_len,
        rule_number=rule_number,
        steps=steps
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    model = CATransformer(seq_len=seq_len)

    train(
        model=model,
        data_loader=train_loader,
        device=device,
        n_epochs=n_epochs,
        lr=lr
    )

    result = evaluate(model, test_loader, device)
    print(result)


if __name__ == "__main__":
    first_experiment()