from data_generation import *
from transformer import *

from visualize import visualize_ca


CA_RULE_30_SAVE_PATH = "results/rule_30.png"
CA_RULE_90_SAVE_PATH = "results/rule_90.png"
CA_RULE_110_SAVE_PATH = "results/rule_110.png"

SHOW_CA_STEPS = 24
SHOW_CA_ITERS = 9



def show_ca(
    rule_number: int,
    n_steps: int = SHOW_CA_STEPS,
    n_iters: int = SHOW_CA_ITERS,
    random_seed: int | None = None,
    save_path: str | None = None,
) -> None:
    """Visualizes the history of a cellular automaton with a given rule number and number of steps."""
    
    if random_seed is not None:
        torch.manual_seed(random_seed)
    
    ca = CellularAutomaton(rule_number)
    initial_states = torch.randint(0, 2, (n_iters, 32), dtype=torch.long)
    visualize_ca(
        ca=ca,
        init_states=initial_states,
        steps=n_steps,
        save_path=save_path
    )


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
    show_ca(30, random_seed=42, save_path=CA_RULE_30_SAVE_PATH)
    show_ca(90, random_seed=42, save_path=CA_RULE_90_SAVE_PATH)
    show_ca(110, random_seed=42, save_path=CA_RULE_110_SAVE_PATH)

    # first_experiment()