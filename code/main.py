from data_generation import *
from transformer import *
from visualize import *


RANDOM_SEED = 42
SEQ_LEN = 32
TRAIN_SIZE = 20000
TEST_SIZE = 2000

SHOW_STEPS = 30
SHOW_N_STATES = 9

CA_RULE_30_SAVE_PATH = "results/rule_30.png"
CA_RULE_90_SAVE_PATH = "results/rule_90.png"
CA_RULE_110_SAVE_PATH = "results/rule_110.png"

TRANSFORMER_RULE_30_SAVE_PATH = "results/transformer_rule_30.png"
TRANSFORMER_RULE_90_SAVE_PATH = "results/transformer_rule_90.png"
TRANSFORMER_RULE_110_SAVE_PATH = "results/transformer_rule_110.png"


def show_ca(
    rule_number: int,
    init_states: torch.Tensor,
    steps: int,
    save_path: str | None = None,
) -> None:
    """Generates and visualizes CA histories."""

    n_samples = init_states.shape[0]
    histories = []
    ca = CellularAutomaton(rule_number)
    
    for idx in range(n_samples):
        history = ca_history(ca, init_states[idx], steps)
        histories.append(history)
    
    visualize_states(
        histories=torch.stack(histories),
        title=f"Celular Automaton Rule {rule_number}",
        save_path=save_path
    )


def train_onestep(
    rule_number: int,
    n_epochs: int = 10,
    lr: float = 1e-3,
) -> CATransformer:
    """Trains a transformer on a specific CA rule and returns the trained model."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = CADataset(
        n_samples=TRAIN_SIZE,
        seq_len=SEQ_LEN,
        rule_number=rule_number,
        steps=1
    )

    model = CATransformer(seq_len=SEQ_LEN)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    train(
        model=model,
        data_loader=train_loader,
        device=device,
        n_epochs=n_epochs,
        lr=lr
    )

    return model


def show_predictions(
    rule_number: int,
    init_states: torch.Tensor,
    steps: int,
    save_path: str | None = None,
) -> None:
    """Visualizes transformer predictions as histories."""

    model = train_onestep(rule_number)
    n_samples = init_states.shape[0]
    histories = []

    for idx in range(n_samples):
        history = transformer_history(model, init_states[idx], steps)
        histories.append(history)
        
    visualize_states(
        histories=torch.stack(histories),
        title=f"Transformer Prediction Rule {rule_number}",
        save_path=save_path
    )


def first_experiment() -> None:
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
    torch.manual_seed(RANDOM_SEED)
    init_states = torch.randint(0, 2, (SHOW_N_STATES, SEQ_LEN), dtype=torch.long)

    # show_ca(
    #     rule_number=30,
    #     init_states=init_states,
    #     steps=SHOW_STEPS,
    #     save_path=CA_RULE_30_SAVE_PATH
    # )
    # show_ca(
    #     rule_number=90,
    #     init_states=init_states,
    #     steps=SHOW_STEPS,
    #     save_path=CA_RULE_90_SAVE_PATH
    # )
    # show_ca(
    #     rule_number=110,
    #     init_states=init_states,
    #     steps=SHOW_STEPS,
    #     save_path=CA_RULE_110_SAVE_PATH
    # )

    show_predictions(
        rule_number=30,
        init_states=init_states,
        steps=SHOW_STEPS,
        save_path=TRANSFORMER_RULE_30_SAVE_PATH
    )
    show_predictions(
        rule_number=90,
        init_states=init_states,
        steps=SHOW_STEPS,
        save_path=TRANSFORMER_RULE_90_SAVE_PATH
    )
    show_predictions(
        rule_number=110,
        init_states=init_states,
        steps=SHOW_STEPS,
        save_path=TRANSFORMER_RULE_110_SAVE_PATH
    )

    # first_experiment()