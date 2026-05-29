import os
import numpy as np

from data_generation import *
from transformer import *
from visualize import *


RANDOM_SEED = 42

SHOW_STEPS = 32
SHOW_N_STATES = 4

RESULTS_DIR = "results"


def visualize_ca_trajectories(
    rule_number: int,
    init_states: np.ndarray,
    steps: int,
    save_path: str | None = None,
) -> None:
    ca = CellularAutomaton(rule_number)

    trajectories = [
        ca.trajectory(init_state, steps=steps, memoize=False)
        for init_state in init_states
    ]
    
    visualize_trajectories(
        trajectories=trajectories,
        title=f"Celular automaton: rule {rule_number}",
        save_path=save_path
    )


def visualize_predictions(
    rule_number: int,
    model: CATransformer,
    init_states: np.ndarray,
    steps: int,
    save_path: str | None = None,
) -> None:
    trajectories = [
        transformer_trajectory(model, init_state, steps=steps)
        for init_state in init_states
    ]

    visualize_trajectories(
        trajectories=trajectories,
        title=f"Transformer predictions: rule {rule_number}",
        save_path=save_path
    )


def visualize_ca_rules(
    rule_numbers: list[int],
    init_state: np.ndarray,
    steps: int,
    save_path: str | None = None,
) -> None:
    trajectories = []
    for rule_number in rule_numbers:
        ca = CellularAutomaton(rule_number)
        trajectory = ca.trajectory(init_state, steps=steps, memoize=False)
        trajectories.append(trajectory)

    labels = [f"Rule {rule_number}" for rule_number in rule_numbers]
    visualize_trajectories(
        trajectories=trajectories,
        labels=labels,
        title=f"CA rules comparison",
        save_path=save_path
    )


def first_experiment(
    do_train: bool = True,
    show_ca: bool = True,
    show_predictions: bool = True,
    show_loss_history: bool = True,
    show_eval_history: bool = True,
    print_eval: bool = True,
    random_seed: int | None = None,
):
    if random_seed is not None:
        torch.manual_seed(random_seed)

    save_dir = f"{RESULTS_DIR}/first_experiment"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rules = [30, 90, 110, 184]
    seq_len = 32
    train_steps = 1

    train_size = 100000
    test_size = 10000
    batch_size = 128

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_model = 64
    n_heads = 4
    n_layers = 2
    d_ff = 128
    dropout = 0.1

    n_epochs = 20
    lr = 1e-3

    init_states = np.random.randint(0, 2, (SHOW_N_STATES, seq_len), dtype=np.uint8)

    if show_ca:
        for rule in rules:
            visualize_ca_trajectories(
                rule_number=rule,
                init_states=init_states,
                steps=SHOW_STEPS,
                save_path=f"{save_dir}/ca_rule_{rule}.png"
            )
    
    if not do_train:
        return

    histories = []

    for rule in rules:
        train_ds = CADataset(train_size, seq_len, rule_number=rule, steps=train_steps)
        test_ds = CADataset(test_size, seq_len, rule_number=rule, steps=train_steps)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        model = CATransformer(
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )

        history = train(
            model=model,
            data_loader=train_loader,
            device=device,
            n_epochs=n_epochs,
            lr=lr,
        )
        histories.append(history)

        if print_eval:
            train_metrics = evaluate(model, train_loader, device)
            test_metrics = evaluate(model, test_loader, device)
            print("=" * 40)
            print(f"RULE {rule}")
            print(f"Train cell accuracy: {train_metrics.cell_accuracy:.4f}, sequence accuracy: {train_metrics.sequence_accuracy:.4f}")
            print(f"Test cell accuracy: {test_metrics.cell_accuracy:.4f}, sequence accuracy: {test_metrics.sequence_accuracy:.4f}")
            print("=" * 40)

        if show_loss_history:
            visualize_loss_history(
                history=history,
                title=f"Loss history: rule {rule}",
                save_path=f"{save_dir}/loss_history_rule_{rule}.png"
            )
        
        if show_eval_history:
            visualize_metrics_history(
                history=history,
                title=f"Evaluation metrics history: rule {rule}",
                save_path=f"{save_dir}/eval_history_rule_{rule}.png"
            )

        if show_predictions:
            visualize_predictions(
                rule_number=rule,
                model=model.to("cpu"),
                init_states=init_states,
                steps=SHOW_STEPS,
                save_path=f"{save_dir}/predictions_rule_{rule}.png"
            )
    
    if show_loss_history:
        visualize_multiple_loss_histories(
            histories={f"Rule {rule}": h for rule, h in zip(rules, histories)},
            title="Loss history comparison",
            save_path=f"{save_dir}/loss_history_comparison.png"
        )

    if show_eval_history:
        visualize_multiple_metrics_histories(
            histories={f"Rule {rule}": h for rule, h in zip(rules, histories)},
            title="Evaluation metrics history comparison",
            save_path=f"{save_dir}/eval_history_comparison.png"
        )


if __name__ == "__main__":
    # visualize_ca_rules(
    #     rule_numbers=[30, 90, 110, 184],
    #     init_state=cpl.init_simple(32),
    #     steps=SHOW_STEPS,
    #     save_path=f"{RESULTS_DIR}/ca_rules_comparison.png"
    # )

    first_experiment(
        do_train=False,
        random_seed=RANDOM_SEED,
    )